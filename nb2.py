# llm_build_and_run_carrier_aware.py
import os, re, json, hashlib, time, math
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# LangChain Azure OpenAI
try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    from langchain.chat_models import AzureChatOpenAI

###############################################################################
# CONFIG (env or defaults)
###############################################################################
FILE_LOCATION = os.getenv("FILE_LOCATION", "")  # if blank, use carrier raw_path
FILE_NAME     = os.getenv("FILE_NAME",     "")
TRANDATE      = os.getenv("TRANDATE",      "2025-10-01")
PAYCODE       = os.getenv("PAYCODE",       "FromApp")
ISSUER        = os.getenv("ISSUER",        "manhattan_life")  # molina | ameritas | manhattan_life

# Ray usage
ENABLE_RAY          = os.getenv("ENABLE_RAY", "auto")   # "auto" | "on" | "off"
RAY_PARTITIONS      = int(os.getenv("RAY_PARTITIONS", "8"))
RAY_MIN_ROWS_TO_USE = int(os.getenv("RAY_MIN_ROWS_TO_USE", "300000"))

# Output
OUT_DIR             = Path(os.getenv("OUT_DIR", "./outbound"))
OUT_FORMAT          = os.getenv("OUT_FORMAT", "parquet")      # "parquet" | "csv"
PARQUET_COMPRESSION = os.getenv("PARQUET_COMPRESSION", "snappy")  # "snappy" | "zstd" | "none"

# LLM files (will be overridden per carrier)
PROMPT_PATH_ENV     = os.getenv("PROMPT_PATH", "")
RULES_NARRATIVE_ENV = os.getenv("RULES_NARRATIVE", "")

# Cache behavior
FORCE_REGEN_RULES   = os.getenv("FORCE_REGEN_RULES", "0") == "1"

OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("./carrier_prompts").mkdir(parents=True, exist_ok=True)

###############################################################################
# PER-CARRIER SETTINGS (controls flattener vs csv)
###############################################################################
CARRIERS = {
    "molina": {
        "prompt_path": "./carrier_prompts/molina_prompt.txt",
        "rules_path":  "./carrier_prompts/molina_rules.json",
        "raw_path":    "./data/molina_raw_data.csv",
        "loader":      "csv"          # one-row header
    },
    "ameritas": {
        "prompt_path": "./carrier_prompts/ameritas_prompt.txt",
        "rules_path":  "./carrier_prompts/ameritas_rules.json",
        "raw_path":    "./data/ameritas_raw_data.csv",
        "loader":      "csv"
    },
    "manhattan_life": {
        "prompt_path": "./carrier_prompts/manhattan_life_prompt.txt",
        "rules_path":  "./carrier_prompts/manhattan_life_rules.json",
        "raw_path":    "/mnt/data/manhattan_life_raw_data.csv",
        "loader":      "two_header"   # only this uses the flattener
    },
}

###############################################################################
# SCHEMA / OPS / SYSTEM PROMPT
###############################################################################
FINAL_COLUMNS = [
    "PolicyNo","PHFirst","PHLast","Status","Issuer","State","ProductType","PlanName",
    "SubmittedDate","EffectiveDate","TermDate","PaySched","PayCode","WritingAgentID",
    "Premium","CommPrem","TranDate","CommReceived","PTD","NoPayMon","Membercount"
]
ALLOWED_OPS = {
    "copy","const","date_mmddyyyy","date_plus_1m_mmddyyyy",
    "name_first_from_full","name_last_from_full","money",
    "membercount_from_commission","blank"
}
SYSTEM_PROMPT = """You are a data transformation agent.
Return STRICT JSON ONLY (no prose). The top-level JSON object must contain EXACTLY the required keys.
For each key return an object with:
- "op": one of [copy,const,date_mmddyyyy,date_plus_1m_mmddyyyy,name_first_from_full,name_last_from_full,money,membercount_from_commission,blank]
- "source": the exact input column name when applicable (for ops that read input)
- "value": for const
If unclear, use {"op":"blank"}.
You MAY also include "PID" as a key if your rules produce it; downstream will map PID→PTD.
Do not add extra keys. Do not omit required keys.
"""

###############################################################################
# UTILITIES
###############################################################################
def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _sig_from_cols(cols: List[str]) -> str:
    joined = "||".join(map(str, cols))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]

def _build_header_index(cols: List[str]) -> Dict[str, str]:
    return {_norm_key(h): h for h in cols}

def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

###############################################################################
# HEADER PROBE + READERS (loader-aware)
###############################################################################
def _fast_read_header(path: str, loader: str) -> List[str]:
    if loader == "csv":
        try:
            df0 = pd.read_csv(path, nrows=0, dtype=str, engine="pyarrow", memory_map=True)
        except Exception:
            df0 = pd.read_csv(path, nrows=0, dtype=str, low_memory=False)
        return list(df0.columns)

    # two_header probe (first two rows → synthetic headers)
    probe = pd.read_csv(path, header=None, nrows=2, dtype=str).fillna("")
    top, bottom = probe.iloc[0].tolist(), probe.iloc[1].tolist()

    ff, last = [], ""
    for x in top:
        x = str(x).strip()
        if x:
            last = x
        ff.append(last)

    cols = []
    for a, b in zip(ff, bottom):
        a, b = str(a).strip(), str(b).strip()
        if not a and not b: name = "unnamed"
        elif not a:         name = b
        elif not b:         name = a
        else:               name = f"{a} {b}"
        name = re.sub(r"\s+", "_", name).replace("/", "_").replace(".", "_").strip()
        name = re.sub(r"\s+$", "", name)
        cols.append(name)
    return cols

def _read_csv_usecols(path: str, usecols: Optional[List[str]], loader: str) -> pd.DataFrame:
    if loader == "csv":
        try:
            return pd.read_csv(path, dtype=str, engine="pyarrow", memory_map=True,
                               usecols=usecols if usecols else None).fillna("")
        except Exception:
            return pd.read_csv(path, dtype=str, low_memory=False,
                               usecols=usecols if usecols else None).fillna("")

    # two_header full read then filter
    tmp = pd.read_csv(path, header=None, dtype=str).fillna("")
    top, bottom = tmp.iloc[0].tolist(), tmp.iloc[1].tolist()

    ff, last = [], ""
    for x in top:
        x = str(x).strip()
        if x:
            last = x
        ff.append(last)

    cols = []
    for a, b in zip(ff, bottom):
        a, b = str(a).strip(), str(b).strip()
        if not a and not b: name = "unnamed"
        elif not a:         name = b
        elif not b:         name = a
        else:               name = f"{a} {b}"
        name = re.sub(r"\s+", "_", name).replace("/", "_").replace(".", "_").strip()
        name = re.sub(r"\s+$", "", name)
        cols.append(name)

    df = tmp.iloc[2:].reset_index(drop=True)
    df.columns = cols
    keep = [c for c in df.columns if not df[c].astype(str).str.strip().eq("").all()]
    df = df[keep]
    if usecols:
        present = [c for c in usecols if c in df.columns]
        return df[present].copy()
    return df

###############################################################################
# LLM
###############################################################################
def build_llm(timeout: int = 30, temperature: float = 0.0) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        request_timeout=timeout,
        temperature=temperature
    )

def llm_generate_rule_spec(headers: List[str], prompt_path: Path, rules_path: Path) -> Dict[str, Any]:
    llm = build_llm()
    payload = {
        "RequiredFields": FINAL_COLUMNS + ["PID"],  # allow PID alias
        "RawHeaders": headers,
        "RulesNarrative": _load_text(rules_path),
        "ExtraPrompt": _load_text(prompt_path),
        "OutputFormat": "Return a JSON object keyed by RequiredFields (plus PID if used)."
    }
    resp = llm.invoke([
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user",   "content": json.dumps(payload, ensure_ascii=False)}
    ])
    content = getattr(resp, "content", str(resp))
    try:
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"LLM did not return valid JSON:\n{content}") from e

###############################################################################
# SPEC NORMALIZATION / BINDING
###############################################################################
_CANON = {_norm_key(k): k for k in FINAL_COLUMNS + ["PID"]}

def canonicalize_spec_keys(spec_in: Dict[str, Any]) -> Dict[str, Any]:
    fixed = {}
    for k, v in spec_in.items():
        nk = _norm_key(k)
        fixed[_CANON.get(nk, k)] = v
    for req in FINAL_COLUMNS:
        if req not in fixed and (req != "PTD" or "PID" not in fixed):
            fixed[req] = {"op":"blank"}
    return fixed

def normalize_rule_spec(spec_in: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in spec_in.items():
        if isinstance(v, dict):
            out[k] = v
        elif isinstance(v, str):
            sv = v.strip()
            out[k] = {"op":"blank"} if sv.lower() in ("blank","tbd") else {"op":"const","value":sv}
        else:
            out[k] = {"op":"blank"}
    return out

def needs_source(op: str) -> bool:
    return op in {
        "copy","date_mmddyyyy","date_plus_1m_mmddyyyy",
        "name_first_from_full","name_last_from_full","money",
        "membercount_from_commission"
    }

def bind_sources_to_headers(headers: List[str], rule_spec_in: Dict[str, Any]) -> Dict[str, Any]:
    norm_map = _build_header_index(headers)
    fixed: Dict[str, Any] = {}
    specN = normalize_rule_spec(rule_spec_in)
    for tgt, spec in specN.items():
        op = str(spec.get("op","")).strip()
        if op not in ALLOWED_OPS:
            fixed[tgt] = {"op":"blank"}; continue
        if not needs_source(op):
            fixed[tgt] = spec; continue

        src = str(spec.get("source","")).strip()
        if not src:
            fixed[tgt] = spec; continue

        if src in headers:
            spec["source"] = src
        else:
            ci = next((h for h in headers if h.lower()==src.lower()), None)
            if ci:
                spec["source"] = ci
            else:
                nk = _norm_key(src)
                if nk in norm_map:
                    spec["source"] = norm_map[nk]
        fixed[tgt] = spec
    return fixed

def promote_pid_to_ptd(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "PID" in spec and ("PTD" not in spec or str(spec["PTD"].get("op","")).lower() in ("","blank")):
        spec["PTD"] = spec["PID"]
    return spec

def collect_usecols(bound_spec: Dict[str, Any]) -> List[str]:
    cols = set()
    for _, spec in bound_spec.items():
        if isinstance(spec, dict) and needs_source(str(spec.get("op","")).strip()):
            src = spec.get("source")
            if src: cols.add(str(src))
    return sorted(cols)

###############################################################################
# TRANSFORM (vectorized / Ray)
###############################################################################
def _to_mmddyyyy(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%m/%d/%Y").fillna("").astype("string")

def _add_one_month_mmddyyyy(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    dtp = dt.apply(lambda x: x + relativedelta(months=1) if pd.notnull(x) else pd.NaT)
    return pd.Series(dtp).dt.strftime("%m/%d/%Y").fillna("").astype("string")

def _parse_case_name_first_last(series: pd.Series):
    s = series.fillna("").astype(str).str.strip()
    comma = s.str.contains(",", regex=False)
    swapped = s.where(~comma, s.str.replace(",", "", regex=False).str.strip())
    def _normalize(name: str) -> str:
        if not name: return ""
        parts = name.split()
        return " ".join(parts[1:] + parts[:1]) if len(parts) >= 2 else name
    normalized = swapped.where(~comma, swapped.map(_normalize))
    toks = normalized.str.split()
    last = toks.str[-1].fillna("")
    first = toks.apply(lambda xs: " ".join(xs[:-1]) if isinstance(xs, list) and len(xs) > 1 else "").fillna("")
    return first.str.title().astype("string"), last.str.title().astype("string")

def _money_to_float_str(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()
    neg_paren = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"[\$,()]", "", regex=True).str.strip()
    num = pd.to_numeric(x, errors="coerce")
    num = np.where(neg_paren, -num, num)
    out = np.where(pd.isna(num), "", np.where(pd.notnull(num), np.vectorize(lambda v: f"{v:.2f}")(num), ""))
    return pd.Series(out, index=s.index, dtype="string")

def _sign_flag_from_money(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()
    neg_paren = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"[\$,()]", "", regex=True).str.strip()
    num = pd.to_numeric(x, errors="coerce")
    num = np.where(neg_paren, -num, num)
    out = np.where(pd.isna(num), "", np.where(num < 0, "-1", "1"))
    return pd.Series(out, index=s.index, dtype="string")

def apply_rules(df: pd.DataFrame, bound_spec: Dict[str, Any]) -> pd.DataFrame:
    out: Dict[str, pd.Series] = {}
    spec = normalize_rule_spec(bound_spec)
    spec = promote_pid_to_ptd(spec)
    def empty(): return pd.Series([""]*len(df), index=df.index, dtype="string")
    for tgt in FINAL_COLUMNS:
        tspec = spec.get(tgt) or (spec.get("PID") if tgt=="PTD" else None)
        if not isinstance(tspec, dict):
            out[tgt] = empty(); continue
        op = str(tspec.get("op","")).strip()
        if op not in ALLOWED_OPS:
            out[tgt] = empty(); continue
        if op == "copy":
            s = tspec.get("source"); out[tgt] = df.get(s, empty()).astype(str)
        elif op == "const":
            out[tgt] = pd.Series([str(tspec.get("value",""))]*len(df), index=df.index, dtype="string")
        elif op == "date_mmddyyyy":
            s = tspec.get("source"); out[tgt] = _to_mmddyyyy(df.get(s, empty()))
        elif op == "date_plus_1m_mmddyyyy":
            s = tspec.get("source"); out[tgt] = _add_one_month_mmddyyyy(df.get(s, empty()))
        elif op == "name_first_from_full":
            s = tspec.get("source"); out[tgt] = _parse_case_name_first_last(df.get(s, empty()))[0]
        elif op == "name_last_from_full":
            s = tspec.get("source"); out[tgt] = _parse_case_name_first_last(df.get(s, empty()))[1]
        elif op == "money":
            s = tspec.get("source"); out[tgt] = _money_to_float_str(df.get(s, empty()))
        elif op == "membercount_from_commission":
            s = tspec.get("source")
            flags = _sign_flag_from_money(df.get(s, empty()))
            out[tgt] = pd.Series(np.where(flags.eq("1"), "1", flags), index=df.index, dtype="string")
        else:
            out[tgt] = empty()
    return pd.DataFrame(out, columns=FINAL_COLUMNS).fillna("").astype("string")

def should_use_ray(n_rows: int) -> bool:
    if ENABLE_RAY == "on":  return True
    if ENABLE_RAY == "off": return False
    return n_rows >= RAY_MIN_ROWS_TO_USE

def apply_rules_parallel(df: pd.DataFrame, bound_spec: Dict[str, Any]) -> pd.DataFrame:
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    spec_ref = ray.put(bound_spec)

    @ray.remote
    def _worker(chunk: pd.DataFrame, spec_ref):
        return apply_rules(chunk, ray.get(spec_ref))

    # Partition reasonably
    p = max(1, RAY_PARTITIONS)
    parts = np.array_split(df, p)
    futures = [_worker.remote(part, spec_ref) for part in parts]
    outs = ray.get(futures)
    return pd.concat(outs, ignore_index=True)

###############################################################################
# MAIN
###############################################################################
def main():
    start = time.perf_counter()
    print(f"=== Start ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"ISSUER={ISSUER}")

    # Resolve carrier settings
    cfg = CARRIERS.get(ISSUER)
    if not cfg:
        raise ValueError(f"Unknown ISSUER '{ISSUER}'. Choose one of: {list(CARRIERS)}")

    # Resolve paths
    prompt_path = Path(PROMPT_PATH_ENV) if PROMPT_PATH_ENV else Path(cfg["prompt_path"])
    rules_path  = Path(RULES_NARRATIVE_ENV) if RULES_NARRATIVE_ENV else Path(cfg["rules_path"])
    raw_path    = FILE_LOCATION if FILE_LOCATION else cfg["raw_path"]
    loader      = cfg["loader"]

    # 1) Probe headers
    headers = _fast_read_header(raw_path, loader)
    sig = _sig_from_cols(headers)

    # 2) Cache path for compiled, header-bound rules
    compiled_path = Path(f"./carrier_prompts/{ISSUER}_compiled_rules__{sig}.json")

    # 3) LLM build or load cached
    if compiled_path.exists() and not FORCE_REGEN_RULES:
        bound_spec = json.loads(compiled_path.read_text(encoding="utf-8"))
        print(f"[Rules] Loaded cached compiled rules → {compiled_path.name}")
    else:
        print("[Rules] Generating with LLM…")
        raw_spec   = llm_generate_rule_spec(headers, prompt_path, rules_path)  # <-- LLM used
        raw_spec   = canonicalize_spec_keys(raw_spec)
        bound_spec = bind_sources_to_headers(headers, raw_spec)
        bound_spec = promote_pid_to_ptd(bound_spec)
        compiled_path.write_text(json.dumps(bound_spec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Rules] Compiled & saved → {compiled_path.name}")

    # 4) Re-read CSV with only needed columns
    usecols = collect_usecols(bound_spec)
    df = _read_csv_usecols(raw_path, usecols if usecols else None, loader)
    n = len(df)
    print(f"[IO] Rows loaded: {n:,} | usecols={len(usecols)} | loader={loader}")

    # 5) Transform
    use_ray = should_use_ray(n)
    if use_ray:
        print("[Exec] Using Ray parallel mode…")
        out_df = apply_rules_parallel(df, bound_spec)
    else:
        out_df = apply_rules(df, bound_spec)

    # 6) Constants
    out_df["TranDate"] = TRANDATE
    out_df["PayCode"]  = PAYCODE
    out_df["Issuer"]   = ISSUER

    # 7) Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / f"{ISSUER}_{sig}"
    if OUT_FORMAT.lower() == "parquet":
        out_path = out_base.with_suffix(".parquet")
        try:
            comp = None if PARQUET_COMPRESSION.lower()=="none" else PARQUET_COMPRESSION
            out_df.to_parquet(out_path, index=False, compression=comp)
        except Exception:
            out_path = out_base.with_suffix(".csv")
            out_df.to_csv(out_path, index=False)
    else:
        out_path = out_base.with_suffix(".csv")
        out_df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - start
    print(f"Rows: {n:,} | Ray: {use_ray} | Out: {out_path.as_posix()} | Elapsed: {elapsed:.2f}s")
    print(f"=== Done ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")

if __name__ == "__main__":
    main()
