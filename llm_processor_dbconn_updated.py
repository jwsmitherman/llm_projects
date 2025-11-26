# processor_dbconn_with_plan_code.py
import warnings
warnings.filterwarnings("ignore")

import os, re, json, hashlib, time, math
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# LangChain Azure OpenAI
try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    from langchain.chat_models import AzureChatOpenAI

# Perf / output config (env)
ENABLE_RAY          = os.getenv("ENABLE_RAY", "auto")      # "auto" | "on" | "off"
RAY_PARTITIONS      = int(os.getenv("RAY_PARTITIONS", "8"))
RAY_MIN_ROWS_TO_USE = int(os.getenv("RAY_MIN_ROWS_TO_USE", "30000"))

BASE_DIR  = Path(__file__).resolve().parent
OUT_DIR   = Path(os.getenv("OUT_DIR", BASE_DIR / "outbound")).resolve()
OUT_FORMAT = os.getenv("OUT_FORMAT", "csv")               # "parquet" | "csv"
PARQUET_COMPRESSION = os.getenv("PARQUET_COMPRESSION", "snappy")

OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("./carrier-prompts").mkdir(parents=True, exist_ok=True)
Path("./uploads").mkdir(parents=True, exist_ok=True)

# Per-carrier Loader modes
CARRIERS = {
    "Molina": {
        "loader": "csv",             # one-row header
    },
    "Ameritas": {
        "loader": "csv"
    },
    "Manhattan Life": {
        "loader": "two_header"       # two-row header (flatten)
    },
}

# ---------------------------------------------------------------------
# schema / ops
# ---------------------------------------------------------------------

# 🔸 CHANGE 1: add PlanCode into FINAL_COLUMNS (after PlanName)
FINAL_COLUMNS = [
    "PolicyNO","PHFirst","PHLast","Status","Issuer","State","ProductType","PlanName","PlanCode",
    "SubmittedDate","EffectiveDate","TermDate","Paysched","PayCode","WritingAgentID",
    "Premium","CommPrem","TranDate","CommReceived","PTD","NoPayMon","Membercount"
]

ALLOWED_OPS = [
    "copy","const","date_mmddyyyy","date_plus_1m_mmddyyyy",
    "name_first_from_full","name_last_from_full","money",
    "membercount_from_commission","blank"
]

SYSTEM_PROMPT = """You are a data transformation agent.
Return STRICT JSON ONLY (no prose). The top-level JSON object must contain EXACTLY the required keys.
For each key return an object with:
- "op": one of [copy,const,date_mmddyyyy,date_plus_1m_mmddyyyy,name_first_from_full,name_last_from_full,
                 money,membercount_from_commission,blank]
- "source": the exact input column name when applicable (for ops that read input)
- "value": for const
If unclear, use {"op":"blank"}.
You MAY also include "PID" as a key if your rules produce it; downstream will map PID -> PID.
Do not add extra keys. Do not omit required keys.
"""

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _sig_from_cols(cols: List[str]) -> str:
    joined = "||".join(map(str, cols))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]

def _build_header_index(cols: List[str]) -> Dict[str, str]:
    return {_norm_key(h): h for h in cols}

def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

# ---------------------------------------------------------------------
# Loader-aware header probe and readers
# ---------------------------------------------------------------------

def _fast_read_header(path: str, loader: str) -> List[str]:
    """
    Fast probe: return list of column names without reading full file.
    Supports "csv" and "two_header".
    """
    if loader == "csv":
        try:
            dfo = pd.read_csv(path, nrows=0, dtype=str, engine="pyarrow", memory_map=True)
        except Exception:
            dfo = pd.read_csv(path, nrows=0, dtype=str, low_memory=False)
        return list(dfo.columns)

    # two_header: first two rows -> synthetic headers
    probe = pd.read_csv(path, header=None, nrows=2, dtype=str).fillna("")
    top, bottom = probe.iloc[0].tolist(), probe.iloc[1].tolist()

    ff, last = [], ""
    for x in top:
        x = str(x).strip()
        if x:
            last = x
        ff.append(last)

    cols: List[str] = []
    for a, b in zip(tuple(ff), bottom):
        a, b = str(a).strip(), str(b).strip()
        if not a and not b:
            name = "unnamed"
        elif not a:
            name = b
        elif not b:
            name = a
        else:
            name = f"{a} {b}"
        name = re.sub(r"[\s+]", "_", name).replace("/", "_").replace(".", "_").strip()
        name = re.sub(r"[^a-zA-Z0-9_]", "", name)
        cols.append(name)

    return cols


def _read_csv_usecols(path: str,
                      usecols: Optional[List[str]],
                      loader: str) -> pd.DataFrame:
    """
    Read CSV with optional usecols and support for two_header loader.
    """
    if loader == "csv":
        try:
            return pd.read_csv(
                path, dtype=str, engine="pyarrow", memory_map=True,
                usecols=usecols if usecols else None
            ).fillna("")
        except Exception:
            return pd.read_csv(
                path, dtype=str, low_memory=False,
                usecols=usecols if usecols else None
            ).fillna("")

    # two_header full read then filter
    tmp = pd.read_csv(path, header=None, dtype=str).fillna("")
    top, bottom = tmp.iloc[0].tolist(), tmp.iloc[1].tolist()

    ff, last = [], ""
    for x in top:
        x = str(x).strip()
        if x:
            last = x
        ff.append(last)

    cols: List[str] = []
    for a, b in zip(tuple(ff), bottom):
        a, b = str(a).strip(), str(b).strip()
        if not a and not b:
            name = "unnamed"
        elif not a:
            name = b
        elif not b:
            name = a
        else:
            name = f"{a} {b}"
        name = re.sub(r"[\s+]", "_", name).replace("/", "_").replace(".", "_").strip()
        name = re.sub(r"[^a-zA-Z0-9_]", "", name)
        cols.append(name)

    df = tmp.iloc[2:].reset_index(drop=True)
    df.columns = cols
    keep = [c for c in df.columns if not df[c].astype(str).str.strip().eq("").all()]
    df = df[keep]

    if usecols:
        present = [c for c in usecols if c in df.columns]
        return df[present].copy()
    return df

# ---------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------

def build_llm(timeout: int = 30, temperature: float = 0.0) -> AzureChatOpenAI:
    llm = AzureChatOpenAI(
        azure_endpoint   = "https://joshu-meub0vlp-swedencentral.cognitiveservices.azure.com/",
        api_key          = "7TtBcUQXFHSbGTKbHVL7IHXSxQn9ODt4SBTrIE4EGaXQLfsVnFqUJ1QJ998BHACfHwKSKJ3w3AAAAAOC0GA0Ed",
        api_version      = "2024-12-01-preview",
        azure_deployment = "gpt-5-mini_ng",
        temperature      = temperature,
        timeout          = timeout,
    )
    return llm


def llm_generate_rule_spec(headers: List[str],
                           prompt_path: Path,
                           rules_path: Path) -> Dict[str, Any]:
    llm = build_llm()
    payload = {
        "RequiredFields": FINAL_COLUMNS + ["PID"],   # allow PID alias (PlanCode now included)
        "RawHeaders": headers,
        "RulesNarrative": _load_text(rules_path),    # narrative JSON or text
        "ExtraPrompt": _load_text(prompt_path),      # extra instructions
        "OutputFormat": "Return a JSON object keyed by RequiredFields (plus PID if used)."
    }
    resp = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
    ])
    content = getattr(resp, "content", str(resp))
    try:
        return json.loads(content)
    except Exception as e:
        raise ValueError(f"LLM did not return valid JSON:\n{content}") from e

# ---------------------------------------------------------------------
# Spec normalization / binding
# ---------------------------------------------------------------------

_CANON = {_norm_key(k): k for k in FINAL_COLUMNS + ["PID"]}

def canonicalize_spec(spec_in: Dict[str, Any]) -> Dict[str, Any]:
    fixed: Dict[str, Any] = {}
    for k, v in spec_in.items():
        nk = _norm_key(k)
        fixed[_CANON.get(nk, k)] = v
    for req in FINAL_COLUMNS:
        if req not in fixed and (req != "PTD" or "PID" not in fixed):
            fixed[req] = {"op": "blank"}
    return fixed


def normalize_rule_spec(spec_in: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in spec_in.items():
        if isinstance(v, dict):
            out[k] = v
        elif isinstance(v, str):
            sv = v.strip()
            out[k] = {"op": "blank"} if sv.lower() in ("blank", "tbd") else {
                "op": "const",
                "value": sv,
            }
        else:
            out[k] = {"op": "blank"}
    return out


def needs_source(op: str) -> bool:
    return op in {
        "copy", "date_mmddyyyy", "date_plus_1m_mmddyyyy",
        "name_first_from_full", "name_last_from_full", "money",
        "membercount_from_commission",
    }


def bind_sources_to_headers(headers: List[str],
                            rule_spec_in: Dict[str, Any]) -> Dict[str, Any]:
    norm_map = _build_header_index(headers)
    fixed: Dict[str, Any] = {}
    spec = normalize_rule_spec(rule_spec_in)

    for tgt, spec in spec.items():
        op = str(spec.get("op", "")).strip()
        if op not in ALLOWED_OPS:
            fixed[tgt] = {"op": "blank"}
            continue
        if not needs_source(op):
            fixed[tgt] = spec
            continue

        src = str(spec.get("source", "")).strip()
        if not src:
            fixed[tgt] = spec
            continue

        if src in headers:
            spec["source"] = src
        else:
            ci = next((h for h in headers if h.lower() == src.lower()), None)
            if ci:
                spec["source"] = ci
            else:
                nk = _norm_key(src)
                if nk in norm_map:
                    spec["source"] = norm_map[nk]

        fixed[tgt] = spec

    return fixed


def promote_pid_to_ptd(spec: Dict[str, Any]) -> Dict[str, Any]:
    if "PID" in spec and ("PTD" not in spec or
                          str(spec["PTD"].get("op", "")).lower() in ("", "blank")):
        spec["PTD"] = spec["PID"]
    return spec


def collect_usecols(bound_spec: Dict[str, Any]) -> List[str]:
    cols: set[str] = set()
    for _, spec in bound_spec.items():
        if isinstance(spec, dict) and needs_source(str(spec.get("op", "")).strip()):
            src = spec.get("source")
            if src:
                cols.add(str(src))
    return sorted(cols)

# ---------------------------------------------------------------------
# Transform (vectorized / Ray)
# ---------------------------------------------------------------------

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
    s = s.where(~comma, s.str.replace(",", "", regex=False).str.strip())

    def _normalize(name: str) -> str:
        if not name:
            return ""
        parts = name.split()
        return " ".join(parts) if len(parts) >= 2 else name

    swapped = s.where(~comma, s.map(_normalize))
    first = swapped.str.split().str[0].fillna("")
    last = swapped.str.split().str[-1].fillna("")
    return first.str.title().astype("string"), last.str.title().astype("string")


def _money_to_float_str(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()
    neg_paren = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"[,\$]", "", regex=True).str.strip()
    num = pd.to_numeric(x, errors="coerce")
    num = np.where(neg_paren, -num, num)
    out = np.where(
        pd.isna(num),
        "",
        np.where(pd.notnull(num), np.vectorize(lambda v: f"{v:.2f}")(num), ""),
    )
    return pd.Series(out, index=s.index, dtype="string")


def _sign_flag_from_money(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()
    neg_paren = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"[,\$]", "", regex=True).str.strip()
    num = pd.to_numeric(x, errors="coerce")
    num = np.where(neg_paren, -num, num)
    out = np.where(pd.isna(num), "", np.where(num < 0, "-1", "1"))
    return pd.Series(out, index=s.index, dtype="string")


def apply_rules(df: pd.DataFrame, bound_spec: Dict[str, Any]) -> pd.DataFrame:
    out: Dict[str, pd.Series] = {}
    spec = normalize_rule_spec(bound_spec)
    spec = promote_pid_to_ptd(spec)

    def empty() -> pd.Series:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    for tgt in FINAL_COLUMNS:
        tspec = spec.get(tgt) or (spec.get("PID") if tgt == "PTD" else None)
        if not isinstance(tspec, dict):
            out[tgt] = empty()
            continue
        op = str(tspec.get("op", "")).strip()
        if op not in ALLOWED_OPS:
            out[tgt] = empty()
            continue

        if op == "copy":
            s = tspec.get("source")
            out[tgt] = df.get(s, empty()).astype(str)
        elif op == "const":
            out[tgt] = pd.Series(
                [str(tspec.get("value", ""))] * len(df),
                index=df.index,
                dtype="string",
            )
        elif op == "date_mmddyyyy":
            s = tspec.get("source")
            out[tgt] = _to_mmddyyyy(df.get(s, empty()))
        elif op == "date_plus_1m_mmddyyyy":
            s = tspec.get("source")
            out[tgt] = _add_one_month_mmddyyyy(df.get(s, empty()))
        elif op == "name_first_from_full":
            s = tspec.get("source")
            out[tgt] = _parse_case_name_first_last(df.get(s, empty()))[0]
        elif op == "name_last_from_full":
            s = tspec.get("source")
            out[tgt] = _parse_case_name_first_last(df.get(s, empty()))[1]
        elif op == "money":
            s = tspec.get("source")
            out[tgt] = _money_to_float_str(df.get(s, empty()))
        elif op == "membercount_from_commission":
            s = tspec.get("source")
            flags = _sign_flag_from_money(df.get(s, empty()))
            out[tgt] = pd.Series(
                np.where(flags.eq("1"), "1", flags),
                index=df.index,
                dtype="string",
            )
        else:
            out[tgt] = empty()

    return pd.DataFrame(out, columns=FINAL_COLUMNS).fillna("").astype("string")

# ---------------------------------------------------------------------
# Ray helpers
# ---------------------------------------------------------------------

def should_use_ray(n_rows: int) -> bool:
    if ENABLE_RAY == "on":
        return True
    if ENABLE_RAY == "off":
        return False
    return n_rows >= RAY_MIN_ROWS_TO_USE


def apply_rules_parallel(df: pd.DataFrame,
                         bound_spec: Dict[str, Any]) -> pd.DataFrame:
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    spec_ref = ray.put(bound_spec)

    @ray.remote
    def _worker(chunk: pd.DataFrame, spec_ref):
        return apply_rules(chunk, ray.get(spec_ref))

    parts = np.array_split(df, max(1, RAY_PARTITIONS))
    futures = [_worker.remote(part, spec_ref) for part in parts]
    futures = [_worker.remote(part, spec_ref) for part in parts]
    outs = ray.get(futures)
    return pd.concat(outs, ignore_index=True)

# ---------------------------------------------------------------------
# Manhattan Life helpers
# ---------------------------------------------------------------------

def extract_manhattan_policy_plan_from_csv(csv_path: str, log: Callable[[str], None]) -> pd.DataFrame:
    """Read Manhattan Life raw CSV, flatten 2-row headers, return ['PolicyNumber','PlanCode']"""
    p = str(csv_path).strip().strip('"').strip("'")
    if p.startswith("/") and not p.startswith("\\\\"):
        # normalize single-leading backslash to UNC
        p = "\\" + p
    log(f"[ManhattanLife] Reading raw CSV: {p}")

    # Try two-row header first (common for Manhattan Life), then fallback
    read_kwargs = dict(dtype=str, engine="python")
    try:
        raw = pd.read_csv(p, header=[0, 1], encoding="utf-8-sig", **read_kwargs)
    except Exception:
        try:
            raw = pd.read_csv(p, header=0, encoding="utf-8-sig", **read_kwargs)
        except Exception:
            raw = pd.read_csv(p, header=0, encoding="latin1", **read_kwargs)

    # ---- Flatten header names
    if isinstance(raw.columns, pd.MultiIndex):
        flat: List[str] = []
        for parts in raw.columns:
            parts = [str(x).strip() for x in parts
                     if x is not None and str(x).strip() != ""]
            name = re.sub(r"\s+", " ", " ".join(parts)).strip()
            flat.append(name)
        raw.columns = flat
    else:
        raw.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.columns]

    # Normalized header map
    norm = {c: re.sub(r"[^A-Za-z0-9]", "", c.lower()) for c in raw.columns}

    # ---- Find PlanCode (prefer headers containing both 'plan' and 'code')
    plan_code_col: Optional[str] = None
    for col, nc in norm.items():
        if "plan" in nc and "code" in nc:
            plan_code_col = col
            break

    if plan_code_col is None:
        # last resort: a column exactly called 'plancode'
        for col, nc in norm.items():
            if nc == "plancode":
                plan_code_col = col
                break

    # ---- Find PolicyNumber
    # 1) exact/near matches
    policy_col: Optional[str] = None
    for col, nc in norm.items():
        if ("policy" in nc and "number" in nc) or nc in ("policynumber", "policyno"):
            policy_col = col
            break

    # 2) heuristic: any 'policy' column with the most numeric-looking values
    if policy_col is None:
        candidates = [c for c in raw.columns if "policy" in c.lower()]
        if candidates:
            def numeric_ratio(series: pd.Series) -> float:
                s = series.dropna().astype(str).str.strip()
                if len(s) == 0:
                    return 0.0
                m = s.str.match(r"^\d{5,}$")  # mostly long numerics
                return m.mean()

            policy_col = max(candidates, key=lambda c: numeric_ratio(raw[c]))
        # if it's clearly not numeric, fall back to first candidate anyway

    if policy_col is None or plan_code_col is None:
        raise ValueError(
            "Could not locate Policy/PlanCode columns.\n"
            f"Seen headers: {list(raw.columns)[:20]}"
        )

    df2 = raw[[policy_col, plan_code_col]].copy()
    df2.columns = ["PolicyNumber", "PlanCode"]
    df2["PolicyNumber"] = df2["PolicyNumber"].astype(str).str.strip()
    df2["PlanCode"] = df2["PlanCode"].astype(str).str.strip().str.upper()
    df2 = df2[df2["PolicyNumber"] != ""].reset_index(drop=True)

    log(
        f"[ManhattanLife] Extracted rows: {len(df2)} | "
        f"cols -> PolicyNumber: {policy_col}, PlanCode: {plan_code_col}"
    )
    return df2


def match_llm_output_to_raw_counts(raw_link_df: pd.DataFrame,
                                   llm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the LLM output has the same number of rows per PolicyNo
    as the raw_link_df (by repeating or trimming rows).
    """
    target_counts = raw_link_df["PolicyNo"].value_counts()
    adjusted: List[pd.DataFrame] = []

    for policy_no, target_n in target_counts.items():
        block = llm_df[llm_df["PolicyNo"] == policy_no]

        if block.empty:
            # No output produced for this policy_no
            continue

        if len(block) == target_n:
            adjusted.append(block)
            continue

        if len(block) > target_n:
            # Trim down
            adjusted.append(block.head(target_n))
        else:
            # Repeat rows to reach target_n
            repeats = (target_n // len(block)) + 1
            expanded = pd.concat([block] * repeats, ignore_index=True)
            adjusted.append(expanded.head(target_n))

    if not adjusted:
        return llm_df.copy()

    return pd.concat(adjusted, ignore_index=True)

# ---------------------------------------------------------------------
# Public entry point the Flask app will call
# ---------------------------------------------------------------------

from manhattan_mapping import get_manhattan_mapping
from stg_plan_mapping_min import stg_plan_mapping_min


def log(line: str) -> None:
    # stream logs immediately to console
    print(line, flush=True)


def run_llm_pipeline(
    *,
    issuer: str,
    paycode: str,
    trandate: str,
    load_task_id: str,
    company_issuer_id: str,
    csv_path: str,
    template_dir: str,
    output_csv_name: str,
    server_name: str,
    database_name: str,
) -> str:
    """
    Returns output file path. Logs progress via `log(line)`.
    `template_dir` should contain `<issuer>_prompt.txt` and `<issuer>_rules.json`.
    """
    start = time.perf_counter()
    log(f"Starting LLM pipeline | issuer={issuer} | csv={csv_path}")

    loader = CARRIERS.get(issuer, {}).get("loader", "csv")
    prompt_path = Path(template_dir) / f"{issuer}_prompt.txt"
    rules_path = Path(template_dir) / f"{issuer}_rules.json"

    if not prompt_path.exists():
        log(f"NOTE: prompt file not found, continuing: {prompt_path}")
    if not rules_path.exists():
        log(f"NOTE: rules file not found, continuing: {rules_path}")

    # 1) Probe headers
    headers = _fast_read_header(csv_path, loader)
    sig = _sig_from_cols(headers)
    compiled_path = Path(template_dir) / f"{issuer}_compiled_rules_{sig}.json"

    # 2) LLM generate or load cached compiled rules
    if compiled_path.exists():
        log(f"[Rules] Loaded cached compiled rules: {compiled_path.name}")
        # 🔸 CHANGE 2: compiled file already contains bound_spec; load it directly
        bound_spec = json.loads(compiled_path.read_text(encoding="utf-8"))
    else:
        log("[Rules] Generating with LLM.")
        raw_spec = llm_generate_rule_spec(headers, prompt_path, rules_path)
        raw_spec = canonicalize_spec(raw_spec)
        bound_spec = bind_sources_to_headers(headers, raw_spec)
        bound_spec = promote_pid_to_ptd(bound_spec)
        compiled_path.write_text(
            json.dumps(bound_spec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"[Rules] Compiled & saved: {compiled_path.name}")

    # 3) Minimal IO re-read
    usecols = collect_usecols(bound_spec)
    df = _read_csv_usecols(csv_path, usecols if usecols else None, loader)
    n = len(df)
    log(f"[IO] Rows loaded: {n} | usecols={len(usecols)} | loader={loader}")

    # 4) Transform (Ray vs single-thread)
    use_ray = should_use_ray(n)
    if use_ray:
        log("[Exec] Using Ray parallel mode.")
        out_df = apply_rules_parallel(df, bound_spec)
    else:
        out_df = apply_rules(df, bound_spec)

    # 5) Add constants
    out_df["TranDate"] = trandate
    out_df["PayCode"] = paycode
    out_df["Issuer"] = issuer
    out_df["ProductType"] = ""
    out_df["PlanName"] = ""
    # 🔸 if PlanCode wasn't produced by rules (e.g., non-Manhattan issuers), ensure column exists
    if "PlanCode" not in out_df.columns:
        out_df["PlanCode"] = ""
    out_df["Note"] = ""

    out_df.to_csv("./out/base_test.csv", index=None)

    # 6) Manhattan Life enrichment
    if issuer == "Manhattan Life":
        log("[INFO] Manhattan Life detected — enriching from DB (PlanCode -> map_df -> PolicyNumber -> out_df).")
        try:
            # raw_link_df now has PolicyNumber, PlanCode
            raw_link_df = extract_manhattan_policy_plan_from_csv(csv_path, log)

            # keep a copy with PlanCode for output logic
            r = raw_link_df.copy()
            r["PolicyNo"] = r["PolicyNumber"]

            # For DB we still need a PlanId column -> copy PlanCode into PlanId
            raw_link_df_db = raw_link_df.rename(columns={"PlanCode": "PlanId"})

            payload = raw_link_df_db[["PolicyNumber", "PlanId"]].copy()
            payload["LoadTaskId"] = load_task_id
            payload = payload[["LoadTaskId", "PolicyNumber", "PlanId"]]

            inserted = stg_plan_mapping_min(
                df=payload,
                server=server_name,
                database=database_name,
                log=log,
            )
            log(f"[ManhattanLife] STG insert-min rows: {inserted}")

            map_df = get_manhattan_mapping(
                Load_task_id=load_task_id,
                company_issuer_id=company_issuer_id,
                server=server_name,
                database=database_name,
                log=log,
            )

            # 🔸 include PlanCode in final column layout
            out_df_cols = [
                "PolicyNo", "PHFirst", "PHLast", "Status", "Issuer", "State",
                "ProductType", "PlanName", "PlanCode",
                "SubmittedDate", "EffectiveDate",
                "TermDate", "Paysched", "PayCode", "WritingAgentID", "Premium",
                "CommPrem", "TranDate", "CommReceived", "PTD", "NoPayMon",
                "Membercount", "Note",
            ]

            out_df2 = out_df.copy()
            # We will recompute PlanName / ProductType; drop them, but keep PlanCode from LLM if present
            out_df2 = out_df2.drop(columns=["PlanName", "ProductType"], errors="ignore")

            if map_df.shape[0] == 0:
                # only payload (raw file) – PlanCode from raw, PlanName/ProductType blank
                payload_df = raw_link_df[["PolicyNumber", "PlanCode"]].copy()
                payload_df.columns = ["PolicyNo", "PlanCode"]
                out_df2 = out_df2.merge(
                    payload_df,
                    on="PolicyNo",
                    how="left",
                )
                out_df2["PlanName"] = ""
                out_df2["ProductType"] = ""
                out_df2 = out_df2[out_df_cols]
                out_df2 = out_df2.fillna("")
                out_df = out_df2.copy()
                out_df = match_llm_output_to_raw_counts(r, out_df)
            else:
                # First merge with map_df (preferred source for PlanName/ProductType, maybe PlanCode)
                out_df2 = out_df2.merge(
                    map_df[["PolicyNo", "PlanName", "ProductType"]],
                    on="PolicyNo",
                    how="left",
                )

                # Always also merge PlanCode from raw file (authoritative)
                payload_df = raw_link_df[["PolicyNumber", "PlanCode"]].copy()
                payload_df.columns = ["PolicyNo", "PlanCode_raw"]
                out_df2 = out_df2.merge(
                    payload_df,
                    on="PolicyNo",
                    how="left",
                )

                # if LLM already produced PlanCode we can prefer that, otherwise use raw
                if "PlanCode" in out_df2.columns:
                    out_df2["PlanCode"] = out_df2["PlanCode"].fillna(out_df2["PlanCode_raw"])
                else:
                    out_df2["PlanCode"] = out_df2["PlanCode_raw"]

                out_df2 = out_df2.drop(columns=["PlanCode_raw"])
                out_df2 = out_df2[out_df_cols]
                out_df2 = out_df2.fillna("")
                out_df = out_df2.copy()
                out_df = match_llm_output_to_raw_counts(r, out_df)

        except Exception as e:
            log(f"[WARN] Manhattan Life enrichment failed: {e}")
            if "PlanName" not in out_df.columns:
                out_df["PlanName"] = ""
            if "ProductType" not in out_df.columns:
                out_df["ProductType"] = ""
            if "PlanCode" not in out_df.columns:
                out_df["PlanCode"] = ""

    # 7) Write output file
    input_path = Path(csv_path)
    out_dir = input_path.parent
    out_path = out_dir / f"{output_csv_name}.csv"

    try:
        out_df.to_csv(out_path, index=False)
        log(f"[Output] file written sucessfully: {out_path}")
    except Exception as e:
        log(f"[ERROR] Failed to write output file: {e}")

    elapsed = time.perf_counter() - start
    log(f"Completed: {out_path.as_posix()} (elapsed {elapsed:.2f}s)")

    return out_path.as_posix()
