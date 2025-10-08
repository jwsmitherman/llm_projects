# fast_run.py
# Goal: <10s end-to-end for typical CSV sizes by removing LLM calls and auditing during the hot path.

import os, re, json, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

###############################################################################
# CONFIG (set these or pass via env)
###############################################################################

FILE_LOCATION = os.getenv("FILE_LOCATION", "./data/manhattan_life_raw_data.csv")
FILE_NAME     = os.getenv("FILE_NAME",     "manhattan_life_raw_data.csv")
TRANDATE      = os.getenv("TRANDATE",      "2025-10-01")
PAYCODE       = os.getenv("PAYCODE",       "FromApp")
ISSUER        = os.getenv("ISSUER",        "manhattan_life")  # molina | ameritas | manhattan_life

# Use Ray only when DF is large enough to offset startup overhead.
ENABLE_RAY            = os.getenv("ENABLE_RAY", "auto")   # "auto" | "on" | "off"
RAY_PARTITIONS        = int(os.getenv("RAY_PARTITIONS", "8"))
RAY_MIN_ROWS_TO_USE   = int(os.getenv("RAY_MIN_ROWS_TO_USE", "300000"))

###############################################################################
# SCHEMA / OPS (unchanged)
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

CARRIERS = {
    "molina": {
        "raw_path": "./data/molina_raw_data.csv",
        "loader": "csv",
        "compiled_rules_path": "./carrier_prompts/molina_compiled_rules.json",
    },
    "ameritas": {
        "raw_path": "./data/ameritas_raw_data.csv",
        "loader": "csv",
        "compiled_rules_path": "./carrier_prompts/ameritas_compiled_rules.json",
    },
    "manhattan_life": {
        "raw_path": "./data/manhattan_life_raw_data.csv",
        "loader": "two_header",
        "compiled_rules_path": "./carrier_prompts/manhattan_life_compiled_rules.json",
    }
}

###############################################################################
# LOADERS (fast)
###############################################################################

def flatten_two_header_csv(path: str) -> pd.DataFrame:
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
        name = re.sub(r"\s+", "_", name)
        name = name.replace("/", "_").replace(".", "_").strip()
        name = re.sub(r"\s+$", "", name)
        cols.append(name)

    df = tmp.iloc[2:].reset_index(drop=True)
    df.columns = cols
    # drop all-blank cols
    keep = [c for c in df.columns if not df[c].astype(str).str.strip().eq("").all()]
    return df[keep]

def fast_read_csv(path: str) -> pd.DataFrame:
    # Use PyArrow engine for speed and low overhead
    try:
        return pd.read_csv(path, dtype=str, engine="pyarrow").fillna("")
    except Exception:
        # fallback
        return pd.read_csv(path, dtype=str).fillna("")

def load_carrier_data(issuer: str) -> Tuple[pd.DataFrame, str]:
    k = issuer.strip().lower()
    if k not in CARRIERS:
        raise ValueError(f"Unsupported issuer '{issuer}'. Known: {list(CARRIERS)}")
    cfg = CARRIERS[k]
    raw_path = Path(cfg["raw_path"]).as_posix()
    if cfg["loader"] == "two_header":
        raw_df = flatten_two_header_csv(raw_path)
    else:
        raw_df = fast_read_csv(raw_path)
    return raw_df, cfg["compiled_rules_path"]

###############################################################################
# HEADER / UTILS
###############################################################################

def _build_header_index(df: pd.DataFrame) -> Dict[str, str]:
    return {re.sub(r"[^a-z0-9]", "", str(h).lower()): h for h in df.columns}

def _get_col(df: pd.DataFrame, header_index: Dict[str, str], name: Optional[str]) -> pd.Series:
    if not name:
        return pd.Series([""] * len(df), index=df.index, dtype="string")
    if name in df.columns:
        return df[name].astype(str)
    for h in df.columns:
        if h.lower() == name.lower():
            return df[h].astype(str)
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if key in header_index:
        return df[header_index[key]].astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="string")

def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

###############################################################################
# VECTORIZED HELPERS (unchanged logic)
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
        if len(parts) >= 2: return " ".join(parts[1:] + parts[:1])
        return name

    normalized = swapped.where(~comma, swapped.map(_normalize))
    tokens = normalized.str.split()
    last = tokens.str[-1].fillna("")
    first = tokens.apply(lambda xs: " ".join(xs[:-1]) if isinstance(xs, list) and len(xs) > 1 else "").fillna("")
    return first.str.title().astype("string"), last.str.title().astype("string")

def _money_to_float_str(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()
    neg_paren = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"[\$,()]", "", regex=True).str.strip()
    num = pd.to_numeric(x, errors="coerce")
    num = np.where(neg_paren, -num, num)
    # Note: using numpy where, then cast to object->string
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

###############################################################################
# SPEC HANDLING (hot path uses already-bound SOURCES)
###############################################################################

def _is_blank_spec(spec: Any) -> bool:
    return not isinstance(spec, dict) or str(spec.get("op", "")).strip().lower() in ("", "blank")

def _promote_pid_to_ptd(spec: Dict[str, Any]) -> Dict[str, Any]:
    # If a historical spec used "PID", allow it to drive PTD when PTD is blank
    if "PID" in spec and ("PTD" not in spec or _is_blank_spec(spec.get("PTD"))):
        spec["PTD"] = spec["PID"]
    return spec

def _header_signature(df: pd.DataFrame) -> str:
    # Stable signature of headers to detect when we can reuse bound rules
    joined = "||".join([str(c) for c in df.columns])
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]

def _load_compiled_rules(compiled_rules_path: str) -> Dict[str, Any]:
    p = Path(compiled_rules_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Compiled rules not found at {compiled_rules_path}. "
            f"Generate once (LLM step) and save, then rerun fast path."
        )
    return json.loads(p.read_text(encoding="utf-8"))

def _ensure_sources_present(bound_spec: Dict[str, Any]) -> Dict[str, Any]:
    # Fast path assumes 'source' already bound for ops that need it; ensure keys exist to avoid KeyErrors.
    fixed = {}
    for tgt, spec in bound_spec.items():
        if not isinstance(spec, dict):
            fixed[tgt] = {"op": "blank"}
            continue
        op = str(spec.get("op", "")).strip()
        if op in {"copy","date_mmddyyyy","date_plus_1m_mmddyyyy","name_first_from_full","name_last_from_full","money","membercount_from_commission"}:
            src = (spec.get("source") or "").strip()
            if not src:
                fixed[tgt] = {"op": "blank"}
                continue
        fixed[tgt] = spec
    return _promote_pid_to_ptd(fixed)

###############################################################################
# EXECUTOR (vectorized)
###############################################################################

def apply_rules_vectorized(df: pd.DataFrame, rule_spec: Dict[str, Any]) -> pd.DataFrame:
    header_index = _build_header_index(df)

    def empty() -> pd.Series:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    out: Dict[str, pd.Series] = {}

    for tgt in FINAL_COLUMNS:
        spec = rule_spec.get(tgt) or (rule_spec.get("PID") if tgt == "PTD" else None)

        if tgt == "PTD":
            pid_spec = rule_spec.get("PID")
            if _is_blank_spec(spec) and not _is_blank_spec(pid_spec):
                spec = pid_spec

        if not isinstance(spec, dict):
            out[tgt] = empty(); continue
        op = str(spec.get("op", "")).strip()
        if op not in ALLOWED_OPS:
            out[tgt] = empty(); continue

        if op == "copy":
            out[tgt] = _get_col(df, header_index, spec.get("source"))
        elif op == "const":
            val = str(spec.get("value", ""))
            out[tgt] = pd.Series([val] * len(df), index=df.index, dtype="string")
        elif op == "date_mmddyyyy":
            out[tgt] = _to_mmddyyyy(_get_col(df, header_index, spec.get("source")))
        elif op == "date_plus_1m_mmddyyyy":
            out[tgt] = _add_one_month_mmddyyyy(_get_col(df, header_index, spec.get("source")))
        elif op == "name_first_from_full":
            out[tgt] = _parse_case_name_first_last(_get_col(df, header_index, spec.get("source")))[0]
        elif op == "name_last_from_full":
            out[tgt] = _parse_case_name_first_last(_get_col(df, header_index, spec.get("source")))[1]
        elif op == "money":
            out[tgt] = _money_to_float_str(_get_col(df, header_index, spec.get("source")))
        elif op == "membercount_from_commission":
            flags = _sign_flag_from_money(_get_col(df, header_index, spec.get("source")))
            out[tgt] = pd.Series(np.where(flags.eq("1"), "1", flags), index=df.index, dtype="string")
        elif op == "blank":
            out[tgt] = empty()

    return pd.DataFrame(out, columns=FINAL_COLUMNS).fillna("").astype("string")

###############################################################################
# OPTIONAL: RAY PARALLELIZATION (only when it's worth it)
###############################################################################

def _should_use_ray(n_rows: int) -> bool:
    if ENABLE_RAY == "on":
        return True
    if ENABLE_RAY == "off":
        return False
    # auto: only if big enough
    return n_rows >= RAY_MIN_ROWS_TO_USE

def apply_rules_parallel(df: pd.DataFrame, bound_spec: Dict[str, Any]) -> pd.DataFrame:
    import ray  # import inside to avoid import cost when not used
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    @ray.remote
    def _apply(chunk: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
        return apply_rules_vectorized(chunk, spec)

    n = len(df)
    if n == 0:
        return df.iloc[0:0]

    parts: List[pd.DataFrame] = []
    splits = np.array_split(df, min(RAY_PARTITIONS, max(1, df.shape[0] // max(1, n // RAY_PARTITIONS))))
    futures = [_apply.remote(part, bound_spec) for part in splits]
    parts = ray.get(futures)
    out = pd.concat(parts, ignore_index=True)
    return out

###############################################################################
# MAIN (hot path)
###############################################################################

def main():
    # 1) Load raw + precompiled rules
    raw_df, compiled_rules_path = load_carrier_data(ISSUER)

    # 2) Build a header signature and pick a cache path (optional: per-header cache)
    sig = _header_signature(raw_df)
    per_header_cache = Path(compiled_rules_path).with_name(
        Path(compiled_rules_path).stem + f"__{sig}.json"
    )

    # Prefer exact header-bound cache, else fall back to generic compiled rules
    if per_header_cache.exists():
        bound_spec = json.loads(per_header_cache.read_text(encoding="utf-8"))
    else:
        bound_spec = _load_compiled_rules(compiled_rules_path)
        bound_spec = _ensure_sources_present(bound_spec)
        # Save a fast per-header copy for next time (no binding needed again)
        try:
            per_header_cache.write_text(json.dumps(bound_spec, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # 3) Execute transform (vectorized / or Ray if large)
    use_ray = _should_use_ray(len(raw_df))
    if use_ray:
        out_df = apply_rules_parallel(raw_df, bound_spec)
    else:
        out_df = apply_rules_vectorized(raw_df, bound_spec)

    # 4) Set constant columns required for this run
    out_df["TranDate"] = TRANDATE
    out_df["PayCode"]  = PAYCODE
    out_df["Issuer"]   = ISSUER

    # 5) Done — return or write (keep I/O tiny)
    # Example: write fast parquet (optional)
    out_path = Path("./outbound") / f"{ISSUER}_{sig}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_parquet(out_path, index=False)
    except Exception:
        # fallback to CSV if pyarrow not present
        out_path = out_path.with_suffix(".csv")
        out_df.to_csv(out_path, index=False)

    print(f"Rows: {len(out_df):,}  | Ray: {use_ray}  | Out: {out_path.as_posix()}")

if __name__ == "__main__":
    main()
