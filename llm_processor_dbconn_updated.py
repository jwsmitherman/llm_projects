import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import hashlib
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Optional: LangChain Azure OpenAI
try:
    from langchain_openai import AzureChatOpenAI
except Exception:  # legacy fallback
    try:
        from langchain.chat_models import AzureChatOpenAI  # type: ignore
    except Exception:
        AzureChatOpenAI = None  # type: ignore

# Optional: Ray for parallelism (not heavily used here, but kept for compatibility)
try:
    import ray
except Exception:  # pragma: no cover
    ray = None  # type: ignore


# ---------------------------------------------------------------------
# Perf / output config (env)
# ---------------------------------------------------------------------

ENABLE_RAY = os.getenv("ENABLE_RAY", "auto")  # "auto" | "on" | "off"
RAY_PARTITIONS = int(os.getenv("RAY_PARTITIONS", "8"))
RAY_MIN_ROWS_TO_USE = int(os.getenv("RAY_MIN_ROWS_TO_USE", "30000"))

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.getenv("OUT_DIR", BASE_DIR / "outbound")).resolve()
OUT_FORMAT = os.getenv("OUT_FORMAT", "csv")  # "parquet" | "csv"
PARQUET_COMPRESSION = os.getenv("PARQUET_COMPRESSION", "snappy")


# ---------------------------------------------------------------------
# Carrier config
# ---------------------------------------------------------------------

CARRIERS: Dict[str, Dict[str, Any]] = {
    "Molina": {"loader": "csv"},
    "Ameritas": {"loader": "csv"},
    "Manhattan Life": {"loader": "two_header"},  # 2-row header CSV
}


# ---------------------------------------------------------------------
# Schema / ops
# ---------------------------------------------------------------------

FINAL_COLUMNS: List[str] = [
    "PolicyNO",
    "PHFirst",
    "PHLast",
    "Status",
    "Issuer",
    "State",
    "ProductType",
    "PlanName",
    "PlanCode",        # <-- NEW: PlanCode included in FINAL_COLUMNS
    "SubmittedDate",
    "EffectiveDate",
    "TermDate",
    "Paysched",
    "PayCode",
    "WritingAgentID",
    "Premium",
    "CommPrem",
    "TranDate",
    "CommReceived",
    "PTD",
    "NoPayMon",
    "Membercount",
]

ALLOWED_OPS: List[str] = [
    "copy",
    "const",
    "date_mmddyyyy",
    "date_plus_1m_mmddyyyy",
    "name_first_from_full",
    "name_last_from_full",
    "money",
    "membercount_from_commission",
    "blank",
]

SYSTEM_PROMPT = """You are a data transformation agent.
Return STRICT JSON ONLY (no prose). The top-level JSON object must contain EXACTLY the required keys.
For each key return an object with:
- "op": one of [copy,const,date_mmddyyyy,date_plus_1m_mmddyyyy,name_first_from_full,name_last_from_full,
                 money,membercount_from_commission,blank]
- "source": the exact input column name when applicable (for ops that read input)
- "value": for const
If unclear, use {"op":"blank"}.
You MAY also include "PID" as a key if your rules produce it; downstream will map PID -> PTD.
Do not add extra keys. Do not omit required keys.
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _fast_read_header(path: str, loader: str) -> List[str]:
    """
    Fast probe: return list of column names without reading full file.
    Supports "csv" and "two_header".
    """
    p = str(path)
    if loader == "csv":
        try:
            dfo = pd.read_csv(p, nrows=0, dtype=str, engine="pyarrow", memory_map=True)
        except Exception:
            dfo = pd.read_csv(p, nrows=0, dtype=str, low_memory=False)
        return list(dfo.columns)

    # two_header: first two rows -> synthetic headers
    probe = pd.read_csv(p, header=None, nrows=2, dtype=str).fillna("")
    top, bottom = probe.iloc[0].tolist(), probe.iloc[1].tolist()

    ff: List[str] = []
    last = ""
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
        name = re.sub(r"\s+", " ", name).strip()
        cols.append(name or "unnamed")

    return cols


def _sig_from_cols(cols: List[str]) -> str:
    norm = [re.sub(r"\s+", " ", c.strip().lower()) for c in cols]
    raw = "|".join(norm).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _create_llm() -> AzureChatOpenAI:
    if AzureChatOpenAI is None:
        raise RuntimeError("AzureChatOpenAI is not available in this environment.")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not (deployment and endpoint and api_key):
        raise RuntimeError("Missing Azure OpenAI environment variables.")

    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        temperature=0.0,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------
# LLM spec generation
# ---------------------------------------------------------------------

def llm_generate_rule_spec(
    headers: List[str],
    prompt_path: Path,
    rules_path: Path,
) -> Dict[str, Any]:
    """
    Call Azure OpenAI to generate a mapping spec from headers.
    """
    extra_prompt = ""
    if prompt_path.exists():
        extra_prompt = prompt_path.read_text(encoding="utf-8")

    explicit_rules: Dict[str, Any] = {}
    if rules_path.exists():
        explicit_rules = json.loads(rules_path.read_text(encoding="utf-8"))

    required_fields = list(FINAL_COLUMNS) + ["PID"]
    payload = {
        "Headers": headers,
        "RequiredFields": required_fields,
        "AllowedOps": ALLOWED_OPS,
        "ExplicitFieldHints": explicit_rules,
        "ExtraPrompt": extra_prompt,
        "OutputFormat": "Return a JSON object keyed by RequiredFields (plus PID if used).",
    }

    llm = _create_llm()
    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    )

    content = getattr(resp, "content", str(resp))
    try:
        return json.loads(content)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"LLM did not return valid JSON:\n{content}") from e


# ---------------------------------------------------------------------
# Spec normalization / binding
# ---------------------------------------------------------------------

def canonicalize_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure spec has a dict per field with 'op', and optional 'source' / 'value'.
    """
    out: Dict[str, Any] = {}
    for field, rule in spec.items():
        if isinstance(rule, str):
            out[field] = {"op": rule}
        elif isinstance(rule, dict):
            op = rule.get("op")
            if op not in ALLOWED_OPS:
                op = "blank"
            obj = {"op": op}
            if "source" in rule:
                obj["source"] = rule["source"]
            if "value" in rule:
                obj["value"] = rule["value"]
            out[field] = obj
        else:
            out[field] = {"op": "blank"}
    return out


def bind_sources_to_headers(headers: List[str], spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to match the 'source' in each rule to an actual header name ignoring minor spacing.
    """
    normalized_header_map: Dict[str, str] = {}
    for h in headers:
        norm = re.sub(r"[^a-z0-9]", "", h.lower())
        normalized_header_map[norm] = h

    bound: Dict[str, Any] = {}
    for field, rule in spec.items():
        rule = dict(rule)
        src = rule.get("source")
        if isinstance(src, str):
            norm_src = re.sub(r"[^a-z0-9]", "", src.lower())
            if norm_src in normalized_header_map:
                rule["source"] = normalized_header_map[norm_src]
        bound[field] = rule
    return bound


def promote_pid_to_ptd(bound_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the spec includes a PID rule, copy it as PTD (Paid to Date).
    """
    out = dict(bound_spec)
    if "PID" in out:
        out["PTD"] = dict(out["PID"])
    return out


# ---------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------

def _apply_op(op: str, v: Any, row: pd.Series, rule: Dict[str, Any]) -> Any:
    if op == "blank":
        return ""
    if op == "const":
        return rule.get("value", "")
    if op == "copy":
        return v or ""
    if op == "date_mmddyyyy":
        if not v:
            return ""
        s = str(v).strip()
        # handle mm/dd/yyyy, yyyy-mm-dd, etc.
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", s)
        if m:
            mm, dd, yy = m.groups()
            if len(yy) == 2:
                yy = "20" + yy
            return f"{int(mm):02d}/{int(dd):02d}/{yy}"
        m2 = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
        if m2:
            yy, mm, dd = map(int, m2.groups())
            return f"{mm:02d}/{dd:02d}/{yy}"
        return s
    if op == "date_plus_1m_mmddyyyy":
        base = _apply_op("date_mmddyyyy", v, row, rule)
        if not base:
            return ""
        try:
            mm, dd, yy = map(int, base.split("/"))
            dt = pd.Timestamp(year=yy, month=mm, day=dd)
            dt2 = dt + relativedelta(months=1)
            return f"{dt2.month:02d}/{dt2.day:02d}/{dt2.year}"
        except Exception:
            return base
    if op == "name_first_from_full":
        s = str(v or "").strip()
        if not s:
            return ""
        parts = s.split()
        if len(parts) <= 1:
            return parts[0]
        # everything except last token
        return " ".join(parts[:-1])
    if op == "name_last_from_full":
        s = str(v or "").strip()
        if not s:
            return ""
        parts = s.split()
        return parts[-1]
    if op == "money":
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        try:
            return float(str(v).replace(",", "").strip())
        except Exception:
            return ""
    if op == "membercount_from_commission":
        # Membercount = 1 unless commission is negative then -1
        amt = _apply_op("money", v, row, rule)
        if amt == "":
            return 1
        try:
            return -1 if float(amt) < 0 else 1
        except Exception:
            return 1
    # default
    return v or ""


def apply_rules(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply compiled spec row-wise to df.
    """
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    spec = canonicalize_spec(spec)

    df2 = df.copy()
    for c in df2.columns:
        df2[c] = df2[c].astype(str)

    out_records: List[Dict[str, Any]] = []

    for _, row in df2.iterrows():
        rec: Dict[str, Any] = {}
        for field in FINAL_COLUMNS + ["PID"]:
            rule = spec.get(field, {"op": "blank"})
            op = rule.get("op", "blank")
            src = rule.get("source")
            val = None
            if src and src in row:
                val = row[src]
            rec[field] = _apply_op(op, val, row, rule)
        # If PID was generated, map to PTD
        if "PID" in rec and "PTD" in FINAL_COLUMNS:
            rec["PTD"] = rec.get("PID", "")
        out_records.append(rec)

    out_df = pd.DataFrame(out_records)

    # normalize column names (PolicyNO vs PolicyNo)
    if "PolicyNO" in out_df.columns and "PolicyNo" not in out_df.columns:
        out_df["PolicyNo"] = out_df["PolicyNO"]
    return out_df


# ---------------------------------------------------------------------
# Manhattan Life helpers
# ---------------------------------------------------------------------

def extract_manhattan_policy_plan_from_csv(csv_path: str, log: Callable[[str], None]) -> pd.DataFrame:
    """
    Read Manhattan Life raw CSV, flatten 2-row headers, return ['PolicyNumber','PlanCode'].
    """
    p = str(csv_path).strip().strip('"').strip("'")
    if p.startswith("/") and not p.startswith("\\\\"):
        # normalize single-leading backslash to UNC if needed
        p = "\\" + p
    log(f"[ManhattanLife] Reading raw CSV: {p}")

    read_kwargs = dict(dtype=str, engine="python")
    try:
        raw = pd.read_csv(p, header=[0, 1], encoding="utf-8-sig", **read_kwargs)
    except Exception:
        try:
            raw = pd.read_csv(p, header=0, encoding="utf-8-sig", **read_kwargs)
        except Exception:
            raw = pd.read_csv(p, header=0, encoding="latin1", **read_kwargs)

    # Flatten headers
    if isinstance(raw.columns, pd.MultiIndex):
        flat: List[str] = []
        for parts in raw.columns:
            parts = [str(x).strip() for x in parts if x is not None and str(x).strip() != ""]
            name = re.sub(r"\s+", " ", " ".join(parts)).strip()
            flat.append(name)
        raw.columns = flat
    else:
        raw.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.columns]

    norm = {c: re.sub(r"[^A-Za-z0-9]", "", c.lower()) for c in raw.columns}

    # PlanCode column
    plan_code_col: Optional[str] = None
    for col, nc in norm.items():
        if "plan" in nc and "code" in nc:
            plan_code_col = col
            break
    if plan_code_col is None:
        for col, nc in norm.items():
            if nc.startswith("plancode"):
                plan_code_col = col
                break

    # PolicyNumber column
    policy_col: Optional[str] = None
    for col, nc in norm.items():
        if "policynumber" in nc or "policyno" in nc or "casenumber" in nc:
            policy_col = col
            break

    if policy_col is None:
        candidates = [c for c in raw.columns if "policy" in c.lower() or "case" in c.lower()]
        if candidates:
            def numeric_ratio(series: pd.Series) -> float:
                s = series.dropna().astype(str).str.strip()
                if len(s) == 0:
                    return 0.0
                m = s.str.match(r"^\d{5,}$")
                return float(m.mean())

            policy_col = max(candidates, key=lambda c: numeric_ratio(raw[c]))

    if policy_col is None or plan_code_col is None:
        raise ValueError(
            "Could not locate Policy/PlanCode columns. "
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


# ---------------------------------------------------------------------
# DB helpers (templates – adjust proc names/params for your env)
# ---------------------------------------------------------------------

def _get_sql_connection(server: str, database: str):
    """
    Return a pyodbc connection if available. You can customize this to match your DSN / driver.
    """
    try:
        import pyodbc  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyodbc is required for DB operations.") from e

    driver = os.getenv("SQL_DRIVER", "{ODBC Driver 17 for SQL Server}")
    user = os.getenv("SQL_USER")
    password = os.getenv("SQL_PASSWORD")

    if user and password:
        conn_str = (
            f"DRIVER={driver};SERVER={server};DATABASE={database};"
            f"UID={user};PWD={password}"
        )
    else:
        # Integrated auth
        conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"

    return pyodbc.connect(conn_str)


def insert_stg_plan_mapping_min(
    df: pd.DataFrame,
    load_task_id: str,
    company_issuer_id: str,
    server: str,
    database: str,
    log: Callable[[str], None],
) -> int:
    """
    Insert PolicyNumber/PlanId mapping rows into a staging table via stored procedure.

    This is a template – adjust proc name and parameters to your environment.
    Expects df with columns: PolicyNumber, PlanId.
    """
    if df.empty:
        return 0

    conn = _get_sql_connection(server, database)
    cur = conn.cursor()
    inserted = 0

    try:
        for _, row in df.iterrows():
            policy = row["PolicyNumber"]
            plan_id = row["PlanId"]
            cur.execute(
                "EXEC dbo.spInsertStgPlanMappingMin ?, ?, ?, ?",
                load_task_id,
                company_issuer_id,
                policy,
                plan_id,
            )
            inserted += 1
        conn.commit()
        log(f"[DB] insert_stg_plan_mapping_min inserted rows: {inserted}")
    finally:
        cur.close()
        conn.close()

    return inserted


def get_manhattan_mapping(
    Load_task_id: str,
    company_issuer_id: str,
    server: str,
    database: str,
    log: Callable[[str], None],
) -> pd.DataFrame:
    """
    Fetch Manhattan Life mapping rows from DB.

    This is a template – adjust proc / query to your schema.
    Expected to return at least: PolicyNo, PlanName, ProductType, optionally PlanCode.
    """
    conn = _get_sql_connection(server, database)
    cur = conn.cursor()
    try:
        cur.execute(
            "EXEC dbo.spGetManhattanMapping ?, ?",
            Load_task_id,
            company_issuer_id,
        )
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame.from_records(rows, columns=cols)
        log(f"[DB] get_manhattan_mapping fetched rows: {len(df)}")
        return df
    finally:
        cur.close()
        conn.close()


# ---------------------------------------------------------------------
# Utility: match_llm_output_to_raw_counts
# ---------------------------------------------------------------------

def match_llm_output_to_raw_counts(raw_link_df: pd.DataFrame, out_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that if a PolicyNumber appears N times in the raw data,
    we have N rows in out_df for that PolicyNo.
    """
    if out_df.empty or raw_link_df.empty:
        return out_df

    counts = (
        raw_link_df["PolicyNumber"]
        .astype(str)
        .str.strip()
        .value_counts()
        .to_dict()
    )

    out_df = out_df.copy()
    out_df["PolicyNo"] = out_df["PolicyNo"].astype(str).str.strip()

    rows: List[pd.Series] = []
    for _, row in out_df.iterrows():
        pol = row["PolicyNo"]
        n = counts.get(pol, 1)
        for _ in range(int(n)):
            rows.append(row.copy())

    return pd.DataFrame(rows, columns=out_df.columns)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

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
    log: Callable[[str], None] = print,
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
        log(f"[Rules] Compiled rules written to: {compiled_path.name}")

    # 3) Load full CSV for transformation
    if loader == "csv":
        df = pd.read_csv(csv_path, dtype=str)
    else:
        # for two_header, we just read with default header; rules are based on flattened names
        df = pd.read_csv(csv_path, dtype=str)

    log(f"[Input] rows={len(df)}, cols={list(df.columns)}")

    # 4) Apply rules to get base out_df
    out_df = apply_rules(df, bound_spec)

    # 5) Add constants / overrides
    out_df["TranDate"] = trandate
    out_df["PayCode"] = paycode
    out_df["Issuer"] = issuer
    # ProductType / PlanName default to blank – DB mapping can fill them
    if "ProductType" not in out_df.columns:
        out_df["ProductType"] = ""
    if "PlanName" not in out_df.columns:
        out_df["PlanName"] = ""
    if "PlanCode" not in out_df.columns:
        out_df["PlanCode"] = ""

    # -----------------------------------------------------------------
    # Manhattan Life special handling: PlanCode + DB mapping
    # -----------------------------------------------------------------
    if issuer.lower().replace(" ", "") in {"manhattanlife", "manhattenlife"}:
        log("[ManhattanLife] Starting PlanCode / mapping enrichment step.")

        raw_link_df = extract_manhattan_policy_plan_from_csv(csv_path, log)
        # raw_link_df has PolicyNumber, PlanCode
        r = raw_link_df.copy()
        r["PolicyNo"] = r["PolicyNumber"]

        # Payload to DB: we send PlanId based on PlanCode
        payload = raw_link_df[["PolicyNumber", "PlanCode"]].copy()
        payload.columns = ["PolicyNumber", "PlanId"]
        # Insert to staging
        try:
            inserted = insert_stg_plan_mapping_min(
                payload,
                load_task_id=load_task_id,
                company_issuer_id=company_issuer_id,
                server=server_name,
                database=database_name,
                log=log,
            )
            log(f"[ManhattanLife] STG insert-min rows: {inserted}")
        except Exception as e:
            log(f"[ManhattanLife][WARN] insert_stg_plan_mapping_min failed: {e}")
            inserted = 0

        # Fetch mapping from DB (may be empty if DB not wired)
        try:
            map_df = get_manhattan_mapping(
                Load_task_id=load_task_id,
                company_issuer_id=company_issuer_id,
                server=server_name,
                database=database_name,
                log=log,
            )
        except Exception as e:
            log(f"[ManhattanLife][WARN] get_manhattan_mapping failed: {e}")
            map_df = pd.DataFrame()

        # Standard output columns including PlanCode
        out_df_cols = [
            "PolicyNo",
            "PHFirst",
            "PHLast",
            "Status",
            "Issuer",
            "State",
            "ProductType",
            "PlanName",
            "PlanCode",
            "SubmittedDate",
            "EffectiveDate",
            "TermDate",
            "Paysched",
            "PayCode",
            "WritingAgentID",
            "Premium",
            "CommPrem",
            "TranDate",
            "CommReceived",
            "PTD",
            "NoPayMon",
            "Membercount",
            "Note",
        ]

        # Ensure PolicyNo exists
        if "PolicyNo" not in out_df.columns and "PolicyNO" in out_df.columns:
            out_df["PolicyNo"] = out_df["PolicyNO"]

        out_df2 = out_df.copy()

        if map_df.empty:
            log("[ManhattanLife] No mapping returned from DB; using raw PlanCode only.")
            # Merge PlanCode from raw_link_df
            payload_df = raw_link_df[["PolicyNumber", "PlanCode"]].copy()
            payload_df.columns = ["PolicyNo", "PlanCode"]
            out_df2 = out_df2.merge(payload_df, on="PolicyNo", how="left")
            out_df2["PlanName"] = out_df2.get("PlanName", "")
            out_df2["ProductType"] = out_df2.get("ProductType", "")
            out_df2 = out_df2[out_df_cols]
            out_df2 = out_df2.fillna("")
            out_df = out_df2.copy()
            out_df = match_llm_output_to_raw_counts(r, out_df)
        else:
            # Normalize map_df columns
            cols_lower = {c.lower(): c for c in map_df.columns}
            need_cols = {}
            for key in ["policyno", "planname", "producttype", "plancode"]:
                if key in cols_lower:
                    need_cols[key] = cols_lower[key]

            # 1) Merge map_df (preferred)
            if "plancode" in need_cols:
                map_df2 = map_df[
                    [
                        need_cols["policyno"],
                        need_cols["planname"],
                        need_cols["producttype"],
                        need_cols["plancode"],
                    ]
                ]
                map_df2.columns = ["PolicyNo", "PlanName", "ProductType", "PlanCode"]
            else:
                map_df2 = map_df[
                    [
                        need_cols["policyno"],
                        need_cols["planname"],
                        need_cols["producttype"],
                    ]
                ]
                map_df2.columns = ["PolicyNo", "PlanName", "ProductType"]

            out_df2 = out_df2.merge(map_df2, on="PolicyNo", how="left")

            # 2) Merge raw PlanCode as fallback
            payload_df = raw_link_df[["PolicyNumber", "PlanCode"]].copy()
            payload_df.columns = ["PolicyNo", "PlanCode_raw"]
            out_df2 = out_df2.merge(payload_df, on="PolicyNo", how="left")

            # 3) If PlanCode from map_df is null, fill from raw PlanCode
            if "PlanCode" in out_df2.columns:
                out_df2["PlanCode"] = out_df2["PlanCode"].fillna(out_df2["PlanCode_raw"])
            else:
                out_df2["PlanCode"] = out_df2["PlanCode_raw"]

            out_df2.drop(columns=["PlanCode_raw"], inplace=True)

            out_df2 = out_df2[out_df_cols]
            out_df2 = out_df2.fillna("")
            out_df = out_df2.copy()
            out_df = match_llm_output_to_raw_counts(r, out_df)

    # -----------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------
    _ensure_out_dir()

    input_path = Path(csv_path)
    out_dir = input_path.parent
    if output_csv_name:
        out_path = out_dir / f"{output_csv_name}.csv"
    else:
        out_path = out_dir / (input_path.stem + "_out.csv")

    try:
        out_df.to_csv(out_path, index=False)
        log(f"[Output] file written sucessfully: {out_path}")
    except Exception as e:
        log(f"[ERROR] Failed to write output file: {e}")

    elapsed = time.perf_counter() - start
    log(f"Completed: {out_path.as_posix()} (elapsed {elapsed:.2f}s)")

    return out_path.as_posix()
