from manhattan_mapping import is_manhattan_issuer
import pandas as pd
import pyodbc

def apply_manhattan_mapping_f(out_df: pd.DataFrame, raw_df: pd.DataFrame, log: Callable[[str], None]) -> pd.DataFrame:
    """Fetch SQL mapping for Manhattan Life and enrich ProductType/PlanName."""
    load_task_id = int(os.getenv("MANHATTAN_LOAD_TASK_ID", "13449"))
    company_issuer_id = int(os.getenv("MANHATTAN_COMPANY_ISSUER_ID", "2204"))
    server = os.getenv("SQL_SERVER", "QWVIDBSQLB401.ngquotit.com")
    database = os.getenv("SQL_DATABASE", "NGCS")
    driver = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    encrypt = os.getenv("SQL_ENCRYPT", "no").lower() in ("1","true","yes","y")
    trust = os.getenv("SQL_TRUST_SERVER_CERT", "yes").lower() in ("1","true","yes","y")

    conn_str = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        "Trusted_Connection=yes;"
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust else 'no'};"
    )

    log(f"[DB] Connecting to SQL Server: {server} | DB={database}")
    with pyodbc.connect(conn_str) as conn:
        sql = f"""
        SELECT DISTINCT
            p.IssuerPlanId   AS PlanCode,
            s.PolicyNumber,
            cpp.ProductName
        FROM dbo.STGPlanMappingFactors s (NOLOCK)
        OUTER APPLY (
            SELECT TOP 1 p.*
            FROM dbo.PlanMappingIssuerDetail p (NOLOCK)
            LEFT JOIN dbo.PlanMappingIssuerDetailState ps (NOLOCK)
                ON p.PlanMappingIssuerDetailId = ps.PlanMappingIssuerDetailId
            WHERE
                (p.IssuerPlanName = s.PlanName OR p.IssuerPlanName IS NULL OR p.IssuerPlanName = '')
                AND (p.IssuerPlanId = s.PlanId OR p.IssuerPlanId IS NULL OR p.IssuerPlanId = '')
                AND (p.[Year] = s.[Year] OR p.[Year] IS NULL)
                AND (p.PercentRate = s.RatePercent OR p.PercentRate IS NULL)
                AND (p.CurrencyRate = s.RateDollar OR p.CurrencyRate IS NULL)
                AND (ps.StateCode = s.StateCode OR ps.StateCode IS NULL)
            ORDER BY
                CASE WHEN p.IssuerPlanName = s.PlanName THEN 1 ELSE 0 END +
                CASE WHEN p.IssuerPlanId   = s.PlanId   THEN 1 ELSE 0 END +
                CASE WHEN p.PercentRate    = s.RatePercent THEN 1 ELSE 0 END +
                CASE WHEN p.[Year]         = s.[Year]   THEN 1 ELSE 0 END +
                CASE WHEN p.CurrencyRate   = s.RateDollar THEN 1 ELSE 0 END +
                CASE WHEN ps.StateCode     = s.StateCode THEN 1 ELSE 0 END
                DESC
        ) p
        JOIN dbo.CompanyProduct cp WITH (NOLOCK)
            ON cp.Deleted = 0
           AND cp.CompanyIssuerId = {company_issuer_id}
        JOIN dbo.CompanyProductPlan cpp WITH (NOLOCK)
            ON cpp.Deleted = 0
           AND cpp.CompanyProductId = cp.CompanyProductId
           AND p.CompanyProductPlanId = cpp.CompanyProductPlanId
        WHERE s.LoadTaskId = {load_task_id};
        """

        df_map = pd.read_sql(sql, conn).fillna("")
        log(f"[DB] Retrieved {len(df_map)} rows from mapping query.")

    if df_map.empty:
        log("[ManhattanLife] No mapping rows returned — skipping enrichment.")
        return out_df

    # Merge SQL results with out_df
    df_map = df_map.drop_duplicates(subset=["PlanCode"]).copy()
    df_map.index = df_map["PlanCode"].astype(str).str.strip()

    if "PlanCode" in raw_df.columns:
        src_key = raw_df["PlanCode"].astype(str).str.strip()
        mapped_policy = src_key.map(df_map["PolicyNumber"]).fillna("")
        mapped_name = src_key.map(df_map["ProductName"]).fillna("")

        if "ProductType" not in out_df.columns:
            out_df["ProductType"] = ""
        if "PlanName" not in out_df.columns:
            out_df["PlanName"] = ""

        out_df["ProductType"] = mapped_policy.where(mapped_policy.ne(""), out_df["ProductType"])
        out_df["PlanName"] = mapped_name.where(mapped_name.ne(""), out_df["PlanName"])
        log(f"[ManhattanLife] Updated {sum(mapped_policy!='')} ProductType/PlanName records.")
    else:
        log("[ManhattanLife] Column 'PlanCode' not found — enrichment skipped.")

    return out_df


# ----------------------------------------------------------------
# Main pipeline — identical to your processor.py, only 2 added blocks
# ----------------------------------------------------------------
def run_llm_pipeline(
    *, issuer: str, paycode: str, trandate: str,
    csv_path: str, template_dir: str,
    log: Callable[[str], None]
) -> str:
    """
    Returns output file path. Logs progress via `log(line)`.
    """
    start = time.perf_counter()
    log(f"Starting LLM pipeline | issuer={issuer} | csv={csv_path}")

    loader = CARRIERS.get(issuer, {}).get("loader", "csv")
    prompt_path = Path(template_dir) / f"{issuer}_prompt.txt"
    rules_path  = Path(template_dir) / f"{issuer}_rules.json"

    if not prompt_path.exists():
        log(f"NOTE: prompt file not found, continuing: {prompt_path}")
    if not rules_path.exists():
        log(f"NOTE: rules file not found, continuing: {rules_path}")

    # 1) Probe headers
    headers = _fast_read_header(csv_path, loader)
    sig = _sig_from_cols(headers)
    compiled_path = Path(template_dir) / f"{issuer}_compiled_rules__{sig}.json"

    # 2) Generate or load compiled rules
    if compiled_path.exists():
        bound_spec = json.loads(compiled_path.read_text(encoding="utf-8"))
        log(f"[Rules] Loaded cached compiled rules → {compiled_path.name}")
    else:
        log("[Rules] Generating with LLM…")
        raw_spec   = llm_generate_rule_spec(headers, prompt_path, rules_path)
        raw_spec   = canonicalize_spec_keys(raw_spec)
        bound_spec = bind_sources_to_headers(headers, raw_spec)
        bound_spec = promote_pid_to_ptd(bound_spec)
        compiled_path.write_text(json.dumps(bound_spec, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[Rules] Compiled & saved → {compiled_path.name}")

    # 3) Read minimal CSV
    usecols = collect_usecols(bound_spec)
    df = _read_csv_usecols(csv_path, usecols if usecols else None, loader)
    n = len(df)
    log(f"[IO] Rows loaded: {n:,} | usecols={len(usecols)} | loader={loader}")

    # 4) Transform with LLM rule logic
    use_ray = should_use_ray(n)
    if use_ray:
        log("[Exec] Using Ray parallel mode…")
        out_df = apply_rules_parallel(df, bound_spec)
    else:
        out_df = apply_rules(df, bound_spec)

    # 5) Add constants
    out_df["TranDate"] = trandate
    out_df["PayCode"]  = paycode
    out_df["Issuer"]   = issuer

    # 6) Manhattan Life enrichment (NEW)
    if is_manhattan_issuer(issuer):
        log("[INFO] Manhattan Life detected — fetching SQL enrichment data.")
        try:
            out_df = apply_manhattan_mapping_f(out_df, df, log)
        except Exception as e:
            log(f"[WARN] Manhattan Life enrichment failed: {e}")

    # 7) Write output file (unchanged)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / f"{issuer}_{sig}"
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
    log(f"Completed → {out_path.as_posix()} (elapsed {elapsed:.2f}s)")
    return out_path.as_posix()
