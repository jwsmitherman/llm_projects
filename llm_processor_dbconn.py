from manhattan_mapping import get_manhattan_mapping
import pandas as pd

def run_llm_pipeline(
    *, issuer: str, paycode: str, trandate: str,
    csv_path: str, template_dir: str,
    log: Callable[[str], None]
) -> str:
    """
    Returns output file path. Logs progress via `log(line)`.
    `template_dir` should contain `<issuer>_prompt.txt` and `<issuer>_rules.json`.
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

    # 2) LLM generate or load cached compiled rules
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

    # 3) Minimal IO re-read
    usecols = collect_usecols(bound_spec)
    df = _read_csv_usecols(csv_path, usecols if usecols else None, loader)
    n = len(df)
    log(f"[IO] Rows loaded: {n:,} | usecols={len(usecols)} | loader={loader}")

    # 4) Transform
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

    # 6) Manhattan Life enrichment
    if issuer == "Manhattan Life":
        log("[INFO] Manhattan Life detected — retrieving SQL mapping.")
        try:
            map_df = get_manhattan_mapping(
                load_task_id=13449,
                company_issuer_id=2204,
                log=log
            )

            if not map_df.empty and "PlanCode" in df.columns:
                map_df = map_df.drop_duplicates(subset=["PlanCode"]).copy()
                map_df.index = map_df["PlanCode"].astype(str).str.strip()

                src_key = df["PlanCode"].astype(str).str.strip()
                mapped_policy = src_key.map(map_df["PolicyNumber"]).fillna("")
                mapped_name = src_key.map(map_df["ProductName"]).fillna("")

                if "ProductType" not in out_df.columns:
                    out_df["ProductType"] = ""
                if "PlanName" not in out_df.columns:
                    out_df["PlanName"] = ""

                out_df["ProductType"] = mapped_policy.where(mapped_policy.ne(""), out_df["ProductType"])
                out_df["PlanName"] = mapped_name.where(mapped_name.ne(""), out_df["PlanName"])
                log(f"[ManhattanLife] Updated {sum(mapped_policy!='')} ProductType/PlanName rows.")
            else:
                log("[ManhattanLife] No PlanCode column or mapping data — skipped.")
        except Exception as e:
            log(f"[WARN] Manhattan Life enrichment failed: {e}")

    # 7) Write output file (same as before)
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


# 6) Manhattan Life enrichment
if issuer == "Manhattan Life":
    log("[INFO] Manhattan Life detected — retrieving SQL mapping.")
    try:
        map_df = get_manhattan_mapping(
            load_task_id=13449,
            company_issuer_id=2204,
            log=log
        )

        # Identify PlanCode column dynamically (in case it's named slightly differently)
        plan_code_col = next((c for c in df.columns if "plan" in c.lower() and "code" in c.lower()), None)

        if not map_df.empty and plan_code_col:
            log(f"[ManhattanLife] Found PlanCode column '{plan_code_col}' and {len(map_df)} mapping rows.")

            # Prepare lookup DataFrame
            map_df = map_df.drop_duplicates(subset=["PlanCode"]).copy()
            map_df.index = map_df["PlanCode"].astype(str).str.strip()

            # Normalize raw source keys
            src_key = df[plan_code_col].astype(str).str.strip()
            mapped_policy = src_key.map(map_df["PolicyNumber"]).fillna("Unknown")
            mapped_name   = src_key.map(map_df["ProductName"]).fillna("Unknown")

            # Ensure columns exist
            if "ProductType" not in out_df.columns:
                out_df["ProductType"] = ""
            if "PlanName" not in out_df.columns:
                out_df["PlanName"] = ""

            # Update mapped fields
            out_df["ProductType"] = mapped_policy.where(mapped_policy.ne(""), "Unknown")
            out_df["PlanName"]    = mapped_name.where(mapped_name.ne(""), "Unknown")

            updated_count = (mapped_policy != "Unknown").sum()
            log(f"[ManhattanLife] Updated {updated_count} ProductType/PlanName rows.")
        else:
            # No mapping data or missing PlanCode column
            log("[ManhattanLife] No PlanCode column or mapping data — populating as 'Unknown'.")
            out_df["ProductType"] = "Unknown"
            out_df["PlanName"] = "Unknown"

    except Exception as e:
        log(f"[WARN] Manhattan Life enrichment failed: {e}")
        out_df["ProductType"] = "Unknown"
        out_df["PlanName"] = "Unknown"
