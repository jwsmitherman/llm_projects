def run_llm_pipeline(
    *,
    issuer: str,
    paycode: str,
    trandate: str,
    load_task_id: str,
    company_issuer_id: str,
    csv_path: str,
    template_dir: str,
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
        compiled_path.write_text(json.dumps(bound_spec, ensure_ascii=False, indent=2, encoding="utf-8"))
        log(f"[Rules] Compiled & saved → {compiled_path.name}")

    # 3) Minimal IO re-read
    usecols = collect_usecols(bound_spec)
    df = _read_csv_usecols(csv_path, usecols if usecols else None, loader)
    n = len(df)
    log(f"[IO] Rows loaded: {n:,} | usecols={len(usecols) if usecols else 'ALL'} | loader={loader}")

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

    # 6) Manhattan Life enrichment (PlanCode → PlanName/ProductType)
    if issuer == "Manhattan Life":
        log("[INFO] Manhattan Life detected — retrieving SQL mapping.")
        try:
            map_df = get_manhattan_mapping(
                load_task_id=load_task_id,
                company_issuer_id=company_issuer_id,
                log=log
            )

            # Find the raw PlanCode column (name may vary)
            plan_code_col = next(
                (c for c in df.columns if "plan" in c.lower() and "code" in c.lower()),
                None
            )

            if not map_df.empty and plan_code_col:
                log(f"[ManhattanLife] Found PlanCode column: {plan_code_col} | mapping rows={len(map_df)}")

                # Validate expected columns in the mapping
                required_cols = {"PlanCode", "PlanName", "ProductType"}
                missing = required_cols - set(map_df.columns)
                if missing:
                    log(f"[WARN][ManhattanLife] Mapping table missing columns: {missing} (will use Unknown)")
                    # Create empty fallbacks if needed
                    for c in missing:
                        map_df[c] = ""

                # Normalize keys on both sides
                map_df = map_df.drop_duplicates(subset=["PlanCode"]).copy()
                map_df["PlanCode_norm"]    = map_df["PlanCode"].astype(str).str.strip().str.upper()
                map_df["PlanName_norm"]    = map_df["PlanName"].astype(str).str.strip()
                map_df["ProductType_norm"] = map_df["ProductType"].astype(str).str.strip()
                map_df = map_df.set_index("PlanCode_norm")

                src_key = df[plan_code_col].astype(str).str.strip().str.upper()

                mapped_plan_name   = src_key.map(map_df["PlanName_norm"]).fillna("Unknown")
                mapped_producttype = src_key.map(map_df["ProductType_norm"]).fillna("Unknown")

                # Ensure destination columns exist on out_df
                if "PlanName" not in out_df.columns:
                    out_df["PlanName"] = ""
                if "ProductType" not in out_df.columns:
                    out_df["ProductType"] = ""

                # Populate (respect Unknown semantics)
                out_df["PlanName"]    = mapped_plan_name.where(mapped_plan_name.ne(""), "Unknown")
                out_df["ProductType"] = mapped_producttype.where(mapped_producttype.ne(""), "Unknown")

                # Diagnostics
                matched = (mapped_plan_name != "Unknown").sum()
                log(f"[ManhattanLife] PlanCode matches applied: {matched}/{len(out_df)}")
                # A few not-matched examples to help debugging
                unmatched = sorted((set(src_key.unique()) - set(map_df.index)))[:5]
                if unmatched:
                    log(f"[ManhattanLife] Sample unmatched PlanCodes: {unmatched}")
            else:
                log("[ManhattanLife] No PlanCode column or empty mapping — setting defaults to 'Unknown'.")
                out_df["PlanName"]    = out_df.get("PlanName", "Unknown")
                out_df["ProductType"] = out_df.get("ProductType", "Unknown")

        except Exception as e:
            log(f"[WARN] Manhattan Life enrichment failed: {e}")
            out_df["PlanName"]    = "Unknown"
            out_df["ProductType"] = "Unknown"

    # 7) Write output file (same as before)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / f"{issuer}_{sig}"
    if OUT_FORMAT.lower() == "parquet":
        out_path = out_base.with_suffix(".parquet")
        try:
            comp = None if PARQUET_COMPRESSION.lower() == "none" else PARQUET_COMPRESSION
            out_df.to_parquet(out_path, index=False, compression=comp)
        except Exception:
            # Fallback to CSV if Parquet writer not available
            out_path = out_base.with_suffix(".csv")
            out_df.to_csv(out_path, index=False)
    else:
        out_path = out_base.with_suffix(".csv")
        out_df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - start
    log(f"Completed: {out_path.as_posix()} (elapsed {elapsed:.2f}s)")
    return out_path.as_posix()
