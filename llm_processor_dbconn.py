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



#################

# 6) Manhattan Life enrichment — query DB and join to out_df by PolicyNumber
if issuer == "Manhattan Life":
    log("[INFO] Manhattan Life detected — enriching PlanName/ProductType from DB via PolicyNumber.")
    try:
        # --- 6.1 Query DB ---
        # Requires: from manhattan_mapping import get_manhattan_mapping (imported at top of file)
        map_df = get_manhattan_mapping(
            load_task_id=load_task_id,
            company_issuer_id=company_issuer_id,
            log=log,
        )

        # --- 6.2 Validate required columns from SQL result ---
        needed_cols = {"PolicyNumber", "PlanName", "ProductType"}
        missing = needed_cols - set(map_df.columns)
        if missing:
            log(f"[WARN][ManhattanLife] SQL result missing columns: {missing}. Creating empty fallbacks.")
            for c in missing:
                map_df[c] = ""

        # --- 6.3 Normalize keys on SQL side ---
        map_norm = (
            map_df[["PolicyNumber", "PlanName", "ProductType"]]
            .copy()
        )
        map_norm["PolicyKey"] = map_norm["PolicyNumber"].astype(str).str.strip()
        map_norm = map_norm.drop_duplicates(subset=["PolicyKey"])

        # --- 6.4 Locate PolicyNumber column in out_df (allow common variants) ---
        if "PolicyNumber" in out_df.columns:
            policy_col = "PolicyNumber"
        elif "PolicyNo" in out_df.columns:
            policy_col = "PolicyNo"
        else:
            policy_col = next(
                (c for c in out_df.columns
                 if "policy" in c.lower() and ("number" in c.lower() or c.lower().endswith("no"))),
                None
            )

        if not policy_col:
            log("[ManhattanLife] No PolicyNumber/PolicyNo column on out_df; enrichment skipped.")
        else:
            # --- 6.5 Merge SQL values to out_df via PolicyNumber ---
            out_df["_PolicyKey"] = out_df[policy_col].astype(str).str.strip()

            out_df = out_df.merge(
                map_norm.rename(columns={"PolicyKey": "_PolicyKey"}),
                on="_PolicyKey",
                how="left",
                suffixes=("", "_sql")
            )

            # --- 6.6 Ensure destination columns exist ---
            if "PlanName" not in out_df.columns:
                out_df["PlanName"] = ""
            if "ProductType" not in out_df.columns:
                out_df["ProductType"] = ""

            # --- 6.7 Overwrite from SQL when present; otherwise keep existing; final fallback -> "Unknown" ---
            has_plan  = out_df["PlanName_sql"].notna() & (out_df["PlanName_sql"].astype(str).str.len() > 0)
            has_ptype = out_df["ProductType_sql"].notna() & (out_df["ProductType_sql"].astype(str).str.len() > 0)

            out_df.loc[has_plan,  "PlanName"]    = out_df.loc[has_plan,  "PlanName_sql"]
            out_df.loc[has_ptype, "ProductType"] = out_df.loc[has_ptype, "ProductType_sql"]

            out_df["PlanName"]    = out_df["PlanName"].replace("", "Unknown").fillna("Unknown")
            out_df["ProductType"] = out_df["ProductType"].replace("", "Unknown").fillna("Unknown")

            updated_rows = int((has_plan | has_ptype).sum())
            log(f"[ManhattanLife] Enrichment applied to {updated_rows} rows (by PolicyNumber).")

            # --- 6.8 Cleanup temp columns ---
            out_df.drop(columns=["PlanName_sql", "ProductType_sql", "_PolicyKey"], inplace=True, errors="ignore")

    except Exception as e:
        log(f"[WARN] Manhattan Life enrichment failed: {e}")
        if "PlanName" not in out_df.columns:
            out_df["PlanName"] = "Unknown"
        else:
            out_df["PlanName"] = out_df["PlanName"].replace("", "Unknown").fillna("Unknown")
        if "ProductType" not in out_df.columns:
            out_df["ProductType"] = "Unknown"
        else:
            out_df["ProductType"] = out_df["ProductType"].replace("", "Unknown").fillna("Unknown")


######

# 6) Manhattan Life enrichment — (A) RAW PlanCode -> map_df  (B) map_df -> out_df by PolicyNumber
if issuer == "Manhattan Life":
    log("[INFO] Manhattan Life detected — enriching from DB (PlanCode -> map_df -> PolicyNumber -> out_df).")
    try:
        # --- A) QUERY DB (works in your setup) ---
        # map_df columns expected: PlanCode, PolicyNumber, PlanName, ProductType
        map_df = get_manhattan_mapping(
            load_task_id=load_task_id,
            company_issuer_id=company_issuer_id,
            log=log,
        )

        # Validate columns from SQL
        for c in ("PlanCode", "PolicyNumber", "PlanName", "ProductType"):
            if c not in map_df.columns:
                log(f"[WARN][ManhattanLife] SQL missing '{c}', creating empty column.")
                map_df[c] = ""

        # --- A) RAW df: find columns and join to map_df on PlanCode ---
        plan_code_col = next((c for c in df.columns if "plan" in c.lower() and "code" in c.lower()), None)
        policy_raw_col = next((c for c in df.columns if "policy" in c.lower() and ("number" in c.lower() or c.lower().endswith("no"))), None)

        if not plan_code_col or not policy_raw_col:
            log(f"[ManhattanLife] Missing RAW columns | plan_code_col={plan_code_col} | policy_raw_col={policy_raw_col}. Skipping enrichment.")
        else:
            # Normalize keys
            raw_link = df[[policy_raw_col, plan_code_col]].copy()
            raw_link["PolicyKey"]   = raw_link[policy_raw_col].astype(str).str.strip()
            raw_link["PlanCodeKey"] = raw_link[plan_code_col].astype(str).str.strip().str.upper()

            map_norm = map_df[["PolicyNumber", "PlanCode", "PlanName", "ProductType"]].copy()
            map_norm["PolicyKey"]   = map_norm["PolicyNumber"].astype(str).str.strip()
            map_norm["PlanCodeKey"] = map_norm["PlanCode"].astype(str).str.strip().str.upper()
            map_norm = map_norm.drop_duplicates(subset=["PlanCodeKey"])

            # Join RAW -> SQL by PlanCode to produce (PolicyKey -> PlanName/ProductType)
            joined = raw_link.merge(
                map_norm[["PlanCodeKey", "PlanName", "ProductType"]],
                on="PlanCodeKey",
                how="left",
            )

            # Build lookup dicts keyed by PolicyNumber (PolicyKey)
            plan_by_policy   = dict(zip(joined["PolicyKey"], joined["PlanName"].fillna("")))
            ptype_by_policy  = dict(zip(joined["PolicyKey"], joined["ProductType"].fillna("")))

            # --- B) APPLY to out_df using PolicyNumber ---
            # Find Policy column in out_df
            if "PolicyNumber" in out_df.columns:
                policy_out_col = "PolicyNumber"
            elif "PolicyNo" in out_df.columns:
                policy_out_col = "PolicyNo"
            else:
                policy_out_col = next((c for c in out_df.columns if "policy" in c.lower() and ("number" in c.lower() or c.lower().endswith("no"))), None)

            if not policy_out_col:
                log("[ManhattanLife] No PolicyNumber/PolicyNo in out_df; cannot apply enrichment.")
            else:
                policy_series = out_df[policy_out_col].astype(str).str.strip()

                # Ensure destination columns
                if "PlanName" not in out_df.columns:    out_df["PlanName"] = ""
                if "ProductType" not in out_df.columns: out_df["ProductType"] = ""

                mapped_plan  = policy_series.map(plan_by_policy)
                mapped_ptype = policy_series.map(ptype_by_policy)

                has_plan  = mapped_plan.notna()  & (mapped_plan.astype(str).str.len()  > 0)
                has_ptype = mapped_ptype.notna() & (mapped_ptype.astype(str).str.len() > 0)

                # Overwrite from DB when present
                out_df.loc[has_plan,  "PlanName"]    = mapped_plan[has_plan]
                out_df.loc[has_ptype, "ProductType"] = mapped_ptype[has_ptype]

                # Final fallback
                out_df["PlanName"]    = out_df["PlanName"].replace("", "Unknown").fillna("Unknown")
                out_df["ProductType"] = out_df["ProductType"].replace("", "Unknown").fillna("Unknown")

                log(f"[ManhattanLife] Applied mappings — PlanName: {int(has_plan.sum())}, ProductType: {int(has_ptype.sum())}")

    except Exception as e:
        log(f"[WARN] Manhattan Life enrichment failed: {e}")
        if "PlanName" not in out_df.columns:    out_df["PlanName"] = "Unknown"
        if "ProductType" not in out_df.columns: out_df["ProductType"] = "Unknown"


####
from pathlib import Path
import pandas as pd
import re

def extract_manhattan_policy_plan_from_csv(csv_path: str, log) -> pd.DataFrame:
    """Read Manhattan Life raw CSV, flatten 2-row headers, return ['PolicyNumber','PlanCode']."""
    p = str(csv_path).strip().strip('"').strip("'")
    if p.startswith("\\") and not p.startswith("\\\\"):
        p = "\\" + p  # normalize single-leading backslash to UNC
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
        flat = []
        for parts in raw.columns:
            parts = [str(x).strip() for x in parts if x is not None and str(x).strip() != ""]
            name = re.sub(r"\s+", " ", " ".join(parts)).strip()
            flat.append(name)
        raw.columns = flat
    else:
        raw.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in raw.columns]

    # Normalized header map
    norm = {c: re.sub(r"[^a-z0-9]", "", c.lower()) for c in raw.columns}

    # ---- Find PlanCode (prefer headers containing both 'plan' and 'code')
    plan_code_col = None
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
    policy_col = None
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
            # if it’s clearly not numeric, fall back to first candidate anyway

    if policy_col is None or plan_code_col is None:
        raise ValueError(
            f"Could not locate Policy/PlanCode columns. "
            f"Seen headers: {list(raw.columns)[:20]}"
        )

    df2 = raw[[policy_col, plan_code_col]].copy()
    df2.columns = ["PolicyNumber", "PlanCode"]
    df2["PolicyNumber"] = df2["PolicyNumber"].astype(str).str.strip()
    df2["PlanCode"]     = df2["PlanCode"].astype(str).str.strip().str.upper()
    df2 = df2[df2["PolicyNumber"] != ""].reset_index(drop=True)

    log(f"[ManhattanLife] Extracted rows: {len(df2)} | cols -> PolicyNumber='{policy_col}', PlanCode='{plan_code_col}'")
    return df2



##############




out_df_col = [
    "PolicyNo", "PHFirst", "PHLast", "Status", "Issuer", "State",
    "ProductType", "PlanName", "SubmittedDate", "EffectiveDate", "TermDate",
    "PaySched", "PayCode", "WritingAgentID", "Premium", "CommPrem",
    "TranDate", "CommReceived", "PTD", "NoPayMon", "Membercount", "Note"
]

# Build fallback mapping from payload once
payload_df = raw_link_df[["PolicyNumber", "PlanId"]].copy()
payload_df.columns = ["PolicyNo", "PlanName"]   # PlanId used as PlanName fallback

if map_df.shape[0] == 0:
    # No rows from DB – use only payload (PlanId) mapping
    out_df = out_df.drop(columns=["PlanName"])
    out_df = out_df.merge(payload_df[["PolicyNo", "PlanName"]],
                          on="PolicyNo", how="left")
    out_df = out_df[out_df_col]
    out_df = out_df.fillna("")
else:
    # Some rows from DB – prefer map_df, fallback to payload for missing
    out_df = out_df.drop(columns=["PlanName"])

    # First merge with map_df (preferred source)
    out_df = out_df.merge(map_df[["PolicyNo", "PlanName"]],
                          on="PolicyNo", how="left")

    # Now merge payload and use it ONLY where map_df left PlanName as null
    out_df = out_df.merge(payload_df[["PolicyNo", "PlanName"]],
                          on="PolicyNo", how="left", suffixes=("", "_payload"))

    # If map_df PlanName is null, fill with payload PlanName (PlanId)
    out_df["PlanName"] = out_df["PlanName"].fillna(out_df["PlanName_payload"])
    out_df = out_df.drop(columns=["PlanName_payload"])

    out_df = out_df[out_df_col]
    out_df = out_df.fillna("")



######

import pandas as pd

def match_llm_output_to_raw_counts(raw_link_df, llm_df):
    """
    Ensures llm_df has exactly the same number of rows for each PolicyNumber
    as raw_link_df.

    - If llm_df has fewer rows → rows are repeated (cycled)
    - If llm_df has more rows  → rows are trimmed
    """

    # Count rows required per policy from raw_link_df
    target_counts = raw_link_df["PolicyNumber"].value_counts()

    adjusted = []

    for policy, target_n in target_counts.items():
        block = llm_df[llm_df["PolicyNumber"] == policy]

        if block.empty:
            # No output produced — create empty rows?
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

    return pd.concat(adjusted, ignore_index=True)


