"""
build_benefits.py

Benefit-targeted parallel extraction.

Public API:  run_benefit_processing(pbp_rows, prompts) -> list[dict]

Strategy
--------
Instead of one huge LLM call (which fragments context across chunks) or one
gigantic single call (which is slow and may not fit), we:

  1. Pre-classify every PBP row into one or more benefit groups using the
     CMS PBP section codes embedded in the `category` field (e.g. "(7a)" → 900).
  2. Fire one focused LLM call per benefit group IN PARALLEL via asyncio.
  3. Each call sees only the rows relevant to its benefit + plan-wide context
     rows (Plan Type, FileName, MOOP, etc.), so input is small (~1-3K tokens).
  4. Each call returns just its rows. Concat the lot.

Result for ANY plan size: ~10-25s wall time (limited by slowest single call,
not number of benefits or rows). Each call is fast because input is small.
Accurate because the LLM sees focused, complete data per benefit.
"""

import asyncio
import json
import random
import re
import time
from collections import defaultdict
from typing import Optional

import httpx
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded config — UPDATE THE TWO PLACEHOLDERS BELOW BEFORE RUNNING
# ─────────────────────────────────────────────────────────────────────────────

AZURE_OPENAI_ENDPOINT    = "<YOUR_ENDPOINT>"
AZURE_OPENAI_API_KEY     = "<YOUR_API_KEY>"
AZURE_OPENAI_DEPLOYMENT  = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_TEMPERATURE = None   # None → omit (works on all model families)

# Carrier prefix override. Leave None to auto-detect from FileName.
CARRIER_PREFIX_OVERRIDE = None

# FileName-prefix → carrier ID. Add carriers as you onboard them.
CARRIER_PREFIX_MAP = {
    "humana":   "MOM",
    "aetna":    "AETNA",
    "uhc":      "UHC",
    "anthem":   "ANTHEM",
    "cigna":    "CIGNA",
    "wellcare": "WELLCARE",
    "kaiser":   "KAISER",
    "bcbs":     "BCBS",
}

# Concurrency. With 500K TPM available, we can run many calls in parallel.
# Each per-benefit call is ~3-5K tokens, so MAX_CONCURRENCY=24 peaks around
# ~120K TPM — well within a 500K-TPM deployment, with room for multiple plans
# to run simultaneously.
MAX_CONCURRENCY = 24

# How many plans to process simultaneously. Plans are independent, so we can
# run them in parallel within the same overall MAX_CONCURRENCY budget. Lower
# this if you have many plans per LoadID and want to avoid TPM bursts.
MAX_PLANS_IN_PARALLEL = 4

# Output cap per call. Each call returns 1-15 rows, so 2K is plenty.
LLM_MAX_TOKENS = 2000

# Retry config
MAX_RETRIES    = 3
BASE_BACKOFF_S = 2.0

# HTTP timeouts (seconds)
HTTP_CONNECT_TIMEOUT = 10.0
HTTP_READ_TIMEOUT    = 90.0
HTTP_WRITE_TIMEOUT   = 30.0

# Output column order
COLS = [
    "planid", "plantypeid", "benefitid", "benefitname",
    "coveragetypeid", "coveragetypedesc",
    "serviceTypeID", "serviceTypeDesc",
    "benefitdesc", "tinyDescription",
]


# ─────────────────────────────────────────────────────────────────────────────
# BENEFIT CATALOG
# Maps benefit IDs to their identifying PBP section codes and category keywords.
# One target per benefit ID = one parallel LLM call.
# ─────────────────────────────────────────────────────────────────────────────

BENEFIT_TARGETS = [
    # (benefitid, benefitname, [pbp_section_codes], [extra_category_keywords], coverage_type)
    # Plan-level / premium / deductibles ---------------------------------------
    ("600",  "Monthly Premium",                    [],         ["Monthly Premium", "Plan Premium"], "4/NA"),
    ("610",  "Health Plan Deductible",             [],         ["Health Plan Deductible", "Medical Deductible"], "1/InNetwork"),
    ("611",  "Drug Deductible",                    [],         ["Drug Deductible", "Rx Deductible", "Enter Deductible Amount"], "4/NA"),
    ("614",  "Health Monthly Premium",             [],         ["Health Monthly Premium", "Health Premium"], "4/NA"),
    ("615",  "Drug Monthly Premium",               [],         ["Drug Monthly Premium", "Rx Premium", "Part D Premium"], "4/NA"),
    ("616",  "Part B Premium Reduction",           [],         ["Part B Premium", "Part B Reduction", "Part B giveback"], "4/NA"),
    ("620",  "Out-of-Pocket Spending Limit",       [],         ["MOOP", "Max Enrollee Cost", "Out of Pocket"], "1/InNetwork"),
    # Pharmacy / Rx ------------------------------------------------------------
    ("700",  "Tier Names",                         [],         ["Rx Tier", "Formulary Tier", "Tier Names"], "4/NA"),
    ("710",  "Initial Coverage",                   [],         ["Initial Coverage Phase", "Rx Setup"], "3/General"),
    ("711",  "Retail Pharmacy",                    [],         ["Retail Pharmacy", "Initial Coverage Phase"], "1/InNetwork"),
    ("730",  "Catastrophic Coverage",              [],         ["Catastrophic Coverage"], "4/NA"),
    ("740",  "Formulary Exception",                [],         ["Formulary Exception"], "4/NA"),
    ("755",  "Initial Coverage Preferred Mail Order", [],      ["Preferred Mail Order"], "1/InNetwork"),
    ("760",  "Initial Coverage Standard Mail Order",  [],      ["Standard Mail Order"], "1/InNetwork"),
    # Inpatient / facility -----------------------------------------------------
    ("800",  "Inpatient Hospital Care",            ["1a"],     ["Inpatient Hospital"], "1/InNetwork"),
    ("810",  "Inpatient Mental Health Care",       ["1b"],     ["Inpatient Mental Health", "Inpatient Psychiatric"], "1/InNetwork"),
    ("820",  "Skilled Nursing Facility",           ["2"],      ["Skilled Nursing", "SNF"], "1/InNetwork"),
    # Professional / outpatient services ---------------------------------------
    ("900",  "Doctor Office Visits Primary",       ["7a"],     ["Primary Care", "PCP"], "1/InNetwork"),
    ("910",  "Doctor Office Visits Specialist",    ["7b"],     ["Specialist", "Specialty Care"], "1/InNetwork"),
    ("911",  "Telehealth",                         ["7d"],     ["Telehealth", "Telemedicine", "Virtual"], "1/InNetwork"),
    ("920",  "Chiropractic Services",              ["8"],      ["Chiropractic"], "1/InNetwork"),
    ("930",  "Podiatry Services",                  ["9a"],     ["Podiatry"], "1/InNetwork"),
    ("940",  "Outpatient Mental Health",           ["4a"],     ["Outpatient Mental Health", "Outpatient Psychiatric"], "1/InNetwork"),
    ("950",  "Outpatient Substance Abuse",         ["4b"],     ["Substance Abuse"], "1/InNetwork"),
    ("960",  "Outpatient Services/Surgery",        ["4c"],     ["Outpatient Surgery", "Outpatient Services", "Ambulatory Surgical"], "1/InNetwork"),
    ("970",  "Ambulance Services",                 ["5"],      ["Ambulance"], "1/InNetwork"),
    ("981",  "Emergency Care",                     ["6a"],     ["Emergency"], "4/NA"),
    ("982",  "Urgently Needed Care",               ["6b"],     ["Urgent Care", "Urgently Needed"], "4/NA"),
    ("990",  "Outpatient Rehabilitation",          ["10"],     ["Outpatient Rehabilitation", "Physical Therapy", "Occupational Therapy", "Speech Therapy", "Cardiac Rehab"], "1/InNetwork"),
    # Equipment / supplies / labs ----------------------------------------------
    ("1000", "Durable Medical Equipment",          ["11a"],    ["Durable Medical Equipment", "DME"], "1/InNetwork"),
    ("1020", "Diabetes Programs and Supplies",     ["11c"],    ["Diabetic", "Diabetes"], "1/InNetwork"),
    ("1030", "Diagnostic Tests, X-Rays, Lab Services and Radiology Services", ["3"], ["Diagnostic", "X-Ray", "Lab", "Radiology"], "1/InNetwork"),
    # Supplemental benefits ----------------------------------------------------
    ("1050", "Fitness",                            ["13b"],    ["Fitness", "SilverSneakers"], "1/InNetwork"),
    ("1060", "Meals",                              ["13c"],    ["Meals", "Meal Benefit"], "1/InNetwork"),
    ("1200", "Kidney Disease",                     ["12"],     ["Kidney", "Renal", "Dialysis"], "1/InNetwork"),
    ("1300", "Dental Services",                    ["16a"],    ["Preventive Dental", "Dental Services"], "1/InNetwork"),
    ("1301", "Dental - Comprehensive",             ["16b"],    ["Comprehensive Dental", "Non-routine"], "1/InNetwork"),
    ("1400", "Hearing Services",                   ["18a", "18b"], ["Hearing Exam", "Hearing Aid", "Fitting/Evaluation"], "1/InNetwork"),
    ("1500", "Vision Services",                    ["17a", "17b"], ["Vision", "Eyewear", "Eyeglasses", "Contact Lenses"], "1/InNetwork"),
    ("1610", "Prosthetic Devices",                 ["11b"],    ["Prosthetic"], "1/InNetwork"),
    ("1700", "Preventive Services",                ["14a", "14b", "14c"], ["Preventive", "Wellness Visit"], "1/InNetwork"),
    ("1800", "Transportation",                     ["15"],     ["Transportation"], "1/InNetwork"),
    ("1900", "Acupuncture",                        ["13a"],    ["Acupuncture"], "1/InNetwork"),
    ("2100", "Over-the-Counter Items",             ["13e"],    ["Over-the-Counter", "OTC"], "1/InNetwork"),
    # Star ratings (constant, not data-driven) ---------------------------------
    ("2110", "Medicare Overall Plan Rating",       [],         ["Plan Rating", "Star Rating"], "4/NA"),
    ("2111", "Medicare Health Plan Rating",        [],         ["Health Plan Rating"], "4/NA"),
    ("2112", "Medicare Drug Plan Rating",          [],         ["Drug Plan Rating"], "4/NA"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Filtering — pick out PBP rows relevant to one benefit
# ─────────────────────────────────────────────────────────────────────────────

# Categories whose rows are needed by EVERY benefit call (plan-wide context)
PLAN_LEVEL_HEADERS = {"Plan Characteristics", "Plan Level Cost Sharing"}


def _row_matches_target(row: dict, section_codes: list, keywords: list) -> bool:
    """True if this PBP row belongs to the target benefit."""
    category = (row.get("category") or "")
    cat_lower = category.lower()

    # Match by parenthesized section code: e.g. "(7a)", "(11c)"
    for code in section_codes:
        if f"({code})" in cat_lower or f"({code}1)" in cat_lower or f"({code}2)" in cat_lower:
            return True

    # Match by keyword in category
    for kw in keywords:
        if kw.lower() in cat_lower:
            return True

    return False


def _filter_rows_for_target(pbp_rows: list, target: tuple) -> list:
    """Return all PBP rows relevant to one benefit target."""
    _, _, section_codes, keywords, _ = target
    matched = [r for r in pbp_rows if _row_matches_target(r, section_codes, keywords)]
    return matched


def _plan_level_rows(pbp_rows: list) -> list:
    """Return rows that are needed by every benefit call (plan-wide context)."""
    return [r for r in pbp_rows if r.get("header") in PLAN_LEVEL_HEADERS]


# ─────────────────────────────────────────────────────────────────────────────
# Carrier detection + plan metadata
# ─────────────────────────────────────────────────────────────────────────────

def _detect_carrier_prefix(file_name: str) -> str:
    if CARRIER_PREFIX_OVERRIDE:
        return CARRIER_PREFIX_OVERRIDE
    if not file_name:
        return "UNKNOWN"
    fname_lower = file_name.lower()
    matches = [(k, v) for k, v in CARRIER_PREFIX_MAP.items() if fname_lower.startswith(k)]
    if matches:
        matches.sort(key=lambda kv: -len(kv[0]))
        return matches[0][1]
    return file_name.split("_")[0].upper() if "_" in file_name else file_name.upper()


def _extract_plan_meta(pbp_rows: list) -> dict:
    file_name = pbp_rows[0].get("FileName", "").strip() if pbp_rows else ""
    carrier_prefix = _detect_carrier_prefix(file_name)

    remainder = file_name[file_name.index("_") + 1:] if "_" in file_name else file_name
    parts = remainder.split("-")
    contract = parts[0] if len(parts) > 0 else ""
    plan = parts[1] if len(parts) > 1 else ""
    seg = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    planid = f"{carrier_prefix}_{contract}_{plan}_{seg}"

    plan_type = "HMO"
    for r in pbp_rows:
        if r.get("header") == "Plan Characteristics" and r.get("field") == "Plan Type":
            plan_type = r.get("value", "HMO").strip()
            break

    return {
        "file_name":      file_name,
        "planid":         planid,
        "plan_type":      plan_type,
        "carrier_prefix": carrier_prefix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Async LLM call with retry
# ─────────────────────────────────────────────────────────────────────────────

async def _call_llm_async(client: httpx.AsyncClient, system_message: str,
                          human_text: str, label: str) -> str:
    url = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    body = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_text},
        ],
        "max_completion_tokens": LLM_MAX_TOKENS,
    }
    if AZURE_OPENAI_TEMPERATURE is not None:
        body["temperature"] = AZURE_OPENAI_TEMPERATURE

    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(url, headers=headers, json=body)

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_ms = resp.headers.get("retry-after-ms")
                retry_s = resp.headers.get("retry-after")
                if retry_ms and retry_ms.isdigit():
                    sleep_s = int(retry_ms) / 1000.0
                elif retry_s and retry_s.isdigit():
                    sleep_s = float(retry_s)
                else:
                    sleep_s = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(sleep_s)
                continue

            if 400 <= resp.status_code < 500:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text
                raise RuntimeError(f"HTTP {resp.status_code}: {err_body}")

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.RequestError as e:
            last_err = e
            await asyncio.sleep(BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1))

    raise RuntimeError(f"[{label}] exhausted {MAX_RETRIES} retries: {last_err}")


def _parse_json_array(raw: str, label: str) -> list:
    text = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        # The model can legitimately return an empty array as "[]" — handle
        # the case where it returns just "no data" prose
        if "no data" in text.lower() or "not applicable" in text.lower() or text == "":
            return []
        raise ValueError(f"[{label}] No JSON array. Preview: {raw[:200]!r}")
    return json.loads(text[start:end])


# ─────────────────────────────────────────────────────────────────────────────
# Per-benefit processor
# ─────────────────────────────────────────────────────────────────────────────

PER_BENEFIT_PROMPT_TEMPLATE = """\
You are extracting ONE specific Medicare Advantage benefit from a subset of PBP rows.

Target benefit:
  benefitid       = {benefit_id}
  benefitname     = {benefit_name}
  default coverage = {coverage_type_id}/{coverage_type_desc}

Plan context:
  carrier_prefix  = "{carrier_prefix}"
  planid          = "{planid}"
  plantypeid      = "{plan_type}"
  file_name       = "{file_name}"

PLAN-WIDE PBP ROWS (use for context, e.g. MOOP, Plan Type):
{plan_level_json}

BENEFIT-SPECIFIC PBP ROWS ({n_specific} rows for this benefit):
{specific_json}

INSTRUCTIONS:
- Produce a JSON array of benefit rows for benefitid={benefit_id} ONLY.
- Some benefits produce multiple rows (e.g. tiers, day ranges, service types).
  Output one row per (serviceTypeID, coveragetypeid) combination found.
- If the data shows the benefit is not covered by this plan, output one row
  with benefitdesc="Not Covered" and tinyDescription="Not Covered".
- If the benefit-specific rows are empty AND there's no plan-wide data
  indicating this benefit, output an empty array: [].
- Apply ALL formatting rules from the system prompt (HTML <b> tags, periodicity
  normalization, $0.00 → $0, etc.).
- Always populate planid="{planid}" and plantypeid="{plan_type}" on every row.

Return ONLY a valid JSON array. No markdown, no explanation.
"""


async def _process_benefit(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    system_message: str,
    target: tuple,
    plan_level_rows: list,
    pbp_rows: list,
    meta: dict,
) -> tuple:
    """Returns (benefitid, list_of_rows). Empty list on failure or no data."""
    benefit_id, benefit_name, _, _, coverage = target
    cov_id, cov_desc = coverage.split("/", 1)
    label = f"benefit {benefit_id} {benefit_name}"

    specific_rows = _filter_rows_for_target(pbp_rows, target)

    # Skip the LLM call entirely if there's nothing relevant for this benefit
    if not specific_rows and not _benefit_might_use_plan_level_only(benefit_id):
        print(f"  [{label}] skipped (no matching PBP rows)")
        return benefit_id, []

    human_text = PER_BENEFIT_PROMPT_TEMPLATE.format(
        benefit_id=benefit_id,
        benefit_name=benefit_name,
        coverage_type_id=cov_id,
        coverage_type_desc=cov_desc,
        carrier_prefix=meta["carrier_prefix"],
        planid=meta["planid"],
        plan_type=meta["plan_type"],
        file_name=meta["file_name"],
        plan_level_json=json.dumps(plan_level_rows, indent=2),
        n_specific=len(specific_rows),
        specific_json=json.dumps(specific_rows, indent=2),
    )

    async with sem:
        t0 = time.monotonic()
        try:
            raw = await _call_llm_async(client, system_message, human_text, label)
            rows = _parse_json_array(raw, label)
            elapsed = time.monotonic() - t0
            print(f"  [{label}] {elapsed:.1f}s → {len(rows)} rows "
                  f"(input: {len(specific_rows)} pbp rows)")
            return benefit_id, rows
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  [{label}] ERROR after {elapsed:.1f}s: {e}")
            return benefit_id, []


# Some benefits (premiums, deductibles, MOOP, star ratings) live in plan-level
# data even when no benefit-specific rows match. Allow them to call the LLM
# even with empty specific_rows.
_PLAN_LEVEL_ONLY_BENEFITS = {
    "600", "610", "611", "614", "615", "616", "620",
    "700", "710", "711", "730", "740", "755", "760",
    "2110", "2111", "2112",
}


def _benefit_might_use_plan_level_only(benefit_id: str) -> bool:
    return benefit_id in _PLAN_LEVEL_ONLY_BENEFITS


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Multi-plan grouping
# ─────────────────────────────────────────────────────────────────────────────

def _group_rows_by_plan(pbp_rows: list) -> dict:
    """
    Group PBP rows by FileName. One LoadID typically contains multiple plans
    (different contract/plan/segment combinations), each with its own FileName.

    Returns: {file_name: [rows]} preserving original row order within each plan.
    """
    groups: dict = defaultdict(list)
    for row in pbp_rows:
        fname = (row.get("FileName") or "").strip()
        if not fname:
            # Rows without a FileName get attached to a sentinel bucket; warn
            # and put them in a plan called "_unknown" so they're not silently lost
            fname = "_unknown"
        groups[fname].append(row)
    return dict(groups)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def _run_one_plan(
    sem_per_call: asyncio.Semaphore,
    client: httpx.AsyncClient,
    system_message: str,
    plan_rows: list,
) -> list:
    """Run benefit-targeted extraction for ONE plan. Returns its benefit rows."""
    meta = _extract_plan_meta(plan_rows)
    plan_level_rows = _plan_level_rows(plan_rows)

    print(f"\n--- Plan: {meta['planid']} ({meta['carrier_prefix']}, {meta['plan_type']}) "
          f"| {len(plan_rows):,} rows | plan-level: {len(plan_level_rows)} ---")

    t0 = time.monotonic()
    tasks = [
        _process_benefit(sem_per_call, client, system_message, target,
                         plan_level_rows, plan_rows, meta)
        for target in BENEFIT_TARGETS
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    plan_rows_out: list = []
    for _, rows in results:
        plan_rows_out.extend(rows)

    benefits_with_rows = sum(1 for _, rows in results if rows)
    print(f"--- {meta['planid']} done: {len(plan_rows_out)} rows from "
          f"{benefits_with_rows}/{len(BENEFIT_TARGETS)} benefits in {elapsed:.1f}s ---")
    return plan_rows_out


async def _run_async(pbp_rows: list, prompts: dict) -> list:
    system_message = prompts["system_prompt"] + "\n\n" + prompts["few_shot_examples"]

    # Group rows by FileName — each unique FileName = one plan
    plan_groups = _group_rows_by_plan(pbp_rows)
    n_plans = len(plan_groups)

    print(f"\n=== LoadID has {n_plans} plan(s) "
          f"({len(pbp_rows):,} total PBP rows) ===")
    for fname, rows in plan_groups.items():
        print(f"   - {fname}: {len(rows):,} rows")
    print(f"max_concurrency (per-call): {MAX_CONCURRENCY} | "
          f"max_plans_in_parallel: {MAX_PLANS_IN_PARALLEL}")

    limits = httpx.Limits(
        max_connections=MAX_CONCURRENCY * 2,
        max_keepalive_connections=MAX_CONCURRENCY * 2,
    )
    timeout = httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT,
        read=HTTP_READ_TIMEOUT,
        write=HTTP_WRITE_TIMEOUT,
        pool=10.0,
    )

    # One semaphore caps total in-flight LLM calls across all plans
    sem_per_call = asyncio.Semaphore(MAX_CONCURRENCY)
    # Another semaphore caps how many plans run at once
    sem_plan = asyncio.Semaphore(MAX_PLANS_IN_PARALLEL)

    async def _bounded_plan_run(plan_rows):
        async with sem_plan:
            return await _run_one_plan(sem_per_call, client, system_message, plan_rows)

    t0 = time.monotonic()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        plan_tasks = [_bounded_plan_run(rows) for rows in plan_groups.values()]
        per_plan_results = await asyncio.gather(*plan_tasks)

    elapsed = time.monotonic() - t0

    all_rows: list = []
    for plan_rows_out in per_plan_results:
        all_rows.extend(plan_rows_out)

    print(f"\n=== ALL DONE: {len(all_rows)} total rows from {n_plans} plan(s) "
          f"in {elapsed:.1f}s ===\n")
    return all_rows


def run_benefit_processing(pbp_rows, prompts: dict) -> list:
    """
    Process PBP rows via benefit-targeted parallel extraction.

    Parameters
    ----------
    pbp_rows : list | dict   — raw PBP rows or input_json dict with "pbp" key
    prompts  : dict          — keys: system_prompt, few_shot_examples, human_template

    Returns
    -------
    list[dict] — benefit rows with the standard 10 columns
    """
    if isinstance(pbp_rows, dict) and "pbp" in pbp_rows:
        pbp_rows = pbp_rows["pbp"]

    rows = asyncio.run(_run_async(pbp_rows, prompts))

    return (
        pd.DataFrame(rows)
        .reindex(columns=COLS)
        .sort_values(["planid", "benefitid", "serviceTypeID"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
