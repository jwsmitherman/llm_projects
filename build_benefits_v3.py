"""
build_benefits.py

Benefit-targeted parallel extraction.

Public API:  run_benefit_processing(pbp_rows, prompts) -> list[dict]

Strategy
--------
Instead of one huge LLM call (which fragments context across chunks) or one
gigantic single call (which is slow and may not fit), we:

  1. Group rows by FileName — each FileName = one plan (a load can have many).
  2. For each plan, pre-classify PBP rows into ~47 benefit groups using CMS
     section codes embedded in `category` (e.g. "(7a)" → 900).
  3. Fire one focused LLM call per benefit IN PARALLEL via asyncio.
  4. Each call sees only the rows relevant to its benefit + plan-wide context
     rows (Plan Type, FileName, MOOP, etc.), so input is small (~1-3K tokens).
  5. Concat the lot — no dedup needed since each call owns a unique benefit.

Result for ANY plan size: ~10-25s wall time per plan. Multiple plans run in
parallel under MAX_PLANS_IN_PARALLEL.
"""

# Build version marker — printed on import so logs make it obvious which
# code is actually running. Bump this whenever you ship a meaningful change.
__BUILD_VERSION__ = "2026-05-12-sweeper-and-retry-v5"
print(f"[build_benefits] loaded build {__BUILD_VERSION__}")

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

# ─────────────────────────────────────────────────────────────────────────────
# BENEFIT CATALOG
# Hybrid approach: plan-level benefits stay hardcoded (no section codes in PBP),
# data-driven benefits get built from the input by mapping section codes →
# benefit IDs at runtime.
# ─────────────────────────────────────────────────────────────────────────────

# Plan-level benefits — these live in plan-level rows or are scattered across
# rows by keyword, not by section code. The "section codes" list is always
# empty; matching is keyword-only.
#
# NOTE: 600/614/2110/2111/2112 (Monthly Premium and Star Ratings) are
# intentionally omitted — these come from CMS plan registry / star ratings
# files, not from PBP data.
PLAN_LEVEL_TARGETS = [
    # (benefitid, benefitname, [section_codes], [keywords], coverage_type)
    ("610",  "Health Plan Deductible",                 [], ["Health Plan Deductible", "Medical Deductible", "Annual Plan Deductible"], "1/InNetwork"),
    ("611",  "Drug Deductible",                        [], ["Drug Deductible", "Rx Deductible", "Enter Deductible Amount"], "4/NA"),
    ("615",  "Drug Monthly Premium",                   [], ["Drug Monthly Premium", "Rx Premium", "Part D Premium"], "4/NA"),
    ("616",  "Part B Premium Reduction",               [], ["Part B Premium", "Part B Reduction", "Part B giveback"], "4/NA"),
    ("620",  "Out-of-Pocket Spending Limit",           [], ["MOOP", "Max Enrollee Cost", "Out of Pocket", "OOP"], "1/InNetwork"),
    ("700",  "Tier Names",                             [], ["Rx Tier", "Formulary Tier", "Tier Names"], "4/NA"),
    ("710",  "Initial Coverage",                       [], ["Initial Coverage Phase", "Rx Setup"], "3/General"),
    ("711",  "Retail Pharmacy",                        [], ["Retail Pharmacy", "Initial Coverage Phase"], "1/InNetwork"),
    ("730",  "Catastrophic Coverage",                  [], ["Catastrophic Coverage"], "4/NA"),
    ("740",  "Formulary Exception",                    [], ["Formulary Exception"], "4/NA"),
    ("755",  "Initial Coverage Preferred Mail Order",  [], ["Preferred Mail Order"], "1/InNetwork"),
    ("760",  "Initial Coverage Standard Mail Order",   [], ["Standard Mail Order"], "1/InNetwork"),
]

# Comprehensive CMS PBP section-code → benefit-ID map.
# Built from the official CMS PBP file schema. Each section code maps to
# exactly one benefit ID + benefit name + default coverage type.
#
# When the data-driven target builder finds a section code in the input
# categories that's in this map, it creates a target for that benefit. Section
# codes not in this map go to the sweeper pass — nothing is silently dropped.
SECTION_CODE_TO_BENEFIT = {
    # Inpatient / facility — section 1, 2
    "1a":     ("800",  "Inpatient Hospital - Acute",          "1/InNetwork"),
    "1b":     ("810",  "Inpatient Hospital - Psychiatric",    "1/InNetwork"),
    "2":      ("820",  "Skilled Nursing Facility",            "1/InNetwork"),
    # Diagnostic / lab / radiology — section 3, 8
    "3":      ("1030", "Diagnostic Tests, Labs, Radiology",   "1/InNetwork"),
    "8a":     ("1030", "Outpatient Diagnostic Procedures",    "1/InNetwork"),
    "8a1":    ("1030", "Outpatient X-Ray Services",           "1/InNetwork"),
    "8a2":    ("1030", "Outpatient Lab Services",             "1/InNetwork"),
    "8b":     ("1030", "Outpatient Radiology",                "1/InNetwork"),
    "8b1":    ("1030", "Outpatient Radiology - Diagnostic",   "1/InNetwork"),
    "8b2":    ("1030", "Outpatient Radiology - Therapeutic",  "1/InNetwork"),
    # Mental health outpatient + substance abuse — section 4
    "4a":     ("940",  "Outpatient Mental Health",            "1/InNetwork"),
    "4b":     ("950",  "Outpatient Substance Abuse",          "1/InNetwork"),
    "4c":     ("960",  "Outpatient Surgery",                  "1/InNetwork"),
    # Ambulance / Emergency — section 5, 6
    "5":      ("970",  "Ambulance Services",                  "1/InNetwork"),
    "5a":     ("970",  "Ground Ambulance",                    "1/InNetwork"),
    "5b":     ("970",  "Air Ambulance",                       "1/InNetwork"),
    "6a":     ("981",  "Emergency Care",                      "4/NA"),
    "6b":     ("982",  "Urgently Needed Care",                "4/NA"),
    # Professional / outpatient visits — section 7
    "7a":     ("900",  "Primary Care Physician Visits",       "1/InNetwork"),
    "7b":     ("910",  "Specialist Visits",                   "1/InNetwork"),
    "7c":     ("910",  "Occupational Therapy",                "1/InNetwork"),
    "7d":     ("911",  "Telehealth - Physician Specialist",   "1/InNetwork"),
    "7g":     ("910",  "Other Health Care Professional",      "1/InNetwork"),
    "7j":     ("911",  "Additional Telehealth Benefits",      "1/InNetwork"),
    # Chiropractic / Podiatry / Outpatient Hospital — section 8, 9
    "8":      ("920",  "Chiropractic Services",               "1/InNetwork"),
    "9a":     ("930",  "Podiatry Services",                   "1/InNetwork"),
    "9a1":    ("960",  "Outpatient Hospital Services",        "1/InNetwork"),
    "9a2":    ("960",  "Observation Services",                "1/InNetwork"),
    "9b":     ("960",  "Ambulatory Surgical Center",          "1/InNetwork"),
    "9d":     ("960",  "Outpatient Blood Services",           "1/InNetwork"),
    # Outpatient Rehab — section 10
    "10":     ("990",  "Outpatient Rehabilitation",           "1/InNetwork"),
    "10a":    ("990",  "Physical Therapy",                    "1/InNetwork"),
    "10b":    ("990",  "Speech Therapy",                      "1/InNetwork"),
    # Equipment / supplies / Diabetic — section 11, 12
    "11a":    ("1000", "Durable Medical Equipment",           "1/InNetwork"),
    "11b":    ("1610", "Prosthetic Devices",                  "1/InNetwork"),
    "11c":    ("1020", "Diabetes Programs and Supplies",      "1/InNetwork"),
    "11d":    ("1020", "Diabetic Therapeutic Shoes/Inserts",  "1/InNetwork"),
    "12":     ("1200", "Kidney Disease / Dialysis",           "1/InNetwork"),
    # Supplemental benefits — section 13, 14
    "13a":    ("1900", "Acupuncture",                         "1/InNetwork"),
    "13b":    ("2100", "Over-the-Counter Items",              "1/InNetwork"),
    "13c":    ("1060", "Meals",                               "1/InNetwork"),
    "13e":    ("2100", "Over-the-Counter Drugs",              "1/InNetwork"),
    "14a":    ("1700", "Preventive Services",                 "1/InNetwork"),
    "14b":    ("1700", "Annual Physical Exam",                "1/InNetwork"),
    "14c":    ("1700", "Preventive Services (other)",         "1/InNetwork"),
    "14c4":   ("1050", "Fitness Benefit",                     "1/InNetwork"),
    "14c7":   ("911",  "Remote Access Technologies",          "1/InNetwork"),
    "14e4":   ("1700", "Digital Rectal Exams",                "1/InNetwork"),
    "14e5":   ("1700", "EKG following Welcome Visit",         "1/InNetwork"),
    # Transportation / Part B drugs — section 15
    "15":     ("1800", "Transportation",                      "1/InNetwork"),
    "15-1":   ("1700", "Medicare Part B Insulin",             "1/InNetwork"),
    # Dental — section 16
    "16a":    ("1300", "Preventive Dental",                   "1/InNetwork"),
    "16b":    ("1301", "Comprehensive Dental",                "1/InNetwork"),
    "16c":    ("1301", "Dental - Comprehensive (other)",      "1/InNetwork"),
    "16c1":   ("1301", "Dental - Restorative Services",       "1/InNetwork"),
    # Vision — section 17
    "17a":    ("1500", "Eye Exams",                           "1/InNetwork"),
    "17a1":   ("1500", "Routine Eye Exams",                   "1/InNetwork"),
    "17b":    ("1500", "Eyewear",                             "1/InNetwork"),
    "17b1":   ("1500", "Contact Lenses",                      "1/InNetwork"),
    "17b2":   ("1500", "Eyeglasses (lenses and frames)",      "1/InNetwork"),
    "17b3":   ("1500", "Eyeglass Lenses",                     "1/InNetwork"),
    "17b4":   ("1500", "Eyeglass Frames",                     "1/InNetwork"),
    # Hearing — section 18
    "18a":    ("1400", "Hearing Exams",                       "1/InNetwork"),
    "18b":    ("1400", "Hearing Aids",                        "1/InNetwork"),
    "18b1":   ("1400", "Prescription Hearing Aids",           "1/InNetwork"),
}


# Regex to extract section codes from a category string like
# "Medicare Services.Inpatient Hospital-Acute (1a)" → "1a"
_SECTION_CODE_RE = re.compile(r"\(([0-9]+[a-z]?[0-9]?(?:-[0-9]+)?)\)")


def _extract_section_codes(category: str) -> list:
    """Pull all section codes out of a category string. Returns lowercase list."""
    return [m.group(1).lower() for m in _SECTION_CODE_RE.finditer(category or "")]


def _build_data_driven_targets(pbp_rows: list) -> list:
    """
    Scan the input PBP rows, find every distinct section code present, and
    build benefit targets dynamically using SECTION_CODE_TO_BENEFIT.

    Returns a list of target tuples in the same format as PLAN_LEVEL_TARGETS:
      (benefitid, benefitname, [section_codes], [keywords], coverage_type)

    Section codes the LLM produces that are NOT in SECTION_CODE_TO_BENEFIT are
    NOT silently dropped — those rows will be picked up by the sweeper pass.
    """
    # Group section codes by the benefit ID they map to
    by_benefit: dict = defaultdict(lambda: {"codes": set(), "name": None, "coverage": "1/InNetwork"})

    for r in pbp_rows:
        cat = r.get("category") or ""
        for code in _extract_section_codes(cat):
            if code in SECTION_CODE_TO_BENEFIT:
                bid, bname, coverage = SECTION_CODE_TO_BENEFIT[code]
                by_benefit[bid]["codes"].add(code)
                # First name wins — multiple section codes can map to same
                # benefit ID (e.g. 1030 covers 3, 8a, 8a1, 8a2, 8b, 8b1, 8b2)
                if by_benefit[bid]["name"] is None:
                    by_benefit[bid]["name"] = bname
                by_benefit[bid]["coverage"] = coverage

    targets = []
    for bid, info in sorted(by_benefit.items()):
        targets.append((
            bid,
            info["name"],
            sorted(info["codes"]),
            [],   # data-driven targets match by section code only
            info["coverage"],
        ))
    return targets


def _build_targets_for_plan(plan_rows: list) -> list:
    """
    Build the combined target list for one plan:
      = PLAN_LEVEL_TARGETS (hardcoded, keyword-matched)
      + data-driven targets discovered from this plan's section codes

    If the same benefit ID appears in both lists, plan-level wins (it has
    explicit keywords; data-driven would just duplicate).
    """
    plan_level_bids = {t[0] for t in PLAN_LEVEL_TARGETS}
    data_targets = _build_data_driven_targets(plan_rows)
    data_targets = [t for t in data_targets if t[0] not in plan_level_bids]
    return PLAN_LEVEL_TARGETS + data_targets


# Backward-compat alias used by callers that still reference the old name.
# Empty list at import time; populated per-plan inside _run_one_plan.
BENEFIT_TARGETS = []


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


SWEEPER_PROMPT_TEMPLATE = """\
You are doing a final-pass extraction. The targeted benefit calls have already
been made for this Medicare Advantage plan. These remaining PBP rows were NOT
covered by any targeted benefit call — but they may still contain benefits
that should be in the output.

Plan context:
  carrier_prefix  = "{carrier_prefix}"
  planid          = "{planid}"
  plantypeid      = "{plan_type}"
  file_name       = "{file_name}"

PLAN-WIDE PBP ROWS (use for context):
{plan_level_json}

REMAINING UNCOVERED PBP ROWS ({n_remaining} rows):
{remaining_json}

INSTRUCTIONS:
- Scan these rows for any Medicare Advantage benefits that produce output rows.
- Use the benefit ID reference from the system prompt to assign benefitid.
- If a row clearly indicates a benefit (e.g. category contains a benefit name
  or section code), extract it.
- If a row is purely metadata, plan-level admin data, or notes (e.g. "Prior
  Authorization", "Referral"), skip it.
- It's OK to return an empty array if these rows have no actionable benefits.
- Apply ALL formatting rules from the system prompt.
- Always populate planid="{planid}" and plantypeid="{plan_type}".

Return ONLY a valid JSON array. No markdown, no explanation.
"""


async def _run_sweeper_pass(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    system_message: str,
    plan_level_rows: list,
    remaining_rows: list,
    meta: dict,
) -> list:
    """
    Final-pass extraction over PBP rows not covered by any targeted benefit
    call. Returns whatever rows the LLM produces — could be empty.
    """
    label = "sweeper"
    if not remaining_rows:
        print(f"  [{label}] no uncovered rows — skipping")
        return []

    # Cap input to keep the call manageable. If there are too many uncovered
    # rows, split into chunks.
    SWEEPER_CHUNK_SIZE = 200
    chunks = [remaining_rows[i:i + SWEEPER_CHUNK_SIZE]
              for i in range(0, len(remaining_rows), SWEEPER_CHUNK_SIZE)]
    print(f"  [{label}] {len(remaining_rows)} uncovered rows in {len(chunks)} chunk(s)")

    async def _process_chunk(chunk_idx: int, chunk_rows: list) -> list:
        chunk_label = f"{label}[{chunk_idx + 1}/{len(chunks)}]"
        human_text = SWEEPER_PROMPT_TEMPLATE.format(
            carrier_prefix=meta["carrier_prefix"],
            planid=meta["planid"],
            plan_type=meta["plan_type"],
            file_name=meta["file_name"],
            plan_level_json=json.dumps(plan_level_rows, indent=2),
            n_remaining=len(chunk_rows),
            remaining_json=json.dumps(chunk_rows, indent=2),
        )
        async with sem:
            t0 = time.monotonic()
            try:
                raw = await _call_llm_async(client, system_message, human_text, chunk_label)
                rows = _parse_json_array(raw, chunk_label)
                elapsed = time.monotonic() - t0
                print(f"  [{chunk_label}] {elapsed:.1f}s → {len(rows)} rows")
                return rows
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"  [{chunk_label}] ERROR after {elapsed:.1f}s: {e}")
                return []

    tasks = [_process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    sweeper_rows: list = []
    for r in results:
        sweeper_rows.extend(r)
    return sweeper_rows


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

            # ── Empty-result retry ────────────────────────────────────────────
            # If the LLM returned [] but the filter found a meaningful number
            # of input rows for this benefit, retry once with a more explicit
            # prompt. This catches the failure mode where the model gets
            # confused by ambiguous input and bails instead of trying.
            if not rows and len(specific_rows) >= 3:
                retry_text = human_text + (
                    f"\n\nIMPORTANT: The input above contains {len(specific_rows)} "
                    f"PBP rows that relate to {benefit_name}. The previous attempt "
                    f"returned no rows. Look carefully — there IS data here for this "
                    f"benefit. Extract whatever you can find. If a value is "
                    f"genuinely missing, output the benefit row with "
                    f'benefitdesc="Not Covered". Do NOT return an empty array.'
                )
                print(f"  [{label}] empty result with {len(specific_rows)} input rows — retrying")
                raw = await _call_llm_async(client, system_message, retry_text, label + "[retry]")
                rows = _parse_json_array(raw, label + "[retry]")

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
    "610", "611", "615", "616", "620",
    "700", "710", "711", "730", "740", "755", "760",
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

    # Build the target list from this plan's actual section codes. This is the
    # core change in v5: instead of asking about 44 hardcoded benefits, we
    # discover what benefits are present in the input and only ask about
    # those. Plan-level benefits (premium, MOOP, tiers) still come from the
    # hardcoded PLAN_LEVEL_TARGETS list.
    plan_targets = _build_targets_for_plan(plan_rows)
    plan_level_count = len(PLAN_LEVEL_TARGETS)
    data_driven_count = len(plan_targets) - plan_level_count

    print(f"\n--- Plan: {meta['planid']} ({meta['carrier_prefix']}, {meta['plan_type']}) "
          f"| {len(plan_rows):,} rows | plan-level: {len(plan_level_rows)} ---")
    print(f"  targets: {len(plan_targets)} total "
          f"({plan_level_count} plan-level + {data_driven_count} data-driven)")

    # Pass 1: targeted per-benefit extraction. Track which rows each target
    # consumed so the sweeper knows what's leftover.
    t0 = time.monotonic()
    covered_row_ids: set = set()
    for r in plan_level_rows:
        covered_row_ids.add(id(r))  # plan-level rows always considered covered
    for target in plan_targets:
        for r in _filter_rows_for_target(plan_rows, target):
            covered_row_ids.add(id(r))

    tasks = [
        _process_benefit(sem_per_call, client, system_message, target,
                         plan_level_rows, plan_rows, meta)
        for target in plan_targets
    ]
    results = await asyncio.gather(*tasks)
    pass1_elapsed = time.monotonic() - t0

    plan_rows_out: list = []
    for _, rows in results:
        plan_rows_out.extend(rows)

    benefits_with_rows = sum(1 for _, rows in results if rows)
    pass1_rows = len(plan_rows_out)
    print(f"  pass1 done: {pass1_rows} rows from "
          f"{benefits_with_rows}/{len(plan_targets)} benefits in {pass1_elapsed:.1f}s")

    # Pass 2: sweeper over uncovered PBP rows
    remaining = [r for r in plan_rows if id(r) not in covered_row_ids]
    print(f"  pass2 input: {len(remaining)} uncovered rows out of {len(plan_rows)} total")

    if remaining:
        t1 = time.monotonic()
        sweeper_rows = await _run_sweeper_pass(
            sem_per_call, client, system_message,
            plan_level_rows, remaining, meta,
        )
        plan_rows_out.extend(sweeper_rows)
        sweeper_elapsed = time.monotonic() - t1
        print(f"  pass2 done: {len(sweeper_rows)} sweeper rows in {sweeper_elapsed:.1f}s")

    total_elapsed = time.monotonic() - t0
    print(f"--- {meta['planid']} done: {len(plan_rows_out)} rows total "
          f"(pass1: {pass1_rows} + sweeper: {len(plan_rows_out) - pass1_rows}) "
          f"in {total_elapsed:.1f}s ---")
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
