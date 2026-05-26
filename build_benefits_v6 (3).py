"""
build_benefits_v6.py
====================
Pipeline that produces IFP2 plan-benefits rows from the new MedicareFeed PBP CSV
input format.

ARCHITECTURE (different from v5.2):
- Python decides which benefits exist in each plan
- Python looks up benefitname, plantypeid, coverageTypedesc, serviceTypeDesc from
  static CSVs (no LLM hallucination of IDs or names)
- LLM ONLY generates benefitdesc and tinyDescription text from input rows

OUTPUT: 10 columns exactly as specified:
    planid, plantypeid, benefitid, benefitname,
    coverageTypeid, coverageTypedesc,
    serviceTypeID, serviceTypeDesc,
    benefitdesc, tinyDescription

USAGE:
    from build_benefits_v6 import run_load
    summary = run_load(
        input_csv_path='input_MedicareFeed_carrierDESR_PBP_LoadID211.csv',
        lookup_dir='./lookups',
        prompts_dir='./prompts',
        output_csv_path='output_benefits_211.csv',
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

__BUILD_VERSION__ = "2026-05-20-v6.8-lean-prompts"


# ----------------------------------------------------------------------------
# Configuration - all overridable via env vars
# ----------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_TEMPERATURE: Optional[float] = None  # gpt-5 family rejects explicit temp

LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "1500"))
MAX_RETRIES       = int(os.getenv("MAX_RETRIES", "5"))      # Was 10 in v6.7 - too patient
BASE_BACKOFF_S    = float(os.getenv("BASE_BACKOFF_S", "2.0"))  # Was 3.0 in v6.7
MAX_BACKOFF_S     = float(os.getenv("MAX_BACKOFF_S", "30.0"))  # Was 60 - cap tighter
MAX_CONCURRENCY   = int(os.getenv("MAX_CONCURRENCY", "2"))
CONCURRENT_PLANS  = int(os.getenv("CONCURRENT_PLANS", "1"))


# ----------------------------------------------------------------------------
# Plan-type mapping. PBP value -> IFP2 planTypeID
# Per user-confirmed mapping table.
# ----------------------------------------------------------------------------
PLAN_TYPE_MAP = {
    "Medicare Prescription Drug Plan": "PDP",
    "PFFS":                            "PFFS",
    "Regional PPO":                    "PPO",
    "HMO":                             "HMO",
    "HMOPOS":                          "POS",
    "Local PPO":                       "PPO",
    "COST":                            "COST",   # best guess per user
}


# ----------------------------------------------------------------------------
# Section-code -> benefit ID. This map says "if I see category text containing
# (7a), produce rows for benefitid 900". The full list will be refined as we
# see more carrier data; for now it covers the categories present in HCSC.
# ----------------------------------------------------------------------------
SECTION_CODE_TO_BENEFIT_ID = {
    # Inpatient / facility
    "1a":   800, "1a1":  800, "1a2":  800,
    "1b":   810,
    "2":    820,
    "3":    830, "3-1": 830, "3-2": 830, "3-3": 830, "3-4": 830,
    # Outpatient
    "4a":  1010, "4a1": 1010, "4a2": 1010, "4b":  1020, "4c":  2100, "4c1": 2100, "4c2": 2100,
    "5a":   950, "5b":   950,
    "6":   1400,
    # Professional services
    "7a":   900, "7b":   920, "7c":   990, "7d":   910, "7e":   940, "7e1":  940, "7e2":  940,
    "7f":   930, "7g":   911, "7h":   940, "7h1":  940, "7h2":  940, "7i":   990, "7j":   911, "7k":   960,
    # Diagnostics / tests
    "8a":  1030, "8a1": 1030, "8a2": 1030, "8b":  1030, "8b1": 1030, "8b2": 1030, "8b3": 1030,
    # Outpatient hospital
    "9a":   981, "9a1":  981, "9a2":  982, "9b":   970, "9c":   960, "9c1":  960, "9c2":  960, "9d":   981,
    # Ambulance + DME + supplies
    "10a": 1000, "10a1":1000, "10a2":1000,
    "11a": 1060, "11b": 1060, "11b1":1060, "11b2":1060, "11c": 1060, "11c1":1060, "11c2":1060,
    # Dialysis + drugs + preventive
    "12":  1200,
    "14a": 1300, "14b": 1300, "14c": 1300, "14c4":1300, "14c7":1300, "14d": 1300,
    "14e1":1300, "14e2":1300, "14e4":1300, "14e5":1300,
    "15":   760, "15-1": 755, "15-2": 760, "15-3": 760,
    # Dental + vision + hearing
    "16a": 1610, "16b": 1610, "16b1":1610, "16b2":1610, "16b4":1610,
    "16c": 1610, "16c1":1610, "16c2":1610, "16c3":1610, "16c4":1610,
    "16c5":1610, "16c7":1610, "16c8":1610, "16c10":1610,
    "17a": 1700, "17a1":1700, "17b": 1700, "17b1":1700, "17b3":1700, "17b4":1700,
    "18a": 1050, "18a1":1050, "18a2":1050,
    "18b": 1800, "18b1":1800,
    # Other
    # (Note: section 19 is omitted - benefit 670 isn't in the canonical list)
}


# ----------------------------------------------------------------------------
# Category-string -> benefit ID for plan-level rows (no section code present)
# These are the rows from "Plan Characteristics", "Rx Setup", etc.
# ----------------------------------------------------------------------------
PLAN_LEVEL_CATEGORIES = {
    # MOOP / OOP rows
    "Contract Year 2026 Medicare-defined MOOP Limits (Local PPO Plan)": 700,
    "LPPO/RPPO Max Enrollee Cost Limit":                                700,
    "Annual Plan Deductible LPPO/RPPO":                                 610,
    "Deductible for LPPO/RPPO Mandatory Supplemental Benefits":         610,
    # Rx
    "Rx Setup":           710,
    "Rx Setup.Tiering":   710,
    "Rx Cost Share":      710,
    "Rx Characteristics": 710,
    "Rx Attestations":    710,
    "Rx Notes":           710,
    "Rx Insulin":         755,
    "Standard Bid":       600,
}


# ----------------------------------------------------------------------------
# Lookup loader
# ----------------------------------------------------------------------------
@dataclass
class Lookups:
    benefit_name:  dict   # {benefit_id: name}
    coverage_type: dict   # {ctID: desc}
    service_type:  dict   # {stID: desc}

    @classmethod
    def from_dir(cls, lookup_dir: str | Path) -> "Lookups":
        d = Path(lookup_dir)
        # keep_default_na=False prevents pandas turning the string "NA" into NaN
        # (the coverage type 4 has description "NA" per the CMS schema)
        bn = pd.read_csv(d / "benefit_id_name.csv",       keep_default_na=False)
        ct = pd.read_csv(d / "coverage_type_lookup.csv",  keep_default_na=False)
        st = pd.read_csv(d / "service_type_lookup.csv",   keep_default_na=False)
        return cls(
            benefit_name  = dict(zip(bn["benefitid"].astype(int),    bn["benefitname"])),
            coverage_type = dict(zip(ct["coverageTypeid"].astype(int), ct["coverageTypedesc"])),
            service_type  = dict(zip(st["serviceTypeID"].astype(int),  st["serviceTypeDesc"])),
        )


# ----------------------------------------------------------------------------
# Section-code parser
# ----------------------------------------------------------------------------
SECTION_CODE_RE = re.compile(r"\(([0-9]+[a-z]?[0-9]*|[0-9]+-[0-9]+|19[ab])\)")


def _extract_section_code(category: str) -> Optional[str]:
    """Pulls e.g. '7a' from 'Primary Care Physician Services (7a) - Medicare'."""
    if not isinstance(category, str):
        return None
    m = SECTION_CODE_RE.search(category)
    return m.group(1) if m else None


def _is_oon(category: str) -> bool:
    """True if this category represents out-of-network rows."""
    if not isinstance(category, str):
        return False
    return "Out-of-Network" in category or "OON" in category


# ----------------------------------------------------------------------------
# Plan-level extraction (Plan Type, Plan ID)
# ----------------------------------------------------------------------------
def _extract_plan_metadata(plan_rows: pd.DataFrame) -> dict:
    """Return planid, plantypeid for this plan from its rows."""
    file_name = plan_rows["FileName"].iloc[0]
    # Get the Plan Type value
    pt_rows = plan_rows[plan_rows["field"] == "Plan Type"]
    pt_value = pt_rows["value"].iloc[0] if not pt_rows.empty else None
    plantypeid = PLAN_TYPE_MAP.get(pt_value, "UNKNOWN")

    # Convert FileName 'HCSC_H0107-003-000' -> planid 'HCSC_H0107_003_0'
    # Carrier prefix kept here; downstream MTM/etc mapping happens in another layer
    parts = file_name.split("_", 1)
    if len(parts) == 2:
        carrier, hcontract = parts[0], parts[1]
        # 'H0107-003-000' -> ['H0107', '003', '000']
        h_parts = hcontract.split("-")
        if len(h_parts) >= 3:
            segment = h_parts[2].lstrip("0") or "0"
            planid = f"{carrier}_{h_parts[0]}_{h_parts[1]}_{segment}"
        else:
            planid = file_name
    else:
        planid = file_name

    return {
        "file_name":  file_name,
        "planid":     planid,
        "plantypeid": plantypeid,
        "plan_type_raw": pt_value,
    }


# ----------------------------------------------------------------------------
# Group plan rows by benefit_id
# ----------------------------------------------------------------------------
@dataclass
class BenefitGroup:
    benefit_id:    int
    section_code:  Optional[str]    # e.g. "7a", or None for plan-level
    rows:          pd.DataFrame      # rows tagged to this benefit
    is_oon:        bool = False


def _group_rows_by_benefit(plan_rows: pd.DataFrame) -> dict[tuple[int, bool], BenefitGroup]:
    """
    Walk every row in this plan, tag with (benefit_id, is_oon), and group.
    Returns: {(benefit_id, is_oon): BenefitGroup}
    """
    tagged_rows: list[tuple[int, bool, int]] = []  # (benefit_id, is_oon, row_idx)

    for idx, row in plan_rows.iterrows():
        cat = row["category"] if isinstance(row["category"], str) else ""
        sc = _extract_section_code(cat)
        oon = _is_oon(cat)

        # Try section code first
        if sc and sc in SECTION_CODE_TO_BENEFIT_ID:
            tagged_rows.append((SECTION_CODE_TO_BENEFIT_ID[sc], oon, idx))
            continue

        # Fall back to plan-level category mapping (exact or prefix match)
        if cat in PLAN_LEVEL_CATEGORIES:
            tagged_rows.append((PLAN_LEVEL_CATEGORIES[cat], False, idx))
            continue
        # Try prefix match for Rx Tier categories etc.
        matched = False
        for prefix, bid in PLAN_LEVEL_CATEGORIES.items():
            if cat.startswith(prefix + "."):
                tagged_rows.append((bid, False, idx))
                matched = True
                break
        if matched:
            continue
        # Otherwise skip - row isn't a benefit we extract

    # Build groups
    groups: dict[tuple[int, bool], BenefitGroup] = {}
    for bid, oon, idx in tagged_rows:
        key = (bid, oon)
        if key not in groups:
            sc = _extract_section_code(plan_rows.at[idx, "category"])
            groups[key] = BenefitGroup(benefit_id=bid, section_code=sc, rows=plan_rows.loc[[idx]], is_oon=oon)
        else:
            groups[key].rows = pd.concat([groups[key].rows, plan_rows.loc[[idx]]])

    return groups


# ----------------------------------------------------------------------------
# Canonical per-benefit defaults from system_prompt.txt
# ----------------------------------------------------------------------------
# For each benefit ID, what coverageTypeid does it use by default?
# This comes straight from system_prompt.txt's "BENEFIT ID AND COVERAGE TYPE
# REFERENCE" section. If is_oon=True we override to 2 (OutOfNetwork) for
# benefits that have an in/out distinction.
BENEFIT_DEFAULT_COVERAGE_TYPE = {
    600: 4, 610: 1, 611: 4, 614: 4, 615: 4, 616: 4, 620: 1,
    700: 4, 710: 3, 711: 1, 730: 4, 740: 4, 755: 1, 760: 1,
    800: 1, 810: 1, 820: 1,
    900: 1, 910: 1, 911: 1, 920: 1, 930: 1, 940: 1,
    950: 1, 960: 1, 970: 1,
    981: 4, 982: 4, 990: 1,
    1000: 1, 1020: 1, 1030: 1, 1050: 1, 1060: 1, 1200: 1,
    1300: 1, 1301: 1, 1400: 1, 1500: 1, 1610: 1,
    1700: 1, 1800: 1, 1900: 1, 2100: 1,
    2110: 4, 2111: 4, 2112: 4,
}

# Benefits where is_oon=True should NOT override coveragetype to 2 because the
# benefit is plan-level/NA in nature (e.g. monthly premium, star ratings).
COVERAGE_TYPE_NA_BENEFITS = {
    600, 611, 614, 615, 616, 700, 730, 740, 981, 982,
    2110, 2111, 2112,
}


# ----------------------------------------------------------------------------
# Determine coverage_type_id / service_type_id from rows + plan metadata
# ----------------------------------------------------------------------------
def _infer_coverage_type(group: BenefitGroup, plantypeid: str) -> int:
    """
    Per-benefit canonical mapping from system_prompt.txt.

    OutOfNetwork override: if the input rows indicate out-of-network AND the
    benefit has an in/out distinction (not in COVERAGE_TYPE_NA_BENEFITS),
    flip to 2.
    """
    default_ct = BENEFIT_DEFAULT_COVERAGE_TYPE.get(group.benefit_id, 1)
    if group.is_oon and group.benefit_id not in COVERAGE_TYPE_NA_BENEFITS:
        return 2
    return default_ct


def _infer_service_type(group: BenefitGroup) -> int:
    """
    Service type heuristic. Most benefits use 0 (N/A). Inpatient + pharmacy
    tiers use specific serviceTypeID values from the system_prompt.

    This is a STARTING POINT - real serviceTypeID assignment varies by
    sub-benefit (e.g. inpatient has 115/125/134/163 for day ranges; pharmacy
    has 20-24 for tiers). Refine this once we see the per-benefit row patterns.
    """
    bid = group.benefit_id
    # Pharmacy tier defaults to N/A (the LLM will use the right one from text)
    # Inpatient default to "days 1 through 20" - common starting range
    if bid in (800, 810, 820):
        return 115  # days 1 through 20
    return 0



# ----------------------------------------------------------------------------
# Build base rows (deterministic - no LLM)
# ----------------------------------------------------------------------------
def _build_base_rows(plan_meta: dict, groups: dict, lookups: Lookups) -> list[dict]:
    """
    For each benefit group, produce one base row with the 8 deterministic
    columns filled in. benefitdesc + tinyDescription will be added by the LLM.
    """
    rows = []
    for (bid, oon), group in sorted(groups.items()):
        ctid = _infer_coverage_type(group, plan_meta["plantypeid"])
        stid = _infer_service_type(group)
        rows.append({
            "planid":           plan_meta["planid"],
            "plantypeid":       plan_meta["plantypeid"],
            "benefitid":        bid,
            "benefitname":      lookups.benefit_name.get(bid, f"Benefit {bid}"),
            "coverageTypeid":   ctid,
            "coverageTypedesc": lookups.coverage_type.get(ctid, "Unknown"),
            "serviceTypeID":    stid,
            "serviceTypeDesc":  lookups.service_type.get(stid, "Standard"),
            # placeholders - LLM fills these
            "benefitdesc":      None,
            "tinyDescription":  None,
            # internal - not in final output
            "_group_rows":      group.rows,
            "_section_code":    group.section_code,
            "_is_oon":          oon,
        })
    return rows


# ----------------------------------------------------------------------------
# LLM call - SINGLE function, generates ONLY benefitdesc + tinyDescription
# ----------------------------------------------------------------------------
async def _call_llm_async(client: httpx.AsyncClient, system_message: str,
                          user_message: str, label: str) -> str:
    url = (
        f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    body = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        "max_completion_tokens": LLM_MAX_TOKENS,
    }
    if AZURE_OPENAI_TEMPERATURE is not None:
        body["temperature"] = AZURE_OPENAI_TEMPERATURE

    last_err:           Optional[Exception] = None
    last_status:        Optional[int]       = None
    last_body_preview:  Optional[str]       = None
    last_finish_reason: Optional[str]       = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(url, headers=headers, json=body)
            last_status = resp.status_code

            if resp.status_code == 429 or resp.status_code >= 500:
                # Capture the body so we know WHY it's throttling
                try:
                    last_body_preview = resp.text[:300] if resp.text else "(empty body)"
                except Exception:
                    last_body_preview = "(could not read body)"
                retry_ms = resp.headers.get("retry-after-ms")
                retry_s  = resp.headers.get("retry-after")
                if retry_ms and retry_ms.isdigit():
                    sleep_s = int(retry_ms) / 1000.0
                elif retry_s and retry_s.isdigit():
                    sleep_s = float(retry_s)
                else:
                    sleep_s = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
                # Cap so a long retry stack doesn't wait minutes per attempt
                sleep_s = min(sleep_s, MAX_BACKOFF_S)
                await asyncio.sleep(sleep_s)
                continue

            if 400 <= resp.status_code < 500:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text
                raise RuntimeError(f"[{label}] HTTP {resp.status_code}: {err_body}")

            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            last_finish_reason = data["choices"][0].get("finish_reason")

            # Empty content with finish_reason=length is a usable signal -
            # the model ran out of tokens before producing output. Treat as retry-worthy.
            if not content.strip() and last_finish_reason == "length":
                last_body_preview = (
                    f"empty content, finish_reason=length (max_completion_tokens={LLM_MAX_TOKENS})"
                )
                await asyncio.sleep(BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1))
                continue

            return content
        except httpx.RequestError as e:
            last_err = e
            await asyncio.sleep(BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1))

    # Build a rich diagnostic message
    parts = [f"[{label}] exhausted {MAX_RETRIES} retries"]
    if last_status is not None:
        parts.append(f"last_status={last_status}")
    if last_finish_reason is not None:
        parts.append(f"finish_reason={last_finish_reason}")
    if last_err is not None:
        parts.append(f"network_error={last_err}")
    if last_body_preview:
        parts.append(f"body={last_body_preview[:200]}")
    raise RuntimeError(" | ".join(parts))


def _build_user_message_for_benefit(base_row: dict) -> str:
    """
    Compact input for the LLM: which benefit, plus the relevant input rows.
    """
    group_rows: pd.DataFrame = base_row["_group_rows"]
    payload_lines = []
    payload_lines.append(f"Benefit: {base_row['benefitid']} - {base_row['benefitname']}")
    payload_lines.append(f"Plan Type: {base_row['plantypeid']}")
    payload_lines.append(f"Coverage: {base_row['coverageTypedesc']}  Service: {base_row['serviceTypeDesc']}")
    payload_lines.append(f"Section Code: {base_row.get('_section_code') or '(plan-level)'}")
    payload_lines.append("Input rows:")
    for _, r in group_rows.iterrows():
        cat = r["category"] if isinstance(r["category"], str) else ""
        fld = r["field"] if isinstance(r["field"], str) else ""
        val = r["value"] if isinstance(r["value"], str) else str(r["value"])
        payload_lines.append(f"  [{cat}] {fld} = {val}")

    payload_lines.append("")
    payload_lines.append(
        'Respond with ONLY a JSON object: '
        '{"benefitdesc": "<html-formatted full description>", "tinyDescription": "<5-15 char short>"}'
    )
    return "\n".join(payload_lines)


def _parse_llm_response(raw: str) -> dict:
    """Parse the LLM's JSON response, defending against markdown fences."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        d = json.loads(s)
        return {
            "benefitdesc":     d.get("benefitdesc", "") or "",
            "tinyDescription": d.get("tinyDescription", "") or "",
        }
    except Exception:
        # Last-resort: try to pull both fields with regex
        bd = re.search(r'"benefitdesc"\s*:\s*"([^"]*)"', s)
        td = re.search(r'"tinyDescription"\s*:\s*"([^"]*)"', s)
        return {
            "benefitdesc":     bd.group(1) if bd else "",
            "tinyDescription": td.group(1) if td else "",
        }


# ----------------------------------------------------------------------------
# Per-plan orchestration
# ----------------------------------------------------------------------------
async def _process_plan_async(client, sem, system_message: str,
                              plan_meta: dict, base_rows: list[dict]) -> list[dict]:
    """For each base row, call LLM once to get its benefitdesc + tinyDescription."""

    async def fill_one(base_row):
        async with sem:
            user_msg = _build_user_message_for_benefit(base_row)
            label = f"{plan_meta['planid']}/bid={base_row['benefitid']}/ctID={base_row['coverageTypeid']}"
            try:
                raw = await _call_llm_async(client, system_message, user_msg, label)
                parsed = _parse_llm_response(raw)
                base_row["benefitdesc"]     = parsed["benefitdesc"]
                base_row["tinyDescription"] = parsed["tinyDescription"]
            except Exception as e:
                base_row["benefitdesc"]     = f"[LLM ERROR: {e}]"
                base_row["tinyDescription"] = ""
        return base_row

    return await asyncio.gather(*(fill_one(r) for r in base_rows))


async def process_one_plan_async(plan_rows: pd.DataFrame, lookups: Lookups,
                                  system_message: str,
                                  client: httpx.AsyncClient,
                                  sem: asyncio.Semaphore,
                                  max_benefits: Optional[int] = None) -> list[dict]:
    """
    Async version - process one plan using an externally-provided client+semaphore.

    max_benefits: if set, only process the first N base rows (fast smoke testing).
    """
    plan_meta = _extract_plan_metadata(plan_rows)
    groups = _group_rows_by_benefit(plan_rows)
    base_rows = _build_base_rows(plan_meta, groups, lookups)

    if max_benefits is not None and max_benefits < len(base_rows):
        print(f"  [{plan_meta['planid']}] LIMITED to first {max_benefits} of {len(base_rows)} benefits")
        base_rows = base_rows[:max_benefits]

    if not base_rows:
        return []

    filled = await _process_plan_async(client, sem, system_message, plan_meta, base_rows)

    # Strip internal fields before returning
    out = []
    for r in filled:
        out.append({k: v for k, v in r.items() if not k.startswith("_")})
    return out


def process_one_plan(plan_rows: pd.DataFrame, lookups: Lookups,
                     system_message: str) -> list[dict]:
    """
    Sync wrapper for process_one_plan_async. Detects whether an event loop is
    already running (e.g. inside a Jupyter notebook) and dispatches accordingly:
      - In a notebook (loop already running): uses nest_asyncio if available
      - In a script (no loop): uses asyncio.run()
    """
    async def _runner():
        sem = asyncio.Semaphore(MAX_CONCURRENCY)
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            return await process_one_plan_async(plan_rows, lookups, system_message, client, sem)

    return _run_async(_runner())


def _run_async(coro):
    """
    Run an async coroutine from either sync or async context.
    Notebooks already have a running loop; scripts don't.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Notebook context - need nest_asyncio to nest the event loop
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            raise RuntimeError(
                "Detected a running event loop (likely Jupyter). "
                "Install nest_asyncio: pip install nest_asyncio"
            )
        return asyncio.run(coro)
    else:
        return asyncio.run(coro)


# ----------------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------------
def run_load(input_csv_path: str | Path,
             lookup_dir:     str | Path,
             prompts_dir:    str | Path,
             output_json_path: str | Path,
             output_csv_path:  Optional[str | Path] = None,
             load_id: Optional[str] = None,
             plan_filter: Optional[list[str]] = None,
             max_benefits_per_plan: Optional[int] = None,
             progress_callback: Optional[callable] = None) -> dict:
    """
    Top-level entry. Reads input CSV, runs all plans, writes:
      - output_json_path: REQUIRED. JSON in {load_id, results: [...]} shape
        for Morteza's SQL ingest (OPENJSON expects this envelope).
      - output_csv_path:  OPTIONAL. Same data as flat CSV for spot-checking.

    load_id:       Optional. If provided, used as the top-level load_id in the
                   JSON envelope. If not provided, extracted from the input
                   file's LoadID column.
    plan_filter:   If provided, only process plans whose FileName is in this list.
                   Useful for dev: pass [first_filename] to test on 1 plan only.
    max_benefits_per_plan: If provided, only process the first N benefits per plan.
                   Use for fast smoke tests - e.g. max_benefits_per_plan=5 -> ~10
                   LLM calls per plan instead of ~60.
    progress_callback: called as fn(planname, n_done, n_total) after each plan.
    """
    print(f"[v6] build version: {__BUILD_VERSION__}")
    t0 = time.monotonic()

    # Load input
    df_input = pd.read_csv(input_csv_path)
    original_rows = len(df_input)
    print(f"[v6] loaded {original_rows:,} input rows from {input_csv_path}")

    # Drop rows with no FileName - these are typically header/footer junk and
    # would crash _extract_plan_metadata with "iloc out of bounds".
    df_input = df_input.dropna(subset=["FileName"]).copy()
    dropped = original_rows - len(df_input)
    if dropped:
        print(f"[v6] dropped {dropped} row(s) with NaN FileName")

    if df_input.empty:
        raise SystemExit("STOP - no rows with valid FileName remain after cleaning")

    # Resolve load_id: caller wins, else infer from data
    def _clean_load_id(v) -> str:
        """Coerce 212.0, '212', '212.0' etc. all to '212' (int-string)."""
        s = str(v).strip()
        # If it's a float-looking string, strip the .0
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return s

    if load_id is not None:
        load_id = _clean_load_id(load_id)
        print(f"[v6] load_id (from parameter): {load_id}")
    else:
        load_id_counts = (
            df_input["LoadID"].dropna().apply(_clean_load_id).value_counts()
        )
        if len(load_id_counts) == 0:
            raise SystemExit("STOP - no load_id provided and input has no LoadID values")
        if len(load_id_counts) > 1:
            top_id = load_id_counts.index[0]
            print(f"[v6] WARNING: input has multiple LoadIDs: "
                  f"{dict(load_id_counts)}; using most-frequent: {top_id!r}")
            load_id = top_id
        else:
            load_id = load_id_counts.index[0]
            print(f"[v6] load_id (from input): {load_id}")

    if plan_filter:
        df_input = df_input[df_input["FileName"].isin(plan_filter)]
        print(f"[v6] filtered to {df_input['FileName'].nunique()} plan(s)")

    # Load lookups
    lookups = Lookups.from_dir(lookup_dir)
    print(f"[v6] lookups loaded: {len(lookups.benefit_name)} benefits, "
          f"{len(lookups.coverage_type)} coverage types, {len(lookups.service_type)} service types")

    # Load prompts
    sys_path = Path(prompts_dir) / "system_prompt_v6.txt"
    if sys_path.exists():
        system_message = sys_path.read_text(encoding="utf-8")
        print(f"[v6] loaded system_prompt_v6.txt ({len(system_message)} chars)")
    else:
        system_message = _default_system_prompt()
        print(f"[v6] WARNING: no system_prompt_v6.txt at {sys_path}, using default")

    # Optionally append few-shot examples
    examples_path = Path(prompts_dir) / "few_shot_examples_v6.txt"
    if examples_path.exists():
        examples = examples_path.read_text(encoding="utf-8")
        system_message = system_message + "\n\n" + examples
        print(f"[v6] appended few_shot_examples_v6.txt ({len(examples)} chars)")

    # Group input rows by plan
    plan_filenames = df_input["FileName"].unique().tolist()
    print(f"[v6] processing {len(plan_filenames)} plan(s) "
          f"with up to {CONCURRENT_PLANS} concurrent")

    # Async runner: shared client, semaphore-limited concurrency across plans
    async def _run_all_plans():
        all_results: list[tuple[str, list[dict], float, Optional[str]]] = []
        sem_calls = asyncio.Semaphore(MAX_CONCURRENCY)
        sem_plans = asyncio.Semaphore(CONCURRENT_PLANS)
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:

            async def _one_plan(fn: str):
                async with sem_plans:
                    plan_rows = df_input[df_input["FileName"] == fn].copy()
                    plan_rows.reset_index(drop=True, inplace=True)
                    t_plan = time.monotonic()
                    try:
                        rows = await process_one_plan_async(
                            plan_rows, lookups, system_message, client, sem_calls,
                            max_benefits=max_benefits_per_plan,
                        )
                        elapsed = time.monotonic() - t_plan
                        return (fn, rows, elapsed, None)
                    except Exception as e:
                        elapsed = time.monotonic() - t_plan
                        tb = traceback.format_exc()
                        return (fn, [], elapsed, f"{e}\n{tb}")

            tasks = [asyncio.create_task(_one_plan(fn)) for fn in plan_filenames]
            n_completed = 0
            for fut in asyncio.as_completed(tasks):
                fn, rows, elapsed, err = await fut
                n_completed += 1
                if err is None:
                    print(f"  [{n_completed}/{len(plan_filenames)}] {fn}: "
                          f"{len(rows)} rows in {elapsed:.1f}s")
                else:
                    print(f"  [{n_completed}/{len(plan_filenames)}] {fn}: "
                          f"FAILED after {elapsed:.1f}s")
                    print(f"    {err}")
                if progress_callback:
                    try:
                        progress_callback(fn, n_completed, len(plan_filenames))
                    except Exception:
                        pass
                all_results.append((fn, rows, elapsed, err))
        return all_results

    all_results = _run_async(_run_all_plans())

    all_output_rows: list[dict] = []
    n_done = 0
    n_failed = 0
    for fn, rows, elapsed, err in all_results:
        if err is None:
            all_output_rows.extend(rows)
            n_done += 1
        else:
            n_failed += 1

    # Write output - JSON envelope is the canonical deliverable for Morteza's SQL ingest
    out_cols = ["planid", "plantypeid", "benefitid", "benefitname",
                "coverageTypeid", "coverageTypedesc",
                "serviceTypeID", "serviceTypeDesc",
                "benefitdesc", "tinyDescription"]

    # Filter each row down to the canonical 10 columns in the required order
    canonical_rows = []
    for r in all_output_rows:
        canonical_rows.append({k: r.get(k) for k in out_cols})

    # Required: JSON in Morteza's envelope shape
    envelope = {
        "load_id": load_id,
        "results": canonical_rows,
    }
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    print(f"[v6] wrote {len(canonical_rows):,} rows to {output_json_path} "
          f"(JSON, with load_id={load_id!r})")

    # Optional: flat CSV for spot-checking in Excel
    csv_path_written: Optional[str] = None
    if output_csv_path and canonical_rows:
        out_df = pd.DataFrame(canonical_rows).reindex(columns=out_cols)
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv_path, index=False)
        csv_path_written = str(output_csv_path)
        print(f"[v6] wrote CSV side artifact to {output_csv_path}")

    total_elapsed = time.monotonic() - t0
    summary = {
        "build_version":    __BUILD_VERSION__,
        "load_id":          load_id,
        "plans_total":      len(plan_filenames),
        "plans_done":       n_done,
        "plans_failed":     n_failed,
        "rows_out":         len(canonical_rows),
        "elapsed_s":        round(total_elapsed, 1),
        "output_json_path": str(output_json_path),
        "output_csv_path":  csv_path_written,
    }
    print(f"[v6] DONE: {summary}")
    return summary


def _default_system_prompt() -> str:
    return (
        "You are a Medicare Advantage benefit description generator. "
        "Given input rows from a CMS PBP file for ONE specific benefit, output a JSON object "
        "with EXACTLY these two fields:\n"
        "  benefitdesc - a complete HTML-formatted description suitable for display to a "
        "consumer. Use <b>$X</b> for dollar amounts and <b>X%</b> for percentages. Be concise.\n"
        "  tinyDescription - a 5-15 character summary (e.g. '$0 copay', '20% per visit', 'No Ded').\n"
        "\n"
        "Rules:\n"
        "  - If the benefit is not covered or amount is zero, say so explicitly\n"
        "  - For copay/coinsurance amounts, format as <b>$X</b> or <b>X%</b>\n"
        "  - For ranges, format as <b>$X</b> - <b>$Y</b>\n"
        "  - Do NOT invent values not in the input rows\n"
        "  - Output ONLY the JSON object, no markdown fences, no explanation"
    )
