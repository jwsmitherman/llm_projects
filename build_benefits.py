"""
build_benefits.py

Callable benefit-processing module used by main.py.

Public API:  run_benefit_processing(pbp_rows, prompts) -> list[dict]

Handles plans of ANY size by:
  1. Chunking PBP rows into batches that fit the context window
  2. Running all chunk LLM calls IN PARALLEL using ThreadPoolExecutor
  3. Merging and deduplicating results

Result: a 4-chunk plan takes ~2.5 min instead of ~10 min.
"""

import json, os, re
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    from langchain.chat_models import AzureChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage


# ── Config ────────────────────────────────────────────────────────────────────

# Rows per chunk — 400 rows ≈ 25k tokens, well under the 272k limit
CHUNK_SIZE   = int(os.environ.get("PBP_CHUNK_SIZE",    "400"))

# Max parallel LLM calls — respect Azure OpenAI rate limits
MAX_WORKERS  = int(os.environ.get("PBP_MAX_WORKERS",   "4"))

COLS = [
    "planid", "plantypeid", "benefitid", "benefitname",
    "coveragetypeid", "coveragetypedesc",
    "serviceTypeID", "serviceTypeDesc",
    "benefitdesc", "tinyDescription",
]


# ── LLM ──────────────────────────────────────────────────────────────────────

def _build_llm():
    return AzureChatOpenAI(
        azure_endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key          = os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version      = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature      = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0")),
        max_tokens       = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "16000")),
    )


# ── Prompt assembly ───────────────────────────────────────────────────────────

def _assemble_prompts(prompts):
    system_message = prompts["system_prompt"] + "\n\n" + prompts["few_shot_examples"]
    human_template = prompts["human_template"]
    human_template = human_template.replace("{few_shot}\n\n", "").replace("{few_shot}", "")
    return system_message, human_template


# ── Plan metadata ─────────────────────────────────────────────────────────────

def _extract_plan_meta(pbp_rows, carrier_prefix="MOM"):
    file_name = pbp_rows[0].get("FileName", "").strip() if pbp_rows else ""
    remainder = file_name[file_name.index("_") + 1:] if "_" in file_name else file_name
    parts     = remainder.split("-")
    contract  = parts[0] if len(parts) > 0 else ""
    plan      = parts[1] if len(parts) > 1 else ""
    seg       = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    planid    = f"{carrier_prefix}_{contract}_{plan}_{seg}"
    plan_type = "HMO"
    for r in pbp_rows:
        if r.get("header") == "Plan Characteristics" and r.get("field") == "Plan Type":
            plan_type = r.get("value", "HMO").strip()
            break
    return {"file_name": file_name, "planid": planid, "plan_type": plan_type}


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_rows(pbp_rows: list, chunk_size: int) -> list:
    """
    Split pbp_rows into chunks, keeping categories together so the LLM
    sees complete benefit data for each category in one call.
    """
    category_groups = defaultdict(list)
    for r in pbp_rows:
        category_groups[r.get("category", "unknown")].append(r)

    chunks  = []
    current = []
    for cat_rows in category_groups.values():
        if len(cat_rows) > chunk_size:
            for i in range(0, len(cat_rows), chunk_size):
                chunks.append(cat_rows[i:i + chunk_size])
            continue
        if len(current) + len(cat_rows) > chunk_size:
            if current:
                chunks.append(current)
            current = list(cat_rows)
        else:
            current.extend(cat_rows)
    if current:
        chunks.append(current)
    return chunks


# ── Single chunk LLM call ─────────────────────────────────────────────────────

def _process_chunk(llm, system_message, human_template,
                   chunk, chunk_idx, n_chunks,
                   carrier_prefix, meta) -> list:
    """
    Send one chunk to the LLM and return parsed rows.
    Called in parallel across chunks.
    """
    chunk_note = (
        f"\n\nNOTE: This is chunk {chunk_idx + 1} of {n_chunks}. "
        f"Process only the rows in this chunk and return ALL benefit rows you can derive."
        if n_chunks > 1 else ""
    )

    human_text = human_template.format_map({
        "carrier_prefix": carrier_prefix,
        "file_name":      meta["file_name"],
        "plan_type":      meta["plan_type"],
        "n_pbp":          len(chunk),
        "pbp_json":       json.dumps(chunk, indent=2) + chunk_note,
    })

    response = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_text),
    ])
    raw = response.content

    # Parse JSON array from response
    text  = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    text  = re.sub(r"\n?```$",       "", text,         flags=re.MULTILINE).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Chunk {chunk_idx+1}: No JSON array in response. Preview: {raw[:300]}")
    return json.loads(text[start:end])


# ── Result merging ────────────────────────────────────────────────────────────

def _merge_results(all_chunk_results: list) -> list:
    """
    Merge results from all chunks.
    Deduplicates by (benefitid, serviceTypeID) — last write wins.
    """
    seen   = {}
    merged = []
    for rows in all_chunk_results:
        for row in rows:
            key = (str(row.get("benefitid", "")), str(row.get("serviceTypeID", "")))
            if key not in seen:
                seen[key] = len(merged)
                merged.append(row)
            else:
                merged[seen[key]] = row
    return merged


# ── Public entry point ────────────────────────────────────────────────────────

def run_benefit_processing(pbp_rows: list, prompts: dict) -> list:
    """
    Process PBP rows through the LLM with no row count limit.

    Parameters
    ----------
    pbp_rows : list   — raw PBP field/value rows for one plan (any size)
    prompts  : dict   — keys: system_prompt, few_shot_examples, human_template

    Returns
    -------
    list[dict] — one dict per (benefitid, serviceTypeID), 10 columns each
    """
    llm                        = _build_llm()
    system_message, human_tmpl = _assemble_prompts(prompts)
    carrier_prefix             = os.environ.get("CARRIER_PREFIX", "MOM")
    meta                       = _extract_plan_meta(pbp_rows, carrier_prefix)

    # Split into chunks
    chunks   = _chunk_rows(pbp_rows, CHUNK_SIZE)
    n_chunks = len(chunks)

    print(f"Processing {len(pbp_rows):,} rows in {n_chunks} chunk(s) "
          f"with up to {MAX_WORKERS} parallel workers")

    # Run all chunks in parallel
    chunk_results = [None] * n_chunks  # preserve order

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _process_chunk,
                llm, system_message, human_tmpl,
                chunk, i, n_chunks,
                carrier_prefix, meta
            ): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                chunk_results[i] = future.result()
                print(f"  Chunk {i+1}/{n_chunks} done — "
                      f"{len(chunk_results[i])} rows returned")
            except Exception as e:
                print(f"  Chunk {i+1}/{n_chunks} ERROR: {e}")
                chunk_results[i] = []  # partial failure — continue with others

    # Merge in order (chunk 0 first, chunk N last)
    merged = _merge_results(chunk_results)
    print(f"Merged total: {len(merged)} unique benefit rows")

    return (
        pd.DataFrame(merged)
        .reindex(columns=COLS)
        .sort_values(["benefitid", "serviceTypeID"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
