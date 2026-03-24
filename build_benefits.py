"""
build_benefits.py

Callable benefit-processing module used by main.py.

Public API:  run_benefit_processing(pbp_rows, prompts) -> list[dict]

Handles plans of ANY size by chunking PBP rows into batches that
fit within the model's context window, making one LLM call per chunk,
then merging and deduplicating the results.
"""

import json, os, re, math
import pandas as pd

try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    from langchain.chat_models import AzureChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage


# ── Config ────────────────────────────────────────────────────────────────────

# Max rows per LLM chunk — keeps each call well under the context limit.
# At ~250 chars/row average, 400 rows ≈ 100k chars ≈ 25k tokens (safe for 272k limit).
CHUNK_SIZE = int(os.environ.get("PBP_CHUNK_SIZE", "400"))

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


# ── LLM invocation ────────────────────────────────────────────────────────────

def _invoke_chain(llm, system_message, human_template, template_vars):
    human_text = human_template.format_map(template_vars)
    response   = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_text),
    ])
    return response.content


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
    Split pbp_rows into chunks of chunk_size.
    Always keeps all rows from the same category together so the LLM
    sees complete benefit data for each category in one chunk.
    """
    # Group rows by category first
    from collections import defaultdict
    category_groups = defaultdict(list)
    for r in pbp_rows:
        cat = r.get("category", "unknown")
        category_groups[cat].append(r)

    # Pack categories into chunks without splitting a category across chunks
    chunks  = []
    current = []
    for cat_rows in category_groups.values():
        # If a single category is larger than chunk_size, split it
        if len(cat_rows) > chunk_size:
            for i in range(0, len(cat_rows), chunk_size):
                chunks.append(cat_rows[i:i + chunk_size])
            continue

        # If adding this category would overflow the chunk, start a new one
        if len(current) + len(cat_rows) > chunk_size:
            if current:
                chunks.append(current)
            current = list(cat_rows)
        else:
            current.extend(cat_rows)

    if current:
        chunks.append(current)

    return chunks


# ── Output parsing ────────────────────────────────────────────────────────────

def _parse_llm_output(raw):
    text  = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    text  = re.sub(r"\n?```$",       "", text,         flags=re.MULTILINE).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in LLM response. Preview:\n{raw[:500]}")
    return json.loads(text[start:end])


# ── Result merging ────────────────────────────────────────────────────────────

def _merge_results(all_rows: list) -> list:
    """
    Merge results from multiple LLM chunk calls.
    Deduplicates by (benefitid, serviceTypeID) — last write wins so
    later chunks can refine earlier partial results.
    """
    seen   = {}
    merged = []
    for row in all_rows:
        key = (str(row.get("benefitid", "")), str(row.get("serviceTypeID", "")))
        if key not in seen:
            seen[key] = len(merged)
            merged.append(row)
        else:
            # Later chunk has more complete data — overwrite
            merged[seen[key]] = row
    return merged


# ── Public entry point ────────────────────────────────────────────────────────

def run_benefit_processing(pbp_rows: list, prompts: dict) -> list:
    """
    Process PBP rows through the LLM and return benefit description rows.
    Automatically chunks large plans into multiple LLM calls so there
    is no row count limit.

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

    # Split into chunks — each chunk fits within the context window
    chunks     = _chunk_rows(pbp_rows, CHUNK_SIZE)
    n_chunks   = len(chunks)
    all_rows   = []

    for i, chunk in enumerate(chunks):
        # Tell the LLM which chunk this is so it knows more data may follow
        chunk_note = (
            f"NOTE: This is chunk {i+1} of {n_chunks}. "
            f"Process only the rows in this chunk. "
            f"Return ALL benefit rows you can derive from this chunk."
            if n_chunks > 1 else ""
        )

        raw = _invoke_chain(llm, system_message, human_tmpl, {
            "carrier_prefix": carrier_prefix,
            "file_name":      meta["file_name"],
            "plan_type":      meta["plan_type"],
            "n_pbp":          len(chunk),
            "pbp_json":       json.dumps(chunk, indent=2) + (
                              f"\n\n{chunk_note}" if chunk_note else ""),
        })

        try:
            parsed = _parse_llm_output(raw)
            all_rows.extend(parsed)
        except Exception as e:
            # Log but continue — partial results better than total failure
            print(f"WARNING: chunk {i+1}/{n_chunks} parse error: {e}")

    # Merge and deduplicate across all chunks
    merged = _merge_results(all_rows)

    return (
        pd.DataFrame(merged)
        .reindex(columns=COLS)
        .sort_values(["benefitid", "serviceTypeID"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
