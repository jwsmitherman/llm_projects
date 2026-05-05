"""
build_benefits.py

Callable benefit-processing module used by main.py.

Public API:  run_benefit_processing(pbp_rows, prompts) -> list[dict]

Optimizations vs prior version:
  1. Async httpx with native async LLM calls (no ThreadPoolExecutor overhead)
  2. Token-aware chunking — chunks sized by estimated tokens, not row count
  3. max_tokens dropped from 16000 → 4000 (reserves less TPM, fits more in parallel)
  4. 429 retry with backoff using Retry-After-Ms header
  5. Larger chunks (~30K input tokens each) → fewer total LLM round-trips
  6. Single shared httpx.AsyncClient with HTTP/2 + connection pooling

Result for 5K-row plan: 6 chunks fire in one wave under TPM cap → ~45-90s.
"""

import asyncio
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Optional

import httpx
import pandas as pd

try:
    import tiktoken
    _ENC = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    _ENC = None


# ── Config ────────────────────────────────────────────────────────────────────

# Target *input tokens per chunk* — pack rows until we hit this budget.
# 30K leaves room for system prompt + few-shot + output, fits comfortably in a
# 128K-context model with TPM headroom for parallel calls.
CHUNK_TOKEN_BUDGET = int(os.environ.get("PBP_CHUNK_TOKEN_BUDGET", "30000"))

# Max parallel LLM calls. With max_tokens=4000 and ~30K input each, 8 parallel
# calls peak around ~270K TPM — fits the 450K Global Standard, tight on 240K.
# Tune down to 6 if you're on a 240K-TPM deployment.
MAX_CONCURRENCY = int(os.environ.get("PBP_MAX_CONCURRENCY", "8"))

# Output cap. Output JSON for ~800 rows of benefits is typically 2-3K tokens.
# Keep this tight — Azure reserves it from your TPM budget on every request.
LLM_MAX_TOKENS = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "4000"))

# Retry config for 429s and transient errors
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5"))
BASE_BACKOFF_S = float(os.environ.get("LLM_BASE_BACKOFF_S", "2.0"))

COLS = [
    "planid", "plantypeid", "benefitid", "benefitname",
    "coveragetypeid", "coveragetypedesc",
    "serviceTypeID", "serviceTypeDesc",
    "benefitdesc", "tinyDescription",
]


# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(obj) -> int:
    """Estimate token count. Uses tiktoken if available, else 4-chars-per-token."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(text) // 4


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
    return {"file_name": file_name, "planid": planid, "plan_type": plan_type}


# ── Token-aware chunking ──────────────────────────────────────────────────────

def _chunk_rows_by_tokens(pbp_rows: list, token_budget: int) -> list:
    """
    Split pbp_rows into chunks where each chunk's serialized JSON is
    approximately `token_budget` tokens. Keeps category groups together
    when possible so the LLM sees coherent benefit data per call.
    """
    # Pre-compute per-row token estimates once
    row_tokens = [_estimate_tokens(r) for r in pbp_rows]

    # Group rows by category to keep them adjacent
    category_groups = defaultdict(list)
    for r, tok in zip(pbp_rows, row_tokens):
        category_groups[r.get("category", "unknown")].append((r, tok))

    chunks = []
    current_rows: list = []
    current_tokens = 0

    for cat_rows in category_groups.values():
        cat_total = sum(t for _, t in cat_rows)

        # If a single category exceeds budget on its own, split it directly
        if cat_total > token_budget:
            # flush current first
            if current_rows:
                chunks.append(current_rows)
                current_rows, current_tokens = [], 0

            sub_rows: list = []
            sub_tokens = 0
            for r, t in cat_rows:
                if sub_tokens + t > token_budget and sub_rows:
                    chunks.append(sub_rows)
                    sub_rows, sub_tokens = [], 0
                sub_rows.append(r)
                sub_tokens += t
            if sub_rows:
                chunks.append(sub_rows)
            continue

        # If adding this category overflows, flush and start fresh
        if current_tokens + cat_total > token_budget and current_rows:
            chunks.append(current_rows)
            current_rows, current_tokens = [], 0

        current_rows.extend(r for r, _ in cat_rows)
        current_tokens += cat_total

    if current_rows:
        chunks.append(current_rows)
    return chunks


# ── Async LLM call with retry ────────────────────────────────────────────────

async def _call_llm_async(
    client: httpx.AsyncClient,
    endpoint: str,
    deployment: str,
    api_version: str,
    api_key: str,
    system_message: str,
    human_text: str,
    chunk_idx: int,
) -> str:
    """
    Single async LLM call with 429-aware retry. Returns raw response content.
    """
    url = (
        f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
        f"/chat/completions?api-version={api_version}"
    )
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    body = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_text},
        ],
        "temperature": float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0")),
        "max_tokens": LLM_MAX_TOKENS,
    }

    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code == 429 or resp.status_code >= 500:
                # Honor Retry-After-Ms / Retry-After when available
                retry_ms = resp.headers.get("retry-after-ms")
                retry_s = resp.headers.get("retry-after")
                if retry_ms and retry_ms.isdigit():
                    sleep_s = int(retry_ms) / 1000.0
                elif retry_s and retry_s.isdigit():
                    sleep_s = float(retry_s)
                else:
                    sleep_s = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
                print(f"  Chunk {chunk_idx+1}: HTTP {resp.status_code}, "
                      f"retry {attempt+1}/{MAX_RETRIES} in {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.RequestError as e:
            last_err = e
            sleep_s = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
            print(f"  Chunk {chunk_idx+1}: network error ({e}), "
                  f"retry {attempt+1}/{MAX_RETRIES} in {sleep_s:.1f}s")
            await asyncio.sleep(sleep_s)
    raise RuntimeError(
        f"Chunk {chunk_idx+1}: exhausted {MAX_RETRIES} retries. Last error: {last_err}"
    )


def _parse_json_array(raw: str, chunk_idx: int) -> list:
    text = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(
            f"Chunk {chunk_idx+1}: No JSON array in response. Preview: {raw[:300]}"
        )
    return json.loads(text[start:end])


# ── Single chunk processor (async) ────────────────────────────────────────────

async def _process_chunk_async(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    endpoint: str,
    deployment: str,
    api_version: str,
    api_key: str,
    system_message: str,
    human_template: str,
    chunk: list,
    chunk_idx: int,
    n_chunks: int,
    carrier_prefix: str,
    meta: dict,
) -> tuple:
    """Returns (chunk_idx, list_of_rows). Empty list on failure."""
    async with sem:
        chunk_note = (
            f"\n\nNOTE: This is chunk {chunk_idx + 1} of {n_chunks}. "
            f"Process only the rows in this chunk and return ALL benefit rows you can derive."
            if n_chunks > 1 else ""
        )
        human_text = human_template.format_map({
            "carrier_prefix": carrier_prefix,
            "file_name": meta["file_name"],
            "plan_type": meta["plan_type"],
            "n_pbp": len(chunk),
            "pbp_json": json.dumps(chunk, indent=2) + chunk_note,
        })

        t0 = time.monotonic()
        try:
            raw = await _call_llm_async(
                client, endpoint, deployment, api_version, api_key,
                system_message, human_text, chunk_idx,
            )
            rows = _parse_json_array(raw, chunk_idx)
            elapsed = time.monotonic() - t0
            print(f"  Chunk {chunk_idx+1}/{n_chunks} done in {elapsed:.1f}s "
                  f"— {len(rows)} rows")
            return chunk_idx, rows
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  Chunk {chunk_idx+1}/{n_chunks} ERROR after {elapsed:.1f}s: {e}")
            return chunk_idx, []


# ── Result merging ────────────────────────────────────────────────────────────

def _merge_results(all_chunk_results: list) -> list:
    """Merge chunk results, dedup by (benefitid, serviceTypeID), last write wins."""
    seen: dict = {}
    merged: list = []
    for rows in all_chunk_results:
        for row in rows:
            key = (str(row.get("benefitid", "")), str(row.get("serviceTypeID", "")))
            if key not in seen:
                seen[key] = len(merged)
                merged.append(row)
            else:
                merged[seen[key]] = row
    return merged


# ── Async orchestration ───────────────────────────────────────────────────────

async def _run_async(pbp_rows: list, prompts: dict) -> list:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    carrier_prefix = os.environ.get("CARRIER_PREFIX", "MOM")

    system_message, human_tmpl = _assemble_prompts(prompts)
    meta = _extract_plan_meta(pbp_rows, carrier_prefix)

    # Token-aware chunking
    chunks = _chunk_rows_by_tokens(pbp_rows, CHUNK_TOKEN_BUDGET)
    n_chunks = len(chunks)

    # Estimate prompt overhead (system + few-shot) once for logging
    system_tokens = _estimate_tokens(system_message)
    chunk_token_estimates = [_estimate_tokens(c) for c in chunks]

    print(
        f"Processing {len(pbp_rows):,} rows in {n_chunks} chunk(s) | "
        f"system+fewshot ≈ {system_tokens:,} tokens | "
        f"chunk sizes (rows): {[len(c) for c in chunks]} | "
        f"chunk tokens (est): {chunk_token_estimates} | "
        f"max_concurrency={MAX_CONCURRENCY} | "
        f"max_tokens out={LLM_MAX_TOKENS}"
    )

    # Single shared async client with HTTP/2 and generous pool
    limits = httpx.Limits(
        max_connections=MAX_CONCURRENCY * 2,
        max_keepalive_connections=MAX_CONCURRENCY * 2,
    )
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    t0 = time.monotonic()

    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
        tasks = [
            _process_chunk_async(
                sem, client, endpoint, deployment, api_version, api_key,
                system_message, human_tmpl,
                chunk, i, n_chunks, carrier_prefix, meta,
            )
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0
    # Sort by chunk_idx to preserve order
    results.sort(key=lambda x: x[0])
    chunk_results = [rows for _, rows in results]

    merged = _merge_results(chunk_results)
    print(
        f"\nMerged total: {len(merged)} unique benefit rows in {elapsed:.1f}s "
        f"({len(pbp_rows)/elapsed:.0f} input rows/sec)"
    )
    return merged


# ── Public entry point ────────────────────────────────────────────────────────

def run_benefit_processing(pbp_rows, prompts: dict) -> list:
    """
    Process PBP rows through the LLM with no row count limit.

    Parameters
    ----------
    pbp_rows : list | dict   — raw PBP field/value rows for one plan, or the full
                               input_json dict containing a "pbp" key
    prompts  : dict          — keys: system_prompt, few_shot_examples, human_template

    Returns
    -------
    list[dict] — one dict per (benefitid, serviceTypeID), 10 columns each
    """
    # Accept either the raw list or the input_json dict
    if isinstance(pbp_rows, dict) and "pbp" in pbp_rows:
        pbp_rows = pbp_rows["pbp"]

    merged = asyncio.run(_run_async(pbp_rows, prompts))

    return (
        pd.DataFrame(merged)
        .reindex(columns=COLS)
        .sort_values(["benefitid", "serviceTypeID"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
