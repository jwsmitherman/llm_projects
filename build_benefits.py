"""
build_benefits.py

Callable benefit-processing module used by main.py.

Public API:  run_benefit_processing(pbp_rows, prompts) -> list[dict]

Strategy:
  - Default: send the entire plan in ONE LLM call. A typical 1,300-row plan
    fits comfortably in gpt-4o's 128K context (~100K input tokens), and the
    LLM produces dramatically better results when it sees complete context
    instead of fragmented chunks.
  - Fallback: if the plan is too big to fit in one call, split into chunks.
    This rarely happens for normal Medicare Advantage plans.

Carrier handling:
  - Carrier prefix is auto-detected from FileName (e.g. "Humana_..." → "Humana"
    by lookup, falling back to the FileName prefix itself).
  - You can override with CARRIER_PREFIX_OVERRIDE if needed.
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

try:
    import tiktoken
    _ENC = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    _ENC = None


# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded config — UPDATE THE TWO PLACEHOLDERS BELOW BEFORE RUNNING
# ─────────────────────────────────────────────────────────────────────────────

# Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT    = "<YOUR_ENDPOINT>"   # e.g. "https://my-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY     = "<YOUR_API_KEY>"
AZURE_OPENAI_DEPLOYMENT  = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
# Set to None to omit (required for some gpt-5 / o-series deployments which
# only accept the default temperature). Set to 0.0 for deterministic gpt-4o output.
AZURE_OPENAI_TEMPERATURE = None

# Carrier prefix override. Leave as None to auto-detect from FileName.
# Set to a string like "MOM" or "AETNA" to force a specific value.
CARRIER_PREFIX_OVERRIDE = None

# FileName-prefix to carrier-id mapping. Add carriers here as you onboard them.
# Match is case-insensitive; the longest matching prefix wins.
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

# Single-call vs chunked strategy
# If the full prompt fits within SINGLE_CALL_TOKEN_LIMIT, send it all in one
# request (best results). Above this, fall back to chunking.
SINGLE_CALL_TOKEN_LIMIT = 110000  # gpt-4o context is 128K — leave headroom for output

# Chunking config (only used if plan exceeds single-call limit)
CHUNK_TOKEN_BUDGET = 30000
MAX_CONCURRENCY    = 8

# Output token cap. Expected output for a full plan is ~10-15K tokens.
# Set high enough to cover the full plan in single-call mode.
LLM_MAX_TOKENS = 16000

# Retry config for 429s and transient errors
MAX_RETRIES    = 3
BASE_BACKOFF_S = 2.0

# Per-request HTTP timeouts (seconds). Single-call mode needs longer reads
# because the LLM is generating more output.
HTTP_CONNECT_TIMEOUT = 10.0
HTTP_READ_TIMEOUT    = 300.0
HTTP_WRITE_TIMEOUT   = 60.0

# Output column order
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


# ── Carrier detection ─────────────────────────────────────────────────────────

def _detect_carrier_prefix(file_name: str) -> str:
    """
    Pick a carrier prefix from FileName. Order of preference:
      1. CARRIER_PREFIX_OVERRIDE if set
      2. Longest match against CARRIER_PREFIX_MAP keys (case-insensitive)
      3. Substring before first underscore (uppercased)
    """
    if CARRIER_PREFIX_OVERRIDE:
        return CARRIER_PREFIX_OVERRIDE

    if not file_name:
        return "UNKNOWN"

    fname_lower = file_name.lower()
    matches = [(k, v) for k, v in CARRIER_PREFIX_MAP.items() if fname_lower.startswith(k)]
    if matches:
        matches.sort(key=lambda kv: -len(kv[0]))  # longest match wins
        return matches[0][1]

    # Fallback: the literal carrier name from the file before "_"
    if "_" in file_name:
        return file_name.split("_")[0].upper()
    return file_name.upper()


# ── Plan metadata ─────────────────────────────────────────────────────────────

def _extract_plan_meta(pbp_rows):
    file_name = pbp_rows[0].get("FileName", "").strip() if pbp_rows else ""
    carrier_prefix = _detect_carrier_prefix(file_name)

    # Strip carrier prefix from filename to extract contract/plan/segment
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


# ── Async LLM call with retry ────────────────────────────────────────────────

async def _call_llm_async(
    client: httpx.AsyncClient,
    system_message: str,
    human_text: str,
    label: str = "single",
) -> str:
    """Single async LLM call with 429-aware retry. Returns raw response content."""
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

            # Retryable: throttling and 5xx
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_ms = resp.headers.get("retry-after-ms")
                retry_s = resp.headers.get("retry-after")
                if retry_ms and retry_ms.isdigit():
                    sleep_s = int(retry_ms) / 1000.0
                elif retry_s and retry_s.isdigit():
                    sleep_s = float(retry_s)
                else:
                    sleep_s = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
                print(f"  [{label}] HTTP {resp.status_code}, "
                      f"retry {attempt+1}/{MAX_RETRIES} in {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)
                continue

            # Non-retryable 4xx: surface Azure's error body
            if 400 <= resp.status_code < 500:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text
                raise RuntimeError(
                    f"HTTP {resp.status_code} from Azure OpenAI. Body: {err_body}"
                )

            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.RequestError as e:
            last_err = e
            sleep_s = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
            print(f"  [{label}] network error ({e}), "
                  f"retry {attempt+1}/{MAX_RETRIES} in {sleep_s:.1f}s")
            await asyncio.sleep(sleep_s)
    raise RuntimeError(
        f"[{label}] exhausted {MAX_RETRIES} retries. Last error: {last_err}"
    )


def _parse_json_array(raw: str, label: str) -> list:
    text = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(
            f"[{label}] No JSON array in response. Preview: {raw[:300]!r}"
        )
    return json.loads(text[start:end])


# ── Result merging (only used in chunked mode) ────────────────────────────────

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


# ── Token-aware chunking (fallback path only) ─────────────────────────────────

def _chunk_rows_by_tokens(pbp_rows: list, token_budget: int) -> list:
    row_tokens = [_estimate_tokens(r) for r in pbp_rows]
    category_groups = defaultdict(list)
    for r, tok in zip(pbp_rows, row_tokens):
        category_groups[r.get("category", "unknown")].append((r, tok))

    chunks = []
    current_rows: list = []
    current_tokens = 0

    for cat_rows in category_groups.values():
        cat_total = sum(t for _, t in cat_rows)
        if cat_total > token_budget:
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
        if current_tokens + cat_total > token_budget and current_rows:
            chunks.append(current_rows)
            current_rows, current_tokens = [], 0
        current_rows.extend(r for r, _ in cat_rows)
        current_tokens += cat_total

    if current_rows:
        chunks.append(current_rows)
    return chunks


# ── Build human prompt ────────────────────────────────────────────────────────

def _build_human_text(human_template: str, rows: list, meta: dict,
                      chunk_idx: int = 0, n_chunks: int = 1) -> str:
    chunk_note = (
        f"\n\nNOTE: This is chunk {chunk_idx + 1} of {n_chunks}. "
        f"Process only the rows in this chunk and return ALL benefit rows you can derive."
        if n_chunks > 1 else ""
    )
    return human_template.format_map({
        "carrier_prefix": meta["carrier_prefix"],
        "file_name":      meta["file_name"],
        "plan_type":      meta["plan_type"],
        "n_pbp":          len(rows),
        "pbp_json":       json.dumps(rows, indent=2) + chunk_note,
    })


# ── Single-call strategy ──────────────────────────────────────────────────────

async def _run_single_call(client: httpx.AsyncClient, system_message: str,
                           human_template: str, pbp_rows: list, meta: dict) -> list:
    human_text = _build_human_text(human_template, pbp_rows, meta)
    print(f"Single-call mode: {len(pbp_rows):,} rows, "
          f"~{_estimate_tokens(human_text):,} input tokens")
    t0 = time.monotonic()
    raw = await _call_llm_async(client, system_message, human_text, label="single")
    elapsed_call = time.monotonic() - t0
    preview = raw[:200].replace("\n", " ")
    print(f"  LLM returned {len(raw):,} chars in {elapsed_call:.1f}s")
    print(f"  raw preview: {preview!r}")

    rows = _parse_json_array(raw, label="single")
    print(f"Parsed {len(rows)} benefit rows")
    return rows


# ── Chunked strategy (fallback) ───────────────────────────────────────────────

async def _process_chunk_async(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    system_message: str,
    human_template: str,
    chunk: list,
    chunk_idx: int,
    n_chunks: int,
    meta: dict,
) -> tuple:
    async with sem:
        human_text = _build_human_text(human_template, chunk, meta, chunk_idx, n_chunks)
        label = f"chunk {chunk_idx+1}/{n_chunks}"
        t0 = time.monotonic()
        try:
            raw = await _call_llm_async(client, system_message, human_text, label)
            rows = _parse_json_array(raw, label)
            elapsed = time.monotonic() - t0
            print(f"  [{label}] done in {elapsed:.1f}s — {len(rows)} rows")
            return chunk_idx, rows
        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"  [{label}] ERROR after {elapsed:.1f}s: {e}")
            return chunk_idx, []


async def _run_chunked(client: httpx.AsyncClient, system_message: str,
                       human_template: str, pbp_rows: list, meta: dict) -> list:
    chunks = _chunk_rows_by_tokens(pbp_rows, CHUNK_TOKEN_BUDGET)
    n_chunks = len(chunks)
    print(f"Chunked mode: {len(pbp_rows):,} rows in {n_chunks} chunk(s) | "
          f"max_concurrency={MAX_CONCURRENCY}")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        _process_chunk_async(sem, client, system_message, human_template,
                             chunk, i, n_chunks, meta)
        for i, chunk in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    return _merge_results([rows for _, rows in results])


# ── Async orchestration ───────────────────────────────────────────────────────

async def _run_async(pbp_rows: list, prompts: dict) -> list:
    system_message, human_tmpl = _assemble_prompts(prompts)
    meta = _extract_plan_meta(pbp_rows)

    system_tokens = _estimate_tokens(system_message)
    print(f"Plan: {meta['planid']} (carrier={meta['carrier_prefix']}, "
          f"plan_type={meta['plan_type']}) | system+fewshot ≈ {system_tokens:,} tokens")

    # Decide single-call vs chunked
    full_input = json.dumps(pbp_rows)
    full_input_tokens = _estimate_tokens(full_input)
    total_estimate = system_tokens + full_input_tokens + LLM_MAX_TOKENS

    print(f"Full payload: {full_input_tokens:,} tokens | "
          f"with system+output: {total_estimate:,} tokens | "
          f"single-call limit: {SINGLE_CALL_TOKEN_LIMIT:,}")

    limits = httpx.Limits(
        max_connections=max(MAX_CONCURRENCY * 2, 4),
        max_keepalive_connections=max(MAX_CONCURRENCY * 2, 4),
    )
    timeout = httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT,
        read=HTTP_READ_TIMEOUT,
        write=HTTP_WRITE_TIMEOUT,
        pool=10.0,
    )

    t0 = time.monotonic()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        if total_estimate <= SINGLE_CALL_TOKEN_LIMIT:
            rows = await _run_single_call(client, system_message, human_tmpl,
                                           pbp_rows, meta)
        else:
            print("Plan too large for single-call mode, falling back to chunking")
            rows = await _run_chunked(client, system_message, human_tmpl,
                                      pbp_rows, meta)

    elapsed = time.monotonic() - t0
    print(f"\nTotal: {len(rows)} unique benefit rows in {elapsed:.1f}s")
    return rows


# ── Public entry point ────────────────────────────────────────────────────────

def run_benefit_processing(pbp_rows, prompts: dict) -> list:
    """
    Process PBP rows through the LLM.

    Parameters
    ----------
    pbp_rows : list | dict   — raw PBP field/value rows for one plan, or the full
                               input_json dict containing a "pbp" key
    prompts  : dict          — keys: system_prompt, few_shot_examples, human_template

    Returns
    -------
    list[dict] — one dict per (benefitid, serviceTypeID), 10 columns each
    """
    if isinstance(pbp_rows, dict) and "pbp" in pbp_rows:
        pbp_rows = pbp_rows["pbp"]

    rows = asyncio.run(_run_async(pbp_rows, prompts))

    return (
        pd.DataFrame(rows)
        .reindex(columns=COLS)
        .sort_values(["benefitid", "serviceTypeID"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
