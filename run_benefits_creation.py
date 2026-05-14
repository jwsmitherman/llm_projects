"""
run_benefits_creation.py — batch runner

Reads inputs from blob, processes plan-by-plan via checkpoint_runner, writes
combined output to blob. Resumable — re-run the same LOAD_ID to pick up where
the previous run stopped.

Required environment variables
-------------------------------
  LOAD_ID
  BLOB_CONNECTION_STRING
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY

Optional environment variables
-------------------------------
  BLOB_PAYLOADS_CONTAINER     (default: payloads)
  BLOB_PROMPTS_CONTAINER      (default: prompts)
  BLOB_OUTBOUND_CONTAINER     (default: outbound)
  BLOB_CHECKPOINTS_CONTAINER  (default: checkpoints)
  FORCE_REPROCESS             (default: "0" — set to "1" to ignore checkpoints)
  MAX_PLANS_THIS_RUN          (default: "" — process ALL plans; set to e.g. "10" to cap)
"""

import io
import json
import os

from azure.storage.blob import BlobServiceClient

from build_benefits import __BUILD_VERSION__ as BUILD_VERSION
from checkpoint_runner import process_load_with_checkpoints

# ============================================================
# Config
# ============================================================

LOAD_ID = os.environ["LOAD_ID"]
BLOB_CONNECTION_STRING = os.environ["BLOB_CONNECTION_STRING"]

PAYLOADS_CONTAINER = os.getenv("BLOB_PAYLOADS_CONTAINER", "payloads")
PROMPTS_CONTAINER  = os.getenv("BLOB_PROMPTS_CONTAINER",  "prompts")

FORCE_REPROCESS    = os.getenv("FORCE_REPROCESS", "0") == "1"
_max               = os.getenv("MAX_PLANS_THIS_RUN") or "0"
MAX_PLANS_THIS_RUN = int(_max) if _max.isdigit() and int(_max) > 0 else None

print(f"=== build_benefits version: {BUILD_VERSION} ===")
print(f"LOAD_ID:            {LOAD_ID}")
print(f"FORCE_REPROCESS:    {FORCE_REPROCESS}")
print(f"MAX_PLANS_THIS_RUN: {MAX_PLANS_THIS_RUN or 'unlimited'}\n")


# ============================================================
# Blob read helpers (load inputs only; outputs handled by runner)
# ============================================================

_BLOB = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)


def read_json_from_blob(container_name: str, blob_name: str):
    raw = _BLOB.get_container_client(container_name).download_blob(blob_name).readall()
    return json.loads(raw.decode("utf-8"))


def read_text_from_blob(container_name: str, blob_name: str) -> str:
    raw = _BLOB.get_container_client(container_name).download_blob(blob_name).readall()
    return raw.decode("utf-8")


# ============================================================
# Step 1: Load inputs from blob
# ============================================================

print(f"--- Loading inputs ---")

pbp_blob = f"medicare_input_loadid{LOAD_ID}.json"
pbp_data = read_json_from_blob(PAYLOADS_CONTAINER, pbp_blob)
pbp_rows = pbp_data["pbp"] if isinstance(pbp_data, dict) and "pbp" in pbp_data else pbp_data
print(f"  payload: '{PAYLOADS_CONTAINER}/{pbp_blob}'  ({len(pbp_rows):,} rows)")

PROMPT_FILES = {
    "system_prompt":     "system_prompt.txt",
    "few_shot_examples": "few_shot_examples.txt",
    "human_template":    "human_template.txt",
}
prompts = {
    key: read_text_from_blob(PROMPTS_CONTAINER, fname)
    for key, fname in PROMPT_FILES.items()
}
for key, fname in PROMPT_FILES.items():
    print(f"  prompt:  '{PROMPTS_CONTAINER}/{fname}'  ({len(prompts[key]):>6,} chars)")


# ============================================================
# Step 2: Run plan-by-plan with checkpointing
# ============================================================

summary = process_load_with_checkpoints(
    load_id=LOAD_ID,
    pbp_rows=pbp_rows,
    prompts=prompts,
    force_reprocess=FORCE_REPROCESS,
    max_plans_this_run=MAX_PLANS_THIS_RUN,
)


# ============================================================
# Step 3: Exit code reflects completeness
# ============================================================

if summary["status"] == "success":
    print(f"\n✓ All plans complete.")
    raise SystemExit(0)
elif summary["status"] == "partial":
    print(f"\n⚠ INCOMPLETE — {summary['n_plans_pending']} plan(s) pending, "
          f"{summary['n_plans_failed']} failed.")
    print(f"  Re-run this script to resume. Existing checkpoints will be reused.")
    print(f"  To force reprocessing of completed plans, set FORCE_REPROCESS=1")
    raise SystemExit(1)
else:
    print(f"\n✗ ERROR")
    raise SystemExit(2)
