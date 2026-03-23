"""
run_benefits_creation.py

Standalone script that reads ALL inputs from Azure Blob Storage,
runs benefit processing, and writes results back to blob.

Blob containers
---------------
  payloads  — medicare_input_loadid{LOAD_ID}.json   (pbp rows)
              benefit_rules_{LOAD_ID}.json           (benefit rules)
  prompts   — system_prompt.txt
              few_shot_examples.txt
              human_template.txt
  outbound  — output_benefits_{LOAD_ID}.json        (results written here)

Required environment variables
-------------------------------
  LOAD_ID                   e.g. "142"
  BLOB_CONNECTION_STRING
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY

Optional environment variables
-------------------------------
  AZURE_OPENAI_DEPLOYMENT   (default: gpt-4o)
  AZURE_OPENAI_API_VERSION  (default: 2024-12-01-preview)
  CARRIER_PREFIX            (default: MOM)
  BLOB_PAYLOADS_CONTAINER   (default: payloads)
  BLOB_PROMPTS_CONTAINER    (default: prompts)
  BLOB_OUTBOUND_CONTAINER   (default: outbound)
"""

import io
import json
import os

import pandas as pd
from azure.storage.blob import BlobServiceClient, ContentSettings

from build_benefits import run_benefit_processing

# ============================================================
# Config — all values from environment variables
# ============================================================

LOAD_ID                = os.environ["LOAD_ID"]
BLOB_CONNECTION_STRING = os.environ["BLOB_CONNECTION_STRING"]

PAYLOADS_CONTAINER = os.getenv("BLOB_PAYLOADS_CONTAINER", "payloads")
PROMPTS_CONTAINER  = os.getenv("BLOB_PROMPTS_CONTAINER",  "prompts")
OUTBOUND_CONTAINER = os.getenv("BLOB_OUTBOUND_CONTAINER", "outbound")

# ============================================================
# Blob helpers
# ============================================================

def _container(name: str):
    return BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING).get_container_client(name)


def read_json_from_blob(container_name: str, blob_name: str):
    raw = _container(container_name).download_blob(blob_name).readall()
    return json.loads(raw.decode("utf-8"))


def read_text_from_blob(container_name: str, blob_name: str) -> str:
    raw = _container(container_name).download_blob(blob_name).readall()
    return raw.decode("utf-8")


def save_json_to_blob(container_name: str, blob_name: str, data) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    _container(container_name).upload_blob(
        name=blob_name,
        data=io.BytesIO(payload),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
    )

# ============================================================
# Step 1: Load PBP rows from blob
# ============================================================

pbp_blob = f"medicare_input_loadid{LOAD_ID}.json"
pbp_data = read_json_from_blob(PAYLOADS_CONTAINER, pbp_blob)

# pbp_data may be the full input_json dict (both keys) or just the pbp list
pbp_rows = pbp_data["pbp"] if isinstance(pbp_data, dict) and "pbp" in pbp_data else pbp_data

print(f"Loaded pbp:   '{PAYLOADS_CONTAINER}/{pbp_blob}'  ({len(pbp_rows):,} rows)")

# ============================================================
# Step 2: Load benefit_rules from blob
# ============================================================

rules_blob    = f"benefit_rules_{LOAD_ID}.json"
benefit_rules = read_json_from_blob(PAYLOADS_CONTAINER, rules_blob)

print(f"Loaded rules: '{PAYLOADS_CONTAINER}/{rules_blob}'  ({len(benefit_rules):,} rows)")

# ============================================================
# Step 3: Build input_json
# ============================================================

input_json = {
    "pbp":           pbp_rows,
    "benefit_rules": benefit_rules,
}

# ============================================================
# Step 4: Load prompts from blob
# ============================================================

PROMPT_FILES = {
    "system_prompt":     "system_prompt.txt",
    "few_shot_examples": "few_shot_examples.txt",
    "human_template":    "human_template.txt",
}

prompts = {
    key: read_text_from_blob(PROMPTS_CONTAINER, filename)
    for key, filename in PROMPT_FILES.items()
}

print(f"\nLoaded prompts from '{PROMPTS_CONTAINER}':")
for key, filename in PROMPT_FILES.items():
    print(f"  {filename:<28}  {len(prompts[key]):>6,} chars")

# ============================================================
# Step 5: Run benefit processing
# ============================================================

print("\nCalling Azure OpenAI...")
output_rows = run_benefit_processing(input_json, prompts)
print(f"Response parsed — {len(output_rows)} benefit rows returned")

# ============================================================
# Step 6: Save results to outbound blob
# ============================================================

out_blob = f"output_benefits_{LOAD_ID}.json"
save_json_to_blob(OUTBOUND_CONTAINER, out_blob, output_rows)
print(f"\nResults saved: '{OUTBOUND_CONTAINER}/{out_blob}'")

# ============================================================
# Step 7: Print summary table
# ============================================================

COLS = [
    "planid", "plantypeid", "benefitid", "benefitname",
    "coveragetypeid", "coveragetypedesc",
    "serviceTypeID", "serviceTypeDesc",
    "benefitdesc", "tinyDescription",
]

df = (
    pd.DataFrame(output_rows)
    .reindex(columns=COLS)
    .sort_values(["benefitid", "serviceTypeID"])
    .reset_index(drop=True)
)

print(f"\n{len(df)} rows total:")
print(df[["benefitid", "benefitname", "serviceTypeDesc", "tinyDescription"]].to_string(index=False))
