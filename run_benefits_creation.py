"""
run_benefits_creation.py

Standalone script for local / notebook testing.
Loads input_json and prompts from disk, calls build_benefits.run_benefit_processing,
and writes the output CSV to ./results/.
"""

import json
import os

import pandas as pd

from build_benefits import run_benefit_processing

# ============================================================
# Config
# ============================================================

INPUT_JSON_PATH = "./payloads/medicare_input_loadid142.json"
PROMPT_DIR      = "./prompts"
RESULTS_DIR     = "./results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Step 1: Load input JSON
# ============================================================

with open(INPUT_JSON_PATH, encoding="utf-8") as f:
    input_json = json.load(f)

load_id = str(input_json["benefit_rules"][0]["LoadID"])

print(f"Loaded: {INPUT_JSON_PATH}")
print(f"  load_id:       {load_id}")
print(f"  pbp rows:      {len(input_json['pbp']):,}")
print(f"  benefit_rules: {len(input_json['benefit_rules']):,}")
print()

# ============================================================
# Step 2: Load prompts from disk
# ============================================================

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


prompts = {
    "system_prompt":     load_prompt("system_prompt.txt"),
    "few_shot_examples": load_prompt("few_shot_examples.txt"),
    "human_template":    load_prompt("human_template.txt"),
}

print(f"system_prompt.txt      {len(prompts['system_prompt']):>6,} chars")
print(f"few_shot_examples.txt  {len(prompts['few_shot_examples']):>6,} chars")
print(f"human_template.txt     {len(prompts['human_template']):>6,} chars")
print()

# ============================================================
# Step 3: Run processing
# ============================================================

print("Calling Azure OpenAI...")
output_rows = run_benefit_processing(input_json, prompts)
print(f"Response parsed — {len(output_rows)} benefit rows returned")
print()

# ============================================================
# Step 4: Write CSV
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

CSV_OUT = os.path.join(RESULTS_DIR, f"output_benefits_loadid_{load_id}.csv")
df.to_csv(CSV_OUT, index=False)

print(f"{len(df)} rows written to {CSV_OUT}")
print(df[["benefitid", "benefitname", "serviceTypeDesc", "tinyDescription"]].to_string(index=False))
