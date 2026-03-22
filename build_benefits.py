"""
build_benefits.py

Callable benefit-processing module used by main.py.

Public API
----------
run_benefit_processing(input_json: dict, prompts: dict) -> list[dict]

    input_json  — {"pbp": [...], "benefit_rules": [...]}
    prompts     — {"system_prompt": str, "few_shot_examples": str, "human_template": str}

    Returns a list of output row dicts (the 10-column benefit descriptions).
"""

import json
import os
import re

import pandas as pd

try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    from langchain.chat_models import AzureChatOpenAI  # type: ignore

from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================
# LLM initialisation
# ============================================================

def _build_llm() -> AzureChatOpenAI:
    """
    Build the Azure OpenAI client from environment variables.
    All credentials are injected via env — nothing is hard-coded.
    """
    return AzureChatOpenAI(
        azure_endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key          = os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version      = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature      = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0")),
        max_tokens       = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "8192")),
    )


# ============================================================
# Prompt assembly
# ============================================================

def _assemble_prompts(prompts: dict) -> tuple[str, str]:
    """
    Merge the three prompt strings into:
      - system_message : system_prompt + few_shot_examples  (passed as plain string)
      - human_template : human_template with {few_shot} stripped out

    Keeping the few-shot content inside the system message avoids LangChain's
    ChatPromptTemplate scanning {{token}} placeholders as missing variables.
    """
    system_prompt   = prompts["system_prompt"]
    few_shot        = prompts["few_shot_examples"]
    human_template  = prompts["human_template"]

    system_message  = system_prompt + "\n\n" + few_shot
    human_template  = human_template.replace("{few_shot}\n\n", "").replace("{few_shot}", "")

    return system_message, human_template


# ============================================================
# LLM invocation
# ============================================================

def _invoke_chain(llm, system_message: str, human_template: str, template_vars: dict) -> str:
    """
    Fill the human template with template_vars via str.format_map() then call
    the LLM directly — no ChatPromptTemplate, so no brace-scanning KeyErrors.
    """
    human_text = human_template.format_map(template_vars)
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_text),
    ]
    response = llm.invoke(messages)
    return response.content


# ============================================================
# Metadata extraction
# ============================================================

def _extract_plan_meta(input_json: dict) -> dict:
    """
    Pull file_name, plan_type, and load_id from the input JSON.
    load_id is read from the pbp rows (LoadID column).
    """
    pbp_rows = input_json["pbp"]

    file_name = pbp_rows[0]["FileName"].strip() if pbp_rows else ""
    load_id   = str(pbp_rows[0]["LoadID"]).strip() if pbp_rows else ""

    plan_type = "HMO"
    for r in pbp_rows:
        if r.get("header") == "Plan Characteristics" and r.get("field") == "Plan Type":
            plan_type = r["value"].strip()
            break

    return {"file_name": file_name, "plan_type": plan_type, "load_id": load_id}


# ============================================================
# PBP filtering
# ============================================================

def _filter_pbp_for_lookup(pbp_rows: list) -> list:
    """
    Keep only the pbp rows the LLM needs for token lookups:
    formulary exceptions, MOOP, deductibles, plan type, tier names.
    Reduces prompt size without losing anything the LLM needs.
    """
    KEEP_CATEGORIES = {
        "rx setup", "rx characteristics", "rx cost share",
        "plan level cost sharing", "tiering",
    }
    KEEP_FIELDS = {
        "what is your formulary exceptions tier?",
        "in network moop amount",
        "enter deductible amount",
        "plan type",
        "indicate each tier for which the deductible will not apply",
    }
    out = []
    for r in pbp_rows:
        cat = r.get("category", "").lower()
        fld = r.get("field",    "").lower()
        if any(c in cat for c in KEEP_CATEGORIES) or any(f in fld for f in KEEP_FIELDS):
            out.append({
                "category": r.get("category"),
                "field":    r.get("field"),
                "value":    r.get("value"),
            })
    return out


# ============================================================
# Output parsing
# ============================================================

def _parse_llm_output(raw: str) -> list:
    """Extract and parse the JSON array from the LLM response."""
    text = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$",       "", text,         flags=re.MULTILINE).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found in LLM response. Raw response:\n{raw[:500]}")
    return json.loads(text[start:end])


# ============================================================
# Output formatting
# ============================================================

COLS = [
    "planid", "plantypeid", "benefitid", "benefitname",
    "coveragetypeid", "coveragetypedesc",
    "serviceTypeID", "serviceTypeDesc",
    "benefitdesc", "tinyDescription",
]

def _to_output_rows(parsed_rows: list) -> list:
    """Sort and normalise the LLM output into the standard 10-column format."""
    df = (
        pd.DataFrame(parsed_rows)
        .reindex(columns=COLS)
        .sort_values(["benefitid", "serviceTypeID"])
        .reset_index(drop=True)
    )
    return df.to_dict(orient="records")


# ============================================================
# Public entry point
# ============================================================

def run_benefit_processing(input_json: dict, prompts: dict) -> list:
    """
    Process a single plan's benefit rules through the LLM and return
    the list of output row dicts.

    Parameters
    ----------
    input_json : dict
        Must contain:
          "pbp"           — list of PBP row dicts
          "benefit_rules" — list of ParseBenefitFileRule row dicts

    prompts : dict
        Must contain string values for:
          "system_prompt"
          "few_shot_examples"
          "human_template"

    Returns
    -------
    list[dict]  — one dict per (BenefitId, TierType) row, 10 columns each
    """
    llm = _build_llm()

    system_message, human_template = _assemble_prompts(prompts)

    meta       = _extract_plan_meta(input_json)
    pbp_lookup = _filter_pbp_for_lookup(input_json["pbp"])

    carrier_prefix = os.environ.get("CARRIER_PREFIX", "MOM")

    raw_response = _invoke_chain(llm, system_message, human_template, {
        "carrier_prefix": carrier_prefix,
        "file_name":      meta["file_name"],
        "plan_type":      meta["plan_type"],
        "n_rules":        len(input_json["benefit_rules"]),
        "rules_json":     json.dumps(input_json["benefit_rules"], indent=2),
        "n_pbp":          len(pbp_lookup),
        "pbp_json":       json.dumps(pbp_lookup, indent=2),
    })

    parsed  = _parse_llm_output(raw_response)
    results = _to_output_rows(parsed)

    return results
