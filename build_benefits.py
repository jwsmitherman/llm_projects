"""
build_benefits.py

Callable benefit-processing module used by main.py.

Public API:  run_benefit_processing(pbp_rows, prompts) -> list[dict]

The LLM derives all benefit descriptions purely from PBP data +
few-shot examples in the prompts. No benefit_rules table needed.
"""

import json, os, re
import pandas as pd

try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    from langchain.chat_models import AzureChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage


def _build_llm():
    return AzureChatOpenAI(
        azure_endpoint   = os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key          = os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_version      = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature      = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0")),
        max_tokens       = int(os.environ.get("AZURE_OPENAI_MAX_TOKENS", "16000")),
    )


def _assemble_prompts(prompts):
    system_message = prompts["system_prompt"] + "\n\n" + prompts["few_shot_examples"]
    human_template = prompts["human_template"]
    human_template = human_template.replace("{few_shot}\n\n", "").replace("{few_shot}", "")
    return system_message, human_template


def _invoke_chain(llm, system_message, human_template, template_vars):
    human_text = human_template.format_map(template_vars)
    response   = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_text),
    ])
    return response.content


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


def _parse_llm_output(raw):
    text  = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    text  = re.sub(r"\n?```$",       "", text,         flags=re.MULTILINE).strip()
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in LLM response. Preview:\n{raw[:500]}")
    return json.loads(text[start:end])


COLS = [
    "planid", "plantypeid", "benefitid", "benefitname",
    "coveragetypeid", "coveragetypedesc",
    "serviceTypeID", "serviceTypeDesc",
    "benefitdesc", "tinyDescription",
]


def run_benefit_processing(pbp_rows: list, prompts: dict) -> list:
    """
    Process PBP rows through the LLM and return benefit description rows.

    Parameters
    ----------
    pbp_rows : list   — raw PBP field/value rows for one plan
    prompts  : dict   — keys: system_prompt, few_shot_examples, human_template

    Returns
    -------
    list[dict] — one dict per (benefitid, serviceTypeID), 10 columns each
    """
    llm                          = _build_llm()
    system_message, human_tmpl   = _assemble_prompts(prompts)
    carrier_prefix               = os.environ.get("CARRIER_PREFIX", "MOM")
    meta                         = _extract_plan_meta(pbp_rows, carrier_prefix)

    raw = _invoke_chain(llm, system_message, human_tmpl, {
        "carrier_prefix": carrier_prefix,
        "file_name":      meta["file_name"],
        "plan_type":      meta["plan_type"],
        "n_pbp":          len(pbp_rows),
        "pbp_json":       json.dumps(pbp_rows, indent=2),
    })

    parsed = _parse_llm_output(raw)
    return (
        pd.DataFrame(parsed)
        .reindex(columns=COLS)
        .sort_values(["benefitid", "serviceTypeID"])
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
