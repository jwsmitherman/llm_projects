import os, re, json
import pandas as pd
from difflib import get_close_matches

try:
    from langchain_openai import AzureChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
except Exception:
    AzureChatOpenAI = None
    SystemMessage = HumanMessage = None

def normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def money_to_number(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            return float(s.rstrip("%")) / 100.0
        except Exception:
            return s
    if s.startswith("$"):
        s = s[1:]
    try:
        return float(s)
    except Exception:
        return x

def pick_best(source_cols, target_col, cache):
    if target_col in cache:
        return cache[target_col]
    matches = get_close_matches(target_col, source_cols, n=1, cutoff=0.72)
    best = matches[0] if matches else None
    cache[target_col] = best
    return best

SYNONYMS = {
    "plan_id": ["contract_plan_id","plan_id","contract_plan","pbp_id","contract_pbpid"],
    "segment_id": ["segment_id","seg_id"],
    "org_marketing_name": ["organization_marketing_name","org_marketing_name","parent_organization"],
    "plan_marketing_name": ["plan_marketing_name","plan_name","marketing_name","plan_marketing"],
    "plan_type": ["plan_type","org_type","product_type"],
    "county": ["county","service_county","service_area_county"],
    "state": ["state","service_state","state_code"],
    "zip": ["zip","zipcode","postal_code"],
    "effective_year": ["year","benefit_year","effective_year"],
    "medical_deductible": ["medical_deductible","deductible_medical","in_network_deductible"],
    "drug_deductible": ["drug_deductible","pharmacy_deductible","rx_deductible"],
    "moop_in_network": ["moop","in_network_moop","max_oop_in_network","oop_max"],
    "pcp_copay": ["pcp_copay","primary_care_copay","primary_care_visit_copay","pcp_visit_copay"],
    "specialist_copay": ["specialist_copay","specialist_visit_copay","spec_copay"],
    "er_copay": ["er_copay","emergency_room_copay","emergency_care_copay"],
    "urgent_care_copay": ["urgent_care_copay","urgent_care_visit_copay"],
    "inpatient_facility_per_stay": ["inpatient_per_stay","inpatient_facility_per_stay","inpatient_hospital_copay_per_stay"],
    "outpatient_surgery_copay": ["outpatient_surgery_copay","ambulatory_surgery_copay","outpatient_facility_copay"],
    "tier1_generic": ["tier1_generic_copay","generic_copay","pref_generic_copay"],
    "tier2_pref_brand": ["tier2_preferred_brand_copay","preferred_brand_copay","pref_brand_copay"],
    "tier3_nonpref_brand": ["tier3_nonpreferred_brand_copay","nonpreferred_brand_copay","nonpref_brand_copay"],
    "tier4_specialty": ["tier4_specialty_copay","specialty_copay"],
    "dental_coverage": ["dental_benefit","dental","comprehensive_dental"],
    "vision_coverage": ["vision_benefit","vision"],
    "hearing_coverage": ["hearing_benefit","hearing"],
}

NUMERIC_LIKE = {
    "medical_deductible","drug_deductible","moop_in_network",
    "pcp_copay","specialist_copay","er_copay","urgent_care_copay",
    "inpatient_facility_per_stay","outpatient_surgery_copay",
    "tier1_generic","tier2_pref_brand","tier3_nonpref_brand","tier4_specialty",
}

def propose_mapping_with_llm(source_cols, target_cols, logs):
    """Return dict[target_col] = best_source_col or None (using Azure OpenAI)."""
    if AzureChatOpenAI is None:
        logs.append("LangChain/AzureChatOpenAI not installed; skipping LLM mapping.")
        return None

    required = ["AZURE_OPENAI_API_KEY","AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_CHAT_DEPLOYMENT"]
    if not all(os.getenv(k) for k in required):
        logs.append("Azure OpenAI env vars missing; skipping LLM mapping.")
        return None

    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0
        )
        sys = SystemMessage(content=(
            "You are a data integration assistant for Medicare PBP → benefits mapping. "
            "Given source and target headers, return ONLY a compact JSON mapping "
            "from target to source. Use null if no clear match. Do not explain."
        ))
        human = HumanMessage(content=f"""
Create a JSON mapping from target → source.
Target columns: {target_cols}
Source columns: {source_cols}

Rules:
- Prefer exact semantic matches (PBP/PUF conventions).
- If multiple candidates exist, pick the most specific.
- If no good match, set value to null.
Return JSON only.
""")
        resp = llm.invoke([sys, human])
        txt = (resp.content or "").strip()
        start = txt.find("{"); end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            txt = txt[start:end+1]
        mapping = json.loads(txt)
        if not isinstance(mapping, dict):
            raise ValueError("LLM did not return a JSON object")
        logs.append("LLM mapping obtained successfully.")
        return mapping
    except Exception as e:
        logs.append(f"LLM mapping error → fallback: {e}")
        return None

def translate(parsed_path: str, target_benchmark_path: str):
    logs = []
    parsed = pd.read_csv(parsed_path)
    target = pd.read_csv(target_benchmark_path)

    target_cols = list(target.columns)
    target_norm = [normalize_name(c) for c in target_cols]

    source_cols = list(parsed.columns)
    source_norm = [normalize_name(c) for c in source_cols]
    norm_to_source = {normalize_name(c): c for c in source_cols}

    logs.append(f"Loaded PARSED_PBP with {len(parsed)} rows and {len(source_cols)} columns.")
    logs.append(f"Loaded TARGET_BENCHMARK with {len(target_cols)} columns.")

    llm_mapping = propose_mapping_with_llm(source_cols, target_cols, logs)

    def resolve_with_rules(tnorm, cache):
        if tnorm in norm_to_source:
            return norm_to_source[tnorm]
        if tnorm in SYNONYMS:
            for cand in SYNONYMS[tnorm]:
                if cand in norm_to_source:
                    return norm_to_source[cand]
        best = pick_best(source_norm, tnorm, cache)
        if best and best in norm_to_source:
            return norm_to_source[best]
        return None

    resolved_sources = {}
    cache = {}

    if llm_mapping:
        logs.append("Reconciling LLM mapping with fallback rules...")
        for tcol in target_cols:
            src = llm_mapping.get(tcol)
            src_final = None
            if src:
                if src in parsed.columns:
                    src_final = src
                else:
                    nsrc = normalize_name(src)
                    src_final = norm_to_source.get(nsrc)
            if not src_final:
                src_final = resolve_with_rules(normalize_name(tcol), cache)
            if src_final:
                resolved_sources[normalize_name(tcol)] = src_final
    else:
        logs.append("LLM unavailable; using rules + fuzzy mapping only.")
        for tcol in target_cols:
            tnorm = normalize_name(tcol)
            src_final = resolve_with_rules(tnorm, cache)
            if src_final:
                resolved_sources[tnorm] = src_final

    out = pd.DataFrame(columns=target_cols)
    for tcol, tnorm in zip(target_cols, target_norm):
        src = resolved_sources.get(tnorm)
        if src is None:
            out[tcol] = None
            continue
        series = parsed[src]
        if tnorm in NUMERIC_LIKE:
            out[tcol] = series.apply(money_to_number)
        else:
            out[tcol] = series

    if "plan_type" in target_norm:
        tname = target_cols[target_norm.index("plan_type")]
        out[tname] = out[tname].astype(str).str.upper().str.replace("MEDICARE ", "", regex=False)

    logs.append(f"Translated to benefits with {len(out)} rows.")
    return out, logs
