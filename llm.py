"""Real model evaluation, replacing the static fixtures.

Drops into the med_nec_api folder next to app.py. Returns the same shape a
fixture block returns, so app.py's envelope needs no change.

    CLAUDE_BASE_URL         https://<resource>.services.ai.azure.com/anthropic
    CLAUDE_DEPLOYMENT_NAME  the deployment name, not the model id
    AZURE_CLIENT_ID         the managed identity client id
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

SCOPE = "https://ai.azure.com/.default"
KNOWLEDGE_PATH = Path(__file__).parent / "med_nec_knowledge.json"

MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "4000"))
TEMPERATURE = float(os.environ.get("CLAUDE_TEMPERATURE", "0"))

_client = None
_prompt_cache: Optional[str] = None
_knowledge_cache: Optional[Dict[str, Any]] = None


class AnthropicIdentity(Anthropic):
    def __init__(self, *, azure_ad_token_provider, base_url, **kwargs):
        self._token_provider = azure_ad_token_provider
        super().__init__(auth_token="placeholder", api_key=None, base_url=base_url, **kwargs)

    @property
    def auth_token(self):
        return self._token_provider()

    @auth_token.setter
    def auth_token(self, value):
        return


def get_client() -> Anthropic:
    global _client
    if _client is not None:
        return _client
    base_url = os.environ.get("CLAUDE_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError("CLAUDE_BASE_URL is not set")
    if not base_url.rstrip("/").endswith("/anthropic"):
        raise RuntimeError("CLAUDE_BASE_URL must end in /anthropic")
    client_id = os.environ.get("AZURE_CLIENT_ID", "").strip() or None
    credential = DefaultAzureCredential(managed_identity_client_id=client_id)
    provider = get_bearer_token_provider(credential, SCOPE)
    _client = AnthropicIdentity(azure_ad_token_provider=provider, base_url=base_url)
    return _client


def get_knowledge() -> Dict[str, Any]:
    global _knowledge_cache
    if _knowledge_cache is None:
        with KNOWLEDGE_PATH.open() as handle:
            _knowledge_cache = json.load(handle)
    return _knowledge_cache


def build_prompt() -> str:
    """Compiled once at startup, not per request."""
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache

    kb = get_knowledge()

    concepts = [
        f'  {c["concept"]} | {c["axis"]} axis | {c["status"]} | phrasing: '
        f'{", ".join(c["terms"][:6])}'
        for c in sorted(kb["concepts"], key=lambda x: x["concept"])
    ]
    presumed = [f'  {p["id"]} | {p["statement"]}' for p in kb["presumed_necessary"]]
    bed = [f'  {e["element"]} | {e["statement"]}' for e in kb["bed_confinement"]["elements"]]
    exclusions = [f'  {e["exclusion_id"]} | {e["statement"]}' for e in kb["exclusions"]]
    rules = [f'  - {r["statement"]}' for r in kb["policy_rules"]]
    denial = [f'  - {d}' for d in kb["denial_criteria"]["elements"]]

    _prompt_cache = f"""
You review order form answers for a non-emergent ground ambulance transport and return a
medical necessity assessment. You judge only what the answers state. You do not infer, you do
not assume, and you handle negation.

The governing question is: {kb["determination_question"]}
For non-emergency transports also ask: {kb["non_emergency_question"]}
{kb["evaluation_principle"]}

You receive every question and answer from one trip at once. Read them together. A criterion
that needs several elements may have them spread across different answers.

CRITERIA THAT SUPPORT NECESSITY:
{chr(10).join(concepts)}

CONDITIONS FOR WHICH NECESSITY MAY BE PRESUMED:
{chr(10).join(presumed)}

BED CONFINEMENT. All three elements are required together. Any one alone is not sufficient,
and bed confinement by itself is neither sufficient nor necessary for coverage.
{chr(10).join(bed)}

SITUATIONS THAT DO NOT ESTABLISH NECESSITY:
{chr(10).join(exclusions)}

DENIAL CRITERIA. A trip fails only when ALL of these are true at once. If any one is false the
trip goes to clinical review rather than a negative determination.
{chr(10).join(denial)}

RULES THAT GOVERN THE ASSESSMENT:
{chr(10).join(rules)}

These phrases are non-specific and never establish anything on their own:
{", ".join(kb["non_specific_phrases"])}

Return THREE things.

1. clinicalSummary. Two sentences in plain English describing the patient's relevant clinical
   condition. Positives only. Omit every answer that reports nothing, says no, or does not
   apply. This is written for a biller to read, not a checklist.

2. meetsMedicalNecessity. True or false.

3. clarifyingQuestion. One question, or null. Return a question only when a single further
   detail could change the determination, and when nothing else already documented settles it.
   If the trip clearly meets criteria, return null. If no answer could rescue it, return null.
   Ask about the patient's condition, never about the form.

OUTPUT valid JSON only, no prose and no code fences, all keys present:
{{
  "clinicalSummary": string,
  "meetsMedicalNecessity": true | false,
  "clarifyingQuestion": string | null,
  "detail": {{
    "determination": "meets" | "does_not_meet" | "needs_review",
    "basis": string,
    "confidence": "High" | "Medium" | "Low",
    "criteriaDocumented": [
      {{"criterionId": string, "citation": string, "evidence": string, "questionId": integer | null}}
    ],
    "exclusionsTriggered": [
      {{"exclusionId": string, "citation": string, "evidence": string, "statement": string}}
    ],
    "questionFindings": [
      {{"questionId": integer | null, "question": string, "answer": string,
        "nonMedNec": true | false,
        "category": "supports_necessity" | "insufficient_alone" | "non_covered" | "administrative" | "unclear",
        "criterionId": string, "citation": string, "evidence": string,
        "reason": string, "confidence": "High" | "Medium" | "Low"}}
    ]
  }}
}}

Every criterion you record as documented must quote the answer text it rests on, verbatim.
A criterion you cannot quote is not documented.
Return one questionFindings entry for every question and answer pair you are given.
""".strip()
    return _prompt_cache


def extract_pairs(trip: Dict[str, Any]) -> List[Dict[str, Any]]:
    legs = trip.get("tripLegs") or []
    leg = legs[0] if legs else {}
    pairs: List[Dict[str, Any]] = []

    for q in leg.get("losQuestions") or []:
        if not isinstance(q, dict):
            continue
        answer = str(q.get("answer") or "").strip()
        if not answer:
            continue
        free_text = str(q.get("answerValue") or "").strip()
        combined = answer
        if free_text and free_text.lower() != answer.lower():
            combined = f"{answer}: {free_text}"
        pairs.append({
            "questionId": q.get("questionId"),
            "question": str(q.get("questionText") or "").strip(),
            "answer": combined,
        })

    if pairs:
        return pairs

    blob = str(leg.get("clinicalData") or "").strip()
    if blob:
        pairs.append({
            "questionId": None,
            "question": "Clinical data, flattened form text",
            "answer": blob,
        })
    return pairs


def build_user_message(trip: Dict[str, Any], pairs: List[Dict[str, Any]],
                       clarification: Optional[Dict[str, str]] = None) -> str:
    legs = trip.get("tripLegs") or []
    leg = legs[0] if legs else {}

    lines = [
        f"Trip request: {trip.get('olosRequestId')}",
        f"Requesting facility: {trip.get('requesterFacility')}",
        f"Level of service requested: {leg.get('levelOfService')}",
        f"Transfer reason: {trip.get('tripReason')}",
        "",
        f"Question and answer pairs ({len(pairs)}):",
    ]
    for i, pair in enumerate(pairs):
        qid = pair["questionId"]
        lines.append(f"{i}. [questionId {qid if qid is not None else 'none'}] "
                     f"QUESTION: {pair['question']}")
        lines.append(f"   ANSWER: {pair['answer']}")

    if clarification:
        lines += [
            "",
            "A clarifying question was already put to the requester and answered.",
            "Take the answer into account and reassess.",
            f"  QUESTION ASKED: {clarification.get('question')}",
            f"  REQUESTER ANSWERED: {clarification.get('answer')}",
            "",
            "Fold the answer into the clinical summary. Return null for clarifyingQuestion;",
            "the requester is only asked once.",
        ]
    return "\n".join(lines)


def strip_fences(raw: str) -> str:
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def validate(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        obj = json.loads(strip_fences(raw))
    except Exception as exc:
        return None, f"unparseable json: {exc}"
    if not isinstance(obj, dict):
        return None, "response is not an object"

    for field in ("clinicalSummary", "meetsMedicalNecessity", "clarifyingQuestion"):
        if field not in obj:
            return None, f"missing field: {field}"
    if not isinstance(obj["meetsMedicalNecessity"], bool):
        return None, "meetsMedicalNecessity is not a boolean"

    detail = obj.get("detail")
    if not isinstance(detail, dict):
        detail = {}
    detail.setdefault("determination", "needs_review")
    detail.setdefault("basis", "")
    detail.setdefault("confidence", "Low")
    for key in ("criteriaDocumented", "exclusionsTriggered", "questionFindings"):
        if not isinstance(detail.get(key), list):
            detail[key] = []

    question = obj["clarifyingQuestion"]
    if isinstance(question, str) and not question.strip():
        question = None

    return {
        "clinicalSummary": str(obj["clinicalSummary"] or "").strip() or None,
        "meetsMedicalNecessity": obj["meetsMedicalNecessity"],
        "clarifyingQuestion": question,
        "detail": detail,
    }, ""


def evaluate_trip(trip: Dict[str, Any],
                  clarification: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Returns the same shape a fixture block returns."""
    pairs = extract_pairs(trip)
    if not pairs:
        return {
            "clinicalSummary": None,
            "meetsMedicalNecessity": False,
            "clarifyingQuestion": None,
            "detail": {
                "determination": "does_not_meet",
                "basis": "The order carried no clinical answers to assess.",
                "confidence": "High",
                "criteriaDocumented": [],
                "exclusionsTriggered": [],
                "questionFindings": [],
            },
        }

    client = get_client()
    user_message = build_user_message(trip, pairs, clarification)

    message = client.messages.create(
        model=os.environ["CLAUDE_DEPLOYMENT_NAME"],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=build_prompt(),
        messages=[{"role": "user", "content": user_message}],
    )
    raw = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")

    result, error = validate(raw)
    if result is None:
        retry = client.messages.create(
            model=os.environ["CLAUDE_DEPLOYMENT_NAME"],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=build_prompt(),
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": raw or ""},
                {"role": "user",
                 "content": f"That response was rejected ({error}). "
                            f"Return only valid JSON matching the schema."},
            ],
        )
        raw = "".join(b.text for b in retry.content if getattr(b, "type", "") == "text")
        result, error = validate(raw)
        message = retry

    if result is None:
        raise RuntimeError(f"model response failed validation twice: {error}")

    result["detail"]["usage"] = {
        "inputTokens": message.usage.input_tokens,
        "outputTokens": message.usage.output_tokens,
        "model": message.model,
    }
    return result
