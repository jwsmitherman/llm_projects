"""Builds the review workbook from a completed batch run.

The sheets exist because of what was asked for in the 29 October review:

  "is there a way to take this blob and turn it into structured data so that we
   can easily see what the input was? Because you can't at scale look at this."

  "I want to look at this for 1000 calls, but in a spreadsheet or somewhere we
   can quickly go through a whole bunch together over an hour or two."

  "here's the question, here's the answer, whether it met the criteria, the
   evidence, the reason, and then the confidence level."

The 6 November review changed the shape of it:

  Dave: "I think these are independent. I think we should score on each category
   for each trip, because one trip could be that they could have an altered LOC,
   they could be unconscious, and they could also have a mobility restriction."

The categories are the six from the in-person meeting doc: mobility, level of
consciousness, medical monitoring, level of service, modifiers, and isolation
precautions. Four are read from the words on the order. Level of service and
modifiers are read off the payload, because the order already states them, and
the modifier scale comes from prompts/med_nec_knowledge.json rather than being
restated here.

  Dave: "If there's a way to pull all this together into a way that we can just
   look it up really succinctly for each one, and then look at 51 after the
   other ... we need it in a way that we can hit a whole bunch of these one
   after the other without jumping around."

  Mukund: "if possible, break down the EPCR data into the columns so that we
   have that ready to view, not just as a blob."

So this writes ONE sheet, one row per trip, and only the columns a reviewer
uses: the six category scores for the TRIP, the order verdict, the ePCR verdict
and the agreement between them, two empty columns for the reviewer's own call,
and the question by question working consolidated into cells rather than spread
across nine columns per question.

Mukund asked for two more things on 10 November, and both are here:

  "the question and answer is not complete because we are not giving all the
   information in the form"            -> the Form as entered column, verbatim

  "you can put this into another column where you can call it as an LLM
   assessment ... that's why I've shaded this separately, so you can group them
   like that"      -> the model's words are their own columns, under a banner

Not changed, on purpose. Dave on the clarifying-question wording: "I'm not
ready to change anything. We should review a whole bunch of these at scale, and
then figure out if a pattern emerges." The prompts are untouched here.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill

# Origin and destination codes read as words, so a reviewer is not decoding
# letters. These are the CMS modifiers.
MODIFIERS = {
    "H": "Hospital", "S": "Skilled nursing facility", "P": "Physician office",
    "R": "Residence", "E": "Residential/domiciliary/custodial facility",
    "N": "Skilled nursing facility", "D": "Diagnostic or therapeutic site",
    "J": "Freestanding ESRD facility", "G": "Hospital-based ESRD facility",
    "X": "Intermediate stop", "I": "Site of transfer",
}

# The six categories from the in-person meeting doc (Categories_in_person_meeting,
# 23 October). Four are decided from the words on the order; the other two are
# read off the payload itself, because the order already carries them as fields:
# level of service is stated on the leg, and the modifiers are the origin and
# destination codes.
#
# The word-matched four. Matched on word boundaries against text normalised to
# plain words, so "altered_mental" (a criterion id) and "Altered mental status?"
# (the question a nurse sees) both hit the same entry. Writing these with
# underscores and matching them raw was the earlier defect: every finding whose
# only clue was in prose fell through to the catch-all.
CATEGORIES = {
    "Mobility": ("bed confined", "bed confinement", "bedbound", "bed bound",
                 "cannot sit", "sit safely", "mobility", "ambulate", "ambulatory",
                 "non ambulatory", "stretcher", "gurney", "contraindicated other means",
                 "weight bearing", "transfer assist", "person assist", "assist",
                 "contracture", "positioning", "wheelchair", "hoyer", "lift",
                 "bariatric", "fall risk", "immobile"),
    "Level of consciousness": (
                  "altered mental", "altered loc", "loc", "mental status", "sedation",
                  "sedated", "confusion", "confused", "unresponsive", "unconscious",
                  "behavioral", "psychiatric", "psych", "consciousness", "gcs",
                  "mentation", "oriented", "orientation", "a o", "dementia",
                  "delirium", "agitated", "agitation", "combative", "sitter",
                  "edo", "emergency detention"),
    "Medical monitoring": (
                   "cardiac", "iv", "intravenous", "oxygen", "o2", "airway",
                   "suction", "ventilator", "vent", "trach", "bipap", "cpap",
                   "medication", "medications", "vitals", "vital signs", "monitor",
                   "monitoring", "monitored", "infusion", "drip", "pump",
                   "restraint", "restraints", "telemetry", "ecg", "ekg", "spo2",
                   "saturation", "titrate", "wound", "drain"),
    "Isolation precautions": (
                   "isolation", "precautions", "contact precautions",
                   "droplet", "airborne", "quarantine", "ppe", "mrsa", "vre",
                   "c diff", "cdiff", "clostridium", "tuberculosis", "tb",
                   "covid", "infectious", "communicable"),
}

CATEGORY_PATTERNS = {
    name: [re.compile(r"\b" + re.escape(term) + r"\b") for term in terms]
    for name, terms in CATEGORIES.items()
}

# The two read from the payload rather than from the words.
DATA_CATEGORIES = ["Level of service", "Modifiers"]

CATEGORY_ORDER = list(CATEGORIES) + DATA_CATEGORIES


def load_modifier_scale() -> List[str]:
    """The origin/destination codes, lowest to highest, from the knowledge file.

    Read rather than restated, so the scale the workbook scores against is the
    same one the model is briefed on. If the file moves, the fallback is that
    same list as of the 23 October knowledge build.
    """
    path = Path(__file__).resolve().parent.parent / "prompts" / "med_nec_knowledge.json"
    try:
        trees = json.loads(path.read_text()).get("decision_trees") or {}
        scale = [entry["code"] for entry
                 in trees.get("modifier_scale_lowest_to_highest") or []]
        if scale:
            return scale
    except Exception:
        pass
    return ["S", "R", "E", "P", "N", "D", "J", "G", "X", "H"]


MODIFIER_SCALE = load_modifier_scale()


def score_level_of_service(los: Any) -> Dict[str, Any]:
    """Level of service as its own category, read off the leg.

    ALS and SCT are ordered when the facility expects advanced intervention en
    route, which is the thing this category is meant to capture. BLS is scored 1
    rather than 0 because the level was stated, it just carries less on its own.
    """
    text = _words(los).strip()
    if not text:
        return {"score": 0, "label": "not stated on the order", "confidence": None,
                "evidence": "", "criteria": ""}
    if "sct" in text or "specialty" in text:
        return {"score": 2, "label": "SCT ordered", "confidence": "High",
                "evidence": str(los), "criteria": "level_of_service"}
    if "als" in text:
        return {"score": 2, "label": "ALS ordered", "confidence": "High",
                "evidence": str(los), "criteria": "level_of_service"}
    if "bls" in text:
        return {"score": 1, "label": "BLS ordered", "confidence": "High",
                "evidence": str(los), "criteria": "level_of_service"}
    return {"score": 1, "label": f"stated as {los}", "confidence": "Medium",
            "evidence": str(los), "criteria": "level_of_service"}


def score_modifiers(pickup: Any, dropoff: Any) -> Dict[str, Any]:
    """Origin and destination as their own category.

    Scored by the rule already in prompts/med_nec_knowledge.json: "Use the
    higher level of care tree when the origin modifier is lower on the scale
    than the destination ... the lower level of care tree when the origin is
    higher than or similar to the destination, such as discharges to a residence
    or nursing home."

    So an upward transport scores 2: it is the case the higher-level tree
    governs. A discharge direction scores 1, meaning the pair alone does not
    carry the trip. Codes missing scores 0, which is a data problem to see.
    """
    origin = str(pickup or "").strip().upper()
    destination = str(dropoff or "").strip().upper()
    words = f"{MODIFIERS.get(origin, origin or '?')} to {MODIFIERS.get(destination, destination or '?')}"

    if not origin or not destination:
        return {"score": 0, "label": "origin or destination code missing",
                "confidence": None, "evidence": words, "criteria": ""}
    if origin not in MODIFIER_SCALE or destination not in MODIFIER_SCALE:
        return {"score": 1, "label": f"{origin} to {destination}, not on the scale",
                "confidence": "Low", "evidence": words, "criteria": "modifiers"}

    here, there = MODIFIER_SCALE.index(origin), MODIFIER_SCALE.index(destination)
    if there > here:
        label = f"{origin} to {destination}, upward, higher level of care tree"
        score, confidence = 2, "High"
    elif there == here:
        label = f"{origin} to {destination}, same level"
        score, confidence = 1, "Medium"
    else:
        label = f"{origin} to {destination}, discharge direction, lower level of care tree"
        score, confidence = 1, "Medium"
    return {"score": score, "label": label, "confidence": confidence,
            "evidence": words, "criteria": "modifiers"}


def _words(*parts: Any) -> str:
    """Lower case, punctuation to spaces. "A&O x1-2" -> " a o x1 2 "."""
    return " " + re.sub(r"[^a-z0-9]+", " ", " ".join(
        str(p or "") for p in parts).lower()).strip() + " "


def categories_for(criterion_id, question, answer, evidence=None,
                   reason=None) -> List[str]:
    """Every category a finding touches, not just the first one matched.

    Dave, 6 November: "these are independent ... one trip could be that they
    could have an altered LOC, they could be unconscious, and they could also
    have a mobility restriction."

    A finding naming both a stretcher and cardiac monitoring belongs in mobility
    AND monitoring. Returning one primary category lost the second, which is
    what the review found. The model's quoted evidence counts too, because that
    is where the second category usually shows up: the question asks about
    special handling, the quote is what the handling was.
    """
    haystack = _words(criterion_id, question, answer, evidence, reason)
    return [name for name, patterns in CATEGORY_PATTERNS.items()
            if any(p.search(haystack) for p in patterns)]


_NEGATIVE = re.compile(
    r"^(no|none|n a|not applicable|negative|denies|unknown|nil|false|0)\b")


def is_negative(answer: Any) -> bool:
    """Whether an answer says the thing was not the case.

    Deliberately narrow: it reads the start of the answer only, so "No" and
    "No (Discharge, Behavioral Transfer or Other Reason)" both count, while an
    answer that happens to contain the word "no" further in does not.
    """
    text = _words(answer).strip()
    return bool(text) and bool(_NEGATIVE.match(text))


def score_category(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll a category's findings up into one score.

    Derived here from what the model already returned per question. The model is
    not asked to score categories, so nothing about the determination changes.

    2  supports    a criterion in this category is documented and cited
    1  insufficient the category is addressed, but not enough on its own
    0  not documented, or asked and answered no

    A category is matched on the words in the question, so a question that names
    the category and is answered "No" lands here too: the EDO question on every
    Texas order mentions detention whatever the answer, and an answer listing
    "Behavioral Transfer" among its options mentions behaviour even when that
    option was not the one chosen. Scoring those 1 would read as "addressed but
    short", which is wrong. Asked and answered no is 0.
    """
    if not findings:
        return {"score": 0, "label": "not documented", "confidence": None,
                "evidence": "", "criteria": ""}

    supporting = [f for f in findings if f.get("nonMedNec") is False
                  and (f.get("criterionId") or f.get("evidence"))]
    if supporting:
        best = supporting
        score, label = 2, "supports"
    elif all(is_negative(f.get("answer")) for f in findings):
        best = findings
        score, label = 0, "asked, answered no"
    else:
        best = findings
        score, label = 1, "insufficient"

    rank = {"High": 3, "Medium": 2, "Low": 1}
    confidence = max((f.get("confidence") for f in best if f.get("confidence")),
                     key=lambda c: rank.get(c, 0), default=None)

    return {
        "score": score,
        "label": label,
        "confidence": confidence,
        "evidence": " | ".join(str(f.get("evidence")) for f in best
                               if f.get("evidence"))[:600],
        "criteria": ", ".join(sorted({str(f.get("criterionId")) for f in best
                                      if f.get("criterionId")})),
    }


def question_of(pair: Dict[str, Any]) -> Any:
    """The question text, whichever key the pair came in under.

    The parser emits "question"; the structured losQuestions path emits
    "questionText". Reading only one of them left every Q column blank for
    production orders, which is the other half of the "question = None" Dave
    saw in the workbook: the first half was the parser, this was the reader.
    """
    return pair.get("question") or pair.get("questionText")


def duplicate_flag(r: Dict[str, Any]) -> str:
    """Whether this trip was submitted more than once, and whether it mattered.

    Josh found these by hand: two payloads under one trip id. Only the latest is
    scored, so the distinction a reviewer needs is whether the earlier
    submission said something different.
    """
    count = r.get("payloadCount") or 1
    if count <= 1:
        return ""
    variants = r.get("payloadVariants") or 1
    if variants > 1:
        return f"YES, {count} submissions, {variants} differ"
    return f"resubmitted {count}x, identical"


# Columns the ePCR extract may carry that no reviewer needs and that would put
# identifiers back into a workbook people email around. The extract is scrubbed
# for identifier patterns inside free text, not column by column, so this is the
# column-level half of that. Substring match, case-insensitive, deny by name.
RECORD_WITHHOLD = (
    "name", "dob", "birth", "ssn", "mrn", "address", "street", "city", "zip",
    "postal", "phone", "email", "insur", "payer", "payor", "payment", "policy",
    "member", "subscriber", "account", "guarantor", "medicaid", "medicare",
    "licen", "crew", "employee", "unit_number", "vehicle",
)

# Shown first, in this order, because this is what decides the verdict. Anything
# else the extract carries follows, so the tab is the whole record rather than a
# curated slice of it.
RECORD_LEAD = [
    "EPCR_Level_Of_Care_Provided", "EPCR_Transport_Method",
    "EPCR_Primary_Impression", "EPCR_Secondary_Impressions",
    "EPCR_Chief_Complaint", "EPCR_Working_Diagnosis",
    "EPCR_Procedures_Performed", "EPCR_Medications_Given",
    "EPCR_Initial_SBP", "EPCR_Initial_DBP", "EPCR_Initial_Heart_Rate",
    "EPCR_Initial_Respiratory_Rate", "EPCR_Initial_SpO2", "EPCR_Initial_GCS",
    "EPCR_Patient_Evaluation_Care", "EPCR_Medical_Surgical_History",
    "EPCR_Current_Medications", "EPCR_Reason_For_Destination",
    "EPCR_Narrative",
]


def record_columns(seen: Dict[str, Any]) -> Dict[str, Any]:
    """The ePCR row broken into its own columns, lead fields first.

    Mukund, 6 November: "if possible, break down the EPCR data into the columns
    so that we have that ready to view, not just as a blob."
    """
    ACRONYMS = {"sbp": "SBP", "dbp": "DBP", "hr": "HR", "spo2": "SpO2",
                "gcs": "GCS", "bp": "BP", "ekg": "ECG", "ecg": "ECG",
                "bls": "BLS", "als": "ALS", "iv": "IV", "los": "LOS",
                "epcr": "ePCR", "id": "id"}

    def label(field: str) -> str:
        words = field.replace("EPCR_", "").replace("TNET_", "").split("_")
        out = [ACRONYMS.get(w.lower(), w.lower()) for w in words if w]
        return " ".join(out)[:1].upper() + " ".join(out)[1:]

    out: Dict[str, Any] = {}
    for field in RECORD_LEAD:
        if field in seen:
            out[label(field)] = seen.get(field)
    # The trip id is already the first column of every tab.
    rest = [f for f in seen
            if f not in RECORD_LEAD
            and "triprequestid" not in f.lower()
            and not any(bad in f.lower() for bad in RECORD_WITHHOLD)]
    for field in sorted(rest):
        out.setdefault(label(field), seen.get(field))
    return out


def agreement(order_meets, record_meets) -> str:
    if order_meets is None or record_meets is None:
        return "no record verdict"
    if order_meets == record_meets:
        return "agree"
    return "order meets, record does not" if order_meets else "record meets, order does not"


def yes_no(value: Any) -> Any:
    """True/False as words. Excel renders a bare bool as 0 or 1 in a mixed
    column, and "0" in a column headed "meets" is a verdict nobody should have
    to decode."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


SHEET = "Med nec review"

# The six scores, in the order of the in-person meeting doc. Short headers: the
# column is a number, the detail lives in the cells beside it.
# The categories that measure what the facility DOCUMENTED. These three are the
# score: each is 0 unless the order says something, so the score has a real zero.
DOC_COLUMNS = [
    ("Score: mobility", "Mobility"),
    ("Score: LOC", "Level of consciousness"),
    ("Score: monitoring", "Medical monitoring"),
]

# Reported beside the score, NOT added into it. Dave, 13 November, scanning the
# sheet: "I don't see anything higher than a two in the individual colored ones
# ... and the total score does go up to eight. I'm just guessing you're adding
# the scores."
#
# He was, and that was the defect. Level of service and the two modifiers are
# read off the payload, and a production order always carries them: level of
# service is never blank, origin and destination codes are never missing. So
# they scored 1 or 2 every time and put a floor of 2 under every total. A trip
# with no clinical documentation at all scored 2 or 3 out of 12, which reads as
# partly documented when it is not documented at all.
#
# They still belong on the sheet. They describe the trip, not the documentation,
# so they are their own columns and out of the score.
ATTRIBUTE_COLUMNS = [
    ("Score: LOS", "Level of service"),
    ("Score: modifiers", "Modifiers"),
]

# Isolation is carried, and is not in the score, because no question on the
# order form asks about isolation precautions. It scored 0 on every trip of the
# 25 October run. A column that can only ever be 0 drags every total down and
# tells a reviewer nothing. It starts counting the day a form question feeds it.
ISOLATION_COLUMN = ("Score: isolation", "Isolation precautions")

SCORE_COLUMNS = DOC_COLUMNS + [ISOLATION_COLUMN] + ATTRIBUTE_COLUMNS

SCORE_LEGEND = ("documentation score 0-6, from mobility, LOC and monitoring   |   "
                "each category: 2 supports, 1 insufficient alone, 0 not documented   |   "
                "LOS, modifiers and isolation are reported but NOT in the score")

# Column groups, banner over them in row 1. Mukund, 10 November: "you are also
# giving LLM assessment here ... and order summary is also an LLM assessment.
# Order clarifying question, which is an LLM assessment again. So that's why
# I've shaded this separately, so you can group them like that."
#
# The point of the grouping is that a reviewer can tell at a glance what the
# nurse entered from what the model concluded. The row is built in this order,
# and the banner is drawn from this list, so the two cannot drift apart.
GROUP_ORDER = [
    ("FROM THE ORDER, as entered", [
        "Trip request id", "Ordered", "LOS", "Trip type", "Duplicate payloads",
        "Questions and answers", "Other form info", "Form chars"]),
    ("LLM ASSESSMENT OF THE ORDER", [
        *[header for header, _ in SCORE_COLUMNS], "Score: documentation",
        "LLM: meets", "LLM: confidence", "LLM: summary",
        "LLM: clarifying question", "LLM: assessment per question",
        "AUDIT: summary not in order", "AUDIT: quotes not in order"]),
    ("ePCR, as documented", [
        "ePCR: level of care provided", "ePCR: primary impression",
        "ePCR: narrative", "ePCR: other fields"]),
    ("LLM ASSESSMENT OF THE ePCR", [
        "LLM ePCR: meets", "LLM ePCR: summary", "LLM ePCR: documentation gaps",
        "Agreement"]),
    ("REVIEWER", ["SME: meets", "SME: notes", "Flags"]),
]

COLUMN_ORDER = [header for _, headers in GROUP_ORDER for header in headers]

# Which groups hold the model's own words rather than the order's.
LLM_GROUPS = {"LLM ASSESSMENT OF THE ORDER", "LLM ASSESSMENT OF THE ePCR"}


def form_cell(scrubbed: Dict[str, Any]) -> str:
    """Everything the form holds, verbatim, not only the pairs we could parse.

    Mukund, 10 November: "shouldn't they have clicked on some radio buttons and
    all of that? ... the question and answer is not complete because we are not
    giving all the information in the form."

    He is right, and the parser is the reason: it recovers question and answer
    pairs, so anything it cannot pair up, including the option lists a nurse
    chose from, never reached the sheet. This prints the de-identified form text
    as it came, so what is in the cell is what is in the payload. The parsed
    pairs stay in their own column for scanning; this is the source for them.
    """
    if not scrubbed:
        return ""
    legs = scrubbed.get("tripLegs") or []
    parts = []
    for n, leg in enumerate(legs, 1):
        chunks = []
        text = str(leg.get("clinicalData") or "").strip()
        if text:
            chunks.append(text)

        # The rest of the form: the drop-downs, toggles and notes that sit
        # outside the question and answer pairs. Same fields the model is now
        # sent, so the cell and the prompt hold the same thing.
        extra = []
        for entry in (leg.get("additionalFields") or []):
            if not isinstance(entry, dict):
                continue
            shown = entry.get("selectedLabel")
            if shown in (None, "", []):
                shown = entry.get("value")
            if shown in (None, "", []):
                continue
            if isinstance(shown, bool):
                shown = "yes" if shown else "no"
            extra.append(f"{entry.get('label') or entry.get('name')}: {shown}")
        if extra:
            chunks.append("Other form fields:\n    " + "\n    ".join(extra))
        # Some orders carry the form as structured questions instead. Print
        # those in full too, options and all, not just the answered ones.
        for q in (leg.get("losQuestions") or []):
            if not isinstance(q, dict):
                continue
            line = str(q.get("questionText") or q.get("question") or "").strip()
            options = q.get("options") or q.get("answerOptions") or []
            if isinstance(options, list) and options:
                shown = [str(o.get("text") if isinstance(o, dict) else o)
                         for o in options]
                line += "\n    options: " + " / ".join(s for s in shown if s)
            answer = str(q.get("answer") or "").strip()
            line += f"\n    answered: {answer}" if answer else "\n    answered: (blank)"
            chunks.append(line)
        if chunks:
            header = f"--- leg {n} ---\n" if len(legs) > 1 else ""
            parts.append(header + "\n\n".join(chunks))
    return "\n\n".join(parts)


def other_form_info(r: Dict[str, Any]) -> str:
    """The form minus the question and answer pairs. No duplication.

    Josh and Mukund, 10 November, looking at Form as entered next to Questions
    and answers: "the same information is here ... we can delete that column."
    They were right about that payload, where the parser recovered the whole
    form, so the two cells read almost identically.

    Deleting the column outright would have cost the other half of it: the order
    details and the drop-downs that are NOT in any question and answer pair, and
    which the previous call asked to have in the sheet. So this is the column
    with the duplication taken out, not the column removed. Everything the model
    was sent, less what the Q&A cell already shows.
    """
    scrubbed = r.get("scrubbed") or {}
    legs = scrubbed.get("tripLegs") or []
    leg = legs[0] if legs else {}
    patient = leg.get("patientInfo") or {}
    lines = []

    for key, label in (("diagnosis", "Diagnosis"),
                       ("patientWeight", "Patient weight"),
                       ("patientHeight", "Patient height"),
                       ("overMaxWeight", "Over max weight")):
        value = patient.get(key)
        if value not in (None, "", []):
            lines.append(f"{label}: {value}")

    for key, label in (("medicareQuestionNums", "Medicare questions answered"),
                       ("medicareQuestionTen", "Medicare question 10"),
                       ("losReason", "Level of service reason"),
                       ("losOverrideReason", "Override reason"),
                       ("specialNeeds", "Special needs"),
                       ("transRequestComment", "Requester comment"),
                       ("facilityPayReason", "Facility pay reason")):
        value = leg.get(key)
        if value not in (None, "", []):
            lines.append(f"{label}: {value if not isinstance(value, list) else ', '.join(map(str, value))}")

    for entry in (leg.get("additionalFields") or []):
        if not isinstance(entry, dict):
            continue
        shown = entry.get("selectedLabel")
        if shown in (None, "", []):
            shown = entry.get("value")
        if shown in (None, "", []):
            continue
        if isinstance(shown, bool):
            shown = "yes" if shown else "no"
        lines.append(f"{entry.get('label') or entry.get('name')}: {shown}")

    return "\n".join(lines)


def qa_cell(r: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
    """The questions and answers alone. No assessment in here.

    Mukund: "I like your idea of separating out. So you can say that hey, you
    know, it supports me, and you can put this into another column where you can
    call it as an LLM assessment."
    """
    blocks, seen = [], set()
    for f in findings:
        question = str(f.get("question") or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        blocks.append(f"Q{len(blocks) + 1}. {question}\n    A: {f.get('answer') or ''}")
    for pair in (r.get("pairs") or []):
        question = str(question_of(pair) or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        blocks.append(f"Q{len(blocks) + 1}. {question}\n    A: {pair.get('answer') or ''}")
    return "\n\n".join(blocks)


def assessment_cell(r: Dict[str, Any], findings: List[Dict[str, Any]],
                    finding_categories: Dict[int, List[str]]) -> str:
    """What the model made of each answer, numbered to match the Q&A cell."""
    order = []
    seen = set()
    for f in findings:
        question = str(f.get("question") or "").strip()
        if question and question not in seen:
            seen.add(question)
            order.append(question)
    for pair in (r.get("pairs") or []):
        question = str(question_of(pair) or "").strip()
        if question and question not in seen:
            seen.add(question)
            order.append(question)
    position = {question: n + 1 for n, question in enumerate(order)}

    blocks = []
    for n, f in enumerate(findings):
        question = str(f.get("question") or "").strip()
        verdict = ("supports necessity" if f.get("nonMedNec") is False
                   else "does not support")
        detail = [", ".join(finding_categories.get(n) or []) or "no category"]
        if f.get("confidence"):
            detail.append(f"{f['confidence']} confidence")
        if f.get("criterionId"):
            detail.append(str(f["criterionId"]))
        lines = [f"Q{position.get(question, n + 1)}: {verdict}  [{'; '.join(detail)}]"]
        if f.get("reason"):
            lines.append(f"    why: {f['reason']}")
        if f.get("evidence"):
            lines.append(f"    quoted: {f['evidence']}")
        if f.get("citation"):
            lines.append(f"    cite: {f['citation']}")
        blocks.append("\n".join(lines))

    answered = {str(question_of(p) or "").strip() for p in (r.get("pairs") or [])}
    for question in answered - {str(f.get("question") or "").strip() for f in findings}:
        if question:
            blocks.append(f"Q{position.get(question, '?')}: no finding returned "
                          "for this answer")
    return "\n\n".join(blocks)



# Claims that either happened or did not. A summary asserting one of these when
# nothing in the order mentions it is a fabrication, not a paraphrase, and this
# is the class of error Mukund found on 10 November:
#
#   "it says that the patient is being transported after a suicide attempt,
#    which means there was an attempt that was made, but the attempt is not
#    documented here ... ideation is not an attempt. So I'm just thinking, how
#    is it saying it's an attempt?"
#
# Descriptive paraphrase is left alone on purpose. A summary saying "altered
# mental status" where the order says "A&O x1-2" is doing its job. These are
# events and interventions, where there is no fair paraphrase.
CLAIM_TERMS = {
    "suicide attempt": ("suicide attempt", "attempted suicide", "self harm attempt",
                        "overdose", "od attempt"),
    "attempted suicide": ("suicide attempt", "attempted suicide"),
    "overdose": ("overdose", "od", "ingestion"),
    "cardiac arrest": ("cardiac arrest", "code blue", "cpr", "rosc", "pulseless"),
    "respiratory arrest": ("respiratory arrest", "apneic", "code blue", "cpr"),
    "intubated": ("intubated", "intubation", "ett", "endotracheal", "ventilator", "vent"),
    "ventilator": ("ventilator", "vent", "intubated", "trach"),
    "unresponsive": ("unresponsive", "unconscious", "gcs 3", "comatose", "obtunded"),
    "comatose": ("comatose", "coma", "unresponsive", "gcs 3"),
    "fracture": ("fracture", "fractured", "fx", "broken"),
    "fractured": ("fracture", "fractured", "fx", "broken"),
    "stroke": ("stroke", "cva", "tia", "cerebrovascular"),
    "seizure": ("seizure", "seizing", "status epilepticus", "postictal"),
    "sepsis": ("sepsis", "septic", "bacteremia"),
    "hemorrhage": ("hemorrhage", "hemorrhaging", "bleed", "bleeding", "gi bleed"),
    "aspiration": ("aspiration", "aspirated"),
    "myocardial infarction": ("myocardial infarction", "mi", "stemi", "nstemi",
                              "heart attack"),
    "heart attack": ("myocardial infarction", "mi", "stemi", "nstemi", "heart attack"),
    "pulmonary embolism": ("pulmonary embolism", "pe"),
    "assault": ("assault", "assaulted", "struck", "battery"),
    "homicidal": ("homicidal", "harm others", "danger to others"),
    "psychosis": ("psychosis", "psychotic", "hallucination", "delusion"),
    "restrained": ("restrained", "restraint", "restraints"),
    "amputation": ("amputation", "amputated"),
}

# Where a weaker term in the order is the likely source of a stronger claim in
# the summary. Named in the flag so a reviewer sees the escalation, not just the
# absence: "attempt (order says ideation)" is the whole finding in four words.
ESCALATED_FROM = {
    "suicide attempt": ("suicidal ideation", "ideation", "si"),
    "attempted suicide": ("suicidal ideation", "ideation", "si"),
    "cardiac arrest": ("chest pain", "cardiac history", "bradycardia", "tachycardia"),
    "respiratory arrest": ("shortness of breath", "sob", "respiratory distress", "oxygen"),
    "intubated": ("oxygen", "bipap", "cpap", "airway"),
    "unresponsive": ("altered", "lethargic", "a o", "confused"),
    "fracture": ("fall", "fell", "pain", "injury"),
    "stroke": ("weakness", "altered", "slurred"),
    "sepsis": ("infection", "fever", "uti"),
    "hemorrhage": ("anemia", "hemoglobin", "blood"),
}


def source_text(r: Dict[str, Any]) -> str:
    """Everything the model was given for this trip, as one normalised string.

    Built from the de-identified payload, so it is exactly what the model read,
    and from the parsed pairs, which is how it was framed. If a claim is not in
    here, the model did not get it from the order.
    """
    def walk(node: Any, out: List[str]) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value, out)
        elif isinstance(node, list):
            for value in node:
                walk(value, out)
        elif isinstance(node, (str, int, float)):
            out.append(str(node))

    collected: List[str] = []
    walk(r.get("scrubbed") or {}, collected)
    for pair in (r.get("pairs") or []):
        collected.append(str(question_of(pair) or ""))
        collected.append(str(pair.get("answer") or ""))
    return _words(" ".join(collected))


def audit_quotes(findings: List[Dict[str, Any]], haystack: str) -> str:
    """Evidence the model quoted that is not in what it was given.

    The prompt requires evidence verbatim, so a quote that is not in the source
    is checkable mechanically, with no judgement and no heuristics.
    """
    missing = []
    for f in findings:
        quote = str(f.get("evidence") or "").strip()
        if not quote:
            continue
        needle = _words(quote).strip()
        if needle and needle not in haystack:
            missing.append(quote[:70])
    return " | ".join(missing)


def audit_summary(summary: Any, haystack: str) -> str:
    """Clinical events asserted in the summary that the order never mentions."""
    text = _words(summary)
    flags = []
    for claim, synonyms in CLAIM_TERMS.items():
        if f" {claim} " not in text:
            continue
        if any(f" {syn} " in haystack for syn in synonyms):
            continue
        weaker = [w for w in ESCALATED_FROM.get(claim, ()) if f" {w} " in haystack]
        flags.append(f"{claim} (order says {weaker[0]})" if weaker
                     else f"{claim} (not in the order)")
    return " | ".join(sorted(set(flags)))


def record_cell(seen: Dict[str, Any], skip: tuple) -> str:
    """The rest of the ePCR as one cell, a field per line."""
    kept = record_columns(seen)
    return "\n".join(f"{name}: {value}" for name, value in kept.items()
                     if str(value or "").strip() and name not in skip)


def flags_cell(r: Dict[str, Any]) -> str:
    """Anything a reviewer should know about the run of this trip."""
    rep = r.get("deid") or {}
    notes = []
    if r.get("leak"):
        notes.append("HELD BACK: an identifier survived scrubbing, nothing was sent")
    error = r.get("error") or (r.get("order") or {}).get("error")
    if error:
        notes.append(f"error: {error}")
    unknown = (rep.get("unknownTop") or []) + (rep.get("unknownLeg") or [])
    if unknown:
        notes.append("unknown fields dropped: " + ", ".join(unknown))
    if not r.get("record"):
        notes.append("no patient care record matched this trip")
    return " | ".join(notes)


def build(results: List[Dict[str, Any]], out_path: Path, scored: bool) -> Path:
    """One sheet, one row per trip, grouped so the order and the model are apart.

    Dave, 6 November: "get this all in one spreadsheet because it's kind of a
    pain in the butt to go back and forth."

    The scores are the trip's, not the questions'. Each of the six categories is
    rolled up to one number for the trip. The form, the parsed questions and the
    model's assessment of them are three separate cells, under banners that say
    which is which.
    """
    rows: List[Dict[str, Any]] = []

    for r in results:
        order = (r.get("order") or {}).get("body") or {}
        record = (r.get("record") or {}).get("body") or {}
        od = order.get("detail") or {}
        rd = record.get("detail") or {}
        findings = od.get("questionFindings") or []

        order_meets = order.get("meetsMedicalNecessity")
        record_meets = record.get("meetsMedicalNecessity") if r.get("record") else None

        by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CATEGORIES}
        finding_categories: Dict[int, List[str]] = {}
        for n, f in enumerate(findings):
            hits = categories_for(f.get("criterionId"), f.get("question"),
                                  f.get("answer"), f.get("evidence"), f.get("reason"))
            finding_categories[n] = hits
            for name in hits:
                by_category[name].append(f)

        scores = {c: score_category(by_category[c]) for c in CATEGORIES}
        # These two are not guessed from wording. The order states them.
        scores["Level of service"] = score_level_of_service(r.get("levelOfService"))
        scores["Modifiers"] = score_modifiers(r.get("pickupCode"), r.get("dropoffCode"))

        modifier_note = scores["Modifiers"]["label"]
        direction = (modifier_note.split(", ", 1)[1] if ", " in modifier_note else "")

        seen = r.get("recordFields") or {}
        lead = ("Level of care provided", "Primary impression", "Narrative")
        haystack = source_text(r)

        row = {
            "Trip request id": r["tripRequestId"],
            "Ordered": r.get("orderedAt"),
            "LOS": r.get("levelOfService"),
            "Trip type": (f"{r.get('pickupCode') or '?'} to {r.get('dropoffCode') or '?'}"
                          + (f" ({direction})" if direction else "")),
            "Duplicate payloads": duplicate_flag(r),
            "Questions and answers": qa_cell(r, findings),
            "Other form info": other_form_info(r),
            # Mukund found payloads with almost nothing in them. This is how a
            # thin one is spotted: sort on it and the blanks come to the top.
            "Form chars": len(form_cell(r.get("scrubbed") or {})),
        }
        # A failed call is not a badly documented trip. When the model never
        # answered, the score cells are left blank rather than written as 0, so
        # the row cannot be read as a weak order. The reason is in Flags.
        failed = bool(r.get("error") or (r.get("order") or {}).get("error")
                      or not findings)
        for header, category in SCORE_COLUMNS:
            row[header] = None if failed else scores[category]["score"]
        row["Score: documentation"] = (
            None if failed else sum(scores[c]["score"] for _, c in DOC_COLUMNS))

        row.update({
            "LLM: meets": yes_no(order_meets),
            "LLM: confidence": od.get("confidence"),
            "LLM: summary": order.get("clinicalSummary"),
            "LLM: clarifying question": order.get("clarifyingQuestion"),
            "LLM: assessment per question": assessment_cell(
                r, findings, finding_categories),
            # The check Mukund asked for on 10 November: "let's kind of complete
            # this check that the question and answers align with the LLM
            # summaries, so that you know we know."
            "AUDIT: summary not in order": audit_summary(
                order.get("clinicalSummary"), haystack),
            "AUDIT: quotes not in order": audit_quotes(findings, haystack),
            "ePCR: level of care provided": seen.get("EPCR_Level_Of_Care_Provided"),
            "ePCR: primary impression": seen.get("EPCR_Primary_Impression"),
            "ePCR: narrative": seen.get("EPCR_Narrative"),
            "ePCR: other fields": record_cell(seen, lead) if r.get("record") else "",
            "LLM ePCR: meets": yes_no(record_meets),
            "LLM ePCR: summary": record.get("clinicalSummary"),
            "LLM ePCR: documentation gaps": " | ".join(
                record.get("documentationGaps") or []),
            "Agreement": agreement(order_meets, record_meets),
            # Empty on purpose: the reviewer's verdict goes beside the model's.
            "SME: meets": "",
            "SME: notes": "",
            "Flags": flags_cell(r),
        })
        rows.append(row)

    frame = pd.DataFrame(rows, columns=COLUMN_ORDER)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=SHEET, index=False)
        format_sheet(writer.sheets[SHEET])

    return out_path


# Columns that hold a paragraph or more. Capped narrow and left unwrapped: the
# row stays one line tall and the cell is read in the formula bar, or by
# widening that one column.
# Eight lines of wrapped text. Enough to read a summary or the first few
# question and answer blocks without expanding the row.
ROW_HEIGHT = 112

LONG_COLUMNS = ("Questions and answers", "Other form info",
                "LLM: summary", "LLM: clarifying question",
                "LLM: assessment per question", "AUDIT: summary not in order",
                "AUDIT: quotes not in order", "ePCR: primary impression",
                "ePCR: narrative", "ePCR: other fields", "LLM ePCR: summary",
                "LLM ePCR: documentation gaps", "Flags", "SME: notes")


def format_sheet(sheet) -> None:
    """Group banner in row 1, headers in row 2, a trip per row after that.

    Mukund, 10 November: "if you highlight all this and give a fixed distance
    here, so that it opens up, and then you can make wrap text here, so it's
    easy to read that way."

    So the long columns wrap and every row is the SAME fixed height. Wrapping
    alone made each row as tall as its longest cell, which is why they were off
    before; a fixed height keeps the rows uniform and shows the first several
    lines of each long cell. A reviewer expands the row, or reads the cell in
    the formula bar, for the rest.
    """
    # The banner row goes in first. Inserting it afterwards would leave the
    # conditional formatting pointing a row above the data it was built for,
    # because openpyxl does not move those ranges when rows shift.
    sheet.insert_rows(1)
    header_row, first_data = 2, 3
    last_row = sheet.max_row
    sheet.freeze_panes = "B3"
    sheet.auto_filter.ref = (
        f"A{header_row}:"
        f"{sheet.cell(row=header_row, column=sheet.max_column).column_letter}{last_row}")

    order_fill = PatternFill("solid", fgColor="DDE5F0")     # from the order
    llm_fill = PatternFill("solid", fgColor="FFE699")       # the model's words
    record_fill = PatternFill("solid", fgColor="E2EFDA")    # the ePCR
    reviewer_fill = PatternFill("solid", fgColor="EDEDED")  # the reviewer's
    score_fill = PatternFill("solid", fgColor="1F3864")

    group_of: Dict[str, str] = {}
    for name, headers in GROUP_ORDER:
        for header in headers:
            group_of[header] = name
    fill_of = {"FROM THE ORDER, as entered": order_fill,
               "LLM ASSESSMENT OF THE ORDER": llm_fill,
               "ePCR, as documented": record_fill,
               "LLM ASSESSMENT OF THE ePCR": llm_fill,
               "REVIEWER": reviewer_fill}

    category_score_letters = []
    spans: Dict[str, List[int]] = {}

    for n in range(1, sheet.max_column + 1):
        cell = sheet.cell(row=header_row, column=n)
        header = str(cell.value or "")
        letter = cell.column_letter
        group = group_of.get(header, "")
        spans.setdefault(group, []).append(n)

        cell.alignment = Alignment(vertical="bottom", wrap_text=False)
        cell.font = Font(bold=True, size=10)
        values = [sheet.cell(row=i, column=n).value
                  for i in range(first_data, min(last_row, first_data + 80) + 1)]

        if header.startswith("Score:"):
            cell.fill = score_fill
            cell.font = Font(bold=True, size=10, color="FFFFFF")
            sheet.column_dimensions[letter].width = 11
            for i in range(first_data, last_row + 1):
                sheet.cell(row=i, column=n).alignment = Alignment(
                    horizontal="center", vertical="top")
            # Only the three that make the score are colour-scaled. The score
            # itself is on a different scale, and isolation is always 0 until a
            # form question feeds it, so colouring it red every row is noise.
            if header in [h for h, _ in DOC_COLUMNS]:
                category_score_letters.append(letter)
            continue

        cell.fill = fill_of.get(group, order_fill)
        width = max([len(str(v)) for v in values if v] + [len(header)], default=10)
        cap = 42 if header in LONG_COLUMNS else 22
        sheet.column_dimensions[letter].width = min(max(width + 2, 11), cap)
        wrap = header in LONG_COLUMNS
        for i in range(first_data, last_row + 1):
            sheet.cell(row=i, column=n).alignment = Alignment(
                vertical="top", wrap_text=wrap)

    # The banner, merged across each group's columns, so what the nurse entered
    # and what the model concluded are visibly different parts of the sheet.
    for name, _ in GROUP_ORDER:
        columns = spans.get(name) or []
        if not columns:
            continue
        left, right = min(columns), max(columns)
        if right > left:
            sheet.merge_cells(start_row=1, start_column=left,
                              end_row=1, end_column=right)
        banner = sheet.cell(row=1, column=left)
        banner.value = (f"{name}   ({SCORE_LEGEND})"
                        if name == "LLM ASSESSMENT OF THE ORDER" else name)
        banner.fill = llm_fill if name in LLM_GROUPS else fill_of.get(name, order_fill)
        banner.font = Font(bold=True, size=9, color="1F3864")
        banner.alignment = Alignment(horizontal="center", vertical="center")

    # Green where a category is supported, red where it is not, so the six
    # scores read as a shape across the row before anything is read as a number.
    if category_score_letters and last_row >= first_data:
        sheet.conditional_formatting.add(
            f"{category_score_letters[0]}{first_data}:"
            f"{category_score_letters[-1]}{last_row}",
            ColorScaleRule(start_type="num", start_value=0, start_color="F8CBAD",
                           mid_type="num", mid_value=1, mid_color="FFE699",
                           end_type="num", end_value=2, end_color="C6E0B4"))

    # One fixed height for every row, so the sheet stays a grid rather than
    # each row growing to its longest cell. Eight lines at 11pt.
    for n in range(first_data, last_row + 1):
        sheet.row_dimensions[n].height = ROW_HEIGHT
