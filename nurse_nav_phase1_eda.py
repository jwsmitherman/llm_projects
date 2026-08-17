# Databricks notebook source

import os, re, json, warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 300)

DATA_DIR = "/Workspace/Users/josh.smitherman@gmr.net/nurse_nav/data"
OUT_DIR  = "/Workspace/Users/josh.smitherman@gmr.net/nurse_nav/results"

SAMPLE_N    = 500
RANDOM_SEED = 42
LLM_MODEL   = "databricks-gpt-oss-120b"

os.makedirs(OUT_DIR, exist_ok=True)

RUN_ID = pd.Timestamp.now().strftime("%Y%m%d_%H%M")


RESULTS = {}


def save(df: pd.DataFrame, name: str, stamp: bool = True) -> str:
    """Register a result frame in memory for the workbook. Does not write a CSV."""
    RESULTS[name] = df.copy()
    print(f"kept   {len(df):>7,} rows -> {name}")
    return name


print("RUN_ID:", RUN_ID)
print("workbook will be saved to ->", OUT_DIR)


import glob

SOURCE_FILE = "data_april2026-aug2026.xlsx"

PRIMARY_INPUT = os.path.join(DATA_DIR, SOURCE_FILE)
PATHS = {"primary": PRIMARY_INPUT, "out_dir": OUT_DIR}

GOLD_FILE = ""
_gold_hits = [f for f in glob.glob(os.path.join(DATA_DIR, "*"))
              if re.match(r"^llm[_ ]validation[_ ]set[_ ]51",
                          os.path.splitext(os.path.basename(f))[0].lower())]
if _gold_hits:
    GOLD_FILE = sorted(_gold_hits, key=os.path.getmtime, reverse=True)[0]
    PATHS["gold"] = GOLD_FILE

print("Source file       ->", SOURCE_FILE)
print("Exists            ->", os.path.exists(PRIMARY_INPUT))
print("Validation set    ->", os.path.basename(GOLD_FILE) if GOLD_FILE else "not found (Section 8 skipped)")

# COMMAND ----------
rows = []
for f in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
    base = os.path.basename(f)
    rows.append({
        "file": base,
        "ext": os.path.splitext(base)[1].lower(),
        "MB": round(os.path.getsize(f) / 1e6, 2),
        "modified": pd.to_datetime(os.path.getmtime(f), unit="s"),
        "timestamped_copy": bool(re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", base)),
        "resolved_as": next((k for k, v in PATHS.items() if v == f), ""),
    })

inventory = pd.DataFrame(rows).sort_values("file")
display(inventory)

print("Formats present:", inventory["ext"].value_counts().to_dict())
dupes = inventory[inventory["timestamped_copy"]]
if len(dupes):
    print(f"\n{len(dupes)} timestamped duplicate(s) present -- "
          "confirm the discovery step picked the right version:")
    display(dupes[["file", "MB", "modified", "resolved_as"]])

# COMMAND ----------
def peek(path: str, n: int = 3):
    print("=" * 90)
    print(os.path.basename(path))
    print("=" * 90)
    if not path or not os.path.exists(path):
        print("  MISSING\n")
        return
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(path)
            print("  sheets:", xl.sheet_names)
            df = xl.parse(xl.sheet_names[0], nrows=200)
        elif path.lower().endswith(".parquet"):
            df = pd.read_parquet(path).head(200)
        else:
            df = pd.read_csv(path, nrows=200, low_memory=False)
        print(f"  rows previewed: {len(df)} | columns ({len(df.columns)}):")
        print(" ", list(df.columns))
        display(df.head(n))
    except Exception as e:
        print("  could not read:", e)
    print()


for key in ["primary", "gold"]:
    if key in PATHS:
        peek(PATHS[key])

# COMMAND ----------
_probe = pd.DataFrame({"check": ["write_test"], "run_id": [RUN_ID]})
_probe_path = os.path.join(OUT_DIR, f"_write_test_{RUN_ID}.csv")

try:
    _probe.to_csv(_probe_path, index=False)
    assert len(pd.read_csv(_probe_path)) == 1
    os.remove(_probe_path)
    print(f"OK -- writable: {OUT_DIR}")
except Exception as e:
    print(f"CANNOT WRITE to {OUT_DIR}\n  {type(e).__name__}: {e}\n")
    print("Falling back to local disk for the workbook.")
    OUT_DIR = "/tmp/nurse_nav_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"OUT_DIR now: {OUT_DIR}")

# COMMAND ----------
def clean_col(col: str) -> str:
    col = re.sub(r"[^\w]+", "_", col.strip())
    return re.sub(r"_+", "_", col).lower()


def parse_protocol_trigger(nmtara_notes: str) -> str:
    if not isinstance(nmtara_notes, str):
        return ""
    m = re.search(r"(?i)trigger:\s*(.*)", nmtara_notes)
    return m.group(1).strip() if m else ""


def choose_case_id(row: pd.Series) -> str:
    for col in ["work_set_id", "external_reference_number"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return ""


def read_any(path: str, sheet: int | str = 0) -> pd.DataFrame:
    """Read whatever format the file happens to be in."""
    low = path.lower()
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    if low.endswith(".parquet"):
        return pd.read_parquet(path)
    if low.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def load_calls(path: str) -> pd.DataFrame:
    df = read_any(path)
    df.columns = [clean_col(x) for x in df.columns]

    if "transaction_response_names" in df.columns:
        df["documented_disposition"] = df["transaction_response_names"]

    df["case_id"] = df.apply(choose_case_id, axis=1)

    for col in [x for x in df.columns if "date" in x or "time" in x]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


calls = load_calls(PRIMARY_INPUT)
print(f"Source : {os.path.basename(PRIMARY_INPUT)}")
print(f"{len(calls):,} rows | {calls.shape[1]} columns")
calls.head(3)

# COMMAND ----------
def find_col(df: pd.DataFrame, exact: List[str], contains: List[str] = None):
    """Resolve a logical column to a real one: exact match first (underscore-insensitive), then substring."""
    norm = lambda x: x.strip("_")
    normalized = {norm(c): c for c in df.columns}
    for c in exact:
        if c in df.columns:
            return c
        if norm(c) in normalized:
            return normalized[norm(c)]
    for pat in (contains or []):
        hits = [c for c in df.columns if pat in c]
        if hits:
            return sorted(hits, key=len)[0]
    return None


COLS = {
    "notes": find_col(calls,
        ["nurses_notes", "nurse_notes", "notes", "note_text", "call_notes",
         "nmtara_notes", "transaction_notes", "comments"],
        ["nurses_note", "note", "comment"]),
    "date": find_col(calls,
        ["transaction_create_date_time_eastern", "transaction_create_date_time",
         "start_of_hold_date_time", "create_date_time", "call_date", "transaction_date"],
        ["create_date", "date_time", "_date"]),
    "nmtara": find_col(calls,
        ["nmtara_response", "nmtara", "nmtara_level",
         "transaction_breakout_including_bls_nmtara_breakout"],
        ["nmtara", "breakout", "tara"]),
    "dispo": find_col(calls,
        ["transaction_response_names", "documented_disposition", "response_macro",
         "response", "disposition", "final_disposition", "outcome"],
        ["response_name", "disposition", "response"]),
    "client": find_col(calls,
        ["market_name", "client_name", "client", "market", "agency", "customer"],
        ["market", "client", "agency"]),
    "episode": find_col(calls,
        ["external_reference_number", "incident_id", "incident_number",
         "patient_id", "encounter_id"],
        ["external_reference", "incident"]),
    "workset": find_col(calls,
        ["work_set_id", "workset_id", "worksetid"],
        ["work_set", "workset"]),
    "protocol": find_col(calls,
        ["cause", "protocol_trigger", "protocol", "chief_complaint", "clinical_pathway"],
        ["cause", "protocol", "complaint", "pathway"]),
}

print(f"{'logical':10s} -> resolved")
print("-" * 60)
for k, v in COLS.items():
    flag = "" if v else "   <-- NOT FOUND"
    print(f"{k:10s} -> {v}{flag}")

print("\n--- all columns available ---")
for c in sorted(calls.columns):
    print(" ", c)

# COMMAND ----------
OVERRIDES = {
}
COLS.update({k: v for k, v in OVERRIDES.items() if v})

NOTES_COL   = COLS["notes"]
DATE_COL    = COLS["date"]
NMTARA_COL  = COLS["nmtara"]
DISPO_COL   = COLS["dispo"]
CLIENT_COL  = COLS["client"]
EPISODE_KEY = COLS["episode"]
PROTO_COL   = COLS["protocol"]

missing = [k for k in ["notes", "dispo"] if not COLS[k]]
if missing:
    raise ValueError(
        f"Cannot proceed without: {missing}. "
        f"Set them in OVERRIDES above. Available columns: {sorted(calls.columns)}"
    )

for k in ["date", "nmtara", "client", "episode"]:
    if not COLS[k]:
        print(f"WARNING: no '{k}' column found -- related sections will be skipped")

if DATE_COL:
    calls[DATE_COL] = pd.to_datetime(calls[DATE_COL], errors="coerce")
    print(f"\nDate range: {calls[DATE_COL].min()}  ->  {calls[DATE_COL].max()}")

calls[NOTES_COL] = calls[NOTES_COL].astype("string")
print(f"Notes populated: {calls[NOTES_COL].notna().mean():.1%}")

# COMMAND ----------
profile = pd.DataFrame({
    "dtype":      calls.dtypes.astype(str),
    "non_null":   calls.notna().sum(),
    "pct_null":   (calls.isna().mean() * 100).round(1),
    "n_unique":   calls.nunique(dropna=True),
})
profile = profile.sort_values("pct_null")
display(profile)

if DATE_COL:
    monthly = calls.dropna(subset=[DATE_COL]).set_index(DATE_COL).resample("MS").size()
    ax = monthly.plot(figsize=(12, 3), marker="o")
    ax.set_title("Call volume by month")
    ax.set_ylabel("work set IDs")
    plt.tight_layout(); plt.show()

# COMMAND ----------
print("Episode key:", EPISODE_KEY or "NONE FOUND -- de-dup will be skipped")

if EPISODE_KEY:
    per_episode = calls.groupby(EPISODE_KEY)["case_id"].nunique()
    print(per_episode.describe())
    print("\nEpisodes by number of work set IDs:")
    print(per_episode.value_counts().sort_index().head(10))

    dup_rate = (per_episode > 1).mean()
    print(f"\n{dup_rate:.1%} of episodes contain more than one work set ID "
          f"-> naive counting inflates volume by ~{(per_episode.sum()/len(per_episode))-1:.1%}")

# COMMAND ----------
OPERATIONAL_MARKERS = [
    "called back", "left voicemail", "vm left", "no answer", "confirmed pickup",
    "lyft", "ride scheduled", "dispatch", "cad", "ems disposition saved",
    "transferred to", "callback number", "faxed", "record updated",
]

CLINICAL_MARKERS = [
    "pain", "denies", "reports", "onset", "history of", "vitals", "bp ",
    "temp", "symptom", "medication", "assessed", "triage", "complains",
]

def marker_hits(text: str, markers: List[str]) -> int:
    if not isinstance(text, str):
        return 0
    low = text.lower()
    return sum(1 for m in markers if m in low)


calls["note_len"]     = calls[NOTES_COL].fillna("").str.len()
calls["op_hits"]      = calls[NOTES_COL].apply(lambda t: marker_hits(t, OPERATIONAL_MARKERS))
calls["clin_hits"]    = calls[NOTES_COL].apply(lambda t: marker_hits(t, CLINICAL_MARKERS))
calls["is_operational_only"] = (calls["op_hits"] > 0) & (calls["clin_hits"] == 0)

print(f"Operational-only work set IDs: {calls['is_operational_only'].mean():.1%}")
print("\n--- sample operational-only note ---")
print(calls.loc[calls["is_operational_only"], NOTES_COL].dropna().head(1).values)
print("\n--- sample clinical note ---")
print(calls.loc[~calls["is_operational_only"], NOTES_COL].dropna().head(1).values)

# COMMAND ----------
analysis = calls[~calls["is_operational_only"]].copy()

if EPISODE_KEY:
    if DATE_COL:
        analysis = analysis.sort_values(DATE_COL)
    analysis = analysis.drop_duplicates(subset=[EPISODE_KEY], keep="first")
else:
    print("No episode key -- keeping every work set ID. Bucket sizes will be inflated; "
          "confirm the correct episode key with Rich before quoting any number.")

print(f"Raw rows        : {len(calls):,}")
print(f"Analysis rows   : {len(analysis):,}  ({len(analysis)/len(calls):.1%} retained)")

# COMMAND ----------
def nmtara_level(x):
    t = str(x)
    m = re.search(r"(?i)n[am]?tara[^0-9]{0,6}(\d)", t)
    if m:
        return int(m.group(1))
    if re.search(r"(?i)self[- ]?care", t):
        return np.nan
    m = re.search(r"(?<![0-9])([0-6])(?![0-9])", t)
    return int(m.group(1)) if m else np.nan

if NMTARA_COL:
    analysis["nmtara_level"] = analysis[NMTARA_COL].apply(nmtara_level)
else:
    analysis["nmtara_level"] = np.nan
    print("No NMTARA column -- NMTARA 6 and override buckets cannot be identified")

dispo_low = analysis[DISPO_COL].fillna("").str.lower()

analysis["is_self_care"] = dispo_low.str.contains("self", na=False)
analysis["is_nmtara6"]   = analysis["nmtara_level"].eq(6)
analysis["is_ambulance"] = dispo_low.str.contains("ambulance|bls|als|911", regex=True, na=False)
analysis["is_amb_override"] = analysis["nmtara_level"].between(1, 5) & analysis["is_ambulance"]

def bucket(row):
    if row["is_nmtara6"]:       return "NMTARA 6 (triage not completed)"
    if row["is_amb_override"]:  return "1-5 ambulance override"
    if row["is_self_care"]:     return "Self-care"
    return "Other"

analysis["phase1_bucket"] = analysis.apply(bucket, axis=1)

summary = (
    analysis["phase1_bucket"].value_counts()
    .rename("calls").to_frame()
    .assign(pct=lambda d: (d["calls"] / len(analysis) * 100).round(1))
)
display(summary)

# COMMAND ----------
if DATE_COL:
    trend = (
        analysis.dropna(subset=[DATE_COL]).set_index(DATE_COL)
                .groupby([pd.Grouper(freq="QS"), "phase1_bucket"]).size()
                .unstack(fill_value=0)
    )
    trend_pct = trend.div(trend.sum(axis=1), axis=0) * 100
    ax = trend_pct.plot(figsize=(12, 4), marker="o")
    ax.set_title("Bucket mix by quarter (% of navigations)")
    ax.set_ylabel("% of calls"); ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.show()
    display(trend_pct.round(1).tail(8))

# COMMAND ----------
if CLIENT_COL:
    by_client = (
        pd.crosstab(analysis[CLIENT_COL], analysis["phase1_bucket"], normalize="index") * 100
    ).round(1)
    by_client["n_calls"] = analysis[CLIENT_COL].value_counts()
    display(by_client.sort_values("n_calls", ascending=False).head(20))
else:
    print("No client/market column found -- request from Rich.")

# COMMAND ----------
def note_quality(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) < 20:
        return "empty/stub"
    if len(text) < 150:
        return "thin"
    if len(text) < 600:
        return "adequate"
    return "rich"

analysis["note_quality"] = analysis[NOTES_COL].apply(note_quality)

QUAL_ORDER = ["empty/stub", "thin", "adequate", "rich"]
qual = (
    pd.crosstab(analysis["phase1_bucket"], analysis["note_quality"], normalize="index") * 100
).round(1).reindex(columns=QUAL_ORDER, fill_value=0.0)
display(qual)

analysis["note_usable"] = analysis["note_quality"].isin(["adequate", "rich"])
print("\nEstimated usable share by bucket:")
print(analysis.groupby("phase1_bucket")["note_usable"].mean().mul(100).round(1))

# COMMAND ----------
if DATE_COL:
    ax = (analysis.dropna(subset=[DATE_COL]).set_index(DATE_COL)
                  .groupby(pd.Grouper(freq="QS"))["note_len"]
                  .median()
                  .plot(figsize=(12, 3), marker="o"))
    ax.set_title("Median nurse note length by quarter")
    ax.set_ylabel("characters")
    plt.tight_layout(); plt.show()

# COMMAND ----------
if "protocol_trigger" not in analysis.columns:
    analysis["protocol_trigger"] = analysis[PROTO_COL] if PROTO_COL else ""

for b in ["Self-care", "NMTARA 6 (triage not completed)", "1-5 ambulance override"]:
    sub = analysis[(analysis["phase1_bucket"] == b) & analysis["note_usable"]]
    if sub.empty:
        continue
    print("=" * 100)
    print(f"{b}   (n={len(sub):,})")
    print("=" * 100)
    for _, r in sub.sample(min(3, len(sub)), random_state=RANDOM_SEED).iterrows():
        print(f"\n[case {r['case_id']}] protocol={r.get('protocol_trigger','')} "
              f"| dispo={r[DISPO_COL]}")
        print(str(r[NOTES_COL])[:1200])
        print("-" * 100)

# COMMAND ----------
REASON_LEXICON = {
    "Patient refused alternative":  ["refused", "declined", "insisted", "wants ambulance", "demanded"],
    "No provider available":        ["no provider", "unavailable", "no one available", "wait time", "queue"],
    "After-hours / closed":         ["closed", "after hours", "not open", "opens at", "overnight"],
    "Mobility / cannot transport":  ["bedbound", "bed bound", "wheelchair", "unable to walk", "cannot ambulate",
                                     "lift assist", "bariatric"],
    "Catheter / device issue":      ["catheter", "foley", "suprapubic", "g-tube", "picc", "ostomy"],
    "Self-transport (not self-care)": ["drive himself", "drive herself", "driving themselves", "family will take",
                                       "will take himself", "will take herself", "own transportation"],
    "Clinical escalation":          ["deteriorat", "worsening", "unstable", "escalat", "red flag"],
    "Technical / call dropped":     ["disconnect", "call dropped", "line went dead", "could not reach",
                                     "poor connection"],
    "Language barrier":             ["interpreter", "language barrier", "does not speak"],
    "Patient unable to participate": ["altered", "confused", "unresponsive", "unable to answer", "intoxicated"],
}

def tag_reasons(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    low = text.lower()
    return [r for r, terms in REASON_LEXICON.items() if any(t in low for t in terms)]

analysis["reason_tags"] = analysis[NOTES_COL].apply(tag_reasons)
analysis["n_reason_tags"] = analysis["reason_tags"].str.len()

print(f"Notes with at least one reason tag: {(analysis['n_reason_tags'] > 0).mean():.1%}")

exploded = analysis.explode("reason_tags").dropna(subset=["reason_tags"]).reset_index(drop=True)
heat = (
    pd.crosstab(exploded["phase1_bucket"], exploded["reason_tags"], normalize="index") * 100
).round(1)
display(heat)

# COMMAND ----------
from openai import OpenAI

DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    if "dbutils" in dir() else os.environ.get("DATABRICKS_TOKEN", "")
)

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-2790612761746757.17.azuredatabricks.net/serving-endpoints",
)

def llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 2500) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
        return json.dumps(content)
    return content

# COMMAND ----------
NAV_REASON_SYSTEM_PROMPT = """
You are a clinical note information extraction assistant for a nurse navigation program.
Your ONLY task is to extract structured facts from the "Nurse Notes" free text.

CRITICAL CONSTRAINTS
- Do NOT provide medical advice, diagnose, or decide what the patient should have received.
- Do NOT guess. Only extract what is explicitly supported by the text.
- Handle negations correctly.
- For every boolean set to true you MUST provide a verbatim evidence quote.
- If the reason for the navigation decision is not documented, set
  documentation.reason_documented = false.

IGNORE these protocol metadata labels as evidence:
"Clinical Pathway:", "Trigger:", "NMTARA:", "Disposition:", "Reference:",
"Corti Date/Time:", "Corti Session ID:", "Home Care Recommendations:",
and workflow lines such as "EMS disposition saved...".

ABBREVIATIONS: CP=chest pain, SOB=shortness of breath, AMS=altered mental status,
LOC=loss of consciousness, N/V=nausea/vomiting, ETOH=alcohol intoxication,
BLS/ALS=basic/advanced life support, UC=urgent care, ED/ER=emergency department.

OUTPUT: valid JSON only, all keys present, booleans never null.

SCHEMA:
{
  "case_id": string,
  "bucket_context": string,
  "call_summary": string,

  "care_setting_actual": {
    "setting": "emergency_department | urgent_care | virtual_care | primary_care | home_no_care | unknown",
    "patient_stayed_home": boolean
  },

  "transport_mode": {
    "mode": "ambulance_als | ambulance_bls | rideshare | self_or_family | none | unknown",
    "arranged_by_program": boolean,
    "ride_initiated_by": "patient_or_family | program_or_nurse | unknown"
  },

  "decision_driver": {
    "patient_refused_recommendation": boolean,
    "patient_requested_ambulance_or_ed": boolean,
    "no_provider_available": boolean,
    "facility_closed_or_after_hours": boolean,
    "no_appointment_available": boolean,
    "mobility_or_transport_barrier": boolean,
    "device_or_procedure_need": boolean,
    "clinical_escalation_by_nurse": boolean,
    "insurance_or_cost_barrier": boolean,
    "language_or_communication_barrier": boolean
  },

  "triage_incomplete_reason": {
    "patient_refused_triage": boolean,
    "call_disconnected_or_technical": boolean,
    "patient_unable_to_participate": boolean,
    "protocol_exclusion": boolean,
    "caller_was_not_patient": boolean
  },

  "documentation": {
    "reason_documented": boolean,
    "note_is_operational_only": boolean,
    "missing_elements": [string]
  },

  "evidence": [
    {"field": string, "value": boolean, "quote": string, "rationale": string}
  ]
}

FIELD NOTES
- care_setting_actual.setting = where the patient ACTUALLY ended up per the note, which may differ
  from the recorded disposition. This is the field that re-classifies mislabeled "self-care".
- patient_stayed_home = true ONLY if the note shows no further care was sought.
- transport_mode captures HOW they got there, separately from WHERE.
- ride_initiated_by = WHO arranged the transport, which decides true self-care:
    * "patient_or_family" ONLY if the note explicitly says the patient or a family member arranged the
      ride themselves (for example "patient ordered own Lyft", "family will drive patient").
    * "program_or_nurse" if the note says the nurse or program arranged it (for example "ride ordered",
      "nurse ordered ride", "arranged transport").
    * "unknown" if the note only says "a ride was ordered" without stating who. Set it to "unknown";
      the analysis treats unknown as program-arranged, because an unattributed ride is assumed to be
      arranged by the program.
  True self-care means the program provided neither care nor transport: the patient handled it
  themselves. A program-arranged or unattributed ride is NOT self-care.
- triage_incomplete_reason applies to NMTARA 6 cases; leave all false otherwise.

Now extract from the following Nurse Notes."""


def build_user_prompt(case_id: str, bucket: str, protocol: str, notes: str) -> str:
    return (
        f"case_id: {case_id}\n"
        f"bucket_context: {bucket}\n"
        f"protocol_trigger: {protocol}\n\n"
        f"Nurse Notes:\n{notes}"
    )

# COMMAND ----------
REQUIRED_TOP_KEYS = {
    "case_id", "bucket_context", "call_summary", "care_setting_actual",
    "transport_mode", "decision_driver", "triage_incomplete_reason",
    "documentation", "evidence",
}

BOOL_SECTIONS = {
    "decision_driver": [
        "patient_refused_recommendation", "patient_requested_ambulance_or_ed",
        "no_provider_available", "facility_closed_or_after_hours",
        "no_appointment_available", "mobility_or_transport_barrier",
        "device_or_procedure_need", "clinical_escalation_by_nurse",
        "insurance_or_cost_barrier", "language_or_communication_barrier",
    ],
    "triage_incomplete_reason": [
        "patient_refused_triage", "call_disconnected_or_technical",
        "patient_unable_to_participate", "protocol_exclusion",
        "caller_was_not_patient",
    ],
}

REASON_LABELS = {
    "patient_refused_triage":          "Patient would not answer the triage questions (typically wanted the ER)",
    "call_disconnected_or_technical":  "The call dropped or hit a technical problem before triage finished",
    "patient_unable_to_participate":   "Patient could not take part (confused, unresponsive, or intoxicated)",
    "protocol_exclusion":              "The protocol did not allow triage for this type of call",
    "caller_was_not_patient":          "The caller was someone other than the patient",
    "clinical_escalation_by_nurse":    "The nurse escalated based on clinical judgement",
    "mobility_or_transport_barrier":   "The patient could not get there another way (bedbound, no transport)",
    "patient_requested_ambulance_or_ed":"The patient asked for an ambulance or the ER",
    "patient_refused_recommendation":  "The patient declined the recommended lower-acuity option",
    "no_provider_available":           "No provider or facility was available",
    "facility_closed_or_after_hours":  "The facility was closed or it was after hours",
    "no_appointment_available":        "No appointment was available",
    "device_or_procedure_need":        "The patient needed a device or procedure (for example a catheter)",
    "insurance_or_cost_barrier":       "Insurance or cost was a barrier",
    "language_or_communication_barrier":"A language or communication barrier",
}


def plain_label(key: str) -> str:
    return REASON_LABELS.get(key, key.replace("_", " ").capitalize())


def validate_extraction(raw: str) -> Tuple[bool, Dict[str, Any], str]:
    try:
        obj = json.loads(raw)
    except Exception as e:
        return False, {}, f"invalid JSON: {e}"

    missing = REQUIRED_TOP_KEYS - set(obj)
    if missing:
        return False, obj, f"missing keys: {sorted(missing)}"

    for section, keys in BOOL_SECTIONS.items():
        sec = obj.get(section, {})
        if not isinstance(sec, dict):
            return False, obj, f"{section} must be an object"
        for k in keys:
            if not isinstance(sec.get(k), bool):
                return False, obj, f"{section}.{k} must be boolean"

    ev_fields = {e.get("field") for e in obj.get("evidence", []) if isinstance(e, dict)}
    for section, keys in BOOL_SECTIONS.items():
        for k in keys:
            if obj[section][k] and f"{section}.{k}" not in ev_fields:
                return False, obj, f"true flag without evidence: {section}.{k}"

    return True, obj, "OK"

# COMMAND ----------
sample = (
    analysis[analysis["phase1_bucket"] != "Other"]
    .groupby("phase1_bucket", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), SAMPLE_N // 3), random_state=RANDOM_SEED))
)
print(f"Extracting {len(sample)} cases")

rows = []
for _, r in sample.iterrows():
    prompt = build_user_prompt(
        r["case_id"], r["phase1_bucket"], r.get("protocol_trigger", ""), r[NOTES_COL]
    )
    try:
        raw = llm_call(NAV_REASON_SYSTEM_PROMPT, prompt)
        ok, obj, msg = validate_extraction(raw)
    except Exception as e:
        ok, obj, msg, raw = False, {}, f"call failed: {e}", ""

    rows.append({
        "case_id": r["case_id"],
        "phase1_bucket": r["phase1_bucket"],
        "extraction_valid": int(ok),
        "extraction_error": "" if ok else msg,
        "extraction_json": raw if ok else "{}",
    })

extractions = pd.DataFrame(rows)
print(f"Valid extractions: {extractions['extraction_valid'].mean():.1%}")
display(extractions.loc[extractions["extraction_valid"] == 0, "extraction_error"].value_counts().head(10))

save(extractions, "llm_extractions_raw")

# COMMAND ----------
if "extractions" not in globals():
    raise RuntimeError("Run the extraction cell in Section 6 first (or load a saved "
                       "llm_extractions_raw_*.csv from the results folder).")

scored = sample.merge(extractions, on=["case_id", "phase1_bucket"], how="left")
valid = scored[scored["extraction_valid"] == 1].copy()
valid["obj"] = valid["extraction_json"].apply(json.loads)

def flat(o, section, key):
    return bool(o.get(section, {}).get(key, False))

for section, keys in BOOL_SECTIONS.items():
    for k in keys:
        valid[f"{section}.{k}"] = valid["obj"].apply(lambda o, s=section, kk=k: flat(o, s, kk))

valid["actual_setting"]  = valid["obj"].apply(lambda o: o.get("care_setting_actual", {}).get("setting", "unknown"))
valid["stayed_home"]     = valid["obj"].apply(lambda o: bool(o.get("care_setting_actual", {}).get("patient_stayed_home", False)))
valid["transport"]       = valid["obj"].apply(lambda o: o.get("transport_mode", {}).get("mode", "unknown"))
valid["arranged_by_program"] = valid["obj"].apply(lambda o: bool(o.get("transport_mode", {}).get("arranged_by_program", False)))
valid["ride_initiated_by"]   = valid["obj"].apply(lambda o: o.get("transport_mode", {}).get("ride_initiated_by", "unknown"))
valid["reason_documented"] = valid["obj"].apply(lambda o: bool(o.get("documentation", {}).get("reason_documented", False)))


def is_true_self_care(row) -> bool:
    """True self-care = the program provided neither care nor transport.
    Per the self-care definition agreed with the business:
      - patient stayed home with no care sought -> true self-care
      - patient went somewhere but arranged their OWN transport -> still self-care
      - program/nurse arranged the ride, OR the note is silent on who did (assume program) -> NOT self-care
    """
    if row["stayed_home"]:
        return True
    if row["arranged_by_program"]:
        return False
    return row["ride_initiated_by"] == "patient_or_family"


valid["true_self_care"] = valid.apply(is_true_self_care, axis=1)

print("Reason documented rate by bucket:")
print(valid.groupby("phase1_bucket")["reason_documented"].mean().mul(100).round(1))

# COMMAND ----------
def actual_outcome(row) -> Dict[str, str]:
    """Read where the patient ended up (and transport) from the extracted note facts.
    This is a plain-language relabel of what the note already says - no new-system logic."""
    setting   = row["actual_setting"]
    transport = row["transport"]

    where = {
        "emergency_department": "Emergency Department",
        "urgent_care":          "Urgent Care",
        "virtual_care":         "Virtual Care",
        "primary_care":         "Primary Care",
        "home_no_care":         "No care setting (true self-care)",
    }.get(setting, "Unknown")

    how = {
        "ambulance_als":  "Ambulance (ALS)",
        "ambulance_bls":  "Ambulance (BLS)",
        "rideshare":      "Program-arranged rideshare",
        "self_or_family": "Self / family transport",
        "none":           "No transport needed",
    }.get(transport, "Unknown")

    return {"where": where, "how": how}


if "valid" not in globals() or len(valid) == 0:
    raise RuntimeError("No validated extractions available -- run Section 6 first.")

outcome_df = valid.join(pd.DataFrame(valid.apply(actual_outcome, axis=1).tolist(), index=valid.index))
display(pd.crosstab(outcome_df["where"], outcome_df["how"]))

# COMMAND ----------
sc = outcome_df[outcome_df["phase1_bucket"] == "Self-care"] if "outcome_df" in globals() else pd.DataFrame()

if len(sc):
    breakdown = (
        sc["where"].value_counts(normalize=True).mul(100).round(1)
        .rename("pct_of_self_care").to_frame()
    )
    breakdown["calls"] = sc["where"].value_counts()
    display(breakdown)

    ax = breakdown["pct_of_self_care"].plot(kind="barh", figsize=(9, 4))
    ax.set_title("What the current 'self-care' bucket actually contains")
    ax.set_xlabel("% of self-care calls")
    plt.tight_layout(); plt.show()

    true_rate = sc["true_self_care"].mean()
    print(f"True self-care (program provided neither care nor transport): {true_rate:.1%}")
    print(f"Mislabeled (program arranged transport, or ride not attributed): {1 - true_rate:.1%}")

# COMMAND ----------
if len(sc):
    def initiator_label(row):
        if row["stayed_home"]:
            return "No ride - stayed home (self-care)"
        if row["ride_initiated_by"] == "patient_or_family" and not row["arranged_by_program"]:
            return "Patient / family arranged ride (self-care)"
        if row["arranged_by_program"] or row["ride_initiated_by"] == "program_or_nurse":
            return "Program / nurse arranged ride (not self-care)"
        return "Ride not attributed - assumed program (not self-care)"

    sc_init = sc.copy()
    sc_init["self_care_class"] = sc_init.apply(initiator_label, axis=1)
    init_breakdown = (
        sc_init["self_care_class"].value_counts(normalize=True).mul(100).round(1)
        .rename("pct_of_self_care").to_frame()
    )
    init_breakdown["calls"] = sc_init["self_care_class"].value_counts()
    display(init_breakdown)
    save(init_breakdown, "self_care_by_ride_initiator")

# COMMAND ----------
def reason_mix_table(frame, section, pct_label):
    keys = [k for k in BOOL_SECTIONS[section] if f"{section}.{k}" in frame.columns]
    shares = {k: round(frame[f"{section}.{k}"].mean() * 100, 1) for k in keys}
    out = pd.DataFrame({
        "reason": [k.replace("_", " ").capitalize() for k in keys],
        "what it means": [plain_label(k) for k in keys],
        pct_label: [shares[k] for k in keys],
    }).sort_values(pct_label, ascending=False).reset_index(drop=True)
    return out


n6 = outcome_df[outcome_df["phase1_bucket"] == "NMTARA 6 (triage not completed)"] if "outcome_df" in globals() else pd.DataFrame()
if len(n6):
    n6_mix = reason_mix_table(n6, "triage_incomplete_reason", "% of NMTARA 6 calls")
    display(n6_mix)
    save(n6_mix, "nmtara6_reason_mix")

ov = outcome_df[outcome_df["phase1_bucket"] == "1-5 ambulance override"] if "outcome_df" in globals() else pd.DataFrame()
if len(ov):
    ov_mix = reason_mix_table(ov, "decision_driver", "% of override calls")
    display(ov_mix)
    save(ov_mix, "override_reason_mix")

# COMMAND ----------
if len(ov) and "protocol_trigger" in ov.columns:
    top_protocols = ov["protocol_trigger"].value_counts().head(12).index
    sub = ov[ov["protocol_trigger"].isin(top_protocols)]
    piv = sub.groupby("protocol_trigger")[
        [f"decision_driver.{k}" for k in BOOL_SECTIONS["decision_driver"]]
    ].mean().mul(100).round(0)
    piv.columns = [c.split(".")[-1].replace("_", " ") for c in piv.columns]
    display(piv)
    save(piv.reset_index(), "override_reasons_by_protocol")

# COMMAND ----------
DRIVER_GROUPS = {
    "Nurse / clinical": ["clinical_escalation_by_nurse"],
    "Patient": ["patient_requested_ambulance_or_ed", "patient_refused_recommendation"],
    "Access / logistics": [
        "no_provider_available", "facility_closed_or_after_hours", "no_appointment_available",
        "mobility_or_transport_barrier", "device_or_procedure_need",
        "insurance_or_cost_barrier", "language_or_communication_barrier",
    ],
}

if len(ov):
    driver_rows = []
    for label, keys in DRIVER_GROUPS.items():
        present_cols = [f"decision_driver.{k}" for k in keys if f"decision_driver.{k}" in ov.columns]
        share = ov[present_cols].any(axis=1).mean() * 100 if present_cols else 0.0
        driver_rows.append({"driver_group": label, "% of override calls": round(share, 1)})
    driver_split = pd.DataFrame(driver_rows)
    display(driver_split)

    all_reason_cols = [f"decision_driver.{k}" for k in BOOL_SECTIONS["decision_driver"]
                       if f"decision_driver.{k}" in ov.columns]
    n_reasons = ov[all_reason_cols].sum(axis=1)
    multi_reason_pct = round((n_reasons > 1).mean() * 100, 1)
    print(f"Override calls with more than one documented reason: {multi_reason_pct}%")
    print(f"Nurse/clinical is the recorded driver in "
          f"{driver_split.loc[driver_split['driver_group']=='Nurse / clinical', '% of override calls'].iloc[0]}% of overrides.")

    driver_split_out = driver_split.copy()
    driver_split_out.loc[len(driver_split_out)] = ["Calls with more than one reason", multi_reason_pct]
    save(driver_split_out, "override_driver_split")

# COMMAND ----------
if len(ov):
    drv_keys = [k for k in BOOL_SECTIONS["decision_driver"] if f"decision_driver.{k}" in ov.columns]

    def reason_combo(row):
        present = [k.replace("_", " ").capitalize() for k in drv_keys if row[f"decision_driver.{k}"]]
        if not present:
            return "None documented"
        return " + ".join(present)

    ov_combo = ov.copy()
    ov_combo["reasons_included"] = ov_combo.apply(reason_combo, axis=1)
    ov_combo["n_reasons"] = ov_combo[[f"decision_driver.{k}" for k in drv_keys]].sum(axis=1)

    combos = (
        ov_combo["reasons_included"].value_counts()
        .rename("calls").to_frame()
        .assign(**{"% of override calls": lambda d: (d["calls"] / len(ov_combo) * 100).round(1)})
        .reset_index().rename(columns={"index": "reasons_included"})
        .head(15)
    )
    display(combos)
    save(combos, "override_reason_combinations")

    single = (ov_combo["n_reasons"] == 1).mean() * 100
    multi = (ov_combo["n_reasons"] > 1).mean() * 100
    print(f"Override calls with a single reason: {single:.1f}%  |  with more than one reason: {multi:.1f}%")

# COMMAND ----------
_dispo = analysis[DISPO_COL].fillna("").str.lower()
analysis["is_urgent_care"] = _dispo.str.contains("urgent", na=False)
analysis["is_virtual"]     = _dispo.str.contains("virtual|telehealth|video", regex=True, na=False)
analysis["is_ed"]          = _dispo.str.contains("emergency|\\bed\\b|\\ber\\b", regex=True, na=False)

def diversion_block(frame, label):
    total = len(frame)
    if total == 0:
        return None
    return {
        "segment": label,
        "calls": total,
        "self_care_pct":   round(frame["is_self_care"].mean() * 100, 1),
        "urgent_care_pct": round(frame["is_urgent_care"].mean() * 100, 1),
        "virtual_pct":     round(frame["is_virtual"].mean() * 100, 1),
        "lower_acuity_pct": round((frame["is_self_care"] | frame["is_urgent_care"] | frame["is_virtual"]).mean() * 100, 1),
        "ambulance_pct":   round(frame["is_ambulance"].mean() * 100, 1),
    }

div_rows = [diversion_block(analysis, "National")]
if CLIENT_COL:
    for cl, grp in analysis.groupby(CLIENT_COL):
        if len(grp) >= MIN_CLIENT_CALLS if "MIN_CLIENT_CALLS" in globals() else len(grp) >= 100:
            div_rows.append(diversion_block(grp, cl))

diversion_insight = pd.DataFrame([r for r in div_rows if r])
display(diversion_insight)
save(diversion_insight, "diversion_insight")

# COMMAND ----------
def reason_examples(section: str, key: str, n: int = 3) -> pd.DataFrame:
    field = f"{section}.{key}"
    if "valid" not in globals() or field not in globals().get("valid", pd.DataFrame()).columns:
        print(f"{field} not available")
        return pd.DataFrame()
    hits = valid[valid[field]].copy()
    rows = []
    for _, r in hits.head(n).iterrows():
        quote = ""
        for e in r["obj"].get("evidence", []):
            if e.get("field") == field:
                quote = e.get("quote", "")
                break
        rows.append({
            "case_id": r["case_id"],
            "reason": key.replace("_", " "),
            "quote": quote,
            "note_snippet": str(r.get(NOTES_COL, ""))[:300],
        })
    return pd.DataFrame(rows)


examples = pd.concat([
    reason_examples("triage_incomplete_reason", "patient_unable_to_participate", 3),
    reason_examples("triage_incomplete_reason", "patient_refused_triage", 2),
    reason_examples("decision_driver", "clinical_escalation_by_nurse", 3),
], ignore_index=True) if "valid" in globals() else pd.DataFrame()

if len(examples):
    display(examples)
    save(examples, "reason_examples")

# COMMAND ----------
if DATE_COL:
    _d = analysis[DATE_COL].dropna()
    if len(_d):
        months = _d.dt.to_period("M")
        coverage = months.value_counts().sort_index().rename("calls").to_frame()
        coverage.index = coverage.index.astype(str)
        coverage.index.name = "month"
        coverage = coverage.reset_index()
        print(f"Date span: {_d.min().date()} to {_d.max().date()}  |  months covered: {months.nunique()}")
        display(coverage)
        save(coverage, "data_coverage")

# COMMAND ----------
if CLIENT_COL:
    def storyline(frame, label):
        n = len(frame)
        if n == 0:
            return None
        dl = frame[DISPO_COL].fillna("").str.lower()
        row = {
            "market": label,
            "calls": n,
            "self_care_pct": round(dl.str.contains("self", na=False).mean() * 100, 1),
            "urgent_care_pct": round(dl.str.contains("urgent", na=False).mean() * 100, 1),
            "ambulance_pct": round(dl.str.contains("ambulance|bls|als|911", regex=True, na=False).mean() * 100, 1),
        }
        if "nmtara_level" in frame.columns:
            row["nmtara6_pct"] = round(frame["nmtara_level"].eq(6).mean() * 100, 1)
        if "is_amb_override" in frame.columns:
            row["amb_override_pct"] = round(frame["is_amb_override"].mean() * 100, 1)
        return row

    rows = [storyline(analysis, "National")]
    for cl, grp in analysis.groupby(CLIENT_COL):
        if len(grp) >= (MIN_CLIENT_CALLS if "MIN_CLIENT_CALLS" in globals() else 100):
            rows.append(storyline(grp, cl))
    client_storylines = pd.DataFrame([r for r in rows if r])
    display(client_storylines)
    save(client_storylines, "client_storylines")

# COMMAND ----------
if "outcome_df" in globals() and "protocol_trigger" in outcome_df.columns:
    home = outcome_df[(outcome_df["phase1_bucket"] == "Self-care") & (outcome_df["stayed_home"])]
    if len(home):
        sc_complaint = (
            home["protocol_trigger"].replace("", "Not stated").value_counts()
            .rename("calls").to_frame()
            .assign(**{"% of stayed-home self-care": lambda d: (d["calls"] / len(home) * 100).round(1)})
            .reset_index().rename(columns={"index": "chief_complaint"})
            .head(12)
        )
        display(sc_complaint)
        save(sc_complaint, "self_care_by_complaint")

# COMMAND ----------
if "outcome_df" in globals():
    prog = outcome_df[(outcome_df["arranged_by_program"]) |
                      (outcome_df["ride_initiated_by"] == "program_or_nurse")]
    if len(prog):
        def transport_bin(row):
            mode = str(row.get("transport", "")).lower()
            setting = str(row.get("actual_setting", "")).lower()
            if "ambulance" in mode or "bls" in mode or "als" in mode:
                return "GMR ambulance"
            if "rideshare" in mode:
                return "Rideshare / Lyft"
            if "emergency" in setting:
                return "To ER (mode not stated)"
            return "Other / not stated"
        prog = prog.copy()
        prog["transport_bin"] = prog.apply(transport_bin, axis=1)
        program_bins = (
            prog["transport_bin"].value_counts()
            .rename("calls").to_frame()
            .assign(**{"% of program-arranged": lambda d: (d["calls"] / len(prog) * 100).round(1)})
            .reset_index().rename(columns={"index": "transport_bin"})
        )
        display(program_bins)
        save(program_bins, "program_transport_bins")

# COMMAND ----------
if "outcome_df" in globals() and "triage_incomplete_reason.call_disconnected_or_technical" in outcome_df.columns:
    drop = outcome_df[outcome_df["triage_incomplete_reason.call_disconnected_or_technical"]]
    n6_all = outcome_df[outcome_df["phase1_bucket"] == "NMTARA 6 (triage not completed)"]
    print(f"Call-drop / technical calls: {len(drop)} "
          f"({(len(drop)/len(n6_all)*100 if len(n6_all) else 0):.1f}% of NMTARA 6 calls read)")
    if CLIENT_COL and CLIENT_COL in drop.columns and len(drop):
        call_drop_focus = (
            drop[CLIENT_COL].value_counts()
            .rename("call_drop_calls").to_frame()
            .reset_index().rename(columns={"index": "market"})
        )
        display(call_drop_focus)
        save(call_drop_focus, "call_drop_focus")

# COMMAND ----------
if "ov" in globals() and "protocol_trigger" in ov.columns and "decision_driver.clinical_escalation_by_nurse" in ov.columns:
    clin = ov[ov["decision_driver.clinical_escalation_by_nurse"]]
    if len(clin):
        clinical_by_complaint = (
            clin["protocol_trigger"].replace("", "Not stated").value_counts()
            .rename("calls").to_frame()
            .assign(**{"% of clinical-escalation overrides": lambda d: (d["calls"] / len(clin) * 100).round(1)})
            .reset_index().rename(columns={"index": "chief_complaint"})
            .head(12)
        )
        display(clinical_by_complaint)
        save(clinical_by_complaint, "clinical_escalation_by_complaint")

# COMMAND ----------
if "ov" in globals() and "protocol_trigger" in ov.columns and len(ov):
    override_top_complaints = (
        ov["protocol_trigger"].replace("", "Not stated").value_counts()
        .rename("calls").to_frame()
        .assign(**{"% of override calls": lambda d: (d["calls"] / len(ov) * 100).round(1)})
        .reset_index().rename(columns={"index": "chief_complaint"})
        .head(10)
    )
    display(override_top_complaints)
    save(override_top_complaints, "override_top_complaints")

# COMMAND ----------
MOBILITY_DRIVERS = {
    "Wheelchair":        ["wheelchair", "w/c"],
    "Bedbound":          ["bedbound", "bed bound", "bed-bound"],
    "Cannot ambulate":   ["cannot ambulate", "unable to ambulate", "non-ambulatory", "nonambulatory", "unable to walk"],
    "Stairs":            ["stairs", "staircase", "second floor", "upstairs"],
    "Bariatric / lift":  ["bariatric", "lift assist", "lift-assist", "heavy", "two person"],
    "No transport":      ["no transport", "no ride", "no car", "no one to drive", "no way to get"],
}
if "ov" in globals() and "decision_driver.mobility_or_transport_barrier" in ov.columns and NOTES_COL in ov.columns:
    mob = ov[ov["decision_driver.mobility_or_transport_barrier"]]
    if len(mob):
        text = mob[NOTES_COL].fillna("").str.lower()
        rows = []
        for label, terms in MOBILITY_DRIVERS.items():
            hits = text.apply(lambda t: any(term in t for term in terms)).sum()
            rows.append({"mobility_driver": label, "calls": int(hits),
                         "% of mobility-barrier overrides": round(hits / len(mob) * 100, 1)})
        mobility_drivers = pd.DataFrame(rows).sort_values("calls", ascending=False).reset_index(drop=True)
        display(mobility_drivers)
        save(mobility_drivers, "mobility_drivers")

# COMMAND ----------
if "ov" in globals() and len(ov):
    def has_any(frame, keys):
        cols = [f"decision_driver.{k}" for k in keys if f"decision_driver.{k}" in frame.columns]
        return frame[cols].any(axis=1) if cols else pd.Series(False, index=frame.index)

    access_keys = ["no_provider_available", "facility_closed_or_after_hours", "no_appointment_available",
                   "insurance_or_cost_barrier"]
    opp = pd.DataFrame({
        "opportunity_group": [
            "Access / appointment (divertible to UC or telehealth)",
            "Mobility / transport (transport solution)",
            "Patient request",
            "Clinical escalation (nurse judgement)",
        ],
        "% of override calls": [
            round(has_any(ov, access_keys).mean() * 100, 1),
            round(has_any(ov, ["mobility_or_transport_barrier", "device_or_procedure_need"]).mean() * 100, 1),
            round(has_any(ov, ["patient_requested_ambulance_or_ed", "patient_refused_recommendation"]).mean() * 100, 1),
            round(has_any(ov, ["clinical_escalation_by_nurse"]).mean() * 100, 1),
        ],
    })
    display(opp)
    save(opp, "override_opportunities")

# COMMAND ----------
if "gold" not in PATHS:
    print("No validation set found in the data folder -- skipping Section 8.")
    gold = pd.DataFrame()
else:
    gold = read_any(PATHS["gold"])
if len(gold):
    gold.columns = [clean_col(x) for x in gold.columns]
    gold["case_id"] = gold.apply(choose_case_id, axis=1)

HUMAN_COL = next((x for x in gold.columns if "human" in x or "manual" in x or "gold" in x), None) if len(gold) else None
print("Human label column:", HUMAN_COL)

if HUMAN_COL and "outcome_df" in globals():
    comp = outcome_df.merge(gold[["case_id", HUMAN_COL]], on="case_id", how="inner")
    print(f"Overlapping cases: {len(comp)}")
    if len(comp):
        agree = (comp["where"].str.lower().str[:4] == comp[HUMAN_COL].astype(str).str.lower().str[:4])
        print(f"Agreement: {agree.mean():.1%}")
        display(pd.crosstab(comp[HUMAN_COL], comp["where"]))
        print("\n--- disagreements to review ---")
        disagree = comp.loc[~agree, ["case_id", HUMAN_COL, "where", "how", "call_summary"]]
        display(disagree.head(15))
        save(disagree, "validation_disagreements")

# COMMAND ----------
audit_rows = []
for _, r in (valid.head(25).iterrows() if "valid" in globals() else []):
    for e in r["obj"].get("evidence", []):
        audit_rows.append({
            "case_id": r["case_id"],
            "bucket": r["phase1_bucket"],
            "field": e.get("field"),
            "value": e.get("value"),
            "quote": str(e.get("quote", ""))[:250],
            "rationale": str(e.get("rationale", ""))[:200],
        })

audit = pd.DataFrame(audit_rows)
display(audit.head(40))
save(audit, "evidence_audit_sample")

# COMMAND ----------
def system_setting(dispo: str) -> str:
    d = str(dispo).lower()
    if "self" in d:                                   return "home_no_care"
    if "urgent" in d:                                 return "urgent_care"
    if "virtual" in d or "telehealth" in d or "video" in d: return "virtual_care"
    if "primary" in d or "pcp" in d:                  return "primary_care"
    if "emergency" in d or "ambulance" in d or "bls" in d or "als" in d or "911" in d or " ed" in d or " er" in d:
        return "emergency_department"
    return "other_unknown"

if "outcome_df" in globals() and len(outcome_df):
    match = outcome_df.copy()
    match["system_setting"] = match[DISPO_COL].apply(system_setting)
    cmp = match[(match["system_setting"] != "other_unknown") & (match["actual_setting"] != "unknown")].copy()
    if len(cmp):
        cmp["agree"] = cmp["system_setting"] == cmp["actual_setting"]
        agree_pct = round(cmp["agree"].mean() * 100, 1)
        print(f"AI setting matches the recorded system code in {agree_pct}% of comparable calls "
              f"(n={len(cmp)}).")

        confusion = pd.crosstab(cmp["system_setting"], cmp["actual_setting"])
        display(confusion)

        summary_rows = (
            cmp.groupby("system_setting")["agree"].agg(["mean", "count"])
            .assign(**{"agree_pct": lambda d: (d["mean"] * 100).round(1)})
            .drop(columns="mean")
            .rename(columns={"count": "calls"})
            .reset_index()
        )
        display(summary_rows)
        save(summary_rows, "categorization_match")
        save(confusion.reset_index(), "categorization_confusion")

# COMMAND ----------
_all = analysis[DISPO_COL].fillna("").str.lower()
coded = {
    "Self-care": _all.str.contains("self", na=False).mean() * 100,
    "NMTARA 6 (triage not completed)": analysis["nmtara_level"].eq(6).mean() * 100
        if "nmtara_level" in analysis.columns else float("nan"),
    "1-5 ambulance override": analysis["is_amb_override"].mean() * 100
        if "is_amb_override" in analysis.columns else float("nan"),
}

abstracted = {}
if "outcome_df" in globals() and len(outcome_df):
    n_total = len(analysis)
    sc_all = outcome_df[outcome_df["phase1_bucket"] == "Self-care"]
    abstracted["Self-care"] = (sc_all["true_self_care"].sum() / n_total * 100) if n_total else float("nan")

    n6_all = outcome_df[outcome_df["phase1_bucket"] == "NMTARA 6 (triage not completed)"]
    n6_conf = n6_all[[f"triage_incomplete_reason.{k}" for k in BOOL_SECTIONS["triage_incomplete_reason"]
                      if f"triage_incomplete_reason.{k}" in n6_all.columns]].any(axis=1).sum() if len(n6_all) else 0
    abstracted["NMTARA 6 (triage not completed)"] = (n6_conf / n_total * 100) if n_total else float("nan")

    ov_all = outcome_df[outcome_df["phase1_bucket"] == "1-5 ambulance override"]
    ov_conf = ov_all[[f"decision_driver.{k}" for k in BOOL_SECTIONS["decision_driver"]
                      if f"decision_driver.{k}" in ov_all.columns]].any(axis=1).sum() if len(ov_all) else 0
    abstracted["1-5 ambulance override"] = (ov_conf / n_total * 100) if n_total else float("nan")

compare = pd.DataFrame({
    "bucket": list(coded.keys()),
    "coded_pct": [round(coded[k], 1) for k in coded],
    "llm_abstracted_pct": [round(abstracted.get(k, float("nan")), 1) for k in coded],
})
compare["gap_pts"] = (compare["llm_abstracted_pct"] - compare["coded_pct"]).round(1)
display(compare)
save(compare, "coded_vs_abstracted")
print("A large gap suggests the Logis codes may not capture every override or self-care case.")

# COMMAND ----------
def kpi_block(frame: pd.DataFrame, label: str) -> Dict[str, Any]:
    total = len(frame)
    if total == 0:
        return {}
    return {
        "segment": label,
        "calls": total,
        "self_care_reported_pct": round(frame["is_self_care"].mean() * 100, 1),
        "nmtara6_pct":            round(frame["is_nmtara6"].mean() * 100, 1),
        "amb_override_pct":       round(frame["is_amb_override"].mean() * 100, 1),
        "ambulance_diversion_pct": round((1 - frame["is_ambulance"].mean()) * 100, 1),
    }

MIN_CLIENT_CALLS = 100

blocks = [kpi_block(analysis, "National")]
if CLIENT_COL:
    for cl, grp in analysis.groupby(CLIENT_COL):
        if len(grp) >= MIN_CLIENT_CALLS:
            blocks.append(kpi_block(grp, cl))

baseline = pd.DataFrame([b for b in blocks if b])
display(baseline.sort_values("calls", ascending=False))
save(baseline, "phase1_baseline_by_client")

# COMMAND ----------
def register(obj_name: str, name: str):
    """Add a frame to the workbook registry by variable name; tolerates skipped cells."""
    obj = globals().get(obj_name)
    if obj is None:
        print(f"skip   {name} (not built)")
        return
    try:
        if len(obj) == 0:
            print(f"skip   {name} (empty)")
            return
        df = obj.to_frame() if isinstance(obj, pd.Series) else obj
        if isinstance(df.index, pd.MultiIndex) or df.index.name or not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
        RESULTS[name] = df
        print(f"kept   {len(df):>7,} rows -> {name}")
    except Exception as e:
        print(f"FAILED {name}: {e}")


register("inventory", "data_folder_inventory")
register("profile",   "field_profile")

if "outcome_df" in globals():
    outcome_cols = [c for c in [
        "case_id", "phase1_bucket", "nmtara_level", "protocol_trigger",
        DISPO_COL, "actual_setting", "stayed_home", "transport",
        "where", "how", "reason_documented",
    ] if c in outcome_df.columns]
    globals()["_outcome_export"] = outcome_df[outcome_cols]
    register("_outcome_export", "case_level_outcomes")

base_cols = [c for c in [
    "case_id", EPISODE_KEY, "phase1_bucket", "nmtara_level", NMTARA_COL,
    DISPO_COL, CLIENT_COL, DATE_COL, "note_len", "note_quality",
    "is_operational_only", "n_reason_tags",
] if c and c in analysis.columns]
globals()["_analysis_export"] = analysis[base_cols]
register("_analysis_export", "analysis_frame")

print("\n" + "=" * 70)
print(f"{len(RESULTS)} frames registered for the workbook")
print("=" * 70)
for k, v in RESULTS.items():
    print(f"  {k:32s} {len(v):>7,} rows")

# COMMAND ----------
WORKBOOK_TABS = [
    ("bucket_summary",              "Bucket Sizes"),
    ("self_care_breakdown",         "Self-Care Breakdown"),
    ("self_care_by_ride_initiator", "Self-Care by Ride"),
    ("nmtara6_reason_mix",          "NMTARA 6 Reasons"),
    ("override_reason_mix",         "Override Reasons"),
    ("override_driver_split",       "Override Driver Split"),
    ("override_reason_combinations","Override Combinations"),
    ("override_reasons_by_protocol","Override by Protocol"),
    ("diversion_insight",           "Diversion Insight"),
    ("bucket_trend_quarterly",      "Bucket Trend"),
    ("phase1_baseline_by_client",   "Baseline by Client"),
    ("note_quality_by_bucket",      "Note Coverage"),
    ("client_storylines",           "Client Storylines"),
    ("self_care_by_complaint",      "Self-Care by Complaint"),
    ("program_transport_bins",      "Program Transport Bins"),
    ("call_drop_focus",             "Call Drop Focus"),
    ("clinical_escalation_by_complaint","Clinical by Complaint"),
    ("override_top_complaints",     "Override Top Complaints"),
    ("mobility_drivers",            "Mobility Drivers"),
    ("override_opportunities",      "Override Opportunities"),
    ("data_coverage",               "Data Coverage"),
    ("categorization_match",        "AI vs System Match"),
    ("coded_vs_abstracted",         "Coded vs Abstracted"),
    ("reason_examples",             "Reason Examples"),
    ("validation_disagreements",    "Validation"),
    ("evidence_audit_sample",       "Evidence Sample"),
]

INTRO_TEXT = [
    ("Nurse Navigation \u2014 Phase 1 Analysis", "title"),
    (f"Run {RUN_ID}", "subtitle"),
    ("", "gap"),
    ("The goal", "h"),
    ("A new operating system goes live before year-end. It measures navigation differently than the "
     "current one, so reported numbers will shift. This analysis builds a historical baseline from the "
     "current system so the shift can be explained to each client before it appears in a dashboard.", "p"),
    ("The current system records WHERE a patient was sent but not WHY. The reason lives only in "
     "free-text nurse notes. This work reads those notes at scale to recover the why. All figures here "
     "are from the current system.", "p"),
    ("", "gap"),
    ("What was done", "h"),
    ("1. Cleaned the historical call data and separated real patient navigations from operational-only "
     "records (a nurse calling a facility or confirming a ride is not a navigation).", "p"),
    ("2. Sized the three buckets in focus: self-care, NMTARA 6 (triage not completed), "
     "and 1-5 ambulance overrides (triage did not call for an ambulance but one was sent).", "p"),
    ("3. Used an AI model to extract the documented reason for each decision, with a verbatim quote "
     "behind every finding so a clinician can check it.", "p"),
    ("4. Read from each note where the patient actually ended up, which exposes mislabeled self-care.", "p"),
    ("5. Grouped override reasons by who drove the decision (nurse, patient, or access).", "p"),
    ("6. Classified self-care by who arranged the ride: a patient-arranged ride is self-care, a "
     "program-arranged or unattributed ride is not.", "p"),
    ("7. Checked the AI categories against the codes the current system already records, and compared the "
     "bucket rates from the Logis codes against the rates read from the notes.", "p"),
    ("", "gap"),
    ("The headline question", "h"),
    ("Of everything currently labelled 'self-care', how much was truly no-care-needed versus a patient "
     "whose note shows they actually went somewhere (ED, urgent care)? See the 'Self-Care Breakdown' tab.", "p"),
    ("", "gap"),
    ("How to read this workbook", "h"),
    ("Each tab answers one question. Percentages are share-of-bucket unless noted. All figures are from "
     "the current system.", "p"),
    ("", "gap"),
    ("Important caveat", "h"),
    ("These numbers are provisional until the full-population data extract is confirmed with Rich and "
     "the reason categories are signed off by clinical leadership. This is a one-time historical "
     "baseline of the current system, not a model for scoring future calls.", "p"),
]


def write_intro(writer, sheet_name="Start Here"):
    wb = writer.book
    ws = wb.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = ws
    ws.hide_gridlines(2)
    ws.set_column("A:A", 3)
    ws.set_column("B:B", 100)
    fmt = {
        "title":    wb.add_format({"bold": True, "font_size": 18, "font_color": "#1F3864"}),
        "subtitle": wb.add_format({"italic": True, "font_size": 10, "font_color": "#808080"}),
        "h":        wb.add_format({"bold": True, "font_size": 12, "font_color": "#2E5496",
                                   "top": 1, "top_color": "#D9D9D9"}),
        "p":        wb.add_format({"font_size": 11, "text_wrap": True, "valign": "top"}),
    }
    r = 1
    for text, kind in INTRO_TEXT:
        if kind == "gap":
            r += 1
            continue
        if kind == "p":
            ws.set_row(r, 30)
        ws.write(r, 1, text, fmt.get(kind, fmt["p"]))
        r += 1


xlsx_path = os.path.join(OUT_DIR, f"Nurse_Nav_Phase1_Analysis_{RUN_ID}.xlsx")

try:
    import xlsxwriter
    engine = "xlsxwriter"
except ImportError:
    engine = "openpyxl"

tabs_written = 0
with pd.ExcelWriter(xlsx_path, engine=engine) as writer:

    if engine == "xlsxwriter":
        write_intro(writer)
    else:
        intro_df = pd.DataFrame({"Nurse Navigation \u2014 Phase 1 Analysis":
                                 [t for t, k in INTRO_TEXT if k != "gap"]})
        intro_df.to_excel(writer, sheet_name="Start Here", index=False)

    for stem, tab in WORKBOOK_TABS:
        df = RESULTS.get(stem)
        if df is None:
            print(f"skip   {tab:22s} (not produced this run)")
            continue
        if df.empty:
            print(f"skip   {tab:22s} (empty)")
            continue

        df.to_excel(writer, sheet_name=tab[:31], index=False,
                    startrow=1 if engine == "xlsxwriter" else 0)

        if engine == "xlsxwriter":
            wb, ws = writer.book, writer.sheets[tab[:31]]
            title_fmt  = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1F3864"})
            header_fmt = wb.add_format({"bold": True, "bg_color": "#2E5496",
                                        "font_color": "white", "border": 1})
            ws.write(0, 0, tab, title_fmt)
            for j, col in enumerate(df.columns):
                maxlen = df[col].astype(str).str.len().max() if len(df) else 10
                ws.write(1, j, str(col), header_fmt)
                ws.set_column(j, j, min(max(len(str(col)), int(maxlen)) + 2, 60))
            ws.freeze_panes(2, 0)

        tabs_written += 1
        print(f"added  {tab}")

print(f"\nWorkbook written: {os.path.basename(xlsx_path)}  ({tabs_written + 1} tabs)")
print("On disk:", os.path.exists(xlsx_path), "->", xlsx_path)

for _old in glob.glob(os.path.join(OUT_DIR, "Nurse_Nav_Phase1_Analysis_*.xlsx")):
    if os.path.abspath(_old) != os.path.abspath(xlsx_path):
        try:
            os.remove(_old)
            print("removed older workbook:", os.path.basename(_old))
        except Exception as e:
            print("could not remove", os.path.basename(_old), "-", e)

print("Only one timestamped Excel is kept in the results folder; no CSVs are saved.")
