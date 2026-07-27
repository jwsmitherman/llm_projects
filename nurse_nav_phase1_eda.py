Goal: Use AI to read historical nurse notes and explain why each navigation decision was made. Today the system records where a patient was sent, but not the reason. We need that baseline so we can prep clients for how the new system will change their reported numbers before it goes live.

Next steps:

Get the full data pull from Rich, then size the three types of calls that will shift most at go-live:
Self-care — patients told to care for themselves at home (we suspect many were mislabeled and actually went to urgent care or the ER)
Incomplete triage — calls where the nurse assessment was never finished, and we don't yet know why
Ambulance overrides — cases where the assessment didn't call for an ambulance, but one was sent anyway
Deliver a client-ready workbook answering the headline question: how much of today's "self-care" was truly no care needed vs. patients who went elsewhere on their own
Validate the results against the reporting team's hand-coded set, and get clinical sign-off on the reason categories before anything goes to a client


# Databricks notebook source
# # Nurse Navigation - Phase 1 Exploratory Data Analysis
#
# **Purpose:** establish a historical baseline for the three buckets that will move most when the new
# operating system goes live: **Self-Care**, **NMTARA 6 (triage not completed)**, and
# **NMTARA 1-5 ambulance overrides**.
#
# **Method:** the existing `nurse_nav_eda.py` pipeline (clean -> LLM extraction -> rules engine -> disposition LLM
# -> rationale rollup) is reused, but pointed at the three buckets and extended with the exploratory steps
# needed before we can trust the output at scale.
#
# **Notebook flow**
#
# | Section | Question it answers |
# |---|---|
# | 1 | What data do we actually have, and over what period? |
# | 2 | What is a "call"? (work set ID de-duplication) |
# | 3 | How big are the three buckets, nationally and by client? |
# | 4 | Are the notes good enough to answer *why*? |
# | 5 | What reasons show up, before we spend money on an LLM? |
# | 6 | Structured extraction of the *why* from the notes |
# | 7 | Recasting history into the new three-decision framework |
# | 8 | Do we agree with the humans? (validation) |
# | 9 | What will the client-facing numbers look like after go-live? |
#
# > Cells that call the LLM endpoint are marked **[COST]**. Run them on a sample first.

# COMMAND ----------

# ## 0. Setup and configuration
#
# Everything environment-specific lives here so the rest of the notebook is portable between the
# Databricks volume and a local copy.

# COMMAND ----------

import os, re, json, warnings
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 300)

# ---------------------------------------------------------------
# CONFIG: Databricks workspace files
#   Workspace > Users > josh.smitherman@gmr.net > nurse_nav > data
# ---------------------------------------------------------------
DATA_DIR = "/Workspace/Users/josh.smitherman@gmr.net/nurse_nav/data"
OUT_DIR  = "/Workspace/Users/josh.smitherman@gmr.net/nurse_nav/results"

SAMPLE_N    = 500      # rows per run while iterating
RANDOM_SEED = 42
LLM_MODEL   = "databricks-gpt-oss-120b"

os.makedirs(OUT_DIR, exist_ok=True)

# Stamp every run so prior runs aren't overwritten
RUN_ID = pd.Timestamp.now().strftime("%Y%m%d_%H%M")


def save(df: pd.DataFrame, name: str, stamp: bool = True) -> str:
    """Write a result to the results folder and report where it landed."""
    fname = f"{name}_{RUN_ID}.csv" if stamp else f"{name}.csv"
    path = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"saved  {len(df):>7,} rows -> {fname}")
    return path


print("RUN_ID:", RUN_ID)
print("results ->", OUT_DIR)


# ---------------------------------------------------------------
# File discovery
#
# Filenames are NOT hardcoded. The folder contains a mix of .csv and .xlsx,
# plus timestamped duplicates that Databricks creates on re-upload
# (e.g. "logis_notes 2026-07-21 12:21:35.xlsx"). We resolve each logical
# dataset to a single newest file so the notebook keeps working as the
# folder changes.
# ---------------------------------------------------------------
import glob

READABLE_EXT = (".csv", ".xlsx", ".xls", ".parquet", ".tsv")

# logical name -> regex for the filename stem (lowercased)
DATASET_PATTERNS = {
    "logis_notes":      r"^logis[_ ]notes",
    "ed_calls_prepped": r"^ed[_ ]calls[_ ]nmtara[_ ]6[_ ]removed[_ ]prepped",
    "ed_calls_removed": r"^ed[_ ]calls[_ ]nmtara[_ ]6[_ ]removed(?![_ ]prepped)",
    "ed_calls_sample":  r"^ed[_ ]calls[_ ]sample(?![_ ]100)",
    "ed_calls_100":     r"^ed[_ ]calls[_ ]sample[_ ]100",
    "gold":             r"^llm[_ ]validation[_ ]set[_ ]51",
    "second_llm_set_1": r"^set[_ ]for[_ ]2nd[_ ]llm",
    "second_llm_set_2": r"^second[_ ]set[_ ]for[_ ]2nd[_ ]llm",
}

# preference order when the same dataset exists in multiple formats
EXT_PREFERENCE = [".parquet", ".csv", ".xlsx", ".xls", ".tsv"]


def discover(data_dir: str) -> Dict[str, str]:
    files = [f for f in glob.glob(os.path.join(data_dir, "*"))
             if f.lower().endswith(READABLE_EXT)]

    resolved = {}
    for name, pattern in DATASET_PATTERNS.items():
        hits = [
            f for f in files
            if re.match(pattern, os.path.splitext(os.path.basename(f))[0].lower())
        ]
        if not hits:
            continue
        # sort newest first, then by preferred format
        hits.sort(key=lambda f: (
            -os.path.getmtime(f),
            EXT_PREFERENCE.index(os.path.splitext(f)[1].lower())
            if os.path.splitext(f)[1].lower() in EXT_PREFERENCE else 99,
        ))
        resolved[name] = hits[0]

    return resolved


PATHS = discover(DATA_DIR)
PATHS["out_dir"] = OUT_DIR

# Primary input for Phase 1. Needs ALL navigations; the ED_calls_* sets have
# NMTARA 6 removed, which is one of the three buckets we must explain.
PRIMARY_INPUT = PATHS.get("logis_notes") or PATHS.get("ed_calls_prepped")

for k, v in PATHS.items():
    print(f"{k:20s} -> {os.path.basename(v) if k != 'out_dir' else v}")
print(f"\nPRIMARY_INPUT       -> {os.path.basename(PRIMARY_INPUT) if PRIMARY_INPUT else 'NOT FOUND'}")

# COMMAND ----------

# ### 0.1 What is actually in the data folder
#
# Lists every file present, flags timestamped duplicates, and shows which one each logical dataset resolved to.
#
# > **Two things to watch.**
# >
# > 1. **Duplicate uploads.** Databricks appends a timestamp when a file is re-uploaded, so
# >    `logis_notes.xlsx` and `logis_notes 2026-07-21 12:21:35.xlsx` can both exist. The discovery step picks
# >    the newest - verify below that it picked the one you intend.
# > 2. **Scope gap (blocking).** The `ED_calls_*` sets are from the earlier ED-appropriateness run and have
# >    **NMTARA 6 removed** - one of the three Phase 1 buckets. They are also ED-focused, so self-care and
# >    1-5 ambulance overrides are likely under-represented. Phase 1 needs a full-population extract from Rich.
# >    Until then, treat bucket sizes as provisional.

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

# Sheet/column preview for each candidate input to pick one to build on
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


for key in ["logis_notes", "ed_calls_prepped", "gold"]:
    if key in PATHS:
        peek(PATHS[key])

# COMMAND ----------

# ### 0.2 Verify we can actually write to the results folder
#
# Runs a throwaway write/read/delete before any analysis. If output is going to fail, it fails here in two
# seconds rather than after an expensive LLM run.

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
    print("Falling back to local disk. Files will be copied to the results folder at the end.")
    OUT_DIR = "/tmp/nurse_nav_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"OUT_DIR now: {OUT_DIR}")

# COMMAND ----------

# ## 1. Load and standardize
#
# Reuses the column-cleaning and protocol-trigger parsing from `nurse_nav_eda.py` so this notebook and the
# production script stay in sync. Do not re-implement these in the notebook - import or copy verbatim.
#
# `read_any()` dispatches on file extension, so nothing downstream cares whether a dataset arrived as
# `.csv`, `.xlsx`, or `.parquet`.

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

    if "nmtara_notes" in df.columns:
        df["protocol_trigger"] = df["nmtara_notes"].apply(parse_protocol_trigger)

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

# ### 1.2 Column resolution - the fix for `KeyError`
#
# **This is what broke the first run.** The notebook was written against expected column names
# (`transaction_create_date_time_eastern`, `nurse_notes`, `nmtara_response`). The actual export uses different
# ones, so Section 2 died on a `KeyError` and nothing downstream ever ran - which is why the results folder
# stayed empty.
#
# Nothing below this cell references a column name directly. Everything goes through `COLS`, which is resolved
# from what is actually in the file. **Check the printout and override anything it gets wrong** - that is the
# one manual step, and it takes thirty seconds.

# COMMAND ----------

def find_col(df: pd.DataFrame, exact: List[str], contains: List[str] = None):
    """Resolve a logical column to a real one: exact match first, then substring."""
    for c in exact:
        if c in df.columns:
            return c
    for pat in (contains or []):
        hits = [c for c in df.columns if pat in c]
        if hits:
            # shortest match is usually the least-qualified name
            return sorted(hits, key=len)[0]
    return None


COLS = {
    "notes": find_col(calls,
        ["nurse_notes", "notes", "note_text", "call_notes", "nmtara_notes"],
        ["note"]),
    "date": find_col(calls,
        ["transaction_create_date_time_eastern", "transaction_create_date_time",
         "create_date_time", "call_date", "transaction_date"],
        ["create_date", "date_time", "_date"]),
    "nmtara": find_col(calls,
        ["nmtara_response", "nmtara", "nmtara_level", "natmara", "mtara"],
        ["nmtara", "tara"]),
    "dispo": find_col(calls,
        ["documented_disposition", "transaction_response_names", "disposition",
         "final_disposition", "outcome"],
        ["disposition", "response_name"]),
    "client": find_col(calls,
        ["client_name", "client", "market", "agency", "customer"],
        ["client", "market", "agency"]),
    "episode": find_col(calls,
        ["external_reference_number", "incident_id", "incident_number",
         "patient_id", "encounter_id"],
        ["external_reference", "incident"]),
    "workset": find_col(calls,
        ["work_set_id", "workset_id", "worksetid"],
        ["work_set", "workset"]),
    "protocol": find_col(calls,
        ["protocol_trigger", "protocol", "chief_complaint", "clinical_pathway"],
        ["protocol", "complaint", "pathway"]),
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

# **Override anything wrong here**, then re-run from this point. `notes` and `dispo` are the two that matter
# most - the whole analysis rests on them.

# COMMAND ----------

# Manual overrides: set a real column name to force it, leave commented to auto-resolve
OVERRIDES = {
    # "notes":  "nurse_notes",
    # "date":   "call_create_datetime",
    # "nmtara": "nmtara_response",
    # "dispo":  "transaction_response_names",
    # "client": "market",
    # "episode": "external_reference_number",
}
COLS.update({k: v for k, v in OVERRIDES.items() if v})

# Bind to the names used throughout the rest of the notebook
NOTES_COL   = COLS["notes"]
DATE_COL    = COLS["date"]
NMTARA_COL  = COLS["nmtara"]
DISPO_COL   = COLS["dispo"]
CLIENT_COL  = COLS["client"]
EPISODE_KEY = COLS["episode"]
PROTO_COL   = COLS["protocol"]

# Hard requirements: fail here, not 200 cells later
missing = [k for k in ["notes", "dispo"] if not COLS[k]]
if missing:
    raise ValueError(
        f"Cannot proceed without: {missing}. "
        f"Set them in OVERRIDES above. Available columns: {sorted(calls.columns)}"
    )

# Soft requirements: skip related sections if missing
for k in ["date", "nmtara", "client", "episode"]:
    if not COLS[k]:
        print(f"WARNING: no '{k}' column found -- related sections will be skipped")

# Normalize types once, up front
if DATE_COL:
    calls[DATE_COL] = pd.to_datetime(calls[DATE_COL], errors="coerce")
    print(f"\nDate range: {calls[DATE_COL].min()}  ->  {calls[DATE_COL].max()}")

calls[NOTES_COL] = calls[NOTES_COL].astype("string")
print(f"Notes populated: {calls[NOTES_COL].notna().mean():.1%}")

# COMMAND ----------

# ### 1.3 Field profile

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

# ## 2. Work set ID de-duplication - what counts as a "call"?
#
# Flagged on the kickoff call: a **work set ID is created every time a nurse picks up the phone**. One patient
# episode can generate several work set IDs within an hour, and some of them are *operational only* - the nurse
# calling a facility, confirming a ride, updating a record - with the patient not on the line.
#
# If we count those as navigations, every denominator in the baseline is wrong. This section builds an
# **episode key** and an **operational vs. clinical** flag before anything is measured.
#
# *Open item for Rich: confirm whether `external_reference_number` is the correct episode-level key.*

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

# ### 2.1 Operational vs. clinical notes
#
# Noah's point: a large share of note text is a record of *what the nurse did* rather than *what the patient
# needed*. A first-pass heuristic separates the two so we know how much genuinely clinical text we have to
# work with. Refine the term lists after eyeballing the samples printed below.

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

# Analysis frame: one clinical work set ID per episode, operational rows dropped
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

# ## 3. Sizing the three buckets
#
# Establishes the "before" picture. Anisa's expectation is Self-Care around **18-20%** of navigations - this
# section confirms it, splits it by client, and shows whether the mix has drifted over time.
#
# Bucket definitions:
#
# - **Self-Care** - documented disposition is self-care
# - **NMTARA 6** - triage not completed (the acuity is unknown, not low)
# - **NMTARA 1-5 ambulance override** - triage did not call for an ambulance, but one was sent

# COMMAND ----------

def nmtara_level(x):
    m = re.search(r"(\d)", str(x))
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

# Bucket mix over time: is history stable enough for one baseline?
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

# Client-level view: each client needs its own story
if CLIENT_COL:
    by_client = (
        pd.crosstab(analysis[CLIENT_COL], analysis["phase1_bucket"], normalize="index") * 100
    ).round(1)
    by_client["n_calls"] = analysis[CLIENT_COL].value_counts()
    display(by_client.sort_values("n_calls", ascending=False).head(20))
else:
    print("No client/market column found -- request from Rich.")

# COMMAND ----------

# ## 4. Can the notes answer *why*?
#
# Anisa's estimate: roughly **25% of notes will be unusable**, and 50-75% coverage is enough to act on. This
# section tests that estimate rather than assuming it, and does so *per bucket* - coverage may be much worse
# for NMTARA 6 (where the call ended early) than for overrides.
#
# The output is a go/no-go input: if a bucket has thin notes, we tell the business that up front instead of
# discovering it in the results.

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

# Has documentation quality improved? (Noah: "pretty good in the last year or two")
if DATE_COL:
    ax = (analysis.dropna(subset=[DATE_COL]).set_index(DATE_COL)
                  .groupby(pd.Grouper(freq="QS"))["note_len"]
                  .median()
                  .plot(figsize=(12, 3), marker="o"))
    ax.set_title("Median nurse note length by quarter")
    ax.set_ylabel("characters")
    plt.tight_layout(); plt.show()

# COMMAND ----------

# ### 4.1 Read the notes
#
# No substitute for this. Pull a stratified sample from each bucket and read them before designing the
# extraction schema - this is where the real reason codes come from.

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

# ## 5. Cheap signal first - keyword pre-pass
#
# Before spending LLM budget on 200k notes, a lexicon pass shows whether the reasons we expect are even
# present in the text, and roughly how often. It also produces the candidate reason-code list that the
# extraction schema in Section 6 is built around.
#
# The lexicons below are seeded from the kickoff discussion (no provider available, patient refusal, no
# after-hours option, mobility/catheter, self-transport). Expand them after reading the samples above.

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

# **How to read the table above.** High keyword coverage in a bucket means the extraction step will have
# something to work with. Low coverage means either our vocabulary is wrong or the reason genuinely is not
# documented - that distinction is worth resolving manually before scaling.

# COMMAND ----------

# ## 6. Structured extraction of the *why*  **[COST]**
#
# This is the existing Step 2 pattern from `nurse_nav_eda.py` - strict-JSON extraction with a mandatory
# evidence quote for every `true` flag - retargeted from ED-appropriateness to the three Phase 1 buckets.
#
# Two design choices carried over deliberately:
#
# 1. **Extraction and judgement stay separate.** The model pulls facts; a deterministic rules engine
#    (Section 7) makes the call. Business rules can then change without re-running the LLM.
# 2. **Every `true` needs a verbatim quote.** This makes the output auditable for clinical review - Dr. Stites
#    or Dr. Troutman can check the evidence rather than trusting a score.

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
    "arranged_by_program": boolean
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
    "nurse_ended_for_acuity": boolean,
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
        "nurse_ended_for_acuity", "caller_was_not_patient",
    ],
}

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

    # every true flag must carry an evidence quote
    ev_fields = {e.get("field") for e in obj.get("evidence", []) if isinstance(e, dict)}
    for section, keys in BOOL_SECTIONS.items():
        for k in keys:
            if obj[section][k] and f"{section}.{k}" not in ev_fields:
                return False, obj, f"true flag without evidence: {section}.{k}"

    return True, obj, "OK"

# COMMAND ----------

# Run on a sample first
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

# Persist immediately: this is the expensive step, don't re-run by accident
save(extractions, "llm_extractions_raw")

# COMMAND ----------

# ### 6.1 Extraction quality gate
#
# Parse-failure rate is the first quality signal. Anything below roughly 95% valid means the prompt or the
# schema needs work before scaling - cheaper to find here than after 200k calls.

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
valid["reason_documented"] = valid["obj"].apply(lambda o: bool(o.get("documentation", {}).get("reason_documented", False)))

print("Reason documented rate by bucket:")
print(valid.groupby("phase1_bucket")["reason_documented"].mean().mul(100).round(1))

# COMMAND ----------

# ## 7. Recasting history into the new three-decision framework
#
# The heart of the change-management story. Today's outcome blends *where* and *how* into one label
# ("rideshare to ER", "self-care"). The new system splits every navigation into three separate answers:
#
# 1. **How fast** does the patient need to be seen (acuity)
# 2. **Where** do they need to be seen (care setting)
# 3. **How** do they get there (transport)
#
# This section applies that split to historical calls so the "before" and "after" numbers are measured the
# same way. The rules are deterministic and live in one place - clinical leadership can adjust them without
# touching the model.

# COMMAND ----------

def recast(row) -> Dict[str, str]:
    """Map a historical call onto the new three-decision framework."""
    setting   = row["actual_setting"]
    transport = row["transport"]

    # WHERE
    where = {
        "emergency_department": "Emergency Department",
        "urgent_care":          "Urgent Care",
        "virtual_care":         "Virtual Care",
        "primary_care":         "Primary Care",
        "home_no_care":         "No care setting (true self-care)",
    }.get(setting, "Unknown")

    # HOW
    how = {
        "ambulance_als":  "Ambulance (ALS)",
        "ambulance_bls":  "Ambulance (BLS)",
        "rideshare":      "Program-arranged rideshare",
        "self_or_family": "Self / family transport",
        "none":           "No transport needed",
    }.get(transport, "Unknown")

    # HOW FAST: proxy from NMTARA until the new acuity field exists
    lvl = row.get("nmtara_level", np.nan)
    if lvl == 0:            how_fast = "Immediate"
    elif lvl in (1, 2):     how_fast = "Within hours"
    elif lvl in (3, 4, 5):  how_fast = "Same day / routine"
    else:                   how_fast = "Not determined"

    return {"where": where, "how": how, "how_fast": how_fast}


if "valid" not in globals() or len(valid) == 0:
    raise RuntimeError("No validated extractions available -- run Section 6 first.")

recast_df = valid.join(pd.DataFrame(valid.apply(recast, axis=1).tolist(), index=valid.index))
display(pd.crosstab(recast_df["where"], recast_df["how"]))

# COMMAND ----------

# ### 7.1 The self-care question
#
# The single most important number in Phase 1: of everything currently labeled **self-care**, how much was
# genuinely *no care needed*, and how much was a patient going somewhere under their own steam?
#
# This determines whether the drop in self-care at go-live is a good-news story (it becomes urgent care and
# virtual care) or one that needs explaining (it becomes ED).

# COMMAND ----------

sc = recast_df[recast_df["phase1_bucket"] == "Self-care"] if "recast_df" in globals() else pd.DataFrame()

if len(sc):
    breakdown = (
        sc["where"].value_counts(normalize=True).mul(100).round(1)
        .rename("pct_of_self_care").to_frame()
    )
    breakdown["calls"] = sc["where"].value_counts()
    display(breakdown)

    ax = breakdown["pct_of_self_care"].plot(kind="barh", figsize=(9, 4))
    ax.set_title("What today's 'self-care' bucket actually contains")
    ax.set_xlabel("% of self-care calls")
    plt.tight_layout(); plt.show()

    true_sc = (sc["where"] == "No care setting (true self-care)").mean()
    print(f"True self-care share: {true_sc:.1%}")
    print(f"Implied restated self-care rate: "
          f"{true_sc * analysis['is_self_care'].mean():.1%} of all navigations "
          f"(vs {analysis['is_self_care'].mean():.1%} reported today)")

# COMMAND ----------

# ### 7.2 NMTARA 6 and ambulance overrides - reason mix
#
# Produces the reason-code distribution for the two remaining buckets. These map directly onto the structured
# override reason codes being built into the new system, so the historical mix becomes the expected baseline
# for the new codes.

# COMMAND ----------

n6 = recast_df[recast_df["phase1_bucket"] == "NMTARA 6 (triage not completed)"] if "recast_df" in globals() else pd.DataFrame()
if len(n6):
    cols = [f"triage_incomplete_reason.{k}" for k in BOOL_SECTIONS["triage_incomplete_reason"]]
    mix = n6[cols].mean().mul(100).round(1).sort_values(ascending=False)
    mix.index = [i.split(".")[-1].replace("_", " ").title() for i in mix.index]
    n6_mix = mix.rename("% of NMTARA 6 calls").to_frame()
    display(n6_mix)
    save(n6_mix.reset_index().rename(columns={"index": "reason"}), "nmtara6_reason_mix")

ov = recast_df[recast_df["phase1_bucket"] == "1-5 ambulance override"] if "recast_df" in globals() else pd.DataFrame()
if len(ov):
    cols = [f"decision_driver.{k}" for k in BOOL_SECTIONS["decision_driver"]]
    mix = ov[cols].mean().mul(100).round(1).sort_values(ascending=False)
    mix.index = [i.split(".")[-1].replace("_", " ").title() for i in mix.index]
    ov_mix = mix.rename("% of override calls").to_frame()
    display(ov_mix)
    save(ov_mix.reset_index().rename(columns={"index": "reason"}), "override_reason_mix")

# COMMAND ----------

# Override reasons by protocol: tests the Riverside catheter pattern
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

# ## 8. Validation against human coding
#
# The manual coding Anisa's team already does is the benchmark. Before any of this reaches a client, we need
# to show agreement on a set the humans coded themselves.
#
# Target: agreement high enough that the remaining disagreements are worth reviewing individually, and every
# extracted flag is traceable to a quote a clinician can check.

# COMMAND ----------

if "gold" not in PATHS:
    print("No validation set found in the data folder -- skipping Section 8.")
    gold = pd.DataFrame()
else:
    gold = read_any(PATHS["gold"])
if len(gold):
    gold.columns = [clean_col(x) for x in gold.columns]
    gold["case_id"] = gold.apply(choose_case_id, axis=1)

# Expected: a human-assigned setting/reason column. Adjust name once the gold file is confirmed.
HUMAN_COL = next((x for x in gold.columns if "human" in x or "manual" in x or "gold" in x), None) if len(gold) else None
print("Human label column:", HUMAN_COL)

if HUMAN_COL and "recast_df" in globals():
    comp = recast_df.merge(gold[["case_id", HUMAN_COL]], on="case_id", how="inner")
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

# ### 8.1 Evidence audit
#
# A sample of extracted flags with their supporting quotes, formatted for clinical review. This is the
# artifact to walk Dr. Stites through - not the accuracy percentage.

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

# ## 9. Before-and-after view for clients
#
# Pulls everything together into the table the account teams will actually use: today's reported numbers next
# to the restated numbers, at national and client level, with the delta and a plain-language driver.
#
# This is the deliverable that lets us tell a client "your self-care rate is going to drop by X points, and
# here is exactly where those calls went" *before* they see it in a dashboard.

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

MIN_CLIENT_CALLS = 100   # hide clients too small for stable percentages

blocks = [kpi_block(analysis, "National")]
if CLIENT_COL:
    for cl, grp in analysis.groupby(CLIENT_COL):
        if len(grp) >= MIN_CLIENT_CALLS:
            blocks.append(kpi_block(grp, cl))

baseline = pd.DataFrame([b for b in blocks if b])
display(baseline.sort_values("calls", ascending=False))
save(baseline, "phase1_baseline_by_client")

# COMMAND ----------

# Restate self-care: apply the sampled recast rate to full volume
if len(sc):
    true_sc_rate = (sc["where"] == "No care setting (true self-care)").mean()

    restated = baseline.copy()
    restated["self_care_restated_pct"] = (restated["self_care_reported_pct"] * true_sc_rate).round(1)
    restated["self_care_delta_pts"]    = (
        restated["self_care_restated_pct"] - restated["self_care_reported_pct"]
    ).round(1)

    display(restated[[
        "segment", "calls", "self_care_reported_pct",
        "self_care_restated_pct", "self_care_delta_pts"
    ]].sort_values("self_care_delta_pts"))

    save(restated, "self_care_restatement")
    print("\nNOTE: applies one national recast rate to every client. "
          "Once the full extraction runs, compute this per client -- "
          "the mix almost certainly varies by market.")

# COMMAND ----------

# ### 9.1 Export everything to the results folder
#
# Writes every artifact this notebook produced to
# `Workspace > Users > josh.smitherman@gmr.net > nurse_nav > results`, each stamped with the run ID so prior
# runs are never overwritten and any number can be traced back to the run that produced it.
#
# | File | What it is | Who it's for |
# |---|---|---|
# | `bucket_summary` | Size of each of the three buckets | Anisa - the "before" picture |
# | `bucket_trend_quarterly` | Bucket mix over time | Anisa - is history stable enough to use as one baseline |
# | `note_quality_by_bucket` | Usable-note share per bucket | Internal - feasibility / coverage caveat |
# | `reason_keyword_coverage` | Keyword pre-pass results | Internal - sanity check before LLM spend |
# | `llm_extractions_raw` | Raw JSON extractions | Internal - the expensive artifact, reusable |
# | `case_level_recast` | Every case mapped to how fast / where / how | Rich - feeds Power BI |
# | `self_care_breakdown` | What self-care actually contained | **Client-facing headline** |
# | `nmtara6_reason_mix` | Why triage wasn't completed | Anisa / new-system reason codes |
# | `override_reason_mix` | Why we overrode to ambulance | Anisa / new-system reason codes |
# | `override_reasons_by_protocol` | Override drivers by chief complaint | Clinical - Dr. Stites / Dr. Troutman |
# | `validation_disagreements` | Cases where we differ from human coding | Anisa - trust building |
# | `evidence_audit_sample` | Extracted flags with supporting quotes | Clinical sign-off |
# | `phase1_baseline_by_client` | KPI baseline, national and per client | Account teams |
# | `self_care_restatement` | Reported vs restated self-care | **Account teams - the change-management number** |

# COMMAND ----------

written = []

def try_save(obj_name: str, name: str):
    """Save a frame by variable name -- tolerates cells that were skipped or failed."""
    obj = globals().get(obj_name)
    if obj is None:
        print(f"skip   {name} (not built -- upstream cell skipped or failed)")
        return
    try:
        if len(obj) == 0:
            print(f"skip   {name} (empty)")
            return
        df = obj.to_frame() if isinstance(obj, pd.Series) else obj
        if isinstance(df.index, pd.MultiIndex) or df.index.name or not isinstance(
            df.index, pd.RangeIndex
        ):
            df = df.reset_index()
        written.append((name, save(df, name), len(df)))
    except Exception as e:
        print(f"FAILED {name}: {e}")


try_save("inventory",  "data_folder_inventory")
try_save("summary",    "bucket_summary")
try_save("trend_pct",  "bucket_trend_quarterly")
try_save("qual",       "note_quality_by_bucket")
try_save("heat",       "reason_keyword_coverage")
try_save("profile",    "field_profile")

# case-level recast: the join key back into Power BI
if "recast_df" in globals():
    recast_cols = [c for c in [
        "case_id", "phase1_bucket", "nmtara_level", "protocol_trigger",
        DISPO_COL, "actual_setting", "stayed_home", "transport",
        "where", "how", "how_fast", "reason_documented",
    ] if c in recast_df.columns]
    globals()["_recast_export"] = recast_df[recast_cols]
    try_save("_recast_export", "case_level_recast")

try_save("breakdown", "self_care_breakdown")

# always export the analysis frame, even if the LLM steps never ran
base_cols = [c for c in [
    "case_id", EPISODE_KEY, "phase1_bucket", "nmtara_level", NMTARA_COL,
    DISPO_COL, CLIENT_COL, DATE_COL, "note_len", "note_quality",
    "is_operational_only", "n_reason_tags",
] if c and c in analysis.columns]
globals()["_analysis_export"] = analysis[base_cols]
try_save("_analysis_export", "analysis_frame")

print("\n" + "=" * 70)
print(f"{len(written)} files written to {OUT_DIR}")
print("=" * 70)
display(pd.DataFrame(written, columns=["artifact", "path", "rows"]))

# Confirm files are on disk; don't trust the write blind
on_disk = sorted(glob.glob(os.path.join(OUT_DIR, f"*{RUN_ID}*")))
print(f"\nFiles present in results folder for run {RUN_ID}: {len(on_disk)}")
for f in on_disk:
    print(f"  {os.path.basename(f):55s} {os.path.getsize(f)/1024:>8,.1f} KB")

if not on_disk:
    print("\nNOTHING WAS WRITTEN. Check section 0.2 -- the folder may not be writable "
          "from this cluster, or an earlier cell failed and these frames never got built.")

# COMMAND ----------

# ### 9.2 Consolidate into a single Excel workbook
#
# The CSVs above are the machine-readable outputs - they feed Power BI and keep each artifact reproducible.
# For anyone who just wants to *read the analysis*, this cell rolls the business-relevant ones into one
# formatted `.xlsx` with a plain-language intro tab.
#
# Only the tabs that help someone understand the findings go in. The raw extraction JSON, the field profile,
# the folder inventory, and the keyword sanity-check stay as CSVs and are left out to keep the workbook clean.
#
# | Tab | What it answers |
# |---|---|
# | **Start Here** | Goal, method, how to read the workbook |
# | **Bucket Sizes** | How big each of the three buckets is |
# | **Self-Care Restated** | Reported vs restated self-care - the headline |
# | **Self-Care Detail** | What today's self-care bucket actually contained |
# | **NMTARA 6 Reasons** | Why triage wasn't completed |
# | **Override Reasons** | Why we overrode to an ambulance |
# | **Override by Protocol** | Override drivers by chief complaint |
# | **Baseline by Client** | KPI baseline, national and per client |
# | **Note Coverage** | Share of usable notes per bucket (feasibility) |
# | **Validation** | Where we differ from human coding |
# | **Evidence Sample** | Extracted flags with supporting quotes |

# COMMAND ----------

# CSVs that go into the reading workbook, in tab order.
# (csv stem without RUN_ID/extension, tab name <= 31 chars)
WORKBOOK_TABS = [
    ("bucket_summary",              "Bucket Sizes"),
    ("self_care_restatement",       "Self-Care Restated"),
    ("self_care_breakdown",         "Self-Care Detail"),
    ("nmtara6_reason_mix",          "NMTARA 6 Reasons"),
    ("override_reason_mix",         "Override Reasons"),
    ("override_reasons_by_protocol","Override by Protocol"),
    ("bucket_trend_quarterly",      "Bucket Trend"),
    ("phase1_baseline_by_client",   "Baseline by Client"),
    ("note_quality_by_bucket",      "Note Coverage"),
    ("validation_disagreements",    "Validation"),
    ("evidence_audit_sample",       "Evidence Sample"),
]

INTRO_TEXT = [
    ("Nurse Navigation \u2014 Phase 1 Analysis", "title"),
    (f"Run {RUN_ID}", "subtitle"),
    ("", "gap"),
    ("The goal", "h"),
    ("A new operating system goes live before year-end. When it does, our reported navigation "
     "numbers will move, because the new system measures decisions differently than the current one. "
     "This analysis builds the historical baseline that lets us explain the shift to each client "
     "before they see it in a dashboard.", "p"),
    ("Today the system records WHERE a patient was sent but not WHY. The reasoning lives only in "
     "free-text nurse notes. This work reads those notes at scale to recover the why.", "p"),
    ("", "gap"),
    ("What was done", "h"),
    ("1. Cleaned the historical call data and separated real patient navigations from operational-only "
     "records (a nurse calling a facility or confirming a ride is not a navigation).", "p"),
    ("2. Sized the three buckets that move most at go-live: self-care, NMTARA 6 (triage not completed), "
     "and 1-5 ambulance overrides (triage did not call for an ambulance but one was sent).", "p"),
    ("3. Used an AI model to extract the documented reason for each decision, with a verbatim quote "
     "behind every finding so a clinician can check it.", "p"),
    ("4. Recast each historical call into the new system's three separate decisions \u2014 how fast, "
     "where, and how the patient gets there \u2014 so before and after are measured the same way.", "p"),
    ("5. Compared results against the cases the reporting team coded by hand.", "p"),
    ("", "gap"),
    ("The headline question", "h"),
    ("Of everything currently labelled 'self-care' (~18-20% of navigations), how much was truly "
     "no-care-needed versus a patient who went somewhere under their own steam? That split decides "
     "whether the drop in self-care at go-live is a good-news story or one we need to prepare clients "
     "for. See the 'Self-Care Restated' tab.", "p"),
    ("", "gap"),
    ("How to read this workbook", "h"),
    ("Each tab answers one question. Percentages are share-of-bucket unless noted. 'Reported' means "
     "today's number; 'restated' means the same calls counted the new way.", "p"),
    ("", "gap"),
    ("Important caveat", "h"),
    ("These numbers are provisional until the full-population data extract is confirmed with Rich and "
     "the reason categories are signed off by clinical leadership. This is a one-time historical "
     "baseline, not a model for scoring future calls.", "p"),
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
    import xlsxwriter  # noqa
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
        matches = glob.glob(os.path.join(OUT_DIR, f"{stem}_{RUN_ID}.csv"))
        if not matches:
            print(f"skip   {tab:22s} (no {stem} csv this run)")
            continue
        try:
            df = pd.read_csv(matches[0])
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
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
present = glob.glob(os.path.join(OUT_DIR, f"Nurse_Nav_Phase1_Analysis_{RUN_ID}.xlsx"))
print("On disk:", bool(present), present[0] if present else "")

# COMMAND ----------

# ## 10. Findings and open items
#
# *Fill in after the first full run. Keep it to what the business needs to decide on.*
#
# ### What the data supports
# - Bucket sizes, national and by client - Section 3
# - Note usability by bucket - Section 4
# - Reason mix for each bucket - Section 7
# - Restated self-care baseline - Section 9
#
# ### Open items
# - [ ] **Rich - data pull (blocking).** Current folder holds ED-appropriateness working sets. Phase 1 needs a
#       full-population extract: all navigations, NMTARA 6 **included**, with disposition, NMTARA level, Cordy
#       protocol, client/market, work set ID, episode key, and nurse notes
# - [ ] **Rich** - confirm the correct episode key and the operational-call flag; validate against Power BI totals
# - [ ] **Nathan Haron** - confirm NMTARA code definitions and Logis field meanings
# - [ ] **Anisa** - confirm the human-coded gold set and its label column
# - [ ] **Dr. Stites / Dr. Troutman** - review the evidence audit sample; sign off on reason categories
# - [ ] Decide whether the historical reason categories become the structured override reason codes in the
#       new build (Shelly's point - findings should feed the new system's design, not just describe the old one)
# - [ ] Confirm whether per-client recast rates are needed, or whether a national rate is defensible
#
# ### Scope note
# Training-data drift is a real constraint: the move from the homegrown protocol to Schmitt Thompson means
# historical notes reflect a different set of questions. Nothing here should be used to train a model that
# scores *future* calls. The purpose is a one-time historical baseline.
