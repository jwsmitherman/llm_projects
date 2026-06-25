# Databricks notebook source
# MAGIC %md
# MAGIC # Ground-to-Air Transport Eligibility Analysis
# MAGIC
# MAGIC **Goal:** From EPCR (image trend / ESO) ground-transport data, identify **long ground trips that
# MAGIC could clinically have gone by air**, and report **eligible trip counts by pickup county and by month**.
# MAGIC
# MAGIC **Eligibility = passes ALL of:**
# MAGIC 1. **Lights & Sirens** (transport mode `eDisposition.18` or response mode `eResponse.24`)
# MAGIC 2. **High-equity critical segment** — Trauma, Stroke, OB (high-risk / pre-term labor), or Pediatrics (<5 yrs)
# MAGIC 3. **Ground** transport that is **long-distance (>50 mi)** *(these files appear pre-filtered to long ground)*
# MAGIC
# MAGIC **Grouping dimension:** `Scene Incident County Name (eScene.21)` — the **pickup** county, not destination.
# MAGIC
# MAGIC ---
# MAGIC ### Open items flagged for the Friday/Monday review (see ASSUMPTIONS cell)
# MAGIC - **Pediatrics <5** needs a patient-age field. It is not in the visible column list — may live in a
# MAGIC   demographics/supporting file. Notebook detects it if present, otherwise reports Peds as N/A.
# MAGIC - Segment keyword lists are driven by the **Primary/Secondary Impression** text — the PROFILING cell
# MAGIC   prints the actual distinct values so the lists can be tuned to the real data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## ASSUMPTIONS & DECISIONS (review these with the team)
# MAGIC 1. **One file = one month.** Month is derived from the file name (Nov 2025 … Apr 2026).
# MAGIC 2. **Data sheet auto-detected.** Sheet1 = tabular trip data; Sheet2 = field legend / single-record view.
# MAGIC 3. **County = Scene/pickup** (`eScene.21`), per the transcript ("we need by pickup, not destination").
# MAGIC 4. **Lights & Sirens** = match in transport mode OR response mode (toggle `LS_REQUIRE_TRANSPORT_ONLY`).
# MAGIC 5. **Trauma** = trauma keyword in impression OR a populated Injury Triage Criteria field (`eInjury.03/04`).
# MAGIC 6. **Distance >50 mi** only applied if a distance column exists; otherwise assumed pre-filtered.
# MAGIC 7. Data read as strings for safe categorical matching; numeric fields coerced where needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Config — edit these

# COMMAND ----------

# If openpyxl is missing on the cluster, uncomment:
# %pip install openpyxl

DATA_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/data"

# Leave as None to auto-pick every .xlsx in DATA_DIR, or pin the exact 6 files:
FILES = None
# FILES = [
#     "Transport-Distances-Main - Nov 2025.xlsx",
#     "Transport-Distances-Main - Dec 2025.xlsx",
#     "Transport-Distances-Main - Jan 2026.xlsx",
#     "Transport-Distances-Main - Feb 2026.xlsx",
#     "Transport-Distances-Main - Mar 2026.xlsx",
#     "Transport-Distances-Main - Apr 2026.xlsx",
# ]

OUTPUT_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/output"

MIN_DISTANCE_MILES = 50
LS_REQUIRE_TRANSPORT_ONLY = False   # True = require Lights&Sirens specifically on the transport leg

SEGMENT_KEYWORDS = {
    "Trauma": ["trauma", "injury", "fracture", "hemorrhage", "laceration", "amputation",
               "burn", "gunshot", "gsw", "stab", "assault", "fall", "head injury", "tbi",
               "crush", "penetrating", "blunt", "spinal", "impalement"],
    "Stroke": ["stroke", "cva", "cerebrovascular", "tia", "transient ischemic",
               "intracranial", "subarachnoid", "ischemic", "hemorrhagic stroke"],
    "OB":     ["obstetric", "pregnancy", "pregnant", "labor", "delivery", "preterm",
               "pre-term", "eclampsia", "preeclampsia", "gestational", "postpartum",
               "miscarriage", "contraction", "ob"],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Column resolver (maps logical names -> real headers by NEMSIS code or keyword)

# COMMAND ----------

import re, os, glob
import pandas as pd

def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

FIELD_SPECS = {
    "scene_county":      {"nemsis": "escene.21",      "keywords": ["scene incident county"]},
    "scene_state":       {"nemsis": "escene.18",      "keywords": ["scene incident state"]},
    "scene_postal":      {"nemsis": "escene.19",      "keywords": ["scene incident postal"]},
    "primary_impr":      {"nemsis": "esituation.11",  "keywords": ["primary impression"]},
    "secondary_impr":    {"nemsis": "esituation.12",  "keywords": ["secondary impression"]},
    "initial_acuity":    {"nemsis": "esituation.13",  "keywords": ["initial patient acuity"]},
    "final_acuity":      {"nemsis": "edisposition.19","keywords": ["final patient acuity"]},
    "resp_mode":         {"nemsis": "eresponse.24",   "keywords": ["additional response mode"]},
    "transport_mode":    {"nemsis": "edisposition.18","keywords": ["additional transport mode"]},
    "transport_disp":    {"nemsis": "edisposition.30","keywords": ["transport disposition"]},
    "injury_tc_12":      {"nemsis": "einjury.03",     "keywords": ["steps 1 and 2", "trauma center"]},
    "injury_tc_34":      {"nemsis": "einjury.04",     "keywords": ["steps 3 and 4", "risk factor"]},
    "gcs_initial":       {"nemsis": None,             "keywords": ["initial total glasgow"]},
    "gcs_last":          {"nemsis": None,             "keywords": ["last total glasgow"]},
    "incident_date":     {"nemsis": "etimes.11",      "keywords": ["patient arrived at destination"]},
    "incident_number":   {"nemsis": "eresponse.03",   "keywords": ["incident number"]},
    "agency_name":       {"nemsis": "eresponse.02",   "keywords": ["ems agency name"]},
    "agency_state":      {"nemsis": "dagency.04",     "keywords": ["agency state"]},
    # optional / may not exist in these files:
    "patient_age":       {"nemsis": "epatient.15",    "keywords": ["patient age", "age ("]},
    "patient_age_units": {"nemsis": "epatient.16",    "keywords": ["age units"]},
    "distance":          {"nemsis": None,             "keywords": ["distance", "mileage", "miles"]},
}

def resolve_columns(df):
    cols = list(df.columns)
    nmap = {c: _norm(c) for c in cols}
    resolved, used = {}, set()
    for logical, spec in FIELD_SPECS.items():
        match = None
        for kw in spec["keywords"]:
            for c in cols:
                if c not in used and kw in nmap[c]:
                    match = c; break
            if match: break
        if not match and spec.get("nemsis"):
            for c in cols:
                if c not in used and spec["nemsis"] in nmap[c]:
                    match = c; break
        resolved[logical] = match
        if match: used.add(match)
    return resolved

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load all monthly files (auto-detect data sheet, tag month from file name)

# COMMAND ----------

MONTH_MAP = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
             "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}

def month_from_name(fname):
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*-?\s*(\d{4})", fname.lower())
    if m:
        return f"{m.group(2)}-{MONTH_MAP[m.group(1)]}", f"{m.group(1).capitalize()} {m.group(2)}"
    return "unknown", os.path.splitext(fname)[0]

def pick_data_sheet(sheets):
    best, best_score = None, -1
    for name, df in sheets.items():
        if df is None or df.shape[0] < 1:
            continue
        r = resolve_columns(df)
        score = sum(1 for v in r.values() if v) + 0.0001 * df.shape[0]
        if r.get("scene_county"):
            score += 5
        if score > best_score:
            best, best_score = name, score
    return best or list(sheets.keys())[0]

def load_all(data_dir, files=None):
    if files:
        paths = [os.path.join(data_dir, f) for f in files]
    else:
        paths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    frames, manifest = [], []
    for p in paths:
        sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        sheet = pick_data_sheet(sheets)
        df = sheets[sheet].dropna(how="all").copy()
        key, label = month_from_name(os.path.basename(p))
        df["__source_file"] = os.path.basename(p)
        df["__month_key"] = key
        df["__month_label"] = label
        df["__sheet"] = sheet
        frames.append(df)
        manifest.append({"file": os.path.basename(p), "sheet": sheet,
                         "rows": df.shape[0], "month": label})
    combined = pd.concat(frames, ignore_index=True)
    return combined, pd.DataFrame(manifest)

combined, manifest = load_all(DATA_DIR, FILES)
print(f"Loaded {len(combined)} rows from {manifest.shape[0]} files.")
try:
    display(manifest)
except Exception:
    print(manifest.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Profile the data (tune the keyword lists from what you see here)

# COMMAND ----------

R = resolve_columns(combined)

print("=== RESOLVED COLUMNS ===")
for k, v in R.items():
    print(f"  {k:18s} -> {v}")
missing = [k for k, v in R.items() if v is None]
print("\nUNRESOLVED:", missing)

def show(x):
    try: display(x)
    except Exception: print(x)

def distinct_counts(logical, top=40):
    c = R.get(logical)
    if not c:
        print(f"[{logical}] not found"); return
    vc = combined[c].fillna("(null)").value_counts().head(top)
    print(f"\n=== {logical} ({c}) — top {top} distinct ===")
    print(vc.to_string())

for f in ["primary_impr", "secondary_impr", "resp_mode", "transport_mode",
          "transport_disp", "initial_acuity", "final_acuity",
          "injury_tc_12", "scene_state", "scene_county"]:
    distinct_counts(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Derive flags & apply eligibility criteria

# COMMAND ----------

def col(name):
    c = R.get(name)
    return combined[c] if c else pd.Series([None] * len(combined), index=combined.index)

def build_regex(keywords):
    parts = []
    for k in keywords:
        k = k.strip()
        parts.append(r"\b" + re.escape(k) + r"\b" if re.fullmatch(r"[a-z0-9]+", k) else re.escape(k))
    return re.compile("|".join(parts))

def populated(series):
    s = series.fillna("").map(_norm).str.replace('"', "", regex=False)
    return ~s.isin(["", "not recorded", "none", "no", "na", "n/a", "not applicable"])

SEG_RX = {seg: build_regex(kws) for seg, kws in SEGMENT_KEYWORDS.items()}
impr = (col("primary_impr").fillna("").map(_norm) + " || " + col("secondary_impr").fillna("").map(_norm))

is_trauma = impr.str.contains(SEG_RX["Trauma"]) | populated(col("injury_tc_12")) | populated(col("injury_tc_34"))
is_stroke = impr.str.contains(SEG_RX["Stroke"])
is_ob     = impr.str.contains(SEG_RX["OB"])

age_col = R.get("patient_age")
if age_col:
    age = pd.to_numeric(col("patient_age"), errors="coerce")
    units = col("patient_age_units").fillna("").map(_norm)
    is_peds = ((units.str.contains("year") & (age < 5)) |
               units.str.contains("month|day|hour|minute|week") |
               (units.eq("") & (age < 5)))
    peds_available = True
else:
    is_peds = pd.Series(False, index=combined.index)
    peds_available = False
    print("WARNING: no patient-age column found — Pediatrics (<5) cannot be evaluated from this file.")

is_target = is_trauma | is_stroke | is_ob | is_peds

tmode = col("transport_mode").fillna("").map(_norm)
rmode = col("resp_mode").fillna("").map(_norm)
ls_transport = tmode.str.contains("lights and sirens")
ls_response  = rmode.str.contains("lights and sirens")
lights_sirens = ls_transport if LS_REQUIRE_TRANSPORT_ONLY else (ls_transport | ls_response)

air_rx = re.compile(r"\b(air|rotor|fixed[- ]?wing|helicopter|flight|medevac|medivac)\b")
disp = col("transport_disp").fillna("").map(_norm)
is_ground = ~(disp.str.contains(air_rx) | tmode.str.contains(air_rx))

if R.get("distance"):
    long_trip = pd.to_numeric(col("distance"), errors="coerce") >= MIN_DISTANCE_MILES
    distance_available = True
else:
    long_trip = pd.Series(True, index=combined.index)
    distance_available = False
    print(f"NOTE: no distance column — assuming files are already filtered to >{MIN_DISTANCE_MILES} mi.")

combined["scene_county"] = col("scene_county")
combined["scene_state"]  = col("scene_state")
combined["is_trauma"] = is_trauma
combined["is_stroke"] = is_stroke
combined["is_ob"]     = is_ob
combined["is_peds"]   = is_peds
combined["lights_sirens"] = lights_sirens
combined["__eligible"] = is_target & lights_sirens & is_ground & long_trip

print("\n=== FILTER FUNNEL (rows) ===")
print(f"  total                         : {len(combined)}")
print(f"  lights & sirens               : {int(lights_sirens.sum())}")
print(f"  ground (non-air)              : {int(is_ground.sum())}")
print(f"  long-distance (>{MIN_DISTANCE_MILES}mi)        : {int(long_trip.sum())}  (available={distance_available})")
print(f"  in target segment             : {int(is_target.sum())}")
print(f"      trauma={int(is_trauma.sum())}  stroke={int(is_stroke.sum())}  "
      f"ob={int(is_ob.sum())}  peds={int(is_peds.sum())} (peds_available={peds_available})")
print(f"  ELIGIBLE (all criteria)       : {int(combined['__eligible'].sum())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. PRIMARY OUTPUT — eligible trips by county x month

# COMMAND ----------

elig = combined[combined["__eligible"]].copy()

order = (combined[["__month_key", "__month_label"]].drop_duplicates()
         .sort_values("__month_key")["__month_label"].tolist())
elig["__month_label"] = pd.Categorical(elig["__month_label"], categories=order, ordered=True)

county_month = pd.pivot_table(
    elig, index=["scene_state", "scene_county"], columns="__month_label",
    values="__eligible", aggfunc="sum", fill_value=0, margins=True, margins_name="Total",
    observed=False,
).astype(int).sort_values("Total", ascending=False)

print("Eligible trips by county x month:")
show(county_month.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Segment breakdown by county

# COMMAND ----------

seg_by_county = (elig.groupby(["scene_state", "scene_county"])[["is_trauma", "is_stroke", "is_ob", "is_peds"]]
                 .sum().astype(int))
seg_by_county["eligible_total"] = elig.groupby(["scene_state", "scene_county"])["__eligible"].sum().astype(int)
seg_by_county = seg_by_county.sort_values("eligible_total", ascending=False)
show(seg_by_county.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Annual extrapolation (4-month sample -> 12-month estimate)

# COMMAND ----------

n_months = elig["__month_key"].nunique()
total_elig = int(elig["__eligible"].sum())
annual = round(total_elig / n_months * 12) if n_months else 0
print(f"Months of data : {n_months}")
print(f"Eligible trips : {total_elig}")
print(f"Annualized est : {annual}  (= {total_elig} / {n_months} * 12)")

annual_by_county = (elig.groupby(["scene_state", "scene_county"])["__eligible"].sum()
                    .reset_index(name="sample_eligible"))
annual_by_county["annualized_estimate"] = (annual_by_county["sample_eligible"] / n_months * 12).round().astype(int)
annual_by_county = annual_by_county.sort_values("annualized_estimate", ascending=False)
show(annual_by_county)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save outputs (CSV + Excel) and line-level eligible trips

# COMMAND ----------

os.makedirs(OUTPUT_DIR, exist_ok=True)

eligible_lines = elig[[c for c in [
    "__source_file", "__month_label", R.get("incident_number"), R.get("agency_name"),
    "scene_state", "scene_county", R.get("primary_impr"), R.get("secondary_impr"),
    R.get("transport_mode"), R.get("resp_mode"), R.get("initial_acuity"), R.get("final_acuity"),
    "is_trauma", "is_stroke", "is_ob", "is_peds",
] if c is not None]].copy()

county_month.reset_index().to_csv(f"{OUTPUT_DIR}/eligible_by_county_month.csv", index=False)
seg_by_county.reset_index().to_csv(f"{OUTPUT_DIR}/eligible_segment_by_county.csv", index=False)
eligible_lines.to_csv(f"{OUTPUT_DIR}/eligible_trips_linelevel.csv", index=False)

try:
    with pd.ExcelWriter(f"{OUTPUT_DIR}/ground_to_air_eligibility_summary.xlsx", engine="openpyxl") as xw:
        county_month.reset_index().to_excel(xw, sheet_name="county_x_month", index=False)
        seg_by_county.reset_index().to_excel(xw, sheet_name="segment_by_county", index=False)
        annual_by_county.to_excel(xw, sheet_name="annualized", index=False)
        eligible_lines.to_excel(xw, sheet_name="eligible_linelevel", index=False)
        manifest.to_excel(xw, sheet_name="file_manifest", index=False)
    print("Wrote Excel summary.")
except Exception as e:
    print("Excel write skipped:", e)

print("Outputs in:", OUTPUT_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. (Optional) Persist as Delta for re-use in the deck pipeline

# COMMAND ----------

# spark.createDataFrame(county_month.reset_index()).write.mode("overwrite") \
#     .saveAsTable("dnasandbox.ground_air.eligible_by_county_month")
