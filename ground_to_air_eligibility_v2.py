# Databricks notebook source
# MAGIC %md
# MAGIC # Ground-to-Air Transport Eligibility Analysis (v2)
# MAGIC
# MAGIC Counts **distinct ground trips that could clinically have gone by air**, by **pickup county** and **month**.
# MAGIC
# MAGIC ### What changed from v1 (and why the numbers were inflated)
# MAGIC 1. **Count distinct incidents, not rows.** The Main extract repeats each incident (UUID) many times
# MAGIC    (one-to-many flatten). v1 counted rows -> 6,722 in Riverside. v2 dedups to one row per **NEMSIS UUID**.
# MAGIC 2. **Trauma flag fixed.** v1 marked a trip trauma whenever the Injury-Triage columns were merely *populated*,
# MAGIC    so "Respiratory Distress" got flagged. v2 uses specific impression values + the **Revised Trauma Score**
# MAGIC    from the Vitals file, and never matches "No Apparent Illness/Injury".
# MAGIC 3. **Stroke flag improved** with the **Stroke Scale** fields (eVitals.29/30) from the Vitals file.
# MAGIC 4. **Pediatrics (<5)** now parsed from the **PCR Narrative** text ("10 y/o male"), since there is no age column.
# MAGIC 5. **Lights & Sirens** now requires the **transport leg** (eDisposition.18), not the drive to scene.
# MAGIC
# MAGIC ### Data layout
# MAGIC - `data/`  -> 6 **Main** files (one per month) = trip rows
# MAGIC - `other_data/` -> 6 **Vitals-&-UUID** + 6 **Narrative-&-UUID** files, joined to Main on `Incident Record NEMSIS UUID`

# COMMAND ----------

# MAGIC %md
# MAGIC ## ASSUMPTIONS (confirm Friday)
# MAGIC 1. Distance >50 mi is **pre-applied** at extract time (transcript: "this is already filtered data" / "the long trips").
# MAGIC 2. **NEMSIS UUID is unique per trip** -> dedup key. Notebook prints the rows-per-incident ratio so you can verify.
# MAGIC 3. Age parsed from narrative is the **first** "N y/o"-type mention; usually the patient, but free text is imperfect.
# MAGIC 4. A trip in multiple segments is counted **once** in the eligible total (segment columns can sum higher).
# MAGIC 5. Acuity filter (drop "Lower Acuity (Green)") is available but **OFF** by default — flip if the team wants it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Config

# COMMAND ----------

# %pip install openpyxl

MAIN_DIR  = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/data"
OTHER_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/other_data"
OUTPUT_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/output"

MAIN_FILES  = None   # None = glob all *.xlsx; or pin a list of filenames
DEDUP_BY_UUID = True
ACUITY_FILTER = False        # True = require Critical/Emergent (drop Lower Acuity/Green)
LS_REQUIRE_TRANSPORT_LEG = True
PEDS_MAX_AGE_YEARS = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Column resolver + helpers

# COMMAND ----------

import re, os, glob
import pandas as pd
import numpy as np

def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

FIELD_SPECS = {
    "uuid":             {"nemsis": None,              "keywords": ["nemsis uuid"]},
    "scene_county":     {"nemsis": "escene.21",       "keywords": ["scene incident county"]},
    "scene_state":      {"nemsis": "escene.18",       "keywords": ["scene incident state"]},
    "primary_impr":     {"nemsis": "esituation.11",   "keywords": ["primary impression"]},
    "secondary_impr":   {"nemsis": "esituation.12",   "keywords": ["secondary impression"]},
    "initial_acuity":   {"nemsis": "esituation.13",   "keywords": ["initial patient acuity"]},
    "final_acuity":     {"nemsis": "edisposition.19", "keywords": ["final patient acuity"]},
    "resp_mode":        {"nemsis": "eresponse.24",    "keywords": ["additional response mode"]},
    "transport_mode":   {"nemsis": "edisposition.18", "keywords": ["additional transport mode"]},
    "transport_disp":   {"nemsis": "edisposition.30", "keywords": ["transport disposition"]},
    "incident_number":  {"nemsis": "eresponse.03",    "keywords": ["incident number"]},
    "agency_name":      {"nemsis": "eresponse.02",    "keywords": ["ems agency name"]},
    # supporting-file fields:
    "narrative":        {"nemsis": "enarrative.01",   "keywords": ["patient care report narrative", "narrative"]},
    "stroke_scale_sc":  {"nemsis": "evitals.29",      "keywords": ["stroke scale score"]},
    "stroke_scale_ty":  {"nemsis": "evitals.30",      "keywords": ["stroke scale type"]},
    "rts":              {"nemsis": "evitals.33",      "keywords": ["revised trauma score"]},
    "gcs_total":        {"nemsis": "evitals.23",      "keywords": ["total glasgow"]},
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

def show(x):
    try: display(x)
    except Exception: print(x)

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
        if r.get("scene_county"): score += 5
        if r.get("uuid"): score += 5
        if score > best_score:
            best, best_score = name, score
    return best or list(sheets.keys())[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Main files (trip rows)

# COMMAND ----------

def load_main(data_dir, files=None):
    paths = [os.path.join(data_dir, f) for f in files] if files else sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    frames, manifest = [], []
    for p in paths:
        if "main" not in os.path.basename(p).lower():
            continue
        sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        sheet = pick_data_sheet(sheets)
        df = sheets[sheet].dropna(how="all").copy()
        key, label = month_from_name(os.path.basename(p))
        df["__source_file"] = os.path.basename(p)
        df["__month_key"] = key
        df["__month_label"] = label
        frames.append(df)
        manifest.append({"file": os.path.basename(p), "sheet": sheet, "rows": df.shape[0], "month": label})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest)

main, main_manifest = load_main(MAIN_DIR, MAIN_FILES)
print(f"Main rows loaded: {len(main)}")
show(main_manifest)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load supporting files (Vitals + Narrative) and aggregate to one row per UUID

# COMMAND ----------

def _load_support(data_dir, name_token):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    frames = []
    for p in paths:
        if name_token not in os.path.basename(p).lower():
            continue
        sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        sheet = pick_data_sheet(sheets)
        frames.append(sheets[sheet].dropna(how="all").copy())
    if not frames:
        print(f"WARNING: no '{name_token}' files found in {data_dir}")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

vitals_raw = _load_support(OTHER_DIR, "vitals")
narr_raw   = _load_support(OTHER_DIR, "narrative")

# ---- Vitals -> per-UUID flags ----
if not vitals_raw.empty:
    rv = resolve_columns(vitals_raw)
    v = pd.DataFrame({"uuid": vitals_raw[rv["uuid"]] if rv.get("uuid") else None})
    v["rts_present"]   = vitals_raw[rv["rts"]].notna() if rv.get("rts") else False
    v["stroke_scale_present"] = False
    if rv.get("stroke_scale_sc"):
        v["stroke_scale_present"] |= vitals_raw[rv["stroke_scale_sc"]].notna()
    if rv.get("stroke_scale_ty"):
        st = vitals_raw[rv["stroke_scale_ty"]].fillna("").map(_norm)
        v["stroke_scale_present"] |= ~st.isin(["", "not recorded", "none", "not applicable"])
    v["gcs"] = pd.to_numeric(vitals_raw[rv["gcs_total"]], errors="coerce") if rv.get("gcs_total") else np.nan
    v = v.dropna(subset=["uuid"])
    vitals_uuid = v.groupby("uuid").agg(
        rts_present=("rts_present", "max"),
        stroke_scale_present=("stroke_scale_present", "max"),
        gcs_min=("gcs", "min"),
    ).reset_index()
    print(f"Vitals: {len(vitals_raw)} rows -> {len(vitals_uuid)} UUIDs")
else:
    vitals_uuid = pd.DataFrame(columns=["uuid", "rts_present", "stroke_scale_present", "gcs_min"])

# ---- Narrative -> per-UUID age ----
AGE_PATTERNS = [
    (r"(\d{1,3})\s*(?:y/?o|yo|year[s\-\s]*old|yr[s]?[\s\-]*old)", 365.0),
    (r"(\d{1,3})\s*(?:m/?o|month[s\-\s]*old)", 30.4),
    (r"(\d{1,3})\s*(?:w/?o|wk|week[s\-\s]*old)", 7.0),
    (r"(\d{1,3})\s*(?:d/?o|day[s\-\s]*old)", 1.0),
]
def parse_age_years(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    t = str(text).lower()
    if re.search(r"\bnewborn\b|\bneonat", t):
        return 0.0
    for pat, days in AGE_PATTERNS:
        m = re.search(pat, t)
        if m:
            return float(m.group(1)) * days / 365.0
    return np.nan

if not narr_raw.empty:
    rn = resolve_columns(narr_raw)
    n = pd.DataFrame({
        "uuid": narr_raw[rn["uuid"]] if rn.get("uuid") else None,
        "narrative": narr_raw[rn["narrative"]] if rn.get("narrative") else "",
    }).dropna(subset=["uuid"])
    n["age_years"] = n["narrative"].map(parse_age_years)
    narr_uuid = n.sort_values("age_years").groupby("uuid").agg(age_years=("age_years", "first")).reset_index()
    print(f"Narrative: {len(narr_raw)} rows -> {len(narr_uuid)} UUIDs; "
          f"age parsed for {narr_uuid['age_years'].notna().sum()}")
else:
    narr_uuid = pd.DataFrame(columns=["uuid", "age_years"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Row-level flags on Main, then collapse to one row per incident

# COMMAND ----------

R = resolve_columns(main)
print("Resolved (main):", {k: v for k, v in R.items() if v})

def col(df, name):
    c = R.get(name)
    return df[c] if c else pd.Series([None] * len(df), index=df.index)

impr = (col(main, "primary_impr").fillna("").map(_norm) + " || " + col(main, "secondary_impr").fillna("").map(_norm))

TRAUMA_RX = re.compile(r"traumatic injury|injury of|injury to|fracture|dislocation|laceration|amputation|"
                       r"avulsion|\bburn|gunshot|\bstab|penetrating|blunt|crush|impalement|head strike|"
                       r"head injury|concussion|\btbi\b")
STROKE_RX = re.compile(r"\bstroke\b|\bcva\b|cerebrovascular")
OB_RX     = re.compile(r"pregnan|obstetric|eclampsia|in labor|preterm|pre-term|peripartum|postpartum|ob /")

no_apparent = impr.str.contains("no apparent")
is_trauma_impr = impr.str.contains(TRAUMA_RX) & ~no_apparent
is_stroke_impr = impr.str.contains(STROKE_RX)
is_ob_impr     = impr.str.contains(OB_RX)

tmode = col(main, "transport_mode").fillna("").map(_norm)
ls_transport = tmode.str.contains("lights and sirens")
if not LS_REQUIRE_TRANSPORT_LEG:
    rmode = col(main, "resp_mode").fillna("").map(_norm)
    ls_transport = ls_transport | rmode.str.contains("lights and sirens")

air_rx = re.compile(r"\b(air|rotor|fixed[- ]?wing|helicopter|flight|medevac|medivac)\b")
disp = col(main, "transport_disp").fillna("").map(_norm)
is_ground = ~(disp.str.contains(air_rx) | tmode.str.contains(air_rx))

ia = col(main, "initial_acuity").fillna("").map(_norm)
fa = col(main, "final_acuity").fillna("").map(_norm)
high_acuity = ia.str.contains("critical|emergent") | fa.str.contains("critical|emergent")

uuid_series = col(main, "uuid")
uuid_key = uuid_series.where(uuid_series.notna(), col(main, "incident_number"))
uuid_key = uuid_key.where(uuid_key.notna(), pd.Series(main.index.astype(str), index=main.index))

m = pd.DataFrame({
    "uuid": uuid_key,
    "scene_state": col(main, "scene_state"),
    "scene_county": col(main, "scene_county"),
    "month_key": main["__month_key"],
    "month_label": main["__month_label"],
    "incident_number": col(main, "incident_number"),
    "agency_name": col(main, "agency_name"),
    "is_trauma_impr": is_trauma_impr.astype(int),
    "is_stroke_impr": is_stroke_impr.astype(int),
    "is_ob_impr": is_ob_impr.astype(int),
    "ls_transport": ls_transport.astype(int),
    "is_ground": is_ground.astype(int),
    "high_acuity": high_acuity.astype(int),
})

if DEDUP_BY_UUID:
    inc = m.groupby("uuid").agg(
        scene_state=("scene_state", "first"),
        scene_county=("scene_county", "first"),
        month_key=("month_key", "first"),
        month_label=("month_label", "first"),
        incident_number=("incident_number", "first"),
        agency_name=("agency_name", "first"),
        is_trauma_impr=("is_trauma_impr", "max"),
        is_stroke_impr=("is_stroke_impr", "max"),
        is_ob_impr=("is_ob_impr", "max"),
        ls_transport=("ls_transport", "max"),
        is_ground=("is_ground", "max"),
        high_acuity=("high_acuity", "max"),
    ).reset_index()
else:
    inc = m.copy()

ratio = len(m) / max(len(inc), 1)
print(f"Main rows: {len(m)}  ->  distinct incidents: {len(inc)}  (rows-per-incident = {ratio:.2f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Join vitals + narrative, build final eligibility

# COMMAND ----------

inc = inc.merge(vitals_uuid, on="uuid", how="left").merge(narr_uuid, on="uuid", how="left")
for c in ["rts_present", "stroke_scale_present"]:
    inc[c] = inc[c].fillna(False).astype(bool)

inc["is_trauma"] = (inc["is_trauma_impr"] == 1) | inc["rts_present"]
inc["is_stroke"] = (inc["is_stroke_impr"] == 1) | inc["stroke_scale_present"]
inc["is_ob"]     = (inc["is_ob_impr"] == 1)
inc["is_peds"]   = inc["age_years"] < PEDS_MAX_AGE_YEARS
inc["is_target"] = inc[["is_trauma", "is_stroke", "is_ob", "is_peds"]].any(axis=1)

acuity_ok = (inc["high_acuity"] == 1) if ACUITY_FILTER else True
inc["eligible"] = inc["is_target"] & (inc["ls_transport"] == 1) & (inc["is_ground"] == 1) & acuity_ok

print("=== FUNNEL (distinct incidents) ===")
print(f"  incidents              : {len(inc)}")
print(f"  lights & sirens (transport): {int((inc['ls_transport']==1).sum())}")
print(f"  ground                 : {int((inc['is_ground']==1).sum())}")
print(f"  in target segment      : {int(inc['is_target'].sum())}")
print(f"     trauma={int(inc['is_trauma'].sum())} stroke={int(inc['is_stroke'].sum())} "
      f"ob={int(inc['is_ob'].sum())} peds={int(inc['is_peds'].sum())}")
print(f"  ELIGIBLE               : {int(inc['eligible'].sum())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. PRIMARY OUTPUT — eligible trips by county x month

# COMMAND ----------

elig = inc[inc["eligible"]].copy()

order = (inc[["month_key", "month_label"]].drop_duplicates()
         .sort_values("month_key")["month_label"].tolist())

county_month = pd.pivot_table(
    elig, index=["scene_state", "scene_county"], columns="month_label",
    values="eligible", aggfunc="sum", margins=True, margins_name="Total",
)
month_cols = [mn for mn in order if mn in county_month.columns]
county_month = (county_month.reindex(columns=month_cols + (["Total"] if "Total" in county_month.columns else []))
                .fillna(0).astype(int).sort_values("Total", ascending=False))
print("Eligible trips by county x month:")
show(county_month.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Segment breakdown + annualized

# COMMAND ----------

seg = (elig.groupby(["scene_state", "scene_county"])[["is_trauma", "is_stroke", "is_ob", "is_peds"]]
       .sum().astype(int))
seg["eligible_total"] = elig.groupby(["scene_state", "scene_county"])["eligible"].sum().astype(int)
seg = seg.sort_values("eligible_total", ascending=False)
show(seg.reset_index())

n_months = elig["month_key"].nunique()
annual = (elig.groupby(["scene_state", "scene_county"])["eligible"].sum().reset_index(name="sample_eligible"))
annual["annualized_estimate"] = (annual["sample_eligible"] / max(n_months, 1) * 12).round().astype(int)
annual = annual.sort_values("annualized_estimate", ascending=False)
print(f"Months of data: {n_months} | total eligible: {int(elig['eligible'].sum())}")
show(annual)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save outputs

# COMMAND ----------

os.makedirs(OUTPUT_DIR, exist_ok=True)
line = elig[["scene_state", "scene_county", "month_label", "incident_number", "agency_name",
             "is_trauma", "is_stroke", "is_ob", "is_peds", "age_years", "gcs_min",
             "rts_present", "stroke_scale_present"]].copy()

county_month.reset_index().to_csv(f"{OUTPUT_DIR}/eligible_by_county_month.csv", index=False)
seg.reset_index().to_csv(f"{OUTPUT_DIR}/eligible_segment_by_county.csv", index=False)
line.to_csv(f"{OUTPUT_DIR}/eligible_trips_linelevel.csv", index=False)
try:
    with pd.ExcelWriter(f"{OUTPUT_DIR}/ground_to_air_eligibility_summary.xlsx", engine="openpyxl") as xw:
        county_month.reset_index().to_excel(xw, sheet_name="county_x_month", index=False)
        seg.reset_index().to_excel(xw, sheet_name="segment_by_county", index=False)
        annual.to_excel(xw, sheet_name="annualized", index=False)
        line.to_excel(xw, sheet_name="eligible_linelevel", index=False)
        main_manifest.to_excel(xw, sheet_name="file_manifest", index=False)
    print("Wrote Excel summary.")
except Exception as e:
    print("Excel write skipped:", e)
print("Outputs in:", OUTPUT_DIR)
