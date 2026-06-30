# Databricks notebook source

# ============================================================================
# Ground-to-Air — Two-Slide Data Pack (LA & San Diego)
#
# Produces ONLY the data needed for the two slides:
#   Slide 1: Los Angeles County    Slide 2: San Diego County (Oceanside)
# Per county, four blocks:
#   1. Ground transport volume (total long-ground + air-eligible, period + per-month)
#   2. Distribution by condition (Trauma / Stroke / OB / Pediatrics)
#   3. Distribution by criticality (Critical-Red / Emergent-Yellow / Lower-Green)
#   4. Short narrative (auto-generated, 2 bullets)
#
# Timeframe: whatever months are in the files (expected Nov 2025 - Apr 2026, ~6 months).
# Underlying logic matches the validated pipeline: dedup by NEMSIS UUID, state/county
# normalization (CA == California), corrected segment flags, transport-leg Lights & Sirens.
# ============================================================================

# COMMAND ----------

# ---- 0. Config ----

# %pip install openpyxl

MAIN_DIR   = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/data"
OTHER_DIR  = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/other_data"
OUTPUT_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/two_slide_output"

FOCUS_COUNTIES = ["Los Angeles", "San Diego"]
TARGET_STATE = "California"
LS_REQUIRE_TRANSPORT_LEG = True
PEDS_MAX_AGE_YEARS = 5
MEANINGFUL_PER_MONTH = 25   # narrative threshold: >= this = "meaningful volume", else "limited"

# COMMAND ----------

# ---- 1. Helpers, resolver, normalization ----

import re, os, glob
import pandas as pd
import numpy as np

def _norm(s): return re.sub(r"\s+", " ", str(s)).strip().lower()

FIELD_SPECS = {
    "uuid":            {"nemsis": None,              "keywords": ["nemsis uuid"]},
    "scene_county":    {"nemsis": "escene.21",       "keywords": ["scene incident county"]},
    "scene_state":     {"nemsis": "escene.18",       "keywords": ["scene incident state"]},
    "primary_impr":    {"nemsis": "esituation.11",   "keywords": ["primary impression"]},
    "secondary_impr":  {"nemsis": "esituation.12",   "keywords": ["secondary impression"]},
    "initial_acuity":  {"nemsis": "esituation.13",   "keywords": ["initial patient acuity"]},
    "final_acuity":    {"nemsis": "edisposition.19", "keywords": ["final patient acuity"]},
    "resp_mode":       {"nemsis": "eresponse.24",    "keywords": ["additional response mode"]},
    "transport_mode":  {"nemsis": "edisposition.18", "keywords": ["additional transport mode"]},
    "transport_disp":  {"nemsis": "edisposition.30", "keywords": ["transport disposition"]},
    "incident_number": {"nemsis": "eresponse.03",    "keywords": ["incident number"]},
    "narrative":       {"nemsis": "enarrative.01",   "keywords": ["patient care report narrative", "narrative"]},
    "stroke_scale_sc": {"nemsis": "evitals.29",      "keywords": ["stroke scale score"]},
    "stroke_scale_ty": {"nemsis": "evitals.30",      "keywords": ["stroke scale type"]},
    "rts":             {"nemsis": "evitals.33",      "keywords": ["revised trauma score"]},
}

def resolve_columns(df):
    cols = list(df.columns); nmap = {c: _norm(c) for c in cols}
    resolved, used = {}, set()
    for logical, spec in FIELD_SPECS.items():
        match = None
        for kw in spec["keywords"]:
            for c in cols:
                if c not in used and kw in nmap[c]: match = c; break
            if match: break
        if not match and spec.get("nemsis"):
            for c in cols:
                if c not in used and spec["nemsis"] in nmap[c]: match = c; break
        resolved[logical] = match
        if match: used.add(match)
    return resolved

def show(x):
    try: display(x)
    except Exception: print(x)

STATE_MAP = {"ca":"California","az":"Arizona","nv":"Nevada","or":"Oregon","tx":"Texas","tn":"Tennessee",
             "fl":"Florida","ny":"New York","la":"Louisiana","ga":"Georgia","wa":"Washington","al":"Alabama",
             "nm":"New Mexico","ok":"Oklahoma","co":"Colorado","ks":"Kansas","ky":"Kentucky","il":"Illinois"}
def norm_state(v):
    s = _norm(v)
    if s in STATE_MAP: return STATE_MAP[s]
    return str(v).strip().title() if v is not None and str(v).strip() else None
def norm_county(v):
    if v is None or str(v).strip() == "": return None
    return re.sub(r"\s+", " ", str(v)).strip().title()

MONTH_MAP = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
             "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
def month_from_name(f):
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*-?\s*(\d{4})", f.lower())
    return (f"{m.group(2)}-{MONTH_MAP[m.group(1)]}", f"{m.group(1).capitalize()} {m.group(2)}") if m else ("unknown", os.path.splitext(f)[0])

def pick_data_sheet(sheets):
    best, bs = None, -1
    for name, df in sheets.items():
        if df is None or df.shape[0] < 1: continue
        r = resolve_columns(df)
        score = sum(1 for v in r.values() if v) + 0.0001*df.shape[0] + (5 if r.get("scene_county") else 0) + (5 if r.get("uuid") else 0)
        if score > bs: best, bs = name, score
    return best or list(sheets.keys())[0]

# COMMAND ----------

# ---- 2. Load Main + supporting ----

def load_set(data_dir, token):
    frames, manifest = [], []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.xlsx"))):
        if token not in os.path.basename(p).lower(): continue
        sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        df = sheets[pick_data_sheet(sheets)].dropna(how="all").copy()
        key, label = month_from_name(os.path.basename(p))
        df["__month_key"], df["__month_label"] = key, label
        frames.append(df); manifest.append({"file": os.path.basename(p), "rows": df.shape[0], "month": label})
    return (pd.concat(frames, ignore_index=True), pd.DataFrame(manifest)) if frames else (pd.DataFrame(), pd.DataFrame())

main, manifest = load_set(MAIN_DIR, "main")
vitals_raw, _ = load_set(OTHER_DIR, "vitals")
narr_raw, _   = load_set(OTHER_DIR, "narrative")
print(f"Main rows: {len(main)} | vitals: {len(vitals_raw)} | narrative: {len(narr_raw)}")
show(manifest)

# COMMAND ----------

# ---- 3. Per-UUID vitals flags + narrative age ----

if not vitals_raw.empty:
    rv = resolve_columns(vitals_raw)
    v = pd.DataFrame({"uuid": vitals_raw[rv["uuid"]] if rv.get("uuid") else None})
    v["rts_present"] = vitals_raw[rv["rts"]].notna() if rv.get("rts") else False
    v["stroke_scale_present"] = False
    if rv.get("stroke_scale_sc"): v["stroke_scale_present"] |= vitals_raw[rv["stroke_scale_sc"]].notna()
    if rv.get("stroke_scale_ty"):
        st = vitals_raw[rv["stroke_scale_ty"]].fillna("").map(_norm)
        v["stroke_scale_present"] |= ~st.isin(["", "not recorded", "none", "not applicable"])
    v = v.dropna(subset=["uuid"])
    vitals_uuid = v.groupby("uuid").agg(rts_present=("rts_present","max"), stroke_scale_present=("stroke_scale_present","max")).reset_index()
else:
    vitals_uuid = pd.DataFrame(columns=["uuid","rts_present","stroke_scale_present"])

AGE_PATTERNS = [(r"(\d{1,3})\s*(?:y/?o|yo|year[s\-\s]*old|yr[s]?[\s\-]*old)", 365.0),
                (r"(\d{1,3})\s*(?:m/?o|month[s\-\s]*old)", 30.4),
                (r"(\d{1,3})\s*(?:w/?o|wk|week[s\-\s]*old)", 7.0),
                (r"(\d{1,3})\s*(?:d/?o|day[s\-\s]*old)", 1.0)]
def parse_age_years(text):
    if text is None or (isinstance(text, float) and np.isnan(text)): return np.nan
    t = str(text).lower()
    if re.search(r"\bnewborn\b|\bneonat", t): return 0.0
    for pat, days in AGE_PATTERNS:
        mm = re.search(pat, t)
        if mm: return float(mm.group(1)) * days / 365.0
    return np.nan

if not narr_raw.empty:
    rn = resolve_columns(narr_raw)
    n = pd.DataFrame({"uuid": narr_raw[rn["uuid"]] if rn.get("uuid") else None,
                      "narrative": narr_raw[rn["narrative"]] if rn.get("narrative") else ""}).dropna(subset=["uuid"])
    n["age_years"] = n["narrative"].map(parse_age_years)
    narr_uuid = n.sort_values("age_years").groupby("uuid").agg(age_years=("age_years","first")).reset_index()
else:
    narr_uuid = pd.DataFrame(columns=["uuid","age_years"])

# COMMAND ----------

# ---- 4. Flags, dedup to incident, normalize ----

R = resolve_columns(main)
def col(name):
    c = R.get(name)
    return main[c] if c else pd.Series([None]*len(main), index=main.index)

impr = (col("primary_impr").fillna("").map(_norm) + " || " + col("secondary_impr").fillna("").map(_norm))
TRAUMA_RX = re.compile(r"traumatic injury|injury of|injury to|fracture|dislocation|laceration|amputation|"
                       r"avulsion|\bburn|gunshot|\bstab|penetrating|blunt|crush|impalement|head strike|head injury|concussion|\btbi\b")
STROKE_RX = re.compile(r"\bstroke\b|\bcva\b|cerebrovascular")
OB_RX     = re.compile(r"pregnan|obstetric|eclampsia|in labor|preterm|pre-term|peripartum|postpartum|ob /")
no_apparent = impr.str.contains("no apparent")

tmode = col("transport_mode").fillna("").map(_norm)
ls = tmode.str.contains("lights and sirens")
if not LS_REQUIRE_TRANSPORT_LEG:
    ls = ls | col("resp_mode").fillna("").map(_norm).str.contains("lights and sirens")
air_rx = re.compile(r"\b(air|rotor|fixed[- ]?wing|helicopter|flight|medevac|medivac)\b")
is_ground = ~(col("transport_disp").fillna("").map(_norm).str.contains(air_rx) | tmode.str.contains(air_rx))

ACUITY_RANK = {"critical (red)": 3, "emergent (yellow)": 2, "lower acuity (green)": 1}
def arank(s): return s.fillna("").map(_norm).map(lambda x: ACUITY_RANK.get(x, 0))
crit_rank = pd.concat([arank(col("initial_acuity")), arank(col("final_acuity"))], axis=1).max(axis=1)
RANK_LABEL = {3:"Critical (Red)", 2:"Emergent (Yellow)", 1:"Lower Acuity (Green)", 0:"Not Recorded"}

uuid_series = col("uuid")
uuid_key = uuid_series.where(uuid_series.notna(), col("incident_number"))
uuid_key = uuid_key.where(uuid_key.notna(), pd.Series(main.index.astype(str), index=main.index))

m = pd.DataFrame({
    "uuid": uuid_key,
    "scene_state": col("scene_state").map(norm_state),
    "scene_county": col("scene_county").map(norm_county),
    "month_key": main["__month_key"], "month_label": main["__month_label"],
    "is_trauma_impr": (impr.str.contains(TRAUMA_RX) & ~no_apparent).astype(int),
    "is_stroke_impr": impr.str.contains(STROKE_RX).astype(int),
    "is_ob_impr": impr.str.contains(OB_RX).astype(int),
    "ls": ls.astype(int), "is_ground": is_ground.astype(int), "crit_rank": crit_rank.astype(int),
})
inc = m.groupby("uuid").agg(
    scene_state=("scene_state","first"), scene_county=("scene_county","first"),
    month_key=("month_key","first"), month_label=("month_label","first"),
    is_trauma_impr=("is_trauma_impr","max"), is_stroke_impr=("is_stroke_impr","max"),
    is_ob_impr=("is_ob_impr","max"), ls=("ls","max"), is_ground=("is_ground","max"),
    crit_rank=("crit_rank","max"),
).reset_index()
print(f"rows {len(m)} -> incidents {len(inc)} (rows-per-incident {len(m)/max(len(inc),1):.2f})")

# COMMAND ----------

# ---- 5. Condition, criticality, air-eligible flag ----

inc = inc.merge(vitals_uuid, on="uuid", how="left").merge(narr_uuid, on="uuid", how="left")
for c in ["rts_present","stroke_scale_present"]: inc[c] = inc[c].fillna(False).astype(bool)

inc["is_trauma"] = (inc["is_trauma_impr"]==1) | inc["rts_present"]
inc["is_stroke"] = (inc["is_stroke_impr"]==1) | inc["stroke_scale_present"]
inc["is_ob"]     = (inc["is_ob_impr"]==1)
inc["is_peds"]   = inc["age_years"] < PEDS_MAX_AGE_YEARS
inc["is_target"] = inc[["is_trauma","is_stroke","is_ob","is_peds"]].any(axis=1)
inc["criticality"] = inc["crit_rank"].map(RANK_LABEL)
def condition(r):
    if r["is_trauma"]: return "Trauma"
    if r["is_stroke"]: return "Stroke"
    if r["is_ob"]:     return "OB"
    if r["is_peds"]:   return "Pediatrics"
    return "Other"
inc["condition"] = inc.apply(condition, axis=1)
inc["air_eligible"] = (inc["is_ground"]==1) & (inc["ls"]==1) & inc["is_target"]

MONTH_ORDER = inc[["month_key","month_label"]].drop_duplicates().sort_values("month_key")["month_label"].tolist()
TIMEFRAME = f"{MONTH_ORDER[0]}-{MONTH_ORDER[-1]} ({len(MONTH_ORDER)} months)" if MONTH_ORDER else "unknown"

# COMMAND ----------

# ---- 6. Two-slide data per county ----

COND_ORDER = ["Trauma","Stroke","OB","Pediatrics"]
CRIT_ORDER = ["Critical (Red)","Emergent (Yellow)","Lower Acuity (Green)","Not Recorded"]

def slide_data(county):
    cty  = inc[(inc["scene_state"]==TARGET_STATE) & (inc["scene_county"]==county)]
    cand = cty[cty["air_eligible"]]
    nm = max(cty["month_key"].nunique(), 1)
    n_long, n_cand = len(cty), len(cand)

    # 1. Volume
    volume = pd.DataFrame([
        {"metric":"Long-ground transports (all)", "period_total":n_long, "avg_per_month":round(n_long/nm,1)},
        {"metric":"Air-eligible (meets filters)", "period_total":n_cand, "avg_per_month":round(n_cand/nm,1)},
    ])
    monthly = (cand.groupby("month_label").size().reindex(MONTH_ORDER).fillna(0).astype(int)
               .rename("air_eligible_trips").reset_index())  # optional trend

    # 2. Condition (air-eligible cohort)
    cond = (cand["condition"].value_counts().reindex(COND_ORDER).fillna(0).astype(int)
            .rename_axis("condition").reset_index(name="trips"))
    cond["pct"] = (cond["trips"]/max(n_cand,1)*100).round(1)

    # 3. Criticality (air-eligible cohort)
    crit = (cand["criticality"].value_counts().reindex(CRIT_ORDER).fillna(0).astype(int)
            .rename_axis("criticality").reset_index(name="trips"))
    crit["pct"] = (crit["trips"]/max(n_cand,1)*100).round(1)

    # 4. Narrative
    avg = n_cand/nm
    red = int(crit.loc[crit["criticality"]=="Critical (Red)","trips"].sum())
    yel = int(crit.loc[crit["criticality"]=="Emergent (Yellow)","trips"].sum())
    ry_pct = round((red+yel)/max(n_cand,1)*100)
    top_cond = cond.sort_values("trips", ascending=False).iloc[0]["condition"] if n_cand>0 else "n/a"
    if avg >= MEANINGFUL_PER_MONTH:
        b1 = (f"{n_cand} air-eligible long-ground transports over {TIMEFRAME} (~{round(avg)}/month) — "
              f"meaningful ground volume representing potential air-conversion opportunity.")
    else:
        b1 = (f"Only {n_cand} air-eligible long-ground transports over {TIMEFRAME} (~{round(avg,1)}/month) — "
              f"limited convertible ground volume, consistent with low long-trip activity in this market.")
    b2 = (f"Clinical mix led by {top_cond}; {ry_pct}% of candidates were high-acuity (Red/Yellow)."
          if n_cand>0 else "Insufficient volume for a clinical-mix readout.")
    return volume, monthly, cond, crit, [b1, b2]

packs, narratives = {}, []
for i, cty in enumerate(FOCUS_COUNTIES, 1):
    volume, monthly, cond, crit, bullets = slide_data(cty)
    packs[cty] = (volume, monthly, cond, crit, bullets)
    narratives.append({"slide": i, "county": cty, "bullet_1": bullets[0], "bullet_2": bullets[1]})
    print(f"\n==================  SLIDE {i}: {cty.upper()} COUNTY  ({TIMEFRAME})  ==================")
    print("1. GROUND TRANSPORT VOLUME"); show(volume)
    print("   (optional) air-eligible by month:"); show(monthly)
    print("2. DISTRIBUTION BY CONDITION (air-eligible cohort)"); show(cond)
    print("3. DISTRIBUTION BY CRITICALITY (air-eligible cohort)"); show(crit)
    print("4. NARRATIVE")
    for b in bullets: print("   -", b)

# COMMAND ----------

# ---- 7. Save one workbook (a sheet per county + a narrative sheet) ----

os.makedirs(OUTPUT_DIR, exist_ok=True)
try:
    with pd.ExcelWriter(f"{OUTPUT_DIR}/two_slide_data.xlsx", engine="openpyxl") as xw:
        pd.DataFrame(narratives).to_excel(xw, sheet_name="narratives", index=False)
        for cty in FOCUS_COUNTIES:
            volume, monthly, cond, crit, _ = packs[cty]
            tag = cty.replace(" ", "_")[:25]; r0 = 0
            for tbl in [volume, monthly, cond, crit]:
                tbl.to_excel(xw, sheet_name=tag, index=False, startrow=r0); r0 += len(tbl) + 2
    print("Wrote two_slide_data.xlsx to", OUTPUT_DIR)
except Exception as e:
    print("Excel write skipped:", e)
