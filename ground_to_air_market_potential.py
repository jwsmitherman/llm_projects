# Databricks notebook source

# ============================================================================
# Ground-to-Air — County Focus & Market Potential (LA & San Diego slide pack)
#
# Purpose: first-pass signal for the underperforming-base case ("don't shut LA/San Diego —
# there is unrealized air volume in long ground trips"). Produces, per focus county:
#   1. Overall long-ground transport volume (total + by month, target timeframe).
#   2. Condition distribution (Trauma / Stroke / OB / Pediatrics / Other) across long trips.
#   3. Criticality distribution (Critical-Red / Emergent-Yellow / Lower-Green).
#   4. Air-candidate subset (passes Lights&Sirens + clinical segment) + conversion sensitivity.
# Plus a QA/diagnostics section (count re-verification, county attribution, normalization audit,
# filter funnel) and an explicit LIMITATIONS section.
#
# Focus counties: Los Angeles, San Diego.  Corridor recheck: Riverside, Imperial, San Bernardino.
# Carries corrected v2 logic: dedup by NEMSIS UUID, fixed segment flags, narrative age,
# transport-leg Lights & Sirens, and state/county normalization (CA == California).
# ============================================================================

# COMMAND ----------

# ---- 0. Config ----

# %pip install openpyxl

MAIN_DIR  = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/data"
OTHER_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/other_data"
OUTPUT_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_air_analysis/output_counties"

FOCUS_COUNTIES   = ["Los Angeles", "San Diego"]                 # one slide each
CONTEXT_COUNTIES = ["Riverside", "Imperial", "San Bernardino"]  # corridor recheck
TARGET_STATE = "California"

DEDUP_BY_UUID = True
LS_REQUIRE_TRANSPORT_LEG = True
ACUITY_FILTER = False               # criticality is reported as a distribution, not used to drop trips
PEDS_MAX_AGE_YEARS = 5
MIN_DISTANCE_MILES = 50             # long-trip threshold (applied only if a distance column exists)
CONVERSION_RATES = [0.10, 0.25, 0.50]   # share of air-candidates that might realistically convert to air

# COMMAND ----------

# ---- 1. Resolver, helpers, normalization ----

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
    "agency_name":     {"nemsis": "eresponse.02",    "keywords": ["ems agency name"]},
    "narrative":       {"nemsis": "enarrative.01",   "keywords": ["patient care report narrative", "narrative"]},
    "stroke_scale_sc": {"nemsis": "evitals.29",      "keywords": ["stroke scale score"]},
    "stroke_scale_ty": {"nemsis": "evitals.30",      "keywords": ["stroke scale type"]},
    "rts":             {"nemsis": "evitals.33",      "keywords": ["revised trauma score"]},
    "distance":        {"nemsis": None,              "keywords": ["distance", "mileage", "miles"]},
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

STATE_MAP = {
    "al":"Alabama","ak":"Alaska","az":"Arizona","ar":"Arkansas","ca":"California","co":"Colorado",
    "ct":"Connecticut","de":"Delaware","fl":"Florida","ga":"Georgia","hi":"Hawaii","id":"Idaho",
    "il":"Illinois","in":"Indiana","ia":"Iowa","ks":"Kansas","ky":"Kentucky","la":"Louisiana",
    "me":"Maine","md":"Maryland","ma":"Massachusetts","mi":"Michigan","mn":"Minnesota","ms":"Mississippi",
    "mo":"Missouri","mt":"Montana","ne":"Nebraska","nv":"Nevada","nh":"New Hampshire","nj":"New Jersey",
    "nm":"New Mexico","ny":"New York","nc":"North Carolina","nd":"North Dakota","oh":"Ohio","ok":"Oklahoma",
    "or":"Oregon","pa":"Pennsylvania","ri":"Rhode Island","sc":"South Carolina","sd":"South Dakota",
    "tn":"Tennessee","tx":"Texas","ut":"Utah","vt":"Vermont","va":"Virginia","wa":"Washington",
    "wv":"West Virginia","wi":"Wisconsin","wy":"Wyoming",
}
def norm_state(v):
    s = _norm(v)
    if s in STATE_MAP: return STATE_MAP[s]
    return str(v).strip().title() if v is not None and str(v).strip() else None

def norm_county(v):
    if v is None or str(v).strip() == "": return None
    return re.sub(r"\s+", " ", str(v)).strip().title()

MONTH_MAP = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
             "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
def month_from_name(fname):
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*-?\s*(\d{4})", fname.lower())
    if m: return f"{m.group(2)}-{MONTH_MAP[m.group(1)]}", f"{m.group(1).capitalize()} {m.group(2)}"
    return "unknown", os.path.splitext(fname)[0]

def pick_data_sheet(sheets):
    best, best_score = None, -1
    for name, df in sheets.items():
        if df is None or df.shape[0] < 1: continue
        r = resolve_columns(df)
        score = sum(1 for v in r.values() if v) + 0.0001 * df.shape[0]
        if r.get("scene_county"): score += 5
        if r.get("uuid"): score += 5
        if score > best_score: best, best_score = name, score
    return best or list(sheets.keys())[0]

# COMMAND ----------

# ---- 2. Load Main + supporting files ----

def load_main(data_dir):
    frames, manifest = [], []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.xlsx"))):
        if "main" not in os.path.basename(p).lower(): continue
        sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        sheet = pick_data_sheet(sheets)
        df = sheets[sheet].dropna(how="all").copy()
        key, label = month_from_name(os.path.basename(p))
        df["__month_key"], df["__month_label"] = key, label
        frames.append(df)
        manifest.append({"file": os.path.basename(p), "rows": df.shape[0], "month": label})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest)

def load_support(data_dir, token):
    frames = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.xlsx"))):
        if token not in os.path.basename(p).lower(): continue
        sheets = pd.read_excel(p, sheet_name=None, dtype=str)
        frames.append(sheets[pick_data_sheet(sheets)].dropna(how="all").copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

main, manifest = load_main(MAIN_DIR)
vitals_raw = load_support(OTHER_DIR, "vitals")
narr_raw   = load_support(OTHER_DIR, "narrative")
print(f"Main rows: {len(main)} | vitals rows: {len(vitals_raw)} | narrative rows: {len(narr_raw)}")
show(manifest)

# COMMAND ----------

# ---- 3. Supporting -> per-UUID vitals flags + narrative age ----

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
    vitals_uuid = v.groupby("uuid").agg(rts_present=("rts_present","max"),
                                        stroke_scale_present=("stroke_scale_present","max")).reset_index()
else:
    vitals_uuid = pd.DataFrame(columns=["uuid","rts_present","stroke_scale_present"])

AGE_PATTERNS = [
    (r"(\d{1,3})\s*(?:y/?o|yo|year[s\-\s]*old|yr[s]?[\s\-]*old)", 365.0),
    (r"(\d{1,3})\s*(?:m/?o|month[s\-\s]*old)", 30.4),
    (r"(\d{1,3})\s*(?:w/?o|wk|week[s\-\s]*old)", 7.0),
    (r"(\d{1,3})\s*(?:d/?o|day[s\-\s]*old)", 1.0),
]
def parse_age_years(text):
    if text is None or (isinstance(text, float) and np.isnan(text)): return np.nan
    t = str(text).lower()
    if re.search(r"\bnewborn\b|\bneonat", t): return 0.0
    for pat, days in AGE_PATTERNS:
        m = re.search(pat, t)
        if m: return float(m.group(1)) * days / 365.0
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

# ---- 4. Flags, dedup to incident, normalize state/county ----

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

# long-trip threshold: applied only if a distance column is actually present
if R.get("distance"):
    dist = pd.to_numeric(col("distance"), errors="coerce")
    long_trip = (dist >= MIN_DISTANCE_MILES).fillna(False)
    DISTANCE_AVAILABLE = True
else:
    long_trip = pd.Series(True, index=main.index)
    DISTANCE_AVAILABLE = False

ACUITY_RANK = {"critical (red)": 3, "emergent (yellow)": 2, "lower acuity (green)": 1}
def acuity_rank(s): return s.fillna("").map(_norm).map(lambda x: ACUITY_RANK.get(x, 0))
crit_rank = pd.concat([acuity_rank(col("initial_acuity")), acuity_rank(col("final_acuity"))], axis=1).max(axis=1)
RANK_LABEL = {3: "Critical (Red)", 2: "Emergent (Yellow)", 1: "Lower Acuity (Green)", 0: "Not Recorded"}

uuid_series = col("uuid")
uuid_key = uuid_series.where(uuid_series.notna(), col("incident_number"))
uuid_key = uuid_key.where(uuid_key.notna(), pd.Series(main.index.astype(str), index=main.index))

m = pd.DataFrame({
    "uuid": uuid_key,
    "raw_state": col("scene_state"),
    "raw_county": col("scene_county"),
    "primary_impr": col("primary_impr"),
    "scene_state": col("scene_state").map(norm_state),
    "scene_county": col("scene_county").map(norm_county),
    "month_key": main["__month_key"], "month_label": main["__month_label"],
    "incident_number": col("incident_number"),
    "is_trauma_impr": (impr.str.contains(TRAUMA_RX) & ~no_apparent).astype(int),
    "is_stroke_impr": impr.str.contains(STROKE_RX).astype(int),
    "is_ob_impr": impr.str.contains(OB_RX).astype(int),
    "ls": ls.astype(int), "is_ground": is_ground.astype(int), "long_trip": long_trip.astype(int),
    "crit_rank": crit_rank.astype(int),
})

if DEDUP_BY_UUID:
    inc = m.groupby("uuid").agg(
        scene_state=("scene_state","first"), scene_county=("scene_county","first"),
        month_key=("month_key","first"), month_label=("month_label","first"),
        incident_number=("incident_number","first"),
        is_trauma_impr=("is_trauma_impr","max"), is_stroke_impr=("is_stroke_impr","max"),
        is_ob_impr=("is_ob_impr","max"), ls=("ls","max"), is_ground=("is_ground","max"),
        primary_impr=("primary_impr","first"),
        long_trip=("long_trip","max"), crit_rank=("crit_rank","max"),
    ).reset_index()
else:
    inc = m.copy()
print(f"rows {len(m)} -> incidents {len(inc)} (rows-per-incident {len(m)/max(len(inc),1):.2f})")

# COMMAND ----------

# ---- 5. Final flags: condition mix, criticality, air-candidate ----

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

# A trip is an AIR CANDIDATE if it is a long ground transport (the dataset), transported lights & sirens,
# and falls in a target clinical segment.  (Acuity is a reported distribution, not a hard filter by default.)
acuity_ok = inc["crit_rank"].isin([2,3]) if ACUITY_FILTER else True
inc["air_candidate"] = (inc["long_trip"]==1) & (inc["is_ground"]==1) & (inc["ls"]==1) & inc["is_target"] & acuity_ok

MONTH_ORDER = (inc[["month_key","month_label"]].drop_duplicates().sort_values("month_key")["month_label"].tolist())
TIMEFRAME = f"{MONTH_ORDER[0]} to {MONTH_ORDER[-1]} ({len(MONTH_ORDER)} months)" if MONTH_ORDER else "unknown"

# COMMAND ----------

# ---- 6. QA / DIAGNOSTICS (re-verify counts, attribution, normalization, filter funnel) ----

print("TIMEFRAME:", TIMEFRAME)
print("DEDUP rows-per-incident:", round(len(m)/max(len(inc),1), 2))
print("DISTANCE column found:", R.get("distance") or "NONE  ->  '>50 mi long-trip' is ASSUMED pre-applied (cannot validate from these files)")

# 6a. State normalization audit (proves CA and California merge)
ALL_COUNTIES = FOCUS_COUNTIES + CONTEXT_COUNTIES
state_audit = (m[m["scene_county"].isin(ALL_COUNTIES)]
               .groupby(["raw_state","scene_state"]).size().reset_index(name="rows")
               .sort_values("rows", ascending=False))
print("\nState normalization audit (raw -> normalized) for target counties:")
show(state_audit)

# 6b. Filter funnel per focus + corridor county (count re-verification)
def funnel(county):
    s = inc[(inc["scene_state"]==TARGET_STATE) & (inc["scene_county"]==county)]
    return {
        "county": county,
        "long_ground_trips": int(len(s)),
        "ground": int((s["is_ground"]==1).sum()),
        "ground_+_L&S": int(((s["is_ground"]==1) & (s["ls"]==1)).sum()),
        "ground_+_L&S_+_segment (air_candidate)": int(s["air_candidate"].sum()),
    }
funnel_df = pd.DataFrame([funnel(c) for c in ALL_COUNTIES]).sort_values("long_ground_trips", ascending=False)
print("\nFilter funnel by county (re-verify counts; Riverside vs Imperial sanity check):")
show(funnel_df)

# COMMAND ----------

# ---- 7. Corridor context: total long-ground vs air-candidate, by county x month ----

corr = inc[(inc["scene_state"]==TARGET_STATE) & (inc["scene_county"].isin(ALL_COUNTIES))].copy()

def county_month_pivot(df):
    p = pd.pivot_table(df, index="scene_county", columns="month_label", values="uuid",
                       aggfunc="count", margins=True, margins_name="Total")
    mc = [x for x in MONTH_ORDER if x in p.columns]
    p = p.reindex(columns=mc + (["Total"] if "Total" in p.columns else [])).fillna(0).astype(int)
    return p.sort_values("Total", ascending=False).reset_index()

print("Long-ground trips by county x month:")
show(county_month_pivot(corr))

print("Air-candidate trips by county x month:")
show(county_month_pivot(corr[corr["air_candidate"]]))

# COMMAND ----------

# ---- 8. Per-county slide pack (FOCUS counties) ----

def pct_table(series, total, name):
    t = series.value_counts().rename_axis(name).reset_index(name="trips")
    t["pct"] = (t["trips"] / max(total, 1) * 100).round(1)
    return t

def county_pack(county):
    cty = inc[(inc["scene_state"]==TARGET_STATE) & (inc["scene_county"]==county)].copy()
    cand = cty[cty["air_candidate"]].copy()
    n_long = len(cty); n_cand = len(cand); nm = max(cty["month_key"].nunique(), 1)

    monthly_long = (cty.groupby("month_label")["uuid"].count()
                    .reindex(MONTH_ORDER).fillna(0).astype(int).rename("long_ground_trips").reset_index())
    monthly_cand = (cand.groupby("month_label")["uuid"].count()
                    .reindex(MONTH_ORDER).fillna(0).astype(int).rename("air_candidate_trips").reset_index())

    condition_dist = pct_table(cty["condition"], n_long, "condition")          # over ALL long trips
    criticality_dist = pct_table(cty["criticality"], n_long, "criticality")    # over ALL long trips

    ann_cand = round(n_cand / nm * 12)
    conv = pd.DataFrame([{
        "capture_rate": f"{int(r*100)}%",
        "potential_air_trips_per_year": round(ann_cand * r),
    } for r in CONVERSION_RATES])

    summary = pd.DataFrame([{
        "county": county, "timeframe": TIMEFRAME,
        "long_ground_trips_total": n_long, "long_ground_avg_per_month": round(n_long/nm, 1),
        "air_candidate_trips_total": n_cand, "air_candidate_avg_per_month": round(n_cand/nm, 1),
        "air_candidate_annualized": ann_cand,
    }])
    return summary, monthly_long, monthly_cand, condition_dist, criticality_dist, conv

packs = {}
for cty in FOCUS_COUNTIES:
    summary, monthly_long, monthly_cand, condition_dist, criticality_dist, conv = county_pack(cty)
    packs[cty] = (summary, monthly_long, monthly_cand, condition_dist, criticality_dist, conv)
    print(f"\n================  {cty.upper()}  ================")
    show(summary)
    print("Long-ground volume by month:"); show(monthly_long)
    print("Air-candidate volume by month:"); show(monthly_cand)
    print("Condition mix (across long trips):"); show(condition_dist)
    print("Criticality mix (across long trips):"); show(criticality_dist)
    print("Air-conversion sensitivity:"); show(conv)

# COMMAND ----------

# ---- 9. DEEPER ANALYSIS (top impressions, condition x criticality) ----

for cty in FOCUS_COUNTIES:
    s = inc[(inc["scene_state"]==TARGET_STATE) & (inc["scene_county"]==cty)]
    print(f"\n----- {cty}: top primary impressions across long-ground trips -----")
    top = (s["primary_impr"].fillna("(blank)").value_counts().head(12)
           .rename_axis("primary_impression").reset_index(name="trips"))
    top["pct"] = (top["trips"] / max(len(s),1) * 100).round(1)
    show(top)
    print(f"{cty}: condition x criticality (long-ground trips)")
    ct = pd.crosstab(s["condition"], s["criticality"])
    cond_order = [c for c in ["Trauma","Stroke","OB","Pediatrics","Other"] if c in ct.index]
    crit_order = [c for c in ["Critical (Red)","Emergent (Yellow)","Lower Acuity (Green)","Not Recorded"] if c in ct.columns]
    show(ct.reindex(index=cond_order, columns=crit_order, fill_value=0).reset_index())

# COMMAND ----------

# ---- 10. CHARTS (saved as PNG for direct PowerPoint insertion) ----

import matplotlib
import matplotlib.pyplot as plt

os.makedirs(OUTPUT_DIR, exist_ok=True)
COND_ORDER = ["Trauma","Stroke","OB","Pediatrics","Other"]
COND_COLORS = {"Trauma":"#C44E52","Stroke":"#4C72B0","OB":"#8172B3","Pediatrics":"#55A868","Other":"#BBBBBB"}
CRIT_ORDER = ["Critical (Red)","Emergent (Yellow)","Lower Acuity (Green)","Not Recorded"]
CRIT_COLORS = {"Critical (Red)":"#C44E52","Emergent (Yellow)":"#E1B12C","Lower Acuity (Green)":"#55A868","Not Recorded":"#999999"}

def save_show(fig, name):
    path = f"{OUTPUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try: display(fig)
    except Exception: pass
    plt.close(fig)

# 10a. Per-county charts
for cty in FOCUS_COUNTIES:
    summary, monthly_long, monthly_cand, condition_dist, criticality_dist, conv = packs[cty]
    tag = cty.replace(" ", "_")

    # monthly volume: long-ground vs air-candidate
    lg = monthly_long.set_index("month_label").reindex(MONTH_ORDER)["long_ground_trips"].fillna(0).values
    cd = monthly_cand.set_index("month_label").reindex(MONTH_ORDER)["air_candidate_trips"].fillna(0).values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MONTH_ORDER)); w = 0.4
    ax.bar(x - w/2, lg, width=w, label="Long-ground trips", color="#4C72B0")
    ax.bar(x + w/2, cd, width=w, label="Air-candidate trips", color="#DD8452")
    for i, val in enumerate(lg): ax.text(i - w/2, val, int(val), ha="center", va="bottom", fontsize=8)
    for i, val in enumerate(cd): ax.text(i + w/2, val, int(val), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(MONTH_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Trips"); ax.set_title(f"{cty} County — Monthly Volume ({TIMEFRAME})"); ax.legend()
    save_show(fig, f"{tag}_monthly_volume")

    # condition mix pie
    cd_s = condition_dist.set_index("condition")["trips"].reindex(COND_ORDER).fillna(0)
    cd_s = cd_s[cd_s > 0]
    if cd_s.sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(cd_s.values, labels=cd_s.index, autopct="%1.0f%%", startangle=90,
               colors=[COND_COLORS[c] for c in cd_s.index], wedgeprops=dict(width=0.45))
        ax.set_title(f"{cty} County — Condition Mix (long trips, n={int(cd_s.sum())})")
        save_show(fig, f"{tag}_condition_mix")

    # criticality bar
    cr_s = criticality_dist.set_index("criticality")["trips"].reindex(CRIT_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(cr_s.index, cr_s.values, color=[CRIT_COLORS[c] for c in cr_s.index])
    for i, val in enumerate(cr_s.values): ax.text(i, val, int(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Trips"); ax.set_title(f"{cty} County — Criticality Mix (long trips)")
    ax.set_xticklabels(cr_s.index, rotation=20, ha="right")
    save_show(fig, f"{tag}_criticality_mix")

    # air-conversion sensitivity
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(conv["capture_rate"], conv["potential_air_trips_per_year"], color="#DD8452")
    for i, val in enumerate(conv["potential_air_trips_per_year"]): ax.text(i, val, int(val), ha="center", va="bottom")
    ax.set_ylabel("Potential air trips / year"); ax.set_title(f"{cty} County — Air-Conversion Potential")
    save_show(fig, f"{tag}_conversion_potential")

# 10b. Corridor comparison (all 5 counties)
tot = corr.groupby("scene_county")["uuid"].count().reindex(ALL_COUNTIES).fillna(0)
can = corr[corr["air_candidate"]].groupby("scene_county")["uuid"].count().reindex(ALL_COUNTIES).fillna(0)
order = tot.sort_values(ascending=True).index
fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(order)); h = 0.4
ax.barh(y + h/2, tot.reindex(order).values, height=h, label="Long-ground", color="#4C72B0")
ax.barh(y - h/2, can.reindex(order).values, height=h, label="Air-candidate", color="#DD8452")
ax.set_yticks(y); ax.set_yticklabels(order)
ax.set_xlabel("Trips (6-month total)"); ax.set_title(f"SoCal Corridor — Long-Ground vs Air-Candidate ({TIMEFRAME})"); ax.legend()
save_show(fig, "corridor_comparison")

print("Charts saved to:", OUTPUT_DIR)

# COMMAND ----------

# ---- 11. LIMITATIONS (state on the slide / in the readout) ----

print("""
LIMITATIONS — this is a directional first-pass signal, NOT a final decision tool:
 1. No precise pickup/drop-off coordinates in this data:
      - cannot model ground-vs-air time savings (e.g., 2 hr ground vs ~40 min rotor wing);
      - cannot confirm which trips are within practical range of a specific air base.
 2. Long-trip (>50 mi) threshold is %s here. Validate the 50-75 mi round-trip proxy with the team.
 3. Competitor "lost flights" (Eric's ADS-B analysis) are a SEPARATE data source, not included.
 4. "Air-candidate" = long ground + Lights&Sirens + clinical segment. Real air-eligibility also depends on
    local protocols, payer/contractual constraints, and clinical judgment NOT captured here.
 5. REQUIRED next step: calibrate with operations / clinical / contracts SMEs. Stakeholders may rule out
    trips on criteria we are not yet capturing. Treat these counts as a starting signal for that conversation.
""" % ("ASSUMED pre-applied (no distance column found)" if not DISTANCE_AVAILABLE else f"applied at {MIN_DISTANCE_MILES} mi"))

# COMMAND ----------

# ---- 12. Save slide-ready workbook ----

os.makedirs(OUTPUT_DIR, exist_ok=True)
try:
    with pd.ExcelWriter(f"{OUTPUT_DIR}/county_slide_pack.xlsx", engine="openpyxl") as xw:
        funnel_df.to_excel(xw, sheet_name="qa_funnel", index=False)
        for cty in FOCUS_COUNTIES:
            summary, monthly_long, monthly_cand, condition_dist, criticality_dist, conv = packs[cty]
            tag = cty.replace(" ", "_")[:25]; r0 = 0
            for tbl in [summary, monthly_long, monthly_cand, condition_dist, criticality_dist, conv]:
                tbl.to_excel(xw, sheet_name=tag, index=False, startrow=r0); r0 += len(tbl) + 2
    print("Wrote county_slide_pack.xlsx")
except Exception as e:
    print("Excel write skipped:", e)

focus = inc[(inc["scene_state"]==TARGET_STATE) & (inc["scene_county"].isin(FOCUS_COUNTIES))]
focus_cols = ["scene_state","scene_county","month_label","incident_number","condition","criticality",
              "is_ground","ls","long_trip","air_candidate","is_trauma","is_stroke","is_ob","is_peds","age_years"]
focus[focus_cols].to_csv(f"{OUTPUT_DIR}/focus_counties_linelevel.csv", index=False)
print("Outputs in:", OUTPUT_DIR)
