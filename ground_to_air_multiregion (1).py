# Databricks notebook source

# ============================================================================
# Ground-to-Air — Two-Slide Data Pack (LA & San Diego)  ·  TRAVEL-TIME GATE
#
# CHANGE FROM PRIOR VERSION (per Mukund call):
#   - Eligibility is now gated on TRAVEL TIME, not mileage.
#   - Field used: "Elapsed Call Time (Arrive Destination Time - Enroute Time)"
#     (minutes). Keep trips with ELAPSED_MIN_LOW <= elapsed <= ELAPSED_MIN_HIGH.
#       * lower bound 60 min  -> drops short trips an aircraft would never serve
#       * upper bound 200 min -> trims outliers / skew Mukund flagged in the data
#   - New reference metric: AVERAGE travel time (min) per county for eligible trips
#     (he explicitly dropped average mileage).
#   - All six Main files share identical columns -> load all of them.
#   - A units sanity-check auto-corrects the classic Excel "day-fraction" storage
#     (e.g. 0.0486 instead of 70 min) before the gate is applied.
#
# Everything else matches the validated pipeline: dedup by NEMSIS UUID,
# state/county normalization (CA == California), corrected segment flags,
# transport-leg Lights & Sirens, condition & criticality readouts.
# ============================================================================

# COMMAND ----------

# ---- 0. Config ----

# %pip install openpyxl

MAIN_DIR   = "/Workspace/Users/josh.smitherman@gmr.net/ground_to_air_analysis/data"
OTHER_DIR  = "/Workspace/Users/josh.smitherman@gmr.net/ground_to_air_analysis/other_data"
OUTPUT_DIR = "/Workspace/Users/josh.smitherman@gmr.net/ground_to_air_analysis/two_slide_output"

# --- Target air bases -> (county, state). One slide per target. ---
# Mukund's batch = Northeast + Pacific + West (6 slides). South included but off by default.
TARGETS = [
    {"region":"Pacific",   "base":"RCH UCLA",                    "county":"Los Angeles", "state":"California"},
    {"region":"Pacific",   "base":"RCH Oceanside",               "county":"San Diego",   "state":"California"},
    {"region":"Northeast", "base":"MTC Colona (Medforce I)",     "county":"Henry",       "state":"Illinois"},
    {"region":"Northeast", "base":"MTC Burlington (Medforce II)","county":"Des Moines",  "state":"Iowa"},
    {"region":"West",      "base":"RCH Porterville",             "county":"Tulare",      "state":"California"},
    {"region":"West",      "base":"RCH Merced",                  "county":"Merced",      "state":"California"},
    {"region":"South",     "base":"AEL Murfreesboro (StatFlight)","county":"Rutherford", "state":"Tennessee"},
    {"region":"South",     "base":"MTC Birmingham (Childrens)",  "county":"Jefferson",   "state":"Alabama"},
]
ACTIVE_REGIONS = {"Northeast", "Pacific", "West", "South"}   # all targets; Mukund asked to confirm NE + South

LS_REQUIRE_TRANSPORT_LEG = True
PEDS_MAX_AGE_YEARS = 5
MEANINGFUL_PER_MONTH = 25   # narrative threshold: >= this = "meaningful volume", else "limited"

# --- travel-time gate (NEW) ---
ELAPSED_MIN_LOW  = 60       # keep trips with travel time >= this many minutes
ELAPSED_MIN_HIGH = 200      # ... and <= this many minutes (trim outliers); set None to disable upper bound

# --- eligibility definition toggles (flip if Ops/Dave Lyons revise the criteria) ---
REQUIRE_SEGMENT      = True   # require high-equity clinical segment (Trauma/Stroke/OB/Peds). Validated default.
REQUIRE_HIGH_ACUITY  = False  # also require Red/Yellow acuity. Mukund said "...and critical"; turn on to enforce.

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
    # --- travel time (NEW) + eTimes fallbacks to compute it if the column is absent ---
    "elapsed_call":    {"nemsis": None,              "keywords": ["elapsed call time", "arrive destination time - enroute"]},
    "t_enroute":       {"nemsis": "etimes.05",       "keywords": ["en route date time"]},
    "t_arrive_dest":   {"nemsis": "etimes.11",       "keywords": ["arrived at destination date time", "patient arrived at destination"]},
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

STATE_MAP = {"al":"Alabama","ak":"Alaska","az":"Arizona","ar":"Arkansas","ca":"California","co":"Colorado",
             "ct":"Connecticut","de":"Delaware","fl":"Florida","ga":"Georgia","hi":"Hawaii","id":"Idaho",
             "il":"Illinois","in":"Indiana","ia":"Iowa","ks":"Kansas","ky":"Kentucky","la":"Louisiana",
             "me":"Maine","md":"Maryland","ma":"Massachusetts","mi":"Michigan","mn":"Minnesota","ms":"Mississippi",
             "mo":"Missouri","mt":"Montana","ne":"Nebraska","nv":"Nevada","nh":"New Hampshire","nj":"New Jersey",
             "nm":"New Mexico","ny":"New York","nc":"North Carolina","nd":"North Dakota","oh":"Ohio","ok":"Oklahoma",
             "or":"Oregon","pa":"Pennsylvania","ri":"Rhode Island","sc":"South Carolina","sd":"South Dakota",
             "tn":"Tennessee","tx":"Texas","ut":"Utah","vt":"Vermont","va":"Virginia","wa":"Washington",
             "wv":"West Virginia","wi":"Wisconsin","wy":"Wyoming"}
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

# ---- travel-time parsing (NEW) ----
def parse_minutes(v):
    """Best-effort -> elapsed minutes. Handles raw minutes, H:MM[:SS], and numeric strings."""
    if v is None: return np.nan
    if isinstance(v, (int, float)):
        return np.nan if (isinstance(v, float) and np.isnan(v)) else float(v)
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"): return np.nan
    if re.fullmatch(r"\d+(\.\d+)?", s): return float(s)
    m = re.fullmatch(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", s)          # H:MM or H:MM:SS
    if m: return int(m.group(1))*60 + int(m.group(2)) + int(m.group(3) or 0)/60.0
    return pd.to_numeric(s, errors="coerce")

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

# ---- guard: fail clearly if no Main files loaded (prevents a cryptic KeyError later) ----
if main.empty:
    all_xlsx = sorted(glob.glob(os.path.join(MAIN_DIR, "*.xlsx")))
    print("!! No Main data loaded — nothing to analyze.")
    print("   MAIN_DIR =", MAIN_DIR)
    print("   .xlsx files at that path:", [os.path.basename(f) for f in all_xlsx] or "(NONE — path is wrong or empty)")
    try:
        parent = os.path.dirname(MAIN_DIR.rstrip("/"))
        print("   sibling folders in parent:", [os.path.basename(p) for p in sorted(glob.glob(os.path.join(parent, "*")))][:20])
    except Exception:
        pass
    raise FileNotFoundError(
        f"No '*Main*.xlsx' files loaded from {MAIN_DIR}. "
        "Point MAIN_DIR at the folder holding the Transport-Distances-Main files "
        "(the path is absolute, so it must be valid from THIS notebook's context), then re-run."
    )

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

# ---- 4. Flags, travel time, dedup to incident, normalize ----

R = resolve_columns(main)
def col(name):
    c = R.get(name)
    return main[c] if c else pd.Series([None]*len(main), index=main.index)

# --- travel time: use Elapsed Call Time column; fall back to (arrive dest - enroute) eTimes ---
if R.get("elapsed_call"):
    elapsed_src = "Elapsed Call Time column"
    elapsed_min_raw = col("elapsed_call").map(parse_minutes)
else:
    elapsed_src = "computed from eTimes.11 - eTimes.05"
    t_dest = pd.to_datetime(col("t_arrive_dest"), errors="coerce")
    t_enr  = pd.to_datetime(col("t_enroute"), errors="coerce")
    elapsed_min_raw = (t_dest - t_enr).dt.total_seconds() / 60.0
print("Travel-time source:", elapsed_src)

# units sanity check: if values look like Excel day-fractions (median < 3), scale to minutes
_pos = elapsed_min_raw.dropna(); _pos = _pos[_pos > 0]
if len(_pos) and _pos.median() < 3:
    print(f"NOTE: median elapsed = {_pos.median():.4f} -> looks like day-fractions; multiplying x1440 to get minutes.")
    elapsed_min_raw = elapsed_min_raw * 1440.0
    _pos = elapsed_min_raw.dropna(); _pos = _pos[_pos > 0]
if len(_pos):
    print("elapsed (min)  median=%.1f  p25=%.1f  p75=%.1f  min=%.1f  max=%.1f"
          % (_pos.median(), _pos.quantile(.25), _pos.quantile(.75), _pos.min(), _pos.max()))

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
    "elapsed_min": pd.to_numeric(elapsed_min_raw, errors="coerce"),
})
inc = m.groupby("uuid").agg(
    scene_state=("scene_state","first"), scene_county=("scene_county","first"),
    month_key=("month_key","first"), month_label=("month_label","first"),
    is_trauma_impr=("is_trauma_impr","max"), is_stroke_impr=("is_stroke_impr","max"),
    is_ob_impr=("is_ob_impr","max"), ls=("ls","max"), is_ground=("is_ground","max"),
    crit_rank=("crit_rank","max"), elapsed_min=("elapsed_min","max"),
).reset_index()
print(f"rows {len(m)} -> incidents {len(inc)} (rows-per-incident {len(m)/max(len(inc),1):.2f})")

# COMMAND ----------

# ---- 5. Condition, criticality, time gate, air-eligible flag ----

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

# --- travel-time gate (NEW) ---
hi = np.inf if ELAPSED_MIN_HIGH is None else ELAPSED_MIN_HIGH
inc["time_eligible"] = inc["elapsed_min"].between(ELAPSED_MIN_LOW, hi)

# base eligibility (pre-time) for the funnel readout, then apply the time gate
base = (inc["is_ground"]==1) & (inc["ls"]==1)
if REQUIRE_SEGMENT:     base = base & inc["is_target"]
if REQUIRE_HIGH_ACUITY: base = base & (inc["crit_rank"] >= 2)
inc["base_eligible"] = base
inc["air_eligible"]  = base & inc["time_eligible"]

MONTH_ORDER = inc[["month_key","month_label"]].drop_duplicates().sort_values("month_key")["month_label"].tolist()
TIMEFRAME = f"{MONTH_ORDER[0]}-{MONTH_ORDER[-1]} ({len(MONTH_ORDER)} months)" if MONTH_ORDER else "unknown"
GATE_TXT = f"{ELAPSED_MIN_LOW}-{'inf' if ELAPSED_MIN_HIGH is None else ELAPSED_MIN_HIGH} min travel time"
print("Eligibility:", "ground + L&S"
      + (" + clinical segment" if REQUIRE_SEGMENT else "")
      + (" + Red/Yellow acuity" if REQUIRE_HIGH_ACUITY else "")
      + f" + [{GATE_TXT}]")

# COMMAND ----------

# ---- 6. Two-slide data per county ----

COND_ORDER = ["Trauma","Stroke","OB","Pediatrics"]
CRIT_ORDER = ["Critical (Red)","Emergent (Yellow)","Lower Acuity (Green)","Not Recorded"]

def time_buckets(s):
    hi_lbl = "inf" if ELAPSED_MIN_HIGH is None else ELAPSED_MIN_HIGH
    return pd.DataFrame([
        {"bucket": "missing travel time",            "trips": int(s.isna().sum())},
        {"bucket": f"< {ELAPSED_MIN_LOW} (dropped)", "trips": int((s < ELAPSED_MIN_LOW).sum())},
        {"bucket": f"{ELAPSED_MIN_LOW}-{hi_lbl} (kept)",
         "trips": int(s.between(ELAPSED_MIN_LOW, np.inf if ELAPSED_MIN_HIGH is None else ELAPSED_MIN_HIGH).sum())},
        {"bucket": f"> {hi_lbl} (dropped)",
         "trips": 0 if ELAPSED_MIN_HIGH is None else int((s > ELAPSED_MIN_HIGH).sum())},
    ])

def slide_data(county, state):
    cty  = inc[(inc["scene_state"]==state) & (inc["scene_county"]==county)]
    base = cty[cty["base_eligible"]]          # pre-time-gate (for the funnel)
    cand = cty[cty["air_eligible"]]           # final eligible cohort
    nm = max(cty["month_key"].nunique(), 1)
    n_long, n_base, n_cand = len(cty), len(base), len(cand)

    # 1. Volume  (long-ground context + air-eligible after the time gate)
    volume = pd.DataFrame([
        {"metric":"Long-ground transports (all)",        "period_total":n_long, "avg_per_month":round(n_long/nm,1)},
        {"metric":"Air-eligible (meets filters)",        "period_total":n_cand, "avg_per_month":round(n_cand/nm,1)},
    ])

    # 1b. Travel-time funnel (how the gate trimmed the pre-time eligible set)
    funnel = time_buckets(base["elapsed_min"])

    # 1c. Average travel time (NEW reference metric) on the final eligible cohort
    et = cand["elapsed_min"].dropna()
    time_summary = pd.DataFrame([{
        "trips_eligible": n_cand,
        "trips_with_time": int(et.notna().sum()),
        "avg_travel_min":  round(et.mean(), 1) if len(et) else None,
        "median_travel_min": round(et.median(), 1) if len(et) else None,
        "min_travel_min":  round(et.min(), 1) if len(et) else None,
        "max_travel_min":  round(et.max(), 1) if len(et) else None,
    }])

    # 1d. Air-eligible by month (trend)
    monthly = (cand.groupby("month_label").size().reindex(MONTH_ORDER).fillna(0).astype(int)
               .rename("air_eligible_trips").reset_index())

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
    avg_t = time_summary["avg_travel_min"].iloc[0]
    avg_t_txt = f" Average travel time ~{avg_t:.0f} min." if avg_t is not None else ""
    if avg >= MEANINGFUL_PER_MONTH:
        b1 = (f"{n_cand} air-eligible long-ground transports over {TIMEFRAME} (~{round(avg)}/month), "
              f"gated to {GATE_TXT} — meaningful ground volume and a potential air-conversion opportunity.{avg_t_txt}")
    else:
        b1 = (f"Only {n_cand} air-eligible long-ground transports over {TIMEFRAME} (~{round(avg,1)}/month), "
              f"gated to {GATE_TXT} — limited convertible ground volume in this market.{avg_t_txt}")
    b2 = (f"Clinical mix led by {top_cond}; {ry_pct}% of candidates were high-acuity (Red/Yellow)."
          if n_cand>0 else "Insufficient volume for a clinical-mix readout.")
    return volume, funnel, time_summary, monthly, cond, crit, [b1, b2]

active = [t for t in TARGETS if t["region"] in ACTIVE_REGIONS]

# ---- 6a. COVERAGE REPORT — distinguish "state not in extract" from a true post-filter zero ----
# rows_in_data == 0 AND state absent  -> CANNOT confirm "no ground transports" (data just isn't in this pull)
# rows_in_data == 0 AND state present -> genuine 0 long-ground transports for that county
# rows_in_data  > 0                   -> ground transports exist; base/air_eligible show how many qualify
states_present = set(inc["scene_state"].dropna().unique())
cov = []
for t in active:
    st = norm_state(t["state"]); cty_n = norm_county(t["county"])
    cty = inc[(inc["scene_state"]==st) & (inc["scene_county"]==cty_n)]
    n = len(cty)
    state_here = st in states_present
    if n > 0:
        status = "ground transports present"
    elif not state_here:
        status = "STATE NOT IN EXTRACT - cannot confirm; needs data pull"
    else:
        status = "0 long-ground transports (state present in data)"
    cov.append({
        "region": t["region"], "base": t["base"], "county": f'{t["county"]}, {t["state"]}',
        "state_in_extract": "yes" if state_here else "no",
        "rows_in_data": n,
        "base_eligible": int(cty["base_eligible"].sum()),
        "air_eligible": int(cty["air_eligible"].sum()),
        "status": status,
    })
coverage = pd.DataFrame(cov)
print("================  COVERAGE REPORT  ================")
print(f"States present in extract: {sorted(states_present)}")
show(coverage)

# COMMAND ----------

# ---- 6b. Two-slide data per target county (write ALL, including zero-volume, so nothing is silently missing) ----

packs, narratives = {}, []
for i, t in enumerate(active, 1):
    cty_norm, st_norm = norm_county(t["county"]), norm_state(t["state"])
    key = f'{t["county"]}, {t["state"]}'
    volume, funnel, time_summary, monthly, cond, crit, bullets = slide_data(cty_norm, st_norm)
    n_cand = int(volume.loc[volume["metric"].str.startswith("Air-eligible"), "period_total"].iloc[0])
    packs[key] = (t, volume, funnel, time_summary, monthly, cond, crit, bullets)
    narratives.append({"slide": i, "region": t["region"], "base": t["base"], "county": key,
                       "air_eligible": n_cand, "bullet_1": bullets[0], "bullet_2": bullets[1]})
    flag = "" if n_cand > 0 else "   [ZERO after filter — see coverage tab for reason]"
    print(f"\n==============  {i}. {t['county'].upper()} COUNTY, {t['state']}  "
          f"[{t['region']} / {t['base']}]  ({TIMEFRAME}){flag}  ==============")
    print("1. GROUND TRANSPORT VOLUME"); show(volume)
    print("1b. TRAVEL-TIME FUNNEL"); show(funnel)
    print("1c. TRAVEL TIME"); show(time_summary)
    print("   air-eligible by month:"); show(monthly)
    print("2. CONDITION"); show(cond)
    print("3. CRITICALITY"); show(crit)
    print("4. NARRATIVE")
    for b in bullets: print("   -", b)

# COMMAND ----------

# ---- 7. Save one workbook (coverage + narratives + a sheet per county, zero-volume included) ----

os.makedirs(OUTPUT_DIR, exist_ok=True)
try:
    with pd.ExcelWriter(f"{OUTPUT_DIR}/multiregion_slide_data.xlsx", engine="openpyxl") as xw:
        coverage.to_excel(xw, sheet_name="coverage", index=False)
        pd.DataFrame(narratives).to_excel(xw, sheet_name="narratives", index=False)
        for key, (t, volume, funnel, time_summary, monthly, cond, crit, _) in packs.items():
            tag = re.sub(r"[^A-Za-z0-9]+", "_", key)[:28]; r0 = 0
            # note row at top so a zero sheet is self-explanatory
            note = pd.DataFrame([{"note": f'{t["region"]} / {t["base"]} — {key}'}])
            note.to_excel(xw, sheet_name=tag, index=False, startrow=r0); r0 += len(note) + 2
            for tbl in [volume, funnel, time_summary, monthly, cond, crit]:
                tbl.to_excel(xw, sheet_name=tag, index=False, startrow=r0); r0 += len(tbl) + 2
    print("Wrote multiregion_slide_data.xlsx to", OUTPUT_DIR, "| county sheets:", len(packs))
except Exception as e:
    print("Excel write skipped:", e)
