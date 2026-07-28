# =============================================================================
# CPMS Non-Identity — Threshold Review (single-cell)
# Jira IDEN-43344 | reflects 2026-07-14 threshold review meeting
#
# Meeting takeaways baked in:
#   * Operating threshold is 0.94 (not 0.90). 0.90 is the floor for sparse records.
#   * Priority follow-up: quantify FP risk for pairs matched on NAME + DOB ONLY,
#     and find the break-even threshold where those stop producing false positives.
#   * DOB permutations matter: ~1.5% of records carry multiple DOBs; the
#     07/20/1969 default is only part of it.
#
# Constraints: read-only Unity Catalog (no table writes); no sparkContext/rdd;
# no GraphFrames; ES-dump tables (fields nested in _source). Outputs: charts +
# tables inline, plus one Excel workbook to the data folder.
# Paste this whole thing into ONE Databricks cell and run.
# =============================================================================

# ------------------------- pip installs --------------------------------------
%pip install --quiet openpyxl xlsxwriter scipy pandas matplotlib

# -----------------------------------------------------------------------------
import os, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt
from pyspark.sql import functions as F, Window as W
from pyspark.sql.types import StructType, StringType, ArrayType

# ============================ CONFIG =========================================
CATALOG, SCHEMA = "eciscor_prod", "pcis_metadata"
PARTIES_TBL = f"{CATALOG}.{SCHEMA}.elasticdump_parties"
PAIRS_TBL   = f"{CATALOG}.{SCHEMA}.elasticdump_party_matches"
PROD_ID_TBL = f"{CATALOG}.{SCHEMA}.elasticdump_identities"

DATA_DIR   = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/irq/data"
SCRATCH    = "dbfs:/tmp/irq_threshold_cache"
LABELS_CSV = f"{DATA_DIR}/threshold_review_labels.csv"   # optional: left_id,right_id,score,label
os.makedirs(DATA_DIR, exist_ok=True)

OPERATING_THRESHOLD = 0.94          # decided in the 2026-07-14 review
FLOOR_THRESHOLD     = 0.90          # sparse-record floor
THRESHOLDS = [round(x, 2) for x in np.arange(0.90, 0.981, 0.01)]

SENTINEL_DOB   = ["1969-07-20", "07/20/1969", "19690720", "1969-07-20T00:00:00"]
SENTINEL_NAMES = ["UNKNOWN", "NONE", "N/A", "TEST"]

MATERIALIZE, REFRESH_CACHE = True, True   # True once to rebuild the bad cache; set back to False after
# One-time: set True to wipe the scratch cache from earlier (broken) runs.
CLEAR_CACHE_ONCE = False
# ---- SPEED CONTROL ----------------------------------------------------------
# DEV_MODE samples the RAW tables BEFORE the expensive JSON parse/explode, so the
# heavy work only touches sampled rows. Target: full notebook < 30 min.
DEV_MODE        = True
PAIR_SAMPLE_FRAC   = 0.03     # 3% of match records (~150k pairs). Lower = faster.
PARTY_SAMPLE_FRAC  = 0.10     # 10% of parties for fill-rate/DOB stats
DEV_SEED           = 42
# Attribute join needs BOTH sides of a sampled pair to be in the parties slice.
# So for the join we keep ALL parties but only the ids referenced by sampled
# pairs (a semi-join), which is cheap. Fill-rate/DOB use the PARTY_SAMPLE slice.
SCHEMA_SAMPLE_ROWS = 200   # rows sampled to infer JSON schema (A# still appears)

# ---- PIN the correct columns (confirmed from 01_resolved_columns) -----------
# The MLaaS score lives on partySearch.candidates[].candidateScore, NOT on
# dedup_results.rules[].partiesWithScores[].score. Force the candidates array so
# the id columns line up with the parties join key.
MANUAL_EXPLODE_COL = "partySearch_candidates"
MANUAL_COLS = {
    "score":    "partySearch_candidates_candidateScore",
    "left_id":  "partySearch_partySourceId",
    "right_id": "partySearch_candidates_partySourceId",
}
MANUAL_NON_IDENTITY_FILTER = None

spark.conf.set("spark.sql.ansi.enabled", "false")  # empty groups -> null, not job abort
spark.conf.set("spark.sql.shuffle.partitions", "64")  # cap shuffle parallelism for a small sample
xls_sheets = {}     # collected tables -> Excel
def band_expr(c):
    s = F.col(c)
    return (F.when(s >= 0.98, "1_match").when(s >= 0.95, "2_goldilocks")
             .when(s >= 0.90, "3_near_match").when(s >= 0.50, "4_higher")
             .when(s >= 0.20, "5_lower").otherwise("6_non_match"))

def show_table(pdf, name):
    xls_sheets[name] = pdf
    print(f"\n----- {name} -----")
    try: display(pdf)
    except Exception: print(pdf.to_string(index=False))
    return pdf

def show_chart(fig):
    try: display(fig)
    except Exception: pass
    plt.close(fig)

# ============================ FLATTEN ========================================
def _infer_ddl(df, n=SCHEMA_SAMPLE_ROWS):
    s = (df.select("_source").where(F.col("_source").isNotNull() & (F.trim("_source") != ""))
           .limit(n).toPandas()["_source"].tolist())
    ddl = spark.range(1).select(F.schema_of_json(F.lit("[" + ",".join(x.strip() for x in s) + "]")).alias("s")).collect()[0]["s"]
    return ddl[6:-1] if ddl.upper().startswith("ARRAY<") else ddl

def parse_source(df):
    t = df.schema["_source"].dataType
    if isinstance(t, StructType): return df.select("_id", "_index", "_source.*")
    return df.withColumn("_s", F.from_json("_source", _infer_ddl(df))).select("_id", "_index", "_s.*")

def expand_structs(df, depth=8):
    for _ in range(depth):
        structs = [f.name for f in df.schema.fields if isinstance(f.dataType, StructType)]
        if not structs: break
        cols = []
        for f in df.schema.fields:
            if f.name in structs:
                for ch in f.dataType.fieldNames():
                    cols.append(F.col(f"`{f.name}`.`{ch}`").alias(f"{f.name}_{ch}"))
            else:
                cols.append(F.col(f"`{f.name}`"))
        df = df.select(*cols)
    return df

def _n(s): return re.sub(r"[^a-z0-9]", "", s.lower())
SCORE_BEST = ["candidatescore", "matchprobability", "mlaasscore"]
SCORE_OK   = ["score", "matchscore", "probability"]
def _srank(name):
    x = _n(name.split("_")[-1])
    if x in SCORE_BEST: return 0
    if x in SCORE_OK:   return 1
    return 2 if "score" in x else None
def scalar_score(df):
    """A usable scalar score column already present. Requires an exact leaf
    (candidateScore / score / matchProbability), not just a 'score' substring,
    so we don't stop early on partiesWithScores-style columns."""
    best, br = None, 99
    for f in df.schema.fields:
        if isinstance(f.dataType, (ArrayType, StructType)): continue
        leaf = _n(f.name.split("_")[-1])
        r = 0 if leaf in SCORE_BEST else (1 if leaf in SCORE_OK else None)
        if r is not None and r < br: best, br = f.name, r
    return best if br <= 1 else None
def _best_leaf_rank_in_element(elem):
    """Best score-rank among the LEAF fields of an array element struct,
    recursing one level so nested arrays are considered by their own leaves."""
    best = 99
    for sub in elem.fields:
        if isinstance(sub.dataType, StructType):
            for s2 in sub.dataType.fields:
                r = _srank(s2.name)
                if r is not None: best = min(best, r)
        elif isinstance(sub.dataType, ArrayType) and isinstance(sub.dataType.elementType, StructType):
            for s2 in sub.dataType.elementType.fields:
                r = _srank(s2.name)
                if r is not None: best = min(best, r + 1)   # penalise deeper nesting
        else:
            r = _srank(sub.name)
            if r is not None: best = min(best, r)
    return best

def find_score_array(df):
    """Pick the array whose element carries the BEST (lowest-rank) score leaf.
    Rank 0 = candidateScore beats rank 1 = a bare 'score' field. This stops the
    exploder wandering into dedup_results.rules[].partiesWithScores."""
    best, br = None, 99
    for f in df.schema.fields:
        if isinstance(f.dataType, ArrayType) and isinstance(f.dataType.elementType, StructType):
            r = _best_leaf_rank_in_element(f.dataType.elementType)
            if r < br: best, br = f.name, r
    return best
def explode_score_array(df, rounds=4):
    df = expand_structs(df)
    for _ in range(rounds):
        tgt = MANUAL_EXPLODE_COL or (None if scalar_score(df) else find_score_array(df))
        if not tgt: break
        print("  exploding score array:", tgt)
        df = expand_structs(df.withColumn(tgt, F.explode_outer(F.col(f"`{tgt}`"))))
        if MANUAL_EXPLODE_COL: break
    return df

def materialize(df, name):
    if not MATERIALIZE: return df
    # Key the cache on the flattened column signature. If the flatten logic
    # changes (e.g. a different array gets exploded), the signature changes and
    # the old parquet is bypassed - no more stale-cache surprises.
    import hashlib
    sig = hashlib.md5((",".join(sorted(df.columns))).encode()).hexdigest()[:8]
    path = f"{SCRATCH}/{name}_{sig}"
    if not REFRESH_CACHE:
        try:
            out = spark.read.parquet(path)
            if len(out.columns) > 0:
                print("cache hit:", path); return out
        except Exception:
            pass
    print("caching:", path)
    df.write.mode("overwrite").parquet(path)
    return spark.read.parquet(path)

if CLEAR_CACHE_ONCE:
    try:
        dbutils.fs.rm(SCRATCH, recurse=True); print("cleared cache:", SCRATCH)
    except Exception as _e:
        print("cache clear skipped:", _e)

# Sample the RAW tables BEFORE parsing JSON. This is the main speed lever:
# from_json + explode only run on the sampled rows, not all 5M+ records.
pairs_raw   = spark.table(PAIRS_TBL)
parties_raw = spark.table(PARTIES_TBL)
ident_raw   = spark.table(PROD_ID_TBL)

if DEV_MODE:
    pairs_raw   = pairs_raw.sample(False, PAIR_SAMPLE_FRAC, seed=DEV_SEED)
    parties_smp = parties_raw.sample(False, PARTY_SAMPLE_FRAC, seed=DEV_SEED)
    print(f"DEV_MODE: pairs {PAIR_SAMPLE_FRAC:.0%}, parties {PARTY_SAMPLE_FRAC:.0%}")
else:
    parties_smp = parties_raw

# Parse + flatten (now on sampled volume in dev mode)
pairs_flat        = explode_score_array(parse_source(pairs_raw))
parties_flat_full = expand_structs(parse_source(parties_raw))     # all parties (for the join)
parties_flat      = expand_structs(parse_source(parties_smp))     # sampled (for fill/DOB stats)
ident_flat        = expand_structs(parse_source(ident_raw))

tag = "dev" if DEV_MODE else "full"
pairs_flat        = materialize(pairs_flat, f"pairs_{tag}")
parties_flat      = materialize(parties_flat, f"parties_{tag}")
ident_flat        = materialize(ident_flat, "identities")
# parties_flat_full stays lazy; it is semi-joined to sampled pair ids below,
# so only the needed party rows are ever read.

# ============================ RESOLVE COLUMNS ================================
SYN = {
 "party_id":["partysourceid","partyid","partyidentifier","id"],
 "score":SCORE_BEST+SCORE_OK, "left_id":["partysourceid","declaredpartyid","instigatingpartyid"],
 "right_id":["candidatepartysourceid","candidateid","candidatepartyid"],
 "first":["firstname","givenname"],"middle":["middlename"],"last":["lastname","surname"],
 "dob":["dateofbirth","dob","birthdate"],"a_number":["anumber","alienumber","aliennumber"],
 "ssn":["ssn"],"fin":["fin"],"eid":["eid"],"i94":["i94"],
 "coc":["countryofcitizenship"],"cob":["countryofbirth"],"receipt":["receipt","receiptnumber"],
 "form":["formtype","formnumber"],"status":["recordstatus","status","partystatus"],
 "identity_id":["identityid"],
}
ES_META = {"_id","_index","_score","_type","_source"}
NO_SUB = {"eid","i94","ssn","fin","cob","coc","dob","form"}
def _leaf(c): return _n(c.split("_")[-1])
def resolve(cols, key):
    cols = [c for c in cols if c not in ES_META]; cands = SYN.get(key, [])
    for cand in cands:
        for c in cols:
            if _leaf(c) == cand: return c
    for cand in cands:
        for c in cols:
            if _n(c) == cand: return c
    if key in NO_SUB: return None
    for cand in cands:
        for c in cols:
            if cand in _n(c): return c
    return None

COLS = {"score": resolve(pairs_flat.columns, "score")}
def _isid(c):
    if c in ES_META: return False
    x = _n(c); return ("partysourceid" in x or "partyid" in x or "candidateid" in x) and "identityid" not in x
ids = [c for c in pairs_flat.columns if _isid(c)]
sp = COLS["score"].rsplit("_", 1)[0] + "_" if COLS.get("score") and "_" in COLS["score"] else ""
same = [c for c in ids if sp and c.startswith(sp)]
other = [c for c in ids if c not in same]
COLS["right_id"] = same[0] if same else (sorted([c for c in ids if "candidate" in _n(c)], key=len)[:1] or [None])[0]
COLS["left_id"]  = (sorted(other, key=lambda c: (0 if "partysearch" in _n(c) else 1, len(c)))[:1] or [None])[0]
for k in ["party_id","first","middle","last","dob","a_number","ssn","fin","eid","i94","coc","cob","receipt","form","status"]:
    COLS[k] = resolve(parties_flat.columns, k)
IDENT_COLS = {k: resolve(ident_flat.columns, k) for k in ["party_id","identity_id"]}
COLS.update(MANUAL_COLS)
# hard-pin the three pair columns so nothing overrides them
for _k in ('score','left_id','right_id'):
    if _k in MANUAL_COLS: COLS[_k] = MANUAL_COLS[_k]
STRONG_IDS = [k for k in ["a_number","ssn","fin"] if COLS.get(k)]

show_table(pd.DataFrame([{"field":k,"resolved_to":v} for k,v in COLS.items()]), "01_resolved_columns")
missing = [k for k in ["score","left_id","right_id"] if not COLS.get(k)]
assert not missing, f"unresolved required PAIRS columns {missing}; PAIRS cols: {pairs_flat.columns}"

NON_IDENTITY_FILTER = MANUAL_NON_IDENTITY_FILTER
if NON_IDENTITY_FILTER is None and COLS.get("status"):
    NON_IDENTITY_FILTER = f"lower(`{COLS['status']}`) like '%non%identity%'"

# ============================ LOAD ==========================================
parties = parties_flat.where(NON_IDENTITY_FILTER) if NON_IDENTITY_FILTER else parties_flat
pairs = pairs_flat.withColumn("score", F.col(COLS["score"]).cast("double")).withColumn("band", band_expr("score"))

# cache the two frames we count repeatedly, then count once each
parties = parties.cache(); pairs = pairs.cache()
_pt = parties_flat_full.count() if not DEV_MODE else None
_pn, _sp = parties.count(), pairs.count()
show_table(pd.DataFrame([
    {"metric":"parties_total","value":(_pt if _pt is not None else "(dev sample - not counted)")},
    {"metric":"parties_non_identity","value":_pn},
    {"metric":"scored_pairs","value":_sp},
]), "00_row_counts")

# ============================ 1. FILL RATES =================================
attrs = [k for k in ["first","middle","last","dob","a_number","ssn","fin","eid","i94","coc","cob","receipt"] if COLS.get(k)]
n = max(_pn, 1)
fill = (parties.select([(F.count(F.when(F.col(COLS[k]).isNotNull() & (F.trim(F.col(COLS[k]).cast("string"))!=""),1))/F.lit(n)).alias(k) for k in attrs])
        .toPandas().T.reset_index()); fill.columns = ["attribute","fill_rate"]
show_table(fill, "02_fill_rates")
fig, ax = plt.subplots(figsize=(9,4))
ax.bar(fill["attribute"], fill["fill_rate"].astype(float),
       color=["#c0392b" if float(v)<0.5 else "#2e86c1" for v in fill["fill_rate"]])
ax.axhline(0.5, ls="--", c="gray"); ax.set_title("Attribute fill rate"); ax.set_ylabel("filled")
plt.xticks(rotation=45, ha="right"); show_chart(fig)

# ============================ 2. DOB PERMUTATIONS ===========================
# Meeting: ~1.5% of records carry multiple DOBs. Measure it directly + sentinel.
if COLS.get("dob") and COLS.get("party_id"):
    dob_stats = (parties.groupBy(COLS["party_id"])
                 .agg(F.countDistinct(COLS["dob"]).alias("distinct_dobs"))
                 .agg(F.avg((F.col("distinct_dobs")>1).cast("int")).alias("pct_multi_dob"),
                      F.max("distinct_dobs").alias("max_dobs")).toPandas())
    sent = parties.select(
        F.count(F.when(F.col(COLS["dob"]).cast("string").isin(SENTINEL_DOB),1)).alias("sentinel_dob_records")
    ).toPandas()
    show_table(pd.concat([dob_stats, sent], axis=1), "03_dob_quality")

# null sentinels before comparison
parties_clean = parties
if COLS.get("dob"):
    parties_clean = parties_clean.withColumn(COLS["dob"],
        F.when(F.col(COLS["dob"]).cast("string").isin(SENTINEL_DOB), None).otherwise(F.col(COLS["dob"])))

# ============================ 3. SCORE BANDS ================================
bc = show_table(pairs.groupBy("band").count().orderBy("band").toPandas(), "04_score_bands")
fig, ax = plt.subplots(figsize=(8,4)); ax.bar(bc["band"], bc["count"], color="#2e86c1")
ax.set_title("Score-band distribution"); plt.xticks(rotation=30, ha="right"); show_chart(fig)

# ============================ 4. EVIDENCE (what agrees) =====================
ajoin = [k for k in ["first","middle","last","dob","a_number","ssn","fin"] if COLS.get(k)]
pid = COLS.get("party_id") or "_id"
# Build the join source from the FULL parties frame, but restrict to just the
# party ids referenced by the (sampled) pairs via a semi-join. Cheap, and it
# guarantees sampled pairs can resolve their attributes (no blank evidence).
join_src = parties_flat_full
if NON_IDENTITY_FILTER:
    join_src = join_src.where(NON_IDENTITY_FILTER)
if COLS.get("dob"):
    join_src = join_src.withColumn(COLS["dob"],
        F.when(F.col(COLS["dob"]).cast("string").isin(SENTINEL_DOB), None).otherwise(F.col(COLS["dob"])))
needed_ids = (pairs.select(F.col(COLS["left_id"]).alias("pid"))
                   .union(pairs.select(F.col(COLS["right_id"]).alias("pid"))).distinct())
join_src = join_src.join(needed_ids, F.col(pid) == F.col("pid"), "left_semi")

L = join_src.select([F.col(pid).alias("_lid")] + [F.col(COLS[k]).alias(f"{k}_l") for k in ajoin])
R = join_src.select([F.col(pid).alias("_rid")] + [F.col(COLS[k]).alias(f"{k}_r") for k in ajoin])
dx = pairs.join(L, pairs[COLS["left_id"]]==F.col("_lid"), "left").join(R, pairs[COLS["right_id"]]==F.col("_rid"), "left")
def agree(f):
    if f not in ajoin: return F.lit(None).cast("int")
    return F.when(F.col(f"{f}_l").isNull() | F.col(f"{f}_r").isNull(), None).otherwise((F.col(f"{f}_l")==F.col(f"{f}_r")).cast("int"))
for f in ["dob","a_number","ssn","fin","last","first"]:
    dx = dx.withColumn(f"agree_{f}", agree(f))
strong = F.lit(0)
for k in STRONG_IDS: strong = strong + F.coalesce(F.col(f"agree_{k}"), F.lit(0))
# name+DOB-only flag = the cohort the meeting flagged as risky
dx = (dx.withColumn("n_strong_agree", strong)
        .withColumn("name_dob_only",
            ((F.col("agree_last")==1) & (F.col("agree_dob")==1) & (F.col("n_strong_agree")==0)).cast("int")))
# Persist the decorated pairs ONCE. Every section below reads this parquet
# instead of recomputing the two attribute joins + explode. Biggest single win.
dx = materialize(dx.select(
        "score", "band", "name_dob_only", "n_strong_agree",
        "agree_dob", "agree_last", "agree_a_number", "agree_ssn", "agree_fin",
        COLS["left_id"], COLS["right_id"]),
    "decorated_pairs_dev" if DEV_MODE else "decorated_pairs")
print("decorated pairs:", dx.count())

near = dx.filter(F.col("band").isin("2_goldilocks","3_near_match"))
near_n = near.count()
if near_n == 0:
    ap = pd.DataFrame({"signal":["(near-match band is empty)"],"rate_in_near_band":[float("nan")]})
else:
    ap = (near.agg(F.avg("agree_dob").alias("dob"), F.avg("agree_last").alias("last"),
                   F.avg("agree_a_number").alias("a_number"), F.avg("agree_ssn").alias("ssn"),
                   F.avg("agree_fin").alias("fin"),
                   F.avg("name_dob_only").alias("name_dob_only_share"),
                   # try_divide-safe: guard the denominator explicitly
                   (F.count(F.when(F.col("n_strong_agree")>0,1)) / F.lit(near_n)).alias("any_strong_id"))
            .toPandas().T.reset_index()); ap.columns = ["signal","rate_in_near_band"]
show_table(ap, "05_evidence_near_band")
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(ap["signal"], pd.to_numeric(ap["rate_in_near_band"], errors="coerce").fillna(0), color="#8e44ad")
ax.set_title("What agrees in the 0.90–0.98 band"); plt.xticks(rotation=30, ha="right"); show_chart(fig)

# ============================ 5. NAME+DOB-ONLY BREAK-EVEN ====================
# THE meeting action item: for pairs matched on name+DOB only, at each threshold
# how many get accepted, and (if labels exist) what is the false-positive rate?
# The break-even = lowest threshold where name+DOB-only FP rate hits 0.
rows = []
labels = None
if os.path.exists(LABELS_CSV):
    labels = pd.read_csv(LABELS_CSV)
    labels["score"] = labels["score"].astype(float); labels["label"] = labels["label"].astype(int)

# One aggregation over the name+DOB-only pairs: a conditional count per
# threshold. Replaces a 9-pass Python loop with a single Spark job.
nd = dx.filter(F.col("name_dob_only")==1)
agg = nd.agg(*[F.count(F.when(F.col("score") >= t, 1)).alias(f"t_{int(round(t*100))}") for t in THRESHOLDS]).toPandas()
nd_pdf = pd.DataFrame({"threshold": THRESHOLDS,
                       "name_dob_only_accepted": [int(agg[f"t_{int(round(t*100))}"][0]) for t in THRESHOLDS]})

if labels is not None and "name_dob_only" in labels.columns:
    lab_nd = labels[labels["name_dob_only"] == 1]
    fp = []
    for t in THRESHOLDS:
        acc = lab_nd[lab_nd["score"] >= t]
        k_fp = int((acc["label"] == 0).sum()); nn = len(acc)
        fp.append({"threshold": t, "reviewed": nn, "false_positives": k_fp,
                   "fp_rate": (k_fp/nn if nn else np.nan)})
    nd_pdf = nd_pdf.merge(pd.DataFrame(fp), on="threshold", how="left")
    clean = nd_pdf[(nd_pdf["reviewed"] > 0) & (nd_pdf["fp_rate"] == 0)]
    be = clean["threshold"].min() if len(clean) else np.nan
    print(f"\n>>> Name+DOB-only break-even threshold (first FP-free): {be}")
else:
    print("\n(name+DOB-only FP rate needs labels with a name_dob_only column; showing volumes only)")

show_table(nd_pdf, "06_name_dob_only_by_threshold")
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(nd_pdf["threshold"], nd_pdf["name_dob_only_accepted"], marker="o", color="#c0392b")
for xv in (FLOOR_THRESHOLD, OPERATING_THRESHOLD):
    ax.axvline(xv, ls="--", c="gray"); ax.text(xv, ax.get_ylim()[1]*0.9, str(xv), rotation=90, va="top", fontsize=8)
ax.invert_xaxis(); ax.set_xlabel("threshold"); ax.set_ylabel("name+DOB-only pairs accepted")
ax.set_title("Name+DOB-only volume by threshold (0.90 floor vs 0.94 operating)"); show_chart(fig)

# ============================ 6. PRECISION/RECALL + CI =======================
def wilson(k, nn, z=1.96):
    if nn == 0: return (np.nan, np.nan)
    p = k/nn; d = 1+z*z/nn; c = p+z*z/(2*nn); m = z*sqrt(p*(1-p)/nn + z*z/(4*nn*nn))
    return (c-m)/d, (c+m)/d
if labels is not None:
    rows = []
    for t in THRESHOLDS:
        pred = (labels["score"] >= t).astype(int)
        tp = int(((pred==1)&(labels["label"]==1)).sum()); fp = int(((pred==1)&(labels["label"]==0)).sum())
        fn = int(((pred==0)&(labels["label"]==1)).sum())
        prec = tp/(tp+fp) if tp+fp else np.nan; rec = tp/(tp+fn) if tp+fn else np.nan
        lo, hi = wilson(fp, tp+fp)
        rows.append({"threshold":t,"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,
                     "fp_rate":(fp/(tp+fp) if tp+fp else np.nan),"fp_ci_low":lo,"fp_ci_high":hi})
    pr = show_table(pd.DataFrame(rows), "07_precision_recall")
    fig, ax = plt.subplots(figsize=(8,4))
    for m_, c_ in [("precision","#c0392b"),("recall","#2e86c1")]:
        ax.plot(pr["threshold"], pr[m_], marker="o", label=m_, color=c_)
    for xv in (FLOOR_THRESHOLD, OPERATING_THRESHOLD): ax.axvline(xv, ls="--", c="gray")
    ax.invert_xaxis(); ax.legend(); ax.set_title("Precision / recall vs threshold"); show_chart(fig)
else:
    lo, hi = wilson(1, 2237)
    show_table(pd.DataFrame([{"note":"no labels file; using meeting figure",
                              "observed_fp":1,"n_reviewed":2237,"fp_rate":1/2237,
                              "fp_ci_low":lo,"fp_ci_high":hi}]), "07_precision_recall")

# ============================ 7. TIERED RULE ================================
rel = ((F.col("agree_a_number")==1) & (F.col("agree_dob")==0)) if (COLS.get("a_number") and COLS.get("dob")) else F.lit(False)
dec = (dx.withColumn("relative_risk", rel).withColumn("decision",
        F.when(F.col("relative_risk"),"review_relative")
         .when((F.col("score")>=OPERATING_THRESHOLD) & (F.col("n_strong_agree")>0),"accept")
         .when(F.col("score")>=0.98,"accept")
         .when(F.col("score")>=OPERATING_THRESHOLD,"review_thin_evidence")
         .otherwise("reject")))
show_table(dec.withColumn("flat", F.when(F.col("score")>=OPERATING_THRESHOLD,"accept").otherwise("reject"))
              .groupBy("flat","decision").count().orderBy("flat","decision").toPandas(), "08_tiered_vs_flat")
dc = show_table(dec.groupBy("decision").count().orderBy(F.desc("count")).toPandas(), "08_decision_counts")
fig, ax = plt.subplots(figsize=(7,4)); ax.bar(dc["decision"], dc["count"], color="#e67e22")
ax.set_title(f"Tiered-rule decisions at {OPERATING_THRESHOLD}"); plt.xticks(rotation=20, ha="right"); show_chart(fig)

# ============================ 8. CLUSTER OVER-MERGE =========================
def connected_components(edges):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as scc
    e = edges.select("src","dst").where(F.col("src").isNotNull() & F.col("dst").isNotNull() & (F.col("src")!=F.col("dst"))).distinct()
    if e.count()==0: return None
    ep = e.toPandas()
    codes, uniq = pd.factorize(pd.concat([ep["src"], ep["dst"]], ignore_index=True))
    h = len(ep); n = len(uniq)
    g = coo_matrix((np.ones(h, np.int8), (codes[:h], codes[h:])), shape=(n,n))
    _, lab = scc(g, directed=False)
    return pd.DataFrame({"id": uniq, "component": lab})

stat = []
for t in [OPERATING_THRESHOLD, FLOOR_THRESHOLD]:
    e = dec.filter(F.col("score")>=t).select(F.col(COLS["left_id"]).alias("src"), F.col(COLS["right_id"]).alias("dst"))
    cc = connected_components(e)
    if cc is None: continue
    sizes = cc.groupby("component").size()
    stat.append({"threshold":t,"clusters":int(sizes.shape[0]),"max_cluster":int(sizes.max()),
                 "p99_cluster":int(sizes.quantile(0.99)),"mean_cluster":round(float(sizes.mean()),2)})
if stat:
    cs = show_table(pd.DataFrame(stat), "09_cluster_stats")
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(cs["threshold"].astype(str), cs["max_cluster"], color="#16a085")
    ax.set_title("Largest identity (over-merge risk) by threshold"); ax.set_ylabel("parties"); show_chart(fig)

# ============================ EXCEL EXPORT ==================================
xlsx_path = f"{DATA_DIR}/cpms_threshold_results.xlsx"
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
    idx = pd.DataFrame({"sheet": list(xls_sheets.keys())})
    idx.to_excel(xw, sheet_name="index", index=False)
    for name, pdf in xls_sheets.items():
        pdf.to_excel(xw, sheet_name=name[:31], index=False)
print("\nExcel written:", xlsx_path)
print("sheets:", list(xls_sheets.keys()))
