# Databricks notebook source
# =============================================================================
# CPMS Non-Identity - Threshold Evaluation EDA
# Jira IDEN-43344 / Confluence "2026 04 - CPMS Non-Identities"
#
# Moves the "lower the MLaaS threshold 0.98 -> 0.90" decision from an eyeballed
# sample to an evidence-decomposed analysis, and quantifies over-merge /
# identity-split at the CLUSTER level, not just pairwise.
#
# CONSTRAINTS BAKED IN:
#   * No "# MAGIC" cells.
#   * READ-ONLY on Unity Catalog. No saveAsTable / CREATE TABLE anywhere.
#     Every artifact (CSV + PNG) is written to DATA_DIR.
#   * No spark.sparkContext / df.rdd  -> blocked on shared-access clusters.
#   * No GraphFrames dependency -> connected components on the DataFrame API.
#   * Source tables are raw Elasticsearch dumps: (_id,_index,_score,_source,_type).
#     Real fields live inside the _source JSON string. NOTE: `_score` is the
#     ELASTICSEARCH relevance score, NOT the MLaaS match score.
#
# The column mapping is RESOLVED AUTOMATICALLY from the flattened schema. If it
# gets something wrong, override in MANUAL_COLS (cell 0) - nothing else changes.
# =============================================================================

# COMMAND ----------

# -----------------------------------------------------------------------------
# 0. CONFIG
# -----------------------------------------------------------------------------
import os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import functions as F, Window as W
from pyspark.sql.types import StructType, StringType, ArrayType

CATALOG = "eciscor_prod"
SCHEMA  = "pcis_metadata"

PARTIES_TBL = CATALOG + "." + SCHEMA + ".elasticdump_parties"
PAIRS_TBL   = CATALOG + "." + SCHEMA + ".elasticdump_party_matches"
PROD_ID_TBL = CATALOG + "." + SCHEMA + ".elasticdump_identities"

# All artifacts land here (the `data` folder in your workspace).
DATA_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/irq/data"
os.makedirs(DATA_DIR, exist_ok=True)
LABELS_CSV = DATA_DIR + "/threshold_review_labels.csv"   # optional reviewed pairs

# Override any auto-resolved column, e.g. {"score": "candidateScore"}.
MANUAL_COLS = {}

# Force the explode column on party_matches, or None to auto-detect.
MANUAL_EXPLODE_COL = None

# Filter to the CPMS non-identity population. Auto-built from the resolved
# status column; set explicitly to override.
MANUAL_NON_IDENTITY_FILTER = None

CURRENT_THRESHOLD  = 0.98        # production today
PROPOSED_THRESHOLD = 0.90        # what the spike is evaluating
THRESHOLDS = [0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]

SENTINEL_DOB   = ["1969-07-20", "07/20/1969", "19690720"]   # the default-DOB pattern
SENTINEL_NAMES = ["UNKNOWN", "NONE", "N/A", "TEST"]

DOB_MISMATCH_BLOCKS_MERGE = True   # father/son-sharing-an-A# guardrail
GIANT_CLUSTER_SIZE = 30            # a CPMS identity this big is suspect
SCHEMA_SAMPLE_ROWS = 500           # A# is ~6% filled; need enough rows to see it

def band_expr(c):
    s = F.col(c)
    return (F.when(s >= 0.98, "1_match").when(s >= 0.95, "2_goldilocks")
             .when(s >= 0.90, "3_near_match").when(s >= 0.50, "4_higher")
             .when(s >= 0.20, "5_lower").otherwise("6_non_match"))

SAVED = []

def save_table(df, name, show=True):
    pdf = df if isinstance(df, pd.DataFrame) else df.toPandas()
    path = DATA_DIR + "/" + name + ".csv"
    pdf.to_csv(path, index=False)
    SAVED.append(path); print("saved:", path)
    if show:
        display(pdf)
    return pdf

def save_fig(fig, name):
    path = DATA_DIR + "/" + name + ".png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    SAVED.append(path); print("saved:", path)
    plt.close(fig)

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1a. RAW SCHEMA DISCOVERY
# -----------------------------------------------------------------------------
for label, tbl in [("PARTIES", PARTIES_TBL), ("PAIRS", PAIRS_TBL), ("IDENTITIES", PROD_ID_TBL)]:
    df = spark.table(tbl)
    print("\n================ " + label + ": " + tbl + " ================")
    df.printSchema()
    df.select("_index").distinct().show(10, truncate=False)

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1b. FLATTEN _source  (+ auto-explode the nested candidate array on PAIRS)
# Schema is inferred by joining sampled _source records into ONE JSON ARRAY
# literal: schema_of_json merges keys across elements, so sparse fields (A# at
# 6% fill) survive. DataFrame API only - no sparkContext, no rdd.
# -----------------------------------------------------------------------------
def _infer_struct_ddl(df, sample_rows=SCHEMA_SAMPLE_ROWS):
    sample = (df.select("_source")
                .where(F.col("_source").isNotNull() & (F.trim(F.col("_source")) != ""))
                .limit(sample_rows).toPandas()["_source"].tolist())
    if not sample:
        raise ValueError("no non-null _source rows to infer schema from")
    arr = "[" + ",".join(s.strip() for s in sample) + "]"
    ddl = (spark.range(1).select(F.schema_of_json(F.lit(arr)).alias("s"))
                .collect()[0]["s"])
    if ddl.upper().startswith("ARRAY<") and ddl.endswith(">"):
        return ddl[len("ARRAY<"):-1]
    return ddl

def flatten_source(df, sample_rows=SCHEMA_SAMPLE_ROWS):
    t = df.schema["_source"].dataType
    if isinstance(t, StructType):
        return df.select("_id", "_index", "_source.*")
    if isinstance(t, StringType):
        ddl = _infer_struct_ddl(df, sample_rows)
        return (df.withColumn("_src", F.from_json(F.col("_source"), ddl))
                  .select("_id", "_index", "_src.*"))
    raise TypeError("unexpected _source type: " + str(t))

parties_flat = flatten_source(spark.table(PARTIES_TBL))
pairs_flat   = flatten_source(spark.table(PAIRS_TBL))
ident_flat   = flatten_source(spark.table(PROD_ID_TBL))

# party_matches likely stores one row per DECLARED party with an ARRAY of scored
# candidates nested inside (433 declared records -> 17,827 score submissions).
array_cols = [f.name for f in pairs_flat.schema.fields if isinstance(f.dataType, ArrayType)]
print("array columns on PAIRS:", array_cols)
EXPLODE_COL = MANUAL_EXPLODE_COL or (array_cols[0] if len(array_cols) == 1 else None)

if EXPLODE_COL:
    print("exploding PAIRS on '" + EXPLODE_COL + "' -> one row per scored pair")
    pairs_flat = (pairs_flat.withColumn("_m", F.explode(F.col(EXPLODE_COL)))
                            .select("*", "_m.*").drop("_m", EXPLODE_COL))

for label, df in [("PARTIES", parties_flat), ("PAIRS", pairs_flat), ("IDENTITIES", ident_flat)]:
    print("\n=== " + label + " flattened (" + str(len(df.columns)) + " cols) ===")
    print(df.columns)

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1c. AUTO-RESOLVE THE COLUMN MAP
# Matches the flattened schema against known synonyms (case/underscore
# insensitive; exact first, then substring). Prints what it resolved - eyeball
# it, and override anything wrong via MANUAL_COLS in cell 0.
# -----------------------------------------------------------------------------
SYNONYMS = {
    "party_id": ["partyid", "partyidentifier", "partykey", "id"],
    "score":    ["candidatescore", "matchprobability", "matchscore", "mlaasscore", "probability", "score"],
    "left_id":  ["declaredpartyid", "instigatingpartyid", "sourcepartyid", "leftid", "partyid"],
    "right_id": ["candidatepartyid", "candidateid", "matchedpartyid", "targetpartyid", "rightid"],
    "first":    ["firstname", "givenname", "forename"],
    "middle":   ["middlename"],
    "last":     ["lastname", "surname", "familyname"],
    "dob":      ["dateofbirth", "dob", "birthdate"],
    "a_number": ["anumber", "alienumber", "aliennumber", "alienregistrationnumber"],
    "ssn":      ["ssn", "socialsecuritynumber"],
    "fin":      ["fin", "fingerprintidnumber"],
    "eid":      ["eid", "encounterid"],
    "i94":      ["i94", "i94number"],
    "coc":      ["countryofcitizenship", "citizenshipcountry"],
    "cob":      ["countryofbirth", "birthcountry"],
    "receipt":  ["receipt", "receiptnumber"],
    "form":     ["formtype", "formnumber", "form"],
    "status":   ["status", "partystatus", "recordstatus", "identitystatus"],
    "identity_id":     ["identityid", "identitykey"],
    "identity_length": ["identitylength", "partycount", "numparties"],
}

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def resolve(cols, key):
    norm = {_norm(c): c for c in cols}
    for cand in SYNONYMS.get(key, []):
        if cand in norm:
            return norm[cand]
    for cand in SYNONYMS.get(key, []):
        for n, real in norm.items():
            if cand in n:
                return real
    return None

PAIR_KEYS  = ["score", "left_id", "right_id"]
PARTY_KEYS = ["party_id", "first", "middle", "last", "dob", "a_number", "ssn",
              "fin", "eid", "i94", "coc", "cob", "receipt", "form", "status"]
IDENT_KEYS = ["party_id", "identity_id", "identity_length"]

COLS = {}
for k in PAIR_KEYS:
    COLS[k] = resolve(pairs_flat.columns, k)
for k in PARTY_KEYS:
    COLS[k] = resolve(parties_flat.columns, k)
IDENT_COLS = {k: resolve(ident_flat.columns, k) for k in IDENT_KEYS}

# left_id and right_id must not collapse onto the same column
if COLS["left_id"] and COLS["left_id"] == COLS["right_id"]:
    alt = [c for c in pairs_flat.columns
           if _norm(c) != _norm(COLS["left_id"]) and "id" in _norm(c)]
    COLS["right_id"] = alt[0] if alt else None

COLS.update(MANUAL_COLS)

resolved = pd.DataFrame(
    [{"field": k, "resolved_to": v, "source": "PAIRS" if k in PAIR_KEYS else "PARTIES"}
     for k, v in COLS.items()] +
    [{"field": "identity." + k, "resolved_to": v, "source": "IDENTITIES"}
     for k, v in IDENT_COLS.items()])
save_table(resolved, "01_resolved_columns")

required = [k for k in PAIR_KEYS if not COLS.get(k)]
if required:
    raise ValueError(
        "Could not resolve required PAIRS columns: " + str(required) +
        "\nPAIRS columns: " + str(pairs_flat.columns) +
        "\nSet them in MANUAL_COLS (cell 0) and re-run.")

STRONG_IDS = [k for k in ["a_number", "ssn", "fin"] if COLS.get(k)]
print("resolved strong identifiers:", STRONG_IDS)

NON_IDENTITY_FILTER = MANUAL_NON_IDENTITY_FILTER
if NON_IDENTITY_FILTER is None and COLS.get("status"):
    NON_IDENTITY_FILTER = "lower(`" + COLS["status"] + "`) like '%non%identity%'"
print("non-identity filter:", NON_IDENTITY_FILTER)

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1. LOAD (read-only, flattened frames)
# -----------------------------------------------------------------------------
parties = parties_flat
if NON_IDENTITY_FILTER:
    parties = parties.where(NON_IDENTITY_FILTER)

pairs = (pairs_flat
         .withColumn("score", F.col(COLS["score"]).cast("double"))
         .withColumn("band", band_expr("score")))

meta = pd.DataFrame([
    {"metric": "parties_total",        "value": parties_flat.count()},
    {"metric": "parties_non_identity", "value": parties.count()},
    {"metric": "scored_pairs",         "value": pairs.count()},
])
save_table(meta, "00_row_counts")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 2. DATA QUALITY - fill rates, fill-by-form, sentinel contamination
# -----------------------------------------------------------------------------
attr_keys = [k for k in ["first","middle","last","dob","a_number","ssn","fin",
                         "eid","i94","coc","cob","receipt"] if COLS.get(k)]
n = parties.count()

fill = (parties.select([
            (F.count(F.when(F.col(COLS[k]).isNotNull() &
                            (F.trim(F.col(COLS[k]).cast("string")) != ""), 1)) / F.lit(max(n, 1))).alias(k)
            for k in attr_keys])
        .toPandas().T.reset_index())
fill.columns = ["attribute", "fill_rate"]
fill_pdf = save_table(fill, "02_fill_rates")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(fill_pdf["attribute"], fill_pdf["fill_rate"].astype(float),
       color=["#c0392b" if float(v) < 0.5 else "#2e86c1" for v in fill_pdf["fill_rate"]])
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_ylabel("fill rate"); ax.set_title("CPMS Non-Identity attribute completeness")
plt.xticks(rotation=45, ha="right")
save_fig(fig, "02_fill_rates")

# COMMAND ----------

# Fill-rate BY FORM: identifier density varies by form, so one global threshold
# penalizes some forms far harder than others.
if COLS.get("form"):
    id_keys = [k for k in ["a_number", "ssn", "fin", "dob"] if COLS.get(k)]
    fbf = (parties.groupBy(COLS["form"]).agg(
                F.count("*").alias("records"),
                *[(F.count(F.when(F.col(COLS[k]).isNotNull(), 1)) / F.count("*")).alias(k + "_fill")
                  for k in id_keys])
           .orderBy(F.desc("records")).limit(25))
    save_table(fbf, "02_fill_by_form")
else:
    print("no form column resolved - skipping fill-by-form")

# COMMAND ----------

# Sentinel contamination. The 1969-07-20 default DOB both fabricates agreement
# AND forces false negatives against records holding a real DOB. Null it BEFORE
# any comparison logic runs.
sent_exprs = []
if COLS.get("dob"):
    sent_exprs.append(F.count(F.when(F.col(COLS["dob"]).cast("string").isin(SENTINEL_DOB), 1)).alias("sentinel_dob"))
if COLS.get("first"):
    sent_exprs.append(F.count(F.when(F.upper(F.col(COLS["first"])).isin(SENTINEL_NAMES), 1)).alias("sentinel_first"))
if sent_exprs:
    save_table(parties.select(*sent_exprs), "02_sentinel_counts")

parties_clean = parties
if COLS.get("dob"):
    parties_clean = parties_clean.withColumn(
        COLS["dob"], F.when(F.col(COLS["dob"]).cast("string").isin(SENTINEL_DOB), None)
                      .otherwise(F.col(COLS["dob"])))
if COLS.get("first"):
    parties_clean = parties_clean.withColumn(
        COLS["first"], F.when(F.upper(F.col(COLS["first"])).isin(SENTINEL_NAMES), None)
                        .otherwise(F.col(COLS["first"])))

# COMMAND ----------

# -----------------------------------------------------------------------------
# 3. SCORE DISTRIBUTION (reconcile against the Confluence bins)
# -----------------------------------------------------------------------------
bc = save_table(pairs.groupBy("band").count().orderBy("band"), "03_score_bands")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(bc["band"], bc["count"], color="#2e86c1")
ax.set_ylabel("pairs"); ax.set_title("Score-band distribution")
plt.xticks(rotation=30, ha="right")
save_fig(fig, "03_score_bands")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 4. MATCH-WEIGHT DECOMPOSITION
# A 0.92 carried only by name+DOB is NOT the same evidence as a 0.92
# corroborated by A#. Join attributes onto each pair and build agreement flags.
# NULL = not comparable (identifier missing on one side) - the dominant case
# here, and the whole reason a flat threshold is risky.
# -----------------------------------------------------------------------------
attr_for_join = [k for k in ["first","middle","last","dob","a_number","ssn","fin"] if COLS.get(k)]
pid = COLS.get("party_id") or "_id"

L = parties_clean.select([F.col(pid).alias("_lid")] +
                         [F.col(COLS[k]).alias(k + "_l") for k in attr_for_join])
R = parties_clean.select([F.col(pid).alias("_rid")] +
                         [F.col(COLS[k]).alias(k + "_r") for k in attr_for_join])

pairs_dx = (pairs
    .join(L, pairs[COLS["left_id"]]  == F.col("_lid"), "left")
    .join(R, pairs[COLS["right_id"]] == F.col("_rid"), "left"))

def agree(field):
    if field not in attr_for_join:
        return F.lit(None).cast("int")
    l, r = F.col(field + "_l"), F.col(field + "_r")
    return F.when(l.isNull() | r.isNull(), None).otherwise((l == r).cast("int"))

for f in ["dob", "a_number", "ssn", "fin", "last", "first"]:
    pairs_dx = pairs_dx.withColumn("agree_" + f, agree(f))

strong_expr = F.lit(0)
for k in STRONG_IDS:
    strong_expr = strong_expr + F.coalesce(F.col("agree_" + k), F.lit(0))
pairs_dx = pairs_dx.withColumn("n_strong_agree", strong_expr)

near = pairs_dx.filter(F.col("band").isin("2_goldilocks", "3_near_match"))
ap = (near.agg(
        F.avg("agree_dob").alias("dob"),
        F.avg("agree_last").alias("last"),
        F.avg("agree_a_number").alias("a_number"),
        F.avg("agree_ssn").alias("ssn"),
        F.avg("agree_fin").alias("fin"),
        (F.count(F.when(F.col("n_strong_agree") > 0, 1)) / F.count("*")).alias("any_strong_id"))
      .toPandas().T.reset_index())
ap.columns = ["field", "agree_rate_near_band"]
ap_pdf = save_table(ap, "04_near_band_agreement")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(ap_pdf["field"], pd.to_numeric(ap_pdf["agree_rate_near_band"], errors="coerce").fillna(0),
       color="#8e44ad")
ax.set_ylabel("agreement rate")
ax.set_title("What actually agrees in the 0.90-0.98 near-match band")
save_fig(fig, "04_near_band_agreement")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 5. AGREEMENT-PATTERN CROSSTAB across score bands
# -----------------------------------------------------------------------------
pat = (pairs_dx.withColumn("pattern", F.concat_ws("+",
            F.when(F.col("agree_dob") == 1, "DOB"),
            F.when(F.col("agree_last") == 1, "LAST"),
            F.when(F.col("agree_a_number") == 1, "A#"),
            F.when(F.col("agree_ssn") == 1, "SSN"),
            F.when(F.col("agree_fin") == 1, "FIN")))
       .withColumn("pattern", F.when(F.col("pattern") == "", "none").otherwise(F.col("pattern")))
       .groupBy("band", "pattern").count()
       .orderBy("band", F.desc("count")))
save_table(pat, "05_agreement_patterns")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 6. PRECISION / RECALL / F1 vs THRESHOLD (from reviewed labels)
# Optional: drop threshold_review_labels.csv in DATA_DIR with columns
#   left_id,right_id,score,label   (label = 1 true match / 0 false positive)
# Falls back to the Confluence figure (1 FP in 2,237) for the CI if absent.
# -----------------------------------------------------------------------------
from math import sqrt

def wilson(k, nn, z=1.96):
    if nn == 0:
        return (float("nan"), float("nan"))
    p = k / nn
    d = 1 + z*z/nn
    c = p + z*z/(2*nn)
    m = z * sqrt(p*(1-p)/nn + z*z/(4*nn*nn))
    return (c - m)/d, (c + m)/d

if os.path.exists(LABELS_CSV):
    lp = pd.read_csv(LABELS_CSV)
    lp["score"] = lp["score"].astype(float)
    lp["label"] = lp["label"].astype(int)

    rows = []
    for t in THRESHOLDS:
        pred = (lp["score"] >= t).astype(int)
        tp = int(((pred == 1) & (lp["label"] == 1)).sum())
        fp = int(((pred == 1) & (lp["label"] == 0)).sum())
        fn = int(((pred == 0) & (lp["label"] == 1)).sum())
        prec = tp/(tp+fp) if tp+fp else float("nan")
        rec  = tp/(tp+fn) if tp+fn else float("nan")
        f1   = 2*prec*rec/(prec+rec) if (prec and rec) else float("nan")
        rows.append((t, tp, fp, fn, prec, rec, f1))
    pr = pd.DataFrame(rows, columns=["threshold","tp","fp","fn","precision","recall","f1"])
    save_table(pr, "06_pr_by_threshold")

    fig, ax = plt.subplots(figsize=(8, 4))
    for m_, c_ in [("precision", "#c0392b"), ("recall", "#2e86c1"), ("f1", "#27ae60")]:
        ax.plot(pr["threshold"], pr[m_], marker="o", label=m_, color=c_)
    ax.invert_xaxis(); ax.set_xlabel("threshold"); ax.set_ylabel("score")
    ax.set_title("Precision / recall / F1 vs threshold"); ax.legend()
    save_fig(fig, "06_pr_by_threshold")

    acc = lp[lp["score"] >= PROPOSED_THRESHOLD]
    k_fp, n_rev = int((acc["label"] == 0).sum()), len(acc)
else:
    print("no labels CSV at " + LABELS_CSV + " - using Confluence figures for the CI only")
    k_fp, n_rev = 1, 2237

lo, hi = wilson(k_fp, n_rev)
save_table(pd.DataFrame([{
    "observed_fp": k_fp,
    "n_reviewed": n_rev,
    "fp_rate": (k_fp / n_rev if n_rev else float("nan")),
    "fp_rate_ci_low": lo,
    "fp_rate_ci_high": hi}]), "06_fp_rate_ci")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 7. TIERED / RULE-AUGMENTED THRESHOLD
# Rather than one flat 0.90:
#   accept               -> >=0.90 AND a strong identifier corroborates
#   accept               -> >=0.98 (today's rule still stands on its own)
#   review_thin_evidence -> >=0.90 but carried by name+DOB alone
#   review_relative      -> A# agrees but DOB disagrees (the father/son FP)
# -----------------------------------------------------------------------------
relative_risk = F.lit(False)
if COLS.get("a_number") and COLS.get("dob") and DOB_MISMATCH_BLOCKS_MERGE:
    relative_risk = ((F.col("agree_a_number") == 1) & (F.col("agree_dob") == 0))

decisioned = (pairs_dx
    .withColumn("relative_risk", relative_risk)
    .withColumn("decision",
        F.when(F.col("relative_risk"), "review_relative")
         .when((F.col("score") >= PROPOSED_THRESHOLD) & (F.col("n_strong_agree") > 0), "accept")
         .when(F.col("score") >= CURRENT_THRESHOLD, "accept")
         .when(F.col("score") >= PROPOSED_THRESHOLD, "review_thin_evidence")
         .otherwise("reject")))

save_table(
    decisioned.withColumn("flat_090",
        F.when(F.col("score") >= PROPOSED_THRESHOLD, "accept").otherwise("reject"))
       .groupBy("flat_090", "decision").count().orderBy("flat_090", "decision"),
    "07_tiered_vs_flat")

dec = save_table(decisioned.groupBy("decision").count().orderBy(F.desc("count")),
                 "07_decision_counts", show=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(dec["decision"], dec["count"], color="#e67e22")
ax.set_ylabel("pairs"); ax.set_title("Tiered-rule decision distribution")
plt.xticks(rotation=20, ha="right")
save_fig(fig, "07_decision_counts")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 8. CLUSTER IMPACT
# Lowering the threshold ADDS EDGES; connected components then merge. The
# pairwise FP rate understates this - one bad edge can fuse two correct
# identities. This is where Huge Gain / Identity Shrink actually come from.
#
# Connected components via iterative min-label propagation on the DataFrame API
# (no GraphFrames, no sparkContext).
# -----------------------------------------------------------------------------
def connected_components(edges, max_iter=20):
    """edges: DataFrame(src, dst) -> DataFrame(id, component)."""
    v = (edges.select(F.col("src").alias("id"))
              .union(edges.select(F.col("dst").alias("id")))
              .distinct()
              .withColumn("component", F.col("id")))

    e = (edges.select("src", "dst")
              .union(edges.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
              .distinct())

    for i in range(max_iter):
        msg = (e.join(v, e["src"] == v["id"], "inner")
                .groupBy(F.col("dst").alias("id"))
                .agg(F.min("component").alias("incoming")))
        nv = (v.alias("v").join(msg.alias("m"), F.col("v.id") == F.col("m.id"), "left")
               .select(F.col("v.id").alias("id"),
                       F.least(F.col("v.component"),
                               F.coalesce(F.col("m.incoming"), F.col("v.component"))).alias("component"))
               .cache())
        changed = (nv.alias("n").join(v.alias("o"), F.col("n.id") == F.col("o.id"))
                     .where(F.col("n.component") != F.col("o.component")).count())
        v = nv
        if changed == 0:
            print("  connected components converged in " + str(i + 1) + " iterations")
            break
    return v

def build_edges(source, threshold):
    return (source.filter(F.col("score") >= threshold)
                  .select(F.col(COLS["left_id"]).alias("src"),
                          F.col(COLS["right_id"]).alias("dst"))
                  .where(F.col("src").isNotNull() & F.col("dst").isNotNull())
                  .where(F.col("src") != F.col("dst"))
                  .distinct())

stat_rows = []
sizes_at_proposed = None
for t in [CURRENT_THRESHOLD, 0.95, 0.92, PROPOSED_THRESHOLD]:
    edges = build_edges(decisioned, t)
    if edges.count() == 0:
        print("no edges at threshold " + str(t))
        continue
    sizes = connected_components(edges).groupBy("component").count().cache()
    s = sizes.agg(F.count("*").alias("n_clusters"),
                  F.max("count").alias("max_size"),
                  F.expr("percentile_approx(count, 0.99)").alias("p99_size"),
                  F.avg("count").alias("mean_size")).toPandas().iloc[0].to_dict()
    stat_rows.append(dict({"threshold": t}, **s))
    if t == PROPOSED_THRESHOLD:
        sizes_at_proposed = sizes.toPandas()
        save_table(sizes.filter(F.col("count") >= GIANT_CLUSTER_SIZE).orderBy(F.desc("count")),
                   "08_giant_clusters", show=False)

if stat_rows:
    cs = save_table(pd.DataFrame(stat_rows), "08_cluster_stats")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cs["threshold"], cs["max_size"], marker="o", color="#c0392b")
    ax.invert_xaxis(); ax.set_xlabel("threshold"); ax.set_ylabel("largest cluster (parties)")
    ax.set_title("Over-merge risk: largest identity vs threshold")
    save_fig(fig, "08_max_cluster_size")

if sizes_at_proposed is not None and len(sizes_at_proposed):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes_at_proposed["count"], bins=range(1, 41), color="#16a085")
    ax.set_xlabel("parties per cluster"); ax.set_ylabel("clusters")
    ax.set_title("Cluster-size distribution at threshold " + str(PROPOSED_THRESHOLD))
    save_fig(fig, "08_cluster_size_hist")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 8b. IDENTITY NET GAIN vs PROD (the Acceptable / Slight / Huge Gain / Shrink bins)
# net_gain = staging cluster size - current PROD identity length.
# Each staging cluster is attributed to the PROD identity most of its members
# already belong to. A cluster spanning SEVERAL PROD identities is itself the
# over-merge signal, so that is counted separately.
# -----------------------------------------------------------------------------
if IDENT_COLS.get("party_id") and IDENT_COLS.get("identity_id"):
    memb = connected_components(build_edges(decisioned, PROPOSED_THRESHOLD))
    staging_len = memb.groupBy("component").agg(F.count("*").alias("staging_len"))

    prod = ident_flat.select(
        F.col(IDENT_COLS["party_id"]).alias("id"),
        F.col(IDENT_COLS["identity_id"]).alias("identity_id"))

    joined = memb.join(prod, "id", "inner")

    per_comp = (joined.groupBy("component", "identity_id").agg(F.count("*").alias("n"))
                      .withColumn("rk", F.row_number().over(
                          W.partitionBy("component").orderBy(F.desc("n"))))
                      .withColumn("n_prod_identities",
                                  F.count("*").over(W.partitionBy("component"))))
    dominant = per_comp.where(F.col("rk") == 1).select("component", "identity_id", "n_prod_identities")
    prod_len = prod.groupBy("identity_id").agg(F.count("*").alias("prod_len"))

    net = (staging_len.join(dominant, "component", "inner")
                      .join(prod_len, "identity_id", "left")
                      .withColumn("prod_len", F.coalesce(F.col("prod_len"), F.lit(0)))
                      .withColumn("net_gain", F.col("staging_len") - F.col("prod_len"))
                      .withColumn("category",
                          F.when(F.col("net_gain") < -10, "Identity Shrink")
                           .when(F.col("net_gain") < 0,   "Smaller Identity")
                           .when(F.col("net_gain") <= 5,  "Acceptable")
                           .when(F.col("net_gain") <= 15, "Slight Gain")
                           .when(F.col("net_gain") <= 30, "Bigger Gain")
                           .otherwise("Huge Gain")))

    ng = save_table(net.groupBy("category").agg(
            F.count("*").alias("clusters"),
            F.avg("net_gain").alias("avg_net_gain"),
            F.max("net_gain").alias("max_net_gain"),
            F.avg("n_prod_identities").alias("avg_prod_identities_spanned")
          ).orderBy(F.desc("clusters")), "08b_net_gain_by_category")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ng["category"], ng["clusters"], color="#2980b9")
    ax.set_ylabel("clusters")
    ax.set_title("Identity net gain at threshold " + str(PROPOSED_THRESHOLD))
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, "08b_net_gain")

    # clusters fusing 2+ distinct PROD identities = the real over-merge population
    save_table(net.where(F.col("n_prod_identities") > 1)
                  .orderBy(F.desc("n_prod_identities")).limit(200),
               "08b_multi_identity_merges", show=False)
else:
    print("identity_id / party_id not resolved on IDENTITIES - skipping net-gain")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 9. MANIFEST
# -----------------------------------------------------------------------------
manifest = pd.DataFrame({"artifact": sorted(set(SAVED))})
manifest.to_csv(DATA_DIR + "/_manifest.csv", index=False)
print(str(len(manifest)) + " artifacts written to " + DATA_DIR)
display(manifest)

# Talking points for the team / OIT / IIMD (AC3):
#  - Flat 0.90 recovers the FN population but is weakest exactly where the
#    evidence is thin (name+DOB only) -> sections 4 and 5.
#  - The tiered rule (7) keeps nearly all the recovery while holding
#    thin-evidence links and catching the A#-shared-by-relatives FP.
#  - The real exposure is cluster-level over-merge (8, 8b), not the pairwise FP
#    rate. Report giant clusters and multi-identity merges next to the 0.045%.
#  - Fix the 1969-07-20 sentinel DOB upstream (2) before any of this ships.
