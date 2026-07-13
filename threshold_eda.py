# Databricks notebook source
# =============================================================================
# CPMS Non-Identity - Threshold Evaluation EDA
# Companion to Jira IDEN-43344 / Confluence "2026 04 - CPMS Non-Identities".
#
# Goal: move the "lower the threshold 0.98 -> 0.90" decision from an eyeballed
# sample to a defensible, evidence-decomposed analysis, and quantify over-merge /
# identity-split at the CLUSTER level rather than only pairwise.
#
# PERSISTENCE POLICY:
#   * READS catalog tables only (spark.table). It NEVER writes to the catalog -
#     no saveAsTable, no CREATE TABLE.
#   * ALL outputs (summary tables as .csv, charts as .png) are written to the
#     workspace `data` folder set in DATA_DIR below.
#
# Cells marked ">>> TODO" are the only places you wire in your real schema.
# =============================================================================

# COMMAND ----------

# -----------------------------------------------------------------------------
# 0. CONFIG
# -----------------------------------------------------------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F, Window as W

# Real sources in Unity Catalog (READ ONLY). Three-level: catalog.schema.table
CATALOG = "eciscor_prod"
SCHEMA  = "pcis_metadata"

PARTIES_TBL = f"{CATALOG}.{SCHEMA}.elasticdump_parties"        # all parties (filter to non-identity below)
PAIRS_TBL   = f"{CATALOG}.{SCHEMA}.elasticdump_party_matches"  # scored candidate pairs (MLaaS/splink)
PROD_ID_TBL = f"{CATALOG}.{SCHEMA}.elasticdump_identities"     # PROD identities (for net-gain)

# IRQ-scoped variants seen in the catalog - likely the pre-filtered non-identity set.
# If elasticdump_parties_irq is already just the CPMS non-identity population you're
# re-triggering, set PARTIES_TBL = IRQ_PARTIES_TBL and skip the status filter.
IRQ_PARTIES_TBL = f"{CATALOG}.{SCHEMA}.elasticdump_parties_irq"
IRQ_INDEX_TBL   = f"{CATALOG}.{SCHEMA}.elasticdump_irq_index"

# >>> TODO: if PARTIES_TBL is the full parties table, set the non-identity filter.
# Set to None if you point PARTIES_TBL at elasticdump_parties_irq instead.
# >>> TODO: set once 1a/1b reveal the real status field inside _source,
# e.g. "partyStatus = 'non-identity'". Leave None until then - the raw ES dump
# has no top-level `status` column, which is what broke the first run.
NON_IDENTITY_FILTER = None

# Labels come from your manual review (~2,237 pairs). We do NOT read/write a catalog
# table for these - drop a CSV in the data folder (LABELS_CSV, defined after DATA_DIR)
# with columns: party_id_l, party_id_r, score, label (1 true match / 0 false positive)

# All artifacts land here. This is the empty `data` folder from your workspace.
# If workspace-file writes are blocked on your runtime, repoint to a Volume/DBFS
# path, e.g. "/Volumes/prod/irq/data" or "/dbfs/tmp/irq_data".
DATA_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/irq/data"
os.makedirs(DATA_DIR, exist_ok=True)
LABELS_CSV = f"{DATA_DIR}/threshold_review_labels.csv"

# Column mapping - rename to whatever your columns are actually called
COLS = {
    "party_id": "party_id", "score": "candidateScore",  # Confluence histogram axis; confirm in §1b
    "left_id": "party_id_l", "right_id": "party_id_r",
    "first": "firstName", "middle": "middleName", "last": "lastName",
    "dob": "dateOfBirth", "a_number": "aNumber", "ssn": "ssn", "fin": "fin",
    "eid": "eid", "i94": "i94", "coc": "countryOfCitizenship",
    "cob": "countryOfBirth", "receipt": "receipt", "form": "formType",
}
STRONG_IDS = ["a_number", "ssn", "fin"]
THRESHOLDS = [0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
SENTINEL_DOB   = ["1969-07-20"]
SENTINEL_NAMES = ["UNKNOWN", "NONE", "N/A", "TEST"]

def band_expr(score_col):
    s = F.col(score_col)
    return (F.when(s >= 0.98, "1_match").when(s >= 0.95, "2_goldilocks")
             .when(s >= 0.90, "3_near_match").when(s >= 0.50, "4_higher")
             .when(s >= 0.20, "5_lower").otherwise("6_non_match"))

# ---- persistence + display helpers (write to DATA_DIR, never to catalog) ----
SAVED = []   # manifest of everything written

def save_table(df, name, show=True):
    """Persist a (small, aggregated) result to DATA_DIR as CSV. Accepts Spark or pandas."""
    pdf = df if isinstance(df, pd.DataFrame) else df.toPandas()
    path = f"{DATA_DIR}/{name}.csv"
    pdf.to_csv(path, index=False)
    SAVED.append(path); print("saved:", path)
    if show:
        display(pdf)
    return pdf

def save_fig(fig, name):
    path = f"{DATA_DIR}/{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    SAVED.append(path); print("saved:", path)
    display(fig); plt.close(fig)

spark.conf.set("spark.sql.shuffle.partitions", "auto")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1a. RAW SCHEMA DISCOVERY  <-- RUN THIS CELL FIRST, ON ITS OWN
# These are raw Elasticsearch dumps: every table is (_id, _index, _score,
# _source, _type). All real fields are nested inside _source.
#
# NOTE: `_score` is the ELASTICSEARCH relevance score, NOT the MLaaS match
# score. The match score (candidateScore) lives INSIDE _source on
# party_matches. Do not band on `_score`.
# -----------------------------------------------------------------------------
for label, tbl in [("PARTIES", PARTIES_TBL), ("PAIRS", PAIRS_TBL), ("IDENTITIES", PROD_ID_TBL)]:
    df = spark.table(tbl)
    print("\n================ " + label + ": " + tbl + " ================")
    df.printSchema()                    # reveals whether _source is a struct or a JSON string
    print("distinct _index values:")
    df.select("_index").distinct().show(10, truncate=False)
    print("sample _source:")
    df.select("_source").show(1, truncate=False)

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1b. FLATTEN _source
# Handles both shapes: _source as a StructType (expand it) or as a JSON string
# (infer schema from a sample, then from_json). After this runs, read the
# printed column lists and fill in COLS in cell 1c.
# -----------------------------------------------------------------------------
from pyspark.sql.types import StructType, StringType

def flatten_source(df, sample_rows=200):
    """Expand _source to top-level columns, keeping _id and _index."""
    src_type = df.schema["_source"].dataType

    if isinstance(src_type, StructType):
        return df.select("_id", "_index", "_source.*")

    if isinstance(src_type, StringType):
        sample = (df.select("_source").where(F.col("_source").isNotNull())
                    .limit(sample_rows).toPandas()["_source"].tolist())
        if not sample:
            raise ValueError("no non-null _source rows to infer schema from")
        inferred = spark.read.json(spark.sparkContext.parallelize(sample)).schema
        return (df.withColumn("src", F.from_json(F.col("_source"), inferred))
                  .select("_id", "_index", "src.*"))

    raise TypeError("unexpected _source type: " + str(src_type))

parties_flat = flatten_source(spark.table(PARTIES_TBL))
pairs_flat   = flatten_source(spark.table(PAIRS_TBL))
ident_flat   = flatten_source(spark.table(PROD_ID_TBL))

for label, df in [("PARTIES", parties_flat), ("PAIRS", pairs_flat), ("IDENTITIES", ident_flat)]:
    print("\n=== " + label + " flattened (" + str(len(df.columns)) + " cols) ===")
    print(df.columns)

# If a field you need is still nested (e.g. candidateScore sits inside an array
# of matches on party_matches), explode it before section 3:
#     pairs_flat = pairs_flat.withColumn("m", F.explode("matches")).select("*", "m.*")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1c. COLUMN MAP  >>> TODO: fill in from the 1b printout, then run onward.
# Set any field you genuinely don't have to None; the agreement logic will then
# treat it as always-missing rather than erroring.
# -----------------------------------------------------------------------------
COLS.update({
    # "score":    "candidateScore",
    # "left_id":  "...",     # declared / instigating party id on party_matches
    # "right_id": "...",     # candidate party id on party_matches
    # "party_id": "...",     # id on parties (may simply be _id)
    # "first": "...", "middle": "...", "last": "...", "dob": "...",
    # "a_number": "...", "ssn": "...", "fin": "...", "form": "...",
})

# Verify the mapping against the flattened schemas before anything expensive runs.
missing_pairs   = [k for k in ["score", "left_id", "right_id"]
                   if COLS.get(k) and COLS[k] not in pairs_flat.columns]
missing_parties = [k for k in ["first", "last", "dob", "a_number", "ssn", "fin"]
                   if COLS.get(k) and COLS[k] not in parties_flat.columns]
print("PAIRS   cols not found:", missing_pairs or "OK")
print("PARTIES cols not found:", missing_parties or "OK")
assert not missing_pairs and not missing_parties, "fix COLS in cell 1c before continuing"

# COMMAND ----------

# -----------------------------------------------------------------------------
# 1. LOAD (read only) - operating on the FLATTENED frames
# -----------------------------------------------------------------------------
parties = parties_flat
if NON_IDENTITY_FILTER:                  # keep only the CPMS non-identity population
    parties = parties.where(NON_IDENTITY_FILTER)

pairs = pairs_flat.withColumn("score", F.col(COLS["score"]).cast("double"))
pairs = pairs.withColumn("band", band_expr("score"))

meta = pd.DataFrame([
    {"metric": "parties", "value": parties.count()},
    {"metric": "scored_pairs", "value": pairs.count()},
])
save_table(meta, "00_row_counts")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 2. DATA QUALITY - fill rates + sentinel detection  (table + chart)
# -----------------------------------------------------------------------------
attr_cols = ["first","middle","last","dob","a_number","ssn","fin","eid","i94","coc","cob","receipt"]
n = parties.count()

fill = parties.select([
    (F.count(F.when(F.col(COLS[c]).isNotNull() & (F.trim(F.col(COLS[c])) != ""), 1)) / F.lit(n)).alias(c)
    for c in attr_cols
]).toPandas().T.reset_index()
fill.columns = ["attribute", "fill_rate"]
fill_pdf = save_table(fill, "02_fill_rates")

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#c0392b" if v < 0.5 else "#2e86c1" for v in fill_pdf["fill_rate"]]
ax.bar(fill_pdf["attribute"], fill_pdf["fill_rate"], color=colors)
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_ylabel("fill rate"); ax.set_title("CPMS Non-Identity attribute completeness")
plt.xticks(rotation=45, ha="right")
save_fig(fig, "02_fill_rates")

# COMMAND ----------

# Fill-rate BY FORM (table) - shows identifier density varies by form type,
# which is why one global threshold penalizes some forms far more than others.
fill_by_form = (parties.groupBy(COLS["form"])
    .agg(F.count("*").alias("records"),
         *[ (F.count(F.when(F.col(COLS[c]).isNotNull(), 1)) / F.count("*")).alias(f"{c}_fill")
            for c in ["a_number","ssn","fin","dob"] ])
    .orderBy(F.desc("records")))
save_table(fill_by_form, "02_fill_by_form")

# COMMAND ----------

# Sentinel contamination (table) + cleaned view that nulls sentinels pre-compare.
sentinel = parties.select(
    F.count(F.when(F.col(COLS["dob"]).isin(SENTINEL_DOB), 1)).alias("sentinel_dob"),
    F.count(F.when(F.upper(F.col(COLS["first"])).isin(SENTINEL_NAMES), 1)).alias("sentinel_first"),
)
save_table(sentinel, "02_sentinel_counts")

parties_clean = (parties
    .withColumn(COLS["dob"],   F.when(F.col(COLS["dob"]).isin(SENTINEL_DOB), None).otherwise(F.col(COLS["dob"])))
    .withColumn(COLS["first"], F.when(F.upper(F.col(COLS["first"])).isin(SENTINEL_NAMES), None).otherwise(F.col(COLS["first"]))))

# COMMAND ----------

# -----------------------------------------------------------------------------
# 3. SCORE DISTRIBUTION  (table + chart, reconciles with Confluence bins)
# -----------------------------------------------------------------------------
band_counts = pairs.groupBy("band").count().orderBy("band")
bc = save_table(band_counts, "03_score_bands")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(bc["band"], bc["count"], color="#2e86c1")
ax.set_ylabel("pairs"); ax.set_title("Score-band distribution")
plt.xticks(rotation=30, ha="right")
save_fig(fig, "03_score_bands")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 4. MATCH-WEIGHT DECOMPOSITION  (the key value-add: what drives near-matches)
# -----------------------------------------------------------------------------
bf_cols = [c for c in pairs.columns if c.startswith("bf_") or c.startswith("gamma_")]
print("splink term-level columns found:", bf_cols)

if not bf_cols:  # fallback: rebuild agreement flags by joining attributes
    L = parties_clean.select([F.col(COLS[k]).alias(f"{k}_l") for k in COLS])
    R = parties_clean.select([F.col(COLS[k]).alias(f"{k}_r") for k in COLS])
    px = (pairs.join(L, pairs[COLS["left_id"]]  == F.col("party_id_l"))
               .join(R, pairs[COLS["right_id"]] == F.col("party_id_r")))
    def agree(field):
        l, r = F.col(f"{field}_l"), F.col(f"{field}_r")
        return F.when(l.isNull() | r.isNull(), None).otherwise((l == r).cast("int"))
    for f in ["dob","a_number","ssn","fin","last","first"]:
        px = px.withColumn(f"agree_{f}", agree(f))
    pairs_dx = px
else:
    pairs_dx = pairs

near = pairs_dx.filter(F.col("band").isin("2_goldilocks","3_near_match"))
agree_profile = near.agg(
    F.avg("agree_dob").alias("dob"), F.avg("agree_last").alias("last"),
    F.avg("agree_a_number").alias("a_number"), F.avg("agree_ssn").alias("ssn"),
    F.avg("agree_fin").alias("fin"),
    (F.count(F.when(F.coalesce(F.col("agree_a_number"),F.col("agree_ssn"),F.col("agree_fin"))==1,1))
       / F.count("*")).alias("any_strong_id"),
).toPandas().T.reset_index()
agree_profile.columns = ["field", "agree_rate_in_near_band"]
ap = save_table(agree_profile, "04_near_band_agreement")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(ap["field"], ap["agree_rate_in_near_band"], color="#8e44ad")
ax.set_ylabel("agreement rate"); ax.set_title("What agrees in the 0.90-0.98 near-match band")
save_fig(fig, "04_near_band_agreement")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 5. AGREEMENT-PATTERN CROSSTAB across bands  (table)
# -----------------------------------------------------------------------------
pat = (pairs_dx.withColumn("pattern", F.concat_ws("+",
        F.when(F.col("agree_dob")==1,"DOB"), F.when(F.col("agree_last")==1,"LAST"),
        F.when(F.col("agree_a_number")==1,"A#"), F.when(F.col("agree_ssn")==1,"SSN"),
        F.when(F.col("agree_fin")==1,"FIN")))
    .groupBy("band","pattern").count().orderBy("band", F.desc("count")))
save_table(pat, "05_agreement_patterns")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 6. PRECISION / RECALL / F1 vs THRESHOLD  (table + chart, from reviewed labels)
# -----------------------------------------------------------------------------
# Labels come from the review CSV in the data folder (no catalog table involved).
# Expected columns: party_id_l, party_id_r, score, label (1 true match / 0 FP).
if not os.path.exists(LABELS_CSV):
    print(f"SKIP §6: no labels file at {LABELS_CSV}. "
          f"Export your ~2,237 reviewed pairs there to enable the PR curve.")
    lp = pd.DataFrame(columns=["l", "r", "score", "label"])
else:
    _raw = pd.read_csv(LABELS_CSV)
    lp = _raw.rename(columns={"party_id_l": "l", "party_id_r": "r"})[["l", "r", "score", "label"]]
    lp["score"] = lp["score"].astype(float); lp["label"] = lp["label"].astype(int)

if len(lp) > 0:
    rows = []
    for t in THRESHOLDS:
        pred = (lp["score"] >= t).astype(int)
        tp = int(((pred==1) & (lp["label"]==1)).sum()); fp = int(((pred==1) & (lp["label"]==0)).sum())
        fn = int(((pred==0) & (lp["label"]==1)).sum())
        prec = tp/(tp+fp) if tp+fp else float("nan")
        rec  = tp/(tp+fn) if tp+fn else float("nan")
        f1   = 2*prec*rec/(prec+rec) if prec and rec else float("nan")
        rows.append((t, tp, fp, fn, prec, rec, f1))
    pr = pd.DataFrame(rows, columns=["threshold","tp","fp","fn","precision","recall","f1"])
    save_table(pr, "06_pr_by_threshold")

    fig, ax = plt.subplots(figsize=(8, 4))
    for m, c in [("precision","#c0392b"), ("recall","#2e86c1"), ("f1","#27ae60")]:
        ax.plot(pr["threshold"], pr[m], marker="o", label=m, color=c)
    ax.invert_xaxis(); ax.set_xlabel("threshold"); ax.set_ylabel("score")
    ax.set_title("Precision / recall / F1 vs threshold"); ax.legend()
    save_fig(fig, "06_pr_by_threshold")

from math import sqrt
def wilson(k, nn, z=1.96):
    if nn == 0: return (float("nan"), float("nan"))
    p = k/nn; d = 1+z*z/nn; c = p + z*z/(2*nn); m = z*sqrt(p*(1-p)/nn + z*z/(4*nn*nn))
    return (c-m)/d, (c+m)/d
# Observed FPs among accepted-at-0.90 in the reviewed set (falls back to the
# Confluence figure of 1 FP in 2,237 if no labels file is present yet).
if len(lp) > 0:
    acc = lp[lp["score"] >= 0.90]
    k_fp, n_rev = int((acc["label"] == 0).sum()), len(acc)
else:
    k_fp, n_rev = 1, 2237
lo, hi = wilson(k_fp, n_rev)
save_table(pd.DataFrame([{"observed_fp": k_fp, "n_reviewed": n_rev,
                          "fp_rate": (k_fp/n_rev if n_rev else float('nan')),
                          "fp_rate_ci_low": lo, "fp_rate_ci_high": hi}]),
           "06_fp_rate_ci")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 7. TIERED / RULE-AUGMENTED THRESHOLD prototype  (table + chart)
# -----------------------------------------------------------------------------
decisioned = (pairs_dx
    .withColumn("strong_agree",
        (F.coalesce(F.col("agree_a_number"),F.lit(0)) + F.coalesce(F.col("agree_ssn"),F.lit(0))
         + F.coalesce(F.col("agree_fin"),F.lit(0))) > 0)
    .withColumn("relative_risk",
        (F.col("agree_a_number")==1) & (F.col("agree_dob")==0)
        & F.col("dob_l").isNotNull() & F.col("dob_r").isNotNull())
    .withColumn("decision",
        F.when(F.col("relative_risk"), "review_relative")
         .when((F.col("score") >= 0.90) & F.col("strong_agree"), "accept")
         .when(F.col("score") >= 0.98, "accept")
         .when(F.col("score") >= 0.90, "review_thin_evidence")
         .otherwise("reject")))

compare = (decisioned.withColumn("flat_090", F.when(F.col("score")>=0.90,"accept").otherwise("reject"))
    .groupBy("flat_090","decision").count().orderBy("flat_090","decision"))
cmp_pdf = save_table(compare, "07_tiered_vs_flat")

dec_counts = decisioned.groupBy("decision").count().orderBy(F.desc("count")).toPandas()
save_table(dec_counts, "07_decision_counts", show=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(dec_counts["decision"], dec_counts["count"], color="#e67e22")
ax.set_ylabel("pairs"); ax.set_title("Tiered-rule decision distribution")
plt.xticks(rotation=20, ha="right")
save_fig(fig, "07_decision_counts")

# COMMAND ----------

# -----------------------------------------------------------------------------
# 8. CLUSTER IMPACT - identity_net_gain as a graph problem  (table + charts)
# -----------------------------------------------------------------------------
spark.sparkContext.setCheckpointDir("/tmp/cpms_cc_checkpoint")
from graphframes import GraphFrame

def cluster_at(threshold):
    edges = (decisioned.filter(F.col("score") >= threshold)
             .selectExpr(f"{COLS['left_id']} as src", f"{COLS['right_id']} as dst"))
    verts = (edges.select(F.col("src").alias("id"))
             .union(edges.select(F.col("dst").alias("id"))).distinct())
    return GraphFrame(verts, edges).connectedComponents().groupBy("component").count()

stat_rows = []
for t in [0.98, 0.95, 0.92, 0.90]:
    sizes = cluster_at(t).cache()
    s = sizes.agg(F.count("*").alias("n_clusters"), F.max("count").alias("max_size"),
                  F.expr("percentile_approx(count,0.99)").alias("p99_size"),
                  F.avg("count").alias("mean_size")).toPandas().iloc[0]
    stat_rows.append({"threshold": t, **s.to_dict()})
    if t == 0.90:  # persist the giant-cluster (over-merge) candidates at the proposed threshold
        save_table(sizes.filter(F.col("count") >= 30).orderBy(F.desc("count")),
                   "08_giant_clusters_at_090", show=False)
        sizes_090 = sizes.toPandas()
cluster_stats = save_table(pd.DataFrame(stat_rows), "08_cluster_stats")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(cluster_stats["threshold"], cluster_stats["max_size"], marker="o", color="#c0392b")
ax.invert_xaxis(); ax.set_xlabel("threshold"); ax.set_ylabel("max cluster size")
ax.set_title("Over-merge risk: largest identity vs threshold")
save_fig(fig, "08_max_cluster_size")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(sizes_090["count"], bins=range(1, 40), color="#16a085")
ax.set_xlabel("parties per cluster"); ax.set_ylabel("clusters")
ax.set_title("Cluster-size distribution at threshold 0.90")
save_fig(fig, "08_cluster_size_hist_090")

# COMMAND ----------

# net-gain recompute skeleton (staging cluster size minus PROD identity length)
staging_sizes = cluster_at(0.90).withColumnRenamed("count","staging_len")
# >>> TODO: PROD_ID_TBL needs (party_id, identity_id, identity_length); map each
# staging component to the dominant PROD identity, diff, band into
# Acceptable/Slight/Bigger/Huge/Smaller/Shrink, then save_table(..., "08_net_gain").

# COMMAND ----------

# -----------------------------------------------------------------------------
# 9. SUMMARY - manifest of everything written to the data folder
# -----------------------------------------------------------------------------
manifest = pd.DataFrame({"artifact": sorted(set(SAVED))})
manifest.to_csv(f"{DATA_DIR}/_manifest.csv", index=False)
print(f"{len(manifest)} artifacts written to {DATA_DIR}")
display(manifest)
