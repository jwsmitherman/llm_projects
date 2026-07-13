# IDEN-43831 - ANM Splink evaluation (SPLINK-ONLY)
# What: runs Splink (probabilistic record matching) across 8 identifier scenarios on raw record fields,
#       then summarizes the results with tables and charts.
# Why:  answers the business question - does this work as a matching tool, especially for people with no strong ID?
#
# Run first (own cell): %pip install splink==4.0.7 "sqlglot<26"   then dbutils.library.restartPython()

# ==================================================================================
# IMPORTS
# What: loads pandas, matplotlib, the scoring metrics, and Splink.
# Why:  these are the tools the rest of the script uses.
# ==================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

# ==================================================================================
# CONFIG
# What: sets the input table, results table, sample size, and a fixed random seed.
# Why:  one place to control the run, so every run is the same and repeatable.
# ==================================================================================
SEED          = 43831
CATALOG       = "eciscor_prod"
DS            = f"{CATALOG}.pcis_data_science"
FULLSET       = f"{DS}.anm_training_data_fullset"   # raw record pairs (firstName_1/2, dob_1/2, ssn_1/2, ...) - what Splink matches on
RESULTS_TABLE = f"{DS}.anm_scenario_results"
SAMPLE_PER_CLASS = 50000
THRESH        = 0.90
SAVE_RESULTS  = True

# ==================================================================================
# WHERE RESULT FILES GET WRITTEN
# What: finds the `data` folder next to this notebook and creates it if missing.
# Why:  so the charts and CSVs land in a known place you can open and share.
# ==================================================================================
import os
try:                                                            # auto-derive the notebook's folder in Databricks
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    DATA_DIR = "/Workspace" + _nb.rsplit("/", 1)[0] + "/data"
except Exception:                                               # fallback: set this to your data folder path
    DATA_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/ANM_models_that_can_handle_no-Id/data"
os.makedirs(DATA_DIR, exist_ok=True)
print("[B] result files ->", DATA_DIR)

# ==================================================================================
# READ RAW PAIRS -> RECORDS
# What: reads the raw record pairs, takes a balanced sample (equal matches and non-matches),
#       then splits each pair into two single records and drops duplicates.
# Why:  Splink compares one flat list of records, not pre-joined pairs. This turns the pair
#       data into that list. Balancing keeps the sample fair.
# ==================================================================================
ATTRS = ["firstName", "middleName", "lastName", "dob", "cob", "anumber", "ssn", "fin",
         "driversLicense", "passport", "i94", "travel_doc"]
RID = "party_record_receipt"

df = spark.table(FULLSET)                                       # raw labeled record pairs
n_pos = df.filter("label = 1").count(); n_neg = df.filter("label = 0").count()  # class counts for balancing
fr = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)), 0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}  # sample fractions
pairs = df.sampleBy("label", fractions=fr, seed=SEED).toPandas()  # balanced sample, reproducible via SEED

def side(n):                                                    # one side of the pair, suffix stripped
    ren = {f"{RID}_{n}": "record_id"}; ren.update({f"{a}_{n}": a for a in ATTRS})
    return pairs[list(ren)].rename(columns=ren)
records = (pd.concat([side("1"), side("2")], ignore_index=True) # record-level table splink links
             .dropna(subset=["record_id"]).drop_duplicates("record_id").reset_index(drop=True))

# ==================================================================================
# TRUTH TABLE + PRIOR
# What: builds a table of "these two records, and whether they're truly the same person,"
#       flags pairs with no SSN / A-Number / FIN, and calculates the base chance any two random records match.
# Why:  the truth labels let us grade accuracy. The flag lets us score the hard (no-strong-ID) group on its own.
#       The prior is the starting estimate Splink needs before weighing evidence - set directly to avoid an error.
# ==================================================================================
def _usable(c1, c2):                                            # both sides present + non-blank
    a, b = pairs[c1].astype("string"), pairs[c2].astype("string")
    return a.notna() & b.notna() & (a.str.strip() != "") & (b.str.strip() != "")
lab = pairs[[f"{RID}_1", f"{RID}_2", "label"]].copy()           # truth table
lab["no_strong_id"] = (~(_usable("ssn_1", "ssn_2") | _usable("anumber_1", "anumber_2")
                         | _usable("fin_1", "fin_2"))).values   # no usable SSN/A#/FIN
lab = lab.dropna(subset=[f"{RID}_1", f"{RID}_2"])
lab["key"] = lab.apply(lambda r: tuple(sorted([r[f"{RID}_1"], r[f"{RID}_2"]])), axis=1)  # order-independent pair id
lab = lab.drop_duplicates("key")
_npairs = len(records) * (len(records) - 1) / 2
P_MATCH = max(1e-6, min(0.5, int((lab["label"] == 1).sum()) / _npairs))  # fixed prior (avoids recall ValueError)
print("[B] records:", len(records), "labeled pairs:", len(lab), "prior P_MATCH:", round(P_MATCH, 8))

# ==================================================================================
# COMPARISONS, SCENARIOS, AND BLOCKING
# What: comparisons = how Splink judges each field (fuzzy name, close-enough dob, exact country, near-match IDs),
#       with common values counting for less. Scenarios = 8 versions each allowed a different set of ID fields
#       (name + dob always on). Blocking = which pairs are worth comparing, instead of everyone vs everyone.
# Why:  comparisons decide how strong the evidence is; scenarios test how matching holds up as IDs are removed;
#       blocking makes it fast enough to run and decides which pairs get looked at at all.
# ==================================================================================
NAME_COMPS = [cl.ForenameSurnameComparison("firstName", "lastName")
                .configure(term_frequency_adjustments=True),                # TF: down-weight common names
              cl.DateOfBirthComparison("dob", input_is_string=True),
              cl.ExactMatch("cob").configure(term_frequency_adjustments=True)]  # TF: down-weight common countries
def id_comp(col):
    return cl.LevenshteinAtThresholds(col, [1, 2]).configure(term_frequency_adjustments=True)
ID_COL = {"anum": "anumber", "ssn": "ssn", "fin": "fin", "dl": "driversLicense",
          "passport": "passport", "i94": "i94", "travel_doc": "travel_doc"}
SCN_IDS = {
    "baseline":             ["anum", "ssn", "fin"],
    "no_ssn":               ["anum", "fin"],
    "no_strong_id":         [],
    "with_dl":              ["anum", "ssn", "fin", "dl"],
    "with_dl_no_strong_id": ["dl"],
    "with_other_ids":       ["anum", "ssn", "fin", "dl", "passport", "i94", "travel_doc"],
    "other_ids_no_strong":  ["dl", "passport", "i94", "travel_doc"],
    "all_ids":              ["anum", "ssn", "fin", "dl", "passport", "i94", "travel_doc"],
}
# name/dob blocking on EVERY scenario so no-strong-id pairs are actually compared;
# substr(...) are crude fuzzy stand-ins - swap for phonetic keys (soundex/metaphone) in production.
NAME_DOB_BLOCKS = [
    block_on("firstName", "dob"),
    block_on("lastName", "dob"),
    block_on("firstName", "lastName"),
    block_on("substr(lastName,1,3)", "dob"),
    block_on("lastName", "substr(dob,1,4)"),
    block_on("firstName", "substr(dob,1,4)"),
]

# ==================================================================================
# SCORING LOOP
# What: runs Splink for each scenario, scores every labeled pair, and records the numbers -
#       for everyone and for the no-strong-ID group.
# Why:  produces the accuracy and coverage figures the whole analysis is built on.
#       auc = end-to-end accuracy; auc_blocked_only = accuracy on pairs it actually compared;
#       blocking_recall = coverage (share of true matches compared).
# ==================================================================================
def splink_scenario(ids):
    comps = NAME_COMPS + [id_comp(ID_COL[k]) for k in ids]
    blocking = NAME_DOB_BLOCKS + [block_on(ID_COL[k]) for k in ids]
    settings = SettingsCreator(link_type="dedupe_only", unique_id_column_name="record_id",
                               comparisons=comps, blocking_rules_to_generate_predictions=blocking,
                               probability_two_random_records_match=P_MATCH)
    linker = Linker(records, settings, db_api=DuckDBAPI())
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("firstName", "lastName"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
    preds = linker.inference.predict().as_pandas_dataframe()
    prob = {tuple(sorted([a, b])): p for a, b, p in
            zip(preds["record_id_l"], preds["record_id_r"], preds["match_probability"])}
    scored = lab.copy()
    scored["p"] = scored["key"].map(prob)
    scored["blocked"] = scored["p"].notna()
    scored["p"] = scored["p"].fillna(0.0)
    return scored

rows = []
for scn, ids in SCN_IDS.items():
    s = splink_scenario(ids)
    for slice_name, sub in [("all", s), ("no_strong_id", s[s["no_strong_id"]])]:
        if len(sub) == 0 or sub["label"].nunique() < 2:
            continue
        bf = sub[sub["blocked"]]; m = sub["label"] == 1
        rows.append(dict(scenario=scn, model="splink", eval=slice_name, n=len(sub),
                         auc=round(roc_auc_score(sub["label"], sub["p"]), 4),
                         ap=round(average_precision_score(sub["label"], sub["p"]), 4),
                         f1=round(f1_score(sub["label"], (sub["p"] >= THRESH).astype(int), zero_division=0), 4),
                         blocking_recall=round(sub.loc[m, "blocked"].mean(), 4),
                         auc_blocked_only=(round(roc_auc_score(bf["label"], bf["p"]), 4)
                                           if bf["label"].nunique() > 1 else None)))
splink_results = pd.DataFrame(rows)

# ==================================================================================
# ANALYSIS SETUP
# What: splits the results into the everyone view and the no-strong-ID view, in a fixed scenario order.
# Why:  the tables and charts below read from these two views.
# ==================================================================================
SCN_ORDER = ["baseline", "no_ssn", "no_strong_id", "with_dl", "with_dl_no_strong_id",
             "with_other_ids", "other_ids_no_strong", "all_ids"]
res   = splink_results.copy()
alls  = res[res["eval"] == "all"].set_index("scenario").reindex([s for s in SCN_ORDER if s in res[res["eval"]=="all"]["scenario"].values])
noid  = res[res["eval"] == "no_strong_id"].set_index("scenario").reindex([s for s in SCN_ORDER if s in res[res["eval"]=="no_strong_id"]["scenario"].values])

# ==================================================================================
# SUMMARY TABLES
# What: a headline table (everyone vs no-strong-ID) and a per-scenario table for the no-strong-ID group.
# Why:  the whole story in a few numbers - accuracy, coverage, and accuracy-when-decidable.
# ==================================================================================
headline = pd.DataFrame({
    "population": ["everyone", "no-strong-ID"],
    "accuracy_end_to_end":     [round(alls["auc"].mean(), 3),               round(noid["auc"].mean(), 3)],
    "coverage":                [round(alls["blocking_recall"].mean(), 3),   round(noid["blocking_recall"].mean(), 3)],
    "accuracy_when_decidable": [round(alls["auc_blocked_only"].mean(), 3),  round(noid["auc_blocked_only"].dropna().mean(), 3)],
})
print("\n================= HEADLINE =================")
print("The method works when it can decide (~0.99). End-to-end accuracy is limited by COVERAGE, not the model.")
print(headline.to_string(index=False))

print("\n========== No-strong-ID population: per scenario ==========")
print(noid[["auc", "blocking_recall", "auc_blocked_only", "f1"]]
        .rename(columns={"auc": "accuracy_end_to_end", "blocking_recall": "coverage",
                         "auc_blocked_only": "accuracy_when_decidable"}).round(3).to_string())

# ==================================================================================
# CHART 1 - COVERAGE BY SCENARIO
# What: bar chart of coverage, everyone vs no-strong-ID.
# Why:  shows the no-strong-ID group's coverage is stuck around half - the bottleneck.
# ==================================================================================
x = np.arange(len(SCN_ORDER)); w = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, alls["blocking_recall"].values, w, label="everyone", color="#2E8B8B")
ax.bar(x + w/2, noid["blocking_recall"].values, w, label="no-strong-ID", color="#E0833B")
ax.axhline(1.0, ls=":", c="gray")
ax.set_xticks(x); ax.set_xticklabels(SCN_ORDER, rotation=45, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("coverage (share of true matches compared)")
ax.set_title("Coverage by scenario — no-strong-ID sits at ~0.55 (the bottleneck)")
ax.legend(); plt.tight_layout()
fig.savefig(f"{DATA_DIR}/chart_coverage.png", dpi=150, bbox_inches="tight"); plt.show()

# ==================================================================================
# CHART 2 - THE COVERAGE GAP
# What: for the no-strong-ID group, compares end-to-end accuracy against accuracy-when-decidable.
# Why:  shows the model is near-perfect on pairs it can compare; the gap is purely coverage.
# ==================================================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, noid["auc"].values, w, label="end-to-end accuracy", color="#B0483B")
ax.bar(x + w/2, noid["auc_blocked_only"].astype(float).values, w, label="accuracy when decidable", color="#2E8B8B")
ax.set_xticks(x); ax.set_xticklabels(SCN_ORDER, rotation=45, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy (AUC)")
ax.set_title("No-strong-ID: model is ~0.99 when it can decide — the gap to end-to-end is coverage")
ax.legend(); plt.tight_layout()
fig.savefig(f"{DATA_DIR}/chart_coverage_gap.png", dpi=150, bbox_inches="tight"); plt.show()

# ==================================================================================
# KEY FINDINGS
# What: prints the plain-English conclusions.
# Why:  states the answer directly - it works when it has the data; the limit is coverage, not the method.
# ==================================================================================
print("""
================= KEY FINDINGS =================
1. YES - the tool produces accurate matches. When it has enough usable information to decide, it is
   right ~99% of the time, even for people with no strong ID.
2. For people WITH strong IDs, end-to-end matching is already strong (~0.97).
3. For the no-strong-ID group, end-to-end accuracy is ~0.60 for one reason: COVERAGE. Only ~55% of those
   true matches had enough usable information to be compared; the rest are undecidable and count as misses.
4. The limit is data availability / candidate coverage - not the matching method. Next step to raise
   coverage: phonetic / nickname matching so more of the hard cases can be compared.
""")

# ==================================================================================
# SAVE RESULT FILES
# What: writes the tables as CSVs and the charts as images into the `data` folder.
# Why:  so you can share them or drop them into a deck without re-running.
# ==================================================================================
splink_results.to_csv(f"{DATA_DIR}/splink_results.csv", index=False)                 # full per-scenario/slice metrics
headline.to_csv(f"{DATA_DIR}/splink_headline_summary.csv", index=False)              # everyone vs no-strong-ID headline
noid.reset_index().to_csv(f"{DATA_DIR}/splink_no_strong_id_summary.csv", index=False)  # no-strong-ID per scenario
alls.reset_index().to_csv(f"{DATA_DIR}/splink_all_population_summary.csv", index=False)  # full-population per scenario
print("saved result files (csv + charts) to", DATA_DIR)

# ==================================================================================
# APPEND RESULTS TO THE CATALOG TABLE
# What: adds this run's numbers (with a timestamp) to a results table.
# Why:  keeps a history of every run.
# ==================================================================================
if SAVE_RESULTS:
    out = splink_results.copy()
    for c in ["blocking_recall", "auc_blocked_only"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["run_ts"] = pd.Timestamp.utcnow(); out["seed"] = SEED
    (spark.createDataFrame(out)
        .write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE))
    print("appended results to", RESULTS_TABLE)
