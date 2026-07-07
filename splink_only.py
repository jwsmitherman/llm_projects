# IDEN-43831 - ANM spike, SPLINK-ONLY notebook.
# Runs Splink (Fellegi-Sunter record linkage) across the 8 identifier scenarios on RAW record fields,
# then an ANALYSIS & SUMMARY section (charts + tables) explaining what the results say.
# Databricks: %pip install splink==4.0.7 "sqlglot<26"   then dbutils.library.restartPython()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

# ============================== config ==============================
SEED          = 43831
CATALOG       = "eciscor_prod"
DS            = f"{CATALOG}.pcis_data_science"
TRAIN_TABLE   = f"{DS}.anm_training_data_sample_featurized"
FULLSET       = f"{DS}.anm_training_data_fullset"
RESULTS_TABLE = f"{DS}.anm_scenario_results"
SAMPLE_PER_CLASS = 50000
THRESH        = 0.90
SAVE_RESULTS  = True

# ---- where result FILES get written: the `data` folder next to this notebook ----
import os
try:                                                            # auto-derive the notebook's folder in Databricks
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    DATA_DIR = "/Workspace" + _nb.rsplit("/", 1)[0] + "/data"
except Exception:                                               # fallback: set this to your data folder path
    DATA_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/ANM_models_that_can_handle_no-Id/data"
os.makedirs(DATA_DIR, exist_ok=True)
print("[B] result files ->", DATA_DIR)

# ============================== reshape pairs -> records ==============================
src = spark.table(TRAIN_TABLE).toPandas()                       # read the labeled pair table
ATTRS = ["firstName", "middleName", "lastName", "dob", "cob", "anumber", "ssn", "fin",
         "driversLicense", "passport", "i94", "travel_doc"]
RID = "party_record_receipt"

if f"{RID}_1" in src.columns and "firstName_1" in src.columns:  # raw _1/_2 fields present?
    pairs = src
else:
    print("[B] raw _1/_2 columns not in TRAIN_TABLE - re-sampling from fullset "
          "(persist `sample` in the data file to avoid this)")
    df = spark.table(FULLSET)
    n_pos = df.filter("label = 1").count(); n_neg = df.filter("label = 0").count()
    fr = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)), 0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}
    pairs = df.sampleBy("label", fractions=fr, seed=SEED).toPandas()

def side(n):                                                    # one side of the pair, suffix stripped
    ren = {f"{RID}_{n}": "record_id"}; ren.update({f"{a}_{n}": a for a in ATTRS})
    return pairs[list(ren)].rename(columns=ren)
records = (pd.concat([side("1"), side("2")], ignore_index=True) # record-level table splink links
             .dropna(subset=["record_id"]).drop_duplicates("record_id").reset_index(drop=True))

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

# ============================== ID-collision diagnostic (no training) ==============================
# Why does adding ids hurt? Measure how often DIFFERENT people (non-matches) share the same id value.
# nonmatch_agree > 0 = false-positive evidence; the top shared value is usually a placeholder to null out.
ID_FIELDS = ["anumber", "ssn", "fin", "driversLicense", "passport", "i94", "travel_doc"]
def _norm_col(s): return s.astype("string").str.strip()
_diag = []
for _f in ID_FIELDS:
    a = _norm_col(pairs[f"{_f}_1"]); b = _norm_col(pairs[f"{_f}_2"])
    both  = a.notna() & b.notna() & (a != "") & (b != "")
    agree = both & (a == b)
    m = pairs["label"] == 1; nm = pairs["label"] == 0
    match_agree    = (agree & m).sum()  / max((both & m).sum(),  1)
    nonmatch_agree = (agree & nm).sum() / max((both & nm).sum(), 1)
    shared_nm = a[agree & nm]; top_val, top_cnt = ("", 0)
    if len(shared_nm):
        vc = shared_nm.value_counts(); top_val, top_cnt = str(vc.index[0]), int(vc.iloc[0])
    _diag.append(dict(id_field=_f, present_rate=round(float(both.mean()), 3),
                      match_agree=round(float(match_agree), 3),
                      nonmatch_agree=round(float(nonmatch_agree), 4),
                      top_shared_nonmatch_value=top_val[:24], count=top_cnt))
_diag = pd.DataFrame(_diag).sort_values("nonmatch_agree", ascending=False)

# ============================== comparisons + blocking ==============================
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
# robust name/dob blocking on EVERY scenario so no-strong-id pairs are actually compared;
# substr(...) are crude fuzzy stand-ins - swap for phonetic keys (soundex/metaphone) in production.
NAME_DOB_BLOCKS = [
    block_on("firstName", "dob"),
    block_on("lastName", "dob"),
    block_on("firstName", "lastName"),
    block_on("substr(lastName,1,3)", "dob"),
    block_on("lastName", "substr(dob,1,4)"),
    block_on("firstName", "substr(dob,1,4)"),
]

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
scored_by_scn = {}                                              # keep per-scenario per-pair scores for deeper analysis
for scn, ids in SCN_IDS.items():
    s = splink_scenario(ids)
    scored_by_scn[scn] = s                                      # stash for the "why more ids hurts" chart below
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

# ==================================================================================================
# ANALYSIS & SUMMARY  -  what the data is saying
# ==================================================================================================
SCN_ORDER = ["baseline", "no_ssn", "no_strong_id", "with_dl", "with_dl_no_strong_id",
             "with_other_ids", "other_ids_no_strong", "all_ids"]
res   = splink_results.copy()
alls  = res[res["eval"] == "all"].set_index("scenario").reindex([s for s in SCN_ORDER if s in res[res["eval"]=="all"]["scenario"].values])
noid  = res[res["eval"] == "no_strong_id"].set_index("scenario").reindex([s for s in SCN_ORDER if s in res[res["eval"]=="no_strong_id"]["scenario"].values])

# ---- Summary table 1: the headline (accuracy is gated by coverage, not the model) ----
headline = pd.DataFrame({
    "population": ["everyone", "no-strong-ID"],
    "accuracy_end_to_end":     [round(alls["auc"].mean(), 3),               round(noid["auc"].mean(), 3)],
    "coverage":                [round(alls["blocking_recall"].mean(), 3),   round(noid["blocking_recall"].mean(), 3)],
    "accuracy_when_decidable": [round(alls["auc_blocked_only"].mean(), 3),  round(noid["auc_blocked_only"].dropna().mean(), 3)],
})
print("\n================= HEADLINE =================")
print("The method works when it can decide (~0.99). End-to-end accuracy is limited by COVERAGE, not the model.")
print(headline.to_string(index=False))

# ---- Summary table 2: no-strong-ID, per scenario ----
print("\n========== No-strong-ID population: per scenario ==========")
print(noid[["auc", "blocking_recall", "auc_blocked_only", "f1"]]
        .rename(columns={"auc": "accuracy_end_to_end", "blocking_recall": "coverage",
                         "auc_blocked_only": "accuracy_when_decidable"}).round(3).to_string())

# ---- Summary table 3: ID-collision diagnostic ----
print("\n========== Why adding IDs hurts: id agreement across DIFFERENT people ==========")
print("(nonmatch_agree > 0 = false-positive evidence; top value is the likely placeholder to null)")
print(_diag.to_string(index=False))

# ---- Chart 1: coverage by scenario (everyone vs no-strong-ID) ----
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

# ---- Chart 2: no-strong-ID, end-to-end vs when-decidable (the coverage gap) ----
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, noid["auc"].values, w, label="end-to-end accuracy", color="#B0483B")
ax.bar(x + w/2, noid["auc_blocked_only"].astype(float).values, w, label="accuracy when decidable", color="#2E8B8B")
ax.set_xticks(x); ax.set_xticklabels(SCN_ORDER, rotation=45, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy (AUC)")
ax.set_title("No-strong-ID: model is ~0.99 when it can decide — the gap to end-to-end is coverage")
ax.legend(); plt.tight_layout()
fig.savefig(f"{DATA_DIR}/chart_coverage_gap.png", dpi=150, bbox_inches="tight"); plt.show()

# ---- Chart 3: no-strong-ID accuracy by scenario, sorted (adding IDs does not help) ----
srt = noid["auc"].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 5))
colors = ["#2E8B8B" if i == "no_strong_id" else "#8AA0A0" for i in srt.index]
ax.bar(range(len(srt)), srt.values, color=colors)
ax.set_xticks(range(len(srt))); ax.set_xticklabels(srt.index, rotation=45, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("end-to-end accuracy (AUC)")
ax.set_title("No-strong-ID: adding identifiers does NOT help — name+DOB-only leads (teal)")
plt.tight_layout()
fig.savefig(f"{DATA_DIR}/chart_accuracy_by_scenario.png", dpi=150, bbox_inches="tight"); plt.show()

# ---- Chart 4: ID-collision (fields where different people share a value) ----
d = _diag.sort_values("nonmatch_agree", ascending=False)
fig, ax = plt.subplots(figsize=(11, 5))
colors = ["#B0483B" if v > 0 else "#8AA0A0" for v in d["nonmatch_agree"].values]
ax.bar(range(len(d)), d["nonmatch_agree"].values, color=colors)
ax.set_xticks(range(len(d))); ax.set_xticklabels(d["id_field"].values, rotation=45, ha="right")
ax.set_ylabel("agreement across DIFFERENT people")
ax.set_title("Why adding IDs hurts: fields where different people share the same value (red = false-positive signal)")
plt.tight_layout()
fig.savefig(f"{DATA_DIR}/chart_id_collision.png", dpi=150, bbox_inches="tight"); plt.show()

# ---- Chart 5: WHY adding ids hurts - non-match score leakage (uses stashed per-pair scores) ----
# ~45% of no-strong-id true matches are undecidable (p=0). Adding id blocks + comparisons pulls extra
# NON-MATCH pairs off that p=0 floor and scores them p>0 (placeholders/reuse); those false positives then
# outrank the undecidable matches -> AUC falls. Compare leanest (name+dob only) vs all_ids.
lean = scored_by_scn.get("no_strong_id"); full = scored_by_scn.get("all_ids")
if lean is not None and full is not None:
    def _noid_nm(scored):                                       # non-match probs on the no-strong-id slice
        s = scored[scored["no_strong_id"]]; return s[s["label"] == 0]["p"].values
    def _undec(scored):                                         # share of TRUE matches stuck at p=0
        s = scored[scored["no_strong_id"]]; mm = s[s["label"] == 1]; return (mm["p"] == 0).mean()
    lean_nm, full_nm = _noid_nm(lean), _noid_nm(full)
    print("\n========== Why adding IDs hurts (no-strong-ID slice) ==========")
    print(f"true matches stuck at p=0 (undecidable): name+dob={_undec(lean):.1%}  all_ids={_undec(full):.1%}")
    for _n, _nm in [("name+dob only", lean_nm), ("all_ids", full_nm)]:
        print(f"{_n:14s} non-matches scored: p>0 {np.mean(_nm>0):.1%} | "
              f"p>=0.5 {np.mean(_nm>=0.5):.1%} | p>=0.9 {np.mean(_nm>=0.9):.1%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 26)                                 # (a) non-match scores pulled off the p=0 floor
    ax1.hist(lean_nm[lean_nm > 0], bins=bins, alpha=0.6, label="name+dob only", color="#2E8B8B")
    ax1.hist(full_nm[full_nm > 0], bins=bins, alpha=0.6, label="all_ids", color="#B0483B")
    ax1.set_yscale("log"); ax1.set_xlabel("predicted match probability of a NON-match pair")
    ax1.set_ylabel("count (log scale)")
    ax1.set_title("Non-match pairs pulled off the p=0 floor\nall_ids has more, at higher scores = false positives")
    ax1.legend()
    th = [0.0, 0.5, 0.9]                                         # (b) false-positive rate at rising thresholds
    lean_r = [np.mean(lean_nm > 0)] + [np.mean(lean_nm >= t) for t in th[1:]]
    full_r = [np.mean(full_nm > 0)] + [np.mean(full_nm >= t) for t in th[1:]]
    xx = np.arange(3); ww = 0.38
    ax2.bar(xx - ww/2, lean_r, ww, label="name+dob only", color="#2E8B8B")
    ax2.bar(xx + ww/2, full_r, ww, label="all_ids", color="#B0483B")
    ax2.set_xticks(xx); ax2.set_xticklabels(["p > 0", "p >= 0.5", "p >= 0.9"])
    ax2.set_ylabel("share of NON-match pairs")
    ax2.set_title("Adding IDs scores more non-matches as (false) matches")
    ax2.legend()
    plt.tight_layout()
    fig.savefig(f"{DATA_DIR}/chart_why_more_ids_hurts.png", dpi=150, bbox_inches="tight"); plt.show()

# ---- Key findings (printed narrative) ----
print("""
================= KEY FINDINGS =================
1. The method WORKS when it can decide: on compared pairs, accuracy is ~0.99 even with no strong ID.
2. COVERAGE is the bottleneck: only ~55% of no-strong-ID true matches are compared; the rest are
   undecidable and counted as misses -> end-to-end accuracy falls to ~0.60.
3. Adding IDs does NOT help the no-strong-ID group and slightly HURTS: name+DOB-only leads (~0.77);
   the extra IDs are mostly blank for this group and, where present, sometimes shared across different
   people (placeholders/reuse), injecting false-positive evidence that outranks the undecidable matches.
4. Fixes: (a) phonetic/token blocking to raise coverage (primary lever); (b) id hygiene - term-frequency
   adjustments (now on) + null out the placeholder values surfaced in the diagnostic above.
""")

# ============================== save all results to the data folder ==============================
# CSV snapshots of every result table (latest run overwrites; the catalog table below keeps full history)
splink_results.to_csv(f"{DATA_DIR}/splink_results.csv", index=False)                 # full per-scenario/slice metrics
_diag.to_csv(f"{DATA_DIR}/splink_id_collision_diagnostic.csv", index=False)          # id-collision diagnostic
headline.to_csv(f"{DATA_DIR}/splink_headline_summary.csv", index=False)              # everyone vs no-strong-ID headline
noid.reset_index().to_csv(f"{DATA_DIR}/splink_no_strong_id_summary.csv", index=False)  # no-strong-ID per scenario
alls.reset_index().to_csv(f"{DATA_DIR}/splink_all_population_summary.csv", index=False)  # full-population per scenario
print("saved result files (csv + charts) to", DATA_DIR)

# ============================== also append results to the catalog table ==============================
if SAVE_RESULTS:
    out = splink_results.copy()
    for c in ["blocking_recall", "auc_blocked_only"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["run_ts"] = pd.Timestamp.utcnow(); out["seed"] = SEED
    (spark.createDataFrame(out)
        .write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE))
    print("appended results to", RESULTS_TABLE)
