# IDEN-43831 ANM spike - probabilistic linkage (the ticket's "e.g. Splink").
# Two approaches in one file, both scoring the same labeled pairs and appending to one results table:
#   Section A - Fellegi-Sunter on the pcs features (apples-to-apples with the GBMs; uses Feat).
#   Section B - the Splink LIBRARY on raw records (its own blocking/comparisons/clustering; uses _1/_2 fields).
# The library can't consume pcs pairwise features, so it reshapes the raw columns; A is the consistent
# comparison, B is the productionization path (blocking + clustering at scale). Both sections run; install splink first.
#
# Databricks: %pip install splink==4.0.7 "sqlglot<26"   then dbutils.library.restartPython()

import numpy as np                                              # arrays + math (log2, clip, etc.)
import pandas as pd                                             # dataframes for the pulled-down table
from sklearn.model_selection import train_test_split           # makes the held-out test split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score  # the 3 scores we report

# ============================== shared setup ==============================
SEED          = 43831                                           # fixed seed so every run splits/samples identically
CATALOG       = "eciscor_prod"                                  # Unity Catalog name
DS            = f"{CATALOG}.pcis_data_science"                  # the schema both tables live in
TRAIN_TABLE   = f"{DS}.anm_training_data_sample_featurized"     # input: the balanced, feature-built sample
FULLSET       = f"{DS}.anm_training_data_fullset"               # fallback source if raw _1/_2 cols aren't saved
RESULTS_TABLE = f"{DS}.anm_scenario_results"                    # output: where every run's scores get appended
SAMPLE_PER_CLASS = 50000                                        # target rows per label if we have to re-sample fullset
THRESH        = 0.90                                            # probability cutoff used only for the f1 score
SAVE_RESULTS  = True                                            # flip False to test without writing to the table

# the 11 pcs similarity features (one per identifier/name field); order is fixed and reused below
FEATURE_NAMES = ["f_dob", "f_cob", "f_anum", "f_ssn", "f_fin",
                 "f_dl", "f_passport", "f_i94", "f_travel_doc", "f_cartos", "f_name"]

src  = spark.table(TRAIN_TABLE).toPandas()                      # read the Spark table into a local pandas df
meta = src[["label", "no_strong_id", "dl_present"]].copy()      # pull the 3 bookkeeping cols (truth + 2 flags)
y    = meta["label"].astype(int).values                         # ground truth: 1 = same person, 0 = different
# split row positions 75/25, keeping the match/non-match ratio in both halves (stratify=y)
tr, te = train_test_split(np.arange(len(src)), test_size=0.25, stratify=y, random_state=SEED)
te_noid = te[meta["no_strong_id"].values[te]]                   # the subset of TEST pairs that have no SSN/A#/FIN
print(f"rows={len(src)} train={len(tr)} test={len(te)} no-strong-id test={len(te_noid)}")  # sanity counts

all_results = []                                                # collect Section A + B result tables, concat at end

# ============================== Section A: Fellegi-Sunter on pcs features ==============================
Feat = src[FEATURE_NAMES].copy()                                # just the 11 feature columns
NAME = ["f_cartos", "f_name", "f_dob", "f_cob"]                 # the always-available signals (name + dob + country)
# each scenario = which feature columns the model is allowed to "see" (simulates missing identifiers)
SCN = {
    "baseline":             NAME + ["f_anum", "f_ssn", "f_fin"],                                  # name + all strong ids
    "no_ssn":               NAME + ["f_anum", "f_fin"],                                           # strong ids minus ssn
    "no_strong_id":         NAME,                                                                 # name only, no ids
    "with_dl":              NAME + ["f_anum", "f_ssn", "f_fin", "f_dl"],                          # strong ids + drivers license
    "with_dl_no_strong_id": NAME + ["f_dl"],                                                      # name + DL only (the key test)
    "with_other_ids":       NAME + ["f_anum", "f_ssn", "f_fin", "f_dl", "f_passport", "f_i94", "f_travel_doc"],  # everything
    "other_ids_no_strong":  NAME + ["f_dl", "f_passport", "f_i94", "f_travel_doc"],               # name + weak ids, no strong
    "all_ids":              FEATURE_NAMES,                                                        # all 11 features
}
ALPHA, K = 0.5, 4                                               # ALPHA = Laplace smoothing; K = number of similarity buckets

def fs_levels(col):                                             # turn one continuous similarity into a 0-3 bucket
    x = Feat[col].astype(float).values                          # the similarity values for this feature
    lv = np.full(len(x), 3, dtype=int)                          # default bucket 3 = "low / disagree"
    lv[(x >= 0.80) & (x < 0.95)] = 2                            # bucket 2 = "medium agreement"
    lv[x >= 0.95] = 1                                           # bucket 1 = "high agreement"
    lv[(x < 0) | np.isnan(x)] = 0                               # bucket 0 = missing (pcs uses -9 sentinel) / null
    return lv                                                   # array of bucket ids, one per pair

Lv = {c: fs_levels(c) for c in FEATURE_NAMES}                   # precompute the bucket array for every feature
ytr = y[tr]                                                     # truth labels for the TRAIN rows only
n_m, n_u = int(ytr.sum()), int((1 - ytr).sum())                # count of matches (m) and non-matches (u) in train
prior_mw = np.log2((n_m / (n_m + n_u)) / (n_u / (n_m + n_u)))   # base log-odds before looking at any feature

mw = {}                                                         # match weights: per feature, per bucket
for c in FEATURE_NAMES:                                         # learn weights one feature at a time
    lv_tr, w = Lv[c][tr], np.zeros(K)                           # this feature's train buckets + empty weight vector
    for k in range(K):                                          # for each of the 4 buckets
        m = (((lv_tr == k) & (ytr == 1)).sum() + ALPHA) / (n_m + ALPHA * K)  # P(bucket k | match), smoothed
        u = (((lv_tr == k) & (ytr == 0)).sum() + ALPHA) / (n_u + ALPHA * K)  # P(bucket k | non-match), smoothed
        w[k] = np.log2(m / u)                                   # the weight = log2 ratio (+ = evidence of match)
    mw[c] = w                                                   # store this feature's 4 weights
# map every pair's bucket back to its learned weight -> one weight value per pair per feature
contrib = pd.DataFrame({c: mw[c][Lv[c]] for c in FEATURE_NAMES})

print("\n[A] Fellegi-Sunter match weights by feature/level (null, high, mid, low):")  # readable weight table
for c in FEATURE_NAMES:                                         # print each feature's 4 weights
    print(f"  {c:13s} " + " ".join(f"{v:+.2f}" for v in mw[c]))

def fs_score(cols, idx):                                        # score a set of pairs using only `cols` features
    W = prior_mw + contrib.iloc[idx][cols].sum(axis=1).values   # total log-odds = prior + sum of feature weights
    return 1.0 / (1.0 + 2.0 ** (-np.clip(W, -50, 50)))          # convert log-odds -> probability (clip avoids overflow)

rows = []                                                       # collect one result row per scenario x slice
for scn, cols in SCN.items():                                   # loop over the 8 scenarios
    for slice_name, idx in [("all", te), ("no_strong_id", te_noid)]:  # score on full test, then the no-id subset
        if len(idx) == 0 or len(np.unique(y[idx])) < 2:         # skip if empty or only one class (AUC undefined)
            continue
        p = fs_score(cols, idx)                                 # predicted match probabilities for these pairs
        rows.append(dict(scenario=scn, model="fellegi_sunter", eval=slice_name, n=len(idx),
                         auc=round(roc_auc_score(y[idx], p), 4),                 # ranking quality (0.5=random,1=perfect)
                         ap=round(average_precision_score(y[idx], p), 4),        # precision-recall summary
                         f1=round(f1_score(y[idx], (p >= THRESH).astype(int), zero_division=0), 4)))  # f1 at THRESH
fs_results = pd.DataFrame(rows)                                 # Section A's results as a dataframe
all_results.append(fs_results)                                  # add to the combined bucket
print("\n[A] fellegi-sunter results:")
print(fs_results.sort_values(["eval", "auc"], ascending=[True, False]).to_string(index=False))  # best AUC on top

# ============================== Section B: Splink library on raw records ==============================
import splink.comparison_library as cl                          # prebuilt comparison templates (name, dob, etc.)
from splink import DuckDBAPI, Linker, SettingsCreator, block_on # core splink objects (DuckDB engine on the driver)

# the raw attribute fields we reshape from the _1/_2 pair layout into single records
ATTRS = ["firstName", "middleName", "lastName", "dob", "cob", "anumber", "ssn", "fin",
         "driversLicense", "passport", "i94", "travel_doc"]
RID = "party_record_receipt"                                    # the per-record id column prefix (_1 / _2)

# splink needs RECORD-level rows; use the training table's raw cols if present, else re-pull from fullset
if f"{RID}_1" in src.columns and "firstName_1" in src.columns:  # check the raw _1/_2 columns were persisted
    pairs = src                                                 # good - reuse the exact same rows as Section A
else:
    print("[B] raw _1/_2 columns not in TRAIN_TABLE - re-sampling from fullset "
          "(persist `sample` in the data file to avoid this)")  # warn: these rows won't be identical to A's
    df = spark.table(FULLSET)                                   # go back to the big labeled source
    n_pos = df.filter("label = 1").count(); n_neg = df.filter("label = 0").count()  # class counts for balancing
    fr = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)), 0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}  # sample fractions
    pairs = df.sampleBy("label", fractions=fr, seed=SEED).toPandas()  # balanced sample, same seed

def side(n):                                                    # pull one side (_1 or _2) and strip the suffix
    ren = {f"{RID}_{n}": "record_id"}; ren.update({f"{a}_{n}": a for a in ATTRS})  # rename map _1/_2 -> bare names
    return pairs[list(ren)].rename(columns=ren)                 # df of just that side's columns, renamed
# stack side 1 on top of side 2, drop blanks, keep each record once -> the deduped record table splink links
records = (pd.concat([side("1"), side("2")], ignore_index=True)
             .dropna(subset=["record_id"]).drop_duplicates("record_id").reset_index(drop=True))

def _usable(c1, c2):                                            # True where BOTH sides have a non-blank value
    a, b = pairs[c1].astype("string"), pairs[c2].astype("string")  # as strings so we can test for blanks
    return a.notna() & b.notna() & (a.str.strip() != "") & (b.str.strip() != "")
lab = pairs[[f"{RID}_1", f"{RID}_2", "label"]].copy()           # the truth table: which two records, and the label
# flag pairs where neither strong id (ssn/anumber/fin) is usable -> the no-strong-id population
lab["no_strong_id"] = (~(_usable("ssn_1", "ssn_2") | _usable("anumber_1", "anumber_2")
                         | _usable("fin_1", "fin_2"))).values
lab = lab.dropna(subset=[f"{RID}_1", f"{RID}_2"])               # drop rows missing a record id
lab["key"] = lab.apply(lambda r: tuple(sorted([r[f"{RID}_1"], r[f"{RID}_2"]])), axis=1)  # order-independent pair key
lab = lab.drop_duplicates("key")                                # one row per unordered pair
# fixed prior from labels (true-match pairs / total possible record pairs); avoids the brittle
# deterministic-rule recall estimate that errors when strict rules over-match on real data
_npairs = len(records) * (len(records) - 1) / 2                 # total possible pairs among all records (n choose 2)
P_MATCH = max(1e-6, min(0.5, int((lab["label"] == 1).sum()) / _npairs))  # prior P(two random records match), clamped
print("[B] records:", len(records), "labeled pairs:", len(lab), "prior P_MATCH:", round(P_MATCH, 8))

# how splink compares each field: fuzzy name, date-of-birth aware, exact country
NAME_COMPS = [cl.ForenameSurnameComparison("firstName", "lastName"),  # handles swaps/typos in first+last
              cl.DateOfBirthComparison("dob", input_is_string=True),  # graded by how close the dates are
              cl.ExactMatch("cob")]                                   # country of birth: match or not
def id_comp(col):                                               # how splink compares an id field
    return cl.LevenshteinAtThresholds(col, [1, 2]).configure(term_frequency_adjustments=True)  # edit-distance + rarity boost
# short scenario key -> real column name, so scenarios can name ids compactly
ID_COL = {"anum": "anumber", "ssn": "ssn", "fin": "fin", "dl": "driversLicense",
          "passport": "passport", "i94": "i94", "travel_doc": "travel_doc"}
# same 8 scenarios as Section A, but expressed as which id columns to include
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

def splink_scenario(ids):                                       # train + predict splink for one scenario's id set
    comps = NAME_COMPS + [id_comp(ID_COL[k]) for k in ids]      # name/dob/cob comparisons + an id comparison per id
    # blocking = which candidate pairs to even consider (compare only records sharing a key, for speed)
    blocking = [block_on("firstName", "dob"), block_on("lastName", "dob")] + \
               [block_on(ID_COL[k]) for k in ids]
    settings = SettingsCreator(link_type="dedupe_only",         # dedupe within one table (vs linking two tables)
                               unique_id_column_name="record_id",
                               comparisons=comps,
                               blocking_rules_to_generate_predictions=blocking,
                               probability_two_random_records_match=P_MATCH)  # the prior we computed above
    linker = Linker(records, settings, db_api=DuckDBAPI())      # build the linker over the record table
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)  # learn "u" weights (agreement among non-matches)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("firstName", "lastName"))  # EM pass 1
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))                     # EM pass 2
    preds = linker.inference.predict().as_pandas_dataframe()    # score all candidate pairs -> match_probability
    # build a lookup: unordered record-id pair -> predicted probability
    prob = {tuple(sorted([a, b])): p for a, b, p in
            zip(preds["record_id_l"], preds["record_id_r"], preds["match_probability"])}
    scored = lab.copy()                                         # start from the truth table
    scored["p"] = scored["key"].map(prob).fillna(0.0)           # attach splink's prob; pairs it never blocked -> 0
    return scored                                              # truth + predicted prob, ready to score

rows = []                                                       # collect Section B result rows
for scn, ids in SCN_IDS.items():                                # loop the 8 scenarios
    s = splink_scenario(ids)                                    # train splink and score the labeled pairs
    for slice_name, sub in [("all", s), ("no_strong_id", s[s["no_strong_id"]])]:  # full set, then no-id subset
        if len(sub) == 0 or sub["label"].nunique() < 2:         # skip if empty or single-class (AUC undefined)
            continue
        rows.append(dict(scenario=scn, model="splink", eval=slice_name, n=len(sub),
                         auc=round(roc_auc_score(sub["label"], sub["p"]), 4),            # ranking quality
                         ap=round(average_precision_score(sub["label"], sub["p"]), 4),   # precision-recall summary
                         f1=round(f1_score(sub["label"], (sub["p"] >= THRESH).astype(int), zero_division=0), 4)))  # f1
splink_results = pd.DataFrame(rows)                             # Section B's results
all_results.append(splink_results)                             # add to the combined bucket
print("\n[B] splink results:")
print(splink_results.sort_values(["eval", "auc"], ascending=[True, False]).to_string(index=False))

# ============================== write combined results ==============================
# stack Section A + B and sort so each scenario's fellegi_sunter vs splink rows sit together per slice
results = pd.concat(all_results, ignore_index=True).sort_values(
    ["eval", "scenario", "model"], ascending=[True, True, True])
print("\ncombined results:")
print(results.to_string(index=False))                          # final table to the cell output

if SAVE_RESULTS:                                                # persist this run unless turned off
    results["run_ts"] = pd.Timestamp.utcnow()                  # stamp every row with run time (separates runs)
    results["seed"] = SEED                                      # record the seed used
    (spark.createDataFrame(results)                            # pandas -> Spark df
        .write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE))  # append, tolerate new cols
    print("\nappended results to", RESULTS_TABLE)
