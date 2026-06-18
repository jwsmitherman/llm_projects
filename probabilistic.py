# IDEN-43831 ANM spike - probabilistic linkage (the ticket's "e.g. Splink").
# Two approaches in one file, both scoring the same labeled pairs and appending to one results table:
#   Section A - Fellegi-Sunter on the pcs features (apples-to-apples with the GBMs; uses Feat).
#   Section B - the Splink LIBRARY on raw records (its own blocking/comparisons/clustering; uses _1/_2 fields).
# The library can't consume pcs pairwise features, so it reshapes the raw columns; A is the consistent
# comparison, B is the productionization path (blocking + clustering at scale). Toggle with RUN_FS / RUN_SPLINK.
#
# Databricks: %pip install splink==4.0.7 "sqlglot<26"   then dbutils.library.restartPython()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ============================== shared setup ==============================
SEED          = 43831
CATALOG       = "eciscor_prod"
DS            = f"{CATALOG}.pcis_data_science"
TRAIN_TABLE   = f"{DS}.anm_training_data_sample_featurized"
FULLSET       = f"{DS}.anm_training_data_fullset"
RESULTS_TABLE = f"{DS}.anm_scenario_results"
SAMPLE_PER_CLASS = 50000
THRESH        = 0.90
SAVE_RESULTS  = True
RUN_FS        = True
RUN_SPLINK    = True

FEATURE_NAMES = ["f_dob", "f_cob", "f_anum", "f_ssn", "f_fin",
                 "f_dl", "f_passport", "f_i94", "f_travel_doc", "f_cartos", "f_name"]

src  = spark.table(TRAIN_TABLE).toPandas()
meta = src[["label", "no_strong_id", "dl_present"]].copy()
y    = meta["label"].astype(int).values
tr, te = train_test_split(np.arange(len(src)), test_size=0.25, stratify=y, random_state=SEED)
te_noid = te[meta["no_strong_id"].values[te]]
print(f"rows={len(src)} train={len(tr)} test={len(te)} no-strong-id test={len(te_noid)}")

all_results = []

# ============================== Section A: Fellegi-Sunter on pcs features ==============================
if RUN_FS:
    Feat = src[FEATURE_NAMES].copy()
    NAME = ["f_cartos", "f_name", "f_dob", "f_cob"]
    SCN = {
        "baseline":             NAME + ["f_anum", "f_ssn", "f_fin"],
        "no_ssn":               NAME + ["f_anum", "f_fin"],
        "no_strong_id":         NAME,
        "with_dl":              NAME + ["f_anum", "f_ssn", "f_fin", "f_dl"],
        "with_dl_no_strong_id": NAME + ["f_dl"],
        "with_other_ids":       NAME + ["f_anum", "f_ssn", "f_fin", "f_dl", "f_passport", "f_i94", "f_travel_doc"],
        "other_ids_no_strong":  NAME + ["f_dl", "f_passport", "f_i94", "f_travel_doc"],
        "all_ids":              FEATURE_NAMES,
    }
    ALPHA, K = 0.5, 4

    def fs_levels(col):
        x = Feat[col].astype(float).values
        lv = np.full(len(x), 3, dtype=int)
        lv[(x >= 0.80) & (x < 0.95)] = 2
        lv[x >= 0.95] = 1
        lv[(x < 0) | np.isnan(x)] = 0
        return lv

    Lv = {c: fs_levels(c) for c in FEATURE_NAMES}
    ytr = y[tr]
    n_m, n_u = int(ytr.sum()), int((1 - ytr).sum())
    prior_mw = np.log2((n_m / (n_m + n_u)) / (n_u / (n_m + n_u)))

    mw = {}
    for c in FEATURE_NAMES:
        lv_tr, w = Lv[c][tr], np.zeros(K)
        for k in range(K):
            m = (((lv_tr == k) & (ytr == 1)).sum() + ALPHA) / (n_m + ALPHA * K)
            u = (((lv_tr == k) & (ytr == 0)).sum() + ALPHA) / (n_u + ALPHA * K)
            w[k] = np.log2(m / u)
        mw[c] = w
    contrib = pd.DataFrame({c: mw[c][Lv[c]] for c in FEATURE_NAMES})

    print("\n[A] Fellegi-Sunter match weights by feature/level (null, high, mid, low):")
    for c in FEATURE_NAMES:
        print(f"  {c:13s} " + " ".join(f"{v:+.2f}" for v in mw[c]))

    def fs_score(cols, idx):
        W = prior_mw + contrib.iloc[idx][cols].sum(axis=1).values
        return 1.0 / (1.0 + 2.0 ** (-np.clip(W, -50, 50)))

    rows = []
    for scn, cols in SCN.items():
        for slice_name, idx in [("all", te), ("no_strong_id", te_noid)]:
            if len(idx) == 0 or len(np.unique(y[idx])) < 2:
                continue
            p = fs_score(cols, idx)
            rows.append(dict(scenario=scn, model="fellegi_sunter", eval=slice_name, n=len(idx),
                             auc=round(roc_auc_score(y[idx], p), 4),
                             ap=round(average_precision_score(y[idx], p), 4),
                             f1=round(f1_score(y[idx], (p >= THRESH).astype(int), zero_division=0), 4)))
    fs_results = pd.DataFrame(rows)
    all_results.append(fs_results)
    print("\n[A] fellegi-sunter results:")
    print(fs_results.sort_values(["eval", "auc"], ascending=[True, False]).to_string(index=False))

# ============================== Section B: Splink library on raw records ==============================
if RUN_SPLINK:
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    ATTRS = ["firstName", "middleName", "lastName", "dob", "cob", "anumber", "ssn", "fin",
             "driversLicense", "passport", "i94", "travel_doc"]
    RID = "party_record_receipt"

    # raw _1/_2 fields must be in the training table; else re-sample them from fullset (same seed)
    if f"{RID}_1" in src.columns and "firstName_1" in src.columns:
        pairs = src
    else:
        print("[B] raw _1/_2 columns not in TRAIN_TABLE - re-sampling from fullset "
              "(persist `sample` in the data file to avoid this)")
        df = spark.table(FULLSET)
        n_pos = df.filter("label = 1").count(); n_neg = df.filter("label = 0").count()
        fr = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)), 0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}
        pairs = df.sampleBy("label", fractions=fr, seed=SEED).toPandas()

    def side(n):
        ren = {f"{RID}_{n}": "record_id"}; ren.update({f"{a}_{n}": a for a in ATTRS})
        return pairs[list(ren)].rename(columns=ren)
    records = (pd.concat([side("1"), side("2")], ignore_index=True)
                 .dropna(subset=["record_id"]).drop_duplicates("record_id").reset_index(drop=True))

    def _usable(c1, c2):
        a, b = pairs[c1].astype("string"), pairs[c2].astype("string")
        return a.notna() & b.notna() & (a.str.strip() != "") & (b.str.strip() != "")
    lab = pairs[[f"{RID}_1", f"{RID}_2", "label"]].copy()
    lab["no_strong_id"] = (~(_usable("ssn_1", "ssn_2") | _usable("anumber_1", "anumber_2")
                             | _usable("fin_1", "fin_2"))).values
    lab = lab.dropna(subset=[f"{RID}_1", f"{RID}_2"])
    lab["key"] = lab.apply(lambda r: tuple(sorted([r[f"{RID}_1"], r[f"{RID}_2"]])), axis=1)
    lab = lab.drop_duplicates("key")
    print("[B] records:", len(records), "labeled pairs:", len(lab))

    NAME_COMPS = [cl.ForenameSurnameComparison("firstName", "lastName"),
                  cl.DateOfBirthComparison("dob", input_is_string=True),
                  cl.ExactMatch("cob")]
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

    def splink_scenario(ids):
        comps = NAME_COMPS + [id_comp(ID_COL[k]) for k in ids]
        blocking = [block_on("firstName", "dob"), block_on("lastName", "dob")] + \
                   [block_on(ID_COL[k]) for k in ids]
        settings = SettingsCreator(link_type="dedupe_only", unique_id_column_name="record_id",
                                   comparisons=comps, blocking_rules_to_generate_predictions=blocking)
        linker = Linker(records, settings, db_api=DuckDBAPI())
        det = [block_on("firstName", "lastName", "dob")] + [block_on(ID_COL[k]) for k in ids[:1]]
        linker.training.estimate_probability_two_random_records_match(det, recall=0.7)
        linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("firstName", "lastName"))
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
        preds = linker.inference.predict().as_pandas_dataframe()
        prob = {tuple(sorted([a, b])): p for a, b, p in
                zip(preds["record_id_l"], preds["record_id_r"], preds["match_probability"])}
        scored = lab.copy()
        scored["p"] = scored["key"].map(prob).fillna(0.0)
        return scored

    rows = []
    for scn, ids in SCN_IDS.items():
        s = splink_scenario(ids)
        for slice_name, sub in [("all", s), ("no_strong_id", s[s["no_strong_id"]])]:
            if len(sub) == 0 or sub["label"].nunique() < 2:
                continue
            rows.append(dict(scenario=scn, model="splink", eval=slice_name, n=len(sub),
                             auc=round(roc_auc_score(sub["label"], sub["p"]), 4),
                             ap=round(average_precision_score(sub["label"], sub["p"]), 4),
                             f1=round(f1_score(sub["label"], (sub["p"] >= THRESH).astype(int), zero_division=0), 4)))
    splink_results = pd.DataFrame(rows)
    all_results.append(splink_results)
    print("\n[B] splink results:")
    print(splink_results.sort_values(["eval", "auc"], ascending=[True, False]).to_string(index=False))

# ============================== write combined results ==============================
results = pd.concat(all_results, ignore_index=True).sort_values(
    ["eval", "scenario", "model"], ascending=[True, True, True])
print("\ncombined results:")
print(results.to_string(index=False))

if SAVE_RESULTS:
    results["run_ts"] = pd.Timestamp.utcnow()
    results["seed"] = SEED
    (spark.createDataFrame(results)
        .write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE))
    print("\nappended results to", RESULTS_TABLE)
