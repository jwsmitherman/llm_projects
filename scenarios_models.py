# IDEN-43831 ANM spike (models). Reads the featurized training table written by the data file,
# runs GBM-class models + baselines across the scenario matrix, evaluates each overall and on the
# no-strong-id slice, prints a leakage check, and appends the results to a table.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED          = 43831
TRAIN_TABLE   = "eciscor_prod.pcis_data_science.anm_training_data_sample_featurized"
RESULTS_TABLE = "eciscor_prod.pcis_data_science.anm_scenario_results"
SAVE_RESULTS  = True

FEATURE_NAMES = ["f_dob", "f_cob", "f_anum", "f_ssn", "f_fin",
                 "f_dl", "f_passport", "f_i94", "f_travel_doc", "f_cartos", "f_name"]

src  = spark.table(TRAIN_TABLE).toPandas()
Feat = src[FEATURE_NAMES].copy()
meta = src[["label", "no_strong_id", "dl_present"]].copy()
# Feat = Feat.replace(-9.0, np.nan)   # optional: treat pcs -9 sentinel as missing (cleaner for GBMs)

# scenario matrix (column subsets of Feat) covering no-ID / Driver's License / other identifiers
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

y = meta["label"].astype(int).values
tr, te = train_test_split(np.arange(len(Feat)), test_size=0.25, stratify=y, random_state=SEED)
noid_mask = meta["no_strong_id"].values
te_noid = te[noid_mask[te]]
print(f"rows={len(Feat)} train={len(tr)} test={len(te)} no-strong-id test={len(te_noid)} "
      f"(pos rate {y[te].mean():.3f} / {(y[te_noid].mean() if len(te_noid) else float('nan')):.3f})")

print("\nsingle-feature separation (AUC; >0.97 = likely crutch / leakage):")
for c in Feat.columns:
    s = Feat.iloc[tr][c].fillna(Feat[c].median())
    auc = roc_auc_score(y[tr], s)
    print(f"  {c:13s} {max(auc, 1 - auc):.3f}")

def build_models():
    m = {}
    try:
        import lightgbm as lgb
        m["lightgbm"] = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                           subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    except Exception as e:
        print("lightgbm unavailable:", e)
    try:
        import xgboost as xgb
        m["xgboost"] = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                         random_state=SEED)
    except Exception as e:
        print("xgboost unavailable:", e)
    try:
        from catboost import CatBoostClassifier
        m["catboost"] = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                                           random_seed=SEED, verbose=0)
    except Exception as e:
        print("catboost unavailable:", e)
    m["hist_gb"] = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=SEED)
    m["random_forest"] = make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))
    m["logreg"] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=1000))
    return m

def best_f1_threshold(y_true, p):
    thr = np.unique(np.round(p, 3))
    thr = thr[(thr > 0) & (thr < 1)]
    if len(thr) == 0:
        return 0.5
    if len(thr) > 300:
        thr = np.quantile(p, np.linspace(0.05, 0.95, 200))
    best_t, best_f = 0.5, -1.0
    for t in thr:
        f = f1_score(y_true, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

rows = []
for scn, cols in SCN.items():
    Xtr = Feat.iloc[tr][cols]
    for name, model in build_models().items():
        model.fit(Xtr, y[tr])
        thr = best_f1_threshold(y[tr], model.predict_proba(Xtr)[:, 1])
        for slice_name, idx in [("all", te), ("no_strong_id", te_noid)]:
            if len(idx) == 0:
                continue
            p = model.predict_proba(Feat.iloc[idx][cols])[:, 1]
            rows.append(dict(
                scenario=scn, model=name, eval=slice_name, n=len(idx),
                auc=round(roc_auc_score(y[idx], p), 4),
                ap=round(average_precision_score(y[idx], p), 4),
                f1=round(f1_score(y[idx], (p >= thr).astype(int), zero_division=0), 4)))

results = pd.DataFrame(rows).sort_values(["eval", "scenario", "auc"], ascending=[True, True, False])
print("\nresults (scenarios x models):")
print(results.to_string(index=False))

noid = results[results["eval"] == "no_strong_id"]
if len(noid):
    print("\nbest model per scenario on no-strong-id slice (AUC):")
    print(noid.sort_values("auc", ascending=False).groupby("scenario").head(1)
              .sort_values("auc", ascending=False).to_string(index=False))

if SAVE_RESULTS:
    results["run_ts"] = pd.Timestamp.utcnow()
    results["seed"] = SEED
    (spark.createDataFrame(results)
        .write.mode("append").option("mergeSchema", "true").saveAsTable(RESULTS_TABLE))
    print("\nsaved results to", RESULTS_TABLE)
