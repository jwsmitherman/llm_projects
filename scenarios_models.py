# IDEN-43831 ANM spike (3/3) models: GBM-class + baselines across every scenario.
# Consumes Feat (feature superset), SCN (scenario -> feature columns), meta (label + no_strong_id).
# One stratified split shared across all scenarios for a fair comparison. Reports overall AUC/AP/F1
# and the same on the no-strong-id slice; plus single-feature separation (f_fin/f_ssn leakage guard).

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 43831
y = meta["label"].astype(int).values
tr, te = train_test_split(np.arange(len(Feat)), test_size=0.25, stratify=y, random_state=SEED)
noid_mask = meta["no_strong_id"].values
te_noid = te[noid_mask[te]]
print(f"train={len(tr)} test={len(te)} no-strong-id test={len(te_noid)} "
      f"(pos rate {y[te].mean():.3f} / {(y[te_noid].mean() if len(te_noid) else float('nan')):.3f})")

print("\nsingle-feature separation (AUC; >0.97 = likely crutch / leakage):")
for c in Feat.columns:
    s = Feat.iloc[tr][c].fillna(Feat[c].median())
    auc = roc_auc_score(y[tr], s)
    print(f"  {c:9s} {max(auc, 1 - auc):.3f}")

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

rows, fitted = [], {}
for scn, cols in SCN.items():
    Xtr = Feat.iloc[tr][cols]
    for name, model in build_models().items():
        model.fit(Xtr, y[tr])
        thr = best_f1_threshold(y[tr], model.predict_proba(Xtr)[:, 1])
        fitted[(scn, name)] = model
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
print("\nresults (all scenarios x models):")
print(results.to_string(index=False))

# best model per scenario on the no-strong-id slice = the AC-1 evidence
noid = results[results["eval"] == "no_strong_id"]
if len(noid):
    print("\nbest model per scenario on no-strong-id slice (AUC):")
    print(noid.sort_values("auc", ascending=False).groupby("scenario").head(1)
              .sort_values("auc", ascending=False).to_string(index=False))

results

