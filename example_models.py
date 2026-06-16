# IDEN-43831 ANM spike: model exploration on the pcs-model-transformed labeled sample.
# Consumes `data` from the data-creation cell (features + label + no_strong_id / dl_both / other_both).
# Ticket angle: can GBM-class models discriminate matches, and do they hold up on the no-strong-ID
# population? Plus a leakage guard (f_fin lesson): if a single id-distance feature separates the
# classes by itself, the model just learns "same id = match" and gives no lift on no-ID records.

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
FEATURES_FULL = ["name_cos", "date_score", "ssn_d", "anum_d", "dl_d", "other_d"]
FEATURES_NOID = ["name_cos", "date_score", "dl_d", "other_d"]
TARGET = "label"

df = data.dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET].astype(int).values
tr, te = train_test_split(df.index.values, test_size=0.25, stratify=y, random_state=SEED)
train, test = df.loc[tr], df.loc[te]
noid_te = test[test["no_strong_id"]]
print(f"train={len(train)} test={len(test)} no-strong-id test={len(noid_te)} "
      f"(pos rate {test[TARGET].mean():.3f} / {noid_te[TARGET].mean():.3f})")

# leakage guard: single-feature separation (direction-agnostic AUC)
print("\nsingle-feature separation (AUC; >0.97 = likely crutch / leakage):")
for c in FEATURES_FULL:
    s = train[c].fillna(train[c].median())
    auc = roc_auc_score(train[TARGET], s)
    print(f"  {c:11s} {max(auc, 1 - auc):.3f}")

# tree models take NaN natively; linear models get imputed + scaled
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
                                       RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1))
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
fitted = {}
for feat_name, feats in [("full", FEATURES_FULL), ("noid", FEATURES_NOID)]:
    for name, model in build_models().items():
        model.fit(train[feats], train[TARGET])
        thr = best_f1_threshold(train[TARGET], model.predict_proba(train[feats])[:, 1])
        fitted[(feat_name, name)] = (model, thr)
        for slice_name, dset in [("all", test), ("no_strong_id", noid_te)]:
            if len(dset) == 0:
                continue
            p = model.predict_proba(dset[feats])[:, 1]
            rows.append(dict(
                model=name, train_feats=feat_name, eval=slice_name, n=len(dset),
                auc=round(roc_auc_score(dset[TARGET], p), 4),
                ap=round(average_precision_score(dset[TARGET], p), 4),
                f1=round(f1_score(dset[TARGET], (p >= thr).astype(int), zero_division=0), 4),
                thr=round(float(thr), 3),
            ))

results = pd.DataFrame(rows).sort_values(["eval", "train_feats", "auc"], ascending=[True, True, False])
print("\nresults:")
print(results.to_string(index=False))

# feature importance from the best available tree model (full features)
for cand in ["lightgbm", "xgboost", "random_forest", "catboost"]:
    key = ("full", cand)
    if key in fitted:
        mdl = fitted[key][0]
        est = mdl.steps[-1][1] if hasattr(mdl, "steps") else mdl
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=FEATURES_FULL).sort_values(ascending=False)
            print(f"\nfeature importance ({cand}, full):")
            print(imp.to_string())
        break

results
