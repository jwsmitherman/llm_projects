# IDEN-43831 ANM spike (data + features). Source = anm_training_data_fullset: already enriched and
# labeled, attributes as <name>_1 / <name>_2. No join needed. Carries DL (driversLicense) and the
# "other identifiers" the ticket calls out (passport, i94, travel_doc) alongside the strong ids.

from pyspark.sql import functions as F
import sys
import pandas as pd
import numpy as np

CATALOG = "eciscor_prod"
DS      = f"{CATALOG}.pcis_data_science"
FULLSET = f"{DS}.anm_training_data_fullset"

SAMPLE_PER_CLASS = 50000
SEED             = 43831
DISCOVER         = True

df = spark.table(FULLSET)
if DISCOVER:
    print("fullset rows:", df.count(), "cols:", len(df.columns))
    print("columns:", sorted(df.columns))

n_pos = df.filter("label = 1").count()
n_neg = df.filter("label = 0").count()
fractions = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)),
             0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}
sample = df.sampleBy("label", fractions=fractions, seed=SEED).toPandas()
print("pop   :", {"matches": n_pos, "nonmatches": n_neg})
print("sample:", sample["label"].value_counts().to_dict())

# ---- pcs-model transform ----------------------------------------------------------------------
if "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov" not in sys.path:
    sys.path.insert(0, "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov")
from pcs_models.data_processing.distance_metrics import (
    norm_levenshtein_sim, exact_string_match, calculate_cartcos, join_name_parts)

# real fullset columns use the _1 / _2 suffix; cob only (no countryOfCitizenship in this table)
FEATURE_NAMES = ["f_dob", "f_cob", "f_anum", "f_ssn", "f_fin",
                 "f_dl", "f_passport", "f_i94", "f_travel_doc", "f_cartos", "f_name"]

def build_all_features(d: pd.DataFrame) -> pd.DataFrame:
    feats = np.empty((len(d), len(FEATURE_NAMES)), dtype="float32")
    for i, (_, r) in enumerate(d.reset_index(drop=True).iterrows()):
        f_dob   = norm_levenshtein_sim([r["dob_1"], r["dob_2"]])
        f_cob   = exact_string_match([r["cob_1"], r["cob_2"]])
        f_anum  = norm_levenshtein_sim([r["anumber_1"], r["anumber_2"]])
        f_ssn   = norm_levenshtein_sim([r["ssn_1"], r["ssn_2"]])
        f_fin   = norm_levenshtein_sim([r["fin_1"], r["fin_2"]])
        f_dl    = norm_levenshtein_sim([r["driversLicense_1"], r["driversLicense_2"]])
        f_pass  = norm_levenshtein_sim([r["passport_1"], r["passport_2"]])
        f_i94   = norm_levenshtein_sim([r["i94_1"], r["i94_2"]])
        f_td    = norm_levenshtein_sim([r["travel_doc_1"], r["travel_doc_2"]])
        nl = [r["firstName_1"], r["middleName_1"], r["lastName_1"],
              r["firstName_2"], r["middleName_2"], r["lastName_2"]]
        f_cartos = calculate_cartcos(nl)
        f_name = norm_levenshtein_sim([join_name_parts(r["firstName_1"], "", ""),
                                       join_name_parts(r["firstName_2"], "", "")])
        feats[i] = [f_dob, f_cob, f_anum, f_ssn, f_fin, f_dl, f_pass, f_i94, f_td, f_cartos, f_name]
    return pd.DataFrame(feats, columns=FEATURE_NAMES)

Feat = build_all_features(sample)

# label + coverage flags. strong ids = anumber / ssn / fin; no_strong_id = none usable on the pair
def both_present(a, b):
    a, b = sample[a].astype("string"), sample[b].astype("string")
    return a.notna() & b.notna() & (a.str.strip() != "") & (b.str.strip() != "")

ssn_ok  = both_present("ssn_1", "ssn_2")
anum_ok = both_present("anumber_1", "anumber_2")
fin_ok  = both_present("fin_1", "fin_2")
meta = pd.DataFrame({"label": sample["label"].astype(int).values,
                     "no_strong_id": (~(ssn_ok | anum_ok | fin_ok)).values,
                     "dl_present": both_present("driversLicense_1", "driversLicense_2").values})

# scenario matrix covering the ticket: no-ID, Driver's License, and other identifiers
NAME = ["f_cartos", "f_name", "f_dob", "f_cob"]          # always-on non-id signal
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

print("\nfeatures:", list(Feat.columns))
print("no-strong-id pairs:", int(meta["no_strong_id"].sum()),
      "| DL present:", int(meta["dl_present"].sum()),
      "| no-strong-id WITH dl:", int((meta["no_strong_id"] & meta["dl_present"]).sum()))
print("\nfeature means by label:")
print(pd.concat([Feat, meta["label"]], axis=1).groupby("label").mean())
