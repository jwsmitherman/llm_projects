# IDEN-43831 ANM spike (data + features). Pair tables hold only (identityid, pr1, pr2, label):
# pr1/pr2 are the party keys per side, identityid is the cluster. ALL attributes (names, dob, ssn,
# fin, aNumber, DL) are joined from a party-attributes table keyed by pr1/pr2, then transformed.

from pyspark.sql import functions as F
import sys
import pandas as pd
import numpy as np

CATALOG     = "eciscor_prod"
DS          = f"{CATALOG}.pcis_data_science"
MD          = f"{CATALOG}.pcis_metadata"
MATCHES     = f"{DS}.anm_training_data_matches"
NONMATCHES  = f"{DS}.anm_training_data_nonmatches"

SAMPLE_PER_CLASS = 50000
SEED             = 43831
DISCOVER         = True

# ---- party-attribute source (set these after DISCOVER) ---------------------------------------
# pr1/pr2 in the pair tables join to PKEY in PARTY. Every feature attribute, incl DL, comes from here.
PARTY  = f"{MD}.rpt_bronze_party"      # >>> table holding party attributes
PKEY   = "identityId"                  # >>> key in PARTY that pr1/pr2 match (try identityId or _id)
DL_COL = "driversLicenseNumber"        # >>> DL column name in PARTY
ATTRS  = ["firstName", "middleName", "lastName", "dateOfBirth", "aNumber", "ssn", "fin",
          "countryOfBirth", "countryOfCitizenship", DL_COL]
# ----------------------------------------------------------------------------------------------

m  = spark.table(MATCHES)
nm = spark.table(NONMATCHES)
if DISCOVER:
    print("matches rows:", m.count(), "cols:", len(m.columns)); m.printSchema()
    print("pair columns:", sorted(m.columns))
    for t in [PARTY, f"{MD}.rpt_bronze_party", f"{MD}.ps_parties_to_upsert", f"{MD}.mv_es_party_identity"]:
        try:
            print(t, "->", sorted(spark.table(t).columns))
        except Exception as e:
            print(t, "ERR", str(e)[:80])

common  = [c for c in m.columns if c in nm.columns]
labeled = (m.select(*common).withColumn("label", F.lit(1))
            .unionByName(nm.select(*common).withColumn("label", F.lit(0))))

n_pos = labeled.filter("label = 1").count()
n_neg = labeled.filter("label = 0").count()
fractions = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)),
             0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}
sampled = labeled.sampleBy("label", fractions=fractions, seed=SEED)

# join party attributes onto each side (pr1 -> *1, pr2 -> *2); DL rides along as just another attr
party = spark.table(PARTY).select([PKEY] + ATTRS).dropDuplicates([PKEY])
s1 = party.select([F.col(PKEY).alias("pr1")] + [F.col(a).alias(a + "1") for a in ATTRS])
s2 = party.select([F.col(PKEY).alias("pr2")] + [F.col(a).alias(a + "2") for a in ATTRS])
sample = sampled.join(s1, on="pr1", how="left").join(s2, on="pr2", how="left").toPandas()

print("pop   :", {"matches": n_pos, "nonmatches": n_neg})
print("sample:", sample["label"].value_counts().to_dict())
print("join miss rate firstName side1/2:",
      sample[["firstName1", "firstName2"]].isna().mean().round(3).to_dict(),
      "(high => wrong PKEY or pr1/pr2 mapping)")

# ---- pcs-model transform ----------------------------------------------------------------------
if "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov" not in sys.path:
    sys.path.insert(0, "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov")
from pcs_models.data_processing.distance_metrics import (
    norm_levenshtein_sim, exact_string_match, calculate_cartcos, join_name_parts)

DL1, DL2 = DL_COL + "1", DL_COL + "2"
assert {DL1, DL2} <= set(sample.columns), f"DL columns {DL1}/{DL2} missing - check PARTY/ATTRS/DL_COL"

FEATURE_NAMES = ["f_dob", "f_anum", "f_cob", "f_coc", "f_ssn", "f_fin", "f_dl", "f_cartos", "f_name"]

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    features = np.empty((len(df), len(FEATURE_NAMES)), dtype="float32")
    for i, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
        f_dob  = norm_levenshtein_sim([row["dateOfBirth1"], row["dateOfBirth2"]])
        f_anum = norm_levenshtein_sim([row["aNumber1"], row["aNumber2"]])
        f_cob  = exact_string_match([row["countryOfBirth1"], row["countryOfBirth2"]])
        f_coc  = exact_string_match([row["countryOfCitizenship1"], row["countryOfCitizenship2"]])
        f_ssn  = norm_levenshtein_sim([row["ssn1"], row["ssn2"]])
        f_fin  = norm_levenshtein_sim([row["fin1"], row["fin2"]])
        f_dl   = norm_levenshtein_sim([row[DL1], row[DL2]])
        name_list = [row["firstName1"], row["middleName1"], row["lastName1"],
                     row["firstName2"], row["middleName2"], row["lastName2"]]
        f_cartos = calculate_cartcos(name_list)
        f_name = norm_levenshtein_sim([join_name_parts(row["firstName1"], "", ""),
                                       join_name_parts(row["firstName2"], "", "")])
        features[i] = [f_dob, f_anum, f_cob, f_coc, f_ssn, f_fin, f_dl, f_cartos, f_name]
    return pd.DataFrame(features, columns=FEATURE_NAMES)

Feat = build_all_features(sample)

def both_present(a, b):
    a, b = sample[a].astype("string"), sample[b].astype("string")
    return a.notna() & b.notna() & (a.str.strip() != "") & (b.str.strip() != "")

ssn_ok  = both_present("ssn1", "ssn2")
anum_ok = both_present("aNumber1", "aNumber2")
fin_ok  = both_present("fin1", "fin2")
meta = pd.DataFrame({"label": sample["label"].astype(int).values,
                     "no_strong_id": (~(ssn_ok | anum_ok | fin_ok)).values,
                     "dl_present": both_present(DL1, DL2).values})

SCN = {
    "baseline":             ["f_dob", "f_anum", "f_cob", "f_coc", "f_ssn", "f_fin", "f_cartos", "f_name"],
    "no_ssn":               ["f_dob", "f_anum", "f_cob", "f_coc",          "f_fin", "f_cartos", "f_name"],
    "no_strong_id":         ["f_dob",                   "f_cob", "f_coc",           "f_cartos", "f_name"],
    "with_dl":              ["f_dob", "f_anum", "f_cob", "f_coc", "f_ssn", "f_fin", "f_dl", "f_cartos", "f_name"],
    "with_dl_no_strong_id": ["f_dob",                   "f_cob", "f_coc", "f_dl",            "f_cartos", "f_name"],
    "dl_only_id":           ["f_dob",                   "f_cob", "f_coc", "f_dl",            "f_cartos", "f_name"],
}

print("\nfeatures:", list(Feat.columns))
print("no-strong-id pairs:", int(meta["no_strong_id"].sum()),
      "| DL present:", int(meta["dl_present"].sum()),
      "| no-strong-id WITH dl:", int((meta["no_strong_id"] & meta["dl_present"]).sum()))
print("\nfeature means by label:")
print(pd.concat([Feat, meta["label"]], axis=1).groupby("label").mean())
