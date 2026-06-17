# IDEN-43831 ANM spike (data + features): labeled sample from real matches/nonmatches,
# DL sourced by join (not present in the pair tables) and transformed as a first-class id feature.
# label = source table (matches=1, nonmatches=0) -> enrich DL -> balanced sample -> feature superset.

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

# ---- DL sourcing -----------------------------------------------------------------------------
# DL is NOT in the matches/nonmatches pair tables, so join it from an identity table keyed by the
# party id on each side. Set these 5 names after DISCOVER (check the schema prints + DL source).
ENRICH_DL = True
DL_SRC    = f"{MD}.mv_es_party_identity"   # >>> table holding driver's license per party
DL_KEY    = "partyId"                      # >>> party key column in DL_SRC
DL_VAL    = "driversLicenseNumber"         # >>> DL column in DL_SRC
PKEY1     = "partyId1"                     # >>> side-A party key in the pair tables
PKEY2     = "partyId2"                     # >>> side-B party key in the pair tables
DL1, DL2  = "dlNumber1", "dlNumber2"       # output column names after the join
# ----------------------------------------------------------------------------------------------

m  = spark.table(MATCHES)
nm = spark.table(NONMATCHES)
if DISCOVER:
    print("matches    rows:", m.count(),  "cols:", len(m.columns)); m.printSchema()
    print("nonmatches rows:", nm.count(), "cols:", len(nm.columns)); nm.printSchema()
    print("pair columns:", sorted(m.columns))
    print("DL source columns:", sorted(spark.table(DL_SRC).columns))

common  = [c for c in m.columns if c in nm.columns]
labeled = (m.select(*common).withColumn("label", F.lit(1))
            .unionByName(nm.select(*common).withColumn("label", F.lit(0))))

if ENRICH_DL:
    dl_ref = (spark.table(DL_SRC)
              .select(F.col(DL_KEY).alias("k"), F.col(DL_VAL).alias("v"))
              .dropDuplicates(["k"]))
    labeled = (labeled
        .join(dl_ref.select(F.col("k").alias(PKEY1), F.col("v").alias(DL1)), on=PKEY1, how="left")
        .join(dl_ref.select(F.col("k").alias(PKEY2), F.col("v").alias(DL2)), on=PKEY2, how="left"))
    print("after DL join, cols:", DL1, DL2, "present:", DL1 in labeled.columns and DL2 in labeled.columns)

n_pos = labeled.filter("label = 1").count()
n_neg = labeled.filter("label = 0").count()
fractions = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)),
             0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}
sample = labeled.sampleBy("label", fractions=fractions, seed=SEED).toPandas()
print("pop   :", {"matches": n_pos, "nonmatches": n_neg})
print("sample:", sample["label"].value_counts().to_dict())

# pcs-model transform ---------------------------------------------------------------------------
if "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov" not in sys.path:
    sys.path.insert(0, "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov")
from pcs_models.data_processing.distance_metrics import (
    norm_levenshtein_sim, exact_string_match, calculate_cartcos, join_name_parts)

assert {DL1, DL2} <= set(sample.columns), \
    f"DL columns {DL1}/{DL2} missing - fix the ENRICH_DL join keys/source above"

# feature superset; DL is a first-class id feature alongside ssn/anum/fin (fixed order)
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
        fname1_clean = join_name_parts(row["firstName1"], "", "")
        fname2_clean = join_name_parts(row["firstName2"], "", "")
        f_name = norm_levenshtein_sim([fname1_clean, fname2_clean])
        features[i] = [f_dob, f_anum, f_cob, f_coc, f_ssn, f_fin, f_dl, f_cartos, f_name]
    return pd.DataFrame(features, columns=FEATURE_NAMES)

Feat = build_all_features(sample)

# label + coverage flags
def both_present(a, b):
    a, b = sample[a].astype("string"), sample[b].astype("string")
    return a.notna() & b.notna() & (a.str.strip() != "") & (b.str.strip() != "")

ssn_ok  = both_present("ssn1", "ssn2")
anum_ok = both_present("aNumber1", "aNumber2")
fin_ok  = both_present("fin1", "fin2")
meta = pd.DataFrame({"label": sample["label"].astype(int).values,
                     "no_strong_id": (~(ssn_ok | anum_ok | fin_ok)).values,
                     "dl_present": both_present(DL1, DL2).values})

# scenario matrix (each = subset of Feat columns), covering the ticket conditions
SCN = {
    "baseline":             ["f_dob", "f_anum", "f_cob", "f_coc", "f_ssn", "f_fin", "f_cartos", "f_name"],
    "no_ssn":               ["f_dob", "f_anum", "f_cob", "f_coc",          "f_fin", "f_cartos", "f_name"],
    "no_strong_id":         ["f_dob",                   "f_cob", "f_coc",           "f_cartos", "f_name"],
    "with_dl":              ["f_dob", "f_anum", "f_cob", "f_coc", "f_ssn", "f_fin", "f_dl", "f_cartos", "f_name"],
    "with_dl_no_strong_id": ["f_dob",                   "f_cob", "f_coc", "f_dl",            "f_cartos", "f_name"],
    "dl_only_id":           ["f_dob",                   "f_cob", "f_coc", "f_dl",            "f_cartos", "f_name"],
}

print("\nfeatures:", list(Feat.columns))
print("scenarios:", {k: v for k, v in SCN.items()})
print("no-strong-id pairs:", int(meta["no_strong_id"].sum()),
      "| DL present:", int(meta["dl_present"].sum()),
      "| no-strong-id WITH dl:", int((meta["no_strong_id"] & meta["dl_present"]).sum()))
print("\nfeature means by label:")
print(pd.concat([Feat, meta["label"]], axis=1).groupby("label").mean())
