# IDEN-43831 ANM spike: build labeled, pcs-model-transformed sample for new-model (no-ID / DL / other-id) exploration
# Source of truth = the real ANM training tables. Label comes from which table a pair is in
# (matches=1, nonmatches=0), then balanced-sample, then transform with pcs_models distance metrics.

from pyspark.sql import functions as F
import pandas as pd
import numpy as np

CATALOG     = "eciscor_prod"
DS          = f"{CATALOG}.pcis_data_science"
MATCHES     = f"{DS}.anm_training_data_matches"
NONMATCHES  = f"{DS}.anm_training_data_nonmatches"
FULLSET     = f"{DS}.anm_training_data_fullset"
OUT_TABLE   = f"{DS}.anm_training_data_sample_featurized"   # new table; existing anm_training_data_sample left alone

SAMPLE_PER_CLASS = 50000
SEED             = 43831
DISCOVER         = True     # first run: inspect real schemas, then set L/R below
SAVE             = False    # flip on once features look right

m  = spark.table(MATCHES)
nm = spark.table(NONMATCHES)

if DISCOVER:
    print("matches    rows:", m.count(),  "cols:", len(m.columns)); m.printSchema()
    print("nonmatches rows:", nm.count(), "cols:", len(nm.columns)); nm.printSchema()

# label + union on shared columns
common  = [c for c in m.columns if c in nm.columns]
labeled = (m.select(*common).withColumn("label", F.lit(1))
            .unionByName(nm.select(*common).withColumn("label", F.lit(0))))

# balanced sample by label
n_pos = labeled.filter("label = 1").count()
n_neg = labeled.filter("label = 0").count()
fractions = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)),
             0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}
sample = labeled.sampleBy("label", fractions=fractions, seed=SEED).toPandas()
print("pop  :", {"matches": n_pos, "nonmatches": n_neg})
print("sample:", sample["label"].value_counts().to_dict())

# transform with pcs-model
# >>> EDIT after DISCOVER: map left (record A) and right (record B) columns from the real schema.
L = dict(first="first_name_1", last="surname_1", dob="dob_1",
         ssn="ssn_1", anum="a_number_1", dl="dl_number_1", other="other_id_1")
R = dict(first="first_name_2", last="surname_2", dob="dob_2",
         ssn="ssn_2", anum="a_number_2", dl="dl_number_2", other="other_id_2")

from pcs_models.data_processing.distance_metrics import id_distance, name_token_cosine_td, date_rule_score

def full_name(row, side):
    f, l = row.get(side["first"]), row.get(side["last"])
    return f"{'' if pd.isna(f) else f} {'' if pd.isna(l) else l}".strip()

def feats(row):
    return pd.Series({
        "name_cos":   name_token_cosine_td(full_name(row, L), full_name(row, R)),
        "date_score": date_rule_score(row[L["dob"]],  row[R["dob"]]),
        "ssn_d":      id_distance(row[L["ssn"]],   row[R["ssn"]]),
        "anum_d":     id_distance(row[L["anum"]],  row[R["anum"]]),
        "dl_d":       id_distance(row[L["dl"]],    row[R["dl"]]),
        "other_d":    id_distance(row[L["other"]], row[R["other"]]),
    })

X = sample.apply(feats, axis=1)

# ticket coverage flags: no shared strong ID (must match on name/dob/DL/other); DL present both sides; other present both sides
no_strong_id = ((sample[L["ssn"]].isna()  | sample[R["ssn"]].isna()) &
                (sample[L["anum"]].isna() | sample[R["anum"]].isna()))
dl_both      = sample[L["dl"]].notna()    & sample[R["dl"]].notna()
other_both   = sample[L["other"]].notna() & sample[R["other"]].notna()

data = pd.concat([sample[["label"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
data["no_strong_id"] = no_strong_id.values
data["dl_both"]      = dl_both.values
data["other_both"]   = other_both.values

print("feature means by label:")
print(data.groupby("label")[["name_cos","date_score","ssn_d","anum_d","dl_d","other_d"]].mean())
print("no-strong-id pairs:", int(data["no_strong_id"].sum()),
      "| DL on both sides:", int(data["dl_both"].sum()),
      "| other-id on both sides:", int(data["other_both"].sum()))
print("no-strong-id label balance:", data[data["no_strong_id"]]["label"].value_counts().to_dict())

if SAVE:
    out = spark.createDataFrame(pd.concat([sample.reset_index(drop=True), data.drop(columns=["label"])], axis=1))
    out.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT_TABLE)
    print("saved:", OUT_TABLE, out.count(), "rows")

data.head()
