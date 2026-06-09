# Databricks notebook source
# MAGIC %md
# MAGIC # anm training data upstream evaluation

# COMMAND ----------

tables = [
    "eciscor_prod.pcis_data_science.anm_training_data_matches",
    "eciscor_prod.pcis_data_science.anm_training_data_nonmatches",
    "eciscor_prod.pcis_data_science.anm_training_data_fullset",
    "eciscor_prod.pcis_data_science.anm_training_data_sample",
    "eciscor_prod.pcis_metadata.mv_es_party_identity",
    "eciscor_prod.pcis_metadata.rpt_bronze_party",
    "eciscor_prod.pcis_metadata.ps_parties_to_upsert",
]

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, BooleanType

def profile(name):
    try:
        df = spark.table(name)
        rows = df.count()
        cols = len(df.columns)
    except Exception as e:
        return {"table": name, "exists": False, "format": None, "rows": None,
                "columns": None, "size_bytes": None, "num_files": None,
                "last_modified": None, "location": str(e)[:200]}
    fmt, size, nfiles, lastmod, loc = None, None, None, None, None
    try:
        d = spark.sql(f"describe detail {name}").collect()[0].asDict()
        fmt = d.get("format")
        size = d.get("sizeInBytes")
        nfiles = d.get("numFiles")
        lastmod = str(d.get("lastModified")) if d.get("lastModified") is not None else None
        loc = d.get("location")
    except Exception:
        fmt = "view"
    return {"table": name, "exists": True, "format": fmt, "rows": rows,
            "columns": cols, "size_bytes": size, "num_files": nfiles,
            "last_modified": lastmod, "location": loc}

# COMMAND ----------

schema = StructType([
    StructField("table", StringType()),
    StructField("exists", BooleanType()),
    StructField("format", StringType()),
    StructField("rows", LongType()),
    StructField("columns", IntegerType()),
    StructField("size_bytes", LongType()),
    StructField("num_files", LongType()),
    StructField("last_modified", StringType()),
    StructField("location", StringType()),
])

summary = spark.createDataFrame([profile(t) for t in tables], schema)
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## schema per table

# COMMAND ----------

for t in tables:
    try:
        print(t)
        spark.table(t).printSchema()
    except Exception as e:
        print(t, "unavailable", str(e)[:120])

# COMMAND ----------

# MAGIC %md
# MAGIC ## latest version and freshness

# COMMAND ----------

hist = None
for t in tables:
    try:
        h = (spark.sql(f"describe history {t}")
             .select(f.lit(t).alias("table"), "version", "timestamp", "operation")
             .orderBy(f.col("version").desc())
             .limit(1))
        hist = h if hist is None else hist.unionByName(h)
    except Exception:
        pass

if hist is not None:
    display(hist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## sampled null rate

# COMMAND ----------

for t in tables:
    try:
        df = spark.table(t).limit(100000)
        total = df.count()
        if total == 0:
            print(t, "empty")
            continue
        nulls = df.select([
            (f.count(f.when(f.col(c).isNull(), c)) / f.lit(total)).alias(c)
            for c in df.columns
        ])
        print(t, total)
        display(nulls)
    except Exception as e:
        print(t, "unavailable", str(e)[:120])
