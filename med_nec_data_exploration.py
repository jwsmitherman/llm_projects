# Databricks notebook source
# =============================================================================
# Medical Necessity EDA - TripMaster clinical fields
#
# CONFIRMED from Vivekkumar's TripMaster_v2 query (7/17/2026). The clinical data
# we needed lives in prod.silver_transbroker.tripleg:
#
#   ClinicalData       <- the nurse free-text "blurb"   *** the AI input ***
#   LosQuestions       <- clinical questions payload (JSON-ish)
#   LevelOfService     <- requested LOS  (joins c_lookup_contractlos.LOS)
#   LosCategory        <- ServiceType    (joins c_lookup_contractlos.LOS)
#   SpecialNeeds       <- special needs / equipment
#   LOSOverride        <- WAS the level of service overridden by the user?
#   LOSOverrideReason  <- why
#
# LOSOverride is a big deal: the workshop assumed there was no derived-vs-
# overridden flag. There is. It directly measures nurses pushing for a higher
# level of service than the rules produced.
#
# Base tables:
#   prod.silver_transbroker.triprequest tr
#   INNER JOIN prod.silver_transbroker.tripleg tl ON tr.TripRequestId = tl.TripRequestId
#   LEFT JOIN prod.silver_transbroker.c_lookup_contractlos cl ON tl.LevelOfService = cl.LOS
#
# ALREADY MATERIALIZED (fastest path - start here):
#   `prod-sandbox`.vivekkumar_patel.temp_tnet_tripmaster
#   filtered to YEAR(RequestDateTime)=2025 and ContractId IN (29,555,326,229)
#     229            = Texas Health Resources
#     29, 555, 326   = MUSC
#   i.e. exactly the two pilot customers.
#
# ORDER: run Part 1, then 2 (the money query), then 3-5.
# =============================================================================


# COMMAND ----------

# -----------------------------------------------------------------------------
# 0. CONFIG
# -----------------------------------------------------------------------------

# Pre-built TripMaster extract (hyphen in catalog name -> must be backticked).
TRIPMASTER = "`prod-sandbox`.vivekkumar_patel.temp_tnet_tripmaster"

# Raw source, if/when you have prod grants and want to re-derive.
TRIPLEG    = "prod.silver_transbroker.tripleg"
TRIPREQ    = "prod.silver_transbroker.triprequest"
LOS_LOOKUP = "prod.silver_transbroker.c_lookup_contractlos"

# Pilot contracts
CONTRACTS = {229: "Texas Health Resources", 29: "MUSC", 555: "MUSC", 326: "MUSC"}

# Where to write the extract for the LLM pipeline
EXTRACT_TABLE = "`prod-sandbox`.vivekkumar_patel.mednec_llm_extract"

from pyspark.sql import functions as F

spark.sql("USE CATALOG prod")
display(spark.sql("SELECT current_catalog(), current_schema()"))


# COMMAND ----------

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def table_exists(fqtn):
    try:
        spark.sql(f"DESCRIBE TABLE {fqtn}")
        return True
    except Exception:
        return False


def profile_table(fqtn, max_cols=60):
    if not table_exists(fqtn):
        print(f"[skip] {fqtn} not found.")
        return
    df = spark.table(fqtn)
    n = df.count()
    cols = df.columns[:max_cols]
    print(f"{fqtn}: {n:,} rows, {len(df.columns)} columns")
    aggs = []
    for c in cols:
        aggs.append(F.count(F.when(F.col(f"`{c}`").isNull(), 1)).alias(f"{c}||n"))
        aggs.append(F.approx_count_distinct(F.col(f"`{c}`")).alias(f"{c}||d"))
    row = df.agg(*aggs).collect()[0].asDict()
    out = [(c, row[f"{c}||n"], round(100.0 * row[f"{c}||n"] / n, 1) if n else None,
            row[f"{c}||d"]) for c in cols]
    display(spark.createDataFrame(out, ["column", "nulls", "null_pct", "approx_distinct"]))


print("helpers ready")


# COMMAND ----------

# =============================================================================
# PART 1 - Confirm the extract and see the clinical fields
# =============================================================================

print("TripMaster exists:", table_exists(TRIPMASTER))
display(spark.sql(f"DESCRIBE TABLE {TRIPMASTER}"))


# COMMAND ----------

profile_table(TRIPMASTER)


# COMMAND ----------

# Row counts by contract - confirms the MUSC / THR split.
display(spark.sql(f"""
    SELECT ContractId, ContractName, count(*) AS trips
    FROM {TRIPMASTER}
    GROUP BY ContractId, ContractName
    ORDER BY trips DESC
"""))


# COMMAND ----------

# The clinical fields, side by side. THIS is what the LLM will read.
display(spark.sql(f"""
    SELECT
        TripRequestId, TripLegId, ContractName, RequestType,
        LevelOfService, LevelOfServiceDescription, ServiceType,
        LOSOverride, LOSOverrideReason,
        SpecialNeeds,
        ClinicalData,
        LosQuestions
    FROM {TRIPMASTER}
    WHERE ClinicalData IS NOT NULL AND length(trim(ClinicalData)) > 0
    LIMIT 50
"""))


# COMMAND ----------

# How populated is each clinical field? If ClinicalData is mostly null, the
# free-text intervention has a much smaller surface than assumed.
display(spark.sql(f"""
    SELECT
        count(*)                                                          AS total_trips,
        sum(CASE WHEN ClinicalData     IS NOT NULL AND length(trim(ClinicalData))     > 0 THEN 1 ELSE 0 END) AS has_clinical_text,
        sum(CASE WHEN LosQuestions     IS NOT NULL AND length(trim(LosQuestions))     > 0 THEN 1 ELSE 0 END) AS has_los_questions,
        sum(CASE WHEN SpecialNeeds     IS NOT NULL AND length(trim(SpecialNeeds))     > 0 THEN 1 ELSE 0 END) AS has_special_needs,
        sum(CASE WHEN LOSOverrideReason IS NOT NULL AND length(trim(LOSOverrideReason)) > 0 THEN 1 ELSE 0 END) AS has_override_reason,
        round(avg(length(ClinicalData)), 1)                               AS avg_clinical_len,
        max(length(ClinicalData))                                         AS max_clinical_len
    FROM {TRIPMASTER}
"""))


# COMMAND ----------

# =============================================================================
# PART 2 - THE MONEY QUERY
# Quantify insufficient documentation. The workshop claimed "general weakness"
# alone may touch ~20% of trips. This measures it against real data.
#
# Terms that do NOT establish medical necessity without an underlying cause and
# a specific functional deficit.
# =============================================================================

INSUFFICIENT = (
    r"(?i)(general(ized)?\s+weakness|^\s*weakness|fall\s*risk|unsteady\s+gait|"
    r"decondition|generally\s+weak|weak\b|per\s+protocol|convenience|"
    r"unable\s+to\s+arrange|no\s+other\s+transport|needs\s+transport)"
)

display(spark.sql(f"""
    SELECT
        ContractName,
        count(*)                                                   AS trips_with_text,
        sum(CASE WHEN ClinicalData RLIKE '{INSUFFICIENT}' THEN 1 ELSE 0 END) AS insufficient_hits,
        round(100.0 * sum(CASE WHEN ClinicalData RLIKE '{INSUFFICIENT}' THEN 1 ELSE 0 END)
              / count(*), 1)                                        AS pct_insufficient
    FROM {TRIPMASTER}
    WHERE ClinicalData IS NOT NULL AND length(trim(ClinicalData)) > 0
    GROUP BY ContractName
    ORDER BY trips_with_text DESC
"""))


# COMMAND ----------

# Same thing, but split by level of service - expect the BLS-vs-wheelchair
# gray area to concentrate the insufficient documentation.
display(spark.sql(f"""
    SELECT
        LevelOfServiceDescription,
        ServiceType,
        count(*)                                                   AS trips,
        sum(CASE WHEN ClinicalData RLIKE '{INSUFFICIENT}' THEN 1 ELSE 0 END) AS insufficient_hits,
        round(100.0 * sum(CASE WHEN ClinicalData RLIKE '{INSUFFICIENT}' THEN 1 ELSE 0 END)
              / count(*), 1)                                        AS pct_insufficient
    FROM {TRIPMASTER}
    WHERE ClinicalData IS NOT NULL AND length(trim(ClinicalData)) > 0
    GROUP BY LevelOfServiceDescription, ServiceType
    ORDER BY trips DESC
"""))


# COMMAND ----------

# Eyeball the actual offending narratives. These become few-shot examples and
# the golden-set seed for the LLM pipeline.
display(spark.sql(f"""
    SELECT ContractName, LevelOfServiceDescription, ClinicalData
    FROM {TRIPMASTER}
    WHERE ClinicalData RLIKE '{INSUFFICIENT}'
    LIMIT 100
"""))


# COMMAND ----------

# Shortest narratives = likeliest to be non-compliant. A quick quality signal.
display(spark.sql(f"""
    SELECT length(ClinicalData) AS len, ContractName,
           LevelOfServiceDescription, ClinicalData
    FROM {TRIPMASTER}
    WHERE ClinicalData IS NOT NULL AND length(trim(ClinicalData)) > 0
    ORDER BY len ASC
    LIMIT 100
"""))


# COMMAND ----------

# =============================================================================
# PART 3 - LOS OVERRIDE ANALYSIS
# LOSOverride tells us when a user pushed past the system-derived level of
# service. This is the "nurses gaming the system" behavior, measurable.
# =============================================================================

display(spark.sql(f"""
    SELECT
        LOSOverride,
        count(*)                                     AS trips,
        round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
    FROM {TRIPMASTER}
    GROUP BY LOSOverride
    ORDER BY trips DESC
"""))


# COMMAND ----------

# Why are they overriding? Free-text reasons, ranked.
display(spark.sql(f"""
    SELECT LOSOverrideReason, count(*) AS trips
    FROM {TRIPMASTER}
    WHERE LOSOverrideReason IS NOT NULL AND length(trim(LOSOverrideReason)) > 0
    GROUP BY LOSOverrideReason
    ORDER BY trips DESC
    LIMIT 50
"""))


# COMMAND ----------

# Override rate by level of service and contract - where is the pressure?
display(spark.sql(f"""
    SELECT
        ContractName,
        LevelOfServiceDescription,
        count(*)                                                  AS trips,
        sum(CASE WHEN LOSOverride = true OR LOSOverride = 1 THEN 1 ELSE 0 END) AS overrides,
        round(100.0 * sum(CASE WHEN LOSOverride = true OR LOSOverride = 1 THEN 1 ELSE 0 END)
              / count(*), 1)                                       AS pct_override
    FROM {TRIPMASTER}
    GROUP BY ContractName, LevelOfServiceDescription
    ORDER BY trips DESC
"""))


# COMMAND ----------

# =============================================================================
# PART 4 - LEVEL OF SERVICE MIX + the BLS vs WHEELCHAIR question
# The single most denial-prone decision per the med-nec meetings.
# =============================================================================

display(spark.sql(f"""
    SELECT
        LevelOfService, LevelOfServiceDescription, ServiceType, ServiceTypeDescription,
        count(*) AS trips,
        round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
    FROM {TRIPMASTER}
    GROUP BY LevelOfService, LevelOfServiceDescription, ServiceType, ServiceTypeDescription
    ORDER BY trips DESC
"""))


# COMMAND ----------

# Special needs drives equipment/LOS. What is actually being requested?
display(spark.sql(f"""
    SELECT SpecialNeeds, count(*) AS trips
    FROM {TRIPMASTER}
    WHERE SpecialNeeds IS NOT NULL AND length(trim(SpecialNeeds)) > 0
    GROUP BY SpecialNeeds
    ORDER BY trips DESC
    LIMIT 50
"""))


# COMMAND ----------

# =============================================================================
# PART 5 - LosQuestions structure
# The per-customer question payload. Non-standardized across contracts (this is
# the "not indexed or mapped" problem from the meetings). Inspect before parsing.
# =============================================================================

display(spark.sql(f"""
    SELECT ContractName, LosQuestions
    FROM {TRIPMASTER}
    WHERE LosQuestions IS NOT NULL AND length(trim(LosQuestions)) > 0
    LIMIT 25
"""))


# COMMAND ----------

# Size of the payload by contract - confirms whether it is the "giant JSON".
display(spark.sql(f"""
    SELECT ContractName,
           count(*)                     AS trips,
           round(avg(length(LosQuestions)), 0) AS avg_len,
           max(length(LosQuestions))    AS max_len
    FROM {TRIPMASTER}
    WHERE LosQuestions IS NOT NULL
    GROUP BY ContractName
    ORDER BY trips DESC
"""))


# COMMAND ----------

# If LosQuestions is valid JSON, this reveals the top-level keys per contract.
# If it errors or returns nulls, the payload is not plain JSON - inspect above.
# display(spark.sql(f"""
#     SELECT ContractName, get_json_object(LosQuestions, '$') AS parsed
#     FROM {TRIPMASTER}
#     WHERE LosQuestions IS NOT NULL
#     LIMIT 10
# """))


# COMMAND ----------

# =============================================================================
# PART 6 - BUILD THE EXTRACT for the LLM pipeline
# Produces exactly the columns scripts 01-04 expect, so you can swap synthetic
# data for real data. No patient identifiers included.
# =============================================================================

spark.sql(f"""
    CREATE OR REPLACE TABLE {EXTRACT_TABLE} AS
    SELECT
        TripRequestId                        AS order_id,
        TripLegId                            AS trip_leg_id,
        ContractId,
        ContractName                         AS market,
        RequestType,
        LevelOfService                       AS requested_los_code,
        LevelOfServiceDescription            AS requested_los,
        ServiceType,
        LOSOverride                          AS los_overridden,
        LOSOverrideReason                    AS los_override_reason,
        SpecialNeeds                         AS special_needs,
        ClinicalData                         AS free_text,
        LosQuestions                         AS clinical_json
    FROM {TRIPMASTER}
    WHERE ClinicalData IS NOT NULL
      AND length(trim(ClinicalData)) > 0
""")

print(f"Wrote {EXTRACT_TABLE}")
display(spark.sql(f"SELECT count(*) AS rows FROM {EXTRACT_TABLE}"))


# COMMAND ----------

display(spark.sql(f"SELECT * FROM {EXTRACT_TABLE} LIMIT 25"))


# COMMAND ----------

# =============================================================================
# PART 7 (optional) - re-derive from source instead of the prebuilt table.
# Only works with prod grants on silver_transbroker. Minimal version of
# Vivekkumar's TripMaster query - just the medical-necessity columns.
# =============================================================================

# display(spark.sql(f"""
#     SELECT
#         tr.TripRequestId, tr.ContractId, tr.RequestType,
#         tl.LevelOfService, cl.Name AS LevelOfServiceDescription,
#         tl.LosCategory AS ServiceType,
#         tl.LOSOverride, tl.LOSOverrideReason,
#         tl.SpecialNeeds, tl.ClinicalData, tl.LosQuestions
#     FROM {TRIPREQ} tr
#     INNER JOIN {TRIPLEG} tl ON tr.TripRequestId = tl.TripRequestId
#     LEFT JOIN {LOS_LOOKUP} cl
#            ON tl.ContractId = cl.ContractId AND tl.LevelOfService = cl.LOS
#     WHERE YEAR(tr.RequestDateTime) = 2025
#       AND tr.ContractId IN (29, 555, 326, 229)
#     LIMIT 100
# """))
