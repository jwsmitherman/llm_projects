# Databricks notebook source
# =============================================================================
# Medical Necessity + Denials + Nurse-Nav - Data Exploration
#
# Discovery-first exploration. The transcripts all stress the same thing: the
# data is NOT indexed or mapped, and the denial data is DIRTY (duplicates, needs
# cleanup). So this notebook gives reusable profiling helpers, then applies them
# to three targets.
#
# WHAT WE ARE TRYING TO ANSWER
#   A. Ground medical necessity (Transport.net / transbroker)
#      - Where do the clinical free-text ("blurb") and the giant JSON live?
#      - Can we derive requested level of service (BLS / wheelchair / ALS / CCT)?
#   B. Integra ground denial data (being loaded into Databricks)
#      - What fields exist? Where are the medical-necessity denial reasons?
#      - How dirty is it (duplicates)? Top denial reasons + BLS -> wheelchair?
#   C. Nurse-navigation 911 triage
#      - Distribution of triage levels 0-6 and self-care / MTARA-6 / override.
#      - Where is the "why" (free-text notes / Cordy protocol codes)?
#      - KPI scaffolding: resolution rate, diversion rate, % referred to nurse-nav.
#
# CONFIRMED prod SCHEMAS (from Vivekkumar's Teams message):
#   silver_transbroker            - ordering / concierge
#   silver_transbrokerdocs        - docs, e.g. tripridesharereceipts
#   silver_transbrokerlogs        - TripLog / audit (LastModifiedDate logic lives here)
#   silver_transportdataservices  - services / API layer
#   silver_resourcetracking       - vehicle / crew tracking
#   silver_patientlookup          - patient demographics  << PHI: do NOT pull into extract
#
# PEOPLE / SOURCES to confirm field definitions
#   Vivekkumar Patel       - knows exactly where transport.net inputs land (take the call)
#   Jeff Pellick (Integra) - denial field definitions + spec file
#   Jen Jones              - CMS criteria + possible denial-code mapping table
#   Nathan Haron           - Logis / call-center terminology (nurse-nav)
#   Rich                   - existing Power BI reports + KPIs (nurse-nav)
#   Adrian / Matt          - data-engineering help with the Integra load
#
# NOTE: names like MTARA-6 / Nimtara / Bingley are phonetic from the
# transcripts - verify exact spelling.
# =============================================================================


# COMMAND ----------

# -----------------------------------------------------------------------------
# 0. CONFIG - edit these as you confirm real names
# -----------------------------------------------------------------------------

# Transport.net - all six confirmed schemas under the prod catalog.
TB_CATALOG = "prod"
TB_SCHEMAS = [
    "silver_transbroker",
    "silver_transbrokerdocs",
    "silver_transbrokerlogs",
    "silver_transportdataservices",
    "silver_resourcetracking",
    "silver_patientlookup",
]

# Schema most likely to hold the clinical free-text + JSON + level of service.
# Used for the SHOW TABLES inventory in Part A. Adjust if discovery points elsewhere.
TB_PRIMARY_SCHEMA = "silver_transbroker"

# Integra denial data (TODO: set once loaded into your playground)
DENIAL_TABLE = "prod_sandbox.vivekkumar_patel.integra_ground_denials"   # TODO confirm

# Nurse-nav triage data (TODO: set once located)
NURSENAV_TABLE = "prod_sandbox.vivekkumar_patel.nurse_nav_calls"        # TODO confirm


# COMMAND ----------

# -----------------------------------------------------------------------------
# 1. REUSABLE PROFILING HELPERS (run once)
# These work on ANY table. Built to be cheap on large/dirty tables: single-pass
# aggregation, approximate distinct counts, capped column width.
# -----------------------------------------------------------------------------

from pyspark.sql import functions as F


def list_tables(catalog, schema):
    """List tables in a schema."""
    display(spark.sql(f"SHOW TABLES IN {catalog}.{schema}"))


def inventory(catalog, schemas):
    """Full table inventory across several schemas (run this before column search)."""
    in_list = ", ".join(f"'{s}'" for s in schemas)
    q = f"""
        SELECT table_schema, table_name
        FROM {catalog}.information_schema.tables
        WHERE table_schema IN ({in_list})
        ORDER BY table_schema, table_name
    """
    display(spark.sql(q))


def find_columns(catalog, schemas, keyword_regex):
    """Search information_schema for columns whose name matches a regex (case-insensitive)."""
    in_list = ", ".join(f"'{s}'" for s in schemas)
    q = f"""
        SELECT table_schema, table_name, column_name, data_type
        FROM {catalog}.information_schema.columns
        WHERE table_schema IN ({in_list})
          AND lower(column_name) RLIKE '{keyword_regex}'
        ORDER BY table_schema, table_name, column_name
    """
    display(spark.sql(q))


def profile_table(fqtn, max_cols=50):
    """Row count + per-column null count and approx distinct count, in a single pass."""
    df = spark.table(fqtn)
    n = df.count()
    cols = df.columns[:max_cols]
    print(f"{fqtn}: {n:,} rows, {len(df.columns)} columns "
          f"({'showing first ' + str(max_cols) if len(df.columns) > max_cols else 'all'})")
    aggs = []
    for c in cols:
        aggs.append(F.count(F.when(F.col(f"`{c}`").isNull(), 1)).alias(f"{c}||nulls"))
        aggs.append(F.approx_count_distinct(F.col(f"`{c}`")).alias(f"{c}||distinct"))
    row = df.agg(*aggs).collect()[0].asDict()
    out = []
    for c in cols:
        nulls = row[f"{c}||nulls"]
        out.append((c, nulls, round(100.0 * nulls / n, 1) if n else None, row[f"{c}||distinct"]))
    prof = spark.createDataFrame(out, ["column", "null_count", "null_pct", "approx_distinct"])
    display(prof)


def duplicate_check(fqtn, key_cols):
    """How many key combinations appear more than once (dirty-data check)."""
    df = spark.table(fqtn)
    dups = df.groupBy(*key_cols).count().filter("count > 1")
    d = dups.count()
    total = df.count()
    print(f"{fqtn}: {d:,} duplicated key groups on {key_cols} (table has {total:,} rows)")
    if d:
        display(dups.orderBy(F.desc("count")).limit(25))


def value_counts(fqtn, col, top=25, where=None):
    """Top values for a column."""
    df = spark.table(fqtn)
    if where:
        df = df.filter(where)
    display(df.groupBy(col).count().orderBy(F.desc("count")).limit(top))


def sample_free_text(fqtn, text_col, n=20, where=None):
    """Sample non-empty free-text values to eyeball the narrative content."""
    df = spark.table(fqtn)
    if where:
        df = df.filter(where)
    df = df.filter(F.col(text_col).isNotNull() & (F.length(F.trim(F.col(text_col))) > 0))
    display(df.select(text_col).limit(n))


print("Helpers ready: list_tables, inventory, find_columns, profile_table, "
      "duplicate_check, value_counts, sample_free_text")


# COMMAND ----------

# =============================================================================
# PART A - Transport.net clinical data (transbroker)
# Find the clinical free-text + JSON + level of service that the AI will read.
# =============================================================================

# 1. Full table inventory across all six schemas - do this FIRST to see the
#    landscape before drilling into columns.
inventory(TB_CATALOG, TB_SCHEMAS)


# COMMAND ----------

# 2. THE KEY SEARCH across all six schemas: columns that likely hold clinical
#    text, the giant JSON, level of service, or market/area.
find_columns(
    TB_CATALOG, TB_SCHEMAS,
    "(pcs|clinic|medical|necess|narrativ|blurb|freetext|free_text|comment|note|reason|"
    "diagnos|oxygen|ambulan|wheelchair|stretcher|los|levelofservice|servicetype|special|"
    "position|mobility|json|survey|question|emergen)"
)


# COMMAND ----------

# 3. Optional: list tables in a single schema you want to eyeball
#    (e.g. logs or transportdataservices, likely homes for the clinical payload).
# list_tables(TB_CATALOG, "silver_transbrokerlogs")
# list_tables(TB_CATALOG, "silver_transportdataservices")


# COMMAND ----------

# 4. Once you identify the order/clinical table from step 2, profile it (replace name):
# profile_table("prod.silver_transbroker.o_concierge_trips")                          # TODO
# sample_free_text("prod.silver_transbroker.o_concierge_trips", "ClinicalNarrative")  # TODO col
# value_counts("prod.silver_transbroker.o_concierge_trips", "RequestedServiceType")   # TODO col


# COMMAND ----------

# =============================================================================
# PART B - Integra ground denial data
# Goal: consolidate, de-dupe, and find the medical-necessity denial reasons +
# the BLS -> wheelchair pattern. Confirm field definitions with Jeff Pellick
# (Integra); mapping table with Jen Jones.
# =============================================================================

# 1. Profile whatever landed (set DENIAL_TABLE in the Config cell first).
profile_table(DENIAL_TABLE)


# COMMAND ----------

# 2. Find the denial-reason and level-of-service fields by name.
find_columns(
    DENIAL_TABLE.split(".")[0],                              # catalog part of DENIAL_TABLE
    [DENIAL_TABLE.split(".")[1]],                            # schema part
    "(deny|denial|reason|remark|carc|rarc|adjust|medical|necess|los|level|hcpcs|cpt|"
    "modifier|gy|claim|payer|payor|status)"
)


# COMMAND ----------

# 3. Dirty-data check: duplicates on the claim key (replace with the real key column(s)).
# duplicate_check(DENIAL_TABLE, ["ClaimId"])                  # TODO real key


# COMMAND ----------

# 4. Top denial reasons overall (replace with the real reason column).
# value_counts(DENIAL_TABLE, "DenialReason", top=40)          # TODO real column


# COMMAND ----------

# 5. BLS -> wheelchair pattern: denial reasons broken out by level of service / HCPCS.
#    Ambulance HCPCS reference: A0428 = BLS non-emergency, A0426 = ALS non-emergency,
#    A0425 = mileage; wheelchair van is often A0130 / T2001-T2005. GY modifier = not med-necessary.
# display(
#     spark.table(DENIAL_TABLE)
#          .groupBy("Hcpcs", "DenialReason")                  # TODO real columns
#          .count().orderBy(F.desc("count")).limit(50)
# )


# COMMAND ----------

# =============================================================================
# PART C - Nurse-navigation 911 triage
# Triage levels 0-6; buckets self-care / MTARA-6 / override; the "why" is in
# free-text + Cordy codes. Confirm terminology with Nathan Haron, KPIs with Rich.
# =============================================================================

# 1. Profile the nurse-nav table (set NURSENAV_TABLE in Config first).
profile_table(NURSENAV_TABLE)


# COMMAND ----------

# 2. Find the triage-level, bucket/disposition, notes, and workset-id fields.
find_columns(
    NURSENAV_TABLE.split(".")[0],
    [NURSENAV_TABLE.split(".")[1]],
    "(triage|level|acuity|bucket|disposition|override|selfcare|self_care|mtara|nimtara|"
    "cordy|protocol|note|comment|reason|workset|result|referral|outcome)"
)


# COMMAND ----------

# 3. Distribution of triage levels and buckets (replace with real column names).
# value_counts(NURSENAV_TABLE, "TriageLevel")                 # TODO real column
# value_counts(NURSENAV_TABLE, "DispositionBucket")           # TODO real column (self-care/MTARA-6/override)
# sample_free_text(NURSENAV_TABLE, "NurseNote", n=30)         # TODO real column


# COMMAND ----------

# 4. KPI scaffolding (fill column names)
#    Resolution rate         = share of calls resolved without ambulance/ED
#    Diversion rate          = share diverted away from ambulance/ED
#    % referred to nurse-nav  = referrals / total calls
#
# df = spark.table(NURSENAV_TABLE)
# total = df.count()
# diverted = df.filter("DispositionBucket in ('self-care','telemedicine','urgent_care')").count()  # TODO
# print(f"Diversion rate: {diverted/total:.1%}  (n={total:,})")


# COMMAND ----------

# =============================================================================
# PART D (optional) - AI categorization of free-text reasons
# Both the denial and nurse-nav work call for using AI to categorize free-text
# into reasons. This takes a small SAMPLE, sends it to the LLM, and buckets it
# into a fixed taxonomy so you can quantify "why". Keep the sample small (cost).
#
# Run this once in its own cell first (this is a notebook magic, run it directly
# in a cell - do NOT paste it as a comment):
#   %pip install -q openai
#   dbutils.library.restartPython()
# =============================================================================

import json
from openai import OpenAI

OPENAI_API_KEY = "sk-REPLACE_ME"       # TODO your key; move to a secret scope after testing
LLM = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_API_KEY)

# Fixed taxonomy for MEDICAL-NECESSITY DENIALS (edit for nurse-nav use).
TAXONOMY = [
    "insufficient_clinical_documentation",   # e.g. 'general weakness' with no cause
    "wrong_level_of_service_bls_vs_wheelchair",
    "wrong_level_of_service_other",
    "missing_or_invalid_pcs_signature",
    "demographic_or_eligibility_error",
    "prior_authorization_missing",
    "not_medically_necessary_other",
    "other_or_unclear",
]

SYS = ("You classify short ambulance-claim denial notes into exactly one category from the provided "
       "list. Respond ONLY as JSON: {\"category\": <one label>, \"rationale\": <=1 sentence}.")


def categorize(text):
    user = f"Categories: {TAXONOMY}\nDenial note: \"{text}\"\nClassify."
    r = client.chat.completions.create(
        model=LLM, temperature=0.0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
    )
    return json.loads(r.choices[0].message.content)


# Example wiring (uncomment once you have the reason column):
# rows = (spark.table(DENIAL_TABLE)
#             .select(F.col("DenialReason").alias("t"))            # TODO real column
#             .filter("t is not null").limit(50).collect())
# results = [{"note": x["t"], **categorize(x["t"])} for x in rows]
# display(spark.createDataFrame(results))

print("Categorizer ready. Uncomment the wiring once the reason column is confirmed.")
