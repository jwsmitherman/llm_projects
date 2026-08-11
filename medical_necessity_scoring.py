# Medical Necessity Scoring - revised schema

# Reframes the non-emergent ground analysis around **medical necessity and transport
# appropriateness** rather than payment or denial. Per the 24 July review:

# - Two scoring axes, not one label: a **mobility** axis (why other transport is
#   contraindicated) and a **monitoring** axis (why this level of service).
# - A single combined score drives the buckets by cutoff: 0 = not_necessary; >= 3 with a
#   named concept = necessary; in between = indeterminate.
# - **Weighted concepts** and a **confidence** score, both driven from one config block.
# - No payment-focused terminology in any output column.

# Read-only throughout. No table writes.

# CMS source documents are defined in a separate module, cms_references.py, keyed by the CMS
# source they come from (BPM10, BPM10_10_2_3, CFR_414_605, RSN, MLN, ...). The notebook imports it
# so citations live in one place. In Databricks, keep both files in the same Workspace folder with
# Files enabled; if import is unavailable, run `%run ./cms_references` in a cell first.
from cms_references import ref_url

# 0. Environment

# Written for Databricks (`spark` session assumed present). The `display()` helper below
# lets the same notebook run in a plain Jupyter/VS Code kernel, where it falls back to a
# pandas-style preview. Delete this cell if running only in Databricks.

try:
    display  # provided by Databricks
except NameError:
    def display(sdf):
        try:
            print(sdf.limit(50).toPandas().to_string(index=False))
        except AttributeError:
            print(sdf)

# 1. Source fields

# The source table is a read-only snapshot. Each field below is cited to the CMS document
# that makes it relevant to the medical necessity determination.

# The catalog name contains a hyphen, so each identifier part must be back-quoted for
# Spark SQL. SOURCE_TABLE keeps the plain name; SOURCE_TABLE_SQL is the quoted form used
# when reading. (Unquoted `prod-sandbox` raises INVALID_IDENTIFIER / SQLSTATE 42602.)
SOURCE_TABLE     = "prod-sandbox.vivekkumar_patel.temp_tnet_tripmaster"
SOURCE_TABLE_SQL = ".".join(f"`{p}`" for p in SOURCE_TABLE.split("."))

# Logical fields mapped to candidate column names. The first candidate present in the table
# wins. This avoids INVALID_IDENTIFIER / UNRESOLVED_COLUMN when the real schema differs.
# Edit the candidate lists if the actual column names are not covered.
FIELD_CANDIDATES = {
    "free_text": ["ClinicalData", "clinical_data", "ClinicalNotes", "Reason", "ReasonForTransport"],
    "los":       ["LevelOfService", "level_of_service", "LOS", "ServiceLevel"],
    "customer":  ["Customer", "CustomerName", "HealthSystem", "Facility", "Account"],
    "order_id":  ["OrderId", "order_id", "OrderID", "TripId", "TripID", "TripLegId"],
}

# 42 CFR / BPM10 citations for the logical fields (see resolution below):
#   free_text -> [BPM10] 10.2.1, 10.2.4 ; [RSN] AM600   (reason must be recorded)
#   los       -> [414.605] ; [410.40](c)                (level billed must be necessary)
#   customer / order_id -> operational only, no CMS basis

# 2. Concept dictionary - the single source of truth

# Every concept carries its axis, weight, group, CMS basis, source tag, source URL, and the
# term pattern matched against ClinicalData. The url column makes the citation explicit next to
# the terms it governs. Weights are 0-3, reflecting how directly the concept establishes the
# criterion (not clinical severity).

#   axis: "mobility"   = transport necessity   [BPM10] 10.2.1 / 10.2.3  (why not a wheelchair van/car)
#         "monitoring" = level of service      [414.605]               (why ALS / CCT rather than BLS)
#         "filler"     = no necessity alone     [MLN] / [BPM10] 10.2.1
#   basis: "named"    = concept is explicit in the cited CMS text
#          "inferred" = derived from the [BPM10] 10.2.1 general test, not separately named

# Concept source URLs, resolved from the reference module (keyed by CMS source).
BPM10   = ref_url("BPM10")
CFR414  = ref_url("CFR_414_605")
MLN     = ref_url("MLN")

# fields: (concept, axis, weight, group, basis, cms_ref, source_url, term_pattern)
CONCEPTS = [
    # --- mobility axis: transport necessity, BPM10 10.2.3 bed-confined test (prongs) + 10.2.1 ---
    # bed_confined: [BPM10] 10.2.3 - "unable to get up from bed without assistance". One prong.
    ("bed_confined",     "mobility",     3, "A", "named",    "BPM10 10.2.3 (prong: get up)",       BPM10,
        r"bed ?(bound|confined)|unable to get up|cannot get out of bed"),
    # mobility_deficit: [BPM10] 10.2.3 - "unable to ambulate" (prong 2); also 10.2.1 contraindication.
    ("mobility_deficit", "mobility",     3, "A", "named",    "BPM10 10.2.3 (prong: ambulate)",     BPM10,
        r"hemipar|hemipleg|paraly|non ?ambulat|unable to (bear weight|ambulate|walk|stand)|fracture|amputat|contracture|bear weight"),
    # cannot_sit: [BPM10] 10.2.3 - "unable to sit in a chair or wheelchair" (prong 3).
    ("cannot_sit",       "mobility",     3, "A", "named",    "BPM10 10.2.3 (prong: sit)",          BPM10,
        r"cannot sit|unable to sit|special positioning|supine|must lie|stretcher|cannot support trunk"),
    # bariatric: not named by CMS; inferred handling requirement under [BPM10] 10.2.1 general test.
    ("bariatric",        "mobility",     2, "A", "inferred", "BPM10 10.2.1 (inferred: handling)",  BPM10,
        r"bariatric|morbid"),
    # wound_ostomy: not named; inferred positioning need under [BPM10] 10.2.1.
    ("wound_ostomy",     "mobility",     1, "A", "inferred", "BPM10 10.2.1 (inferred: position)",  BPM10,
        r"wound|ostomy|ulcer|decubitus|drain"),
    # behavioral: not named; inferred safety need under [BPM10] 10.2.1. Weakest - CMS wants a
    #   physical limitation, so this is a candidate to move to Group B on SME review.
    ("behavioral",       "mobility",     1, "A", "inferred", "BPM10 10.2.1 (inferred: safety)",    BPM10,
        r"dementia|alzheimer|combative|agitat|altered mental|flight risk|elope"),

    # --- monitoring axis: level of service, 42 CFR 414.605 definitions ---
    # ventilator: [414.605] ALS2 (airway management) / SCT (respiratory care) territory.
    ("ventilator",       "monitoring",   3, "A", "named",    "414.605 (ALS2 airway / SCT)",        CFR414,
        r"ventilat|vent|trach|intubat"),
    # suctioning: [414.605] SCT - care beyond the EMT-Paramedic scope of practice.
    ("suctioning",       "monitoring",   3, "A", "named",    "414.605 (SCT)",                      CFR414,
        r"suction"),
    # iv_medication: [414.605] ALS2 - administration of 3+ IV medications / central line.
    ("iv_medication",    "monitoring",   3, "A", "named",    "414.605 (ALS2 IV meds)",             CFR414,
        r"\biv\b|infusion|drip|heparin|antibiotic|tpn"),
    # cardiac: [414.605] ALS assessment / ALS2 cardiac procedures (monitoring, defib, pacing).
    ("cardiac",          "monitoring",   2, "A", "named",    "414.605 (ALS assessment)",           CFR414,
        r"cardiac|telemetry|ekg|ecg|nstemi|stemi|arrhythm|afib"),
    # oxygen: not a named necessity concept; [BPM10] 10.2.1 - chronic O2 alone does not qualify.
    ("oxygen",           "monitoring",   1, "A", "inferred", "BPM10 10.2.1 (O2 alone insufficient)", BPM10,
        r"oxygen|\bo2\b|lpm|nasal cannula|bipap|cpap"),
    # isolation: not named; inferred under [BPM10] 10.2.1 - infection control, not a patient contraindication.
    ("isolation",        "monitoring",   1, "A", "inferred", "BPM10 10.2.1 (infection control)",   BPM10,
        r"isolation|mrsa|c\.? ?diff|precaution"),

    # --- filler: appears in text but establishes no necessity on its own ---
    # weakness_only: [MLN] / MAC guidance - "severe generalized weakness" counts only with a
    #   qualifier and functional consequence; the bare term is vague and of little value on review.
    ("weakness_only",    "filler",       0, "B", "named",    "MLN / MAC (vague weakness)",         MLN,
        r"general(iz)?e?d? weakness|generally weak|^\s*weak"),
    # fall_risk_only: [MLN] / MAC guidance - a risk is not a contraindication to other transport.
    ("fall_risk_only",   "filler",       0, "B", "named",    "MLN / MAC (risk != contra)",         MLN,
        r"fall risk|unsteady|deconditio"),
    # nonclinical: [BPM10] 10.2.1 / [410.40](d) - a physician order alone does not prove necessity.
    ("nonclinical",      "filler",       0, "B", "named",    "BPM10 10.2.1 (order != necessity)",  BPM10,
        r"per protocol|convenience|no other transport|unable to arrange|family request"),
]

# Thresholds - tune here. A single named, weight-3 concept (score 3) is a clear reason.
NECESSARY_CUTOFF  = 3   # total_score at/above this, with a named concept, = necessary

# 3. Scope - non-emergent ground only

# Rideshare, air, and emergent trips are excluded; they do not carry the [BPM10] 10.2.1
# non-emergency documentation requirement. Adjust the filter to the real column values.

from pyspark.sql import functions as F

df = spark.table(SOURCE_TABLE_SQL)

# Resolve each logical field to a real column. First candidate present wins; unresolved fields
# come back as None and any output depending on them is skipped rather than erroring.
_cols_lower = {c.lower(): c for c in df.columns}
def resolve(field):
    for cand in FIELD_CANDIDATES[field]:
        if cand.lower() in _cols_lower:
            return _cols_lower[cand.lower()]
    return None

FREE_TEXT_COL = resolve("free_text")
LOS_COL       = resolve("los")
CUSTOMER_COL  = resolve("customer")
ORDER_ID_COL  = resolve("order_id")

print("Resolved columns:")
print(f"  free_text -> {FREE_TEXT_COL}")
print(f"  los       -> {LOS_COL}")
print(f"  customer  -> {CUSTOMER_COL}")
print(f"  order_id  -> {ORDER_ID_COL}")
if FREE_TEXT_COL is None:
    raise ValueError(
        "No free-text column found. Add the real column name to "
        "FIELD_CANDIDATES['free_text']. Available columns: " + ", ".join(df.columns)
    )

# Scope: non-emergent GROUND AMBULANCE only. Exclusions are explicit and configurable below,
# because the real level-of-service field uses short codes (BLS, ALS, WC, EMG, FWQUOTE, ...) that
# a generic pattern does not reliably catch. The distinct-value print confirms them against data.

raw_count = df.count()

# Show what level-of-service values actually exist, so scope is verified rather than assumed.
if LOS_COL is not None:
    print("Distinct " + LOS_COL + " values (top 40 by count):")
    df.groupBy(LOS_COL).count().orderBy(F.desc("count")).show(40, truncate=False)

# --- Exclusion 1: air / non-ground. Not subject to the ground ambulance necessity test. ---
NON_GROUND_LOS = ["FWQUOTE", "ORGAN"]          # fixed-wing quote, organ transport
NON_GROUND_PATTERN = r"air|fixed ?wing|rotor|helicopter|flight|rideshare|uber|lyft|livery|taxi"

# --- Exclusion 2: emergent. Outside the NON-emergent scope ([BPM10] 10.2.1). ---
# Catches standalone EMG and the -EMG suffix (ALS-EMG, CCT-EMG), plus spelled-out emergent.
EMERGENT_LOS_PATTERN = r"(^|[-_ ])emg($|[-_ ])|emergen"

# --- Exclusion 3: non-ambulance ground. A wheelchair van or ambulatory transport is the cheaper
# alternative, so ambulance necessity concepts do not apply and produce all-indeterminate noise.
# Excluded by default. Set INCLUDE_NON_AMBULANCE = True to keep them in scope. ---
NON_AMBULANCE_LOS = ["WC", "AMBLTY"]           # wheelchair van, ambulatory
INCLUDE_NON_AMBULANCE = False

# --- Unidentified codes: kept in scope and flagged. Identify with the business before treating
# the headline as final - e.g. ST (large, all-indeterminate) and CC_CF_C5. ---
REVIEW_LOS = ["ST", "CC_CF_C5"]

if LOS_COL is not None:
    los_l = F.lower(F.coalesce(F.col(LOS_COL), F.lit("")))
    exclude_exact = [c.lower() for c in NON_GROUND_LOS]
    if not INCLUDE_NON_AMBULANCE:
        exclude_exact += [c.lower() for c in NON_AMBULANCE_LOS]
    df = df.filter(~los_l.isin(exclude_exact))
    df = df.filter(~los_l.rlike(NON_GROUND_PATTERN))
    df = df.filter(~los_l.rlike(EMERGENT_LOS_PATTERN))
    # Flag any review codes still present, so they are not silently trusted.
    present_review = [c for c in REVIEW_LOS
                      if df.filter(los_l == c.lower()).limit(1).count() > 0]
    if present_review:
        print("REVIEW: unidentified level-of-service codes still in scope - "
              "identify before finalising: " + ", ".join(present_review))
else:
    print("WARNING: no level-of-service column resolved - scope exclusions NOT applied.")

# Also remove emergent trips via a trip-type / priority flag if one is present.
EMERGENT_PATTERN = r"emergen|911|lights|code ?3"
for trip_type_col in ("TripType", "trip_type", "TransportType", "Priority", "CallType"):
    if trip_type_col in df.columns:
        df = df.filter(~F.lower(F.coalesce(F.col(trip_type_col), F.lit(""))).rlike(EMERGENT_PATTERN))
        print(f"Applied emergent exclusion on column: {trip_type_col}")
        break
else:
    print("NOTE: no trip-type column found - emergent exclusion relies on the LOS codes above.")

df = df.withColumn("_text", F.lower(F.coalesce(F.col(FREE_TEXT_COL), F.lit(""))))
df = df.withColumn("_has_text", F.length(F.trim(F.col("_text"))) > 0)

scope_count = df.count()
has_text_count = df.filter(F.col("_has_text")).count()
print(f"\nRaw rows:            {raw_count:,}")
print(f"In scope:            {scope_count:,}  ({scope_count/raw_count*100:.1f}% of raw)")
print(f"  with free text:    {has_text_count:,}  ({has_text_count/max(scope_count,1)*100:.1f}% of scope)")
print("Excluded: air/non-ground (" + ", ".join(NON_GROUND_LOS) + " + patterns), emergent (EMG), "
      + ("non-ambulance (" + ", ".join(NON_AMBULANCE_LOS) + ")" if not INCLUDE_NON_AMBULANCE else "non-ambulance KEPT"))

# Guardrail - catch a scope that collapses to too few rows.
if scope_count < 0.2 * raw_count:
    print("\nWARNING: scope kept under 20% of rows. Check the distinct level-of-service values "
          "above - an exclusion may be dropping valid ground rows, or the wrong column resolved.")
if has_text_count < 0.5 * max(scope_count, 1):
    print("\nWARNING: over half of in-scope orders have no free text. Confirm FREE_TEXT_COL "
          "resolved to the real reason-for-transport field (see the 'Resolved columns' output).")
    print("Sample of the resolved free-text field:")
    df.select(FREE_TEXT_COL).filter(F.col("_has_text")).show(5, truncate=80)

# 4. Concept tagging

# One boolean column per concept, the term pattern matched against ClinicalData with `rlike`.
# The CMS citation for each pattern is in the CONCEPTS table above.

for name, axis, weight, group, basis, ref, url, pattern in CONCEPTS:
    df = df.withColumn(f"c_{name}", F.col("_text").rlike(pattern) & F.col("_has_text"))

# 5. Scores, classification, confidence, and level-of-service recommendation

# All derived from the config. The bucket is driven by one combined score plus one named flag.

def axis_score(axis):
    terms = [F.when(F.col(f"c_{n}"), F.lit(w)).otherwise(F.lit(0))
             for n, a, w, *_ in CONCEPTS if a == axis]
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr

# Component scores (shown so the total is fully traceable). mobility + monitoring = total_score
# exactly - there is no bonus or adjustment.
df = df.withColumn("mobility_score",   axis_score("mobility"))    # [BPM10] 10.2.1 / 10.2.3
df = df.withColumn("monitoring_score", axis_score("monitoring"))  # [414.605]
df = df.withColumn("total_score", F.col("mobility_score") + F.col("monitoring_score"))

# named_score: points from NAMED CMS concepts only (those explicit in CMS text). Shown because a
# named concept is required to reach the necessary band - this column is what drives that.
named_a = [n for n, a, w, g, b, *_ in CONCEPTS if g == "A" and b == "named"]
named_terms = [F.when(F.col(f"c_{n}"), F.lit(w)).otherwise(F.lit(0))
               for n, a, w, g, b, *_ in CONCEPTS if g == "A" and b == "named"]
named_expr = named_terms[0]
for t in named_terms[1:]:
    named_expr = named_expr + t
df = df.withColumn("named_score", named_expr)
df = df.withColumn("named_a_hits", sum([F.col(f"c_{n}").cast("int") for n in named_a]))
df = df.withColumn("has_named_concept", (F.col("named_a_hits") > 0).cast("int"))

# unmatched_text: 1 when text was entered but nothing scored (words no rule recognized). Purely
# informational - it does NOT change the bucket - and flags the language-model target set.
filler_cols = [f"c_{n}" for n, a, *_ in CONCEPTS if a == "filler"]
df = df.withColumn("any_filler", sum([F.col(c).cast("int") for c in filler_cols]) > 0)
df = df.withColumn(
    "unmatched_text",
    (F.col("_has_text") & (F.col("total_score") == 0) & ~F.col("any_filler")).cast("int"),
)

# necessity_class: driven by total_score and the named flag. Simple cutoffs:
#   total_score >= NECESSARY_CUTOFF and a named concept present -> necessary
#   total_score == 0                                            -> not_necessary
#   otherwise                                                   -> indeterminate
df = df.withColumn(
    "necessity_class",
    F.when((F.col("total_score") >= NECESSARY_CUTOFF) & (F.col("has_named_concept") == 1),
           F.lit("necessary"))
     .when(F.col("total_score") == 0, F.lit("not_necessary"))
     .otherwise(F.lit("indeterminate")),
)

# confidence: score-based quality flag (not a CMS construct).
df = df.withColumn(
    "confidence",
    F.when((F.col("total_score") >= NECESSARY_CUTOFF) & (F.col("has_named_concept") == 1), F.lit("high"))
     .when(F.col("total_score") >= 1, F.lit("medium"))
     .otherwise(F.lit("low")),
)

# recommended_los: level of service implied by the monitoring axis, per [414.605] definitions.
#   Descriptive only until CMS BLS/ALS eligibility criteria are confirmed - not a billing call.
df = df.withColumn(
    "recommended_los",
    F.when(F.col("c_ventilator") | F.col("c_suctioning"), F.lit("CCT/SCT"))            # [414.605] SCT/ALS2
     .when((F.col("monitoring_score") >= 2) | F.col("c_iv_medication"), F.lit("ALS"))  # [414.605] ALS1/ALS2
     .when(F.col("mobility_score") >= 1, F.lit("BLS"))                                 # [414.605] BLS
     .otherwise(F.lit("none/indeterminate")),
)

# gy_disposition: relates the class to the GY-modifier process (4 Aug review-process meeting).
# not_necessary = no documented reason (a GY candidate); necessary = documented reason present;
# indeterminate = partial / review. Describes the order text at order time, not a billing call.
df = df.withColumn(
    "gy_disposition",
    F.when(F.col("necessity_class") == "not_necessary", F.lit("no documented reason - GY candidate"))
     .when(F.col("necessity_class") == "necessary", F.lit("documented reason present"))
     .otherwise(F.lit("partial - review")),
)

# classification_method: how the labels were assigned. Keyword/term matching today; the
# unmatched_text orders are the set a language model would categorise in a later pass.
df = df.withColumn("classification_method", F.lit("keyword_match"))

# why_labeled: for each order, the concepts that matched and the exact text that triggered each -
# e.g. "mobility_deficit[unable to ambulate]; oxygen[o2]". This is the reviewer's view of HOW an
# order was labeled: the actual words behind each concept. Empty when nothing matched (the
# text_unmatched case), which is itself the signal that the rules found nothing to go on.
_match_exprs = []
for _n, _a, _w, _g, _b, _ref, _url, _pat in CONCEPTS:
    _m = F.regexp_extract(F.col("_text"), _pat, 0)
    _match_exprs.append(
        F.when(F.col(f"c_{_n}"), F.concat(F.lit(_n + "["), _m, F.lit("]"))).otherwise(F.lit(None))
    )
df = df.withColumn("why_labeled", F.concat_ws("; ", F.array(*_match_exprs)))

# 6. Summary outputs (display only - nothing is written)

# Necessity distribution
display(
    df.groupBy("necessity_class")
      .agg(F.count("*").alias("orders"))
      .orderBy(F.desc("orders"))
)

# Unmatched-text breakdown - words present that no rule scored (the language-model opportunity).
display(
    df.groupBy("unmatched_text").agg(F.count("*").alias("orders")).orderBy("unmatched_text")
)

# Necessity by customer (skipped if no customer column resolved)
if CUSTOMER_COL is not None:
    display(
        df.groupBy(CUSTOMER_COL, "necessity_class")
          .agg(F.count("*").alias("orders"))
          .orderBy(CUSTOMER_COL, "necessity_class")
    )
else:
    print("Skipped necessity-by-customer: no customer column resolved.")

# Transport appropriateness: recommended vs requested level of service ([414.605])
if LOS_COL is not None:
    display(
        df.groupBy(LOS_COL, "recommended_los")
          .agg(F.count("*").alias("orders"))
          .orderBy(F.desc("orders"))
    )
else:
    display(
        df.groupBy("recommended_los")
          .agg(F.count("*").alias("orders"))
          .orderBy(F.desc("orders"))
    )

# 7. Example order texts per category

# For the 24 July action item: sample real free-text per class so stakeholders can see how each
# category is defined. Sampled, de-identified review only - not written back.

_example_cols = [c for c in [ORDER_ID_COL, LOS_COL] if c is not None] + [
    "total_score", "mobility_score", "monitoring_score", "named_score", "has_named_concept", "confidence", "unmatched_text", FREE_TEXT_COL
]
for cls in ["necessary", "indeterminate", "not_necessary"]:
    print("=" * 70)
    print(cls.upper())
    display(
        df.filter(F.col("necessity_class") == cls)
          .select(*_example_cols)
          .limit(15)
    )

# 8. Export summary workbook - med_nec_buckets_v2.xlsx

# Writes the aggregates that back the slides to a five-tab workbook. This is an OUTPUT FILE, not
# a table write - the source table is never modified. Mirrors the original med_nec_buckets.xlsx,
# updated to the medical-necessity schema (necessity_class, mobility/monitoring, appropriateness).

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from datetime import datetime
import re as _re

# openpyxl rejects ASCII control characters (except tab/newline/carriage return) with
# IllegalCharacterError. Real order text can contain them (stray glyphs from copy/paste, form
# artifacts like the character before "L1 kyphoplasty"), so strip them from every string cell
# before any Excel write. Applied to workbook builders and to the review/CSV frames.
_ILLEGAL_XLSX = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def _clean(v):
    if isinstance(v, str):
        return _ILLEGAL_XLSX.sub("", v)
    return v

def sanitize_df(pdf):
    import pandas.api.types as _pt
    pdf = pdf.copy()
    for c in pdf.columns:
        # Clean any column that is not purely numeric/bool (dtype may be 'object' or 'string').
        if not (_pt.is_numeric_dtype(pdf[c]) or _pt.is_bool_dtype(pdf[c])):
            pdf[c] = pdf[c].map(_clean)
    return pdf

# Run timestamp (YYYYMMDD_HHMMSS) appended to every output file so runs stay grouped and do not
# overwrite each other. Set once here and reused for the workbook, detail CSV, and review workbook.
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_XLSX = f"med_nec_buckets_v2_{RUN_TS}.xlsx"
# Destination directory. Workspace paths (/Workspace/...) are a real mounted filesystem, so the
# workbook is written there directly with pandas/openpyxl - no dbutils.fs.cp needed.
# For a Volume or DBFS destination instead, use e.g.:
#   "/Volumes/prod-sandbox/vivekkumar_patel/exports"   (Unity Catalog volume)
#   "dbfs:/FileStore/med_nec"                            (download via /files/ URL)
OUTPUT_DIR = "/Workspace/Users/josh.smitherman@gmr.net/med_nec/data"

total = df.count()

def with_pct(sdf, count_col="orders"):
    pdf = sdf.toPandas()
    pdf["pct_of_scope"] = (pdf[count_col] / total * 100).round(1)
    return pdf

# --- Summary tab: necessity, confidence, indeterminate reasons ---
nec_pdf = with_pct(
    df.groupBy("necessity_class").agg(F.count("*").alias("orders")).orderBy(F.desc("orders"))
)
conf_pdf = with_pct(
    df.groupBy("confidence").agg(F.count("*").alias("orders")).orderBy(F.desc("orders"))
)
indet_pdf = with_pct(
    df.groupBy("unmatched_text").agg(F.count("*").alias("orders")).orderBy(F.desc("orders"))
)

# --- Concepts tab: dictionary with counts and CMS citations (backs the concept slide) ---
concept_sums = df.agg(
    *[F.sum(F.col(f"c_{n}").cast("int")).alias(n) for n, *_ in CONCEPTS]
).collect()[0].asDict()
concept_pdf = pd.DataFrame([
    {"concept": n, "group": g, "axis": a, "weight": w, "basis": b,
     "cms_ref": ref, "source_url": url, "match_terms": pat,
     "tags": int(concept_sums[n] or 0),
     "pct_of_orders": round((concept_sums[n] or 0) / total * 100, 1)}
    for (n, a, w, g, b, ref, url, pat) in CONCEPTS
]).sort_values("tags", ascending=False).reset_index(drop=True)

# --- Mobility reasons tab: what reasons for mobility appear in the order text, and how often.
# Requested in the 10 Aug meeting (text analysis of reasons for mobility). Reuses the mobility-axis
# concept tags already computed - no new matching. Overall counts, plus counts among necessary
# orders, so the documented reasons that carry necessity are visible. ---
mobility_concepts = [(n, w, b) for n, a, w, g, b, *_ in CONCEPTS if a == "mobility"]
nec_df = df.filter(F.col("necessity_class") == "necessary")
nec_total = nec_df.count()
mob_sums_all = df.agg(
    *[F.sum(F.col(f"c_{n}").cast("int")).alias(n) for n, *_ in mobility_concepts]
).collect()[0].asDict()
mob_sums_nec = nec_df.agg(
    *[F.sum(F.col(f"c_{n}").cast("int")).alias(n) for n, *_ in mobility_concepts]
).collect()[0].asDict()
mobility_pdf = pd.DataFrame([
    {"mobility_reason": n, "weight": w, "basis": b,
     "orders": int(mob_sums_all[n] or 0),
     "pct_of_scope": round((mob_sums_all[n] or 0) / total * 100, 1),
     "orders_necessary": int(mob_sums_nec[n] or 0),
     "pct_of_necessary": round((mob_sums_nec[n] or 0) / max(nec_total, 1) * 100, 1)}
    for (n, w, b) in mobility_concepts
]).sort_values("orders", ascending=False).reset_index(drop=True)


# --- Where tab: by customer, by level of service, transport appropriateness ---
where_blocks = []
if CUSTOMER_COL is not None:
    where_blocks.append((
        "Necessity by customer",
        df.groupBy(CUSTOMER_COL, "necessity_class").agg(F.count("*").alias("orders"))
          .orderBy(CUSTOMER_COL, "necessity_class").toPandas()
    ))
if LOS_COL is not None:
    where_blocks.append((
        "Necessity by level of service",
        df.groupBy(LOS_COL, "necessity_class").agg(F.count("*").alias("orders"))
          .orderBy(LOS_COL, "necessity_class").toPandas()
    ))
    where_blocks.append((
        "Transport appropriateness (requested vs recommended)",
        df.groupBy(LOS_COL, "recommended_los").agg(F.count("*").alias("orders"))
          .orderBy(F.desc("orders")).toPandas()
    ))

# --- Detail tab: row-level trace. Original order data -> every concept label -> scores ->
# necessity class -> GY disposition, so a reviewer can walk a single order end to end.
# The xlsx tab holds a stratified sample (DETAIL_SAMPLE_N per class) to stay openable; set
# WRITE_FULL_DETAIL_CSV to also write every scored order to a CSV beside the workbook. ---
DETAIL_SAMPLE_N = 150
WRITE_FULL_DETAIL_CSV = True

concept_names = [n for n, *_ in CONCEPTS]
context_cols = [c for c in [ORDER_ID_COL, LOS_COL, CUSTOMER_COL] if c is not None]
# concept hit columns, cast to 1/0 and named by the concept (the "labels")
label_cols = [F.col(f"c_{n}").cast("int").alias(n) for n in concept_names]
derived_cols = ["mobility_score", "monitoring_score", "total_score",
                "named_score", "has_named_concept", "necessity_class",
                "unmatched_text", "confidence", "recommended_los",
                "gy_disposition", "why_labeled", "classification_method"]
detail_select = ([F.col(c) for c in context_cols] + label_cols
                 + [F.col(c) for c in derived_cols] + [F.col(FREE_TEXT_COL)])

detail_pdf = pd.concat([
    df.filter(F.col("necessity_class") == cls).select(*detail_select).limit(DETAIL_SAMPLE_N).toPandas()
    for cls in ["necessary", "indeterminate", "not_necessary"]
], ignore_index=True)

# --- Examples tab: sampled free-text per class ---
ex_cols = [c for c in [ORDER_ID_COL, LOS_COL] if c is not None] + [
    "necessity_class", "total_score", "mobility_score", "monitoring_score",
    "named_score", "has_named_concept", "confidence", "unmatched_text", FREE_TEXT_COL,
]
examples_pdf = pd.concat([
    df.filter(F.col("necessity_class") == cls).select(*ex_cols).limit(20).toPandas()
    for cls in ["necessary", "indeterminate", "not_necessary"]
], ignore_index=True)

# --- Definitions tab: embedded so the workbook is self-explaining ---
definitions_pdf = pd.DataFrame([
    ["necessity_class = necessary", "A named CMS concept meets the clear threshold, or the full 10.2.3 test is met in the order text.", "BPM10 10.2.3 / 10.2.1; 414.605"],
    ["necessity_class = not_necessary", "Text present with a clinical reason field but only filler, no Group A concept.", "RSN AM600; MLN"],
    ["necessity_class = indeterminate", "Some signal but below the clear threshold, or text present that matched no term.", "BPM10 10.2.1"],
    ["unmatched_text", "1 = text present but nothing scored (words no rule recognized). Informational; the language-model target. Does not change the bucket.", "BPM10 10.2.1"],
    ["mobility_score", "Weighted sum of mobility-axis concepts - why other transport is contraindicated.", "BPM10 10.2.1 / 10.2.3"],
    ["monitoring_score", "Weighted sum of monitoring-axis concepts - why this level of service.", "414.605"],
    ["total_score", "mobility_score + monitoring_score. Drives the bucket via cutoffs.", "composite (provisional)"],
    ["named_score", "Points from named CMS concepts only. A named concept (named_score > 0) is required for necessary.", "BPM10 / 414.605"],
    ["has_named_concept", "1 if any named CMS concept matched. Gates the necessary band.", "BPM10 / 414.605"],
    ["confidence", "Internal quality flag on the classification (high / medium / low). Not a CMS construct.", "n/a"],
    ["recommended_los", "Level of service implied by the monitoring axis. Descriptive, not a billing call.", "414.605; 410.40(c)"],
    ["gy_disposition", "GY process relation: not_necessary = no documented reason (GY candidate); necessary = documented reason present; indeterminate = partial, review.", "review-process 4 Aug"],
    ["GY modifier", "Billing marks a non-medically-necessary Medicare trip GY: transport provided, payment not requested. Enables billing the facility/patient and building a record.", "review-process 4 Aug"],
    ["PCS", "Physician Certification Statement - the order-time certification. This analysis reads the order-time (PCS-side) free text.", "410.40(d); review-process"],
    ["PCR", "Patient Care Report - the crew's documentation in ImageTrend. A separate source, NOT in this table. Billing requires the PCR to support the PCS.", "review-process 4 Aug"],
    ["Routing", "Non-emergency Medicare trips go to Hemlata Khatri's team (under Jen); Medicaid, commercial and self-pay go to Integra. Coding, with medical-necessity confirmation, is the first step for both.", "review-process 4 Aug"],
    ["classification_method", "How labels were assigned: keyword_match today. text_unmatched orders are the set a language model would categorise next.", "n/a"],
    ["Group A vs Group B", "A = a clinical reason (12 concepts). B = vague filler that confers no necessity (3).", "BPM10 10.2.1; MLN"],
    ["named vs inferred", "named = explicit in CMS text. inferred = derived from the 10.2.1 general test.", "BPM10 10.2.1"],
    ["Limitation", "Necessity here reflects ORDER-TIME documentation only. The final billing determination also depends on the crew PCR supporting the PCS, which is not in this dataset.", "review-process 4 Aug"],
    ["Scope", f"Non-emergent ground ambulance only. {total:,} orders in scope, 2024 to present. Wheelchair, ambulatory, air and emergent codes excluded.", "BPM10 10.2.1"],
], columns=["term", "definition", "cms_ref"])

# --- Scoring tab: weight table, aggregation formula, and score distribution ---
weight_pdf = pd.DataFrame([
    {"concept": n, "axis": a, "group": g, "basis": b, "weight": w, "cms_ref": ref}
    for (n, a, w, g, b, ref, url, pat) in CONCEPTS
]).sort_values(["axis", "weight"], ascending=[True, False]).reset_index(drop=True)

formula_pdf = pd.DataFrame([
    ["mobility_score", "sum of weights of matched mobility-axis concepts", "BPM10 10.2.1 / 10.2.3"],
    ["monitoring_score", "sum of weights of matched monitoring-axis concepts", "414.605"],
    ["total_score", "mobility_score + monitoring_score", "composite (provisional)"],
    ["necessary", f"total_score >= {NECESSARY_CUTOFF} AND a named concept present", "BPM10 / 414.605"],
    ["not_necessary", "total_score == 0", "RSN AM600"],
    ["indeterminate", "everything else", "BPM10 10.2.1"],
    ["note", "weights and bonus are tunable modelling choices, not CMS figures - validate with SMEs", "n/a"],
], columns=["element", "definition", "cms_ref"])

total_dist_pdf = with_pct(
    df.groupBy("total_score").agg(F.count("*").alias("orders"))
      .orderBy("total_score")
)

score_by_class_pdf = (
    df.groupBy("necessity_class")
      .agg(F.round(F.avg("total_score"), 2).alias("avg_total_score"),
           F.min("total_score").alias("min_score"),
           F.max("total_score").alias("max_score"),
           F.count("*").alias("orders"))
      .orderBy(F.desc("avg_total_score")).toPandas()
)

sheets = {
    "Definitions": [("Definitions and CMS basis", definitions_pdf)],
    "Summary":     [("Necessity", nec_pdf), ("Confidence", conf_pdf),
                    ("Indeterminate - why", indet_pdf)],
    "Scoring":     [("Concept weights", weight_pdf),
                    ("Aggregation formula", formula_pdf),
                    ("Total score distribution", total_dist_pdf),
                    ("Total score by necessity class", score_by_class_pdf)],
    "Concepts":    [("Concept dictionary - terms, weights, CMS basis", concept_pdf)],
    "Mobility reasons": [("Reasons for mobility - frequency overall and among necessary orders",
                          mobility_pdf)],
    "Detail":      [(f"Row-level trace: order -> labels -> class -> GY disposition "
                     f"(sample of {DETAIL_SAMPLE_N} per class)", detail_pdf)],
    "Where":       where_blocks or [("No customer/LOS column resolved", pd.DataFrame({"note": ["set FIELD_CANDIDATES"]}))],
    "Examples":    [("Sampled order text per class", examples_pdf)],
}

def _coerce(v):
    if pd.isna(v):
        return None
    v = v.item() if hasattr(v, "item") else v
    return _clean(v)

def build_workbook(path, sheets):
    wb = Workbook(); wb.remove(wb.active)
    for name, blocks in sheets.items():
        ws = wb.create_sheet(name[:31]); r = 1
        for title, pdf in blocks:
            if title:
                ws.cell(r, 1, title).font = Font(bold=True); r += 1
            for j, col in enumerate(pdf.columns, 1):
                ws.cell(r, j, str(col)).font = Font(bold=True)
            r += 1
            for _, row in pdf.iterrows():
                for j, val in enumerate(row, 1):
                    ws.cell(r, j, _coerce(val))
                r += 1
            r += 1
        for col_cells in ws.columns:
            w = min(70, max((len(str(c.value)) for c in col_cells if c.value is not None), default=8) + 2)
            ws.column_dimensions[col_cells[0].column_letter].width = w
    wb.save(path)

dest = f"{OUTPUT_DIR.rstrip('/')}/{OUTPUT_XLSX}"
# Always build to a driver-local path first, then copy to the destination. Direct writes to
# /Workspace and /Volumes FUSE paths are unreliable on serverless compute.
local_path = f"/tmp/{OUTPUT_XLSX}"
build_workbook(local_path, sheets)
import os, shutil
if OUTPUT_DIR.startswith("/Workspace/") or OUTPUT_DIR.startswith("/dbfs/") or OUTPUT_DIR.startswith("/Volumes/"):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        shutil.copyfile(local_path, dest)
    except Exception:
        dbutils.fs.cp(f"file:{local_path}", dest)
    print("Saved workbook to:", dest)
else:
    try:
        dbutils.fs.mkdirs(OUTPUT_DIR)
        dbutils.fs.cp(f"file:{local_path}", dest)
        print("Saved workbook to:", dest)
        if OUTPUT_DIR.startswith("dbfs:/FileStore"):
            print("Download via: <workspace-url>/files/" + dest.split("/FileStore/", 1)[1])
    except Exception as e:
        print("Wrote local copy:", local_path)
        print("Copy to", dest, "failed - set OUTPUT_DIR to a writable path.")
        print("Reason:", e)

# Full row-level detail as CSV beside the workbook (every scored order, not just the sample).
if WRITE_FULL_DETAIL_CSV:
    csv_name = OUTPUT_XLSX.replace(".xlsx", "_detail.csv")
    csv_dest = f"{OUTPUT_DIR.rstrip('/')}/{csv_name}"
    local_csv = f"/tmp/{csv_name}"
    sanitize_df(df.select(*detail_select).toPandas()).to_csv(local_csv, index=False)
    try:
        if OUTPUT_DIR.startswith("/Workspace/") or OUTPUT_DIR.startswith("/dbfs/") or OUTPUT_DIR.startswith("/Volumes/"):
            shutil.copyfile(local_csv, csv_dest)
        else:
            dbutils.fs.cp(f"file:{local_csv}", csv_dest)
        print("Saved full detail CSV to:", csv_dest)
    except Exception as e:
        print("Wrote local detail CSV:", local_csv, "| copy failed:", e)

# 9. Review workbook - for walking through actual orders and how each was labeled

# A separate, scrollable workbook aimed at the business review, not analysis. Each tab is a
# curated view: the free text is shown first, then the class, the GY disposition, and why_labeled
# (the exact words that triggered each concept). Header rows are frozen so labels stay visible
# while scrolling. Built with pandas for speed on the large tabs.

from openpyxl.utils import get_column_letter

BUILD_REVIEW_WORKBOOK = True
REVIEW_XLSX = f"med_nec_review_{RUN_TS}.xlsx"
MAX_REVIEW_ROWS = 200000   # safety cap for the "All with text" tab

if BUILD_REVIEW_WORKBOOK:
    review_cols = [c for c in [ORDER_ID_COL, LOS_COL, CUSTOMER_COL] if c is not None] + [
        FREE_TEXT_COL, "necessity_class", "gy_disposition", "why_labeled",
        "total_score", "named_score", "mobility_score", "monitoring_score",
        "has_named_concept", "confidence", "unmatched_text", "recommended_los",
    ]

    def review_frame(spark_df, limit=None):
        d = spark_df.select(*review_cols)
        if limit:
            d = d.limit(limit)
        return d.toPandas()

    with_text = df.filter(F.col("_has_text"))
    # Borderline: orders near the necessary/not-necessary boundary - the cases worth scrutiny.
    #   single-concept clears, indeterminate orders with a real signal, and a scored concept
    #   sitting alongside filler wording.
    borderline = df.filter(
        ((F.col("necessity_class") == "necessary") & (F.col("total_score") == NECESSARY_CUTOFF))
        | ((F.col("necessity_class") == "indeterminate") & (F.col("total_score") >= 2))
        | ((F.col("total_score") >= 1) & F.col("any_filler"))
    )

    guide_pdf = pd.DataFrame([
        ["Purpose", "Walk through actual orders and see how each was labeled for medical necessity."],
        ["free text (ClinicalData)", "The reason-for-transport text entered at order time. Read this first."],
        ["necessity_class", "Label for the order text: necessary / indeterminate / not_necessary."],
        ["gy_disposition", "GY process relation: no documented reason (GY candidate) / documented reason present / partial, review."],
        ["why_labeled", "The concepts that matched and the exact words that triggered each. Empty = the rules found nothing (text_unmatched)."],
        ["total_score", "mobility_score + monitoring_score. total_score == 0 -> not_necessary; >= 3 with a named concept -> necessary; else indeterminate."],
        ["Tab: All with text", "Every order that has free text."],
        ["Tab: Not necessary", "Orders labeled not_necessary - no documented reason in the order text."],
        ["Tab: Rules missed", "text_unmatched - text present that no keyword rule read. The language-model candidates."],
        ["Tab: Borderline", "Orders near the decision boundary - single-concept clears, near-miss indeterminates, reason-plus-filler."],
        ["Limitation", "Reflects ORDER-TIME documentation only. Final billing also depends on the crew PCR supporting the PCS, which is not in this data."],
    ], columns=["item", "meaning"])

    review_tabs = {
        "How to read": guide_pdf,
        "All with text": review_frame(with_text, MAX_REVIEW_ROWS),
        "Not necessary": review_frame(df.filter(F.col("necessity_class") == "not_necessary")),
        "Rules missed": review_frame(df.filter(F.col("unmatched_text") == 1)),
        "Borderline": review_frame(borderline),
    }

    def build_review(path, tabs):
        # Write to a driver-local path first, then the caller copies it to the destination.
        # Direct pandas writes to /Workspace or /Volumes FUSE paths are unreliable on serverless.
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, pdf in tabs.items():
                sheet = (name[:31] or "Sheet")
                if pdf is None or len(pdf) == 0:
                    pdf = pd.DataFrame({"note": ["no rows for this view"]})
                pdf = sanitize_df(pdf)
                pdf.to_excel(xw, sheet_name=sheet, index=False)
                ws = xw.sheets[sheet]
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                ws.freeze_panes = "A2"
                for i, col in enumerate(pdf.columns, 1):
                    L = get_column_letter(i)
                    if col in (FREE_TEXT_COL, "why_labeled"):
                        ws.column_dimensions[L].width = 60
                    else:
                        ws.column_dimensions[L].width = min(26, max(len(str(col)) + 2, 10))

    review_dest = f"{OUTPUT_DIR.rstrip('/')}/{REVIEW_XLSX}"
    local_review = f"/tmp/{REVIEW_XLSX}"
    build_review(local_review, review_tabs)
    if OUTPUT_DIR.startswith("/Workspace/") or OUTPUT_DIR.startswith("/dbfs/") or OUTPUT_DIR.startswith("/Volumes/"):
        # Copy the finished file to the mounted destination (works on serverless).
        import shutil
        try:
            shutil.copyfile(local_review, review_dest)
        except Exception:
            dbutils.fs.cp(f"file:{local_review}", review_dest)
        print("Saved review workbook to:", review_dest)
        for nm, pdf in review_tabs.items():
            print(f"  {nm}: {len(pdf):,} rows")
    else:
        try:
            dbutils.fs.cp(f"file:{local_review}", review_dest)
            print("Saved review workbook to:", review_dest)
        except Exception as e:
            print("Wrote local review workbook:", local_review, "| copy failed:", e)

# 10. Genie table (optional) - persist the scored data as a curated table for AI/BI Genie

# Genie queries a real table or view, so this is the one place the pipeline writes. It is OFF by
# default to preserve the read-only default. Set WRITE_GENIE_TABLE = True and a target you can
# write to. Writes only this derived output - it never touches the source table.

WRITE_GENIE_TABLE = False
GENIE_TABLE = "`prod-sandbox`.`vivekkumar_patel`.`med_nec_genie`"

if WRITE_GENIE_TABLE:
    concept_names_all = [n for n, *_ in CONCEPTS]
    genie_cols = (
        [c for c in [ORDER_ID_COL, LOS_COL, CUSTOMER_COL] if c is not None]
        + ["necessity_class", "gy_disposition", "total_score", "mobility_score",
           "monitoring_score", "named_score", "has_named_concept", "unmatched_text",
           "confidence", "recommended_los", "why_labeled", "classification_method"]
        + [F.col(f"c_{n}").cast("int").alias(n) for n in concept_names_all]
        + [F.col(FREE_TEXT_COL).alias("clinical_text")]
    )
    genie_df = df.select(*[F.col(c) if isinstance(c, str) else c for c in genie_cols])
    genie_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GENIE_TABLE)
    print("Wrote Genie table:", GENIE_TABLE, "-", genie_df.count(), "rows")

    # Column comments - Genie relies on these to understand the data. Keep them factual.
    col_comments = {
        "necessity_class": "Label for the order text: necessary / not_necessary / indeterminate.",
        "gy_disposition": "GY-process relation: not_necessary=no documented reason (GY candidate); necessary=documented reason present; indeterminate=partial.",
        "total_score": "mobility_score + monitoring_score. 0=not_necessary; >=3 with a named concept=necessary; else indeterminate.",
        "mobility_score": "Sum of matched mobility-axis concept weights (why other transport is contraindicated). BPM10 10.2.1/10.2.3.",
        "monitoring_score": "Sum of matched monitoring-axis concept weights (why this level of service). 42 CFR 414.605.",
        "named_score": "Points from named CMS concepts only. A named concept is required for necessary.",
        "has_named_concept": "1 if any named CMS concept matched, else 0.",
        "unmatched_text": "1 if text was entered but nothing scored (words no rule recognized). The language-model target set.",
        "confidence": "Score-based quality flag: high / medium / low.",
        "recommended_los": "Level of service implied by monitoring concepts (descriptive, not a billing call).",
        "why_labeled": "The concepts that matched and the exact text that triggered each.",
        "clinical_text": "The free-text reason-for-transport entered at order time (order-time / PCS-side documentation).",
    }
    for n, a, w, g, b, ref, url, pat in CONCEPTS:
        col_comments[n] = f"1 if the order text matched the {n} concept ({a} axis, weight {w}, {b}). Basis: {ref}."
    for col, comment in col_comments.items():
        safe = comment.replace("'", "")
        try:
            spark.sql(f"ALTER TABLE {GENIE_TABLE} ALTER COLUMN {col} COMMENT '{safe}'")
        except Exception as e:
            print(f"  comment on {col} skipped:", e)
    spark.sql(
        f"COMMENT ON TABLE {GENIE_TABLE} IS "
        "'Non-emergent ground ambulance orders scored for medical-necessity documentation against "
        "CMS criteria (order-time text only). One row per order. Labels describe the documentation, "
        "not the trip, and are not a billing determination.'"
    )
    print("Applied column and table comments. Point a Genie space at:", GENIE_TABLE)

# 11. LLM approach (optional) - second opinion on the orders the rules could not read

# The scoring path above is the rule-based approach. This is the LLM approach, using the same
# pattern as the NurseNav project (OpenAI SDK -> Databricks serving endpoint, extract-then-judge):
# the model extracts facts with an evidence quote for every flag, then the SAME cutoffs assign the
# label. It runs ONLY on the unmatched_text set - orders where text was entered but no rule scored
# it - because that is where scoring adds nothing and the LLM can. Running the two side by side is
# the rule-based vs LLM experiment. OFF by default (it calls a paid endpoint). SAMPLE first.
# All LLM code is inlined here (and imports openai) so it only loads when RUN_LLM is True.

RUN_LLM = False
LLM_SAMPLE_N = 200       # cap the rules-missed set while validating the prompt; None = all
LLM_WRITE_CSV = True
LLM_MODEL = "databricks-gpt-oss-120b"      # same model NurseNav uses
LLM_BASE_URL = "https://adb-2790612761746757.17.azuredatabricks.net/serving-endpoints"

if RUN_LLM:
    import json as _json
    from openai import OpenAI

    # --- LLM client and call (OpenAI SDK -> Databricks serving endpoint, NurseNav pattern) ---
    _token = (
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        if "dbutils" in dir() else os.environ.get("DATABRICKS_TOKEN", "")
    )
    _client = OpenAI(api_key=_token, base_url=LLM_BASE_URL)

    def llm_call(system_prompt, user_prompt, max_tokens=1500):
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_prompt}],
            temperature=0.1, max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return _json.dumps(content)
        return content

    # --- Extraction prompt: the model pulls facts only; the rules below assign the label ---
    MED_NEC_SYSTEM_PROMPT = """
You are a clinical documentation information extraction assistant for non-emergent ground ambulance orders.
Your ONLY task is to extract structured facts from the free-text transport reason recorded at order time.

CRITICAL CONSTRAINTS
- Do NOT provide medical advice, diagnose, or decide whether the transport was appropriate.
- Do NOT guess. Only extract what is explicitly supported by the text.
- Handle negations correctly (for example "no oxygen" is false).
- For every boolean set to true you MUST provide a verbatim evidence quote from the text.
- If no clinical reason is documented, set documentation.reason_documented = false.

ABBREVIATIONS: BLS/ALS=basic/advanced life support, O2=oxygen, IV=intravenous, w/c=wheelchair,
amb=ambulate/ambulatory, ETT=endotracheal tube, trach=tracheostomy.

OUTPUT: valid JSON only, all keys present, booleans never null.

SCHEMA:
{
  "order_id": string,
  "text_summary": string,
  "mobility": {
    "bed_confined": boolean, "cannot_ambulate": boolean, "cannot_sit": boolean,
    "bariatric_handling": boolean, "wound_or_ostomy_positioning": boolean,
    "behavioral_or_altered_mental": boolean
  },
  "monitoring": {
    "ventilator_or_airway": boolean, "airway_suctioning": boolean, "iv_medication": boolean,
    "cardiac_monitoring": boolean, "oxygen": boolean, "isolation_precautions": boolean
  },
  "filler_only": {
    "weakness_only": boolean, "fall_risk_only": boolean, "nonclinical_reason": boolean
  },
  "documentation": { "reason_documented": boolean, "note_is_operational_only": boolean },
  "evidence": [ {"field": string, "value": boolean, "quote": string} ]
}

FIELD NOTES
- mobility captures why the patient cannot travel by wheelchair van or car.
- monitoring captures clinical care needed during transport (drives level of service).
- filler_only captures vague terms that carry no necessity on their own.
- Set reason_documented = false when the field is empty or contains only filler.

Now extract from the following order text."""

    def build_user_prompt(order_id, text):
        return f"order_id: {order_id}\n\nOrder text:\n{text}"

    # --- Validation: strict JSON, all sections present, every true flag carries evidence ---
    BOOL_SECTIONS = {
        "mobility": ["bed_confined", "cannot_ambulate", "cannot_sit", "bariatric_handling",
                     "wound_or_ostomy_positioning", "behavioral_or_altered_mental"],
        "monitoring": ["ventilator_or_airway", "airway_suctioning", "iv_medication",
                       "cardiac_monitoring", "oxygen", "isolation_precautions"],
        "filler_only": ["weakness_only", "fall_risk_only", "nonclinical_reason"],
    }
    REQUIRED_TOP_KEYS = {"order_id", "text_summary", "mobility", "monitoring",
                         "filler_only", "documentation", "evidence"}
    # Named concepts (explicit in CMS text) - only these can carry an order to necessary.
    NAMED_FIELDS = {
        "mobility.bed_confined", "mobility.cannot_ambulate", "mobility.cannot_sit",
        "monitoring.ventilator_or_airway", "monitoring.airway_suctioning",
        "monitoring.iv_medication", "monitoring.cardiac_monitoring",
    }
    LLM_WEIGHTS = {
        "mobility.bed_confined": 3, "mobility.cannot_ambulate": 3, "mobility.cannot_sit": 3,
        "mobility.bariatric_handling": 2, "mobility.wound_or_ostomy_positioning": 1,
        "mobility.behavioral_or_altered_mental": 1,
        "monitoring.ventilator_or_airway": 3, "monitoring.airway_suctioning": 3,
        "monitoring.iv_medication": 3, "monitoring.cardiac_monitoring": 2,
        "monitoring.oxygen": 1, "monitoring.isolation_precautions": 1,
    }

    def validate_extraction(raw):
        try:
            obj = _json.loads(raw)
        except Exception as e:
            return False, {}, f"invalid JSON: {e}"
        missing = REQUIRED_TOP_KEYS - set(obj)
        if missing:
            return False, obj, f"missing keys: {sorted(missing)}"
        for section, keys in BOOL_SECTIONS.items():
            sec = obj.get(section, {})
            if not isinstance(sec, dict):
                return False, obj, f"{section} must be an object"
            for k in keys:
                if not isinstance(sec.get(k), bool):
                    return False, obj, f"{section}.{k} must be boolean"
        ev_fields = {e.get("field") for e in obj.get("evidence", []) if isinstance(e, dict)}
        for section, keys in BOOL_SECTIONS.items():
            for k in keys:
                if obj[section][k] and f"{section}.{k}" not in ev_fields:
                    return False, obj, f"true flag without evidence: {section}.{k}"
        return True, obj, "OK"

    # --- Rules engine: the SAME cutoffs as the scoring path, applied to extracted facts ---
    def judge(obj):
        mob = sum(LLM_WEIGHTS[f"mobility.{k}"] for k in BOOL_SECTIONS["mobility"] if obj["mobility"][k])
        mon = sum(LLM_WEIGHTS[f"monitoring.{k}"] for k in BOOL_SECTIONS["monitoring"] if obj["monitoring"][k])
        total = mob + mon
        has_named = any(
            obj[sec][k] for sec in ("mobility", "monitoring") for k in BOOL_SECTIONS[sec]
            if f"{sec}.{k}" in NAMED_FIELDS
        )
        if total >= NECESSARY_CUTOFF and has_named:
            cls = "necessary"
        elif total == 0:
            cls = "not_necessary"
        else:
            cls = "indeterminate"
        return {"necessity_class_llm": cls, "mobility_score_llm": mob,
                "monitoring_score_llm": mon, "total_score_llm": total,
                "has_named_concept_llm": int(has_named)}

    # --- Run on the rules-missed set ---
    id_col = ORDER_ID_COL or "OrderId"
    missed_pdf = (
        df.filter(F.col("unmatched_text") == 1)
          .select(F.col(id_col).alias("order_id"), F.col(FREE_TEXT_COL).alias("text"),
                  F.col("necessity_class"))
          .toPandas()
    )
    if LLM_SAMPLE_N:
        missed_pdf = missed_pdf.head(LLM_SAMPLE_N)
    print(f"LLM approach: extracting {len(missed_pdf):,} rules-missed orders via {LLM_MODEL}")

    rows = []
    for _, r in missed_pdf.iterrows():
        prompt = build_user_prompt(str(r["order_id"]), r["text"] or "")
        try:
            raw = llm_call(MED_NEC_SYSTEM_PROMPT, prompt)
            ok, obj, msg = validate_extraction(raw)
        except Exception as e:
            ok, obj, msg = False, {}, f"call failed: {e}"
        rec = {"order_id": r["order_id"], "necessity_class": r["necessity_class"],
               "extraction_valid": int(ok), "extraction_error": "" if ok else msg,
               "clinical_text": r["text"]}
        if ok:
            rec.update(judge(obj))
            rec["text_summary"] = obj.get("text_summary", "")
            rec["evidence"] = "; ".join(
                f'{e.get("field")}[{e.get("quote")}]' for e in obj.get("evidence", [])
                if isinstance(e, dict)
            )
        rows.append(rec)

    llm_pdf = pd.DataFrame(rows)
    valid_rate = llm_pdf["extraction_valid"].mean() if len(llm_pdf) else 0.0
    print(f"Valid extractions: {valid_rate:.1%}  (target >= 95% before scaling)")
    if valid_rate < 0.95 and len(llm_pdf):
        print("  Below target - review extraction_error before running the full set:")
        print(llm_pdf.loc[llm_pdf.extraction_valid == 0, "extraction_error"].value_counts().head(10))

    ok_pdf = llm_pdf[llm_pdf.extraction_valid == 1]
    if len(ok_pdf):
        print("Rule-based label vs LLM label on the rules-missed set:")
        print(ok_pdf.groupby(["necessity_class", "necessity_class_llm"]).size()
                    .reset_index(name="orders").sort_values("orders", ascending=False).to_string(index=False))

    if LLM_WRITE_CSV and len(llm_pdf):
        llm_csv = f"med_nec_llm_{RUN_TS}.csv"
        local_llm = f"/tmp/{llm_csv}"
        sanitize_df(llm_pdf).to_csv(local_llm, index=False)
        llm_dest = f"{OUTPUT_DIR.rstrip('/')}/{llm_csv}"
        try:
            if OUTPUT_DIR.startswith("/Workspace/") or OUTPUT_DIR.startswith("/dbfs/") or OUTPUT_DIR.startswith("/Volumes/"):
                shutil.copyfile(local_llm, llm_dest)
            else:
                dbutils.fs.cp(f"file:{local_llm}", llm_dest)
            print("Saved LLM results to:", llm_dest)
        except Exception as e:
            print("Wrote local LLM CSV:", local_llm, "| copy failed:", e)

# Notes / open items

# - Weights and thresholds are provisional. They encode the CMS hierarchy (named [BPM10] 10.2.3
#   concepts strongest, inferred [BPM10] 10.2.1 concepts weakest) but should be validated by the
#   SMEs and, once available, against actual denial outcomes in [RSN].
# - text_unmatched is the LLM target - text present that no term captured. That count is the
#   concrete case for language-model classification over term matching.
# - recommended_los is descriptive until CMS BLS/ALS eligibility criteria in [414.605] and
#   [410.40](c) are confirmed; it surfaces requested-vs-recommended mismatches, not billing calls.
# - behavioral, oxygen, isolation, wound_ostomy, bariatric are inferred from [BPM10] 10.2.1 and
#   are the first candidates for SME challenge; behavioral is the weakest.
# - Second free-text field: the review raised expanding beyond ClinicalData. Add fields to the
#   match step only after confirming they are point-of-order per [BPM10] 10.2.4, not post-hoc.
