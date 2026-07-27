# Medical Necessity Scoring - revised schema

# Reframes the non-emergent ground analysis around **medical necessity and transport
# appropriateness** rather than payment or denial. Per the 24 July review:

# - Two scoring axes, not one label: a **mobility** axis (why other transport is
#   contraindicated) and a **monitoring** axis (why this level of service).
# - A **two-bucket** determination - clearly necessary / clearly not necessary - with an
#   explicit **indeterminate** middle to be refined over time.
# - **Weighted concepts** and a **confidence** score, both driven from one config block.
# - No payment-focused terminology in any output column.

# Read-only throughout. No table writes.

# CMS source documents (every field, concept, and term below cites one of these):
#   [BPM10]  Medicare Benefit Policy Manual, Pub. 100-02, Ch. 10 - Ambulance Services
#            https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c10.pdf
#            - 10.2.1 general medical necessity test; 10.2.3 bed-confined three-prong test.
#   [410.40] 42 CFR 410.40 - Coverage of ambulance services (levels of service, physician cert.)
#            https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-B/part-410/subpart-B/section-410.40
#   [414.605] 42 CFR 414.605 - Definitions of BLS, ALS1, ALS2, SCT, ALS assessment/intervention
#            https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-B/part-414/subpart-H/section-414.605
#   [CPM15]  Medicare Claims Processing Manual, Pub. 100-04, Ch. 15 - Ambulance (adjudication)
#            https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/clm104c15ambulance.pdf
#   [RSN]    CMS Ambulance Transport Reason Codes and Statements (denial codes, e.g. AM600)
#            https://www.cms.gov/files/document/ambulance-transport-reason-codes-statements.pdf
#   [MLN]    CMS MLN Ambulance Services - provider compliance tips (documentation guidance)
#            https://www.cms.gov/training-education/medicare-learning-networkr-mln/compliance/medicare-provider-compliance-tips/ambulance-services

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

SOURCE_TABLE   = "prod-sandbox.vivekkumar_patel.temp_tnet_tripmaster"

# ClinicalData: free-text reason for transport. This is the documentation CMS reviews for
#   medical necessity; per [BPM10] 10.2.1 and 10.2.4, the reason other transport is
#   contraindicated must be recorded. Denial code AM600 in [RSN] fires when it is absent.
FREE_TEXT_COL  = "ClinicalData"      # cite: [BPM10] 10.2.1, 10.2.4 ; [RSN] AM600

# LevelOfService: BLS / ALS / CCT etc. The level billed must itself be medically necessary,
#   with the covered levels defined in [414.605] and [410.40](c).
LOS_COL        = "LevelOfService"    # cite: [414.605] ; [410.40](c)

CUSTOMER_COL   = "Customer"          # operational grouping only - no CMS basis
ORDER_ID_COL   = "OrderId"           # operational key only - no CMS basis

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

BPM10   = "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c10.pdf"
CFR414  = "https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-B/part-414/subpart-H/section-414.605"
MLN     = "https://www.cms.gov/training-education/medicare-learning-networkr-mln/compliance/medicare-provider-compliance-tips/ambulance-services"

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
MOBILITY_CLEAR    = 3   # mobility score at/above this = clearly necessary on the mobility axis
MONITORING_CLEAR  = 3   # monitoring score at/above this = clearly necessary on the monitoring axis
INDETERMINATE_MIN = 1   # any Group A signal below the clear thresholds = indeterminate

# The three prongs of the bed-confined test. [BPM10] 10.2.3 requires ALL THREE to be met, and
# states bed confinement is neither sufficient nor necessary by itself - it is one factor.
BED_CONFINED_PRONGS = ["bed_confined", "mobility_deficit", "cannot_sit"]  # cite: [BPM10] 10.2.3

# 3. Scope - non-emergent ground only

# Rideshare, air, and emergent trips are excluded; they do not carry the [BPM10] 10.2.1
# non-emergency documentation requirement. Adjust the filter to the real column values.

from pyspark.sql import functions as F

df = spark.table(SOURCE_TABLE)

ground_los = [
    "Basic life support", "Advanced life support", "Critical care transport",
    "Basic Life Support - Concierge", "Team", "Ambulatory",
]  # covered ground levels per [414.605] / [410.40](c)
df = df.filter(F.lower(F.col(LOS_COL)).isin([s.lower() for s in ground_los]))
# emergent / rideshare exclusion - adapt predicate to the real flag column
if "TripType" in df.columns:
    df = df.filter(~F.lower(F.col("TripType")).rlike("emergen|rideshare|air|fixed wing|rotor"))

df = df.withColumn("_text", F.lower(F.coalesce(F.col(FREE_TEXT_COL), F.lit(""))))
df = df.withColumn("_has_text", F.length(F.trim(F.col("_text"))) > 0)

print("Orders in scope:", df.count())

# 4. Concept tagging

# One boolean column per concept, the term pattern matched against ClinicalData with `rlike`.
# The CMS citation for each pattern is in the CONCEPTS table above.

for name, axis, weight, group, basis, ref, url, pattern in CONCEPTS:
    df = df.withColumn(f"c_{name}", F.col("_text").rlike(pattern) & F.col("_has_text"))

# 5. Scores, classification, confidence, and level-of-service recommendation

# All derived from the config - no hand-coded per-concept logic below this point.

def axis_score(axis):
    terms = [F.when(F.col(f"c_{n}"), F.lit(w)).otherwise(F.lit(0))
             for n, a, w, *_ in CONCEPTS if a == axis]
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr

# mobility_score: sum of mobility-axis concept weights. Basis: [BPM10] 10.2.1 / 10.2.3.
df = df.withColumn("mobility_score",   axis_score("mobility"))
# monitoring_score: sum of monitoring-axis concept weights. Basis: [414.605].
df = df.withColumn("monitoring_score", axis_score("monitoring"))

# named vs inferred / filler counts, for confidence and the clear threshold.
# named = concept explicit in CMS text; only named concepts can reach clearly_necessary alone.
named_a = [n for n, a, w, g, b, *_ in CONCEPTS if g == "A" and b == "named"]
df = df.withColumn("named_a_hits", sum([F.col(f"c_{n}").cast("int") for n in named_a]))
df = df.withColumn("has_named_a", F.col("named_a_hits") > 0)
df = df.withColumn("any_a", (F.col("mobility_score") + F.col("monitoring_score")) > 0)

# bed_confined_full: all three [BPM10] 10.2.3 prongs present = the full objective standard.
prong_expr = F.col(f"c_{BED_CONFINED_PRONGS[0]}")
for p in BED_CONFINED_PRONGS[1:]:
    prong_expr = prong_expr & F.col(f"c_{p}")
df = df.withColumn("bed_confined_full", prong_expr)  # cite: [BPM10] 10.2.3

filler_cols = [f"c_{n}" for n, a, *_ in CONCEPTS if a == "filler"]
df = df.withColumn("any_filler", sum([F.col(c).cast("int") for c in filler_cols]) > 0)

# necessity_class: clearly_necessary / clearly_not_necessary / indeterminate.
# clearly_necessary requires the full [BPM10] 10.2.3 test, OR a named CMS concept at the clear
#   threshold - inferred concepts alone cannot reach it.
# clearly_not_necessary maps to [RSN] AM600 territory: no documented reason other transport
#   was contraindicated (empty, or filler only).
df = df.withColumn(
    "necessity_class",
    F.when(
        F.col("bed_confined_full") |
        (F.col("has_named_a") &
         ((F.col("mobility_score") >= MOBILITY_CLEAR) | (F.col("monitoring_score") >= MONITORING_CLEAR))),
        F.lit("clearly_necessary"),          # cite: [BPM10] 10.2.3 / 10.2.1 ; [414.605]
    ).when(
        ~F.col("any_a") & (~F.col("_has_text") | F.col("any_filler")),
        F.lit("clearly_not_necessary"),      # cite: [RSN] AM600 ; [MLN]
    ).otherwise(F.lit("indeterminate")),
)

# indeterminate_reason: text_unmatched (text present, no term matched - the LLM target) vs
#   weak_or_inferred_only (some signal, below the clear threshold).
df = df.withColumn(
    "indeterminate_reason",
    F.when(F.col("necessity_class") != "indeterminate", F.lit(None))
     .when(F.col("_has_text") & ~F.col("any_a") & ~F.col("any_filler"), F.lit("text_unmatched"))
     .otherwise(F.lit("weak_or_inferred_only")),
)

# confidence in the classification itself (not a CMS construct - internal quality flag).
df = df.withColumn(
    "confidence",
    F.when(F.col("bed_confined_full"), F.lit("high"))
     .when((F.col("named_a_hits") >= 2), F.lit("high"))
     .when(~F.col("_has_text"), F.lit("high"))
     .when(F.col("named_a_hits") == 1, F.lit("medium"))
     .when(F.col("any_filler") & ~F.col("any_a"), F.lit("medium"))
     .otherwise(F.lit("low")),
)

# recommended_los: level of service implied by the monitoring axis, per [414.605] definitions.
#   ventilator/suctioning -> CCT/SCT ; multiple monitoring or IV meds -> ALS ; else BLS.
#   Descriptive only until CMS BLS/ALS eligibility criteria are confirmed - not a billing call.
df = df.withColumn(
    "recommended_los",
    F.when(F.col("c_ventilator") | F.col("c_suctioning"), F.lit("CCT/SCT"))          # cite: [414.605] SCT/ALS2
     .when((F.col("monitoring_score") >= 2) | F.col("c_iv_medication"), F.lit("ALS"))  # cite: [414.605] ALS1/ALS2
     .when(F.col("mobility_score") >= INDETERMINATE_MIN, F.lit("BLS"))               # cite: [414.605] BLS
     .otherwise(F.lit("none/indeterminate")),
)

# 6. Summary outputs (display only - nothing is written)

# Necessity distribution
display(
    df.groupBy("necessity_class")
      .agg(F.count("*").alias("orders"))
      .orderBy(F.desc("orders"))
)

# Indeterminate breakdown - text_unmatched is the LLM opportunity ([BPM10] 10.2.1 reason present
# but not captured by term matching)
display(
    df.filter(F.col("necessity_class") == "indeterminate")
      .groupBy("indeterminate_reason")
      .agg(F.count("*").alias("orders"))
)

# Necessity by customer
display(
    df.groupBy(CUSTOMER_COL, "necessity_class")
      .agg(F.count("*").alias("orders"))
      .orderBy(CUSTOMER_COL, "necessity_class")
)

# Transport appropriateness: recommended vs requested level of service ([414.605])
display(
    df.groupBy(LOS_COL, "recommended_los")
      .agg(F.count("*").alias("orders"))
      .orderBy(F.desc("orders"))
)

# 7. Example order texts per category

# For the 24 July action item: sample real free-text per class so stakeholders can see how each
# category is defined. Sampled, de-identified review only - not written back.

for cls in ["clearly_necessary", "indeterminate", "clearly_not_necessary"]:
    print("=" * 70)
    print(cls.upper())
    display(
        df.filter(F.col("necessity_class") == cls)
          .select(ORDER_ID_COL, LOS_COL, "mobility_score", "monitoring_score",
                  "confidence", "indeterminate_reason", FREE_TEXT_COL)
          .limit(15)
    )

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
