# ================================================================================================
# IDEN-45868 — SPONSOR IDENTITIES: STATUS QUO MATCHING OF MULTIPLE I-865 FILINGS
#
# Question (from the business): If sponsors become their own identities, how likely is it that
# a sponsor's multiple I-865 filings end up matched into one identity — using the matching we
# have today — and what changes if the threshold is lowered?
#
# Approach: Use production results, not a re-run of the model.
# - I-865 filings come from C3 (ecc3conods_mod.petapp, FORM_NUMBER = 'I865'). The filer — the
# sponsor — is stored in the BEN_* columns.
# - Each ingested I-865 already has a scored record in pcis_metadata.mv_es_mlaas_scores, with
# an identityId. We join on the receipt number at the end of _id.
# - SSN is treated as ground truth for "same sponsor" (the business confirmed SSN is the only
# reliable sponsor identifier).
#
# How to run: Run the cells top to bottom. Each chart displays inline and is saved as a PNG to
# results/ next to this notebook as soon as it is drawn; the Excel workbook is saved there at the end.
#
# Copy each section (between the ==== lines) into its own notebook cell and run them in order.
# ================================================================================================


# ================================================================================================
# 1. CONFIGURATION
# Tables, thresholds, and policies used throughout. Change these here only.
# ================================================================================================
# Purpose: set the tables and thresholds the whole analysis uses.
# Thresholds come from "2026 09 - All Systems Threshold Combinations and Analysis" (Confluence).

import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

CATALOG = "eciscor_prod"
SCORES_TABLE = f"{CATALOG}.pcis_metadata.mv_es_mlaas_scores"
RECEIPT_REGEX = "_([A-Za-z]{3}[0-9]{10})$"

CURRENT_THRESHOLD = 0.98
SYSTEM_THRESHOLDS = {
    "ar11": 0.98, "camino": 0.98, "cis2": 0.95, "c3": 0.98,
    "c4": 0.98, "cpms": 0.90, "elis": 0.95, "global": 0.95,
}
CUSTOM_EXCEPTIONS = {
    frozenset({"cpms", "cis2"}): 0.95,
    frozenset({"cpms", "elis"}): 0.95,
}
POLICIES = ["current", "top_down", "bottom_up", "custom", "flat_0.95", "flat_0.90"]
THRESHOLD_LINES = {"0.90": 0.90, "0.95": 0.95, "0.98 (current)": 0.98}


# ================================================================================================
# 2. HELPERS
# Where results are saved, how charts are stored, and how a pair's threshold is chosen under
# each policy.
# ================================================================================================
# Purpose: helper functions for saving results/charts and picking the threshold for a pair.
# Policies: current = 0.98 everywhere; top_down = stricter of the two systems; bottom_up = looser of the two;
# custom = bottom_up but CPMS vs CIS2/ELIS stays at 0.95; flat_x = one threshold for every pair.

RESULTS_DIR = Path("/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/sponsor_analysis/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
CHARTS = {}
TABLES = {}


def save_chart(fig, name):
    path = RESULTS_DIR / f"{name}_{RUN_STAMP}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    CHARTS[name] = path
    print("chart saved to", path)
    plt.show()


def label_bars(ax, fmt="{:,.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h and not np.isnan(h):
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=8)


def pair_threshold(policy, sys_a, sys_b):
    ta = SYSTEM_THRESHOLDS.get(sys_a, CURRENT_THRESHOLD)
    tb = SYSTEM_THRESHOLDS.get(sys_b, CURRENT_THRESHOLD)
    if policy == "current":
        return CURRENT_THRESHOLD
    if policy == "top_down":
        return max(ta, tb)
    if policy == "bottom_up":
        return min(ta, tb)
    if policy == "custom":
        return CUSTOM_EXCEPTIONS.get(frozenset({sys_a, sys_b}), min(ta, tb))
    if policy.startswith("flat_"):
        return float(policy.split("_")[1])
    raise ValueError(policy)


# ================================================================================================
# 3. BUILD THE DATA
# Three views: the I-865 sponsors, the latest stored score per record pair, and the two joined
# by receipt number.
# ================================================================================================
# Purpose: build the working datasets in Spark.
#   i865          - one row per I-865 receipt with the sponsor's name, DOB, and a cleaned SSN (ssn_key)
#   scores        - latest stored MLaaS score per (record, candidate) pair, with system and receipt pulled from the ids
#   i865_in_pcis  - each I-865 matched to its PCIS record and identityId (null if the I-865 was never scored)

I865_SQL = f"""
select
    upper(RECEIPT_NUMBER) as receipt_number,
    coalesce(BEN_FIRST_NAME, PET_FIRST_NAME) as first_name,
    coalesce(BEN_LAST_NAME, PET_LAST_NAME) as last_name,
    coalesce(cast(BEN_DATE_OF_BIRTH as string), cast(PET_DATE_OF_BIRTH as string)) as dob,
    coalesce(BEN_SSN, PET_SSN) as ssn
from {CATALOG}.ecc3conods_mod.petapp
where FORM_NUMBER = 'I865'
qualify row_number() over (partition by RECEIPT_NUMBER order by DWH_UPD_DT desc) = 1
"""

spark.sql(f"""
    create or replace temp view i865 as
    select
        receipt_number,
        upper(trim(first_name)) as first_name,
        upper(trim(last_name)) as last_name,
        date_format(coalesce(
            try_to_timestamp(trim(dob), 'yyyyMMdd'),
            try_to_timestamp(trim(dob), 'yyyy-MM-dd'),
            try_to_timestamp(trim(dob), 'MM/dd/yyyy'),
            try_to_timestamp(trim(dob))
        ), 'yyyy-MM-dd') as dob_norm,
        case when regexp_replace(ssn, '[^0-9]', '') rlike '^[0-9]{{7,9}}$'
              and lpad(regexp_replace(ssn, '[^0-9]', ''), 9, '0') not in ('000000000', '111111111', '123456789', '999999999')
              and substr(lpad(regexp_replace(ssn, '[^0-9]', ''), 9, '0'), 1, 3) not in ('000', '666')
             then lpad(regexp_replace(ssn, '[^0-9]', ''), 9, '0') end as ssn_key
    from ({I865_SQL})
""")

spark.sql(f"""
    create or replace temp view scores as
    select
        identityId as identity_id,
        _id as record_key,
        candidateSourceId as candidate_key,
        cast(candidateScore as double) as score,
        timestamp,
        rule_name,
        lower(split(_id, '_')[0]) as record_system,
        lower(split(candidateSourceId, '_')[0]) as candidate_system,
        upper(regexp_extract(_id, '{RECEIPT_REGEX}', 1)) as record_receipt,
        upper(regexp_extract(candidateSourceId, '{RECEIPT_REGEX}', 1)) as candidate_receipt
    from {SCORES_TABLE}
    qualify row_number() over (partition by _id, candidateSourceId order by timestamp desc) = 1
""")

spark.sql("""
    create or replace temp view record_identity as
    select record_key, record_system, record_receipt, identity_id, rule_name
    from scores
    qualify row_number() over (partition by record_key order by timestamp desc) = 1
""")

spark.sql("""
    create or replace temp view i865_in_pcis as
    select f.*, r.record_key, r.record_system, r.identity_id, r.rule_name
    from i865 f
    left join record_identity r on r.record_receipt = f.receipt_number
""")
print("views ready: i865, scores, record_identity, i865_in_pcis")


# ================================================================================================
# 4. COVERAGE — HOW MUCH OF THE I-865 DATA CAN WE ANALYZE?
# Counts I-865 receipts, how many have an SSN and DOB, and how many were found in the PCIS
# scores table. If found_in_scores is near zero, I-865 filers were not ingested as parties and
# this approach cannot be used.
# ================================================================================================
# Purpose: show how complete the I-865 data is and how much of it exists in PCIS today.

cov = spark.sql("""
    select
        count(distinct receipt_number) as i865_receipts,
        count(distinct case when ssn_key is not null then receipt_number end) as with_ssn,
        count(distinct case when dob_norm is not null then receipt_number end) as with_dob,
        count(distinct case when record_key is not null then receipt_number end) as found_in_scores,
        count(distinct case when record_key is not null and ssn_key is not null then receipt_number end) as found_with_ssn,
        count(distinct ssn_key) as distinct_sponsors,
        count(distinct identity_id) as distinct_identities
    from i865_in_pcis
""").toPandas().T.reset_index()
cov.columns = ["metric", "value"]
cov["value"] = cov["value"].astype(float)
total = cov.loc[cov.metric == "i865_receipts", "value"].iloc[0] or 1.0
cov["pct_of_receipts"] = (cov["value"] / total).round(4)
cov.loc[cov.metric.isin(["distinct_sponsors", "distinct_identities"]), "pct_of_receipts"] = np.nan
TABLES["coverage"] = cov
display(cov)

bars = cov[cov.metric.isin(["i865_receipts", "with_ssn", "with_dob", "found_in_scores", "found_with_ssn"])]
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(bars["metric"], bars["value"], color=["#4c72b0", "#55a868", "#8172b2", "#c44e52", "#dd8452"])
label_bars(ax)
ax.set_title("I-865 receipts: data completeness and presence in PCIS scores")
ax.set_ylabel("receipts")
save_chart(fig, "01_coverage")


# ================================================================================================
# 5. WHICH MODEL VERSION SCORED THESE RECORDS?
# The scores table holds results from more than one model version (e.g. dedup_model_v1.0.0 and
# v1.0.1). This shows how the I-865 records split across versions, so results can be read in
# context.
# ================================================================================================
# Purpose: show which dedup model version produced the stored scores for I-865 records.

versions = spark.sql("""
    select coalesce(rule_name, 'not in scores') as rule_name, count(distinct receipt_number) as i865_receipts
    from i865_in_pcis group by 1 order by 2 desc
""").toPandas()
TABLES["model_versions"] = versions
display(versions)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(versions["rule_name"], versions["i865_receipts"], color="#4c72b0")
for i, v in enumerate(versions["i865_receipts"]):
    ax.annotate(f"{v:,.0f}", (v, i), va="center", fontsize=8)
ax.set_title("I-865 receipts by model version")
ax.set_xlabel("receipts")
ax.invert_yaxis()
save_chart(fig, "02_model_versions")


# ================================================================================================
# 6. HOW MANY I-865S DOES EACH SPONSOR FILE?
# Groups I-865s by sponsor (SSN). Sponsors with 2+ filings are the population the business
# question is about.
# ================================================================================================
# Purpose: one row per sponsor (SSN) with how many I-865s they filed, how many are in PCIS,
# how many identities those records sit in, and the fiscal year of their first filing.

spark.sql("""
    create or replace temp view i865_per_sponsor as
    select
        ssn_key,
        count(distinct receipt_number) as i865_filings,
        count(distinct case when record_key is not null then receipt_number end) as filings_in_pcis,
        count(distinct identity_id) as identities,
        min(case when record_key is not null then substr(receipt_number, 4, 2) end) as first_fy
    from i865_in_pcis
    where ssn_key is not null
    group by 1
""")

filings_dist = spark.sql("""
    select least(i865_filings, 10) as i865_filings_per_sponsor, count(*) as sponsors
    from i865_per_sponsor group by 1 order by 1
""").toPandas()
filings_dist["i865_filings_per_sponsor"] = filings_dist["i865_filings_per_sponsor"].astype(int).astype(str).replace("10", "10+")
TABLES["i865_per_sponsor"] = filings_dist
display(filings_dist)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(filings_dist["i865_filings_per_sponsor"], filings_dist["sponsors"], color="#55a868")
label_bars(ax)
ax.set_title("I-865 filings per sponsor (by SSN)")
ax.set_xlabel("I-865 filings")
ax.set_ylabel("sponsors")
save_chart(fig, "03_filings_per_sponsor")


# ================================================================================================
# 7. STATUS QUO — ARE A SPONSOR'S MULTIPLE I-865S IN ONE IDENTITY TODAY?
# This is the core answer. For sponsors with 2+ I-865s in PCIS: *linked* = all their records
# share one identityId; *split* = spread across several identities. The second chart shows how
# badly split sponsors are fragmented.
# ================================================================================================
# Purpose: measure today's matching outcome for sponsors with 2+ I-865s in PCIS,
# and how many identities each sponsor's records are spread across.

status = spark.sql("""
    select
        count(*) as sponsors_with_i865,
        sum(case when i865_filings > 1 then 1 else 0 end) as sponsors_2plus_i865,
        sum(case when filings_in_pcis > 1 then 1 else 0 end) as sponsors_2plus_in_pcis,
        sum(case when filings_in_pcis > 1 and identities = 1 then 1 else 0 end) as linked_one_identity,
        sum(case when filings_in_pcis > 1 and identities > 1 then 1 else 0 end) as split_across_identities
    from i865_per_sponsor
""").toPandas().T.reset_index()
status.columns = ["metric", "value"]
status["value"] = status["value"].astype(float)
base = status.loc[status.metric == "sponsors_2plus_in_pcis", "value"].iloc[0] or 1.0
status["pct_of_2plus_in_pcis"] = np.where(status.metric.isin(["linked_one_identity", "split_across_identities"]),
                                          (status["value"] / base).round(4), np.nan)
TABLES["status_quo"] = status
display(status)

identities_dist = spark.sql("""
    select least(identities, 5) as identities_per_sponsor, count(*) as sponsors
    from i865_per_sponsor where filings_in_pcis > 1
    group by 1 order by 1
""").toPandas()
identities_dist["identities_per_sponsor"] = identities_dist["identities_per_sponsor"].astype(int).astype(str).replace("5", "5+")
TABLES["identities_per_sponsor"] = identities_dist
display(identities_dist)

linked = status.loc[status.metric == "linked_one_identity", "value"].iloc[0]
split = status.loc[status.metric == "split_across_identities", "value"].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].pie([linked, split], labels=["linked (1 identity)", "split (2+ identities)"], autopct="%1.1f%%",
            colors=["#55a868", "#c44e52"], startangle=90)
axes[0].set_title("Sponsors with 2+ I-865s in PCIS: status quo")
axes[1].bar(identities_dist["identities_per_sponsor"], identities_dist["sponsors"], color="#8172b2")
label_bars(axes[1])
axes[1].set_title("Identities per sponsor")
axes[1].set_xlabel("identities holding the sponsor's I-865 records")
axes[1].set_ylabel("sponsors")
save_chart(fig, "04_status_quo")


# ================================================================================================
# 8. HAS MATCHING GOTTEN BETTER OR WORSE OVER TIME?
# Linked rate for sponsors with 2+ I-865s, grouped by the fiscal year of their first I-865
# (taken from the receipt number).
# ================================================================================================
# Purpose: show the linked rate by the fiscal year of the sponsor's first I-865, to spot older data that matches worse.

by_year = spark.sql("""
    select first_fy, count(*) as sponsors,
           avg(case when identities = 1 then 1.0 else 0.0 end) as linked_rate
    from i865_per_sponsor
    where filings_in_pcis > 1 and first_fy rlike '^[0-9]{2}$'
    group by 1
""").toPandas()
by_year["fiscal_year"] = by_year["first_fy"].astype(int).map(lambda y: 1900 + y if y > 50 else 2000 + y)
by_year = by_year.sort_values("fiscal_year")[["fiscal_year", "sponsors", "linked_rate"]].reset_index(drop=True)
by_year["linked_rate"] = by_year["linked_rate"].round(4)
TABLES["linked_by_year"] = by_year
display(by_year)

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(by_year["fiscal_year"].astype(str), by_year["sponsors"], color="#c9d6ea", label="sponsors")
ax1.set_ylabel("sponsors")
ax2 = ax1.twinx()
ax2.plot(by_year["fiscal_year"].astype(str), by_year["linked_rate"], color="#4c72b0", marker="o", label="linked rate")
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("linked rate")
ax1.set_title("Linked rate by fiscal year of first I-865")
ax1.tick_params(axis="x", rotation=45)
save_chart(fig, "05_linked_by_year")


# ================================================================================================
# 9. SPLIT SPONSORS — COULD A DIFFERENT THRESHOLD HAVE LINKED THEM?
# For every pair of same-SSN I-865 records in different identities, looks up the stored score
# between them.
# - *Never candidates*: search never returned one record for the other, so no threshold change
# can link them.
# - *Scored*: the stored score decides whether each policy would have linked them.
# ================================================================================================
# Purpose: for same-sponsor (same SSN) record pairs, find the stored score between them and test each threshold policy.

pairs = spark.sql("""
    with recs as (
        select distinct ssn_key, record_key, record_system, identity_id
        from i865_in_pcis
        where ssn_key is not null and record_key is not null
    ),
    p as (
        select a.ssn_key, a.record_key as key_a, b.record_key as key_b,
               a.record_system as sys_a, b.record_system as sys_b,
               a.identity_id = b.identity_id as same_identity
        from recs a
        join recs b on a.ssn_key = b.ssn_key and a.record_key < b.record_key
    )
    select p.*, greatest(s1.score, s2.score) as stored_score
    from p
    left join scores s1 on s1.record_key = p.key_a and s1.candidate_key = p.key_b
    left join scores s2 on s2.record_key = p.key_b and s2.candidate_key = p.key_a
""").toPandas()
pairs["same_identity"] = pairs["same_identity"].astype(bool)
pairs["stored_score"] = pairs["stored_score"].astype(float)
pairs["has_stored_score"] = pairs["stored_score"].notna()
split_pairs = pairs[~pairs["same_identity"]].copy()

rows = []
for p in POLICIES:
    thr = np.array([pair_threshold(p, a, b) for a, b in zip(split_pairs["sys_a"], split_pairs["sys_b"])])
    hit = split_pairs["has_stored_score"].to_numpy() & (split_pairs["stored_score"].fillna(-1).to_numpy() >= thr)
    rows.append({"policy": p, "split_pairs": len(split_pairs),
                 "never_candidates": int((~split_pairs["has_stored_score"]).sum()),
                 "scored": int(split_pairs["has_stored_score"].sum()),
                 "would_link": int(hit.sum()),
                 "would_link_rate": round(hit.mean(), 4) if len(split_pairs) else np.nan})
split_by_policy = pd.DataFrame(rows)
TABLES["split_by_policy"] = split_by_policy
display(split_by_policy)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
nc = int((~split_pairs["has_stored_score"]).sum())
sc = int(split_pairs["has_stored_score"].sum())
axes[0].bar(["never candidates", "scored"], [nc, sc], color=["#c44e52", "#4c72b0"])
label_bars(axes[0])
axes[0].set_title("Split same-sponsor pairs: were they ever compared?")
axes[0].set_ylabel("pairs")

scored_vals = split_pairs.loc[split_pairs["has_stored_score"], "stored_score"]
axes[1].hist(scored_vals, bins=np.linspace(0, 1, 41), color="#8172b2")
for name, t in THRESHOLD_LINES.items():
    axes[1].axvline(t, linestyle="--", linewidth=1, label=name, color={"0.90": "#dd8452", "0.95": "#55a868"}.get(name, "#c44e52"))
axes[1].legend(fontsize=8)
axes[1].set_title("Stored scores of split same-sponsor pairs")
axes[1].set_xlabel("stored score")
axes[1].set_ylabel("pairs")

axes[2].bar(split_by_policy["policy"], split_by_policy["would_link"], color="#55a868")
label_bars(axes[2])
axes[2].set_title("Split pairs each policy would link")
axes[2].set_ylabel("pairs")
axes[2].tick_params(axis="x", rotation=30)
save_chart(fig, "06_split_pairs")

TABLES["same_sponsor_pairs"] = pairs.drop(columns=["ssn_key"])


# ================================================================================================
# 10. RISK OF LOWERING — WOULD DIFFERENT SPONSORS START MERGING?
# Stored scores ≥ 0.90 between I-865 records with different SSNs. These are merges each policy
# would allow — the name + DOB risk raised on the call.
# ================================================================================================
# Purpose: count different-sponsor (different SSN) record pairs each policy would merge, and show their scores.

risk = spark.sql("""
    with recs as (
        select distinct ssn_key, record_key from i865_in_pcis
        where ssn_key is not null and record_key is not null
    )
    select s.record_key, s.candidate_key, s.record_system, s.candidate_system, s.score
    from scores s
    join recs a on a.record_key = s.record_key
    join recs b on b.record_key = s.candidate_key
    where a.ssn_key <> b.ssn_key and s.score >= 0.90
""").toPandas()
risk["score"] = risk["score"].astype(float)

rows = []
for p in POLICIES:
    thr = np.array([pair_threshold(p, a, b) for a, b in zip(risk["record_system"], risk["candidate_system"])])
    rows.append({"policy": p, "different_ssn_pairs_matched": int((risk["score"].to_numpy() >= thr).sum())})
risk_by_policy = pd.DataFrame(rows)
TABLES["lowering_risk"] = risk_by_policy
TABLES["different_ssn_pairs"] = risk
display(risk_by_policy)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(risk_by_policy["policy"], risk_by_policy["different_ssn_pairs_matched"], color="#c44e52")
label_bars(axes[0])
axes[0].set_title("Different-SSN pairs each policy would merge")
axes[0].set_ylabel("pairs")
axes[0].tick_params(axis="x", rotation=30)

axes[1].hist(risk["score"], bins=np.linspace(0.90, 1.0, 21), color="#c44e52")
for name, t in THRESHOLD_LINES.items():
    axes[1].axvline(t, linestyle="--", linewidth=1, label=name, color={"0.90": "#dd8452", "0.95": "#55a868"}.get(name, "#4c72b0"))
axes[1].legend(fontsize=8)
axes[1].set_title("Scores of different-SSN pairs (>= 0.90)")
axes[1].set_xlabel("stored score")
axes[1].set_ylabel("pairs")
save_chart(fig, "07_lowering_risk")


# ================================================================================================
# 11. TRADE-OFF BY POLICY
# Puts the gain (split same-sponsor pairs that would link) next to the risk (different-sponsor
# pairs that would merge) for each policy.
# ================================================================================================
# Purpose: compare gain vs. risk for every threshold policy in one view.

tradeoff = split_by_policy[["policy", "would_link"]].merge(risk_by_policy, on="policy")
tradeoff.columns = ["policy", "same_sponsor_pairs_linked (gain)", "different_sponsor_pairs_merged (risk)"]
TABLES["tradeoff"] = tradeoff
display(tradeoff)

x = np.arange(len(tradeoff))
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - 0.2, tradeoff.iloc[:, 1], width=0.4, color="#55a868", label="gain: same-sponsor pairs linked")
ax.bar(x + 0.2, tradeoff.iloc[:, 2], width=0.4, color="#c44e52", label="risk: different-sponsor pairs merged")
label_bars(ax)
ax.set_xticks(x)
ax.set_xticklabels(tradeoff["policy"])
ax.set_ylabel("pairs")
ax.set_title("Gain vs. risk by threshold policy")
ax.legend()
save_chart(fig, "08_tradeoff")


# ================================================================================================
# 12. KEY FINDINGS AND EXPORT
# Plain-language summary built from the numbers above, then every table and chart is written
# to one Excel workbook (charts on their own sheet) plus PNG files in results/.
# ================================================================================================
# Purpose: write the headline findings in plain language and save all tables and charts.

def metric(df, name):
    return float(df.loc[df.metric == name, "value"].iloc[0])

receipts = metric(cov, "i865_receipts")
found = metric(cov, "found_in_scores")
multi = metric(status, "sponsors_2plus_in_pcis")
linked_n = metric(status, "linked_one_identity")
split_n = metric(status, "split_across_identities")
never = int((~split_pairs["has_stored_score"]).sum())
gain = split_by_policy.set_index("policy")["would_link"]
loss = risk_by_policy.set_index("policy")["different_ssn_pairs_matched"]

findings = pd.DataFrame({"finding": [
    f"{found:,.0f} of {receipts:,.0f} I-865 receipts ({found / max(receipts, 1):.1%}) have a scored record in PCIS.",
    f"{multi:,.0f} sponsors have 2+ I-865s in PCIS; {linked_n / max(multi, 1):.1%} are in one identity today and {split_n / max(multi, 1):.1%} are split.",
    f"{len(split_pairs):,} same-sponsor record pairs are split; {never:,} ({never / max(len(split_pairs), 1):.1%}) were never compared, so no threshold change can link them.",
    f"Lowering to 0.95 would link {gain['flat_0.95']:,} split pairs and merge {loss['flat_0.95']:,} different-sponsor pairs.",
    f"Lowering to 0.90 would link {gain['flat_0.90']:,} split pairs and merge {loss['flat_0.90']:,} different-sponsor pairs.",
]})
TABLES = {"key_findings": findings, **TABLES}
pd.set_option("display.max_colwidth", None)
display(findings)

config = pd.DataFrame(
    [["run time", RUN_STAMP], ["scores table", SCORES_TABLE], ["receipt regex on _id", RECEIPT_REGEX],
     ["current threshold", CURRENT_THRESHOLD], ["I-865 source", I865_SQL.strip()]]
    + [[f"threshold {k}", v] for k, v in SYSTEM_THRESHOLDS.items()]
    + [[f"custom {' / '.join(sorted(k))}", v] for k, v in CUSTOM_EXCEPTIONS.items()],
    columns=["setting", "value"])
TABLES["config"] = config

from openpyxl.drawing.image import Image as XLImage

def write_workbook(path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in TABLES.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
        ws = writer.book.create_sheet("charts", 1)
        row = 1
        for name, chart_path in CHARTS.items():
            ws.cell(row=row, column=1, value=name)
            img = XLImage(str(chart_path))
            img.width, img.height = img.width * 0.6, img.height * 0.6
            ws.add_image(img, f"A{row + 1}")
            row += int(img.height / 20) + 4


xlsx_path = RESULTS_DIR / f"sponsor_i865_status_quo_{RUN_STAMP}.xlsx"
try:
    write_workbook(xlsx_path)
except OSError:
    local_copy = Path("/tmp") / xlsx_path.name
    write_workbook(local_copy)
    shutil.copy(local_copy, xlsx_path)
print("workbook saved to", xlsx_path)
