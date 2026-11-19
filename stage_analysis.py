# Databricks notebook source
# ================================================================================================
# SPONSOR MATCHING STUDY - 10M ELIS SPONSOR RECORDS IN STAGE
# Builds on IDEN-45868 (status quo for sponsors, measured on stored prod MLaaS scores).
#
# Background (sponsor matching study planning call):
#   - The pipeline's splitter now splits sponsors off the beneficiary party record, the same way
#     petitioners are split. Each sponsor becomes its own party record. Its party source ID is the
#     beneficiary's party source ID plus "SPON-<n>" (n = 1, 2, 3 ... per sponsor on the record).
#   - Case data (receipt, form type) is copied from the beneficiary record. Only the person data
#     and the filing role type (sponsor) are new. Sponsors are NOT linked to their beneficiary on
#     purpose: the study measures what matches organically.
#   - The splitter passes every sponsor record through without a sparse check. Dedup's sparse
#     checker was not changed, so sparse sponsors should show up flagged sparse in the party index.
#   - Stage was resynced from prod on 11/05 (snapshot runbook) before the large load.
#   - 10M beneficiary records with sponsor data were pushed through stage
#     (list: pcis_sandbox.elis_sponsor_testing_1112, built from staging_v2_elis_sponsor_details).
#
# What this notebook measures, in order:
#   Part 1  What was sent: the 10M beneficiary keys, their sponsors, field completeness, and how
#           many sponsors appear on more than one receipt (the ones that should link).
#   Part 2  What the pipeline received: the generated payloads for the run, every sponsor in the
#           sponsors array (not only the first), and their completeness.
#   Part 3  What landed in the stage party index: sponsor party records found vs expected, sparse
#           flag, filing role type, and records with no identity.
#   Part 4  How stage resolved them: identity composition, sponsors on 2+ receipts linked vs split,
#           sponsors joining the same person's existing (non-sponsor) records, and merge risks
#           (2+ SSNs in one identity, sponsor merged with the beneficiary on its own receipt).
#   Part 5  Comparison with the IDEN-45868 status quo, review samples, findings, Excel export.
#
# Stage data is read from the OpenSearch dump tables in Databricks (Harness "pcis_elasticsearch
# dump" job). Section 6 checks the dump contains sponsor party records; if it predates the sponsor
# load, Parts 3-4 stop with a message and Parts 1-2 still run.
#
# SSN is the "same sponsor" key. Exported tables hold counts and record keys only: no SSNs, names
# or DOBs. The notebook only reads tables; it creates none. Intermediates are cached in memory as
# temporary views for this session. The only output is one Excel workbook in RESULTS_DIR.
# ================================================================================================

# COMMAND ----------

# ================================================================================================
# 1. CONFIGURATION
# Leave STAGE_PARTY_TABLE / STAGE_IDENTITY_TABLE as None to auto-discover the dump tables
# (section 6 prints what it picked). DETAIL_OVERRIDES and PARTY_OVERRIDES fix a field when the
# auto-mapping in sections 3 and 7 picks the wrong one, for example:
#   DETAIL_OVERRIDES = {"ssn": "sponsor_ssn"}                       (column of the details table)
#   PARTY_OVERRIDES  = {"identity_id": "get_json_object(_source, '$.identityId')"}
# LOG_SPONSOR_PATHS are the JSON paths inside one element of party.sponsors in the payload
# (taken from the "check payloads sent for sponsors" cell of the IDEN-46817 notebook).
# ================================================================================================

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pyspark.sql import functions as F
from pyspark.sql import types as T

CATALOG = "eciscor_prod"
TEST_SET_TABLE = f"{CATALOG}.pcis_sandbox.elis_sponsor_testing_1112"
SPONSOR_DETAILS_TABLE = f"{CATALOG}.pcis_metadata.staging_v2_elis_sponsor_details"
GENERATE_LOG_TABLE = f"{CATALOG}.pcis_metadata.staging_v2_pcis_generate_log"
LOG_SOURCE_SYSTEM = "ELIS2"
LOG_RUN_IDS = [1070833185663608]          # None = every run for LOG_SOURCE_SYSTEM
LOG_DATA_COL = "_1.Data"

STAGE_PARTY_TABLE = None                  # e.g. f"{CATALOG}.pcis_metadata.elasticdump_parties"
STAGE_IDENTITY_TABLE = None               # e.g. f"{CATALOG}.pcis_metadata.elasticdump_identities"
STAGE_SCORES_TABLE = None                 # optional, same columns as mv_es_mlaas_scores

RESULTS_DIR = Path("/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/sponsor_analysis/results")

SPONSOR_ID_REGEX = r"SPON-?[0-9]+"                         # sponsor party records in stage
BASE_KEY_REGEX = r"([0-9]+_[0-9]+_[A-Za-z]{3}[0-9]{10})"   # id_for_key inside any party _id
RECEIPT_REGEX = r"([A-Za-z]{3}[0-9]{10})"
SSN_TYPE_RX = "SSN|SOCIAL"

DETAIL_OVERRIDES = {}
PARTY_OVERRIDES = {}
PARSE_ALL_PARTIES = True                  # SSN + identity for every party (needed for Part 4 joins)
SAMPLE_DOCS = 2000
MAX_RECORDS_PER_SSN = 200
REVIEW_SAMPLE_SIZE = 100

LOG_SPONSOR_PATHS = {
    "form_type": "$.formType.source.formType",
    "sponsor_type": "$.sponsorType.source.sponsorType",
    "qualified": "$.qualified.source.qualified",
    "identifiers": "$.identifiers",
    "identifier_value": "$.source.identifier",
    "identifier_type": "$.source.identifierTypeCd",
    "dob": "$.dobs[0].source.dob",
    "cob": "$.cobs[0].source.cobCd",
    "coc": "$.cocs[0].source.countryOfCitizenshipCd",
    "first_name": "$.name[0].source.firstName",
    "last_name": "$.name[0].source.lastName",
    "zip5": "$.address[0].source.address.zip5",
    "state": "$.address[0].source.address.state",
}

# IDEN-45868 status quo (prod, stored MLaaS scores, sponsors found through the I-865 receipt link).
# Edit if that notebook is rerun.
EARLIER_STATUS_QUO = {"sponsors_2plus_records": 761, "linked_one_identity": 726, "split_across_identities": 35}

CURRENT_THRESHOLD = 0.98

# COMMAND ----------

# ================================================================================================
# 2. HELPERS
# Saving, work tables, SQL builders for cleaning SSN / DOB / names (same rules as IDEN-45868),
# column auto-mapping for flat tables, and JSON path discovery for the OpenSearch dump documents.
# ================================================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
CHART_DIR = Path("/tmp") / f"sponsor10m_charts_{RUN_STAMP}"   # charts go into the workbook only
CHART_DIR.mkdir(parents=True, exist_ok=True)
CHARTS, TABLES, FINDINGS = {}, {}, []


def save_chart(fig, name):
    path = CHART_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    CHARTS[name] = path
    plt.show()


def label_bars(ax, fmt="{:,.0f}"):
    for p in ax.patches:
        h = p.get_height()
        if h and not np.isnan(h):
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=8)


def label_hbars(ax, fmt="{:,.0f}"):
    for p in ax.patches:
        w = p.get_width()
        if w and not np.isnan(w):
            ax.annotate(fmt.format(w), (w, p.get_y() + p.get_height() / 2), va="center", fontsize=8)


def work(df, name):
    """Cache an intermediate in memory and register it as a temporary view (no table is created)."""
    out = df.cache()
    out.count()
    out.createOrReplaceTempView(name)
    return out


def keep(name, df):
    if not isinstance(df, pd.DataFrame):
        df = df.toPandas()
    TABLES[name] = df
    display(df)
    return df


def one(sql):
    r = spark.sql(sql).collect()
    return r[0][0] if r else None


def ssn_sql(col):
    digits = f"regexp_replace(cast({col} as string), '[^0-9]', '')"
    padded = f"lpad({digits}, 9, '0')"
    return (f"case when {digits} rlike '^[0-9]{{7,9}}$' "
            f"and {padded} not in ('000000000', '111111111', '123456789', '999999999') "
            f"and substr({padded}, 1, 3) not in ('000', '666') then {padded} end")


def dob_sql(col):
    s = f"regexp_replace(trim(cast({col} as string)), '[.]0+$', '')"
    epoch = (f"case when {s} rlike '^-?[0-9]{{11,13}}$' then timestamp_millis(cast({s} as bigint)) "
             f"when {s} rlike '^-?[0-9]{{9,10}}$' then timestamp_seconds(cast({s} as bigint)) end")
    ts = (f"coalesce({epoch}, try_to_timestamp({s}, 'yyyyMMdd'), try_to_timestamp({s}, 'yyyy-MM-dd'), "
          f"try_to_timestamp({s}, 'MM/dd/yyyy'), try_to_timestamp({s}, 'M/d/yyyy'), try_to_timestamp({s}, 'yyyy/MM/dd'), "
          f"try_to_timestamp(substr({s}, 1, 10), 'yyyy-MM-dd'), try_to_timestamp({s}))")
    d = f"date_format({ts}, 'yyyy-MM-dd')"
    return f"case when {d} > '1900-01-01' and {d} <= date_format(current_date(), 'yyyy-MM-dd') then {d} end"


def clean_sql(col):
    return f"nullif(regexp_replace(upper(coalesce(cast({col} as string), '')), '[^A-Z]', ''), '')"


def pattern_sql(col):
    """Shape of a value with no content: digits -> 9, letters -> A (e.g. 9999-99-99)."""
    return f"coalesce(regexp_replace(regexp_replace(trim(cast({col} as string)), '[0-9]', '9'), '[A-Za-z]', 'A'), '(null)')"


def filled_sql(col):
    return f"nullif(trim(cast({col} as string)), '')"


def null_sql(col):
    return "cast(null as string)" if col is None else col


def consistency_sql(names, dobs):
    return (f"case when {names} = 0 then 'name missing' "
            f"when {names} = 1 and {dobs} = 1 then 'same name, same DOB' "
            f"when {names} = 1 and {dobs} = 0 then 'same name, no DOB' "
            f"when {names} = 1 and {dobs} > 1 then 'same name, DOB differs' "
            f"when {dobs} <= 1 then 'name differs' else 'name and DOB differ' end")


CONSISTENCY_ORDER = ["same name, same DOB", "same name, no DOB", "same name, DOB differs",
                     "name differs", "name and DOB differ", "name missing"]

FIELD_SPECS = {
    "ssn": r"(^|\.)(ssn|social_?security(_?(number|nbr|num|no))?)$",
    "first_name": r"(^|\.)(first(_?(name|nm))?|given_?name|frst_?nm)$",
    "middle_name": r"(^|\.)(middle(_?(name|nm))?|mid_?nm)$",
    "last_name": r"(^|\.)(last(_?(name|nm))?|family_?name|surname)$",
    "dob": r"(^|\.)(dob|date_?of_?birth|birth_?date|brth_?dt)$",
    "a_number": r"(^|\.)(a_?number|a_?nbr|alien_?(number|nbr)|anumber)$",
    "form_type": r"(^|\.)form(_?(type|number|nbr|nm|name|id))?$",
    "sponsor_type": r"sponsor_?type",
    "qualified": r"qualif",
    "receipt": r"(^|\.)receipt(_?(number|nbr|num|no))?$",
    "identity_id": r"(^|\.)identity_?id$",
    "sparse": r"sparse",
    "role": r"(^|\.)(filing_?)?role(_?type)?(_?(cd|code))?$",
    "system": r"(^|\.)(source_?)?system(_?(name|cd|code))?$",
}
TYPE_RX = r"(^|_)(type|type_?cd|type_?code|identifier_?type(_?cd)?|id_?type|kind|category)$"
VALUE_RX = r"^(value|number|nbr|num|identifier|id|val|text)$"


def map_columns(cols, fields, overrides, exclude_rx=r"id_for_key"):
    """Pick a column of a flat table for each field by name. Returns {field: column} and a report."""
    mapping, rows = {}, []
    for f in fields:
        if f in overrides:
            mapping[f] = overrides[f]
            rows.append({"field": f, "column": overrides[f], "how": "override"})
            continue
        hits = [c for c in cols if re.search(FIELD_SPECS[f], c.lower()) and not re.search(exclude_rx, c.lower())]
        hits.sort(key=len)
        if hits:
            mapping[f] = f"`{hits[0]}`"
            rows.append({"field": f, "column": hits[0], "how": "matched by name"})
            continue
        if f in ("ssn", "a_number"):
            val = next((c for c in cols if re.fullmatch(r"(identifier|identifier_?(value|nbr|number)|id_?value)", c.lower())), None)
            typ = next((c for c in cols if re.fullmatch(r"identifier_?type(_?(cd|code))?|id_?type(_?cd)?", c.lower())), None)
            if val and typ:
                rx = SSN_TYPE_RX if f == "ssn" else "ALIEN|A_?NUMBER|ANUMBER|A-NUMBER"
                mapping[f] = f"case when upper(`{typ}`) rlike '{rx}' then `{val}` end"
                rows.append({"field": f, "column": f"{val} where {typ} ~ {rx}", "how": "identifier + type columns"})
                continue
        mapping[f] = None
        rows.append({"field": f, "column": "", "how": "not found"})
    return mapping, pd.DataFrame(rows)


def _jp(path, key):
    return f"{path}.{key}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else f"{path}['{key}']"


def flatten_doc(obj, path="$", segs=(), out=None, arrays=None):
    """Leaf JSON paths of one document (first array element only) and every array of objects."""
    out = {} if out is None else out
    arrays = {} if arrays is None else arrays
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_doc(v, _jp(path, k), segs + (k,), out, arrays)
    elif isinstance(obj, list):
        items = [x for x in obj if x is not None]
        if items and isinstance(items[0], dict):
            arrays.setdefault(path, (segs, {}))
            sub_out = {}
            flatten_doc(items[0], "$", (), sub_out, {})
            arrays[path][1].update({k: v for k, v in sub_out.items()})
        if items:
            flatten_doc(items[0], f"{path}[0]", segs + ("[]",), out, arrays)
        else:
            out.setdefault(path, (segs, None))
    else:
        out.setdefault(path, (segs, obj))
    return out, arrays


def seg_text(segs):
    return ".".join(s for s in segs if s != "[]").lower()


def discover_paths(json_strings):
    """Union of leaf paths and object arrays over sample documents, with how often each appears."""
    leaves, arrays, n = {}, {}, 0
    for s in json_strings:
        try:
            d = json.loads(s)
        except Exception:
            continue
        n += 1
        out, arr = flatten_doc(d)
        for p, (segs, v) in out.items():
            e = leaves.setdefault(p, {"segs": segs, "docs": 0, "example_type": type(v).__name__})
            e["docs"] += 1
        for p, (segs, sub) in arr.items():
            e = arrays.setdefault(p, {"segs": segs, "children": {}, "docs": 0})
            e["children"].update({k: v[0] for k, v in sub.items()})
            e["docs"] += 1
    return leaves, arrays, n


def json_identifier_expr(src, arrays, exclude_rx):
    """SSN from an identifier list: [{type: 'SSN', value: '...'}] (type/value may sit under 'source')."""
    ranked = sorted(arrays.items(), key=lambda kv: (0 if "identif" in seg_text(kv[1]["segs"]) else 1, len(kv[0])))
    for path, info in ranked:
        if exclude_rx and re.search(exclude_rx, seg_text(info["segs"])):
            continue
        t = next((p for p, segs in info["children"].items() if re.search(TYPE_RX, (segs[-1] if segs else "").lower())), None)
        v = next((p for p, segs in info["children"].items()
                  if re.search(VALUE_RX, (segs[-1] if segs else "").lower()) and p != t), None)
        if t and v:
            e = (f"get_json_object(try_element_at(filter(from_json(get_json_object({src}, '{path}'), 'array<string>'), "
                 f"x -> upper(get_json_object(x, '{t}')) rlike '{SSN_TYPE_RX}'), 1), '{v}')")
            return e, f"{path}[{t} ~ {SSN_TYPE_RX}].{v}", (path, t)
    return None, None, None


def map_json(src, leaves, arrays, fields, overrides, exclude_rx=None):
    """Pick a JSON path for each field by name; SSN may come from an identifier list."""
    mapping, rows, id_type = {}, [], None
    for f in fields:
        if f in overrides:
            mapping[f] = overrides[f]
            rows.append({"field": f, "expression": overrides[f], "source_path": "", "how": "override"})
            continue
        hits = [(p, i) for p, i in leaves.items()
                if re.search(FIELD_SPECS[f], seg_text(i["segs"]))
                and not (exclude_rx and re.search(exclude_rx, seg_text(i["segs"])))]
        hits.sort(key=lambda x: (-x[1]["docs"], x[0].count("["), len(x[0])))
        if hits:
            p = hits[0][0]
            mapping[f] = f"get_json_object({src}, '{p}')"
            rows.append({"field": f, "expression": mapping[f], "source_path": p, "how": "matched by name"})
            continue
        if f == "ssn":
            e, src_path, id_type = json_identifier_expr(src, arrays, exclude_rx)
            if e:
                mapping[f] = e
                rows.append({"field": f, "expression": e, "source_path": src_path, "how": "identifier list"})
                continue
        mapping[f] = None
        rows.append({"field": f, "expression": "", "source_path": "", "how": "not found"})
    return mapping, pd.DataFrame(rows), id_type


def find_tables(schema, name_rx):
    try:
        rows = spark.sql(f"show tables in {schema}").collect()
    except Exception:
        return []
    return sorted(r["tableName"] for r in rows if re.search(name_rx, r["tableName"].lower()) and not r["isTemporary"])


def bar_chart(pdf, x, y, title, name, ylabel="", color="#4c72b0", rotate=0, horizontal=False):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if horizontal:
        ax.barh(pdf[x].astype(str), pdf[y], color=color)
        label_hbars(ax)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    else:
        ax.bar(pdf[x].astype(str), pdf[y], color=color)
        label_bars(ax)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=rotate)
    ax.set_title(title)
    save_chart(fig, name)


def pct(a, b):
    return (a / b) if b else float("nan")


def metric(df, name, col="value"):
    if df is None or df.empty or name not in set(df["metric"]):
        return float("nan")
    return float(df.loc[df["metric"] == name, col].iloc[0])


def long_form(pdf):
    out = pdf.T.reset_index()
    out.columns = ["metric", "value"]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def norm_key_sql(col):
    """Party key from the base key onward, so 'elis2_<key>SPON-1' and '<key>SPON-1' compare equal."""
    return f"coalesce(nullif(regexp_extract({col}, '([0-9]+_[0-9]+_[A-Za-z]{{3}}[0-9]{{10}}.*)$', 1), ''), {col})"

# COMMAND ----------

# ================================================================================================
# 3. PART 1 - WHAT WAS SENT: THE 10M TEST SET AND ITS SPONSORS
# The test set holds beneficiary keys (id_for_key). Each beneficiary key can carry several
# sponsors (one row per sponsor in the details table). Columns of the details table are mapped by
# name; check the mapping table and set DETAIL_OVERRIDES if a field is wrong or missing.
# ================================================================================================

detail_cols = spark.table(SPONSOR_DETAILS_TABLE).columns
print("details table columns:", detail_cols)
DCOL, detail_map = map_columns(detail_cols, ["ssn", "first_name", "middle_name", "last_name", "dob", "a_number",
                                             "sponsor_type", "qualified", "form_type"], DETAIL_OVERRIDES)
keep("p1_detail_field_map", detail_map)

sent = spark.sql(f"""
    select d.id_for_key as base_key,
           d.sponsor_id_for_key as sponsor_key,
           regexp_extract(d.id_for_key, '{RECEIPT_REGEX}', 1) as receipt,
           {ssn_sql(null_sql(DCOL['ssn']))} as ssn_key,
           {clean_sql(null_sql(DCOL['first_name']))} as first_name,
           {clean_sql(null_sql(DCOL['last_name']))} as last_name,
           {dob_sql(null_sql(DCOL['dob']))} as dob,
           {filled_sql(null_sql(DCOL['a_number']))} as a_number,
           {filled_sql(null_sql(DCOL['sponsor_type']))} as sponsor_type,
           lower(cast({null_sql(DCOL['qualified'])} as string)) as qualified,
           {filled_sql(null_sql(DCOL['form_type']))} as form_type
    from {SPONSOR_DETAILS_TABLE} d
    where d.id_for_key in (select id_for_key from {TEST_SET_TABLE})
""")
sent = work(sent, "sent")

p1_summary = keep("p1_sent_summary", spark.sql(f"""
    select (select count(distinct id_for_key) from {TEST_SET_TABLE}) as test_set_beneficiary_keys,
           count(distinct base_key) as keys_with_sponsor_rows,
           count(*) as sponsor_rows,
           count(distinct sponsor_key) as distinct_sponsor_keys,
           count(distinct receipt) as receipts,
           count(distinct ssn_key) as distinct_sponsor_ssns
    from sent
"""))
p1 = p1_summary.iloc[0]
EXPECTED_SPONSORS = int(p1["distinct_sponsor_keys"])

keep("p1_detail_value_shapes", spark.sql(f"""
    select 'dob' as field, {pattern_sql(null_sql(DCOL['dob']))} as value_shape, count(*) as sponsor_rows
    from {SPONSOR_DETAILS_TABLE} group by 1, 2
    union all
    select 'identifier_type', coalesce(upper(cast({null_sql(next((f"`{c}`" for c in detail_cols if re.search(r"type_?cd$", c.lower()) and "identif" in c.lower()), None))} as string)), '(null)'), count(*)
    from {SPONSOR_DETAILS_TABLE} group by 1, 2
    order by 1, 3 desc
""").toPandas().groupby("field", group_keys=False).head(10))

per_key = keep("p1_sponsors_per_beneficiary", spark.sql("""
    select least(n, 4) as sponsors_on_record, count(*) as beneficiary_keys
    from (select base_key, count(distinct sponsor_key) as n from sent group by 1) group by 1 order by 1
"""))
per_key["sponsors_on_record"] = per_key["sponsors_on_record"].astype(int).astype(str).replace("4", "4+")

keep("p1_sponsor_type", spark.sql("""
    select coalesce(sponsor_type, '(blank)') as sponsor_type, coalesce(qualified, '(blank)') as qualified,
           count(distinct sponsor_key) as sponsors
    from sent group by 1, 2 order by 3 desc
"""))

fill = spark.sql("""
    select count(distinct sponsor_key) as sponsors,
           count(distinct case when ssn_key is not null then sponsor_key end) as valid_ssn,
           count(distinct case when dob is not null then sponsor_key end) as valid_dob,
           count(distinct case when first_name is not null and last_name is not null then sponsor_key end) as first_and_last_name,
           count(distinct case when a_number is not null then sponsor_key end) as a_number,
           count(distinct case when ssn_key is null and dob is null then sponsor_key end) as no_ssn_and_no_dob
    from sent
""").toPandas().T.reset_index().set_axis(["field", "sponsors"], axis=1)
fill["sponsors"] = fill["sponsors"].astype(int)
fill["pct_of_sponsors"] = (fill["sponsors"] / max(EXPECTED_SPONSORS, 1)).round(4)
keep("p1_field_completeness", fill)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].bar(per_key["sponsors_on_record"], per_key["beneficiary_keys"], color="#4c72b0")
label_bars(axes[0])
axes[0].set_title("Sponsors per beneficiary record in the test set")
axes[0].set_xlabel("sponsors on the record")
axes[0].set_ylabel("beneficiary records")
f2 = fill[fill["field"] != "sponsors"]
axes[1].barh(f2["field"], f2["pct_of_sponsors"] * 100, color="#55a868")
label_hbars(axes[1], "{:.1f}%")
axes[1].invert_yaxis()
axes[1].set_xlim(0, 110)
axes[1].set_title(f"Sponsor field completeness ({EXPECTED_SPONSORS:,} sponsors)")
axes[1].set_xlabel("% of sponsors")
save_chart(fig, "01_sent")

# COMMAND ----------

# ================================================================================================
# 4. PART 1 - SPONSORS ON MORE THAN ONE RECEIPT (THE ONES THAT SHOULD LINK)
# A sponsor SSN on 2+ receipts in the test set is one person sponsoring more than once. These are
# the sponsor records stage should put in one identity. Name and DOB consistency across those
# receipts shows how much the other fields agree. SSNs on more than MAX_RECORDS_PER_SSN receipts
# are listed separately (shared or placeholder values) and left out of later linking counts.
# ================================================================================================

ssn_groups = work(spark.sql(f"""
    select ssn_key, count(distinct sponsor_key) as sponsor_records, count(distinct receipt) as receipts,
           count(distinct concat_ws('|', first_name, last_name)) as names,
           count(distinct dob) as dobs
    from sent where ssn_key is not null group by 1
"""), "sent_ssn_groups")

repeat = keep("p1_repeat_sponsors", spark.sql(f"""
    select count(*) as sponsor_ssns,
           count_if(receipts > 1) as ssns_on_2plus_receipts,
           sum(case when receipts > 1 then sponsor_records else 0 end) as sponsor_records_on_those_ssns,
           count_if(receipts > {MAX_RECORDS_PER_SSN}) as ssns_over_size_cap,
           max(receipts) as max_receipts_one_ssn
    from sent_ssn_groups
"""))
rep = repeat.iloc[0]

recv = keep("p1_receipts_per_sponsor_ssn", spark.sql("""
    select case when receipts >= 6 then '6+' else cast(receipts as string) end as receipts_per_ssn, count(*) as sponsor_ssns
    from sent_ssn_groups group by 1 order by min(receipts)
"""))
cons = keep("p1_name_dob_consistency", spark.sql(f"""
    select {consistency_sql('names', 'dobs')} as consistency, count(*) as sponsor_ssns
    from sent_ssn_groups where receipts > 1 and receipts <= {MAX_RECORDS_PER_SSN} group by 1
"""))
cons = cons.set_index("consistency").reindex(CONSISTENCY_ORDER).fillna(0).reset_index()
cons["pct"] = (cons["sponsor_ssns"] / max(cons["sponsor_ssns"].sum(), 1)).round(4)
TABLES["p1_name_dob_consistency"] = cons

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].bar(recv["receipts_per_ssn"], recv["sponsor_ssns"], color="#4c72b0")
label_bars(axes[0])
axes[0].set_yscale("log")
axes[0].set_title("Receipts per sponsor SSN (log scale)")
axes[0].set_xlabel("receipts carrying the SSN")
axes[0].set_ylabel("sponsor SSNs")
axes[1].barh(cons["consistency"], cons["sponsor_ssns"], color="#8172b2")
label_hbars(axes[1])
axes[1].invert_yaxis()
axes[1].set_title("Name / DOB across receipts, sponsor SSNs on 2+ receipts")
axes[1].set_xlabel("sponsor SSNs")
save_chart(fig, "02_repeat_sponsors")

# COMMAND ----------

# ================================================================================================
# 5. PART 2 - WHAT THE PIPELINE RECEIVED: GENERATED PAYLOADS FOR THE RUN
# Every element of party.sponsors is read (the check cell in IDEN-46817 read only sponsors[0]).
# Payloads are matched to the test set by the id_for_key found in sourceId / systemCaseIdentifier.
# ================================================================================================

log_df = spark.table(GENERATE_LOG_TABLE)
data_type = log_df.select(F.expr(LOG_DATA_COL).alias("d")).schema[0].dataType
DATA_EXPR = LOG_DATA_COL if isinstance(data_type, T.StringType) else f"to_json({LOG_DATA_COL})"
print("payload column type:", data_type.simpleString(), "-> read as", DATA_EXPR)

run_filter = f"source_system = '{LOG_SOURCE_SYSTEM}'"
if LOG_RUN_IDS:
    run_filter += f" and run_id in ({', '.join(str(r) for r in LOG_RUN_IDS)})"

P = LOG_SPONSOR_PATHS
payload = work(spark.sql(f"""
    with m as (
        select {DATA_EXPR} as d from {GENERATE_LOG_TABLE} where {run_filter}
    ), e as (
        select get_json_object(d, '$.notificationType') as notification_type,
               get_json_object(d, '$.sourceId') as source_id,
               get_json_object(d, '$.systemCaseIdentifier') as case_id,
               posexplode_outer(from_json(get_json_object(d, '$.party.sponsors'), 'array<string>')) as (pos, s)
        from m
    )
    select notification_type, source_id, case_id, pos,
           coalesce(nullif(regexp_extract(coalesce(source_id, ''), '{BASE_KEY_REGEX}', 1), ''),
                    nullif(regexp_extract(coalesce(case_id, ''), '{BASE_KEY_REGEX}', 1), '')) as base_key,
           s is not null as has_sponsor,
           get_json_object(s, '{P['form_type']}') as form_type,
           get_json_object(s, '{P['sponsor_type']}') as sponsor_type,
           get_json_object(s, '{P['qualified']}') as qualified,
           {ssn_sql(f"get_json_object(try_element_at(filter(from_json(get_json_object(s, '{P['identifiers']}'), 'array<string>'), x -> upper(get_json_object(x, '{P['identifier_type']}')) rlike '{SSN_TYPE_RX}'), 1), '{P['identifier_value']}')")} as ssn_key,
           get_json_object(try_element_at(from_json(get_json_object(s, '{P['identifiers']}'), 'array<string>'), 1), '{P['identifier_type']}') as first_identifier_type,
           {dob_sql(f"get_json_object(s, '{P['dob']}')")} as dob,
           {pattern_sql(f"get_json_object(s, '{P['dob']}')")} as dob_shape,
           {pattern_sql(f"get_json_object(s, '{P['zip5']}')")} as zip5_shape,
           {clean_sql(f"get_json_object(s, '{P['first_name']}')")} as first_name,
           {clean_sql(f"get_json_object(s, '{P['last_name']}')")} as last_name,
           {filled_sql(f"get_json_object(s, '{P['zip5']}')")} as zip5
    from e
"""), "payload")

p2 = keep("p2_payload_summary", spark.sql(f"""
    select count(distinct concat_ws('|', source_id, case_id)) as payload_messages,
           count(distinct case when has_sponsor then concat_ws('|', source_id, case_id) end) as messages_with_sponsors,
           count_if(has_sponsor) as sponsors_in_payloads,
           count_if(has_sponsor and pos > 0) as sponsors_after_first_position,
           count_if(has_sponsor and ssn_key is not null) as sponsors_with_valid_ssn,
           count_if(has_sponsor and dob is not null) as sponsors_with_valid_dob,
           count_if(has_sponsor and first_name is not null and last_name is not null) as sponsors_with_first_last_name,
           count_if(has_sponsor and zip5 is not null) as sponsors_with_zip5,
           count(distinct p.base_key) as payload_beneficiary_keys,
           count(distinct t.id_for_key) as test_set_keys_in_payloads
    from payload p left join (select distinct id_for_key from {TEST_SET_TABLE}) t on t.id_for_key = p.base_key
""")).iloc[0]

keep("p2_identifier_types", spark.sql("""
    select coalesce(first_identifier_type, '(none)') as first_identifier_type, count(*) as sponsors
    from payload where has_sponsor group by 1 order by 2 desc limit 20
"""))
keep("p2_payload_value_shapes", spark.sql("""
    select 'dob' as field, dob_shape as value_shape, count(*) as sponsors from payload where has_sponsor group by 1, 2
    union all
    select 'zip5', zip5_shape, count(*) from payload where has_sponsor group by 1, 2
    order by 1, 3 desc
""").toPandas().groupby("field", group_keys=False).head(10))

sp_samples = [r[0] for r in spark.sql(f"""
    select s from (select explode(from_json(get_json_object({DATA_EXPR}, '$.party.sponsors'), 'array<string>')) as s
                   from {GENERATE_LOG_TABLE} where {run_filter} limit 500) limit 1000
""").collect()]
sp_leaves, _, sp_n = discover_paths(sp_samples)
keep("p2_sponsor_json_paths", pd.DataFrame([{"path": p, "in_pct_of_sample": round(i["docs"] / max(sp_n, 1), 3),
                                             "example_type": i["example_type"]} for p, i in sp_leaves.items()])
     .sort_values("path").reset_index(drop=True))

keep("p2_notification_types", spark.sql("""
    select coalesce(notification_type, '(none)') as notification_type, count(distinct concat_ws('|', source_id, case_id)) as messages
    from payload group by 1 order by 2 desc
"""))

cov = keep("p2_sent_vs_payload_by_key", spark.sql("""
    with s as (select base_key, count(distinct sponsor_key) as sent from sent group by 1),
         p as (select base_key, count_if(has_sponsor) as received from payload where base_key is not null group by 1)
    select case when p.base_key is null then 'not in payloads'
                when p.received = 0 then 'payload has no sponsors'
                when p.received < s.sent then 'fewer sponsors than sent'
                when p.received = s.sent then 'same sponsor count'
                else 'more sponsors than sent' end as outcome,
           count(*) as beneficiary_keys, sum(s.sent) as sponsors_sent, sum(coalesce(p.received, 0)) as sponsors_in_payload
    from s left join p on p.base_key = s.base_key group by 1 order by 2 desc
"""))
if p2["test_set_keys_in_payloads"] == 0:
    print("NOTE: no payload matched a test-set key. Check BASE_KEY_REGEX against sourceId / systemCaseIdentifier:")
    display(spark.sql("select source_id, case_id from payload limit 5").toPandas())

pf = pd.DataFrame({"field": ["valid SSN", "valid DOB", "first + last name", "ZIP5"],
                   "pct": [pct(p2[c], p2["sponsors_in_payloads"]) * 100 for c in
                           ["sponsors_with_valid_ssn", "sponsors_with_valid_dob", "sponsors_with_first_last_name", "sponsors_with_zip5"]]})
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].barh(cov["outcome"], cov["beneficiary_keys"], color="#4c72b0")
label_hbars(axes[0])
axes[0].invert_yaxis()
axes[0].set_title("Test-set beneficiary records: sponsors sent vs in payload")
axes[0].set_xlabel("beneficiary records")
axes[1].barh(pf["field"], pf["pct"], color="#55a868")
label_hbars(axes[1], "{:.1f}%")
axes[1].invert_yaxis()
axes[1].set_xlim(0, 110)
axes[1].set_title(f"Sponsor fields in payloads ({p2['sponsors_in_payloads']:,.0f} sponsors)")
axes[1].set_xlabel("% of sponsors in payloads")
save_chart(fig, "03_payloads")

# COMMAND ----------

# ================================================================================================
# 6. STAGE DUMP READINESS
# Finds the stage party and identity dump tables and checks they contain sponsor party records
# (_id matching SPONSOR_ID_REGEX). If none are found the dump was taken before the sponsor load:
# rerun the Harness dump job and the Databricks load (snapshot runbook Parts 8-9), then rerun
# from here. Parts 3-4 are skipped until then.
# ================================================================================================

meta = f"{CATALOG}.pcis_metadata"
if STAGE_PARTY_TABLE is None:
    c = [t for t in find_tables(meta, r"elasticdump.*part") if not re.search(r"match|archive|temp|tmp", t)]
    STAGE_PARTY_TABLE = f"{meta}.{c[0]}" if c else None
if STAGE_IDENTITY_TABLE is None:
    c = [t for t in find_tables(meta, r"elasticdump.*identit") if not re.search(r"archive|temp|tmp", t)]
    STAGE_IDENTITY_TABLE = f"{meta}.{c[0]}" if c else None
print("stage party table:   ", STAGE_PARTY_TABLE)
print("stage identity table:", STAGE_IDENTITY_TABLE)

STAGE_READY = False
readiness = {"party_table": str(STAGE_PARTY_TABLE), "identity_table": str(STAGE_IDENTITY_TABLE)}
if STAGE_PARTY_TABLE:
    r = spark.sql(f"""select count(*) as party_docs, count_if(_id rlike '{SPONSOR_ID_REGEX}') as sponsor_party_docs,
                             count(distinct _index) as indexes from {STAGE_PARTY_TABLE}""").collect()[0]
    readiness.update(r.asDict())
    readiness["party_indexes"] = ", ".join(x[0] for x in spark.sql(f"select distinct _index from {STAGE_PARTY_TABLE} limit 10").collect())
    STAGE_READY = r["sponsor_party_docs"] > 0
keep("p3_stage_readiness", pd.DataFrame([readiness]).astype(str).T.reset_index().set_axis(["item", "value"], axis=1))
if not STAGE_READY:
    print("STOP: no sponsor party records in the stage dump. Parts 3-4 are skipped; rerun the dump after the sponsor load.")

# COMMAND ----------

# ================================================================================================
# 7. PART 3 - MAP THE STAGE PARTY DOCUMENT
# Samples sponsor and non-sponsor party documents, lists their JSON paths, and maps the fields
# used below by name. Paths under any "sponsor" key are excluded so a beneficiary's sponsors array
# is not read as the beneficiary's own data. Check the mapping table; fix with PARTY_OVERRIDES.
# ================================================================================================

PCOL = {}
if STAGE_READY:
    samples = [r[0] for r in spark.sql(f"""
        (select _source from {STAGE_PARTY_TABLE} where _id rlike '{SPONSOR_ID_REGEX}' limit {SAMPLE_DOCS // 2})
        union all
        (select _source from {STAGE_PARTY_TABLE} where not _id rlike '{SPONSOR_ID_REGEX}' limit {SAMPLE_DOCS // 2})
    """).collect()]
    leaves, arrays, n_docs = discover_paths(samples)
    keep("p3_party_json_paths", pd.DataFrame([{"path": p, "in_pct_of_sample": round(i["docs"] / max(n_docs, 1), 3),
                                                "example_type": i["example_type"]} for p, i in leaves.items()])
         .sort_values("path").reset_index(drop=True))
    PCOL, party_map, ID_TYPE_PATH = map_json("_source", leaves, arrays,
                                             ["identity_id", "sparse", "role", "form_type", "receipt", "system",
                                              "ssn", "first_name", "last_name", "dob"],
                                             PARTY_OVERRIDES, exclude_rx=r"sponsor")
    keep("p3_party_field_map", party_map)
    if ID_TYPE_PATH:
        arr_path, type_path = ID_TYPE_PATH
        keep("p3_party_identifier_types", spark.sql(f"""
            select upper(get_json_object(x, '{type_path}')) as identifier_type, count(*) as identifiers
            from (select explode(from_json(get_json_object(_source, '{arr_path}'), 'array<string>')) as x
                  from {STAGE_PARTY_TABLE} where _id rlike '{SPONSOR_ID_REGEX}' limit 200000)
            group by 1 order by 2 desc limit 20
        """))

# COMMAND ----------

# ================================================================================================
# 8. PART 3 - PARSED PARTY RECORDS AND WHAT LANDED
# One row per stage party record: base key (id_for_key), whether it is a sponsor split, receipt,
# role, sparse flag, identity, SSN, name, DOB. With PARSE_ALL_PARTIES = False only sponsor records
# and the other records on the test-set receipts are parsed (smaller in memory; section 10's
# "joined existing identity" then has no records from other receipts to compare with).
# Landing compares sponsor records per beneficiary key in stage with sponsors sent (section 3).
# ================================================================================================

if STAGE_READY:
    scope = "" if PARSE_ALL_PARTIES else (
        f"where _id rlike '{SPONSOR_ID_REGEX}' or regexp_extract(_id, '{BASE_KEY_REGEX}', 1) in (select id_for_key from {TEST_SET_TABLE})")
    parties = work(spark.sql(f"""
        select _id as party_id,
               regexp_extract(_id, '{BASE_KEY_REGEX}', 1) as base_key,
               _id rlike '{SPONSOR_ID_REGEX}' as is_sponsor,
               coalesce(nullif(regexp_extract(_id, '{RECEIPT_REGEX}', 1), ''), {null_sql(PCOL.get('receipt'))}) as receipt,
               {null_sql(PCOL.get('identity_id'))} as identity_id,
               lower(cast({null_sql(PCOL.get('sparse'))} as string)) as sparse,
               {filled_sql(null_sql(PCOL.get('role')))} as role,
               {filled_sql(null_sql(PCOL.get('form_type')))} as form_type,
               {ssn_sql(null_sql(PCOL.get('ssn')))} as ssn_key,
               {clean_sql(null_sql(PCOL.get('first_name')))} as first_name,
               {clean_sql(null_sql(PCOL.get('last_name')))} as last_name,
               {dob_sql(null_sql(PCOL.get('dob')))} as dob
        from {STAGE_PARTY_TABLE} {scope}
    """), "parties")

    # identity link: from the party document, or else from the identity documents
    IDENTITY_SOURCE = "party document"
    if PCOL.get("identity_id") is None and STAGE_IDENTITY_TABLE:
        id_samples = [r[0] for r in spark.sql(f"select _source from {STAGE_IDENTITY_TABLE} limit {SAMPLE_DOCS}").collect()]
        id_leaves, id_arrays, _ = discover_paths(id_samples)
        key_rx = re.compile(BASE_KEY_REGEX)
        link_expr = None
        for doc in id_samples[:200]:
            try:
                d = json.loads(doc)
            except Exception:
                continue
            for p, (segs, val) in flatten_doc(d)[0].items():
                if "[0]" in p and isinstance(val, str) and key_rx.search(val):
                    arr = p[:p.rindex("[0]")]
                    sub = p[p.rindex("[0]") + 3:]
                    link_expr = (arr, "$" + sub if sub else None)
                    break
            if link_expr:
                break
        if link_expr:
            arr, sub = link_expr
            member = "x" if sub is None else f"get_json_object(x, '{sub}')"
            links = work(spark.sql(f"""
                select distinct {member} as party_id, _id as identity_id
                from (select _id, explode(from_json(get_json_object(_source, '{arr}'), 'array<string>')) as x
                      from {STAGE_IDENTITY_TABLE})
            """), "identity_links")
            parties = work(spark.sql(f"""
                select p.party_id, p.base_key, p.is_sponsor, p.receipt, l.identity_id, p.sparse, p.role, p.form_type,
                       p.ssn_key, p.first_name, p.last_name, p.dob
                from parties p left join identity_links l on {norm_key_sql('l.party_id')} = {norm_key_sql('p.party_id')}
            """), "parties_linked")
            spark.table("parties_linked").createOrReplaceTempView("parties")
            IDENTITY_SOURCE = f"identity document {arr}{'' if sub is None else ' ' + sub}"
        else:
            IDENTITY_SOURCE = "not found"
    print("identity link taken from:", IDENTITY_SOURCE)

    sp = work(spark.sql(f"""
        select p.*, t.id_for_key is not null as in_test_set
        from parties p left join (select distinct id_for_key from {TEST_SET_TABLE}) t on t.id_for_key = p.base_key
        where p.is_sponsor
    """), "sponsor_parties")

    p3 = keep("p3_landing_summary", spark.sql(f"""
        select {EXPECTED_SPONSORS} as sponsors_sent,
               count_if(in_test_set) as sponsor_records_in_stage_for_test_set,
               count_if(not in_test_set) as sponsor_records_in_stage_outside_test_set,
               count_if(in_test_set and identity_id is not null) as with_identity,
               count_if(in_test_set and identity_id is null) as without_identity,
               count_if(in_test_set and sparse in ('true', '1', 'y', 'yes')) as flagged_sparse,
               count_if(in_test_set and ssn_key is not null) as with_valid_ssn,
               count_if(in_test_set and dob is not null) as with_valid_dob
        from sponsor_parties
    """)).iloc[0]

    landing = keep("p3_landing_by_beneficiary", spark.sql("""
        with s as (select base_key, count(distinct sponsor_key) as sent from sent group by 1),
             l as (select base_key, count(*) as landed from sponsor_parties where in_test_set group by 1)
        select case when l.base_key is null then 'no sponsor record in stage'
                    when l.landed < s.sent then 'fewer than sent'
                    when l.landed = s.sent then 'all sponsors landed'
                    else 'more than sent' end as outcome,
               count(*) as beneficiary_keys, sum(s.sent) as sponsors_sent, sum(coalesce(l.landed, 0)) as sponsor_records_in_stage
        from s left join l on l.base_key = s.base_key group by 1 order by 2 desc
    """))
    keep("p3_sponsor_role_values", spark.sql("""
        select coalesce(role, '(blank)') as role, coalesce(sparse, '(blank)') as sparse, count(*) as sponsor_records
        from sponsor_parties where in_test_set group by 1, 2 order by 3 desc
    """))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    axes[0].barh(landing["outcome"], landing["beneficiary_keys"], color="#4c72b0")
    label_hbars(axes[0])
    axes[0].invert_yaxis()
    axes[0].set_title("Test-set beneficiary records: sponsor records found in stage")
    axes[0].set_xlabel("beneficiary records")
    lv = pd.DataFrame({"stage": ["sent", "in stage", "with identity", "flagged sparse"],
                       "n": [EXPECTED_SPONSORS, p3["sponsor_records_in_stage_for_test_set"], p3["with_identity"], p3["flagged_sparse"]]})
    axes[1].bar(lv["stage"], lv["n"], color=["#999999", "#4c72b0", "#55a868", "#c44e52"])
    label_bars(axes[1])
    axes[1].set_title("Sponsor records: sent -> stage party index")
    axes[1].set_ylabel("sponsor records")
    save_chart(fig, "04_landing")

# COMMAND ----------

# ================================================================================================
# 9. PART 4 - HOW STAGE RESOLVED THE SPONSOR RECORDS: IDENTITY COMPOSITION
# Each test-set sponsor record is placed in one category, first match wins: no identity; alone;
# with non-sponsor records from other receipts (an existing person record); with a record on its
# own receipt only (petitioner / beneficiary split from the same beneficiary record); with other
# sponsor records only. The second table breaks the own-receipt merges down by role and SSN.
# ================================================================================================

if STAGE_READY:
    members = work(spark.sql("""
        with ids as (select distinct identity_id from sponsor_parties where in_test_set and identity_id is not null)
        select p.* from parties p join ids on ids.identity_id = p.identity_id
    """), "sponsor_identity_members")

    comp = work(spark.sql("""
        select identity_id, count(*) as records, count_if(is_sponsor) as sponsor_records,
               count_if(not is_sponsor) as other_records,
               count(distinct ssn_key) as ssns,
               count(distinct case when is_sponsor then ssn_key end) as sponsor_ssns
        from sponsor_identity_members group by 1
    """), "identity_comp")

    sp_identity = keep("p4_sponsor_record_identity_type", spark.sql("""
        with o as (
            select s.party_id,
                   count_if(not m.is_sponsor and m.base_key = s.base_key) as own_receipt_records,
                   count_if(not m.is_sponsor and m.base_key <> s.base_key) as other_receipt_records
            from sponsor_parties s join sponsor_identity_members m on m.identity_id = s.identity_id
            where s.in_test_set group by 1),
        t as (
            select s.party_id,
                   case when c.identity_id is null then 'no identity'
                        when c.records = 1 then 'alone (new identity)'
                        when o.other_receipt_records > 0 then 'with non-sponsor records from other receipts'
                        when o.own_receipt_records > 0 then 'with a record on its own receipt only'
                        else 'with other sponsor records only' end as identity_type
            from sponsor_parties s left join identity_comp c on c.identity_id = s.identity_id
            left join o on o.party_id = s.party_id
            where s.in_test_set)
        select identity_type, count(*) as sponsor_records from t group by 1 order by 2 desc
    """))
    sp_identity["pct"] = (sp_identity["sponsor_records"] / max(sp_identity["sponsor_records"].sum(), 1)).round(4)
    TABLES["p4_sponsor_record_identity_type"] = sp_identity

    same_receipt = keep("p4_same_receipt_merges", spark.sql("""
        select coalesce(o.role, '(blank role)') as other_record_role,
               count(distinct s.party_id) as sponsor_records,
               count(distinct case when s.ssn_key is not null and s.ssn_key = o.ssn_key then s.party_id end) as same_ssn,
               count(distinct case when s.ssn_key is not null and o.ssn_key is not null and s.ssn_key <> o.ssn_key then s.party_id end) as different_ssn
        from sponsor_parties s
        join sponsor_identity_members o
          on o.identity_id = s.identity_id and o.base_key = s.base_key and not o.is_sponsor
        where s.in_test_set
        group by 1 order by 2 desc
    """))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(sp_identity["identity_type"], sp_identity["sponsor_records"], color="#4c72b0")
    label_hbars(ax)
    ax.invert_yaxis()
    ax.set_title("Where each test-set sponsor record resolved in stage")
    ax.set_xlabel("sponsor records")
    save_chart(fig, "05_identity_type")

# COMMAND ----------

# ================================================================================================
# 10. PART 4 - SAME SPONSOR ON 2+ RECEIPTS: LINKED OR SPLIT
# Sponsor records grouped by SSN. A sponsor on 2+ receipts is linked when all their sponsor
# records sit in one identity. The second view adds the person's non-sponsor records with the same
# SSN on other receipts (as beneficiary, petitioner, etc.; records split from the sponsor's own
# receipt are left out): did the sponsor records join the person's existing identity? Name / DOB consistency is shown for the split ones. Same cap as IDEN-45868.
# ================================================================================================

if STAGE_READY:
    sg = work(spark.sql(f"""
        select ssn_key, count(*) as records, count(distinct receipt) as receipts,
               count(distinct identity_id) as identities,
               count(distinct concat_ws('|', first_name, last_name)) as names, count(distinct dob) as dobs
        from sponsor_parties where in_test_set and ssn_key is not null and identity_id is not null
        group by 1
    """), "stage_sponsor_ssn_groups")

    linking = long_form(spark.sql(f"""
        select count(*) as sponsor_ssns_resolved,
               count_if(receipts > 1 and receipts <= {MAX_RECORDS_PER_SSN}) as sponsors_2plus_records,
               count_if(receipts > 1 and receipts <= {MAX_RECORDS_PER_SSN} and identities = 1) as linked_one_identity,
               count_if(receipts > 1 and receipts <= {MAX_RECORDS_PER_SSN} and identities > 1) as split_across_identities,
               count_if(receipts > {MAX_RECORDS_PER_SSN}) as sponsors_over_size_cap
        from stage_sponsor_ssn_groups
    """).toPandas())
    multi = metric(linking, "sponsors_2plus_records")
    linking["pct_of_2plus"] = np.where(linking["metric"].isin(["linked_one_identity", "split_across_identities"]),
                                       (linking["value"] / max(multi, 1)).round(4), np.nan)
    keep("p4_sponsor_linking", linking)

    ident_dist = keep("p4_identities_per_sponsor", spark.sql(f"""
        select case when identities >= 5 then '5+' else cast(identities as string) end as identities_per_sponsor,
               count(*) as sponsor_ssns
        from stage_sponsor_ssn_groups where receipts > 1 and receipts <= {MAX_RECORDS_PER_SSN}
        group by 1 order by min(identities)
    """))
    split_cons = keep("p4_split_name_dob", spark.sql(f"""
        select {consistency_sql('names', 'dobs')} as consistency,
               count_if(identities = 1) as linked, count_if(identities > 1) as split
        from stage_sponsor_ssn_groups where receipts > 1 and receipts <= {MAX_RECORDS_PER_SSN} group by 1
    """))
    split_cons = split_cons.set_index("consistency").reindex(CONSISTENCY_ORDER).fillna(0).astype(int).reset_index()
    split_cons["split_rate"] = (split_cons["split"] / (split_cons["linked"] + split_cons["split"]).replace(0, np.nan)).round(4)
    TABLES["p4_split_name_dob"] = split_cons

    person = long_form(spark.sql(f"""
        with s as (select distinct ssn_key from sponsor_parties where in_test_set and ssn_key is not null),
             own as (select distinct ssn_key, base_key from sponsor_parties where in_test_set and ssn_key is not null),
             all_recs as (select p.ssn_key, p.is_sponsor, p.identity_id from parties p join s on s.ssn_key = p.ssn_key
                          left anti join own on own.ssn_key = p.ssn_key and own.base_key = p.base_key and not p.is_sponsor
                          where p.identity_id is not null),
             g as (select ssn_key, count_if(is_sponsor) as sponsor_records, count_if(not is_sponsor) as other_records,
                          count(distinct identity_id) as identities,
                          count(distinct case when is_sponsor then identity_id end) as sponsor_identities,
                          size(array_intersect(collect_set(case when is_sponsor then identity_id end),
                                               collect_set(case when not is_sponsor then identity_id end))) as shared_identities
                   from all_recs group by 1)
        select count(*) as sponsor_ssns,
               count_if(other_records > 0) as ssns_with_non_sponsor_records,
               count_if(other_records > 0 and identities = 1) as all_records_one_identity,
               count_if(other_records > 0 and shared_identities > 0) as sponsor_joined_existing_identity,
               count_if(other_records > 0 and shared_identities = 0) as sponsor_apart_from_existing_records
        from g where sponsor_records + other_records <= {MAX_RECORDS_PER_SSN}
    """).toPandas())
    keep("p4_sponsor_vs_existing_records", person)

    linked, split = metric(linking, "linked_one_identity"), metric(linking, "split_across_identities")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    if linked + split > 0:
        axes[0].pie([linked, split], labels=["linked (1 identity)", "split (2+ identities)"], autopct="%1.1f%%",
                    colors=["#55a868", "#c44e52"], startangle=90)
    axes[0].set_title(f"Stage: sponsors on 2+ receipts\n{multi:,.0f} sponsor SSNs")
    axes[1].bar(ident_dist["identities_per_sponsor"], ident_dist["sponsor_ssns"], color="#4c72b0")
    label_bars(axes[1])
    axes[1].set_title("Identities per sponsor (2+ receipts)")
    axes[1].set_xlabel("identities")
    axes[1].set_ylabel("sponsor SSNs")
    pv = person[person["metric"].isin(["sponsor_joined_existing_identity", "sponsor_apart_from_existing_records"])]
    axes[2].bar(["joined existing", "apart from existing"], pv["value"], color=["#55a868", "#c44e52"])
    label_bars(axes[2])
    axes[2].set_title("Sponsors who also have non-sponsor records")
    axes[2].set_ylabel("sponsor SSNs")
    save_chart(fig, "06_linking")

# COMMAND ----------

# ================================================================================================
# 11. PART 4 - MERGE RISK
# Identities holding a test-set sponsor record that also hold 2+ different valid SSNs, and the
# same split by who carries them (2+ sponsor SSNs vs a sponsor SSN next to another record's SSN).
# Section 9 already lists sponsor records merged with a record on their own receipt by role.
# ================================================================================================

if STAGE_READY:
    risk = keep("p4_merge_risk", long_form(spark.sql("""
        select count(*) as identities_with_test_set_sponsor,
               count_if(ssns > 1) as identities_with_2plus_ssns,
               count_if(sponsor_ssns > 1) as identities_with_2plus_sponsor_ssns,
               count_if(ssns > 1 and sponsor_ssns <= 1) as sponsor_ssn_differs_from_other_record,
               sum(case when ssns > 1 then records else 0 end) as records_in_those_identities
        from identity_comp
    """).toPandas()))
    ssn_per_identity = keep("p4_ssns_per_identity", spark.sql("""
        select case when ssns >= 4 then '4+' else cast(ssns as string) end as distinct_ssns, count(*) as identities
        from identity_comp group by 1 order by min(ssns)
    """))

# COMMAND ----------

# ================================================================================================
# 12. OPTIONAL - STAGE SCORES FOR SPLIT SPONSORS
# Runs only when STAGE_SCORES_TABLE is set (same columns as mv_es_mlaas_scores: _id,
# candidateSourceId, candidateScore, timestamp). For every pair of same-SSN sponsor records in
# different identities: the latest stored score between them, in the IDEN-45868 bands.
# ================================================================================================

if STAGE_READY and STAGE_SCORES_TABLE:
    band = ("case when score is null then 'not compared' when score >= 0.98 then '0.98+' "
            "when score >= 0.95 then '0.95-0.98' when score >= 0.90 then '0.90-0.95' else 'below 0.90' end")
    split_bands = keep("p4_split_pair_score_bands", spark.sql(f"""
        with g as (select ssn_key from stage_sponsor_ssn_groups where identities > 1 and receipts <= {MAX_RECORDS_PER_SSN}),
             r as (select s.ssn_key, s.party_id, s.identity_id from sponsor_parties s join g on g.ssn_key = s.ssn_key where s.in_test_set),
             p as (select a.ssn_key, a.party_id as ka, b.party_id as kb from r a join r b
                   on a.ssn_key = b.ssn_key and a.party_id < b.party_id and a.identity_id <> b.identity_id),
             sc as (select _id as k1, candidateSourceId as k2, cast(candidateScore as double) as score,
                           row_number() over (partition by _id, candidateSourceId order by `timestamp` desc) as rn
                    from {STAGE_SCORES_TABLE} where _id rlike '{SPONSOR_ID_REGEX}' or candidateSourceId rlike '{SPONSOR_ID_REGEX}')
        select band, count(*) as pairs, count(distinct ssn_key) as sponsors from (
            select p.ssn_key, {band.replace('score', 'greatest(s1.score, s2.score)')} as band
            from p left join sc s1 on s1.rn = 1 and s1.k1 = p.ka and s1.k2 = p.kb
                   left join sc s2 on s2.rn = 1 and s2.k1 = p.kb and s2.k2 = p.ka) group by 1
    """))

# COMMAND ----------

# ================================================================================================
# 13. PART 5 - COMPARISON WITH THE IDEN-45868 STATUS QUO
# IDEN-45868 measured sponsors on prod using stored MLaaS scores, reaching sponsors only through a
# receipt they filed themselves (I-865). Stage measures sponsor records created from every ELIS
# sponsor on the 10M receipts. The record sets differ; the table puts the rates side by side.
# ================================================================================================

if STAGE_READY:
    e = EARLIER_STATUS_QUO
    compare = keep("p5_compare_status_quo", pd.DataFrame([
        {"measure": "record set", "iden_45868_prod": "prod party records reached through the I-865 receipt link",
         "stage_10m": "stage sponsor records split from 10M ELIS beneficiary records"},
        {"measure": "sponsors with 2+ records", "iden_45868_prod": f"{e['sponsors_2plus_records']:,}",
         "stage_10m": f"{multi:,.0f}"},
        {"measure": "linked (one identity)", "iden_45868_prod": f"{e['linked_one_identity']:,} ({pct(e['linked_one_identity'], e['sponsors_2plus_records']):.1%})",
         "stage_10m": f"{linked:,.0f} ({pct(linked, multi):.1%})"},
        {"measure": "split (2+ identities)", "iden_45868_prod": f"{e['split_across_identities']:,} ({pct(e['split_across_identities'], e['sponsors_2plus_records']):.1%})",
         "stage_10m": f"{split:,.0f} ({pct(split, multi):.1%})"},
    ]))

# COMMAND ----------

# ================================================================================================
# 14. PART 5 - REVIEW SAMPLES
# Up to REVIEW_SAMPLE_SIZE record keys per category for manual review in the stage dashboards
# (receipt, party _id, identity _id only). Categories: sponsors split across identities,
# identities with 2+ SSNs, sponsor merged with a record on its own receipt, beneficiary records
# whose sponsors did not land in stage, and sponsor records without an identity.
# ================================================================================================

if STAGE_READY:
    n = REVIEW_SAMPLE_SIZE
    review = keep("p5_review_samples", spark.sql(f"""
        (select 'sponsor split across identities' as category, s.receipt, s.party_id, s.identity_id
         from sponsor_parties s join (select ssn_key from stage_sponsor_ssn_groups
                                      where identities > 1 and receipts <= {MAX_RECORDS_PER_SSN} order by rand(7) limit {n // 4}) g
           on g.ssn_key = s.ssn_key where s.in_test_set)
        union all
        (select 'identity with 2+ SSNs', m.receipt, m.party_id, m.identity_id
         from sponsor_identity_members m join (select identity_id from identity_comp where ssns > 1 order by rand(7) limit {n // 4}) c
           on c.identity_id = m.identity_id)
        union all
        (select 'sponsor merged with record on own receipt', s.receipt, s.party_id, s.identity_id
         from sponsor_parties s join sponsor_identity_members o
           on o.identity_id = s.identity_id and o.base_key = s.base_key and not o.is_sponsor
         where s.in_test_set order by rand(7) limit {n})
        union all
        (select 'sponsors not in stage', regexp_extract(s.base_key, '{RECEIPT_REGEX}', 1), s.base_key, cast(null as string)
         from (select distinct base_key from sent) s left anti join (select distinct base_key from sponsor_parties) l
           on l.base_key = s.base_key order by rand(7) limit {n})
        union all
        (select 'sponsor record without identity', receipt, party_id, identity_id
         from sponsor_parties where in_test_set and identity_id is null order by rand(7) limit {n})
    """))

# COMMAND ----------

# ================================================================================================
# 15. FINDINGS AND EXPORT
# Statements of what the numbers above show, then every table and chart in one Excel workbook
# (findings first, charts second) in RESULTS_DIR. This workbook is the only file written.
# ================================================================================================

FINDINGS.append(f"Test set: {p1['test_set_beneficiary_keys']:,.0f} beneficiary records; {p1['keys_with_sponsor_rows']:,.0f} have "
                f"sponsor rows, carrying {EXPECTED_SPONSORS:,} sponsors on {p1['receipts']:,.0f} receipts.")
fv = fill.set_index("field")["pct_of_sponsors"]
FINDINGS.append(f"Sent sponsors: valid SSN on {fv.get('valid_ssn', 0):.1%}, valid DOB on {fv.get('valid_dob', 0):.1%}, "
                f"first and last name on {fv.get('first_and_last_name', 0):.1%}; {fv.get('no_ssn_and_no_dob', 0):.1%} have neither SSN nor DOB.")
FINDINGS.append(f"{rep['sponsor_ssns']:,.0f} distinct sponsor SSNs; {rep['ssns_on_2plus_receipts']:,.0f} appear on 2+ receipts "
                f"({rep['sponsor_records_on_those_ssns']:,.0f} sponsor records). {rep['ssns_over_size_cap']:,.0f} SSNs are on more than "
                f"{MAX_RECORDS_PER_SSN} receipts (max {rep['max_receipts_one_ssn']:,.0f}).")
ct = cons.set_index("consistency")["pct"]
FINDINGS.append(f"Sponsor SSNs on 2+ receipts: same name and DOB on every receipt {ct.get('same name, same DOB', 0):.1%}; "
                f"name differs on at least one {ct.get('name differs', 0) + ct.get('name and DOB differ', 0):.1%}.")
FINDINGS.append(f"Payloads for the run: {p2['sponsors_in_payloads']:,.0f} sponsors in {p2['messages_with_sponsors']:,.0f} messages; "
                f"{p2['sponsors_after_first_position']:,.0f} are in position 2+ of the sponsors array. "
                f"{p2['test_set_keys_in_payloads']:,.0f} of {p1['test_set_beneficiary_keys']:,.0f} test-set keys found in payloads.")
if STAGE_READY:
    FINDINGS.append(f"Stage party index: {p3['sponsor_records_in_stage_for_test_set']:,.0f} sponsor records for the test set vs "
                    f"{EXPECTED_SPONSORS:,} sent ({pct(p3['sponsor_records_in_stage_for_test_set'], EXPECTED_SPONSORS):.1%}); "
                    f"{p3['flagged_sparse']:,.0f} flagged sparse; {p3['without_identity']:,.0f} have no identity.")
    for r in sp_identity.itertuples():
        FINDINGS.append(f"Sponsor records {r.identity_type}: {r.sponsor_records:,.0f} ({r.pct:.1%}).")
    FINDINGS.append(f"Sponsors on 2+ receipts in stage: {multi:,.0f}; linked in one identity {linked:,.0f} ({pct(linked, multi):.1%}), "
                    f"split {split:,.0f} ({pct(split, multi):.1%}). IDEN-45868 prod status quo: "
                    f"{pct(EARLIER_STATUS_QUO['linked_one_identity'], EARLIER_STATUS_QUO['sponsors_2plus_records']):.1%} linked "
                    f"of {EARLIER_STATUS_QUO['sponsors_2plus_records']:,}.")
    FINDINGS.append(f"Sponsor SSNs that also have non-sponsor records: {metric(person, 'ssns_with_non_sponsor_records'):,.0f}; "
                    f"sponsor record joined the existing identity for {metric(person, 'sponsor_joined_existing_identity'):,.0f}, "
                    f"stayed apart for {metric(person, 'sponsor_apart_from_existing_records'):,.0f}.")
    FINDINGS.append(f"Identities holding a test-set sponsor with 2+ different SSNs: {metric(risk, 'identities_with_2plus_ssns'):,.0f} "
                    f"of {metric(risk, 'identities_with_test_set_sponsor'):,.0f}.")
    for r in same_receipt.itertuples():
        FINDINGS.append(f"Sponsor records merged with a '{r.other_record_role}' record on their own receipt: {r.sponsor_records:,.0f} "
                        f"(same SSN {r.same_ssn:,.0f}, different SSN {r.different_ssn:,.0f}).")
else:
    FINDINGS.append("Stage dump has no sponsor party records: landing and identity results not produced. Rerun the dump after the sponsor load.")

findings = pd.DataFrame({"finding": FINDINGS})
pd.set_option("display.max_colwidth", None)
display(findings)

config = pd.DataFrame([
    ["run time", RUN_STAMP], ["test set", TEST_SET_TABLE], ["sponsor details", SPONSOR_DETAILS_TABLE],
    ["payload log", GENERATE_LOG_TABLE], ["payload filter", run_filter], ["stage party table", str(STAGE_PARTY_TABLE)],
    ["stage identity table", str(STAGE_IDENTITY_TABLE)], ["stage scores table", str(STAGE_SCORES_TABLE)],
    ["identity link from", globals().get("IDENTITY_SOURCE", "n/a")], ["sponsor _id regex", SPONSOR_ID_REGEX],
    ["base key regex", BASE_KEY_REGEX], ["SSN identifier type regex", SSN_TYPE_RX],
    ["parse all parties", PARSE_ALL_PARTIES], ["tables created", "none (in-memory temporary views only)"], ["max records per SSN", MAX_RECORDS_PER_SSN],
    ["IDEN-45868 status quo", json.dumps(EARLIER_STATUS_QUO)],
], columns=["setting", "value"])
config["value"] = config["value"].astype(str)
TABLES = {"key_findings": findings, **TABLES, "config": config}

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


xlsx_path = RESULTS_DIR / f"sponsor10m_stage_{RUN_STAMP}.xlsx"
try:
    write_workbook(xlsx_path)
except OSError:
    local_copy = Path("/tmp") / xlsx_path.name
    write_workbook(local_copy)
    shutil.copy(local_copy, xlsx_path)
print("workbook saved to", xlsx_path)
