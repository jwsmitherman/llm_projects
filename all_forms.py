# Databricks notebook source
# ================================================================================================
# IDEN-45868 - SPONSOR IDENTITIES ACROSS ALL SPONSOR FORMS (STATUS QUO)
#
# Expands the I-865 study to every sponsor record available today. Two parts, both run:
#   Part A - sponsor filings from the source systems: C3 I-865, C3 I-864 (r2i864), ELIS I-864
#            affidavits (spnsr_affdvt_supt), ELIS I-134 (coe.sponsorinfo1)
#   Part B - PCIS party records that carry sponsor data (the "sponsors" array on the party record)
#
# The stored-score status quo (one identity vs split, and what 0.95 / 0.90 would change) is then
# measured for two record sets:
#   Set 1 - C3 I-865 filers: the PCIS record is the sponsor (reproduces the I-865 study)
#   Set 2 - every PCIS party record whose own SSN belongs to a sponsor from Part A or Part B
#           (sponsors who already exist in PCIS in another role)
#
# SSN is the "same sponsor" key. Exported tables hold counts only: no SSNs, names or DOBs.
# Run the cells top to bottom. Charts save as PNG to RESULTS_DIR as they are drawn; the Excel
# workbook is saved there at the end.
# ================================================================================================

# COMMAND ----------

# ================================================================================================
# 1. CONFIGURATION
# Part B settings: leave PARTY_TABLE, PARTY_ID_EXPR and SPONSORS_EXPR as None to auto-discover
# them (section 9 prints what it picked). Set them to override. FIELD_OVERRIDES maps a field to
# a SQL expression when auto-mapping picks the wrong field, for example:
#   {"sponsor.ssn": "s.identifiers.ssn", "party.ssn": "biographicInfo.ssn"}
# "s" is one element of the sponsors array; "party.*" expressions run on the party table row.
# ================================================================================================

import re
import shutil
from datetime import datetime
from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pyspark.sql import functions as F
from pyspark.sql import types as T

CATALOG = "eciscor_prod"
SCORES_TABLE = f"{CATALOG}.pcis_metadata.mv_es_mlaas_scores"
RECEIPT_REGEX = "_([A-Za-z]{3}[0-9]{10})$"
RESULTS_DIR = Path("/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/sponsor_analysis/results")

PARTY_TABLE = None
PARTY_ID_EXPR = None
SPONSORS_EXPR = None
FIELD_OVERRIDES = {}

MAX_RECORDS_PER_SSN = 200
SHARED_SSN_FILINGS = 50

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

# COMMAND ----------

# ================================================================================================
# 2. HELPERS
# Saving, SQL builders for cleaning SSN / DOB / names, the threshold policy rules, and the field
# auto-mapper used on the party record in Part B.
# ================================================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
CHARTS = {}
TABLES = {}
FINDINGS = []


def save_chart(fig, name):
    path = RESULTS_DIR / f"sponsor_all_forms_{name}_{RUN_STAMP}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    CHARTS[name] = path
    print("chart saved to", path)
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


def persist(df, name):
    try:
        df = df.cache()
        df.count()
    except Exception:
        pass
    df.createOrReplaceTempView(name)
    return df


def keep(name, df):
    TABLES[name] = df
    display(df)
    return df


def ssn_sql(col):
    digits = f"regexp_replace(cast({col} as string), '[^0-9]', '')"
    padded = f"lpad({digits}, 9, '0')"
    return (f"case when {digits} rlike '^[0-9]{{7,9}}$' "
            f"and {padded} not in ('000000000', '111111111', '123456789', '999999999') "
            f"and substr({padded}, 1, 3) not in ('000', '666') then {padded} end")


def dob_sql(col):
    s = f"regexp_replace(trim(cast({col} as string)), '[.]0+$', '')"
    ts = (f"coalesce(try_to_timestamp({s}, 'yyyyMMdd'), try_to_timestamp({s}, 'yyyy-MM-dd'), "
          f"try_to_timestamp({s}, 'MM/dd/yyyy'), try_to_timestamp({s}))")
    d = f"date_format({ts}, 'yyyy-MM-dd')"
    return f"case when {d} > '1900-01-01' and {d} <= date_format(current_date(), 'yyyy-MM-dd') then {d} end"


def clean_sql(col):
    return f"nullif(regexp_replace(upper(coalesce(cast({col} as string), '')), '[^A-Z]', ''), '')"


def anum_sql(col):
    digits = f"regexp_replace(upper(cast({col} as string)), '[^0-9]', '')"
    return f"case when {digits} rlike '^[0-9]{{8,9}}$' and cast({digits} as bigint) > 0 then concat('A', lpad({digits}, 9, '0')) end"


def filled_sql(col):
    return f"nullif(trim(cast({col} as string)), '')"


def band_sql(col):
    return (f"case when {col} is null then 'not compared' when {col} >= 0.98 then '0.98+' "
            f"when {col} >= 0.95 then '0.95-0.98' when {col} >= 0.90 then '0.90-0.95' else 'below 0.90' end")


BAND_ORDER = ["not compared", "0.98+", "0.95-0.98", "0.90-0.95", "below 0.90"]
BAND_LOW = {"0.98+": 0.98, "0.95-0.98": 0.95, "0.90-0.95": 0.90, "below 0.90": 0.0}


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


def policy_counts(bands):
    out = {}
    for p in POLICIES:
        n = 0
        for r in bands.itertuples():
            low = BAND_LOW.get(r.band)
            if low is not None and low >= pair_threshold(p, r.sa, r.sb) - 1e-9:
                n += int(r.pairs)
        out[p] = n
    return out


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
    "dob": r"(^|\.)(dob|date_?of_?birth|birth_?date|brth_?dt)$|birth\w*\.date$",
    "a_number": r"(^|\.)(a_?number|a_?nbr|alien_?(number|nbr)|anumber)$",
    "cob": r"country_?of_?birth|birth\w*\.country(\.iso_?code)?$|(^|\.)cob$",
    "coc": r"country_?of_?citizenship|citizenship(\.country)?(\.iso_?code)?$|(^|\.)coc$",
    "form_type": r"(^|\.)form(_?(type|number|nbr|nm|name|id))?$",
    "sponsor_type": r"sponsor_?type|(^|\.)type$|relationship",
    "qualified": r"qualif",
    "address": r"(^|\.)(street(_?\w+)?|address_?line_?1|line_?1|address1)$|address",
    "receipt_number": r"receipt",
}
ID_KINDS = {"ssn": "SSN|SOCIAL", "a_number": "ALIEN|A_?NUMBER|ANUMBER|A-NUMBER"}
TYPE_RX = r"^(type|type_?code|code|identifier_?type|id_?type|kind|category)$"
VALUE_RX = r"^(value|number|nbr|num|identifier|id|val|text)$"
PARTY_EXCLUDE_RX = r"sponsor|relat|attorney|preparer|interpret|spouse|child|parent|famil|beneficiar|petitioner|dependent|household|member"


def leaf_paths(dtype, segs=()):
    if isinstance(dtype, T.StructType):
        for f in dtype.fields:
            yield from leaf_paths(f.dataType, segs + (f.name,))
    elif isinstance(dtype, T.ArrayType) and isinstance(dtype.elementType, T.StructType):
        yield from leaf_paths(dtype.elementType, segs + ("[]",))
    else:
        yield segs, dtype


def path_text(segs):
    return ".".join(s for s in segs if s != "[]").lower()


def path_expr(root, segs):
    expr = root
    for s in segs:
        if s == "[]":
            expr = f"(try_element_at({expr}, 1))"
        else:
            expr = f"`{s}`" if expr is None else f"{expr}.`{s}`"
    return expr


def identifier_lookup(dtype, root, kind_rx, exclude):
    leaves = [(s, d) for s, d in leaf_paths(dtype) if not (exclude and re.search(exclude, path_text(s)))]
    for segs, dt in leaves:
        if isinstance(dt, T.MapType) and "identif" in path_text(segs):
            e = path_expr(root, segs)
            keys = ["SSN", "ssn", "SOCIAL_SECURITY_NUMBER"] if "SSN" in kind_rx else ["A_NUMBER", "ALIEN_NBR", "ANUMBER", "aNumber"]
            return "coalesce(" + ", ".join(e + "['" + k + "']" for k in keys) + ")", path_text(segs) + "[key]"
    groups = {}
    for segs, dt in leaves:
        if "[]" in segs:
            i = len(segs) - 1 - segs[::-1].index("[]")
            if len(segs) - i == 2:
                groups.setdefault(segs[:i], []).append(segs[-1])
    ranked = sorted(groups.items(), key=lambda kv: (0 if "identif" in path_text(kv[0]) else 1, len(kv[0])))
    for prefix, children in ranked:
        t = next((c for c in children if re.search(TYPE_RX, c.lower())), None)
        v = next((c for c in children if re.search(VALUE_RX, c.lower())), None)
        if t and v:
            arr = path_expr(root, prefix)
            e = (f"(try_element_at(filter({arr}, x -> upper(cast(x.`{t}` as string)) rlike '{kind_rx}'), 1)).`{v}`")
            return e, f"{path_text(prefix)}[{t} ~ {kind_rx}].{v}"
    return None, None


def auto_map(dtype, root, fields, prefix, exclude=None):
    leaves = [(s, d) for s, d in leaf_paths(dtype)
              if not isinstance(d, T.MapType) and not (exclude and re.search(exclude, path_text(s)))]
    mapping, rows = {}, []
    for name in fields:
        key = f"{prefix}.{name}"
        if key in FIELD_OVERRIDES:
            mapping[name] = FIELD_OVERRIDES[key]
            rows.append({"field": key, "expression": FIELD_OVERRIDES[key], "source_path": "", "how": "override"})
            continue
        hits = [(s, d) for s, d in leaves if re.search(FIELD_SPECS[name], path_text(s))
                and (name in ID_KINDS or "identif" not in path_text(s))]
        hits.sort(key=lambda x: (x[0].count("[]"), len(x[0])))
        if hits:
            segs, dt = hits[0]
            e = path_expr(root, segs)
            if isinstance(dt, T.ArrayType):
                e = f"try_element_at({e}, 1)"
            mapping[name] = e
            rows.append({"field": key, "expression": e, "source_path": path_text(segs), "how": "matched by name"})
            continue
        if name in ID_KINDS:
            e, src = identifier_lookup(dtype, root, ID_KINDS[name], exclude)
            if e:
                mapping[name] = e
                rows.append({"field": key, "expression": e, "source_path": src, "how": "identifier list"})
                continue
        rows.append({"field": key, "expression": "", "source_path": "", "how": "not found"})
    return mapping, pd.DataFrame(rows)


def find_columns(name_regex):
    q = (f"select table_schema, table_name, column_name, data_type from {CATALOG}.information_schema.columns "
         f"where lower(column_name) rlike '{name_regex}' and table_schema <> 'information_schema'")
    try:
        return spark.sql(q).toPandas()
    except Exception:
        rows = []
        for db in spark.catalog.listDatabases():
            for t in spark.catalog.listTables(db.name):
                if t.isTemporary:
                    continue
                for c in spark.catalog.listColumns(t.name, db.name):
                    if re.search(name_regex, c.name.lower()):
                        rows.append((db.name, t.name, c.name, c.dataType))
        return pd.DataFrame(rows, columns=["table_schema", "table_name", "column_name", "data_type"])


def metric(df, name, col="value"):
    if df is None or df.empty or name not in set(df["metric"]):
        return float("nan")
    return float(df.loc[df["metric"] == name, col].iloc[0])


def long_form(pdf):
    out = pdf.T.reset_index()
    out.columns = ["metric", "value"]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out

# COMMAND ----------

# ================================================================================================
# 3. PCIS SCORES AND IDENTITIES
# scores          - latest stored MLaaS score per (record, candidate) pair
# record_identity - each PCIS record's latest identityId, source system, receipt and model version
# ================================================================================================

spark.sql(f"""
    create or replace temp view scores as
    select identity_id, record_key, candidate_key, score, ts, rule_name, record_system, candidate_system, record_receipt
    from (
        select identityId as identity_id, _id as record_key, candidateSourceId as candidate_key,
               cast(candidateScore as double) as score, `timestamp` as ts, rule_name,
               lower(split(_id, '_')[0]) as record_system,
               lower(split(candidateSourceId, '_')[0]) as candidate_system,
               upper(regexp_extract(_id, '{RECEIPT_REGEX}', 1)) as record_receipt,
               row_number() over (partition by _id, candidateSourceId order by `timestamp` desc) as rn
        from {SCORES_TABLE}
    ) where rn = 1
""")

spark.sql("""
    create or replace temp view record_identity as
    select record_key, record_system, record_receipt, identity_id, rule_name
    from (
        select record_key, record_system, record_receipt, identity_id, rule_name,
               row_number() over (partition by record_key order by ts desc) as rn
        from scores
    ) where rn = 1
""")
print("views ready: scores, record_identity")

# COMMAND ----------

# ================================================================================================
# 4. PART A - SPONSOR FILINGS FROM THE SOURCE SYSTEMS
# One row per sponsor filing, in a common layout. Each source loads on its own; a source that
# fails is reported in source_status and the rest continue.
#   C3 I-865   - petapp FORM_NUMBER = 'I865'; the filer (sponsor) is in the BEN_* columns
#   C3 I-864   - r2i864 (legacy and modern C3); attached_form is the form on the same receipt;
#                DOB falls back to person via i864 when r2i864.dob is empty
#   ELIS I-864 - spnsr_affdvt_supt with the sponsor's name, DOB and A-number from pers tables
#   ELIS I-134 - coe.sponsorinfo1 (pre-built extract)
# ================================================================================================

C3 = f"{CATALOG}.ecc3conods_mod"
ELIS = f"{CATALOG}.ecelis2"

SOURCES = {
    "C3 I-865": f"""
        select 'C3 I-865' as source, 'c3' as source_system, 'I-865' as form_type, 'I865' as attached_form,
               concat('c3_i865_', RECEIPT_NUMBER) as filing_id, RECEIPT_NUMBER as receipt_number,
               coalesce(BEN_FIRST_NAME, PET_FIRST_NAME) as first_name,
               coalesce(BEN_MIDDLE_NAME, PET_MIDDLE_NAME) as middle_name,
               coalesce(BEN_LAST_NAME, PET_LAST_NAME) as last_name,
               coalesce(cast(BEN_DATE_OF_BIRTH as string), cast(PET_DATE_OF_BIRTH as string)) as dob_raw,
               coalesce(BEN_SSN, PET_SSN) as ssn_raw,
               coalesce(BEN_A_NUMBER, PET_A_NUMBER) as a_number_raw,
               BEN_COUNTRY_OF_BIRTH as cob
        from (
            select *, row_number() over (partition by RECEIPT_NUMBER order by DWH_UPD_DT desc) as rn
            from {C3}.petapp where FORM_NUMBER = 'I865'
        ) where rn = 1
    """,
    "C3 I-864": f"""
        select 'C3 I-864' as source, 'c3' as source_system, 'I-864' as form_type,
               coalesce(ci.FORM_NUMBER, 'unknown') as attached_form,
               concat('c3_r2i864_', cast(r.r2i864_id as string)) as filing_id, r.receipt_number,
               r.first_name, r.middle_name, r.last_name,
               coalesce(nullif(trim(cast(r.dob as string)), ''), cast(pp.dob as string)) as dob_raw,
               r.ssn as ssn_raw, r.a_number as a_number_raw, cast(null as string) as cob
        from (
            select *, row_number() over (partition by r2i864_id order by dwh_upd_dt desc) as rn
            from {C3}.r2i864
        ) r
        left join (
            select RECEIPT_NUMBER, FORM_NUMBER,
                   row_number() over (partition by RECEIPT_NUMBER order by DWH_UPD_DT desc) as rn
            from {C3}.case_info
        ) ci on ci.RECEIPT_NUMBER = r.receipt_number and ci.rn = 1
        left join (
            select i.RECEIPT_NUMBER, lpad(regexp_replace(p.ssn, '[^0-9]', ''), 9, '0') as ssn9, max(p.dob) as dob
            from {C3}.i864 i join {C3}.person p on p.person_id = i.PERSON_ID
            group by 1, 2
        ) pp on pp.RECEIPT_NUMBER = r.receipt_number
            and pp.ssn9 = lpad(regexp_replace(r.ssn, '[^0-9]', ''), 9, '0')
        where r.rn = 1
    """,
    "ELIS I-864": f"""
        select 'ELIS I-864' as source, 'elis' as source_system, 'I-864' as form_type, 'unknown' as attached_form,
               filing_id, receipt_number, first_name, middle_name, last_name, dob_raw, ssn_raw, a_number_raw, cob
        from (
            select concat('elis_affdvt_', cast(s.SPNSR_AFFDVT_SUPT_ID as string)) as filing_id,
                   cast(s.BNFT_REQ_IV_PKT_ID as string) as receipt_number,
                   n.FRST_NM as first_name, n.MID_NM as middle_name, n.LAST_NM as last_name,
                   cast(coalesce(b1.BRTH_DT, b2.BRTH_DT) as string) as dob_raw,
                   s.SPNSR_SSN as ssn_raw, coalesce(s.SPNSR_ALN_NBR, p.ALN_NBR) as a_number_raw,
                   p.BRTH_CNTRY_CD as cob,
                   row_number() over (
                       partition by s.SPNSR_AFFDVT_SUPT_ID
                       order by n.ACTV_PERS_NM_IND desc, n.PERS_NM_EFFTV_DT desc,
                                b2.PRIMARY_BRTH_DT_IND desc, s.dwh_upd_dt desc) as rn
            from {ELIS}.spnsr_affdvt_supt s
            left join {ELIS}.pers p on p.PRTY_ID = s.SPNSR_PRTY_ID
            left join {ELIS}.pers_nm n on n.PRTY_ID = s.SPNSR_PRTY_ID
            left join {ELIS}.pers_brth_dt b1 on b1.PERS_BRTH_DT_ID = p.PERS_BRTH_DT_ID
            left join {ELIS}.pers_brth_dt b2 on b2.PERS_ID = s.SPNSR_PRTY_ID
        ) where rn = 1
    """,
    "ELIS I-134": f"""
        select 'ELIS I-134' as source, 'elis' as source_system, coalesce(frm_nbr, 'I-134') as form_type,
               coalesce(cast(Case_Filing_Type as string), 'unknown') as attached_form,
               concat('coe_i134_', cast(monotonically_increasing_id() as string)) as filing_id,
               receipt_number, first_name, cast(null as string) as middle_name, last_name,
               dob_raw, ssn_raw, a_number_raw, cob
        from (
            select distinct frm_nbr, Case_Filing_Type, cast(I134rec_num as string) as receipt_number,
                   frst_nm as first_name, last_nm as last_name,
                   coalesce(cast(DOB as string), cast(brth_dt as string)) as dob_raw,
                   SSN as ssn_raw, alinbr as a_number_raw, brth_cntry_cd as cob
            from {CATALOG}.coe.sponsorinfo1
        )
    """,
}

STD_COLS = ["source", "source_system", "form_type", "attached_form", "filing_id", "receipt_number",
            "first_name", "middle_name", "last_name", "dob_raw", "ssn_raw", "a_number_raw", "cob"]

status_rows, frames = [], []
for name, sql in SOURCES.items():
    try:
        df = spark.sql(sql).select(*[F.col(c).cast("string").alias(c) for c in STD_COLS])
        df.limit(1).collect()
        frames.append(df)
        status_rows.append({"source": name, "status": "loaded", "error": ""})
    except Exception as e:
        status_rows.append({"source": name, "status": "failed", "error": str(e).strip().splitlines()[0][:300]})
source_status = pd.DataFrame(status_rows)
display(source_status)

reduce(lambda a, b: a.unionByName(b), frames).createOrReplaceTempView("sponsor_filings_raw")
filings = spark.sql(f"""
    select source, source_system, form_type, coalesce(attached_form, 'unknown') as attached_form, filing_id,
           upper(trim(receipt_number)) as receipt_number,
           {clean_sql('first_name')} as first_clean, {clean_sql('last_name')} as last_clean,
           {dob_sql('dob_raw')} as dob_norm, {ssn_sql('ssn_raw')} as ssn_key,
           {anum_sql('a_number_raw')} as anum_key, upper(trim(cob)) as cob
    from sponsor_filings_raw
""").withColumn("name_key", F.when(F.col("first_clean").isNotNull() & F.col("last_clean").isNotNull(),
                                   F.concat_ws("|", "first_clean", "last_clean")))
persist(filings, "sponsor_filings")
print("view ready: sponsor_filings")

# COMMAND ----------

# ================================================================================================
# 5. PART A - DATA COMPLETENESS BY SOURCE
# How many sponsor filings each source holds and how many carry an SSN, DOB, name and A-number.
# The second table shows which forms the C3 I-864 affidavits are attached to.
# ================================================================================================

comp = spark.sql("""
    select source, count(*) as filings, count(distinct receipt_number) as receipts,
           count_if(ssn_key is not null) as with_ssn, count_if(dob_norm is not null) as with_dob,
           count_if(name_key is not null) as with_name, count_if(anum_key is not null) as with_a_number,
           count(distinct ssn_key) as distinct_ssn
    from sponsor_filings group by source order by filings desc
""").toPandas()
for c in ["with_ssn", "with_dob", "with_name", "with_a_number"]:
    comp[f"pct_{c[5:]}"] = (comp[c] / comp["filings"]).round(4)
keep("a_completeness", comp)
source_status = source_status.merge(comp[["source", "filings"]], on="source", how="left")
TABLES["source_status"] = source_status

attached = spark.sql("""
    select source, attached_form, count(*) as filings, count_if(ssn_key is not null) as with_ssn
    from sponsor_filings where source in ('C3 I-864', 'ELIS I-134')
    group by 1, 2 order by filings desc limit 40
""").toPandas()
keep("a_attached_forms", attached)

pct_cols = ["pct_ssn", "pct_dob", "pct_name", "pct_a_number"]
x = np.arange(len(comp))
fig, ax = plt.subplots(figsize=(11, 4.5))
for i, c in enumerate(pct_cols):
    ax.bar(x + (i - 1.5) * 0.2, comp[c] * 100, width=0.2, label=c.replace("pct_", ""))
label_bars(ax, "{:.0f}")
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n{n:,.0f} filings" for s, n in zip(comp["source"], comp["filings"])])
ax.set_ylabel("% of filings")
ax.set_ylim(0, 110)
ax.set_title("Sponsor filings by source: share with each identifier")
ax.legend(ncol=4, fontsize=8)
save_chart(fig, "01_source_completeness")

# COMMAND ----------

# ================================================================================================
# 6. PART A - REPEAT SPONSORS ACROSS FILINGS AND FORMS
# One row per sponsor SSN. Shows how many filings each sponsor has, which sources they appear in,
# and how many SSNs carry so many filings that they are likely shared or placeholder values.
# ================================================================================================

sponsor_a = spark.sql("""
    select ssn_key, count(distinct filing_id) as filings, count(distinct receipt_number) as receipts,
           count(distinct source) as sources, concat_ws(' + ', array_sort(collect_set(source))) as source_combo,
           count(distinct name_key) as names, count(distinct last_clean) as last_names,
           count(distinct dob_norm) as dobs
    from sponsor_filings where ssn_key is not null group by ssn_key
""")
persist(sponsor_a, "sponsor_a")

repeat_a = long_form(spark.sql(f"""
    select count(*) as sponsors, count_if(filings > 1) as sponsors_2plus_filings,
           count_if(sources > 1) as sponsors_in_2plus_sources,
           count_if(filings > {SHARED_SSN_FILINGS}) as ssns_over_{SHARED_SSN_FILINGS}_filings
    from sponsor_a
""").toPandas())
keep("a_repeat_summary", repeat_a)

filings_dist = spark.sql("""
    select least(filings, 10) as filings_per_sponsor, count(*) as sponsors from sponsor_a group by 1 order by 1
""").toPandas()
filings_dist["filings_per_sponsor"] = filings_dist["filings_per_sponsor"].astype(int).astype(str).replace("10", "10+")
keep("a_filings_per_sponsor", filings_dist)

combos = spark.sql("""
    select source_combo, count(*) as sponsors, count_if(filings > 1) as sponsors_2plus_filings
    from sponsor_a group by 1 order by sponsors desc limit 15
""").toPandas()
keep("a_source_combinations", combos)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
axes[0].bar(filings_dist["filings_per_sponsor"], filings_dist["sponsors"], color="#4c72b0")
label_bars(axes[0])
axes[0].set_title("Sponsor filings per sponsor (all sources, by SSN)")
axes[0].set_xlabel("filings")
axes[0].set_ylabel("sponsors")
axes[1].barh(combos["source_combo"], combos["sponsors"], color="#55a868")
label_hbars(axes[1])
axes[1].invert_yaxis()
axes[1].set_title("Sponsors by the sources they appear in")
axes[1].set_xlabel("sponsors")
save_chart(fig, "02_repeat_sponsors")

# COMMAND ----------

# ================================================================================================
# 7. PART A - DO A SPONSOR'S FILINGS CARRY THE SAME NAME AND DOB?
# For sponsors with 2+ filings: whether the first+last name and DOB agree across filings. Same
# SSN with the same name and DOB is the easiest case for matching into one sponsor identity;
# differing names or DOBs lean harder on the SSN.
# ================================================================================================

cons_a = spark.sql(f"""
    select case when sources > 1 then 'multiple sources' else source_combo end as source_group,
           {consistency_sql('names', 'dobs')} as consistency, count(*) as sponsors
    from sponsor_a where filings > 1 and filings <= {SHARED_SSN_FILINGS}
    group by 1, 2
""").toPandas()
cons_a_pivot = (cons_a.pivot_table(index="source_group", columns="consistency", values="sponsors", aggfunc="sum", fill_value=0)
                .reindex(columns=[c for c in CONSISTENCY_ORDER if c in set(cons_a["consistency"])]))
cons_a_pivot["total"] = cons_a_pivot.sum(axis=1)
keep("a_name_dob_consistency", cons_a_pivot.reset_index())

overall = cons_a.groupby("consistency")["sponsors"].sum().reindex(CONSISTENCY_ORDER).dropna()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(overall.index, overall.values, color=["#55a868", "#8fbf8f", "#dd8452", "#c44e52", "#8c2d2d", "#999999"][:len(overall)])
label_bars(ax)
ax.set_title("Sponsors with 2+ filings: do name and DOB agree across filings?")
ax.set_ylabel("sponsors")
ax.tick_params(axis="x", rotation=15)
save_chart(fig, "03_name_dob_consistency")

# COMMAND ----------

# ================================================================================================
# 8. PART A - SAME NAME AND DOB, DIFFERENT SSN
# Exact first+last name and DOB shared by filings with different SSNs. At 0.90, an exact name and
# DOB match with no other identifier can match (the concern raised on the threshold call), so
# these groups are where a lower threshold could merge different sponsors. SSN typos also land here.
# ================================================================================================

collide = long_form(spark.sql("""
    with g as (
        select name_key, dob_norm, count(distinct ssn_key) as ssns, count_if(ssn_key is null) as no_ssn_filings,
               count(*) as filings
        from sponsor_filings where name_key is not null and dob_norm is not null
        group by 1, 2
    )
    select count(*) as name_dob_groups,
           count_if(ssns >= 2) as groups_with_2plus_ssns,
           coalesce(sum(case when ssns >= 2 then ssns end), 0) as ssns_in_those_groups,
           coalesce(sum(case when ssns >= 2 then filings end), 0) as filings_in_those_groups,
           count_if(ssns >= 1 and no_ssn_filings > 0) as groups_mixing_ssn_and_no_ssn,
           count_if(ssns = 0) as groups_without_any_ssn
    from g
""").toPandas())
keep("a_name_dob_collisions", collide)

bars = collide[collide["metric"].isin(["name_dob_groups", "groups_with_2plus_ssns", "groups_mixing_ssn_and_no_ssn", "groups_without_any_ssn"])]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(bars["metric"], bars["value"], color=["#4c72b0", "#c44e52", "#dd8452", "#999999"])
label_bars(ax)
ax.set_title("Sponsor filings grouped by exact name + DOB")
ax.set_ylabel("name + DOB groups")
ax.tick_params(axis="x", rotation=10)
save_chart(fig, "04_name_dob_collisions")

# COMMAND ----------

# ================================================================================================
# 9. PART B - FIND THE PARTY RECORDS THAT CARRY SPONSOR DATA
# Looks for a table with a sponsors column, preferring PCIS schemas and party tables, then picks
# the party id column by testing which one matches _id in the scores table. Prints the candidates,
# the choice, the sponsors element schema, and how each field was mapped. Check the mapping and
# use FIELD_OVERRIDES in section 1 if a field is wrong.
# ================================================================================================

PART_B_READY = False
party_candidates = pd.DataFrame()
if PARTY_TABLE is None or SPONSORS_EXPR is None:
    found = find_columns("sponsor")
    if not found.empty:
        found = found[~found["table_name"].str.contains(r"backup|bkup|copy|_[0-9]{8}$", case=False, regex=True)]
        found["rank"] = (found["table_schema"].str.contains("pcis", case=False) * 4
                         + found["table_name"].str.contains("party", case=False) * 3
                         + found["column_name"].str.lower().isin(["sponsors", "sponsor"]) * 2
                         + found["data_type"].astype(str).str.lower().str.match(r"(array|struct|string)") * 1)
        party_candidates = found.sort_values("rank", ascending=False).head(20).reset_index(drop=True)
        display(party_candidates)
        if PARTY_TABLE is not None:
            tn = PARTY_TABLE.split(".")[-1]
            own = party_candidates[party_candidates["table_name"].str.lower() == tn.lower()]
            if not own.empty and SPONSORS_EXPR is None:
                SPONSORS_EXPR = f"`{own.iloc[0]['column_name']}`"
        elif not party_candidates.empty and party_candidates.iloc[0]["rank"] >= 4:
            best = party_candidates.iloc[0]
            PARTY_TABLE = f"{CATALOG}.{best['table_schema']}.{best['table_name']}"
            SPONSORS_EXPR = SPONSORS_EXPR or f"`{best['column_name']}`"

if PARTY_TABLE and SPONSORS_EXPR:
    party_schema = spark.table(PARTY_TABLE).schema
    if PARTY_ID_EXPR is None:
        id_rx = r"^(_id|id|party_?source_?id|source_?id|party_?id|party_?record_?id|record_?id)$|sourceid$"
        id_cols = [f.name for f in party_schema.fields if isinstance(f.dataType, T.StringType) and re.search(id_rx, f.name.lower())]
        trials = []
        for c in id_cols:
            hits = spark.sql(f"""
                select count(*) from (select `{c}` as k from {PARTY_TABLE} where `{c}` is not null limit 2000) x
                left semi join (select _id from {SCORES_TABLE}) s on s._id = x.k
            """).first()[0]
            trials.append({"column": c, "sample_matches_in_scores": hits})
        id_trials = pd.DataFrame(trials, columns=["column", "sample_matches_in_scores"])
        display(id_trials)
        if not id_trials.empty:
            PARTY_ID_EXPR = f"`{id_trials.sort_values('sample_matches_in_scores', ascending=False).iloc[0]['column']}`"

    sponsors_type = spark.sql(f"select {SPONSORS_EXPR} as x from {PARTY_TABLE} limit 0").schema["x"].dataType
    sponsors_sql = SPONSORS_EXPR
    if isinstance(sponsors_type, T.StringType):
        samples = [r[0] for r in spark.sql(
            f"select {SPONSORS_EXPR} as x from {PARTY_TABLE} where length({SPONSORS_EXPR}) > 2 limit 200").collect()]
        if samples:
            ddl = spark.range(1).select(F.schema_of_json(F.lit(max(samples, key=len)))).first()[0]
            sponsors_sql = f"from_json({SPONSORS_EXPR}, '{ddl}')"
            sponsors_type = spark.sql(f"select {sponsors_sql} as x from {PARTY_TABLE} limit 0").schema["x"].dataType
    if isinstance(sponsors_type, T.StructType):
        sponsors_sql = f"array({sponsors_sql})"
        sponsors_type = T.ArrayType(sponsors_type)

    if PARTY_ID_EXPR and isinstance(sponsors_type, T.ArrayType) and isinstance(sponsors_type.elementType, T.StructType):
        PART_B_READY = True
        elem_type = sponsors_type.elementType
        sponsor_map, sponsor_map_tbl = auto_map(elem_type, "s", list(FIELD_SPECS), "sponsor")
        top_cols = set(re.findall(r"`([^`]+)`", SPONSORS_EXPR)) | set(re.findall(r"^\s*(\w+)\s*$", SPONSORS_EXPR))
        party_only = T.StructType([f for f in party_schema.fields if f.name not in top_cols])
        party_map, party_map_tbl = auto_map(party_only, None, ["ssn"], "party", PARTY_EXCLUDE_RX)
        field_map = pd.concat([sponsor_map_tbl, party_map_tbl], ignore_index=True)
        print("party table     :", PARTY_TABLE)
        print("party id        :", PARTY_ID_EXPR)
        print("sponsors        :", sponsors_sql[:300])
        print("sponsor element schema:")
        print(elem_type.simpleString()[:3000])
        keep("b_field_mapping", field_map)

if not PART_B_READY:
    print("Part B skipped: no party table with a usable sponsors array was found. "
          "Set PARTY_TABLE, PARTY_ID_EXPR and SPONSORS_EXPR in section 1 and re-run from section 9.")

# COMMAND ----------

# ================================================================================================
# 10. PART B - SPONSOR FIELDS ON PCIS PARTY RECORDS: HOW SPARSE ARE THEY?
# One row per sponsor on a party record, with the parent party record's identity. Fill rates show
# which sponsor fields would be available to matching if sponsors become their own party records.
# ================================================================================================

if PART_B_READY:
    spark.sql(f"""
        create or replace temp view party_src as
        select {PARTY_ID_EXPR} as party_key, {sponsors_sql} as sponsors from {PARTY_TABLE}
    """)
    spark.sql("""
        create or replace temp view party_sponsor_raw as
        select party_key, pos as sponsor_seq, s
        from (select party_key, sponsors from party_src where sponsors is not null and size(sponsors) > 0)
        lateral view posexplode(sponsors) t as pos, s
    """)
    mapped = ", ".join(f"cast({sponsor_map[k]} as string) as {k}_raw" if k in sponsor_map else f"cast(null as string) as {k}_raw"
                       for k in FIELD_SPECS)
    spark.sql(f"""
        create or replace temp view party_sponsor_fields as
        select party_key, sponsor_seq, {mapped}
        from (select *, row_number() over (partition by party_key, sponsor_seq order by party_key) as rn
              from party_sponsor_raw) where rn = 1
    """)
    party_sponsors = spark.sql(f"""
        select f.*, lower(split(f.party_key, '_')[0]) as party_system, ri.identity_id as parent_identity_id,
               {ssn_sql('f.ssn_raw')} as ssn_key, {dob_sql('f.dob_raw')} as dob_norm,
               {clean_sql('f.first_name_raw')} as first_clean, {clean_sql('f.last_name_raw')} as last_clean,
               {anum_sql('f.a_number_raw')} as anum_key
        from party_sponsor_fields f
        left join record_identity ri on ri.record_key = f.party_key
    """).withColumn("name_key", F.when(F.col("first_clean").isNotNull() & F.col("last_clean").isNotNull(),
                                       F.concat_ws("|", "first_clean", "last_clean")))
    persist(party_sponsors, "party_sponsors")

    fill_fields = [k for k in FIELD_SPECS if k in sponsor_map]
    fill_exprs = ", ".join(
        [f"count_if(ssn_key is not null) as valid_ssn", f"count_if(dob_norm is not null) as valid_dob",
         f"count_if(name_key is not null) as first_and_last_name", f"count_if(anum_key is not null) as valid_a_number"]
        + [f"count_if({filled_sql(k + '_raw')} is not null) as {k}_filled" for k in fill_fields])
    b_fill = spark.sql(f"""
        select party_system, count(distinct party_key) as party_records, count(*) as sponsors_on_records,
               count(distinct parent_identity_id) as parent_identities, {fill_exprs}
        from party_sponsors group by party_system order by sponsors_on_records desc
    """).toPandas()
    keep("b_fill_by_system", b_fill)

    b_fill_all = spark.sql(f"select count(*) as sponsors_on_records, {fill_exprs} from party_sponsors").toPandas()
    tot = max(float(b_fill_all["sponsors_on_records"].iloc[0]), 1.0)
    b_fill_pct = (b_fill_all.drop(columns=["sponsors_on_records"]).T.reset_index())
    b_fill_pct.columns = ["field", "sponsors_with_value"]
    b_fill_pct["pct"] = (b_fill_pct["sponsors_with_value"] / tot).round(4)
    keep("b_fill_overall", b_fill_pct)

    per_record = spark.sql("""
        select least(n, 3) as sponsors_per_record, count(*) as party_records
        from (select party_key, count(distinct sponsor_seq) as n from party_sponsors group by party_key) group by 1 order by 1
    """).toPandas()
    per_record["sponsors_per_record"] = per_record["sponsors_per_record"].astype(int).astype(str).replace("3", "3+")
    keep("b_sponsors_per_record", per_record)

    if "form_type" in sponsor_map:
        keep("b_by_form", spark.sql("""
            select coalesce(form_type_raw, 'blank') as form_type, count(*) as sponsors_on_records,
                   count_if(ssn_key is not null) as valid_ssn, count_if(dob_norm is not null) as valid_dob
            from party_sponsors group by 1 order by 2 desc limit 30
        """).toPandas())

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(b_fill_pct))))
    ax.barh(b_fill_pct["field"], b_fill_pct["pct"] * 100, color="#4c72b0")
    label_hbars(ax, "{:.1f}")
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of sponsors on party records")
    ax.set_title(f"Sponsor fields on PCIS party records ({tot:,.0f} sponsors)")
    save_chart(fig, "05_party_sponsor_fill")
else:
    print("Part B skipped.")

# COMMAND ----------

# ================================================================================================
# 11. PART B - REPEAT SPONSORS ON PARTY RECORDS
# One row per sponsor SSN found on party records: how many party records carry them, how many
# different parent identities they sponsor, and whether name and DOB agree across those records.
# ================================================================================================

if PART_B_READY:
    sponsor_b = spark.sql("""
        select ssn_key, count(distinct party_key) as party_records, count(distinct parent_identity_id) as parent_identities,
               count(distinct name_key) as names, count(distinct dob_norm) as dobs
        from party_sponsors where ssn_key is not null group by ssn_key
    """)
    persist(sponsor_b, "sponsor_b")
    b_repeat = long_form(spark.sql(f"""
        select count(*) as sponsors, count_if(party_records > 1) as sponsors_on_2plus_records,
               count_if(parent_identities > 1) as sponsors_of_2plus_identities,
               count_if(party_records > {SHARED_SSN_FILINGS}) as ssns_over_{SHARED_SSN_FILINGS}_records
        from sponsor_b
    """).toPandas())
    keep("b_repeat_summary", b_repeat)
    cons_b = spark.sql(f"""
        select {consistency_sql('names', 'dobs')} as consistency, count(*) as sponsors
        from sponsor_b where party_records > 1 and party_records <= {SHARED_SSN_FILINGS} group by 1
    """).toPandas().set_index("consistency").reindex(CONSISTENCY_ORDER).dropna().reset_index()
    keep("b_name_dob_consistency", cons_b)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(cons_b["consistency"], cons_b["sponsors"], color="#8172b2")
    label_bars(ax)
    ax.set_title("Sponsors on 2+ party records: do name and DOB agree?")
    ax.set_ylabel("sponsors")
    ax.tick_params(axis="x", rotation=15)
    save_chart(fig, "06_party_sponsor_consistency")
else:
    print("Part B skipped.")

# COMMAND ----------

# ================================================================================================
# 12. RECORD SETS FOR THE STORED-SCORE STATUS QUO
#   Set 1 - C3 I-865 filers: I-865 receipts joined to their PCIS record (the record is the sponsor)
#   Set 2 - PCIS party records whose own SSN matches any sponsor SSN from Part A or Part B
# Each set has one row per (sponsor SSN, PCIS record) with the record's identity.
# ================================================================================================

spark.sql("""
    create or replace temp view rs_i865_filers as
    select distinct f.ssn_key as group_key, ri.record_key, ri.record_system, ri.identity_id, ri.rule_name
    from sponsor_filings f
    join record_identity ri on ri.record_receipt = f.receipt_number
    where f.source = 'C3 I-865' and f.ssn_key is not null
""")

ssn_sources_sql = "select distinct ssn_key, source from sponsor_filings where ssn_key is not null"
if PART_B_READY:
    ssn_sources_sql += (" union select distinct ssn_key, concat('PCIS party sponsors (', coalesce(party_system, 'unknown'), ')') "
                        "from party_sponsors where ssn_key is not null")
persist(spark.sql(ssn_sources_sql), "sponsor_ssn_sources")

SET2_READY = False
if PART_B_READY and "ssn" in party_map:
    party_ssn = spark.sql(f"""
        select distinct party_key, ssn_key
        from (select {PARTY_ID_EXPR} as party_key, {ssn_sql(party_map['ssn'])} as ssn_key from {PARTY_TABLE})
        where ssn_key is not null and party_key is not null
    """)
    persist(party_ssn, "party_ssn")
    spark.sql("""
        create or replace temp view rs_sponsor_in_pcis as
        select distinct ps.ssn_key as group_key, ri.record_key, ri.record_system, ri.identity_id, ri.rule_name
        from party_ssn ps
        join (select distinct ssn_key from sponsor_ssn_sources) ss on ss.ssn_key = ps.ssn_key
        join record_identity ri on ri.record_key = ps.party_key
    """)
    set2_cov = long_form(spark.sql("""
        select (select count(distinct ssn_key) from sponsor_ssn_sources) as sponsor_ssns_all_sources,
               (select count(distinct p.ssn_key) from party_ssn p
                  join (select distinct ssn_key from sponsor_ssn_sources) s on s.ssn_key = p.ssn_key) as sponsor_ssns_on_a_party_record,
               (select count(distinct group_key) from rs_sponsor_in_pcis) as sponsor_ssns_with_scored_record,
               (select count(distinct record_key) from rs_sponsor_in_pcis) as scored_party_records
    """).toPandas())
    keep("set2_coverage", set2_cov)
    SET2_READY = True
else:
    print("Set 2 skipped: needs Part B and a party-level SSN (set FIELD_OVERRIDES['party.ssn'] if the mapping missed it).")

print("record sets ready:", "rs_i865_filers" + (", rs_sponsor_in_pcis" if SET2_READY else ""))

# COMMAND ----------

# ================================================================================================
# 13. STATUS QUO FUNCTION
# For a record set: linked (all of a sponsor's records in one identity) vs split; for split
# sponsors, every same-SSN record pair in different identities with its stored score band; and
# different-SSN record pairs scoring 0.90+ (risk). Pairs are counted once regardless of which
# record was scored against which. Threshold policies are applied to the score bands.
# Sponsors with more than MAX_RECORDS_PER_SSN records are left out of the pair counts.
# ================================================================================================

RESULTS = {}


def status_quo(view, label, title):
    recs = spark.sql(f"""select distinct group_key, record_key, record_system, identity_id, rule_name
                         from {view} where identity_id is not null and group_key is not null""")
    persist(recs, f"{label}_recs")
    sc = spark.sql(f"""
        select record_key, candidate_key, score from (
            select _id as record_key, candidateSourceId as candidate_key, cast(candidateScore as double) as score,
                   row_number() over (partition by _id, candidateSourceId order by `timestamp` desc) as rn
            from {SCORES_TABLE}
            where _id in (select record_key from {label}_recs) and candidateSourceId in (select record_key from {label}_recs)
        ) where rn = 1
    """)
    persist(sc, f"{label}_scores")
    groups = spark.sql(f"""select group_key, count(distinct record_key) as records, count(distinct identity_id) as identities
                           from {label}_recs group by group_key""")
    persist(groups, f"{label}_groups")

    summary = long_form(spark.sql(f"""
        select a.*, b.scored_records from (
            select count(*) as sponsors_with_scored_record,
                   count_if(records > 1) as sponsors_2plus_records,
                   count_if(records > 1 and identities = 1) as linked_one_identity,
                   count_if(records > 1 and identities > 1) as split_across_identities,
                   count_if(records > {MAX_RECORDS_PER_SSN}) as sponsors_over_size_cap
            from {label}_groups) a
        cross join (select count(distinct record_key) as scored_records from {label}_recs) b
    """).toPandas())
    multi = metric(summary, "sponsors_2plus_records")
    summary["pct_of_2plus"] = np.where(summary["metric"].isin(["linked_one_identity", "split_across_identities"]),
                                       (summary["value"] / max(multi, 1)).round(4), np.nan)

    ident = spark.sql(f"""select least(identities, 5) as identities_per_sponsor, count(*) as sponsors
                          from {label}_groups where records > 1 group by 1 order by 1""").toPandas()
    ident["identities_per_sponsor"] = ident["identities_per_sponsor"].astype(int).astype(str).replace("5", "5+")

    split_pairs = spark.sql(f"""
        with g as (select group_key from {label}_groups
                   where records > 1 and identities > 1 and records <= {MAX_RECORDS_PER_SSN}),
        r as (select x.* from {label}_recs x join g on g.group_key = x.group_key),
        p as (select a.group_key, a.record_key as ka, b.record_key as kb, a.record_system as sa, b.record_system as sb
              from r a join r b on a.group_key = b.group_key and a.record_key < b.record_key and a.identity_id <> b.identity_id)
        select p.*, greatest(s1.score, s2.score) as score
        from p
        left join {label}_scores s1 on s1.record_key = p.ka and s1.candidate_key = p.kb
        left join {label}_scores s2 on s2.record_key = p.kb and s2.candidate_key = p.ka
    """)
    persist(split_pairs, f"{label}_split_pairs")
    bands = spark.sql(f"select sa, sb, {band_sql('score')} as band, count(*) as pairs from {label}_split_pairs group by 1, 2, 3").toPandas()
    band_tbl = spark.sql(f"""select {band_sql('score')} as band, count(*) as pairs, count(distinct group_key) as sponsors
                             from {label}_split_pairs group by 1""").toPandas()
    band_tbl = band_tbl.set_index("band").reindex(BAND_ORDER).fillna(0).astype(int).reset_index()
    band_tbl["pct_of_pairs"] = (band_tbl["pairs"] / max(band_tbl["pairs"].sum(), 1)).round(4)
    hist = spark.sql(f"""select floor(score * 50) / 50 as bin, count(*) as pairs from {label}_split_pairs
                         where score is not null group by 1 order by 1""").toPandas()

    risk = spark.sql(f"""
        with u as (select least(record_key, candidate_key) as k1, greatest(record_key, candidate_key) as k2, max(score) as score
                   from {label}_scores where record_key <> candidate_key group by 1, 2),
        k as (select distinct group_key, record_key from {label}_recs)
        select lower(split(u.k1, '_')[0]) as sa, lower(split(u.k2, '_')[0]) as sb, {band_sql('u.score')} as band,
               count(distinct u.k1, u.k2) as pairs
        from u join k a on a.record_key = u.k1 join k b on b.record_key = u.k2
        where a.group_key <> b.group_key and u.score >= 0.90
        group by 1, 2, 3
    """).toPandas()
    risk_tbl = risk.groupby("band", as_index=False)["pairs"].sum() if not risk.empty else pd.DataFrame({"band": [], "pairs": []})

    gain, loss = policy_counts(bands), policy_counts(risk)
    pol = pd.DataFrame({"policy": POLICIES,
                        "same_ssn_split_pairs_at_or_above": [gain[p] for p in POLICIES],
                        "added_vs_current": [gain[p] - gain["current"] for p in POLICIES],
                        "different_ssn_pairs_at_or_above": [loss[p] for p in POLICIES],
                        "different_ssn_added_vs_current": [loss[p] - loss["current"] for p in POLICIES]})

    versions = spark.sql(f"""select coalesce(rule_name, 'none') as rule_name, count(distinct record_key) as records
                             from {label}_recs group by 1 order by 2 desc""").toPandas()
    by_system = spark.sql(f"""select record_system, count(distinct record_key) as records, count(distinct group_key) as sponsors
                              from {label}_recs group by 1 order by 2 desc""").toPandas()

    for k, v in {"summary": summary, "identities": ident, "split_bands": band_tbl, "policies": pol,
                 "risk_bands": risk_tbl, "model_versions": versions, "systems": by_system}.items():
        TABLES[f"{label[:14]}_{k}"] = v
    print(title)
    for v in [summary, band_tbl, pol]:
        display(v)

    linked = metric(summary, "linked_one_identity")
    split = metric(summary, "split_across_identities")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    if linked + split > 0:
        axes[0].pie([linked, split], labels=["linked (1 identity)", "split (2+ identities)"], autopct="%1.1f%%",
                    colors=["#55a868", "#c44e52"], startangle=90)
    axes[0].set_title(f"{title}\nsponsors with 2+ records: {multi:,.0f}")
    axes[1].bar(band_tbl["band"], band_tbl["pairs"], color=["#999999", "#c44e52", "#dd8452", "#e6b35c", "#4c72b0"])
    label_bars(axes[1])
    axes[1].set_title("Split same-SSN pairs by stored score")
    axes[1].set_ylabel("record pairs")
    axes[1].tick_params(axis="x", rotation=15)
    axes[2].bar(hist["bin"], hist["pairs"], width=0.018, color="#8172b2", align="edge")
    for t, c in [(0.90, "#dd8452"), (0.95, "#55a868"), (0.98, "#c44e52")]:
        axes[2].axvline(t, linestyle="--", linewidth=1, color=c, label=f"{t:.2f}")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Stored scores of split pairs that were compared")
    axes[2].set_xlabel("stored score")
    axes[2].set_ylabel("record pairs")
    save_chart(fig, f"07_status_quo_{label}")

    RESULTS[label] = {"title": title, "summary": summary, "bands": band_tbl, "policies": pol, "risk": risk_tbl}
    return RESULTS[label]

# COMMAND ----------

# ================================================================================================
# 14. STATUS QUO - SET 1: C3 I-865 FILERS
# Should closely match the I-865 deck (17,136 sponsors with 2+ records, 99.3% linked, 3,039 split
# pairs). Different-SSN counts are now unique pairs, so they can be lower than the deck's 231 / 250.
# ================================================================================================

status_quo("rs_i865_filers", "set1_i865_filers", "Set 1: C3 I-865 filers")

# COMMAND ----------

# ================================================================================================
# 15. STATUS QUO - SET 2: SPONSORS ALREADY IN PCIS (ALL SPONSOR FORMS)
# Every PCIS party record whose own SSN is a sponsor SSN from any form. Answers: for people who
# are sponsors, are their existing PCIS records already in one identity, and what would 0.95 /
# 0.90 change? The second table breaks the linked rate down by the form the sponsor SSN came from.
# ================================================================================================

if SET2_READY:
    status_quo("rs_sponsor_in_pcis", "set2_sponsors_in_pcis", "Set 2: sponsors already in PCIS")
    keep("set2_by_sponsor_source", spark.sql("""
        select s.source, count(*) as sponsors_with_scored_record,
               count_if(g.records > 1) as sponsors_2plus_records,
               count_if(g.records > 1 and g.identities = 1) as linked_one_identity,
               count_if(g.records > 1 and g.identities > 1) as split_across_identities,
               round(count_if(g.records > 1 and g.identities = 1) / nullif(count_if(g.records > 1), 0), 4) as linked_rate
        from set2_sponsors_in_pcis_groups g
        join sponsor_ssn_sources s on s.ssn_key = g.group_key
        group by 1 order by 2 desc
    """).toPandas())
else:
    print("Set 2 skipped.")

# COMMAND ----------

# ================================================================================================
# 16. THRESHOLD TRADE-OFF ACROSS RECORD SETS
# For each record set, what 0.95 and 0.90 add compared with today's 0.98: same-SSN split pairs
# that would newly reach the threshold (gain) next to different-SSN pairs that would (risk).
# ================================================================================================

trade = pd.concat([r["policies"].assign(record_set=r["title"]) for r in RESULTS.values()], ignore_index=True)
trade = trade[["record_set"] + [c for c in trade.columns if c != "record_set"]]
keep("tradeoff_all_sets", trade)

flat = trade[trade["policy"].isin(["flat_0.95", "flat_0.90"])]
labels = [f"{s}\n{p.replace('flat_', 'lower to ')}" for s, p in zip(flat["record_set"], flat["policy"])]
x = np.arange(len(flat))
fig, ax = plt.subplots(figsize=(max(8, 2.6 * len(flat)), 4.5))
ax.bar(x - 0.2, flat["added_vs_current"], width=0.4, color="#4c72b0", label="same-SSN split pairs added (gain)")
ax.bar(x + 0.2, flat["different_ssn_added_vs_current"], width=0.4, color="#c44e52", label="different-SSN pairs added (risk)")
label_bars(ax)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("record pairs added vs 0.98")
ax.set_title("What lowering the threshold adds, by record set")
ax.legend(fontsize=8)
save_chart(fig, "08_tradeoff")

# COMMAND ----------

# ================================================================================================
# 17. KEY FINDINGS AND EXPORT
# Plain-language findings from the numbers above, then every table and chart goes into one Excel
# workbook (findings first, charts second) in RESULTS_DIR.
# ================================================================================================

for r in comp.itertuples():
    FINDINGS.append(f"{r.source}: {r.filings:,.0f} sponsor filings; {r.pct_ssn:.1%} have a valid SSN, "
                    f"{r.pct_dob:.1%} a DOB, {r.pct_name:.1%} a first and last name.")
FINDINGS.append(f"{metric(repeat_a, 'sponsors'):,.0f} distinct sponsor SSNs across sources; "
                f"{metric(repeat_a, 'sponsors_2plus_filings'):,.0f} have 2+ filings and "
                f"{metric(repeat_a, 'sponsors_in_2plus_sources'):,.0f} appear in 2+ sources.")
cons_tot = overall.sum()
if cons_tot:
    FINDINGS.append(f"Of sponsors with 2+ filings, {overall.get('same name, same DOB', 0) / cons_tot:.1%} carry the same name and DOB "
                    f"on every filing, {overall.get('same name, no DOB', 0) / cons_tot:.1%} the same name with no DOB, and "
                    f"{(overall.get('name differs', 0) + overall.get('name and DOB differ', 0)) / cons_tot:.1%} a different name on at least one filing.")
FINDINGS.append(f"{metric(collide, 'groups_with_2plus_ssns'):,.0f} exact name + DOB combinations are shared by 2+ different SSNs "
                f"({metric(collide, 'ssns_in_those_groups'):,.0f} SSNs).")
if PART_B_READY:
    fp = b_fill_pct.set_index("field")["pct"]
    FINDINGS.append(f"PCIS party records with sponsor data ({PARTY_TABLE}): {b_fill['party_records'].sum():,.0f} records carrying "
                    f"{tot:,.0f} sponsors; valid SSN on {fp.get('valid_ssn', 0):.1%}, DOB on {fp.get('valid_dob', 0):.1%}, "
                    f"first and last name on {fp.get('first_and_last_name', 0):.1%}.")
else:
    FINDINGS.append("PCIS party records with sponsor data: not found automatically; Part B and Set 2 did not run.")
for label, r in RESULTS.items():
    s, b, p = r["summary"], r["bands"].set_index("band"), r["policies"].set_index("policy")
    multi = metric(s, "sponsors_2plus_records")
    FINDINGS.append(f"{r['title']}: {multi:,.0f} sponsors with 2+ scored records; "
                    f"{metric(s, 'linked_one_identity') / max(multi, 1):.1%} in one identity, "
                    f"{metric(s, 'split_across_identities') / max(multi, 1):.1%} split.")
    total_pairs = int(b["pairs"].sum())
    FINDINGS.append(f"{r['title']}: {total_pairs:,} split same-SSN pairs; {int(b.loc['not compared', 'pairs']):,} never compared, "
                    f"{int(b.loc['0.98+', 'pairs']):,} already score 0.98+.")
    FINDINGS.append(f"{r['title']}: lowering to 0.95 adds {p.loc['flat_0.95', 'added_vs_current']:,} same-SSN pairs and "
                    f"{p.loc['flat_0.95', 'different_ssn_added_vs_current']:,} different-SSN pairs; lowering to 0.90 adds "
                    f"{p.loc['flat_0.90', 'added_vs_current']:,} and {p.loc['flat_0.90', 'different_ssn_added_vs_current']:,}.")

findings = pd.DataFrame({"finding": FINDINGS})
pd.set_option("display.max_colwidth", None)
display(findings)

config = pd.DataFrame(
    [["run time", RUN_STAMP], ["scores table", SCORES_TABLE], ["receipt regex on _id", RECEIPT_REGEX],
     ["party table", str(PARTY_TABLE)], ["party id", str(PARTY_ID_EXPR)], ["sponsors", str(SPONSORS_EXPR)],
     ["max records per SSN in pair counts", MAX_RECORDS_PER_SSN], ["shared SSN cut-off (filings)", SHARED_SSN_FILINGS],
     ["current threshold", CURRENT_THRESHOLD]]
    + [[f"threshold {k}", v] for k, v in SYSTEM_THRESHOLDS.items()]
    + [[f"custom {' / '.join(sorted(k))}", v] for k, v in CUSTOM_EXCEPTIONS.items()]
    + [[f"source SQL: {k}", " ".join(v.split())] for k, v in SOURCES.items()],
    columns=["setting", "value"])
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


xlsx_path = RESULTS_DIR / f"sponsor_all_forms_status_quo_{RUN_STAMP}.xlsx"
try:
    write_workbook(xlsx_path)
except OSError:
    local_copy = Path("/tmp") / xlsx_path.name
    write_workbook(local_copy)
    shutil.copy(local_copy, xlsx_path)
print("workbook saved to", xlsx_path)
