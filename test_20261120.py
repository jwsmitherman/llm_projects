# Databricks notebook source
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

IDENTITY_HOST = "https://<identity-opensearch-host>"
IDENTITY_INDEX = "iis-identity-api-alias"
PARTY_HOST = "https://<party-opensearch-host>"
PARTY_INDEX = "aim-dedup-parties-20250519"
AUTH_TOKEN = "<base64 user:pass>"
VERIFY_SSL = True

IDENTITY_IDS = [
    "47cce36642ad465783dcfda3a1f87a17",
    "7f5dae357d7c4eaf96fc8800b8deb310",
]
TARGET_IDENTITY_ID = "47cce36642ad465783dcfda3a1f87a17"
TARGET_PARTY_ID = "cis2_78484947"
EXTRA_PARTY_IDS = ["cis2_78484947"]

A_NUMBER = "A091788118"
LAST_NAME = "LEGARRETA GONZALEZ"
DOB_YYYYMMDD = "19680514"

ANUMBER_FIELD = "_search.identifiers.ALIEN_NBR"
DOB_FIELD = "_search.dateOfBirth"
LAST_NAME_FIELD = "biographicInfo.name.last.keyword"

CSP = "candidateSearchableProperties"
PARTY_FIELDS = ["identityId", "partySourceId", "recordStatus", "aNumber", "receiptNumber", "esTimeStamp", "sourceSystem", "irqStatus", "irqResolved"]

PAYLOAD_TABLE = "pcis_metadata.mv_payload_ids"
PAYLOAD_ENV = "prod"

OUTPUT_PATHS = {
    "step0_identity_mapping": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step0_identity_mapping.json",
    "step0_party_mapping": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step0_party_mapping.json",
    "step1a_identity_mget": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1a_identity_mget.json",
    "step1b_party_mget": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1b_party_mget.json",
    "step1b_party_owners_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1b_party_owners.csv",
    "step1c_parties_pointing_at_47cce36642ad465783dcfda3a1f87a17": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1c_parties_pointing_at_47cce36642ad465783dcfda3a1f87a17.json",
    "step1c_parties_pointing_at_7f5dae357d7c4eaf96fc8800b8deb310": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1c_parties_pointing_at_7f5dae357d7c4eaf96fc8800b8deb310.json",
    "step1c_parties_by_identity_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1c_parties_by_identity.csv",
    "step1d_parties_by_anumber": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1d_parties_by_anumber.json",
    "step1d_parties_by_anumber_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step1d_parties_by_anumber.csv",
    "step2_raw_index_tests": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step2_raw_index_tests.json",
    "step2_raw_index_tests_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step2_raw_index_tests.csv",
    "step2_verdict": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step2_verdict.json",
    "step4_payload_timeline": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step4_payload_timeline.json",
    "step4_payload_timeline_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step4_payload_timeline.csv",
    "step5_scan_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step5_scan.csv",
    "step5_scan_summary_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step5_scan_summary.csv",
    "step5b_raw_anumber_check_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step5b_raw_anumber_check.csv",
    "step6_full_candidates": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step6_full_candidates.json",
    "step6_mlaas_request": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step6_mlaas_request.json",
    "step6_mlaas_response": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step6_mlaas_response.json",
    "step6_mlaas_scores_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step6_mlaas_scores.csv",
    "step6_mlaas_best_csv": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results/search_suppression_20261120/step6_mlaas_best.csv",
}

MASK_PHI = False
PHI_COLUMNS = ["aNumber", "firstName", "middleName", "lastName", "dob", "fullName", "joinedName", "joinedNameWithFull"]

SCAN_MAX_BATCHES = 20
SCAN_BATCH_SIZE = 500
SCAN_SLEEP_SECONDS = 0.2
SCAN_OUTPUT_TABLE = None

DEDUP_REPO_PATH = "/Workspace/Repos/<user>/pcis-matrix-identity-index/<path-to-dedup-modules>"
MLAAS_URL = "https://<mlaas-host>/<dedup-model-endpoint>"
MLAAS_HEADERS = {"Content-Type": "application/json", "Authorization": "<mlaas auth>"}
MLAAS_COMPARE_IDENTITY_IDS = ["7f5dae357d7c4eaf96fc8800b8deb310"]
MODEL_THRESHOLD = None
CPMS_THRESHOLD = None

# COMMAND ----------

def auth_header(token):
    return token if token.lower().startswith("basic ") else f"Basic {token}"

HEADERS = {"Authorization": auth_header(AUTH_TOKEN), "Content-Type": "application/json"}

def os_call(method, host, path, body=None, params=None, ok404=False):
    r = requests.request(method, f"{host}/{path}", headers=HEADERS, json=body, params=params, verify=VERIFY_SSL, timeout=120)
    if ok404 and r.status_code == 404:
        return r.json()
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:2000]}")
    return r.json()

def total_of(res):
    t = res["hits"]["total"]
    return t["value"] if isinstance(t, dict) else t

def save(key, obj):
    path = OUTPUT_PATHS[key]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"saved {path}")
    return path

def save_csv(key, df):
    path = OUTPUT_PATHS[key]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.toPandas() if hasattr(df, "toPandas") else df
    mask(out).to_csv(path, index=False)
    print(f"saved {path}")
    return path

def mask(df):
    if not MASK_PHI:
        return df
    out = df.copy()
    for c in PHI_COLUMNS:
        if c in out.columns:
            out[c] = out[c].map(lambda v: None if v in (None, "") else "***")
    return out

def flatten_mapping(props, prefix=""):
    rows = []
    for name, spec in (props or {}).items():
        path = f"{prefix}{name}"
        rows.append((path, spec.get("type", "object")))
        for sub, sspec in (spec.get("fields") or {}).items():
            rows.append((f"{path}.{sub}", sspec.get("type")))
        if "properties" in spec:
            rows.extend(flatten_mapping(spec["properties"], f"{path}."))
    return rows

def mapping_fields(host, index):
    m = os_call("GET", host, f"{index}/_mapping")
    rows = []
    for idx, spec in m.items():
        for path, typ in flatten_mapping(spec["mappings"].get("properties")):
            rows.append({"index": idx, "field": path, "type": typ})
    return pd.DataFrame(rows).drop_duplicates()

def party_ids_of(identity_source):
    out = []
    for p in identity_source.get("parties", []) or []:
        out.append(p.get("partyId") if isinstance(p, dict) else p)
    return [p for p in out if p]

def pf(name):
    return f"{CSP}.{name}"

def pget(src, name):
    return ((src or {}).get(CSP) or {}).get(name)

# COMMAND ----------

print("STEP 0  identity index mapping")
identity_map = mapping_fields(IDENTITY_HOST, IDENTITY_INDEX)
save("step0_identity_mapping", identity_map.to_dict("records"))
pattern = r"(?i)alien|anumber|dateofbirth|dob|name\.last|parties|shoulddelete|status|phantom|orphan|deleted|active"
display(identity_map[identity_map["field"].str.contains(pattern, regex=True)])

# COMMAND ----------

print("STEP 0  party index mapping")
party_map = mapping_fields(PARTY_HOST, PARTY_INDEX)
save("step0_party_mapping", party_map.to_dict("records"))
display(party_map[party_map["field"].str.startswith(CSP) & party_map["field"].str.contains(r"(?i)identity|status|anumber|partysourceid|receipt|estimestamp|sourcesystem", regex=True)])

party_types = dict(zip(party_map["field"], party_map["type"]))

def kw(field):
    if party_types.get(field) == "keyword":
        return field
    if party_types.get(f"{field}.keyword") == "keyword":
        return f"{field}.keyword"
    if field not in party_types:
        raise ValueError(f"{field} not in party mapping")
    return field

PARTY_IDENTITY_FIELD = kw(pf("identityId"))
PARTY_STATUS_FIELD = kw(pf("recordStatus"))
PARTY_ANUMBER_FIELD = kw(pf("aNumber"))
print(f"PARTY_IDENTITY_FIELD = {PARTY_IDENTITY_FIELD}")
print(f"PARTY_STATUS_FIELD = {PARTY_STATUS_FIELD}")
print(f"PARTY_ANUMBER_FIELD = {PARTY_ANUMBER_FIELD}")
PARTY_INCLUDES = ",".join(pf(f) for f in PARTY_FIELDS)

# COMMAND ----------

print("STEP 1a  identity documents")
identity_mget = os_call("POST", IDENTITY_HOST, f"{IDENTITY_INDEX}/_mget", {"ids": IDENTITY_IDS})
save("step1a_identity_mget", identity_mget)

identity_docs = {}
for d in identity_mget["docs"]:
    identity_docs[d["_id"]] = d
    src = d.get("_source", {})
    print(f"\n{d['_id']}  found={d.get('found')}  index={d.get('_index')}  version={d.get('_version')}")
    print(f"top-level keys: {sorted(src.keys())}")
    print(f"_search block: {json.dumps(src.get('_search'), default=str)[:1500] if not MASK_PHI else '(masked)'}")
    print(f"parties listed ({len(party_ids_of(src))}):")
    for p in src.get("parties", []) or []:
        print(f"   {p}")

TARGET_CONCRETE_INDEX = identity_docs.get(TARGET_IDENTITY_ID, {}).get("_index", IDENTITY_INDEX)

# COMMAND ----------

def party_row(d, listed=None):
    src = d.get("_source", {}) or {}
    owner = pget(src, "identityId")
    row = {
        "partyId": d["_id"],
        "found": d.get("found", True),
        "current_identity": owner,
        "recordStatus": pget(src, "recordStatus"),
        "passes_processed_filter": pget(src, "recordStatus") == "processed",
        "sourceSystem": pget(src, "sourceSystem"),
        "aNumber": pget(src, "aNumber"),
        "receiptNumber": pget(src, "receiptNumber"),
        "esTimeStamp": pget(src, "esTimeStamp"),
        "irqStatus": pget(src, "irqStatus"),
        "_version": d.get("_version"),
    }
    if listed is not None:
        row["listed_on_identity"] = ",".join(listed)
        row["owner_matches_listing"] = owner in listed
    return row

print("STEP 1b  current owner and recordStatus of every listed party")
listed_on = {}
for iid, d in identity_docs.items():
    for pid in party_ids_of(d.get("_source", {})):
        listed_on.setdefault(pid, []).append(iid)
for pid in EXTRA_PARTY_IDS:
    listed_on.setdefault(pid, [])

party_mget = os_call("POST", PARTY_HOST, f"{PARTY_INDEX}/_mget", {"ids": list(listed_on)}, params={"_source_includes": PARTY_INCLUDES})
save("step1b_party_mget", party_mget)
party_owner_df = pd.DataFrame([party_row(d, listed_on.get(d["_id"], [])) for d in party_mget["docs"]]).sort_values(["listed_on_identity", "partyId"])
save_csv("step1b_party_owners_csv", party_owner_df)
display(mask(party_owner_df))

# COMMAND ----------

print("STEP 1c  parties pointing at each identity")
reverse_rows = []
for iid in IDENTITY_IDS:
    res = os_call("POST", PARTY_HOST, f"{PARTY_INDEX}/_search", {
        "size": 200,
        "_source": [pf(f) for f in PARTY_FIELDS],
        "query": {"term": {PARTY_IDENTITY_FIELD: iid}},
    })
    save(f"step1c_parties_pointing_at_{iid}", res)
    for h in res["hits"]["hits"]:
        reverse_rows.append({"identity": iid, **party_row(h)})
    print(f"{iid}: {total_of(res)}")
save_csv("step1c_parties_by_identity_csv", pd.DataFrame(reverse_rows))
display(mask(pd.DataFrame(reverse_rows)))

# COMMAND ----------

print("STEP 1d  every party record carrying the A-number, with recordStatus and identity presence")
anum_res = os_call("POST", PARTY_HOST, f"{PARTY_INDEX}/_search", {
    "size": 500,
    "_source": [pf(f) for f in PARTY_FIELDS],
    "query": {"bool": {"should": [
        {"term": {PARTY_ANUMBER_FIELD: A_NUMBER}},
        {"term": {PARTY_ANUMBER_FIELD: A_NUMBER.lstrip("Aa")}},
    ], "minimum_should_match": 1}},
})
save("step1d_parties_by_anumber", anum_res)
anum_df = pd.DataFrame([party_row(h) for h in anum_res["hits"]["hits"]])
if not anum_df.empty:
    owner_ids = sorted({i for i in anum_df["current_identity"] if i})
    present = {}
    if owner_ids:
        chk = os_call("POST", IDENTITY_HOST, f"{IDENTITY_INDEX}/_mget", {"ids": owner_ids}, params={"_source": "false"})
        present = {d["_id"]: d.get("found", False) for d in chk["docs"]}
    anum_df["owner_in_identity_index"] = anum_df["current_identity"].map(lambda i: present.get(i, False))
    save_csv("step1d_parties_by_anumber_csv", anum_df)
    display(mask(anum_df.sort_values(["current_identity", "partyId"])))
    display(anum_df.groupby(["recordStatus", "owner_in_identity_index"], dropna=False).size().reset_index(name="parties"))
else:
    print("no party records found for the A-number; check PARTY_ANUMBER_FIELD format")

# COMMAND ----------

print("STEP 2  does the raw identity index match the target identity")
digits = A_NUMBER.lstrip("Aa")
tests = {
    "anumber_with_A": {"match": {ANUMBER_FIELD: A_NUMBER}},
    "anumber_digits_only": {"match": {ANUMBER_FIELD: digits}},
    "last_name_exact": {"term": {LAST_NAME_FIELD: LAST_NAME}},
    "dob_exact": {"term": {DOB_FIELD: DOB_YYYYMMDD}},
    "last_name_and_dob": {"bool": {"must": [{"term": {LAST_NAME_FIELD: LAST_NAME}}, {"term": {DOB_FIELD: DOB_YYYYMMDD}}]}},
}

step2_rows = []
step2_raw = {}
for name, q in tests.items():
    pinned = os_call("POST", IDENTITY_HOST, f"{IDENTITY_INDEX}/_search", {
        "size": 1, "_source": False,
        "query": {"bool": {"filter": [{"ids": {"values": [TARGET_IDENTITY_ID]}}], "must": [q]}},
    })
    open_q = os_call("POST", IDENTITY_HOST, f"{IDENTITY_INDEX}/_search", {"size": 50, "_source": False, "query": q})
    open_ids = [h["_id"] for h in open_q["hits"]["hits"]]
    expl = os_call("POST", IDENTITY_HOST, f"{TARGET_CONCRETE_INDEX}/_explain/{TARGET_IDENTITY_ID}", {"query": q}, ok404=True)
    step2_raw[name] = {"query": q, "pinned": pinned, "open": open_q, "explain": expl}
    step2_rows.append({
        "test": name,
        "target_matches": total_of(pinned) > 0,
        "explain_matched": expl.get("matched"),
        "open_query_hits": len(open_ids),
        "target_in_open_top50": TARGET_IDENTITY_ID in open_ids,
        "other_identity_in_open_top50": [i for i in IDENTITY_IDS if i != TARGET_IDENTITY_ID and i in open_ids],
        "open_ids_top10": open_ids[:10],
    })
save("step2_raw_index_tests", step2_raw)
step2_df = pd.DataFrame(step2_rows)
save_csv("step2_raw_index_tests_csv", step2_df)
display(step2_df)

# COMMAND ----------

print("STEP 2  verdict")
target_src = identity_docs.get(TARGET_IDENTITY_ID, {}).get("_source", {})
search_block_present = bool(target_src.get("_search"))
anumber_hit = bool(step2_df.loc[step2_df["test"].isin(["anumber_with_A", "anumber_digits_only"]), "target_matches"].any())
name_dob_hit = bool(step2_df.loc[step2_df["test"] == "last_name_and_dob", "target_matches"].any())
owners = party_owner_df[party_owner_df["listed_on_identity"].str.contains(TARGET_IDENTITY_ID, na=False)]
n_listed = len(owners)
n_point_here = int((owners["current_identity"] == TARGET_IDENTITY_ID).sum())
target_party = party_owner_df[party_owner_df["partyId"] == TARGET_PARTY_ID]
target_status = target_party["recordStatus"].iloc[0] if not target_party.empty else None
target_owner = target_party["current_identity"].iloc[0] if not target_party.empty else None

print(f"target identity: {TARGET_IDENTITY_ID}")
print(f"target party {TARGET_PARTY_ID}: recordStatus={target_status}, identityId={target_owner}")
print(f"_search block present: {search_block_present}")
print(f"raw index matches on A-number: {anumber_hit}")
print(f"raw index matches on last name + DOB: {name_dob_hit}")
print(f"parties listed on target: {n_listed}; pointing back to target: {n_point_here}; pointing elsewhere: {n_listed - n_point_here}")
if target_status != "processed":
    print(f"RESULT: target party is '{target_status}', so a recordStatus=processed filter would hide it; recordStatus hypothesis holds for this case")
else:
    print("RESULT: target party is processed, so a recordStatus=processed filter does not explain this case")
if anumber_hit and name_dob_hit:
    print("RESULT: raw index matches; identity is removed downstream in pcis-search-service")
elif not search_block_present:
    print("RESULT: target identity has no _search block; identity document build/update issue")
else:
    print("RESULT: _search block exists but does not match; inspect explain output and field types from Step 0")
if n_listed and n_point_here < n_listed:
    print("RESULT: target identity parties array is stale")
save("step2_verdict", {
    "target_identity": TARGET_IDENTITY_ID,
    "target_party": TARGET_PARTY_ID,
    "target_party_record_status": target_status,
    "target_party_identity_id": target_owner,
    "search_block_present": search_block_present,
    "raw_anumber_match": anumber_hit,
    "raw_last_name_dob_match": name_dob_hit,
    "parties_listed": n_listed,
    "parties_pointing_back": n_point_here,
})

# COMMAND ----------

print("STEP 4  payload timeline")
source_ids = set()
for pid in listed_on:
    if "_" in pid:
        source_ids.add(pid.split("_", 1)[1].lower())
        source_ids.add(pid.rsplit("_", 1)[-1].lower())
in_list = ",".join(f"'{s}'" for s in sorted(source_ids)) or "''"
timeline_sql = f"""
select systemName, sourceId, id_type, identifier, dt_utc, PartitionKey, uuid
from {PAYLOAD_TABLE}
where env = '{PAYLOAD_ENV}'
  and (identifier like '%{digits}%' or lower(sourceId) in ({in_list}))
order by dt_utc
"""
print(timeline_sql)
timeline_df = spark.sql(timeline_sql)
display(timeline_df)
save("step4_payload_timeline", timeline_df.toPandas().to_dict("records"))
save_csv("step4_payload_timeline_csv", timeline_df)

# COMMAND ----------

print("STEP 5  scope scan for stale identities and non-processed parties")

def scroll_identities():
    first = os_call("POST", IDENTITY_HOST, f"{IDENTITY_INDEX}/_search", {"size": SCAN_BATCH_SIZE, "_source": ["parties"], "query": {"match_all": {}}}, params={"scroll": "5m"})
    sid = first["_scroll_id"]
    page = first
    try:
        while page["hits"]["hits"]:
            yield page["hits"]["hits"]
            page = os_call("POST", IDENTITY_HOST, "_search/scroll", {"scroll": "5m", "scroll_id": sid})
            sid = page["_scroll_id"]
    finally:
        requests.delete(f"{IDENTITY_HOST}/_search/scroll", headers=HEADERS, json={"scroll_id": sid}, verify=VERIFY_SSL, timeout=60)

scan_rows = []
started = time.time()
scan_includes = f"{pf('identityId')},{pf('recordStatus')}"
for i, hits in enumerate(scroll_identities()):
    if SCAN_MAX_BATCHES is not None and i >= SCAN_MAX_BATCHES:
        break
    pids = sorted({p for h in hits for p in party_ids_of(h["_source"])})
    owner = {}
    for j in range(0, len(pids), 1000):
        chunk = pids[j:j + 1000]
        res = os_call("POST", PARTY_HOST, f"{PARTY_INDEX}/_mget", {"ids": chunk}, params={"_source_includes": scan_includes})
        for d in res["docs"]:
            s = d.get("_source") or {}
            owner[d["_id"]] = (pget(s, "identityId"), pget(s, "recordStatus")) if d.get("found") else (None, None)
    for h in hits:
        listed = party_ids_of(h["_source"])
        here = [p for p in listed if owner.get(p, (None, None))[0] == h["_id"]]
        missing = [p for p in listed if owner.get(p, (None, None))[0] is None]
        elsewhere = [p for p in listed if p not in here and p not in missing]
        non_processed = [p for p in listed if owner.get(p, (None, None))[1] not in (None, "processed")]
        scan_rows.append({
            "identity_id": h["_id"],
            "n_listed": len(listed),
            "n_point_here": len(here),
            "n_point_elsewhere": len(elsewhere),
            "n_missing": len(missing),
            "n_here_processed": sum(1 for p in here if owner.get(p, (None, None))[1] == "processed"),
            "n_non_processed": len(non_processed),
            "non_processed_statuses": ",".join(sorted({owner[p][1] for p in non_processed})),
            "elsewhere_party_ids": ",".join(elsewhere[:20]),
        })
    time.sleep(SCAN_SLEEP_SECONDS)
    print(f"batch {i + 1}: {len(scan_rows)} identities scanned, {time.time() - started:.0f}s")

scan_df = spark.createDataFrame(pd.DataFrame(scan_rows))
stale_df = scan_df.filter("n_point_elsewhere > 0 or n_missing > 0 or n_non_processed > 0").orderBy("n_point_elsewhere", ascending=False)
display(stale_df)
summary = scan_df.selectExpr(
    "count(*) as identities_scanned",
    "sum(case when n_point_elsewhere > 0 then 1 else 0 end) as with_parties_elsewhere",
    "sum(case when n_missing > 0 then 1 else 0 end) as with_missing_parties",
    "sum(case when n_non_processed > 0 then 1 else 0 end) as with_non_processed_parties",
    "sum(case when n_point_elsewhere > 0 and n_here_processed > 0 then 1 else 0 end) as stale_but_has_processed_owned_party",
)
save_csv("step5_scan_csv", scan_df)
save_csv("step5_scan_summary_csv", summary)
display(summary)
display(scan_df.filter("n_non_processed > 0").groupBy("non_processed_statuses").count().orderBy("count", ascending=False))
if SCAN_OUTPUT_TABLE:
    scan_df.write.mode("overwrite").saveAsTable(SCAN_OUTPUT_TABLE)
    print(f"wrote {SCAN_OUTPUT_TABLE}")

# COMMAND ----------

print("STEP 5b  raw A-number search check for stale identities with processed owned parties")
check_rows = []
candidates = stale_df.filter("n_here_processed > 0 and n_point_elsewhere > 0").limit(25).collect()
for r in candidates:
    owned = os_call("POST", PARTY_HOST, f"{PARTY_INDEX}/_search", {
        "size": 50, "_source": [pf("aNumber"), pf("recordStatus")],
        "query": {"bool": {"filter": [{"term": {PARTY_IDENTITY_FIELD: r["identity_id"]}}, {"term": {PARTY_STATUS_FIELD: "processed"}}]}},
    })
    anums = sorted({pget(h["_source"], "aNumber") for h in owned["hits"]["hits"] if pget(h["_source"], "aNumber")})
    for a in anums:
        hit = os_call("POST", IDENTITY_HOST, f"{IDENTITY_INDEX}/_search", {
            "size": 0,
            "query": {"bool": {"filter": [{"ids": {"values": [r["identity_id"]]}}], "must": [{"match": {ANUMBER_FIELD: a}}]}},
        })
        check_rows.append({"identity_id": r["identity_id"], "aNumber": a, "raw_index_matches": total_of(hit) > 0, "n_point_elsewhere": r["n_point_elsewhere"]})
save_csv("step5b_raw_anumber_check_csv", pd.DataFrame(check_rows))
display(mask(pd.DataFrame(check_rows)))

# COMMAND ----------

print("STEP 6  MLaaS scoring of target party against parties of the compare identities")
import sys
if DEDUP_REPO_PATH not in sys.path:
    sys.path.insert(0, DEDUP_REPO_PATH)
from party_array_data import get_party_dict, get_parties_dict, get_mlaas_declared_array, get_mlaas_existing_parties
from mlaas_comm import generate_mlaas_message

compare_ids = sorted({r["partyId"] for r in reverse_rows if r["identity"] in MLAAS_COMPARE_IDENTITY_IDS and r["partyId"] != TARGET_PARTY_ID})
fc_res = os_call("POST", PARTY_HOST, f"{PARTY_INDEX}/_mget", {"ids": [TARGET_PARTY_ID] + compare_ids}, params={"_source_includes": "fullCandidate"})
save("step6_full_candidates", fc_res)
full_candidates = {d["_id"]: d["_source"]["fullCandidate"] for d in fc_res["docs"] if d.get("found") and (d.get("_source") or {}).get("fullCandidate")}
if TARGET_PARTY_ID not in full_candidates:
    raise ValueError(f"{TARGET_PARTY_ID} has no fullCandidate in the party index")
compare_found = [p for p in compare_ids if p in full_candidates]
print(f"declared: {TARGET_PARTY_ID}; candidates: {compare_found}")

declared_dict = get_party_dict(full_candidates[TARGET_PARTY_ID])
existing_dict = get_parties_dict([full_candidates[p] for p in compare_found])
declared_array = get_mlaas_declared_array(declared_dict)
existing_parties = get_mlaas_existing_parties(existing_dict)
candidate_key = [k[0] for k in existing_parties.keys()]
mlaas_message = generate_mlaas_message(declared_array, list(existing_parties.values()))
print(f"declared permutations: {len(mlaas_message.get('declared', []))}; candidate permutations: {len(mlaas_message.get('candidates', []))}")

for c in mlaas_message.get("candidates", []):
    idx = int(c["candidateId"])
    c["partySourceId"] = candidate_key[idx] if 0 <= idx < len(candidate_key) else None
seen = set()
unique_candidates = []
for c in mlaas_message.get("candidates", []):
    sig = tuple(sorted((k, json.dumps(v, sort_keys=True, default=str)) for k, v in c.items() if k != "candidateId"))
    if sig not in seen:
        seen.add(sig)
        unique_candidates.append(c)
mlaas_message["candidates"] = unique_candidates
for c in mlaas_message["candidates"]:
    c.pop("partySourceId", None)
save("step6_mlaas_request", mlaas_message)

if not mlaas_message.get("candidates") or not mlaas_message.get("declared"):
    print("RESULT: no valid permutations; dedup would skip MLaaS for this pair")
else:
    resp = requests.post(MLAAS_URL, headers=MLAAS_HEADERS, json=mlaas_message, verify=VERIFY_SSL, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"MLaaS -> {resp.status_code}: {resp.text[:2000]}")
    mlaas_output = resp.json()
    save("step6_mlaas_response", mlaas_output)
    score_rows = []
    for out in mlaas_output if isinstance(mlaas_output, list) else [mlaas_output]:
        model = f"{out.get('model', 'Mlaas')}_{out.get('packageVersion', '')}"
        for c in out.get("candidates", []):
            idx = int(c["candidateId"])
            psid = candidate_key[idx] if 0 <= idx < len(candidate_key) else None
            score_rows.append({"model": model, "candidateId": idx, "partyId": psid, "score": c.get("score")})
    scores_df = pd.DataFrame(score_rows)
    best = scores_df.groupby(["model", "partyId"], dropna=False)["score"].max().reset_index().sort_values("score", ascending=False)
    if MODEL_THRESHOLD is not None:
        def threshold_for(psid):
            if CPMS_THRESHOLD is not None and ("cpms" in (psid or "") or "cpms" in TARGET_PARTY_ID):
                return CPMS_THRESHOLD
            return MODEL_THRESHOLD
        best["threshold"] = best["partyId"].map(threshold_for)
        best["would_match"] = best["score"] >= best["threshold"]
    save_csv("step6_mlaas_scores_csv", scores_df)
    save_csv("step6_mlaas_best_csv", best)
    display(scores_df)
    display(best)
