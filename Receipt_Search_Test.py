# Databricks notebook source

# ============================================================================
# RECEIPT SEARCH TEST  (standalone diagnostic, one cell)
#
# Purpose: search specific receipt numbers DIRECTLY against OpenSearch and see exactly what comes
# back - no log parsing, no comparison. For each receipt it runs TWO queries so you can see the
# difference the main script's ranking question hinges on:
#
#   1) TEMPLATE  - builds the receipt query from the yaml config (the same way the main script does).
#   2) EXACT     - a plain exact term match on _search.identifiers.RECEIPT_NBR (a control).
#
# If the right person is #1 on EXACT but ranks lower (or not top) with TEMPLATE, the config's receipt
# clause is fuzzy/low-boost and needs the high-boost exact-receipt tier PSS uses. It prints the top
# hits (id, name, dob, score) and the exact query sent for each, so you can eyeball and compare to PSS.
# ============================================================================
import requests, json, re, os
from datetime import datetime
import pandas as pd

# ------------------------- CONFIG -------------------------------------------
AUTH_TOKEN = "PASTE_BASE64_TOKEN_HERE"
ENDPOINT   = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search"
SEARCH_CONFIG_YAML = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-max-clause-test.yaml"
RESULTS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"   # Excel written here
RECEIPT_FIELD = "_search.identifiers.RECEIPT_NBR"
ID_FIELD   = "identityId"
TOP_N      = 10
SHOW_QUERY = True                 # print the exact JSON query sent
VERIFY_TLS = True

# >>> paste the receipt numbers to test here <<<
RECEIPTS = [
    "IOE0937357943",
    "IOE0932090935",
    "MSC2490633122",
    "MSC2490014794",
    # add/replace with the receipts you want to check
]

auth = AUTH_TOKEN if AUTH_TOKEN.startswith("Basic ") else "Basic " + AUTH_TOKEN
HEADERS = {"Content-Type":"application/json","Authorization":auth}
CONFIGURED = AUTH_TOKEN != "PASTE_BASE64_TOKEN_HERE"

# ------------------------- template loader (same as main script) ------------
PH=re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")
def load_template_from_yaml(path):
    txt=open(path).read()
    try:
        import yaml
        tpl=yaml.safe_load(txt)["search-config"]["similar-query-template"]
        json.loads(re.sub(r'\{\{[A-Z_0-9]+\}\}','X', tpl)); return tpl
    except Exception: pass
    key=re.search(r'similar-query-template:\s*\|', txt); start=txt.index("{", key.end())
    depth=0; i=start; n=len(txt); instr=False; esc=False
    while i<n:
        c=txt[i]
        if instr:
            if esc: esc=False
            elif c=="\\": esc=True
            elif c=='"': instr=False
        else:
            if c=='"': instr=True
            elif c=="{":
                if txt[i:i+2]=="{{": i+=2; continue
                depth+=1
            elif c=="}":
                if txt[i:i+2]=="}}": i+=2; continue
                depth-=1
                if depth==0:
                    tpl=txt[start:i+1]; json.loads(re.sub(r'\{\{[A-Z_0-9]+\}\}','X',tpl)); return tpl
        i+=1
    raise ValueError(f"{os.path.basename(path)} did not parse (malformed - re-copy a clean file from the repo).")

def _has_ph(n):
    if isinstance(n,str): return bool(PH.search(n))
    if isinstance(n,list): return any(_has_ph(x) for x in n)
    if isinstance(n,dict): return any(_has_ph(v) for v in n.values())
    return False
def _empty_bool(n):
    b=n.get("bool") if isinstance(n,dict) else None
    return isinstance(b,dict) and not any(isinstance(b.get(k),list) and b[k] for k in ("must","should","filter","must_not"))
def _prune(n):
    if isinstance(n,dict) and isinstance(n.get("bool"),dict):
        b=n["bool"]
        for key in ("must","should"):
            if isinstance(b.get(key),list):
                for c in b[key]: _prune(c)
                b[key]=[c for c in b[key] if not _has_ph(c) and not _empty_bool(c)]
                if not b[key]: del b[key]
    for v in (n.values() if isinstance(n,dict) else n if isinstance(n,list) else []):
        if isinstance(v,(dict,list)): _prune(v)
def _strip(n):
    if isinstance(n,dict):
        for k,v in list(n.items()):
            if isinstance(v,str): n[k]=PH.sub("",v)
            elif isinstance(v,(dict,list)): _strip(v)
    elif isinstance(n,list):
        for i,v in enumerate(n):
            if isinstance(v,str): n[i]=PH.sub("",v)
            elif isinstance(v,(dict,list)): _strip(v)
def render_template(template_text, receipt, size=TOP_N):
    # fill only the receipt identifier slot; everything else prunes out
    params={"SIMILAR_SIZE":str(size),"IDENTIFIER_NAME_1":"RECEIPT_NBR","IDENTIFIER_VALUE_1":receipt}
    s=PH.sub(lambda m:(params[m.group(1)] if params.get(m.group(1)) else m.group(0)), template_text)
    s=re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
    if s.lstrip()[:1]!="{": s="{"+s+"}"
    s=re.sub(r",(\s*[}\]])", r"\1", s)
    q=json.loads(s); _prune(q); _strip(q); q["size"]=size; q["track_total_hits"]=True; return q

def exact_query(receipt, size=TOP_N):
    return {"size":size,"track_total_hits":True,
            "query":{"term":{RECEIPT_FIELD:{"value":receipt}}}}

def run(body):
    r=requests.post(ENDPOINT, headers=HEADERS, json=body, verify=VERIFY_TLS, timeout=120)
    if r.status_code>=400: return None, f"HTTP {r.status_code}: {r.text[:200]}"
    j=r.json(); total=(j.get("hits",{}).get("total",{}) or {}).get("value")
    hits=[]
    for h in j.get("hits",{}).get("hits",[]):
        src=h.get("_source",{}); nm=(src.get("biographicInfo",{}) or {}).get("name",{}) or {}
        dob=src.get("_search",{}).get("dateOfBirth","") if isinstance(src.get("_search"),dict) else src.get("dateOfBirth","")
        hits.append({"id":str(src.get(ID_FIELD,"")),
                     "name":" ".join(x for x in [nm.get("first"),nm.get("middle"),nm.get("last")] if x),
                     "dob":str(dob or ""),"score":h.get("_score")})
    return {"total":total,"hits":hits}, None

def show(label, receipt, body):
    if SHOW_QUERY:
        print(f"  [{label}] query: {json.dumps(body)[:300]}{'...' if len(json.dumps(body))>300 else ''}")
    res,err=run(body)
    rows=[]
    if err:
        print(f"  [{label}] ERROR: {err}")
        rows.append({"receipt":receipt,"method":label.strip(),"rank":None,"name":f"ERROR: {err}",
                     "dob":"","score":None,"total_matches":None,"query_sent":json.dumps(body)})
        return rows
    print(f"  [{label}] total matches: {res['total']}   top {min(TOP_N,len(res['hits']))}:")
    if not res["hits"]:
        print("      (no results)")
        rows.append({"receipt":receipt,"method":label.strip(),"rank":None,"name":"(no results)",
                     "dob":"","score":None,"total_matches":res["total"],"query_sent":json.dumps(body)})
    for rank,h in enumerate(res["hits"][:TOP_N], start=1):
        print(f"      {rank:2}. score={h['score']!s:>8}  {h['name']:<45} dob={h['dob']:<9} id={h['id'][:16]}")
        rows.append({"receipt":receipt,"method":label.strip(),"rank":rank,"name":h["name"],"dob":h["dob"],
                     "score":h["score"],"total_matches":res["total"],"id":h["id"],
                     "query_sent":json.dumps(body) if rank==1 else ""})
    return rows

# ------------------------- RUN ----------------------------------------------
if not CONFIGURED:
    print("Set AUTH_TOKEN, confirm SEARCH_CONFIG_YAML and the RECEIPTS list, then re-run.")
else:
    try:
        tpl=load_template_from_yaml(SEARCH_CONFIG_YAML)
        print(f"Loaded receipt query template from {os.path.basename(SEARCH_CONFIG_YAML)}.\n")
    except Exception as e:
        tpl=None; print(f"Could not load the yaml template: {e}\nFalling back to EXACT-only.\n")
    all_rows=[]
    for rcpt in RECEIPTS:
        print(f"RECEIPT {rcpt}")
        if tpl:
            all_rows += show("TEMPLATE (yaml config)", rcpt, render_template(tpl, rcpt))
        all_rows += show("EXACT term match       ", rcpt, exact_query(rcpt))
        print()
    print("Read it like this: if EXACT returns the right person at rank 1 but TEMPLATE ranks them lower "
          "or misses, the config's receipt clause is fuzzy/low-boost - it needs the high-boost exact-receipt "
          "tier PSS uses. Compare the TEMPLATE query above to what PSS builds for the same receipt.")

    # save every hit (both methods) to a timestamped Excel so the results are kept, not just printed
    results=pd.DataFrame(all_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out=os.path.join(RESULTS_DIR, f"Receipt_Search_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        results.to_excel(xl, sheet_name="Receipt results", index=False)
    print(f"\nExcel written: {out}")
