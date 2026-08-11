# Databricks notebook source

# ============================================================================
# SEARCH RESULTS: APPLICATION API vs RAW OPENSEARCH  (one cell, runs both)
#
# CONFIRMED from PSS run + Jay: PSS builds ONE tiered query and POSTs it to the OpenSearch _search
# endpoint. Every tier holds a clause for each field the caller supplied - INCLUDING receipt
# (_search.identifiers.RECEIPT_NBR). Receipt search is NOT missing from the real query; it was missing
# from the stale template this notebook used, which is why our receipt searches returned nothing while
# Jay's returned the right person.
#
# This script now gets results BOTH ways for every search and reports them side by side:
#   raw  -> POSTs ONE tiered query (from a master template that mirrors what PSS builds, receipt
#           included) straight to the _search endpoint - the same call Jay makes.
#   api  -> calls the application /search endpoint (PSS builds the query for you). Use once the URL +
#           body are confirmed; then api and raw should agree because both use the real query.
#
# Each result is scored against the INPUT terms (name fuzzy + DOB) - a COMPARISON, not an accuracy
# score (no ground truth). The master template is a close mirror of PSS, not PSS itself; for exact
# fidelity, prefer the api path (or paste the current generated query into the master template).
# ============================================================================
import requests, json, re, os, glob
from datetime import datetime, date
from difflib import SequenceMatcher
import pandas as pd

# ------------------------- CONFIG -------------------------------------------
AUTH_TOKEN = "PASTE_BASE64_TOKEN_HERE"

# Option A - application /search endpoint (does exact+similar+merge+DOB sort for you)
SEARCH_API_ENDPOINT = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/search"   # confirm real URL
SEARCH_CLIENT_ID    = "max-clause-test"
SEARCH_METHOD_TYPE  = "advancedSearch"

# raw path - hits the SAME OpenSearch _search endpoint PSS uses, sending the query built from the REAL
# repo config (search-max-clause-test.yaml, the reduced-tier template with the max-clause fix). PSS
# fills generic identifier slots: _search.identifiers.{{IDENTIFIER_NAME_1}} with {{IDENTIFIER_VALUE_1}},
# so a receipt maps to RECEIPT_NBR and an A-number to ALIEN_NBR - which is why receipt searches work.
OPENSEARCH_ENDPOINT   = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search"
SEARCH_CONFIG_YAML    = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-max-clause-test.yaml"

PROD_LOGS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs"
RESULTS_DIR   = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"
ID_FIELD   = "identityId"
TOP_N      = 10                     # window used to score the "top result"
RESULT_SIZE = 100                   # how many results to RETURN per search (OpenSearch caps size at 10000;
                                    # track_total_hits still reports the true total beyond this)
NAME_THRESHOLD = 0.85
VERIFY_TLS = True

auth = AUTH_TOKEN if AUTH_TOKEN.startswith("Basic ") else "Basic " + AUTH_TOKEN
HEADERS = {"Content-Type":"application/json","Authorization":auth}
CONFIGURED = AUTH_TOKEN != "PASTE_BASE64_TOKEN_HERE"

# ------------------------- RULE-BASED YARDSTICK (unchanged) -----------------
def _ratio(a,b): return SequenceMatcher(None,(a or "").upper(),(b or "").upper()).ratio()
def name_match(inf,inm,inl, rf,rm,rl, thr=NAME_THRESHOLD):
    itk=[t for t in (str(inf).split()+str(inm).split()+str(inl).split()) if t]
    rtk=[t for t in (str(rf).split()+str(rm).split()+str(rl).split()) if t]
    if not itk or not rtk: return 0.0, False
    sc=[max((_ratio(t,r) for r in rtk), default=0) for t in itk]
    return round(sum(sc)/len(sc),2), all(s>=thr for s in sc)
def _digit_flip(a,b): return len(a)==len(b) and sum(1 for x,y in zip(a,b) if x!=y)==1
def dob_match(ind,resd):
    ind=(ind or "").replace("-",""); resd=(resd or "").replace("-","")
    if not ind or not resd: return "n/a"
    if ind==resd: return "exact"
    if _digit_flip(ind,resd): return "digit-flip"
    return "no"
def is_good(ng,ds): return bool(ng) and ds in ("exact","digit-flip","n/a")

# ------------------------- LOAD + PARSE LOGS --------------------------------
_START=re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')
def _g(pat,s,d="",grp=1):
    m=re.search(pat,s); return m.group(grp) if m else d
def parse_record(rec):
    consumer=_g(r'^([A-Z0-9_]+),',rec); rec=rec.replace('""','"')
    i=rec.find('"result":'); terms=rec[:i] if i>=0 else rec; result=rec[i:] if i>=0 else ""
    mid=_g(r'"personMiddleName":(null|"[^"]*")',terms,"null")
    fields={"FIRSTNAME":_g(r'"personGivenName":"([^"]*)"',terms),
            "MIDDLENAME":"" if mid in ("null","") else mid.strip('"'),
            "LASTNAME":_g(r'"personSurName":"([^"]*)"',terms),
            "ANUMBER":_g(r'"type":"ALIEN_NBR","value":"([^"]*)"',terms),
            "RECEIPT":_g(r'"type":"RECEIPT_NBR","value":"([^"]*)"',terms),
            "DOB":(_g(r'"dob":"(\d{4}-\d{2}-\d{2})"',terms) or "").replace("-",""),
            "COB":_g(r'"cobs":\["([^"]*)"\]',terms),"COC":_g(r'"cocs":\["([^"]*)"\]',terms)}
    prod_id=_g(r'"identityId":"([0-9a-fA-F]{16,})"',result,None)
    rnm=re.search(r'"name":\{[^}]*"first":"([^"]*)"[^}]*("middle":"([^"]*)")?[^}]*"last":"([^"]*)"',result)
    prod={"id":prod_id,"first":rnm.group(1) if rnm else "","middle":(rnm.group(3) if rnm and rnm.group(3) else "") if rnm else "",
          "last":rnm.group(4) if rnm else "",
          "dob":(_g(r'"dateOfBirth":"?(\d{8})',result) or _g(r'"dob":"(\d{4}-\d{2}-\d{2})"',result).replace("-",""))}
    return {"consumer":consumer,"fields":fields,"prod":prod}
def load_cases(folder):
    cases=[]
    for path in sorted(glob.glob(os.path.join(folder,"*.csv"))):
        txt=open(path,encoding="utf-8",errors="replace").read()
        starts=[m.start() for m in _START.finditer(txt)]
        recs=[txt[s:(starts[i+1] if i+1<len(starts) else len(txt))] for i,s in enumerate(starts)]
        for r in recs:
            c=parse_record(r)
            if c["prod"]["id"] or any(c["fields"][k] for k in ("FIRSTNAME","LASTNAME","ANUMBER","RECEIPT")):
                cases.append(c)
    return cases

def _person_from_source(src):
    nm=(src.get("biographicInfo",{}) or {}).get("name",{}) or {}
    dob=src.get("_search",{}).get("dateOfBirth","") if isinstance(src.get("_search"),dict) else src.get("dateOfBirth","")
    return {"id":str(src.get(ID_FIELD,"")),"first":nm.get("first",""),"middle":nm.get("middle",""),
            "last":nm.get("last",""),"dob":str(dob or "")}

# ------------------------- OPTION A: application /search endpoint ------------
def build_api_body(f):
    body={"page":0,"pageSize":RESULT_SIZE,"clientId":SEARCH_CLIENT_ID,"searchMethodType":SEARCH_METHOD_TYPE}
    nm={}
    if f["FIRSTNAME"]: nm["first"]=f["FIRSTNAME"]
    if f["MIDDLENAME"]: nm["middle"]=f["MIDDLENAME"]
    if f["LASTNAME"]: nm["last"]=f["LASTNAME"]
    if nm: body["names"]=[nm]
    if f["DOB"]: body["dobs"]=[{"dob":f"{f['DOB'][:4]}-{f['DOB'][4:6]}-{f['DOB'][6:8]}"}]
    if f["COB"]: body["cobs"]=[f["COB"]]
    if f["COC"]: body["cocs"]=[f["COC"]]
    ids=[]
    if f["ANUMBER"]: ids.append({"type":"ALIEN_NBR","value":f["ANUMBER"]})
    if f["RECEIPT"]: ids.append({"type":"RECEIPT_NBR","value":f["RECEIPT"]})
    if ids: body["identifiers"]=ids
    return body
def results_api(f):
    r=requests.post(SEARCH_API_ENDPOINT, headers=HEADERS, json=build_api_body(f), verify=VERIFY_TLS, timeout=120)
    if r.status_code>=400: return [], 0
    j=r.json()
    out=[]; total=0
    for block in ("exactMatches","similarMatches"):   # exact first, then similar (as the API returns)
        blk=j.get(block,{}) or {}
        total += blk.get("totalElements") or len(blk.get("content",[]) or [])
        for item in blk.get("content",[]) or []:
            nm=(item.get("biographicInfo",{}) or {}).get("name",{}) or {}
            out.append({"id":str(item.get(ID_FIELD,"")),"first":nm.get("first",""),"middle":nm.get("middle",""),
                        "last":nm.get("last",""),"dob":str(item.get("dateOfBirth","") or "")})
    return out, total

# ------------------------- OPTION B: raw OpenSearch, replicate merge+sort ----
PH=re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")

def load_template_from_yaml(path):
    """Extract the similar-query-template JSON from the repo config yaml.
       Tries a proper YAML parse first (works on a clean repo file), then falls back to brace-balancing.
       If the file is malformed (e.g. a copy-paste that dropped a brace), fails with a clear message."""
    txt=open(path).read()
    # 1) clean file: let YAML parse it
    try:
        import yaml
        tpl=yaml.safe_load(txt)["search-config"]["similar-query-template"]
        json.loads(re.sub(r'\{\{[A-Z_0-9]+\}\}','X', tpl))   # sanity: is it valid JSON?
        return tpl
    except Exception:
        pass
    # 2) fallback: brace-balance from the first { after the key, skipping {{ }} and quoted strings
    key=re.search(r'similar-query-template:\s*\|', txt)
    if not key: raise ValueError("similar-query-template not found in the yaml.")
    start=txt.index("{", key.end())
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
                    tpl=txt[start:i+1]
                    json.loads(re.sub(r'\{\{[A-Z_0-9]+\}\}','X', tpl)); return tpl
        i+=1
    # 3) genuinely malformed - tell the user plainly
    clean=re.sub(r'\{\{[A-Z_0-9]+\}\}','X', txt)
    raise ValueError(f"The template in {os.path.basename(path)} does not parse - it looks malformed "
                     f"(opening braces {clean.count('{')} vs closing {clean.count('}')}). This usually means "
                     f"the file lost characters when it was copied into Databricks. Re-copy a clean version "
                     f"straight from the repo (download the raw file rather than paste it).")

def identifier_params(f):
    """Map the input's identifiers into the generic slots PSS uses (RECEIPT_NBR / ALIEN_NBR / ...)."""
    ids=[]
    if f.get("ANUMBER"): ids.append(("ALIEN_NBR", f["ANUMBER"]))
    if f.get("RECEIPT"): ids.append(("RECEIPT_NBR", f["RECEIPT"]))
    p={}
    for i,(nm,val) in enumerate(ids[:2], start=1):
        p[f"IDENTIFIER_NAME_{i}"]=nm; p[f"IDENTIFIER_VALUE_{i}"]=val
    return p
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
def _has_real_clauses(n):
    if isinstance(n,dict):
        if "bool" in n and isinstance(n["bool"],dict):
            b=n["bool"]
            if any(isinstance(b.get(k),list) and b[k] for k in ("must","should","filter")): return True
        if any(k in n for k in ("match","term","match_phrase","multi_match","prefix","fuzzy","range")): return True
        return any(_has_real_clauses(v) for v in n.values())
    if isinstance(n,list): return any(_has_real_clauses(v) for v in n)
    return False
def render(template_text, f, size=RESULT_SIZE):
    p={k:("" if v is None else str(v)) for k,v in f.items()}
    p.update(identifier_params(f))          # fill IDENTIFIER_NAME_1/VALUE_1/... the way PSS does
    s=PH.sub(lambda m:(p[m.group(1)] if p.get(m.group(1)) else m.group(0)), template_text)
    s=re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
    if s.lstrip()[:1]!="{": s="{"+s+"}"
    s=re.sub(r",(\s*[}\]])", r"\1", s)
    q=json.loads(s); _prune(q); _strip(q); q["size"]=size; q["track_total_hits"]=True; return q
def _run_raw(template_text, f):
    body=render(template_text, f)
    if not body.get("query") or not _has_real_clauses(body["query"]): return [], 0
    r=requests.post(OPENSEARCH_ENDPOINT, headers=HEADERS, json=body, verify=VERIFY_TLS, timeout=120)
    if r.status_code>=400: return [], 0
    j=r.json()
    total=(j.get("hits",{}).get("total",{}) or {}).get("value") or 0
    return [_person_from_source(h.get("_source",{})) for h in j.get("hits",{}).get("hits",[])], total
def _dob_sort_key(person, input_dob):
    st=dob_match(input_dob, person.get("dob"))
    return {"exact":0,"digit-flip":1,"n/a":2,"no":3}.get(st,3)
def results_raw(f, master_txt):
    """Send ONE tiered query (like PSS builds) to the OpenSearch _search endpoint. The query already
       sorts by _score, so results come back ranked the same way PSS returns them."""
    results, total = _run_raw(master_txt, f) if master_txt else ([],0)
    return results, total

# ------------------------- SCORE ONE TOP RESULT -----------------------------
def score_top(f, results, prod_id=None):
    if not results: return {"returned":"(no result / not supported)","matched":"","dob":"n/a","good":None,
                            "returned_count":0,"prod_rank":None}
    top=results[0]
    ns,ng=name_match(f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"], top["first"],top["middle"],top["last"])
    ds=dob_match(f["DOB"], top["dob"])
    matched=[]
    rtok=[t for t in (str(top["first"]).split()+str(top["middle"]).split()+str(top["last"]).split()) if t]
    if f["FIRSTNAME"] and all(max((_ratio(t,r) for r in rtok),default=0)>=NAME_THRESHOLD for t in f["FIRSTNAME"].split()): matched.append("first")
    if f["LASTNAME"] and all(max((_ratio(t,r) for r in rtok),default=0)>=NAME_THRESHOLD for t in f["LASTNAME"].split()): matched.append("last")
    if ds in ("exact","digit-flip"): matched.append("dob")
    # where does the identity production returned appear in THIS method's fuller result set?
    ids=[r["id"] for r in results]
    prod_rank=(ids.index(prod_id)+1) if (prod_id and prod_id in ids) else None
    return {"returned":" ".join(x for x in [top["first"],top["middle"],top["last"]] if x),
            "matched":", ".join(matched),"dob":ds,"good":is_good(ng,ds),
            "returned_count":len(results),"prod_rank":prod_rank}

# ------------------------- RUN ----------------------------------------------
if not CONFIGURED:
    print("Set AUTH_TOKEN (and the /search URL + exact/similar templates), then re-run.")
else:
    master_txt = load_template_from_yaml(SEARCH_CONFIG_YAML) if os.path.exists(SEARCH_CONFIG_YAML) else None
    if not master_txt:
        print("raw: search config yaml not found - check SEARCH_CONFIG_YAML.")
    cases=load_cases(PROD_LOGS_DIR)
    print(f"Loaded {len(cases)} searches. Running BOTH api and raw for each...\n")

    rows=[]
    for c in cases:
        f=c["fields"]
        row={"consumer":c["consumer"],
             "input_name":" ".join(x for x in [f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"]] if x),
             "input_dob":f["DOB"],"input_anumber":f["ANUMBER"],"input_receipt":f["RECEIPT"]}
        # prod baseline (from the log, not truth)
        sc=score_top(f,[c["prod"]] if c["prod"]["id"] else [])
        row.update({"prod_returned":sc["returned"],"prod_matched":sc["matched"],"prod_dob":sc["dob"],"prod_good":sc["good"]})
        prod_id=c["prod"]["id"]
        # api (application /search endpoint)
        try: res,total=results_api(f)
        except Exception: res,total=[],0
        sc=score_top(f,res,prod_id); row.update({"api_returned":sc["returned"],"api_matched":sc["matched"],
            "api_dob":sc["dob"],"api_returned_count":sc["returned_count"],"api_total_hits":total,
            "api_prod_rank":sc["prod_rank"],"api_good":sc["good"]})
        # raw (OpenSearch _search direct, ONE tiered query built like PSS - includes receipt)
        try: res,total=results_raw(f, master_txt)
        except Exception: res,total=[],0
        sc=score_top(f,res,prod_id); row.update({"raw_returned":sc["returned"],"raw_matched":sc["matched"],
            "raw_dob":sc["dob"],"raw_returned_count":sc["returned_count"],"raw_total_hits":total,
            "raw_prod_rank":sc["prod_rank"],"raw_good":sc["good"]})
        row["api_raw_agree"] = (row["api_returned"]==row["raw_returned"])
        rows.append(row)
    detail=pd.DataFrame(rows)

    good_cols=[c for c in detail.columns if c.endswith("_good") and c!="prod_good"]
    summ=[]
    for grp,sub in list(detail.groupby("consumer"))+[("OVERALL",detail)]:
        r={"consumer":grp,"searches":len(sub),"prod_good_pct":round(100*sub["prod_good"].dropna().mean(),1) if sub["prod_good"].notna().any() else None}
        for gc in good_cols:
            g=sub[gc].dropna()
            r[gc.replace("_good","_good_pct")]=round(100*g.mean(),1) if len(g) else None
        summ.append(r)
    summary=pd.DataFrame(summ)
    print("GOOD-MATCH rate: prod baseline vs api vs raw (comparison, not accuracy)")
    display(summary)
    print(f"\napi and raw returned the SAME top person on {int(detail['api_raw_agree'].sum())} of {len(detail)} searches.")
    print("\nPER-SEARCH DETAIL")
    display(detail)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out=os.path.join(RESULTS_DIR, f"Search_API_vs_Raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        detail.to_excel(xl, sheet_name="Per-search detail", index=False)
    print(f"\nExcel written: {out}")
    print("\nNOTE: Option A (api) reflects what consumers actually get - two queries, merge, and the Java "
          "DOB sort - and covers receipt lookups. Option B (raw) replicates that logic against OpenSearch "
          "directly. Where api and raw disagree, the app layer is doing something the raw query does not.")
