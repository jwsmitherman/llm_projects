# Databricks notebook source

# ============================================================================
# QUERY COMPARISON TEST  (one cell)
#
# IMPORTANT FRAMING (from the Aug 4 script accuracy meeting):
#   There is NO source of truth. Production (the legacy "core search") is known to be wrong in
#   some cases, so its result is a BASELINE, not a correct answer. This notebook therefore does
#   NOT measure "accuracy" and does NOT build a confusion matrix. It measures, for each query
#   version, how well the TOP result it returns MATCHES THE INPUT SEARCH TERMS, using a shared
#   rule-based / fuzzy yardstick. It then compares versions and highlights where they DIFFER.
#
# The rule-based definition of a "good" match (agreed as a starting point, tune as needed):
#   - NAME: every input name token fuzzily appears in the returned name (handles 2-part surnames
#           like "Munoz Garcia").
#   - DOB : exact, a single-digit flip ("fat finger", how the consumers think), or not provided.
#           (v7 also treats +/- 6 months as close; that is reported separately, not counted as good.)
#   A result is "good" if the name matches AND the DOB is exact / digit-flip / not provided.
#
# Compares any set of query versions against this same yardstick, e.g.
#   prod (legacy core search)  vs  v7 (current template)  vs  v7 + enhancements.
# v6 is intentionally excluded. A 15-20 case subset is emitted as a regression / smoke test.
# ============================================================================
import requests, json, re, os, glob
from datetime import datetime
import pandas as pd
from difflib import SequenceMatcher
from datetime import date

# ------------------------- CONFIG -------------------------------------------
ENDPOINT   = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search"
AUTH_TOKEN = "PASTE_BASE64_TOKEN_HERE"
# Query versions to compare, label -> template file. Josh currently has the v7 template; use it.
# When Arvin's tuning versions are shared as files, add them here to compare side by side:
#   A = Chris's original (current default)          C = B with tiers 10 and 13 combined
#   B = clauses removed for performance             D = C plus a name + high-DOB-boost tier (latest)
# (max_expansions=10 on the fuzzy/prefix tier is what keeps the query under the max clause count.)
# Any template file that isn't found is skipped with a warning, so missing versions don't stop the run.
VERSIONS = {
    "v7": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-templatev7.txt",
    # "D_dob_boost": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-template-D.txt",
}
INCLUDE_PROD_BASELINE = True    # score what production returned (from the log) as the "prod" column
PROD_LOGS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs"
RESULTS_DIR   = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"
ID_FIELD   = "identityId"
NAME_THRESHOLD = 0.85           # fuzzy ratio a name token must reach to count as matched
DOB_RANGE_MONTHS = 6            # v7's "close" DOB window (reported, not counted as good)
REGRESSION_N = 20               # size of the smoke-test subset to emit
MASK_PHI   = True
VERIFY_TLS = True
PLACEHOLDER_ANUMBERS = {"A000000007","A00000000","A000000000"}

auth = AUTH_TOKEN if AUTH_TOKEN.startswith("Basic ") else "Basic " + AUTH_TOKEN
HEADERS = {"Content-Type": "application/json", "Authorization": auth}
CONFIGURED = AUTH_TOKEN != "PASTE_BASE64_TOKEN_HERE"

# ------------------------- RULE-BASED FUZZY YARDSTICK -----------------------
def _ratio(a,b): return SequenceMatcher(None,(a or "").upper(),(b or "").upper()).ratio()
def name_match(inf,inm,inl, rf,rm,rl, thr=NAME_THRESHOLD):
    itk=[t for t in (str(inf).split()+str(inm).split()+str(inl).split()) if t]
    rtk=[t for t in (str(rf).split()+str(rm).split()+str(rl).split()) if t]
    if not itk or not rtk: return 0.0, False
    sc=[max((_ratio(t,r) for r in rtk), default=0) for t in itk]
    return round(sum(sc)/len(sc),2), all(s>=thr for s in sc)
def _digit_flip(a,b): return len(a)==len(b) and sum(1 for x,y in zip(a,b) if x!=y)==1
def _within_months(a,b,m):
    try:
        da=date(int(a[:4]),int(a[4:6]),int(a[6:8])); db=date(int(b[:4]),int(b[4:6]),int(b[6:8]))
        return abs((da-db).days)<=m*31
    except Exception: return False
def dob_match(ind,resd):
    if not ind or not resd: return "n/a"
    if ind==resd: return "exact"
    if _digit_flip(ind,resd): return "digit-flip"
    if _within_months(ind,resd,DOB_RANGE_MONTHS): return f"within {DOB_RANGE_MONTHS}mo"
    return "no"
def is_good(name_good, dob_status): return bool(name_good) and dob_status in ("exact","digit-flip","n/a")

# ------------------------- LOAD + PARSE LOGS --------------------------------
_START=re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')
def _g(pat,s,d="",grp=1):
    m=re.search(pat,s); return m.group(grp) if m else d
def parse_record(rec):
    consumer=_g(r'^([A-Z0-9_]+),',rec); rec=rec.replace('""','"')
    i=rec.find('"result":'); terms=rec[:i] if i>=0 else rec; result=rec[i:] if i>=0 else ""
    mid=_g(r'"personMiddleName":(null|"[^"]*")',terms,"null")
    anum=_g(r'"type":"ALIEN_NBR","value":"([^"]*)"',terms)
    if anum in PLACEHOLDER_ANUMBERS: anum=""
    fields={"FIRSTNAME":_g(r'"personGivenName":"([^"]*)"',terms),
            "MIDDLENAME":"" if mid in ("null","") else mid.strip('"'),
            "LASTNAME":_g(r'"personSurName":"([^"]*)"',terms),
            "ANUMBER":anum,"RECEIPT":_g(r'"type":"RECEIPT_NBR","value":"([^"]*)"',terms),
            "DOB":(_g(r'"dob":"(\d{4}-\d{2}-\d{2})"',terms) or "").replace("-",""),
            "COB":_g(r'"cobs":\["([^"]*)"\]',terms),"COC":_g(r'"cocs":\["([^"]*)"\]',terms)}
    # production's returned top result (baseline, NOT truth): its id + name + dob from the log
    prod_id=_g(r'"identityId":"([0-9a-fA-F]{16,})"',result,None)
    rnm=re.search(r'"name":\{[^}]*"first":"([^"]*)"[^}]*("middle":"([^"]*)")?[^}]*"last":"([^"]*)"',result)
    prod={"id":prod_id,
          "first":rnm.group(1) if rnm else "","middle":(rnm.group(3) if rnm and rnm.group(3) else "") if rnm else "",
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

# ------------------------- RENDER + SEARCH (per template) --------------------
PH=re.compile(r"\{\{\s*([A-Z_]+)\s*\}\}")
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
def render(template_text, fields):
    p={k:("" if v is None else str(v)) for k,v in fields.items()}
    s=PH.sub(lambda m:(p[m.group(1)] if p.get(m.group(1)) else m.group(0)), template_text)
    s=re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
    if s.lstrip()[:1]!="{": s="{"+s+"}"
    s=re.sub(r",(\s*[}\]])", r"\1", s)
    q=json.loads(s); _prune(q); _strip(q); q["size"]=1; return q     # only the top result is scored
def top_result(template_text, fields):
    body=render(template_text, fields)
    if "query" not in body or not body["query"]: return None
    r=requests.post(ENDPOINT, headers=HEADERS, json=body, verify=VERIFY_TLS, timeout=120)
    if r.status_code>=400: return None
    hits=r.json().get("hits",{}).get("hits",[])
    if not hits: return {"id":"","first":"","middle":"","last":"","dob":""}
    src=hits[0].get("_source",{}); nm=(src.get("biographicInfo",{}) or {}).get("name",{}) or {}
    return {"id":str(src.get(ID_FIELD,"")),"first":nm.get("first",""),"middle":nm.get("middle",""),
            "last":nm.get("last",""),"dob":str(src.get("_search",{}).get("dateOfBirth","") if isinstance(src.get("_search"),dict) else src.get("dateOfBirth",""))}

def _tok_match(in_val, res_tokens, thr=NAME_THRESHOLD):
    """Did this input term (e.g. first name) fuzzily appear in the returned name tokens?"""
    it=[t for t in str(in_val).split() if t]
    if not it: return None                      # not provided in the input -> not applicable
    return all(max((_ratio(t,r) for r in res_tokens), default=0)>=thr for t in it)

def score_result(fields, res):
    """Score one returned top result against the INPUT search terms (the shared yardstick),
       and record which specific terms (first, last, DOB) matched — the 'compare matched terms' ask."""
    if not res:
        return {"name_score":None,"name_match":None,"dob_status":"no result","good":False,"returned_name":"",
                "first_matched":None,"last_matched":None,"matched_terms":""}
    rtok=[t for t in (str(res.get("first")).split()+str(res.get("middle")).split()+str(res.get("last")).split()) if t]
    ns,ng=name_match(fields["FIRSTNAME"],fields["MIDDLENAME"],fields["LASTNAME"],
                     res.get("first"),res.get("middle"),res.get("last"))
    ds=dob_match(fields["DOB"], res.get("dob"))
    first_ok=_tok_match(fields["FIRSTNAME"], rtok)
    last_ok =_tok_match(fields["LASTNAME"],  rtok)
    # a plain list of which input terms the returned result matched
    matched=[]
    if first_ok: matched.append("first")
    if last_ok:  matched.append("last")
    if ds in ("exact","digit-flip"): matched.append("dob")
    return {"name_score":ns,"name_match":ng,"dob_status":ds,"good":is_good(ng,ds),
            "returned_name":" ".join(x for x in [res.get("first"),res.get("middle"),res.get("last")] if x),
            "first_matched":first_ok,"last_matched":last_ok,"matched_terms":", ".join(matched)}

# ------------------------- RUN ----------------------------------------------
if not CONFIGURED:
    print("Set AUTH_TOKEN and the VERSIONS templates in CONFIG, then re-run.")
else:
    templates={}
    for lbl,fp in VERSIONS.items():
        try:
            templates[lbl]=open(fp).read()
        except FileNotFoundError:
            print(f"  skipping '{lbl}': template not found at {fp}")
    if not templates:
        raise FileNotFoundError("No template files found. Fix the paths in VERSIONS (Josh has search-templatev7.txt).")
    cases=load_cases(PROD_LOGS_DIR)
    print(f"Loaded {len(cases)} searches. Scoring {list(templates)} against the input terms...\n")

    rows=[]
    for c in cases:
        f=c["fields"]
        qt=[]
        if f["FIRSTNAME"] or f["LASTNAME"]: qt.append("name")
        if f["ANUMBER"]: qt.append("A-number")
        if f["RECEIPT"]: qt.append("receipt")
        if f["DOB"]: qt.append("DOB")
        if f["COB"]: qt.append("COB")
        row={"consumer":c["consumer"],"query_type":" + ".join(qt) if qt else "none",
             "input_name":" ".join(x for x in [f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"]] if x),
             "input_dob":f["DOB"],"input_cob":f["COB"],"input_anumber":f["ANUMBER"],"input_receipt":f["RECEIPT"]}
        # prod baseline (from the log, not truth): returned name + which input terms it matched
        if INCLUDE_PROD_BASELINE:
            sc=score_result(f, c["prod"])
            row.update({"prod_returned":sc["returned_name"],"prod_matched_terms":sc["matched_terms"],
                        "prod_name_score":sc["name_score"],"prod_dob":sc["dob_status"],"prod_good":sc["good"]})
        # each query version: returned name + which input terms it matched
        for lbl,txt in templates.items():
            try: res=top_result(txt, f)
            except Exception: res=None
            sc=score_result(f, res)
            row.update({f"{lbl}_returned":sc["returned_name"],f"{lbl}_matched_terms":sc["matched_terms"],
                        f"{lbl}_name_score":sc["name_score"],f"{lbl}_dob":sc["dob_status"],f"{lbl}_good":sc["good"]})
        rows.append(row)
    detail=pd.DataFrame(rows)

    # summary: "good match to input" rate per version, by consumer + overall (NOT accuracy)
    good_cols=[c for c in detail.columns if c.endswith("_good")]
    summ=[]
    for grp,sub in list(detail.groupby("consumer"))+[("OVERALL",detail)]:
        r={"consumer":grp,"searches":len(sub)}
        for gc in good_cols:
            r[gc.replace("_good","_good_pct")]=round(100*sub[gc].mean(),1) if len(sub) else None
        summ.append(r)
    summary=pd.DataFrame(summ)
    print("GOOD-MATCH-TO-INPUT rate by version (rule-based yardstick; NOT accuracy, NO ground truth)")
    display(summary)

    # where versions DIFFER on the good/not-good call (the useful signal, per the meeting)
    if len(good_cols)>=2:
        detail["versions_disagree"]=detail[good_cols].nunique(axis=1)>1
        print(f"\n{int(detail['versions_disagree'].sum())} of {len(detail)} searches: the versions DISAGREE "
              f"on whether the returned result is a good match to the input.")
    print("\nPER-SEARCH DETAIL")
    display(detail)

    # PHI mask for sharing
    def _ini(s): return " ".join(w[0]+"." for w in str(s).split()) if s and str(s)!="nan" else s
    out_detail=detail.copy()
    if MASK_PHI:
        for col in [c for c in out_detail.columns if "name" in c.lower() or "returned" in c.lower()]:
            out_detail[col]=out_detail[col].map(_ini)
        if "input_anumber" in out_detail: out_detail["input_anumber"]=out_detail["input_anumber"].map(lambda s:("A*****"+str(s)[-3:]) if s else s)
        if "input_receipt" in out_detail: out_detail["input_receipt"]=out_detail["input_receipt"].map(lambda s:("*****"+str(s)[-3:]) if s else s)
        if "input_dob" in out_detail: out_detail["input_dob"]=out_detail["input_dob"].map(lambda s:(str(s)[:4]) if s else s)

    # regression / smoke-test subset (15-20 cases) for future template changes
    regression=out_detail.head(REGRESSION_N)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out=os.path.join(RESULTS_DIR, f"Query_Comparison_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Good match by version", index=False)
        out_detail.to_excel(xl, sheet_name="Per-search detail", index=False)
        regression.to_excel(xl, sheet_name="Regression set", index=False)
    print(f"\nExcel written: {out}")
    print("\nNOTE: this is a COMPARISON, not an accuracy score. There is no ground truth. A 'good' result "
          "means the returned top record matched the input search terms by the rule above. Use the "
          "Per-search detail to state, case by case, why one version's result looks better than another.")
