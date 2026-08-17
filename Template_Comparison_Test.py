+# Databricks notebook source

import requests, json, re, os, glob
from datetime import datetime
from difflib import SequenceMatcher
import pandas as pd

ENVIRONMENTS = {
    "staging": {
        "auth_token":          "PASTE_STAGING_TOKEN",
        "service_endpoint":    "https://pcis-search-service-staging.apps.k8s.uscis.dhs.gov/search",
        "opensearch_endpoint": "",
    },
    "prod": {
        "auth_token":          "PASTE_PROD_TOKEN",
        "service_endpoint":    "",
        "opensearch_endpoint": "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search",
    },
}

TEMPLATE_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates"

TEMPLATES = {
    "default":         {"file": f"{TEMPLATE_DIR}/search-default.yaml"},
    "first":           {"file": f"{TEMPLATE_DIR}/search-first.yaml"},
    "max-clause-test": {"file": f"{TEMPLATE_DIR}/search-max-clause-test.yaml"},
    "reduced-tiers":   {"file": f"{TEMPLATE_DIR}/search-reduced-tiers.yaml"},
    "ui":              {"file": f"{TEMPLATE_DIR}/search-ui.yaml"},
    "v7":              {"file": f"{TEMPLATE_DIR}/search-templatev7.txt"},
    "v6":              {"file": f"{TEMPLATE_DIR}/search-templatev6.txt"},
}

RUN_SERVICE = True
RUN_DIRECT  = True

PROD_LOGS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs"
RESULTS_DIR   = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"

LOG_FILES = [
    "BHUB.csv",
    "CRIS.csv",
    "ELIS.csv",
    "FIRST.csv",
    "GLOBAL.csv",
    "UIPATH.csv",
]

LOG_FILE_EXCLUDE = r"\s1\.csv$"
LOG_FILE_INCLUDE = r"\.csv$"

ONLY_CONSUMERS = []

DIAGNOSE_MISSES = True

ID_FIELD      = "identityId"
RESULT_SIZE   = 100
NAME_THRESHOLD = 0.85
VERIFY_TLS    = True
TIMEOUT_S     = 120
MAX_SEARCHES  = 0

AUTOFIX_BRACES      = True
MAX_AUTOFIX_BRACES  = 3
AUTOFIX_SELF_NESTED = True

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
    prod={"id":prod_id,"first":rnm.group(1) if rnm else "",
          "middle":(rnm.group(3) if rnm and rnm.group(3) else "") if rnm else "",
          "last":rnm.group(4) if rnm else "",
          "dob":(_g(r'"dateOfBirth":"?(\d{8})',result) or _g(r'"dob":"(\d{4}-\d{2}-\d{2})"',result).replace("-",""))}
    return {"consumer":consumer,"fields":fields,"prod":prod}
def select_log_files(folder):
    on_disk={os.path.basename(p):p for p in sorted(glob.glob(os.path.join(folder,"*.csv")))}
    used=[]; left_out=[]; missing=[]
    if LOG_FILES:
        for name in LOG_FILES:
            if name in on_disk: used.append(on_disk[name])
            else: missing.append(name)
        for name,p in on_disk.items():
            if name not in LOG_FILES: left_out.append((name,"not in LOG_FILES"))
    else:
        for name,p in on_disk.items():
            if not re.search(LOG_FILE_INCLUDE, name):
                left_out.append((name,"did not match the include pattern")); continue
            if LOG_FILE_EXCLUDE and re.search(LOG_FILE_EXCLUDE, name):
                left_out.append((name,"matched the exclude pattern")); continue
            used.append(p)
    return used, left_out, missing

def load_cases(folder):
    files, left_out, missing = select_log_files(folder)
    print(f"Log files used ({len(files)}): {[os.path.basename(p) for p in files]}")
    if missing:
        print(f"NOT FOUND on disk, named in LOG_FILES but absent ({len(missing)}): {missing}")
    if left_out:
        print(f"Log files left out ({len(left_out)}): {[n for n,_ in left_out]}")
    if not files:
        raise ValueError(f"No log files selected from {folder}. Check LOG_FILES against the file names on disk.")
    cases=[]
    per_file={}
    for path in files:
        txt=open(path,encoding="utf-8",errors="replace").read()
        starts=[m.start() for m in _START.finditer(txt)]
        recs=[txt[s:(starts[i+1] if i+1<len(starts) else len(txt))] for i,s in enumerate(starts)]
        kept=0
        for r in recs:
            c=parse_record(r)
            if ONLY_CONSUMERS and c["consumer"] not in ONLY_CONSUMERS: continue
            if c["prod"]["id"] or any(c["fields"][k] for k in ("FIRSTNAME","LASTNAME","ANUMBER","RECEIPT")):
                c["source_file"]=os.path.basename(path)
                cases.append(c); kept+=1
        per_file[os.path.basename(path)]={"rows_found":len(recs),"rows_usable":kept}
        if kept==0:
            print(f"WARNING {os.path.basename(path)} produced no usable rows out of {len(recs)} found. "
                  f"Check that the file matches the expected audit log layout.")
    if ONLY_CONSUMERS: print(f"Consumer filter active: {ONLY_CONSUMERS}")
    file_stats=pd.DataFrame([{"file":k,**v} for k,v in per_file.items()])
    return cases, [os.path.basename(p) for p in files], left_out, missing, file_stats

def _person_from_source(src, hit=None):
    nm=(src.get("biographicInfo",{}) or {}).get("name",{}) or {}
    dob=src.get("_search",{}).get("dateOfBirth","") if isinstance(src.get("_search"),dict) else src.get("dateOfBirth","")
    p={"id":str(src.get(ID_FIELD,"")),"first":nm.get("first",""),"middle":nm.get("middle",""),
       "last":nm.get("last",""),"dob":str(dob or ""),"tiers":""}
    if hit:
        mq=hit.get("matched_queries")
        if isinstance(mq,list): p["tiers"]=", ".join(str(x) for x in mq)
    return p
def _person_from_api_item(item):
    if not isinstance(item,dict): return {"id":"","first":"","middle":"","last":"","dob":"","tiers":""}
    if "biographicInfo" in item or "_search" in item: return _person_from_source(item)
    nm=item.get("name") if isinstance(item.get("name"),dict) else {}
    dob=item.get("dateOfBirth") or item.get("dob") or ""
    return {"id":str(item.get(ID_FIELD,"") or item.get("id","")),
            "first":nm.get("first",item.get("first","")) or "",
            "middle":nm.get("middle",item.get("middle","")) or "",
            "last":nm.get("last",item.get("last","")) or "",
            "dob":str(dob).replace("-",""),"tiers":""}

PH=re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")

def quote_bare_placeholders(s):
    out=[]; i=0; n=len(s); instr=False; esc=False
    while i<n:
        c=s[i]
        if instr:
            if esc: esc=False
            elif c=="\\": esc=True
            elif c=='"': instr=False
            out.append(c); i+=1; continue
        if c=='"': instr=True; out.append(c); i+=1; continue
        m=PH.match(s,i)
        if m: out.append('"'+m.group(0)+'"'); i=m.end(); continue
        out.append(c); i+=1
    return "".join(out)
def _probe(t): return PH.sub("X", quote_bare_placeholders(t))

def brace_depth(s):
    depth=0; instr=False; esc=False
    for c in s:
        if instr:
            if esc: esc=False
            elif c=="\\": esc=True
            elif c=='"': instr=False
            continue
        if c=='"': instr=True
        elif c=="{": depth+=1
        elif c=="}": depth-=1
    return depth

SELF_NESTED_RE = re.compile(r'("([^"\n]+)"\s*:\s*\{)(\s*)"\2"\s*:\s*\{')
def find_self_nested_text(t):
    return [{"field":m.group(2),"line":t[:m.start()].count("\n")+1,"span":(m.start(),m.end())}
            for m in SELF_NESTED_RE.finditer(t)]
def repair_self_nested(t, hits):
    for h in sorted(hits, key=lambda x: -x["span"][0]):
        s,e=h["span"]
        inner_open=t.index("{", t.index(f'"{h["field"]}"', t.index(f'"{h["field"]}"', s)+1))
        depth=0; i=inner_open; end=None; instr=False; esc=False
        while i<len(t):
            c=t[i]
            if instr:
                if esc: esc=False
                elif c=="\\": esc=True
                elif c=='"': instr=False
            else:
                if c=='"': instr=True
                elif c=="{": depth+=1
                elif c=="}":
                    depth-=1
                    if depth==0: end=i; break
            i+=1
        if end is None: continue
        t = t[:s] + f'"{h["field"]}": {{' + t[inner_open+1:end] + "}" + t[end+1:]
    return t

def extract_template(txt, is_yaml):
    if not is_yaml:
        return txt[txt.index("{"):], "plain template file"
    try:
        import yaml
        t=yaml.safe_load(txt)["search-config"]["similar-query-template"]
        if t: return t, "yaml parser"
    except Exception:
        pass
    key=re.search(r'similar-query-template\s*:\s*[|>]?', txt)
    if not key: raise ValueError("similar-query-template not found under search-config.")
    return txt[txt.index("{", key.end()):], "brace scan (yaml parser could not read the file)"

def load_config_scalars(txt, is_yaml):
    out={}
    if is_yaml:
        try:
            import yaml
            cfg=(yaml.safe_load(txt) or {}).get("search-config",{}) or {}
            for k,v in cfg.items():
                if k=="similar-query-template": continue
                if isinstance(v,(str,int,float,bool)): out["{{"+k.upper().replace("-","_")+"}}"]=str(v)
        except Exception:
            head=txt.split("similar-query-template")[0]
            for m in re.finditer(r"^\s{2,}([A-Za-z0-9\-]+)\s*:\s*([^\n|>#]+)$", head, re.MULTILINE):
                out["{{"+m.group(1).upper().replace("-","_")+"}}"]=m.group(2).strip()
    out.setdefault("{{SIMILAR_SIZE}}", str(RESULT_SIZE))
    return out

def leaf_clause_count(node):
    if isinstance(node,dict):
        if any(k in node for k in ("match","term","prefix","fuzzy","multi_match","match_phrase","range")): return 1
        return sum(leaf_clause_count(v) for v in node.values())
    if isinstance(node,list): return sum(leaf_clause_count(v) for v in node)
    return 0

def load_template(label, spec):
    path=spec["file"]
    is_yaml=path.lower().endswith((".yaml",".yml"))
    rec={"template":label,"file":os.path.basename(path),"is_config":is_yaml,"loaded":False,
         "source":"","brace_balance":None,"repaired_braces":0,"repaired_self_nested":0,
         "tiers":"","tier_count":None,"leaf_clauses":None,"max_expansions":None,
         "fuzzy_clauses":None,"prefix_clauses":None,"problems":"",
         "client_id":spec.get("client_id") or (re.sub(r"^search-","",os.path.splitext(os.path.basename(path))[0])
                                               if is_yaml else None)}
    problems=[]
    if not os.path.exists(path):
        rec["problems"]="file not found at this path"; return rec,None,{}
    txt=open(path).read()
    try:
        tpl,how=extract_template(txt,is_yaml)
    except Exception as e:
        rec["problems"]=f"could not extract the template: {e}"; return rec,None,{}
    rec["source"]=how
    if how.startswith("brace scan"):
        problems.append("the yaml parser could not read this file, usually a line inside the template block "
                        "dedented to column 0; the service uses a yaml parser too, so confirm the deployed copy")

    nested=find_self_nested_text(tpl)
    if nested:
        detail=", ".join(f"'{h['field']}' at template line {h['line']}" for h in nested)
        if AUTOFIX_SELF_NESTED:
            tpl=repair_self_nested(tpl,nested); rec["repaired_self_nested"]=len(nested)
            problems.append(f"a field name is nested inside itself: {detail}. OpenSearch rejects the clause, so "
                            f"every search reaching that tier fails and returns nothing. Corrected for this run; "
                            f"remove the duplicated key line in the source file.")
        else:
            problems.append(f"a field name is nested inside itself: {detail}")

    d=brace_depth(_probe(tpl)); rec["brace_balance"]=d
    if d>0:
        if AUTOFIX_BRACES and d<=MAX_AUTOFIX_BRACES:
            tpl+="}"*d; rec["repaired_braces"]=d
            problems.append(f"the template was {d} closing brace(s) short and was closed for this run; "
                            f"the source file is still malformed")
        else:
            problems.append(f"the template is {d} closing brace(s) short and cannot be used")
            rec["problems"]="; ".join(problems); return rec,None,{}
    elif d<0:
        problems.append(f"the template has {abs(d)} extra closing brace(s) and cannot be used")
        rec["problems"]="; ".join(problems); return rec,None,{}

    try:
        parsed=json.loads(_probe(tpl))
    except Exception as e:
        problems.append(f"the template is not valid JSON: {e}")
        rec["problems"]="; ".join(problems); return rec,None,{}

    tiers=re.findall(r'"_name":\s*"([^"]+)"', tpl)
    rec.update({"loaded":True,"tiers":", ".join(tiers),"tier_count":len(tiers),
                "leaf_clauses":leaf_clause_count(parsed.get("query",{})),
                "max_expansions":tpl.count("max_expansions"),
                "fuzzy_clauses":tpl.count('"fuzziness"'),
                "prefix_clauses":tpl.count('"prefix"'),
                "problems":"; ".join(problems)})
    return rec, tpl, load_config_scalars(txt,is_yaml)

def identifier_params(f):
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
def render(tpl, f, scalars, size=RESULT_SIZE):
    s=tpl
    for ph,val in scalars.items(): s=s.replace(ph,val)
    p={k:("" if v is None else str(v)) for k,v in f.items()}
    p.update(identifier_params(f))
    s=PH.sub(lambda m:(p[m.group(1)] if p.get(m.group(1)) else m.group(0)), s)
    s=re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
    s=quote_bare_placeholders(s)
    if s.lstrip()[:1]!="{": s="{"+s+"}"
    s=re.sub(r",(\s*[}\]])", r"\1", s)
    q=json.loads(s); _prune(q); _strip(q); q["size"]=size; q["track_total_hits"]=True; return q

def _fmt_dob(d):
    d=(d or "").replace("-","")
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d)==8 else ""
def pick_search_method(f):
    has_name=any(f.get(k) for k in ("FIRSTNAME","MIDDLENAME","LASTNAME"))
    if any(f.get(k) for k in ("ANUMBER","RECEIPT")) and not has_name and not f.get("DOB"):
        return "identifierSearch"
    return "advancedSearch"
def build_service_body(f, client_id):
    body={"page":0,"clientId":client_id,"searchMethodType":pick_search_method(f)}
    nm={}
    if f.get("FIRSTNAME"):  nm["first"]=f["FIRSTNAME"]
    if f.get("MIDDLENAME"): nm["middle"]=f["MIDDLENAME"]
    if f.get("LASTNAME"):   nm["last"]=f["LASTNAME"]
    if nm: body["names"]=[nm]
    d=_fmt_dob(f.get("DOB"))
    if d: body["dobs"]=[{"dob":d}]
    if f.get("COB"): body["cobs"]=[f["COB"]]
    if f.get("COC"): body["cocs"]=[f["COC"]]
    ids=[]
    if f.get("ANUMBER"): ids.append({"type":"ALIEN_NBR","value":f["ANUMBER"]})
    if f.get("RECEIPT"): ids.append({"type":"RECEIPT_NBR","value":f["RECEIPT"]})
    if ids: body["identifiers"]=ids
    return body

def call_service(run, f):
    body=build_service_body(f, run["client_id"])
    try:
        r=requests.post(run["url"], headers=run["headers"], json=body, verify=VERIFY_TLS, timeout=TIMEOUT_S)
    except Exception as e:
        return [],0,None,f"{type(e).__name__}: {e}"[:400],{"method":body["searchMethodType"]},body
    meta={"method":body["searchMethodType"]}
    if r.status_code>=400: return [],0,r.status_code,(r.text or "")[:400],meta,body
    try: j=r.json()
    except Exception: return [],0,r.status_code,"response was not JSON: "+(r.text or "")[:300],meta,body
    meta["client_id_returned"]=j.get("clientId")
    people=[]; total=0
    for key in ("exactMatches","similarMatches"):
        c=j.get(key); content=(c or {}).get("content") if isinstance(c,dict) else None
        content=content if isinstance(content,list) else []
        meta[key]=len(content); total+=(c or {}).get("totalElements",0) or 0
        people.extend(_person_from_api_item(x) for x in content)
    return people,total,r.status_code,"",meta,body

def call_direct(run, f, tpl, scalars):
    if not tpl: return [],0,None,"no template loaded",{},None
    try: body=render(tpl,f,scalars)
    except Exception as e: return [],0,None,f"build error: {e}"[:400],{},None
    if not body.get("query") or not _has_real_clauses(body["query"]):
        return [],0,None,"query built with no real clauses (all tiers pruned)",{},body
    try:
        r=requests.post(run["url"], headers=run["headers"], json=body, verify=VERIFY_TLS, timeout=TIMEOUT_S)
    except Exception as e:
        return [],0,None,f"{type(e).__name__}: {e}"[:400],{},body
    if r.status_code>=400: return [],0,r.status_code,(r.text or "")[:400],{},body
    j=r.json()
    total=(j.get("hits",{}).get("total",{}) or {}).get("value") or 0
    hits=[_person_from_source(h.get("_source",{}),h) for h in j.get("hits",{}).get("hits",[])]
    return hits,total,r.status_code,"",{"clauses":leaf_clause_count(body.get("query",{})),"took":j.get("took")},body

def fetch_identity(env_url, headers, identity_id):
    if not env_url or not identity_id: return None, "no lookup endpoint"
    body={"size":1,"query":{"term":{ID_FIELD:{"value":identity_id}}},
          "_source":{"includes":["identityId","biographicInfo.*","_search.*"]}}
    try:
        r=requests.post(env_url, headers=headers, json=body, verify=VERIFY_TLS, timeout=TIMEOUT_S)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:200]
    if r.status_code>=400: return None, f"status {r.status_code}: {(r.text or '')[:150]}"
    hits=r.json().get("hits",{}).get("hits",[])
    if not hits: return None, ""
    return _person_from_source(hits[0].get("_source",{})), ""

def classify_miss(f, indexed, lookup_error):
    if lookup_error: return "could not check the index", ""
    if indexed is None:
        return "identity is not in this index", ("the record production returned does not exist here, so no template "
                                                 "could have found it")
    notes=[]
    _,name_ok = name_match(f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"],
                           indexed["first"],indexed["middle"],indexed["last"])
    ds=dob_match(f["DOB"], indexed["dob"])
    if not name_ok and (f["FIRSTNAME"] or f["LASTNAME"]):
        notes.append(f"indexed name is '{' '.join(x for x in [indexed['first'],indexed['middle'],indexed['last']] if x)}'")
    if ds=="no":
        notes.append(f"indexed date of birth is '{indexed['dob']}'")
    if notes:
        return "identity is indexed differently from the search terms", "; ".join(notes)
    return "identity is indexed and matches the search terms", ("the record is present and consistent with the "
                                                                "input, so this is a query or ranking gap")

def diagnose(deduped, long, runs, template_recs):
    lookup={}
    for r in runs:
        if r["path"]=="direct" and r["env"] not in lookup:
            lookup[r["env"]]={"url":r["url"],"headers":r["headers"]}
    rows=[]
    for c in deduped:
        f=c["fields"]; pid=c["prod"]["id"]
        key=(c["consumer"],f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"],f["ANUMBER"],f["RECEIPT"],f["DOB"])
        sub=long[(long["consumer"]==c["consumer"]) & (long["input_name"]==
                 " ".join(x for x in [f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"]] if x)) &
                 (long["input_receipt"]==f["RECEIPT"]) & (long["input_anumber"]==f["ANUMBER"]) &
                 (long["input_dob"]==f["DOB"])]
        if not len(sub) or sub["found_log_identity"].any():
            continue
        best_rank=None
        for env,cfg in lookup.items():
            indexed,err=fetch_identity(cfg["url"],cfg["headers"],pid)
            reason,detail=classify_miss(f,indexed,err)
            rows.append({"consumer":c["consumer"],"source_file":c.get("source_file",""),
                         "input_name":" ".join(x for x in [f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"]] if x),
                         "input_dob":f["DOB"],"input_anumber":f["ANUMBER"],"input_receipt":f["RECEIPT"],
                         "log_identity_id":pid,"log_returned":c["prod"]["first"]+" "+c["prod"]["last"],
                         "environment":env,"reason":reason,"detail":detail,
                         "indexed_name":(" ".join(x for x in [indexed["first"],indexed["middle"],indexed["last"]] if x)
                                         if indexed else ""),
                         "indexed_dob":indexed["dob"] if indexed else "",
                         "lookup_error":err,
                         "runs_attempted":int(len(sub)),
                         "templates_returning_nothing":int((sub["returned_count"]==0).sum()),
                         "what_was_returned_instead":sub["returned"].iloc[0]})
    return pd.DataFrame(rows)

def score_top(f, results, prod_id=None, error=""):
    if not results:
        return {"returned":("(call failed)" if error else "(no result / not supported)"),
                "matched":"","dob":"n/a","good":None,"returned_count":0,"rank":None,"top_id":"","tiers":""}
    top=results[0]
    _,ng=name_match(f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"],top["first"],top["middle"],top["last"])
    ds=dob_match(f["DOB"],top["dob"])
    matched=[]
    rtok=[t for t in (str(top["first"]).split()+str(top["middle"]).split()+str(top["last"]).split()) if t]
    if f["FIRSTNAME"] and all(max((_ratio(t,r) for r in rtok),default=0)>=NAME_THRESHOLD for t in f["FIRSTNAME"].split()): matched.append("first")
    if f["LASTNAME"] and all(max((_ratio(t,r) for r in rtok),default=0)>=NAME_THRESHOLD for t in f["LASTNAME"].split()): matched.append("last")
    if ds in ("exact","digit-flip"): matched.append("dob")
    ids=[r["id"] for r in results]
    return {"returned":" ".join(x for x in [top["first"],top["middle"],top["last"]] if x) or "(id only, no name in response)",
            "matched":", ".join(matched),"dob":ds,"good":is_good(ng,ds),"returned_count":len(results),
            "rank":(ids.index(prod_id)+1) if (prod_id and prod_id in ids) else None,
            "top_id":top["id"],"tiers":top.get("tiers","")}

def build_runs(template_recs):
    runs=[]; skipped=[]
    for env,cfg in ENVIRONMENTS.items():
        tok=(cfg.get("auth_token") or "").strip()
        if not tok or tok.startswith("PASTE"):
            skipped.append((f"{env} (all)","no auth token set")); continue
        auth=tok if tok.startswith("Basic ") else "Basic "+tok
        base={"Content-Type":"application/json","Authorization":auth}

        svc=(cfg.get("service_endpoint") or "").strip()
        if RUN_SERVICE and svc and "<" not in svc:
            if svc.rstrip("/").endswith("_search"):
                raise ValueError(f"{env} service_endpoint points at an OpenSearch _search URL. The service path "
                                 f"sends the service request contract, which OpenSearch cannot read.")
            h=dict(base)
            for label,rec in template_recs.items():
                if not rec["client_id"]:
                    skipped.append((f"{label} / {env} / service","plain template file, no config for the service to load by name"))
                    continue
                runs.append({"key":f"{label}|{env}|service","template":label,"env":env,"path":"service",
                             "url":svc,"headers":h,"client_id":rec["client_id"]})
        elif RUN_SERVICE:
            skipped.append((f"{env} / service","service endpoint not set"))

        osq=(cfg.get("opensearch_endpoint") or "").strip()
        if RUN_DIRECT and osq and "<" not in osq:
            if not osq.rstrip("/").endswith("_search"):
                raise ValueError(f"{env} opensearch_endpoint does not end in _search. This is almost always the "
                                 f"service URL in the wrong slot. The service reads any body as its own contract, "
                                 f"finds no search terms in query DSL, and returns an empty result with a success "
                                 f"code, which looks exactly like a search that found nobody.")
            for label,rec in template_recs.items():
                if not rec["loaded"]:
                    skipped.append((f"{label} / {env} / direct",f"template did not load")); continue
                runs.append({"key":f"{label}|{env}|direct","template":label,"env":env,"path":"direct",
                             "url":osq,"headers":dict(base),"client_id":None})
        elif RUN_DIRECT:
            skipped.append((f"{env} / direct","opensearch endpoint not set"))
    return runs, skipped

template_recs={}; templates={}; scalars_by={}
for label,spec in TEMPLATES.items():
    rec,tpl,sc=load_template(label,spec)
    template_recs[label]=rec; templates[label]=tpl; scalars_by[label]=sc
config_check=pd.DataFrame(list(template_recs.values()))

print("TEMPLATE CHECK")
print(config_check[["template","file","loaded","tier_count","leaf_clauses","max_expansions",
                    "fuzzy_clauses","repaired_braces","repaired_self_nested","client_id"]].to_string(index=False))
for _,r in config_check.iterrows():
    if r["problems"]: print(f"\n  [{r['template']}] {r['problems']}")
print()

runs, skipped = build_runs(template_recs)
if not runs:
    print("Nothing runnable. Set the auth tokens and endpoints, then re-run.")
    for k,w in skipped: print(f"  skipped {k}: {w}")
else:
    print(f"{len(runs)} runs: {len(set(r['template'] for r in runs))} templates x "
          f"{len(set(r['env'] for r in runs))} environments x {len(set(r['path'] for r in runs))} paths")
    if skipped:
        print("Skipped:")
        for k,w in skipped: print(f"  {k}: {w}")
    print()

    probe={"FIRSTNAME":"MARIA","MIDDLENAME":"","LASTNAME":"GARCIA","ANUMBER":"","RECEIPT":"",
           "DOB":"19800101","COB":"","COC":""}
    pf=[]
    deployed={}
    for r in runs:
        if r["path"]=="service":
            res,_,st,err,meta,_=call_service(r,probe)
            cid=meta.get("client_id_returned")
            ok = (not cid) or (cid==r["client_id"])
            deployed[r["key"]]=ok
            note="" if ok else (f"template not deployed in {r['env']}: asked for '{r['client_id']}', "
                                f"the service used '{cid}'")
        else:
            res,_,st,err,meta,_=call_direct(r,probe,templates[r["template"]],scalars_by[r["template"]])
            deployed[r["key"]]=True; note=""
        pf.append({"run":r["key"],"template":r["template"],"environment":r["env"],"path":r["path"],
                   "url":r["url"],"status":st,"results":len(res),"error":err[:140],"note":note})
    preflight=pd.DataFrame(pf)
    print("PREFLIGHT"); print(preflight[["run","status","results","error","note"]].to_string(index=False)); print()
    nd=preflight[preflight["note"]!=""]
    if len(nd):
        print(f"{len(nd)} service runs are asking for a template that is not deployed in that environment. "
              f"Those runs are still executed and reported, but they are marked as running the default template, "
              f"not the one named. The direct path covers those templates in that environment.\n")

    cases, files_used, files_left_out, files_missing, file_stats = load_cases(PROD_LOGS_DIR)
    seen={}; deduped=[]
    for c in cases:
        f=c["fields"]
        k=(c["consumer"],f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"],f["ANUMBER"],f["RECEIPT"],f["DOB"],f["COB"],f["COC"])
        if k in seen: seen[k]["dup_count"]+=1; continue
        c2=dict(c); c2["dup_count"]=1; seen[k]=c2; deduped.append(c2)
    if MAX_SEARCHES: deduped=deduped[:MAX_SEARCHES]
    total_log_rows=sum(c["dup_count"] for c in deduped)
    print(f"{len(cases)} log rows -> {len(deduped)} distinct searches "
          f"(collapsed {len(cases)-len(deduped)} duplicates).")
    print(f"{len(runs)} runs x {len(deduped)} searches = {len(runs)*len(deduped)} calls.\n")

    rows=[]; queries=[]
    for c in deduped:
        f=c["fields"]; prod_id=c["prod"]["id"]
        base={"consumer":c["consumer"],"dup_count":c["dup_count"],
              "input_name":" ".join(x for x in [f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"]] if x),
              "input_dob":f["DOB"],"input_anumber":f["ANUMBER"],"input_receipt":f["RECEIPT"],
              "log_identity_id":prod_id or ""}
        sc=score_top(f,[c["prod"]] if prod_id else [])
        base.update({"log_returned":sc["returned"],"log_matched":sc["matched"],
                     "log_dob":sc["dob"],"log_good":sc["good"]})
        for r in runs:
            if r["path"]=="service":
                res,total,st,err,meta,sent=call_service(r,f)
            else:
                res,total,st,err,meta,sent=call_direct(r,f,templates[r["template"]],scalars_by[r["template"]])
            s=score_top(f,res,prod_id,err)
            rows.append({**base,"template":r["template"],"environment":r["env"],"path":r["path"],
                         "run":r["key"],
                         "template_actually_used":meta.get("client_id_returned") if r["path"]=="service" else r["template"],
                         "status":st,"returned":s["returned"],"matched":s["matched"],"dob":s["dob"],
                         "good":s["good"],"returned_count":s["returned_count"],"total_hits":total,
                         "log_rank":s["rank"],"found_log_identity":s["rank"] is not None,
                         "top_id":s["top_id"],"tiers_matched":s["tiers"],
                         "clauses":meta.get("clauses"),"took_ms":meta.get("took"),
                         "search_method":meta.get("method"),"error":err})
            queries.append({"run":r["key"],"consumer":c["consumer"],"input_name":base["input_name"],
                            "input_anumber":f["ANUMBER"],"input_receipt":f["RECEIPT"],
                            "status":st,"error":err,"body_sent":json.dumps(sent) if sent else ""})
    long=pd.DataFrame(rows)

    def rates(sub):
        n=len(sub); rows_=sub["dup_count"].sum()
        found=sub["found_log_identity"].sum()
        found_w=sub.loc[sub["found_log_identity"],"dup_count"].sum()
        top1=(sub["log_rank"]==1).sum()
        return {"distinct_searches":int(n),"log_rows":int(rows_),
                "found":int(found),"pct_of_searches":round(100*found/n,1) if n else None,
                "found_weighted":int(found_w),
                "pct_of_log_rows":round(100*found_w/rows_,1) if rows_ else None,
                "ranked_first":int(top1),
                "pct_ranked_first":round(100*top1/n,1) if n else None,
                "errors":int((sub["error"].fillna("")!="").sum())}

    score=[]
    for (tpl,env,path),sub in long.groupby(["template","environment","path"]):
        actually=sub["template_actually_used"].dropna().unique().tolist()
        score.append({"template":tpl,"environment":env,"path":path,
                      "template_actually_used":", ".join(str(a) for a in actually) or tpl,
                      **rates(sub)})
    scorecard=pd.DataFrame(score).sort_values(["path","environment","pct_of_searches"],
                                              ascending=[True,True,False])

    diffs=[]
    for (tpl,path),sub in long.groupby(["template","path"]):
        envs={e:rates(g) for e,g in sub.groupby("environment")}
        if "staging" in envs and "prod" in envs:
            diffs.append({"template":tpl,"path":path,
                          "staging_pct":envs["staging"]["pct_of_searches"],
                          "prod_pct":envs["prod"]["pct_of_searches"],
                          "staging_minus_prod":round(envs["staging"]["pct_of_searches"]-envs["prod"]["pct_of_searches"],1),
                          "staging_ranked_first_pct":envs["staging"]["pct_ranked_first"],
                          "prod_ranked_first_pct":envs["prod"]["pct_ranked_first"]})
    env_diff=pd.DataFrame(diffs).sort_values("staging_minus_prod",ascending=False) if diffs else pd.DataFrame(
        columns=["template","path","staging_pct","prod_pct","staging_minus_prod"])

    long["run_label"]=long["template"]+" | "+long["environment"]+" | "+long["path"]
    cm=[]
    for consumer,sub in long.groupby("consumer"):
        row={"consumer":consumer,"distinct_searches":int(sub["run_label"].value_counts().iloc[0]),
             "log_rows":int(sub[sub["run_label"]==sub["run_label"].iloc[0]]["dup_count"].sum())}
        for rl,g in sub.groupby("run_label"):
            row[rl]=round(100*g["found_log_identity"].sum()/len(g),1) if len(g) else None
        cm.append(row)
    consumer_matrix=pd.DataFrame(cm)

    hard=[]
    for key,sub in long.groupby(["consumer","input_name","input_dob","input_anumber","input_receipt"]):
        if not sub["found_log_identity"].any():
            hard.append({"consumer":key[0],"input_name":key[1],"input_dob":key[2],
                         "input_anumber":key[3],"input_receipt":key[4],
                         "log_returned":sub["log_returned"].iloc[0],
                         "log_identity_id":sub["log_identity_id"].iloc[0],
                         "runs_attempted":len(sub),
                         "all_returned_something":bool((sub["returned_count"]>0).all())})
    never_found=pd.DataFrame(hard)

    miss_diag = diagnose(deduped, long, runs, template_recs) if DIAGNOSE_MISSES else pd.DataFrame()
    if len(miss_diag):
        miss_reasons = (miss_diag.groupby(["consumer","environment","reason"]).size()
                        .reset_index(name="searches").sort_values(["consumer","searches"],ascending=[True,False]))
    else:
        miss_reasons = pd.DataFrame(columns=["consumer","environment","reason","searches"])

    errors=long[long["error"].fillna("")!=""][
        ["run","consumer","input_name","input_anumber","input_receipt","status","error"]]

    print("SCORECARD, every template in every environment on the same searches")
    display(scorecard)
    if len(env_diff):
        print("\nSTAGING MINUS PROD, by template")
        display(env_diff)
    print("\nBY CONSUMER")
    display(consumer_matrix)
    if len(never_found):
        print(f"\n{len(never_found)} searches were not matched by ANY template in ANY environment")
        display(never_found)
    if len(miss_reasons):
        print("\nWHY THOSE SEARCHES WERE MISSED, by consumer")
        display(miss_reasons)
    if len(errors):
        print(f"\n{len(errors)} calls failed")
        display(errors.head(25))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out=os.path.join(RESULTS_DIR, f"Template_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

    runinfo=pd.DataFrame(
        [{"field":"run_local_time","value":datetime.now().isoformat()},
         {"field":"templates_tested","value":", ".join(sorted(set(r["template"] for r in runs)))},
         {"field":"environments","value":", ".join(sorted(set(r["env"] for r in runs)))},
         {"field":"paths","value":", ".join(sorted(set(r["path"] for r in runs)))},
         {"field":"runs_skipped","value":"; ".join(f"{k} ({w})" for k,w in skipped) or "none"},
         {"field":"prod_logs_dir","value":PROD_LOGS_DIR},
         {"field":"log_files_used","value":", ".join(files_used)},
         {"field":"log_files_missing","value":", ".join(files_missing) or "none"},
         {"field":"consumer_filter","value":", ".join(ONLY_CONSUMERS) or "none"},
         {"field":"log_files_left_out","value":"; ".join(f"{n} ({w})" for n,w in files_left_out) or "none"},
         {"field":"distinct_searches","value":len(deduped)},
         {"field":"log_rows","value":total_log_rows},
         {"field":"total_calls","value":len(long)},
         {"field":"result_size","value":RESULT_SIZE},
         {"field":"name_threshold","value":NAME_THRESHOLD},
         {"field":"failed_calls","value":len(errors)},
         {"field":"templates_repaired","value":int(config_check["repaired_self_nested"].fillna(0).gt(0).sum()
                                                   + config_check["repaired_braces"].fillna(0).gt(0).sum())}]
        + [{"field":f"url_{r['key']}","value":r["url"]} for r in runs])

    detail_cols=["consumer","dup_count","input_name","input_dob","input_anumber","input_receipt",
                 "log_returned","log_matched","log_dob","log_good","log_identity_id",
                 "environment","path","template_actually_used","status","returned","matched","dob","good",
                 "returned_count","total_hits","log_rank","found_log_identity","top_id","tiers_matched",
                 "clauses","took_ms","search_method","error"]

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        scorecard.to_excel(xl, sheet_name="Scorecard", index=False)
        if len(env_diff): env_diff.to_excel(xl, sheet_name="Staging vs prod", index=False)
        consumer_matrix.to_excel(xl, sheet_name="By consumer", index=False)
        config_check.to_excel(xl, sheet_name="Template check", index=False)

        for label in TEMPLATES.keys():
            sub=long[long["template"]==label]
            if not len(sub): continue
            name=label[:31]
            sub.sort_values(["consumer","input_name","environment","path"])[detail_cols]\
               .to_excel(xl, sheet_name=name, index=False)
        if len(never_found): never_found.to_excel(xl, sheet_name="Never matched", index=False)
        if len(miss_reasons): miss_reasons.to_excel(xl, sheet_name="Miss reasons", index=False)
        if len(miss_diag): miss_diag.to_excel(xl, sheet_name="Miss diagnosis", index=False)
        errors.to_excel(xl, sheet_name="Errors", index=False)
        file_stats.to_excel(xl, sheet_name="Log files", index=False)
        preflight.to_excel(xl, sheet_name="Preflight", index=False)
        pd.DataFrame(queries).to_excel(xl, sheet_name="Queries sent", index=False)
        runinfo.to_excel(xl, sheet_name="Run info", index=False)

    print(f"\nExcel written: {out}")
    print("\nHOW TO READ IT. Scorecard ranks every template in every environment on the same searches. "
          "Staging vs prod gives the difference per template. By consumer shows where a template is weak even "
          "when it looks strong overall. One tab per template holds every search for that template, with a row "
          "per environment and path. Template check records what each file contains and anything that had to be "
          "corrected to make it usable. Where a service run shows a template_actually_used other than the one "
          "requested, that template is not deployed in that environment and the row describes the default "
          "template instead; use the direct rows for that template there.")
