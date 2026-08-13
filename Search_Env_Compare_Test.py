# Databricks notebook source

# ============================================================================
# SEARCH TEST: staging vs prod, PSS api vs raw OpenSearch, one workbook
#
# For each production-log search this runs every configured path and puts the
# results side by side, with the production log as the baseline.
#
# A path is an environment, a yaml config, and a layer:
#
#   <env>_<config>_api  - POST to the PSS /search endpoint. PSS builds the
#         query itself in Java from the yaml named by clientId. This is PSS.
#         The request body is the PSS contract: names, dobs, cobs, cocs and
#         identifiers are ARRAYS, and dates are hyphenated (YYYY-MM-DD).
#   <env>_<config>_raw  - the query built here from the same yaml and sent to
#         the OpenSearch _search endpoint. A close mirror of PSS, NOT PSS.
#         This is the only path whose query text can be inspected and diffed.
#
# THREE FAILURE MODES THIS GUARDS AGAINST, all of which look identical in a
# spreadsheet (an empty result with no error):
#
#   1. ENDPOINT IN THE WRONG SLOT. The two layers take completely different
#      request bodies. Sending OpenSearch DSL to the PSS service returns
#      nothing: PSS reads the body as its own contract, finds no search terms,
#      and answers empty with a 200. Every URL is checked against its layer
#      before the run starts.
#   2. A BROKEN YAML. A template that lost a brace, or that has a field name
#      nested inside itself, either fails to load or is rejected by OpenSearch
#      at query time. Every config is validated and reported before the run.
#   3. AN UNRECOGNIZED clientId. PSS falls back to the default config without
#      raising an error, so a run reports the default's behavior under the
#      variant's name. The clientId PSS echoes back is recorded per row.
#
# ERRORS ARE NEVER FOLDED INTO EMPTY RESULTS. A failed call is recorded with
# its status code and response text and reads as "(call failed)".
#
# TIER ATTRIBUTION: the templates name each tier with _name, so OpenSearch
# returns matched_queries per hit. The raw path records which tier produced
# the top result, which is the fastest way to see whether a relevance change
# did what it intended.
#
# Duplicate log rows are collapsed; dup_count shows how many merged. This is a
# COMPARISON against what production returned, not an accuracy score - there is
# no ground truth, and production is known to be wrong in some cases.
# ============================================================================
import requests, json, re, os, glob
from datetime import datetime
from difflib import SequenceMatcher
import pandas as pd

# ------------------------- ENVIRONMENTS -------------------------------------
# Leave a URL blank or containing "<" to skip that path. Skipped paths are
# listed in the run info tab so a partial run is never mistaken for a full one.
ENVIRONMENTS = {
    "staging": {
        "auth_token":          "PASTE_STAGING_TOKEN",
        "service_endpoint":    "https://pcis-search-service-staging.apps.k8s.uscis.dhs.gov/search",
        "opensearch_endpoint": "",     # staging _search URL, when known
    },
    "prod": {
        "auth_token":          "PASTE_PROD_TOKEN",
        "service_endpoint":    "",     # prod PSS /search URL, when known
        "opensearch_endpoint": "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search",
    },
}

# ------------------------- APIGEE FRONT DOOR --------------------------------
# The front door is the same host plus the search path, with the key sent in
# the x-apigee-apikey header. When a token is set for an environment, the api
# path for that environment goes through the proxy instead of straight to the
# service - which is the path consuming systems actually use, and the only one
# where proxy quota can appear.
#
# The key is left blank here on purpose. It belongs in the vault, not in a
# notebook that gets shared, screenshotted, or committed.
APIGEE = {
    "staging": {
        "endpoint": "https://pcis-search-service-staging.apps.k8s.uscis.dhs.gov/search",
        "header":   "x-apigee-apikey",
        "token":    "",
    },
    "prod": {
        "endpoint": "",
        "header":   "x-apigee-apikey",
        "token":    "",
    },
}

# searchMethodType selection.
#   "auto"             - choose per search from the fields present (recommended)
#   "advancedSearch"   - force for every search
#   "identifierSearch" - force for every search
SEARCH_METHOD_MODE = "auto"

# ------------------------- CONFIGS UNDER TEST -------------------------------
# label -> yaml path. The clientId sent to PSS is derived from the file name
# (search-max-clause-test.yaml -> max-clause-test) unless client_id is given.
# Add more entries to compare configs against each other in the same run.
CONFIGS = {
    "max_clause_test": {
        "yaml": "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-max-clause-test.yaml",
        "client_id": None,
    },
    # "reduced_tiers": {"yaml": ".../search-reduced-tiers.yaml", "client_id": None},
    # "default":       {"yaml": ".../search-default.yaml",       "client_id": None},
}

# A template that is a brace short can be closed automatically so the run can
# proceed. The repair is always reported and the run is marked as using a
# repaired template. Set to False to stop instead.
AUTOFIX_BRACES = True
MAX_AUTOFIX_BRACES = 3

# A field name nested inside itself can be collapsed automatically so the run
# can proceed. Always reported; the file still needs fixing at source.
AUTOFIX_SELF_NESTED = True

PROD_LOGS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs"
RESULTS_DIR   = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"

ID_FIELD    = "identityId"
RESULT_SIZE = 100         # results RETURNED per search; track_total_hits reports the true total
NAME_THRESHOLD = 0.85
VERIFY_TLS  = True
TIMEOUT_S   = 120
MAX_SEARCHES = 0          # 0 = all; set a small number for a quick smoke run

# ------------------------- RULE-BASED YARDSTICK -----------------------------
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

def _person_from_source(src, hit=None):
    nm=(src.get("biographicInfo",{}) or {}).get("name",{}) or {}
    dob=src.get("_search",{}).get("dateOfBirth","") if isinstance(src.get("_search"),dict) else src.get("dateOfBirth","")
    p={"id":str(src.get(ID_FIELD,"")),"first":nm.get("first",""),"middle":nm.get("middle",""),
       "last":nm.get("last",""),"dob":str(dob or ""),"tiers":""}
    if hit:
        mq=hit.get("matched_queries")
        if isinstance(mq,list): p["tiers"]=", ".join(str(x) for x in mq)
        p["score"]=hit.get("_score")
    return p
def _person_from_api_item(item):
    """PSS result items may carry the full biographic block or only an id. Both are
       handled; a name is used for scoring when present, and the id alone still
       supports the rank comparison against what production returned."""
    if not isinstance(item,dict): return {"id":"","first":"","middle":"","last":"","dob":"","tiers":""}
    if "biographicInfo" in item or "_search" in item: return _person_from_source(item)
    nm=item.get("name") if isinstance(item.get("name"),dict) else {}
    dob=item.get("dateOfBirth") or item.get("dob") or ""
    return {"id":str(item.get(ID_FIELD,"") or item.get("id","")),
            "first":nm.get("first",item.get("first","")) or "",
            "middle":nm.get("middle",item.get("middle","")) or "",
            "last":nm.get("last",item.get("last","")) or "",
            "dob":str(dob).replace("-",""),"tiers":""}

# ------------------------- TEMPLATE FROM THE YAML CONFIG --------------------
PH=re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")

def quote_bare_placeholders(s):
    """A placeholder can sit in a VALUE position, e.g. "boost": {{EXACT_MATCH_BOOST}}.
       Left bare it is not valid JSON, so a perfectly good config fails to parse and
       looks malformed. Placeholders inside a quoted string are left alone; those
       outside one are wrapped in quotes so the document parses and the clause
       holding an unfilled placeholder can then be pruned normally."""
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
def _probe(tpl): return PH.sub("X", quote_bare_placeholders(tpl))

def brace_depth(s):
    """Net brace depth ignoring braces inside quoted strings. 0 means balanced."""
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

def find_self_nested_text(tpl):
    """Find a field name nested immediately inside itself in the raw text, e.g.

         "term": {
           "_search.dateOfBirth": {
             "_search.dateOfBirth": {   <-- duplicated key
               "value": "{{DOB}}"

       This is worth detecting before the JSON parse rather than after, because
       it breaks the document in two ways at once. OpenSearch rejects the clause
       at query time, AND the duplicated key opens a brace level that the closing
       braces never account for - so the same mistake shows up as a brace
       imbalance somewhere far from the real cause. Reported with line numbers
       relative to the start of the template."""
    hits=[]
    for m in SELF_NESTED_RE.finditer(tpl):
        line=tpl[:m.start()].count("\n")+1
        hits.append({"field":m.group(2),"template_line":line,"span":(m.start(),m.end())})
    return hits

def repair_self_nested(tpl, hits):
    """Remove the duplicated key and its matching closing brace, innermost first
       so earlier offsets stay valid."""
    for h in sorted(hits, key=lambda x: -x["span"][0]):
        s,e = h["span"]
        inner_open = tpl.index("{", tpl.index(f'"{h["field"]}"', tpl.index(f'"{h["field"]}"', s)+1))
        depth=0; i=inner_open; end=None
        instr=False; esc=False
        while i < len(tpl):
            c=tpl[i]
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
        inner_body = tpl[inner_open+1:end]
        tpl = tpl[:s] + f'"{h["field"]}": {{' + inner_body + "}" + tpl[end+1:]
    return tpl

def find_self_nested_fields(node, path=""):
    """Same defect, found on a parsed document. Kept for configs that do parse."""
    found=[]
    if isinstance(node,dict):
        for k,v in node.items():
            if k in ("term","match","prefix","fuzzy","match_phrase") and isinstance(v,dict) and len(v)==1:
                fld=next(iter(v)); inner=v[fld]
                if isinstance(inner,dict) and len(inner)==1 and next(iter(inner))==fld:
                    found.append(f"{path}/{k}: field '{fld}' is nested inside itself")
            found+=find_self_nested_fields(v, f"{path}/{k}")
    elif isinstance(node,list):
        for i,v in enumerate(node): found+=find_self_nested_fields(v, f"{path}[{i}]")
    return found

def extract_template(txt):
    """Pull similar-query-template out of the config. A proper YAML parse is tried
       first; it fails when a line inside the block scalar is dedented to column 0,
       which a hand-edited or copy-pasted file often is. The fallback takes
       everything after the key."""
    try:
        import yaml
        tpl=yaml.safe_load(txt)["search-config"]["similar-query-template"]
        if tpl: return tpl, "yaml parser"
    except Exception:
        pass
    key=re.search(r'similar-query-template\s*:\s*[|>]?', txt)
    if not key: raise ValueError("similar-query-template not found under search-config.")
    start=txt.index("{", key.end())
    return txt[start:], "brace scan (yaml parser could not read the file)"

def load_config_scalars(txt):
    """The yaml carries boosts and sizes as scalars next to the template, and a
       template may refer to them as placeholders. These must be substituted BEFORE
       pruning: a clause holding an unfilled {{EXACT_MATCH_BOOST}} looks unfilled and
       is pruned away, quietly stripping whole tiers. Key exact-match-boost fills
       {{EXACT_MATCH_BOOST}}."""
    out={}
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

def load_config(label, spec):
    """Load and validate one yaml. Returns a record describing what was found so
       problems are visible in the workbook rather than discovered as empty rows."""
    path=spec["yaml"]
    rec={"config":label,"yaml":path,"loaded":False,"source":"","brace_balance":None,
         "repaired_braces":0,"self_nested":"","repaired_self_nested":0,
         "tiers":"","leaf_clauses":None,"max_expansions":None,
         "placeholders":"","problems":"",
         "client_id":spec.get("client_id") or re.sub(r"^search-","",os.path.splitext(os.path.basename(path))[0])}
    problems=[]
    if not os.path.exists(path):
        rec["problems"]="yaml not found at this path"; return rec, None, {}
    txt=open(path).read()
    try:
        tpl, how = extract_template(txt)
    except Exception as e:
        rec["problems"]=f"could not extract the template: {e}"; return rec, None, {}
    rec["source"]=how
    if how.startswith("brace scan"):
        problems.append("the yaml parser could not read this file, usually because a line inside the "
                        "template block is dedented to column 0; PSS uses a yaml parser too, so confirm "
                        "the deployed copy is not affected")

    # Self-nested fields are checked FIRST. The duplicated key opens a brace
    # level that the closing braces never account for, so this one mistake
    # surfaces as both a rejected clause and a brace imbalance reported far
    # from its real cause. Fixing it usually fixes both.
    nested=find_self_nested_text(tpl)
    rec["self_nested"]="; ".join(f"{h['field']} at template line {h['template_line']}" for h in nested)
    if nested:
        detail=", ".join(f"'{h['field']}' (template line {h['template_line']})" for h in nested)
        if AUTOFIX_SELF_NESTED:
            tpl=repair_self_nested(tpl, nested); rec["repaired_self_nested"]=len(nested)
            problems.append(f"a field name is nested inside itself: {detail}. OpenSearch rejects this clause, "
                            f"so every search reaching that tier fails and returns nothing. It was collapsed "
                            f"automatically for this run; remove the duplicated key line in the repo copy.")
        else:
            problems.append(f"a field name is nested inside itself: {detail}. OpenSearch rejects this clause.")

    depth=brace_depth(_probe(tpl))
    rec["brace_balance"]=depth
    if depth>0:
        if AUTOFIX_BRACES and depth<=MAX_AUTOFIX_BRACES:
            tpl=tpl+("}"*depth); rec["repaired_braces"]=depth
            problems.append(f"the template was {depth} closing brace(s) short and was closed automatically "
                            f"for this run; the file itself is still malformed and needs fixing at source")
        else:
            problems.append(f"the template is {depth} closing brace(s) short and cannot be used")
            rec["problems"]="; ".join(problems); return rec, None, {}
    elif depth<0:
        problems.append(f"the template has {abs(depth)} extra closing brace(s) and cannot be used")
        rec["problems"]="; ".join(problems); return rec, None, {}

    try:
        parsed=json.loads(_probe(tpl))
    except Exception as e:
        problems.append(f"the template is not valid JSON: {e}")
        rec["problems"]="; ".join(problems); return rec, None, {}

    for n in find_self_nested_fields(parsed):
        problems.append(f"OpenSearch will reject this clause: {n}")

    rec["loaded"]=True
    rec["tiers"]=", ".join(re.findall(r'"_name":\s*"([^"]+)"', tpl))
    rec["leaf_clauses"]=leaf_clause_count(parsed.get("query",{}))
    rec["max_expansions"]=tpl.count("max_expansions")
    rec["placeholders"]=", ".join(sorted(set(PH.findall(tpl))))
    rec["problems"]="; ".join(problems)
    return rec, tpl, load_config_scalars(txt)

def identifier_params(f):
    """Map the input's identifiers into the generic slots the templates use."""
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
def render(template_text, f, scalars, size=RESULT_SIZE):
    s=template_text
    for ph,val in scalars.items(): s=s.replace(ph,val)     # config scalars first, before pruning
    p={k:("" if v is None else str(v)) for k,v in f.items()}
    p.update(identifier_params(f))
    s=PH.sub(lambda m:(p[m.group(1)] if p.get(m.group(1)) else m.group(0)), s)
    s=re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
    s=quote_bare_placeholders(s)
    if s.lstrip()[:1]!="{": s="{"+s+"}"
    s=re.sub(r",(\s*[}\]])", r"\1", s)
    q=json.loads(s); _prune(q); _strip(q); q["size"]=size; q["track_total_hits"]=True; return q

# ------------------------- PATH SETUP ---------------------------------------
def build_paths(config_recs):
    paths=[]; skipped=[]
    for env,cfg in ENVIRONMENTS.items():
        tok=(cfg.get("auth_token") or "").strip()
        if not tok or tok.startswith("PASTE"):
            skipped.append((f"{env}_*","no auth token set for this environment")); continue
        auth = tok if tok.startswith("Basic ") else "Basic "+tok
        base={"Content-Type":"application/json","Authorization":auth}

        svc=(cfg.get("service_endpoint") or "").strip()
        ap=APIGEE.get(env) or {}
        via_proxy=bool(ap.get("token") and ap.get("endpoint"))
        if via_proxy: svc=ap["endpoint"]
        if svc and "<" not in svc:
            if svc.rstrip("/").endswith("_search"):
                raise ValueError(f"{env} service_endpoint points at an OpenSearch _search URL ({svc}). "
                                 f"The api layer sends the PSS request contract, which OpenSearch cannot read.")
            h=dict(base)
            if via_proxy: h[ap["header"]]=ap["token"]
            for label,rec in config_recs.items():
                paths.append({"key":f"{env}_{label}_api","env":env,"config":label,"layer":"api",
                              "url":svc,"headers":h,"client_id":rec["client_id"]})
        else:
            skipped.append((f"{env}_*_api","service endpoint not set"))

        osq=(cfg.get("opensearch_endpoint") or "").strip()
        if osq and "<" not in osq:
            if not osq.rstrip("/").endswith("_search"):
                raise ValueError(
                    f"{env} opensearch_endpoint does not end in _search ({osq}). This is almost always the "
                    f"PSS service URL pasted into the wrong slot. The raw layer sends OpenSearch query DSL; "
                    f"the PSS service reads any body as its own contract, finds no search terms in DSL, and "
                    f"returns an empty result with a 200 - which looks exactly like a search that found "
                    f"nobody. Correct the URL before running.")
            for label,rec in config_recs.items():
                if not rec["loaded"]:
                    skipped.append((f"{env}_{label}_raw",f"config '{label}' did not load")); continue
                paths.append({"key":f"{env}_{label}_raw","env":env,"config":label,"layer":"raw",
                              "url":osq,"headers":dict(base),"client_id":None})
        else:
            skipped.append((f"{env}_*_raw","opensearch endpoint not set"))
    return paths, skipped

# ------------------------- CALLS --------------------------------------------
# Both layers return (results, total, status, error, meta, body_sent).
# A failure is NEVER folded into an empty result.
def _fmt_dob(d):
    """PSS expects a hyphenated date; the logs carry YYYYMMDD. The unhyphenated
       form is accepted and matches nothing."""
    d=(d or "").replace("-","")
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d)==8 else ""

def pick_search_method(f):
    """The service exposes more than one search method, and the method has to
       match the shape of the request:

         identifierSearch - the request carries only identifiers (a receipt or
                            an A-number) and no name or date of birth. This is
                            what the CRIS, UIPATH and GLOBAL log rows look like.
         advancedSearch   - the request carries a name, a date of birth, or a
                            combination of terms.

       Sending advancedSearch for an identifier-only request is accepted and
       returns nothing useful, because the query the service builds for it
       expects name terms that are not there. That failure is silent, so it is
       decided here from the fields present rather than assumed."""
    if SEARCH_METHOD_MODE != "auto":
        return SEARCH_METHOD_MODE
    has_name = any(f.get(k) for k in ("FIRSTNAME","MIDDLENAME","LASTNAME"))
    has_dob  = bool(f.get("DOB"))
    has_id   = any(f.get(k) for k in ("ANUMBER","RECEIPT"))
    if has_id and not has_name and not has_dob:
        return "identifierSearch"
    return "advancedSearch"

def build_api_body(f, client_id):
    """The PSS contract: names, dobs, cobs, cocs and identifiers are ARRAYS."""
    method=pick_search_method(f)
    body={"page":0,"clientId":client_id,"searchMethodType":method}
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

def call_api(path, f):
    body=build_api_body(f, path["client_id"])
    try:
        r=requests.post(path["url"], headers=path["headers"], json=body, verify=VERIFY_TLS, timeout=TIMEOUT_S)
    except Exception as e:
        return [],0,None,f"{type(e).__name__}: {e}"[:400],{"method":body.get("searchMethodType")},body
    if r.status_code>=400:
        return [],0,r.status_code,(r.text or "")[:400],{"method":body.get("searchMethodType")},body
    try: j=r.json()
    except Exception:
        return [],0,r.status_code,"response was not JSON: "+(r.text or "")[:300],{"method":body.get("searchMethodType")},body
    meta={"client_id_returned":j.get("clientId"),"method":body.get("searchMethodType")}
    people=[]; total=0
    for key in ("exactMatches","similarMatches"):
        c=j.get(key); content=(c or {}).get("content") if isinstance(c,dict) else None
        content=content if isinstance(content,list) else []
        meta[key]=len(content); total+=(c or {}).get("totalElements",0) or 0
        people.extend(_person_from_api_item(x) for x in content)
    return people,total,r.status_code,"",meta,body

def call_raw(path, f, template_text, scalars):
    if not template_text: return [],0,None,"no template loaded",{},None
    try: body=render(template_text, f, scalars)
    except Exception as e: return [],0,None,f"build error: {e}"[:400],{},None
    if not body.get("query") or not _has_real_clauses(body["query"]):
        return [],0,None,"query built with no real clauses (all tiers pruned)",{},body
    try:
        r=requests.post(path["url"], headers=path["headers"], json=body, verify=VERIFY_TLS, timeout=TIMEOUT_S)
    except Exception as e:
        return [],0,None,f"{type(e).__name__}: {e}"[:400],{},body
    if r.status_code>=400: return [],0,r.status_code,(r.text or "")[:400],{},body
    j=r.json()
    total=(j.get("hits",{}).get("total",{}) or {}).get("value") or 0
    hits=[_person_from_source(h.get("_source",{}), h) for h in j.get("hits",{}).get("hits",[])]
    meta={"clauses":leaf_clause_count(body.get("query",{})),"took":j.get("took")}
    return hits,total,r.status_code,"",meta,body

# ------------------------- SCORE ONE TOP RESULT -----------------------------
def score_top(f, results, prod_id=None, error=""):
    if not results:
        return {"returned":("(call failed)" if error else "(no result / not supported)"),
                "matched":"","dob":"n/a","good":None,"returned_count":0,"prod_rank":None,
                "top_id":"","tiers":""}
    top=results[0]
    ns,ng=name_match(f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"], top["first"],top["middle"],top["last"])
    ds=dob_match(f["DOB"], top["dob"])
    matched=[]
    rtok=[t for t in (str(top["first"]).split()+str(top["middle"]).split()+str(top["last"]).split()) if t]
    if f["FIRSTNAME"] and all(max((_ratio(t,r) for r in rtok),default=0)>=NAME_THRESHOLD for t in f["FIRSTNAME"].split()): matched.append("first")
    if f["LASTNAME"] and all(max((_ratio(t,r) for r in rtok),default=0)>=NAME_THRESHOLD for t in f["LASTNAME"].split()): matched.append("last")
    if ds in ("exact","digit-flip"): matched.append("dob")
    ids=[r["id"] for r in results]
    prod_rank=(ids.index(prod_id)+1) if (prod_id and prod_id in ids) else None
    label=" ".join(x for x in [top["first"],top["middle"],top["last"]] if x) or "(id only, no name in response)"
    return {"returned":label,"matched":", ".join(matched),"dob":ds,"good":is_good(ng,ds),
            "returned_count":len(results),"prod_rank":prod_rank,"top_id":top["id"],
            "tiers":top.get("tiers","")}

# ------------------------- RUN ----------------------------------------------
config_recs={}; templates={}; scalars_by={}
for label,spec in CONFIGS.items():
    rec,tpl,sc = load_config(label,spec)
    config_recs[label]=rec; templates[label]=tpl; scalars_by[label]=sc
config_check=pd.DataFrame(list(config_recs.values()))

print("CONFIG CHECK")
print(config_check[["config","loaded","brace_balance","repaired_braces","repaired_self_nested",
                    "tiers","leaf_clauses","max_expansions","client_id"]].to_string(index=False))
for _,r in config_check.iterrows():
    if r["problems"]: print(f"\n  [{r['config']}] {r['problems']}")
print()

paths, skipped = build_paths(config_recs)
if not paths:
    print("No path is runnable. Set at least one auth token and endpoint pair, then re-run.")
    for k,why in skipped: print(f"  skipped {k}: {why}")
else:
    print(f"Paths: {[p['key'] for p in paths]}")
    if skipped: print(f"Skipped: {skipped}")
    print()

    # preflight: one call per path, so a bad host, wrong alias, expired token or
    # misplaced URL surfaces in seconds rather than after every row is a miss.
    probe={"FIRSTNAME":"MARIA","MIDDLENAME":"","LASTNAME":"GARCIA","ANUMBER":"","RECEIPT":"",
           "DOB":"19800101","COB":"","COC":""}
    pf=[]
    for p in paths:
        if p["layer"]=="api":
            res,_,st,err,meta,_=call_api(p,probe)
            cid=meta.get("client_id_returned")
            note=("" if not cid or not p["client_id"] or cid==p["client_id"] else
                  f"clientId '{p['client_id']}' not recognized; PSS used '{cid}'")
        else:
            res,_,st,err,meta,_=call_raw(p,probe,templates[p["config"]],scalars_by[p["config"]]); note=""
        pf.append({"path":p["key"],"layer":p["layer"],"url":p["url"],"status":st,
                   "results":len(res),"error":err[:140],"note":note})
    preflight_df=pd.DataFrame(pf)
    print("PREFLIGHT"); print(preflight_df.to_string(index=False)); print()

    cases=load_cases(PROD_LOGS_DIR)
    seen={}; deduped=[]
    for c in cases:
        f=c["fields"]
        k=(c["consumer"],f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"],f["ANUMBER"],f["RECEIPT"],f["DOB"],f["COB"],f["COC"])
        if k in seen: seen[k]["dup_count"]+=1; continue
        c2=dict(c); c2["dup_count"]=1; seen[k]=c2; deduped.append(c2)
    if MAX_SEARCHES: deduped=deduped[:MAX_SEARCHES]
    print(f"Loaded {len(cases)} log rows -> {len(deduped)} distinct searches "
          f"(collapsed {len(cases)-len(deduped)} duplicates).")
    print(f"Running {len(paths)} paths x {len(deduped)} searches = {len(paths)*len(deduped)} calls...\n")

    rows=[]; queries=[]
    for c in deduped:
        f=c["fields"]; prod_id=c["prod"]["id"]
        row={"consumer":c["consumer"],"dup_count":c.get("dup_count",1),
             "input_name":" ".join(x for x in [f["FIRSTNAME"],f["MIDDLENAME"],f["LASTNAME"]] if x),
             "input_dob":f["DOB"],"input_anumber":f["ANUMBER"],"input_receipt":f["RECEIPT"]}
        sc=score_top(f,[c["prod"]] if prod_id else [])
        row.update({"log_returned":sc["returned"],"log_matched":sc["matched"],
                    "log_dob":sc["dob"],"log_good":sc["good"],"log_identity_id":prod_id or ""})

        for p in paths:
            k=p["key"]
            if p["layer"]=="api":
                res,total,st,err,meta,sent=call_api(p,f)
            else:
                res,total,st,err,meta,sent=call_raw(p,f,templates[p["config"]],scalars_by[p["config"]])
            sc=score_top(f,res,prod_id,err)
            row[f"{k}_status"]=st; row[f"{k}_returned"]=sc["returned"]
            row[f"{k}_matched"]=sc["matched"]; row[f"{k}_dob"]=sc["dob"]
            row[f"{k}_good"]=sc["good"]; row[f"{k}_returned_count"]=sc["returned_count"]
            row[f"{k}_total_hits"]=total; row[f"{k}_log_rank"]=sc["prod_rank"]
            row[f"{k}_top_id"]=sc["top_id"]; row[f"{k}_error"]=err
            if p["layer"]=="api":
                row[f"{k}_exact_count"]=meta.get("exactMatches")
                row[f"{k}_similar_count"]=meta.get("similarMatches")
                row[f"{k}_client_id_used"]=meta.get("client_id_returned")
                row[f"{k}_method"]=meta.get("method")
            else:
                row[f"{k}_tier"]=sc["tiers"]          # which named tier produced the top hit
                row[f"{k}_clauses"]=meta.get("clauses")
                row[f"{k}_took_ms"]=meta.get("took")
                row[f"{k}_buildable"]=bool(sent) and _has_real_clauses((sent or {}).get("query",{}))
            queries.append({"path":k,"consumer":c["consumer"],"input_name":row["input_name"],
                            "input_anumber":f["ANUMBER"],"input_receipt":f["RECEIPT"],
                            "status":st,"error":err,"body_sent":json.dumps(sent) if sent else ""})
        rows.append(row)
    detail=pd.DataFrame(rows)

    # ---------------------- CROSS-PATH AGREEMENT ----------------------------
    keys=[p["key"] for p in paths]; meta_by={p["key"]:p for p in paths}
    pairs=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            a,b=keys[i],keys[j]; pa,pb=meta_by[a],meta_by[b]
            eq=lambda fld: pa[fld]==pb[fld]
            if eq("layer") and eq("config") and not eq("env"): kind="same config and layer, different environment"
            elif eq("env") and eq("config") and not eq("layer"): kind="same environment and config, different layer"
            elif eq("env") and eq("layer") and not eq("config"): kind="same environment and layer, different config"
            else: kind="multiple differences"
            ok=(detail[f"{a}_error"].fillna("")=="") & (detail[f"{b}_error"].fillna("")=="")
            sub=detail[ok]
            st=(sub[f"{a}_top_id"]==sub[f"{b}_top_id"]) if len(sub) else pd.Series(dtype=bool)
            row={"path_a":a,"path_b":b,"comparison":kind,"comparable_searches":int(len(sub)),
                 "same_top_result":int(st.sum()) if len(sub) else 0,
                 "different_top_result":int((~st).sum()) if len(sub) else 0,
                 "a_found_log_identity":int(sub[f"{a}_log_rank"].notna().sum()) if len(sub) else 0,
                 "b_found_log_identity":int(sub[f"{b}_log_rank"].notna().sum()) if len(sub) else 0}
            row["agreement_pct"]=round(100*row["same_top_result"]/row["comparable_searches"],1) if row["comparable_searches"] else None
            pairs.append(row)
            detail[f"agree_{a}_vs_{b}"]=(detail[f"{a}_top_id"]==detail[f"{b}_top_id"])
    agreement=pd.DataFrame(pairs)

    # ---------------------- TIER USAGE --------------------------------------
    tier_rows=[]
    for p in paths:
        col=f"{p['key']}_tier"
        if p["layer"]!="raw" or col not in detail: continue
        for val,cnt in detail[col].fillna("").value_counts().items():
            tier_rows.append({"path":p["key"],"tier_that_produced_top_result":val or "(none / no result)",
                              "searches":int(cnt)})
    tiers_df=pd.DataFrame(tier_rows) if tier_rows else pd.DataFrame(
        columns=["path","tier_that_produced_top_result","searches"])

    # ---------------------- SEARCH METHOD MIX -------------------------------
    method_rows=[]
    for p in paths:
        col=f"{p['key']}_method"
        if p["layer"]!="api" or col not in detail: continue
        for val,cnt in detail[col].fillna("").value_counts().items():
            sub=detail[detail[col]==val]
            method_rows.append({"path":p["key"],"searchMethodType":val or "(unknown)","searches":int(cnt),
                                "found_log_identity":int(sub[f"{p['key']}_log_rank"].notna().sum()),
                                "empty_results":int((sub[f"{p['key']}_returned_count"].fillna(0)==0).sum())})
    methods_df=pd.DataFrame(method_rows) if method_rows else pd.DataFrame(
        columns=["path","searchMethodType","searches","found_log_identity","empty_results"])

    # ---------------------- SUMMARY -----------------------------------------
    summ=[]
    for grp,sub in list(detail.groupby("consumer"))+[("OVERALL",detail)]:
        r={"consumer":grp,"distinct_searches":len(sub),"log_rows":int(sub["dup_count"].sum()),
           "log_good_pct":round(100*sub["log_good"].dropna().mean(),1) if sub["log_good"].notna().any() else None}
        for p in paths:
            k=p["key"]
            r[f"{k}_good_pct"]=round(100*sub[f"{k}_good"].dropna().mean(),1) if sub[f"{k}_good"].notna().any() else None
            r[f"{k}_found_log_id"]=int(sub[f"{k}_log_rank"].notna().sum())
            r[f"{k}_errors"]=int((sub[f"{k}_error"].fillna("")!="").sum())
        summ.append(r)
    summary=pd.DataFrame(summ)

    # ---------------------- ERRORS ------------------------------------------
    err_rows=[]
    for _,d in detail.iterrows():
        for p in paths:
            e=d.get(f"{p['key']}_error") or ""
            if e: err_rows.append({"path":p["key"],"consumer":d["consumer"],"input_name":d["input_name"],
                                   "input_receipt":d["input_receipt"],"input_anumber":d["input_anumber"],
                                   "status":d.get(f"{p['key']}_status"),"error":e})
    errors=pd.DataFrame(err_rows) if err_rows else pd.DataFrame(
        columns=["path","consumer","input_name","input_receipt","input_anumber","status","error"])

    print("GOOD-MATCH rate by consumer, log baseline vs each path. A comparison, not accuracy.")
    display(summary)
    print("\nCROSS-PATH AGREEMENT (top result identity, failed calls excluded)")
    display(agreement)
    if len(tiers_df):
        print("\nWHICH TIER PRODUCED THE TOP RESULT")
        display(tiers_df)
    if len(methods_df):
        print("\nSEARCH METHOD USED (identifier-only searches use identifierSearch)")
        display(methods_df)
    if len(errors):
        print(f"\n{len(errors)} calls FAILED. Reported as failures, not as searches that found nothing.")
        display(errors.head(25))
    for p in paths:
        k=p["key"]
        if p["layer"]=="api" and f"{k}_client_id_used" in detail and detail[f"{k}_client_id_used"].notna().any():
            bad=detail[detail[f"{k}_client_id_used"].notna() & (detail[f"{k}_client_id_used"]!=p["client_id"])]
            if len(bad): print(f"\nWARNING [{k}]: PSS reported a different clientId than '{p['client_id']}' on "
                               f"{len(bad)} searches. The config was not recognized and the default was used.")
    print("\nPER-SEARCH DETAIL")
    display(detail)

    # ---------------------- WRITE -------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out=os.path.join(RESULTS_DIR, f"Search_env_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    runinfo=[{"field":"run_local_time","value":datetime.now().isoformat()},
             {"field":"paths_run","value":", ".join(keys)},
             {"field":"paths_skipped","value":"; ".join(f"{k} ({w})" for k,w in skipped) or "none"},
             {"field":"prod_logs_dir","value":PROD_LOGS_DIR},
             {"field":"distinct_searches","value":len(detail)},
             {"field":"result_size","value":RESULT_SIZE},
             {"field":"name_threshold","value":NAME_THRESHOLD},
             {"field":"failed_calls","value":len(errors)},
             {"field":"templates_repaired","value":int(config_check["repaired_braces"].fillna(0).gt(0).sum())}]
    for p in paths: runinfo.append({"field":f"url_{p['key']}","value":p["url"]})
    runinfo=pd.DataFrame(runinfo)

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        agreement.to_excel(xl, sheet_name="Path agreement", index=False)
        tiers_df.to_excel(xl, sheet_name="Tier usage", index=False)
        methods_df.to_excel(xl, sheet_name="Search method", index=False)
        detail.to_excel(xl, sheet_name="Per-search detail", index=False)
        pd.DataFrame(queries).to_excel(xl, sheet_name="Queries sent", index=False)
        errors.to_excel(xl, sheet_name="Errors", index=False)
        config_check.to_excel(xl, sheet_name="Config check", index=False)
        preflight_df.to_excel(xl, sheet_name="Preflight", index=False)
        runinfo.to_excel(xl, sheet_name="Run info", index=False)
    print(f"\nExcel written: {out}")
    print("\nREADING THIS: '<env>_<config>_api' is PSS itself - PSS builds the query in Java from the config "
          "named by clientId. '<env>_<config>_raw' is the query built here from the same yaml and sent to "
          "OpenSearch, a close mirror of PSS but NOT PSS. Same config and layer across environments shows "
          "whether staging behaves like production. Same environment and config across layers shows whether "
          "the mirrored query matches what PSS builds. Index size and content differ between environments, "
          "so a difference between staging and production is not by itself a defect in the query.")
