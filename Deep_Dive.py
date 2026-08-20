# Databricks notebook source
import requests, json, re, os, time, unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher
from requests.adapters import HTTPAdapter
import pandas as pd

ENVS = {
    "staging": {
        "service": "https://pcis-search-service-staging.apps.k8s.uscis.dhs.gov/search",
        "service_auth": "oauth",
        "oauth_url": "https://oauth-preprod.uscis.dhs.gov/uaa/oauth/token",
        "oauth_id": "svc-oauth-pcisui-preprod",
        "oauth_secret": "PASTE_CLIENT_SECRET",
        "opensearch": "https://opensearch-identity-staging.pcis.uscis.dhs.gov/iis-identity-api-alias/_search",
        "opensearch_token": "",
    },
    "prod": {
        "service": "",
        "service_auth": "basic",
        "service_token": "",
        "opensearch": "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search",
        "opensearch_token": "PASTE_PROD_BASIC",
    },
}

TEMPLATE_FILE = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-default.yaml"
CLIENT_ID = "default"

ELIS_FILES = [
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/ELIS.csv",
]
COMPARE_FILES = [
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/BHUB.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/FIRST.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/CRIS.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/GLOBAL.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/UIPATH.csv",
]
RESULTS = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"

SIZE, TIMEOUT, CONCURRENCY = 100, 120, 8
NAME_THRESHOLD = 0.85
RETRY_STATUSES = (429, 502, 503, 504)
MAX_RETRIES, RETRY_BACKOFF_S = 3, 1

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_connections=CONCURRENCY, pool_maxsize=CONCURRENCY, max_retries=0))
PH = re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")
NESTED = re.compile(r'("([^"\n]+)"\s*:\s*\{)(\s*)"\2"\s*:\s*\{')
START = re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')

def grab(pat, s, d=""):
    m = re.search(pat, s); return m.group(1) if m else d
def ratio(a, b): return SequenceMatcher(None, (a or "").upper(), (b or "").upper()).ratio()

def norm_dob(v):
    if v is None: return ""
    if isinstance(v, dict):
        y = str(v.get("year") or v.get("yyyy") or "").strip()
        m = str(v.get("month") or v.get("mm") or "").strip()
        d = str(v.get("day") or v.get("dd") or "").strip()
        if y and m and d: return f"{y.zfill(4)}{m.zfill(2)}{d.zfill(2)}"
        return ""
    s = str(v).strip()
    if not s: return ""
    digits = re.sub(r"\D", "", s)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return digits
    if re.fullmatch(r"\d{8}", s): return s
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m: return f"{m.group(3)}{m.group(1).zfill(2)}{m.group(2).zfill(2)}"
    m = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m: return f"{m.group(1)}{m.group(2).zfill(2)}{m.group(3).zfill(2)}"
    return digits if len(digits) == 8 else ""

def dob_compare(a, b):
    a, b = norm_dob(a), norm_dob(b)
    if not a or not b: return "n/a"
    if a == b: return "exact"
    if len(a) == len(b) and sum(1 for x, y in zip(a, b) if x != y) == 1: return "digit-flip"
    return "no"

def _toks(*vals):
    return [t for t in " ".join(str(v or "") for v in vals).split() if t]

def name_matches(f, p, detail=False):
    fi, mi, la = _toks(f.get("FIRSTNAME")), _toks(f.get("MIDDLENAME")), _toks(f.get("LASTNAME"))
    pf, pm, pl = _toks(p.get("first")), _toks(p.get("middle")), _toks(p.get("last"))
    if not (fi + mi + la) or not (pf + pm + pl):
        return (None, "one side has no name") if detail else None
    def ok(tokens, pool):
        pool = pool or []
        return all(max((ratio(t, r) for r in pool), default=0) >= NAME_THRESHOLD for t in tokens)
    first_ok = ok(fi, pf + pm) if fi else True
    last_ok = ok(la, pl + pm) if la else True
    middle_ok = True if not mi or not pm else ok(mi, pm + pf)
    why = ""
    if not first_ok: why = "first name differs"
    elif not last_ok: why = "last name differs"
    elif not middle_ok: why = "middle name differs"
    elif mi and not pm: why = "matched, record has no middle name to compare"
    result = first_ok and last_ok and middle_ok
    return (result, why) if detail else result

def has_nonascii(s):
    return any(ord(ch) > 127 for ch in s or "")
def strip_accents(s):
    return "".join(ch for ch in unicodedata.normalize("NFD", s or "") if unicodedata.category(ch) != "Mn")

def name_profile(f):
    first = (f["FIRSTNAME"] or "").split()
    mid = (f["MIDDLENAME"] or "").split()
    last = (f["LASTNAME"] or "").split()
    full = " ".join(first + mid + last)
    return {
        "first_token_count": len(first),
        "middle_token_count": len(mid),
        "last_token_count": len(last),
        "total_name_tokens": len(first) + len(mid) + len(last),
        "two_or_more_first_names": len(first) > 1,
        "two_or_more_last_names": len(last) > 1,
        "compound_name": len(first) > 1 or len(last) > 1,
        "has_hyphen": "-" in full,
        "has_apostrophe": "'" in full,
        "has_accent_or_nonascii": has_nonascii(full),
        "shortest_token_len": min([len(t) for t in first + mid + last], default=0),
        "longest_token_len": max([len(t) for t in first + mid + last], default=0),
    }

def load_logs(files):
    cases = []
    for path in files:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"MISSING {path}"); continue
        txt = open(path, encoding="utf-8", errors="replace").read()
        st = [m.start() for m in START.finditer(txt)]
        recs = [txt[s:(st[i+1] if i+1 < len(st) else len(txt))] for i, s in enumerate(st)]
        n = 0
        for rec in recs:
            consumer = grab(r'^([A-Z0-9_]+),', rec)
            rec = rec.replace('""', '"')
            i = rec.find('"result":')
            terms, result = (rec[:i], rec[i:]) if i >= 0 else (rec, "")
            mid = grab(r'"personMiddleName":(null|"[^"]*")', terms, "null")
            f = {"FIRSTNAME": grab(r'"personGivenName":"([^"]*)"', terms),
                 "MIDDLENAME": "" if mid in ("null", "") else mid.strip('"'),
                 "LASTNAME": grab(r'"personSurName":"([^"]*)"', terms),
                 "ANUMBER": grab(r'"type":"ALIEN_NBR","value":"([^"]*)"', terms),
                 "RECEIPT": grab(r'"type":"RECEIPT_NBR","value":"([^"]*)"', terms),
                 "DOB": grab(r'"dob":"(\d{4}-\d{2}-\d{2})"', terms).replace("-", ""),
                 "COB": grab(r'"cobs":\["([^"]*)"\]', terms),
                 "COC": grab(r'"cocs":\["([^"]*)"\]', terms)}
            pid = grab(r'"identityId":"([0-9a-fA-F]{16,})"', result)
            nm = re.search(r'"name":\{[^}]*"first":"([^"]*)"[^}]*"last":"([^"]*)"', result)
            total = grab(r'"totalIdentities"\s*:\s*(\d+)', result)
            scores = [float(x) for x in re.findall(r'"score"\s*:\s*([0-9.]+)', result)]
            all_ids = re.findall(r'"identityId":"([0-9a-fA-F]{16,})"', result)
            if not pid and not any(f[k] for k in ("FIRSTNAME", "LASTNAME", "ANUMBER", "RECEIPT")):
                continue
            cases.append({"source_file": name, "consumer": consumer, "f": f, "pid": pid,
                          "pname": f"{nm.group(1)} {nm.group(2)}" if nm else "",
                          "prod_total_identities": int(total) if total else None,
                          "prod_top_score": scores[0] if scores else None,
                          "prod_second_score": scores[1] if len(scores) > 1 else None,
                          "prod_returned_ids": len(all_ids)})
            n += 1
        print(f"{name}: {len(recs)} records, {n} usable")
    seen, out = {}, []
    for c in cases:
        k = (c["source_file"], c["consumer"]) + tuple(c["f"].values())
        if k in seen: seen[k]["dup"] += 1
        else:
            c["dup"] = 1; seen[k] = c; out.append(c)
    print(f"{len(cases)} rows -> {len(out)} distinct searches")
    return out

def quote_bare(s):
    out, i, instr, esc = [], 0, False, False
    while i < len(s):
        c = s[i]
        if instr:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': instr = False
            out.append(c); i += 1; continue
        if c == '"': instr = True; out.append(c); i += 1; continue
        m = PH.match(s, i)
        if m: out.append('"' + m.group(0) + '"'); i = m.end(); continue
        out.append(c); i += 1
    return "".join(out)
def depth(s):
    d, instr, esc = 0, False, False
    for c in s:
        if instr:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': instr = False
            continue
        if c == '"': instr = True
        elif c == "{": d += 1
        elif c == "}": d -= 1
    return d

def load_template(path):
    txt = open(path).read()
    try:
        import yaml
        tpl = yaml.safe_load(txt)["search-config"]["similar-query-template"]
    except Exception:
        k = re.search(r'similar-query-template\s*:\s*[|>]?', txt)
        tpl = txt[txt.index("{", k.end()):]
    scal = {}
    try:
        import yaml
        for k2, v in (yaml.safe_load(txt) or {}).get("search-config", {}).items():
            if k2 != "similar-query-template" and isinstance(v, (str, int, float, bool)):
                scal["{{" + k2.upper().replace("-", "_") + "}}"] = str(v)
    except Exception:
        head = txt.split("similar-query-template")[0]
        for m in re.finditer(r"^\s{2,}([A-Za-z0-9\-]+)\s*:\s*([^\n|>#]+)$", head, re.M):
            scal["{{" + m.group(1).upper().replace("-", "_") + "}}"] = m.group(2).strip()
    scal.setdefault("{{SIMILAR_SIZE}}", str(SIZE))
    for m in list(NESTED.finditer(tpl))[::-1]:
        fld, st = m.group(2), m.start()
        io_ = tpl.index("{", tpl.index(f'"{fld}"', tpl.index(f'"{fld}"', st) + 1))
        d, i, end, instr, esc = 0, io_, None, False, False
        while i < len(tpl):
            c = tpl[i]
            if instr:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': instr = False
            else:
                if c == '"': instr = True
                elif c == "{": d += 1
                elif c == "}":
                    d -= 1
                    if d == 0: end = i; break
            i += 1
        if end:
            tpl = tpl[:st] + f'"{fld}": {{' + tpl[io_+1:end] + "}" + tpl[end+1:]
            print(f"repaired '{fld}' nested inside itself in {os.path.basename(path)}")
    d = depth(PH.sub("X", quote_bare(tpl)))
    if 0 < d <= 3: tpl += "}" * d
    return tpl, scal

def has_ph(n):
    if isinstance(n, str): return bool(PH.search(n))
    if isinstance(n, list): return any(has_ph(x) for x in n)
    if isinstance(n, dict): return any(has_ph(v) for v in n.values())
    return False
def prune(n):
    if isinstance(n, dict) and isinstance(n.get("bool"), dict):
        b = n["bool"]
        for k in ("must", "should"):
            if isinstance(b.get(k), list):
                for c in b[k]: prune(c)
                b[k] = [c for c in b[k] if not has_ph(c)]
                if not b[k]: del b[k]
    for v in (n.values() if isinstance(n, dict) else n if isinstance(n, list) else []):
        if isinstance(v, (dict, list)): prune(v)
def strip_ph(n):
    if isinstance(n, dict):
        for k, v in list(n.items()):
            if isinstance(v, str): n[k] = PH.sub("", v)
            elif isinstance(v, (dict, list)): strip_ph(v)
    elif isinstance(n, list):
        for i, v in enumerate(n):
            if isinstance(v, str): n[i] = PH.sub("", v)
            elif isinstance(v, (dict, list)): strip_ph(v)

def build_dsl(tpl, f, scal):
    s = tpl
    for p, v in scal.items(): s = s.replace(p, v)
    p = dict(f)
    ids = [("ALIEN_NBR", f["ANUMBER"]), ("RECEIPT_NBR", f["RECEIPT"])]
    for i, (nm, val) in enumerate([x for x in ids if x[1]][:2], 1):
        p[f"IDENTIFIER_NAME_{i}"], p[f"IDENTIFIER_VALUE_{i}"] = nm, val
    s = PH.sub(lambda m: p.get(m.group(1)) or m.group(0), s)
    s = re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', quote_bare(s))
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    q = json.loads(s); prune(q); strip_ph(q); q["size"] = SIZE; q["track_total_hits"] = True
    return q

def build_service(f):
    has_name = any(f[k] for k in ("FIRSTNAME", "MIDDLENAME", "LASTNAME"))
    method = "identifierSearch" if (f["ANUMBER"] or f["RECEIPT"]) and not has_name and not f["DOB"] else "advancedSearch"
    b = {"page": 0, "size": SIZE, "clientId": CLIENT_ID, "searchMethodType": method}
    nm = {k: f[v] for k, v in (("first", "FIRSTNAME"), ("middle", "MIDDLENAME"), ("last", "LASTNAME")) if f[v]}
    if nm: b["names"] = [nm]
    if len(f["DOB"]) == 8: b["dobs"] = [{"dob": f"{f['DOB'][:4]}-{f['DOB'][4:6]}-{f['DOB'][6:]}"}]
    if f["COB"]: b["cobs"] = [f["COB"]]
    if f["COC"]: b["cocs"] = [f["COC"]]
    ids = [{"type": t, "value": f[k]} for t, k in (("ALIEN_NBR", "ANUMBER"), ("RECEIPT_NBR", "RECEIPT")) if f[k]]
    if ids: b["identifiers"] = ids
    return b, method

def person(src, hit=None):
    nm = (src.get("biographicInfo", {}) or {}).get("name", {}) or {}
    sd = src.get("_search", {})
    p = {"id": str(src.get("identityId") or ""), "first": nm.get("first") or "",
         "middle": nm.get("middle") or "", "last": nm.get("last") or "",
         "dob": norm_dob(sd.get("dateOfBirth") if isinstance(sd, dict) else sd)}
    if hit:
        p["score"] = hit.get("_score")
        mq = hit.get("matched_queries")
        p["tiers"] = ", ".join(str(x) for x in mq) if isinstance(mq, list) else ""
    return p

def post_with_retry(url, headers, body):
    last = None
    for attempt in range(MAX_RETRIES + 1):
        r = SESSION.post(url, headers=headers, json=body, timeout=TIMEOUT)
        if r.status_code not in RETRY_STATUSES: return r
        last = r
        if attempt < MAX_RETRIES: time.sleep(RETRY_BACKOFF_S * (2 ** attempt))
    return last

_tok = {}
def oauth_token(env, c):
    if env in _tok and datetime.now().timestamp() < _tok[env][1]: return _tok[env][0]
    r = requests.post(c["oauth_url"], data={"grant_type": "client_credentials"},
                      auth=(c["oauth_id"], c["oauth_secret"]),
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=TIMEOUT)
    if r.status_code >= 400: raise RuntimeError(f"{env} token failed {r.status_code}: {r.text[:200]}")
    j = r.json()
    _tok[env] = (j["access_token"], datetime.now().timestamp() + int(j.get("expires_in", 3600)) - 60)
    return _tok[env][0]
def basic(t): return t if t.startswith("Basic ") else "Basic " + t

def search_direct(url, headers, tpl, f, scal):
    body = build_dsl(tpl, f, scal)
    r = post_with_retry(url, headers, body)
    if r.status_code >= 400: return [], 0, r.status_code, r.text[:300]
    j = r.json()
    hits = [person(h.get("_source", {}), h) for h in j["hits"]["hits"]]
    return hits, (j["hits"]["total"] or {}).get("value", 0), r.status_code, ""

def search_service(url, headers, f):
    body, method = build_service(f)
    r = post_with_retry(url, headers, body)
    if r.status_code >= 400: return [], 0, r.status_code, r.text[:300], method
    j = r.json()
    ex = (j.get("exactMatches") or {}).get("content") or []
    sim = (j.get("similarMatches") or {}).get("content") or []
    def api_person(x):
        if "biographicInfo" in x or "_search" in x: return person(x)
        nm = x.get("name") if isinstance(x.get("name"), dict) else {}
        return {"id": str(x.get("identityId") or x.get("id") or ""), "first": nm.get("first") or "",
                "middle": nm.get("middle") or "", "last": nm.get("last") or "",
                "dob": norm_dob(x.get("dateOfBirth")),
                "score": x.get("score"), "tiers": ""}
    tot = sum((j.get(k) or {}).get("totalElements", 0) or 0 for k in ("exactMatches", "similarMatches"))
    return [api_person(x) for x in list(ex) + list(sim)], tot, r.status_code, "", method

def lookup_identity(url, headers, pid):
    body = {"size": 1, "query": {"term": {"identityId": {"value": pid}}}}
    try:
        r = post_with_retry(url, headers, body)
        hits = r.json().get("hits", {}).get("hits", [])
        return person(hits[0]["_source"]) if hits else None
    except Exception:
        return None

tpl, scal = load_template(TEMPLATE_FILE)

paths = []
for env, c in ENVS.items():
    if c.get("service"):
        try:
            h = {"Content-Type": "application/json"}
            h["Authorization"] = ("Bearer " + oauth_token(env, c)) if c.get("service_auth") == "oauth" \
                else basic(c.get("service_token", ""))
            if not str(c.get("oauth_secret", c.get("service_token", ""))).startswith("PASTE"):
                paths.append({"key": f"{env}_service", "env": env, "kind": "service",
                              "url": c["service"], "h": h})
        except Exception as e:
            print(f"{env} service auth failed: {e}")
    if c.get("opensearch") and not str(c.get("opensearch_token", "")).startswith("PASTE") \
       and c.get("opensearch_token"):
        paths.append({"key": f"{env}_direct", "env": env, "kind": "direct",
                      "url": c["opensearch"], "h": {"Content-Type": "application/json",
                                                    "Authorization": basic(c["opensearch_token"])}})
print(f"paths: {[p['key'] for p in paths]}")
if not paths: raise SystemExit("No endpoint is configured. Set a token and re-run.")

elis = load_logs(ELIS_FILES)
others = load_logs(COMPARE_FILES) if COMPARE_FILES else []
if not elis: raise SystemExit("No ELIS searches loaded. Check ELIS_FILES.")

lookup_path = next((p for p in paths if p["kind"] == "direct"), None)

def analyse(case, path):
    f, pid = case["f"], case["pid"]
    if path["kind"] == "service":
        res, tot, st, err, method = search_service(path["url"], path["h"], f)
    else:
        res, tot, st, err = search_direct(path["url"], path["h"], tpl, f, scal)
        method = "direct query"
    ids = [r["id"] for r in res]
    rank = ids.index(pid) + 1 if pid and pid in ids else None
    top = res[0] if res else None
    target = next((r for r in res if r["id"] == pid), None)

    def best_name_ratio(r):
        it = [t for t in " ".join(str(f.get(k) or "") for k in ("FIRSTNAME", "MIDDLENAME", "LASTNAME")).split() if t]
        rt = [t for t in " ".join(str(r.get(k) or "") for k in ("first", "middle", "last")).split() if t]
        if not it or not rt: return None
        return round(min(max(ratio(t, x) for x in rt) for t in it), 2)

    same_person = [(i + 1, r) for i, r in enumerate(res)
                   if name_matches(f, r) and dob_compare(f["DOB"], r["dob"]) in ("exact", "digit-flip", "n/a")]
    same_person_rank = same_person[0][0] if same_person else None
    same_person_id = same_person[0][1]["id"] if same_person else ""
    same_person_name = (" ".join(x for x in [same_person[0][1]["first"], same_person[0][1]["middle"],
                                             same_person[0][1]["last"]] if x) if same_person else "")

    indexed = None
    if lookup_path and pid and rank is None:
        indexed = lookup_identity(lookup_path["url"], lookup_path["h"], pid)

    if rank == 1:
        reason = "returned first"
    elif rank:
        reason = f"returned at position {rank}"
    elif not pid:
        reason = "no identity recorded in the log"
    elif same_person_rank == 1:
        reason = "same person returned first under a different record id"
    elif same_person_rank:
        reason = f"same person returned at position {same_person_rank} under a different record id"
    elif indexed is None:
        reason = "identity not present in this index"
    else:
        nm_ok = name_matches(f, indexed)
        dc = dob_compare(f["DOB"], indexed["dob"])
        if nm_ok is False and dc == "no": reason = "indexed name and date of birth both differ"
        elif nm_ok is False: reason = "indexed name differs from the search terms"
        elif dc == "no": reason = "indexed date of birth differs from the search terms"
        else: reason = "record present and consistent, query did not surface it"

    row = {"source_file": case["source_file"], "consumer": case["consumer"],
           "input_first": f["FIRSTNAME"], "input_middle": f["MIDDLENAME"], "input_last": f["LASTNAME"],
           "input_dob": f["DOB"], "input_anumber": f["ANUMBER"], "input_receipt": f["RECEIPT"],
           "input_cob": f["COB"], "input_coc": f["COC"],
           "search_criteria": ", ".join(k for k, v in
                                        (("first", f["FIRSTNAME"]), ("middle", f["MIDDLENAME"]),
                                         ("last", f["LASTNAME"]), ("dob", f["DOB"]),
                                         ("alien_nbr", f["ANUMBER"]), ("receipt_nbr", f["RECEIPT"]),
                                         ("cob", f["COB"]), ("coc", f["COC"])) if v),
           "identifier_count": sum(1 for k in ("ANUMBER", "RECEIPT") if f[k]),
           "search_method": method,
           "path": path["key"], "environment": path["env"],
           "log_returned": case["pname"], "log_identity_id": pid,
           "prod_total_identities": case["prod_total_identities"],
           "prod_top_score": case["prod_top_score"],
           "id_matched": rank is not None, "rank": rank,
           "same_person_found": same_person_rank is not None,
           "same_person_rank": same_person_rank,
           "same_person_id": same_person_id,
           "same_person_name": same_person_name,
           "found_either_way": rank is not None or same_person_rank is not None,
           "outcome": reason,
           "top_dob": top["dob"] if top else "",
           "top_name_match": name_matches(f, top) if top else None,
           "top_dob_compare": dob_compare(f["DOB"], top["dob"]) if top else "n/a",
           "top_weakest_name_token_score": best_name_ratio(top) if top else None,
           "name_match_detail": (name_matches(f, top, detail=True)[1] if top else "no results"),
           "why_not_same_person": ("" if (top and name_matches(f, top) and
                                          dob_compare(f["DOB"], top["dob"]) in ("exact", "digit-flip", "n/a"))
                                   else "no results" if not top
                                   else "top result has no name in the response" if best_name_ratio(top) is None
                                   else "name differs" if not name_matches(f, top)
                                   else f"date of birth differs, searched {f['DOB']} and record has {top['dob'] or 'none'}"),
           "top_returned": " ".join(x for x in [top["first"], top["middle"], top["last"]] if x) if top else
                           ("(call failed)" if err else "(no result)"),
           "top_id": top["id"] if top else "",
           "top_score": top.get("score") if top else None,
           "target_score": target.get("score") if target else None,
           "score_gap_to_top": (round(top["score"] - target["score"], 2)
                                if top and target and top.get("score") is not None
                                and target.get("score") is not None else None),
           "tiers_matched_on_top": top.get("tiers", "") if top else "",
           "tiers_matched_on_target": target.get("tiers", "") if target else "",
           "indexed_name": (" ".join(x for x in [indexed["first"], indexed["middle"], indexed["last"]] if x)
                            if indexed else ""),
           "indexed_dob": indexed["dob"] if indexed else "",
           "returned_count": len(res), "total_hits": tot, "status": st, "error": err}
    row.update(name_profile(f))
    return row

tasks = [(c, p) for c in elis for p in paths]
print(f"{len(elis)} ELIS searches x {len(paths)} paths = {len(tasks)} calls")
with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    rows = list(pool.map(lambda t: analyse(*t), tasks))
detail = pd.DataFrame(rows)

ok = detail[(detail["error"] == "") & (detail["log_identity_id"] != "")]

outcomes = (ok.groupby(["path", "outcome"]).size().reset_index(name="searches")
              .sort_values(["path", "searches"], ascending=[True, False]))
outcomes["share_pct"] = (100 * outcomes["searches"] /
                         outcomes.groupby("path")["searches"].transform("sum")).round(1)

def _pct(a, b): return (100 * pd.to_numeric(a, errors="coerce") /
                        pd.to_numeric(b, errors="coerce")).round(1)

rows = []
for path, g in ok.groupby("path"):
    for crit, sub in list(g.groupby("search_criteria")) + [("ALL CRITERIA", g)]:
        n = len(sub)
        m = sub[sub["id_matched"]]
        r = m["rank"].dropna()
        rows.append({"path": path, "search_criteria": crit, "searches": n,
                     "id_matched": len(m), "id_not_matched": n - len(m),
                     "id_match_rate_pct": round(100 * len(m) / n, 1),
                     "returned_first": int((r == 1).sum()),
                     "returned_first_pct": round(100 * (r == 1).sum() / n, 1),
                     "returned_in_top_10": int((r <= 10).sum()),
                     "returned_in_top_10_pct": round(100 * (r <= 10).sum() / n, 1),
                     "returned_position_11_plus": int((r > 10).sum()),
                     "not_returned": n - len(m),
                     "median_rank_when_found": float(r.median()) if len(r) else None,
                     "worst_rank_when_found": int(r.max()) if len(r) else None,
                     "median_results_returned": float(sub["returned_count"].median()),
                     "median_total_hits": float(sub["total_hits"].median())})
summary = pd.DataFrame(rows)
summary["is_total"] = summary["search_criteria"] == "ALL CRITERIA"
summary = summary.sort_values(["path", "is_total", "searches"],
                              ascending=[True, False, False]).drop(columns=["is_total"])

review = (ok.groupby("path")
            .agg(searches=("id_matched", "size"),
                 same_person_different_record=("same_person_found", "sum"),
                 found_either_way=("found_either_way", "sum"))
            .reset_index())
review["same_person_pct"] = _pct(review["same_person_different_record"], review["searches"])
review["found_either_way_pct"] = _pct(review["found_either_way"], review["searches"])
review["basis"] = ("name and date of birth compared in this notebook at a 0.85 similarity threshold. "
                   "This is a review aid, not the matching rule the template applies.")

profile_cols = ["two_or_more_first_names", "two_or_more_last_names", "compound_name",
                "has_hyphen", "has_apostrophe", "has_accent_or_nonascii"]
name_effect = []
for col in profile_cols:
    for val, g in ok.groupby(col):
        name_effect.append({"name_feature": col, "value": bool(val), "searches": len(g),
                            "matched": int(g["id_matched"].sum()),
                            "match_rate_pct": round(100 * g["id_matched"].sum() / len(g), 1),
                            "median_name_tokens": float(g["total_name_tokens"].median())})
name_effect = pd.DataFrame(name_effect)

by_tokens = (ok.groupby("total_name_tokens")
               .agg(searches=("id_matched", "size"), matched=("id_matched", "sum"))
               .reset_index())
by_tokens["match_rate_pct"] = (100 * by_tokens["matched"] / by_tokens["searches"]).round(1)

by_criteria = (ok.groupby(["search_criteria", "path"])
                 .agg(searches=("id_matched", "size"), matched=("id_matched", "sum"),
                      median_results=("returned_count", "median"))
                 .reset_index())
by_criteria["match_rate_pct"] = (100 * by_criteria["matched"] / by_criteria["searches"]).round(1)

compare = pd.DataFrame()
if others:
    prof = []
    for c in others + elis:
        p = name_profile(c["f"])
        p.update({"consumer": c["consumer"], "identifier_count":
                  sum(1 for k in ("ANUMBER", "RECEIPT") if c["f"][k]),
                  "has_name": bool(c["f"]["FIRSTNAME"] or c["f"]["LASTNAME"]),
                  "has_dob": bool(c["f"]["DOB"])})
        prof.append(p)
    pf = pd.DataFrame(prof)
    compare = (pf.groupby("consumer")
                 .agg(searches=("total_name_tokens", "size"),
                      median_name_tokens=("total_name_tokens", "median"),
                      pct_two_or_more_first=("two_or_more_first_names", lambda x: round(100*x.mean(), 1)),
                      pct_two_or_more_last=("two_or_more_last_names", lambda x: round(100*x.mean(), 1)),
                      pct_compound_name=("compound_name", lambda x: round(100*x.mean(), 1)),
                      pct_with_accent=("has_accent_or_nonascii", lambda x: round(100*x.mean(), 1)),
                      pct_two_identifiers=("identifier_count", lambda x: round(100*(x > 1).mean(), 1)),
                      pct_with_name=("has_name", lambda x: round(100*x.mean(), 1)),
                      pct_with_dob=("has_dob", lambda x: round(100*x.mean(), 1)))
                 .reset_index())

misses = ok[~ok["found_either_way"]][
    ["source_file", "input_first", "input_middle", "input_last", "input_dob",
     "input_anumber", "input_receipt", "search_criteria", "path", "outcome",
     "log_returned", "log_identity_id", "same_person_name", "same_person_id", "same_person_rank",
     "indexed_name", "indexed_dob",
     "top_returned", "top_id", "top_dob", "top_name_match", "top_dob_compare",
     "top_weakest_name_token_score", "name_match_detail", "why_not_same_person",
     "top_score", "returned_count", "total_hits",
     "total_name_tokens", "two_or_more_first_names", "two_or_more_last_names"]]

print("\nID MATCH RATE BY SEARCH CRITERIA")
print("Match means the identity record production returned came back. Nothing else is assumed.")
display(summary)
print("\nREVIEW AID, NOT A MATCH RATE")
print("The columns below use a name and date of birth comparison written in this notebook, not the "
      "matching rules inside the template. Use them to investigate individual searches, not to report on.")
display(review)
print("\nWHY EACH SEARCH ENDED THE WAY IT DID")
print("Outcomes that mention the same person use this notebook's name and date of birth comparison, "
      "not the template's matching rules.")
display(outcomes)
print("\nDOES NAME SHAPE AFFECT THE RESULT")
print("Scored on the identity record match, so this one does not depend on any assumed rule.")
display(name_effect)
print("\nMATCH RATE BY NUMBER OF NAME TOKENS"); display(by_tokens)
print("\nMATCH RATE BY SEARCH CRITERIA"); display(by_criteria)
if len(compare):
    print("\nHOW ELIS SEARCHES DIFFER FROM OTHER CONSUMERS"); display(compare)
if len(misses):
    print(f"\n{len(misses)} MISSES IN DETAIL"); display(misses.head(50))

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, f"ELIS_deep_dive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xl:
    summary.to_excel(xl, sheet_name="ID match by criteria", index=False)
    review.to_excel(xl, sheet_name="Review aid", index=False)
    outcomes.to_excel(xl, sheet_name="Outcomes (review aid)", index=False)
    name_effect.to_excel(xl, sheet_name="Name shape effect", index=False)
    by_tokens.to_excel(xl, sheet_name="By name tokens", index=False)
    by_criteria.to_excel(xl, sheet_name="By search criteria", index=False)
    if len(compare): compare.to_excel(xl, sheet_name="ELIS vs other consumers", index=False)
    if len(misses): misses.to_excel(xl, sheet_name="Misses (review aid)", index=False)
    detail.to_excel(xl, sheet_name="Per search detail", index=False)
print(f"\nSaved: {out}")
