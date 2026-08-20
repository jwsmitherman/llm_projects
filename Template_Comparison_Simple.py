# Databricks notebook source
import requests, json, re, os, glob, time, hashlib
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from requests.adapters import HTTPAdapter
import pandas as pd

ENVS = {
    "staging": {
        "service": "https://pcis-search-service-staging.apps.k8s.uscis.dhs.gov/search",
        "service_auth": "oauth",
        "oauth_url": "https://oauth-preprod.uscis.dhs.gov/uaa/oauth/token",
        "oauth_id": "svc-oauth-pcisui-preprod",
        "oauth_secret": "PASTE_CLIENT_SECRET",
        "oauth_scope": "",
        "opensearch": "",
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

TEMPLATE_FILES = [
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-default.yaml",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-max-clause-test.yaml",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-reduced-tiers.yaml",
]
LOG_FILES = [
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/bhub-a.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/bhub-b.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/bhub-c.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/cris.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/first-a.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/first-b.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/global.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/larger/uipath.csv",
]
RESULTS = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"

SIZE, TIMEOUT = 100, 120
NAME_THRESHOLD = 0.85
SERVICE_SIZE_FIELD = "size"
DEDUPE = True

CONCURRENCY = 16
RETRY_STATUSES = (429, 502, 503, 504)
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1
SAMPLE_PER_CONSUMER = 0
PROGRESS_EVERY = 2000
EXCEL_ROW_LIMIT = 200000

REUSE_RAW_CSV = ""

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_connections=CONCURRENCY, pool_maxsize=CONCURRENCY, max_retries=0))

PH = re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")
NESTED = re.compile(r'("([^"\n]+)"\s*:\s*\{)(\s*)"\2"\s*:\s*\{')

START = re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')
def grab(pat, s, d=""):
    m = re.search(pat, s); return m.group(1) if m else d

def load_logs():
    cases = []
    for path in LOG_FILES:
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f"MISSING {path}"); continue
        txt = open(path, encoding="utf-8", errors="replace").read()
        st = [m.start() for m in START.finditer(txt)]
        recs = [txt[s:(st[i+1] if i+1 < len(st) else len(txt))] for i, s in enumerate(st)]
        mentions = txt.count("CORE_SEARCH")
        print(f"{name}: {os.path.getsize(path):,} bytes, {txt.count(chr(10))+1:,} lines, "
              f"{len(recs):,} records matched, {os.path.getsize(path)//max(len(recs),1):,} bytes per record")
        if mentions > len(recs):
            print(f"  CHECK CORE_SEARCH appears {mentions:,} times but {len(recs):,} records were captured. "
                  f"Extra mentions may be inside a payload, or records may not start at the beginning of a line.")
        if len(recs) < (txt.count(chr(10)) + 1) / 2:
            print(f"  WARNING {name} has {txt.count(chr(10))+1:,} lines but only {len(recs):,} records "
                  f"matched the CONSUMER,APP,CORE_SEARCH pattern. Most of this file is not being read.")
        n = 0
        for rec in recs:
            consumer = grab(r'^([A-Z0-9_]+),', rec); rec = rec.replace('""', '"')
            i = rec.find('"result":'); terms, result = (rec[:i], rec[i:]) if i >= 0 else (rec, "")
            mid = grab(r'"personMiddleName":(null|"[^"]*")', terms, "null")
            f = {"FIRSTNAME": grab(r'"personGivenName":"([^"]*)"', terms),
                 "MIDDLENAME": "" if mid in ("null", "") else mid.strip('"'),
                 "LASTNAME": grab(r'"personSurName":"([^"]*)"', terms),
                 "ANUMBER": grab(r'"type":"ALIEN_NBR","value":"([^"]*)"', terms),
                 "RECEIPT": grab(r'"type":"RECEIPT_NBR","value":"([^"]*)"', terms),
                 "DOB": grab(r'"dob":"(\d{4}-\d{2}-\d{2})"', terms).replace("-", ""),
                 "COB": grab(r'"cobs":\["([^"]*)"\]', terms), "COC": grab(r'"cocs":\["([^"]*)"\]', terms)}
            pid = grab(r'"identityId":"([0-9a-fA-F]{16,})"', result)
            nm = re.search(r'"name":\{[^}]*"first":"([^"]*)"[^}]*"last":"([^"]*)"', result)
            log_ids = re.findall(r'"identityId":"([0-9a-fA-F]{16,})"', result)
            log_scores = [float(x) for x in re.findall(r'"score"\s*:\s*([0-9.]+)', result)]
            log_total = grab(r'"totalIdentities"\s*:\s*(\d+)', result)
            log_dob = grab(r'"dateOfBirth"\s*:\s*"?(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{8})', result)
            if not pid and not any(f[k] for k in ("FIRSTNAME", "LASTNAME", "ANUMBER", "RECEIPT")): continue
            cases.append({"source_file": name, "consumer": consumer, "f": f, "pid": pid,
                          "pname": f"{nm.group(1)} {nm.group(2)}" if nm else "",
                          "log_dob": log_dob,
                          "log_total_identities": int(log_total) if log_total else None,
                          "log_top_score": log_scores[0] if log_scores else None,
                          "log_results_listed": len(log_ids),
                          "log_client_id": grab(r'"clientId"\s*:\s*"?([^",}]*)', terms),
                          "log_suppress_phantom": grab(r'"suppressPhantomIdentities"\s*:\s*(true|false)', terms),
                          "log_search_method": grab(r'"searchMethodType"\s*:\s*"([^"]*)"', terms)}); n += 1
        print(f"  {n:,} usable from {name}")
        if recs and n == 0:
            print(f"  WARNING {name} parsed {len(recs)} records but none were usable. "
                  f"Check the file layout.")
        if not recs:
            print(f"  WARNING {name} produced no records. The row pattern did not match anything. "
                  f"Check that rows start with CONSUMER,APP,CORE_SEARCH,")
    if not DEDUPE:
        for c in cases: c["dup"] = 1
        print(f"{len(cases)} rows, dedupe off, all rows kept")
        return cases
    seen, out = {}, []
    for c in cases:
        k = (c["source_file"], c["consumer"]) + tuple(c["f"].values())
        if k in seen: seen[k]["dup"] += 1
        else: c["dup"] = 1; seen[k] = c; out.append(c)
    seen_consumers = sorted({c["consumer"] for c in cases})
    print(f"Consumers present: {seen_consumers}")
    missing = [x for x in ("BHUB", "CRIS", "ELIS", "FIRST", "GLOBAL", "UIPATH") if x not in seen_consumers]
    if missing:
        print(f"NOTE no searches loaded for {missing}. Those consumers cannot be assessed in this run.")
    print(f"{len(cases)} rows -> {len(out)} distinct searches "
          f"({len(cases) - len(out)} were repeats of a search already counted)")
    return out

def probe(t): return PH.sub("X", quote_bare(t))
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
    name = os.path.basename(path)
    if not os.path.exists(path): return None, {}, "file not found"
    txt = open(path).read()
    notes = []
    if name.lower().endswith((".yaml", ".yml")):
        tpl = None
        try:
            import yaml
            tpl = yaml.safe_load(txt)["search-config"]["similar-query-template"]
        except Exception:
            k = re.search(r'similar-query-template\s*:\s*[|>]?', txt)
            if not k: return None, {}, "similar-query-template not found"
            tpl = txt[txt.index("{", k.end()):]
            notes.append("yaml parser could not read this file")
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
    else:
        tpl, scal = txt[txt.index("{"):], {}
    scal.setdefault("{{SIMILAR_SIZE}}", str(SIZE))

    for m in list(NESTED.finditer(tpl))[::-1]:
        fld, s = m.group(2), m.start()
        io_ = tpl.index("{", tpl.index(f'"{fld}"', tpl.index(f'"{fld}"', s) + 1))
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
            tpl = tpl[:s] + f'"{fld}": {{' + tpl[io_+1:end] + "}" + tpl[end+1:]
            notes.append(f"repaired '{fld}' nested inside itself, fix the source file")
    d = depth(probe(tpl))
    if 0 < d <= 3: tpl += "}" * d; notes.append(f"added {d} closing brace(s), fix the source file")
    try: json.loads(probe(tpl))
    except Exception as e: return None, {}, f"not valid JSON: {e}"
    return tpl, scal, "; ".join(notes)

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

def build_service(f, client_id):
    has_name = any(f[k] for k in ("FIRSTNAME", "MIDDLENAME", "LASTNAME"))
    method = "identifierSearch" if (f["ANUMBER"] or f["RECEIPT"]) and not has_name and not f["DOB"] else "advancedSearch"
    b = {"page": 0, SERVICE_SIZE_FIELD: SIZE, "clientId": client_id, "searchMethodType": method}
    nm = {k: f[v] for k, v in (("first", "FIRSTNAME"), ("middle", "MIDDLENAME"), ("last", "LASTNAME")) if f[v]}
    if nm: b["names"] = [nm]
    if len(f["DOB"]) == 8: b["dobs"] = [{"dob": f"{f['DOB'][:4]}-{f['DOB'][4:6]}-{f['DOB'][6:]}"}]
    if f["COB"]: b["cobs"] = [f["COB"]]
    if f["COC"]: b["cocs"] = [f["COC"]]
    ids = [{"type": t, "value": f[k]} for t, k in (("ALIEN_NBR", "ANUMBER"), ("RECEIPT_NBR", "RECEIPT")) if f[k]]
    if ids: b["identifiers"] = ids
    return b, method

_tok = {}
def oauth_token(env, c):
    if env in _tok and datetime.now().timestamp() < _tok[env][1]:
        return _tok[env][0]
    data = {"grant_type": "client_credentials"}
    if c.get("oauth_scope"): data["scope"] = c["oauth_scope"]
    r = requests.post(c["oauth_url"], data=data, auth=(c["oauth_id"], c["oauth_secret"]),
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"{env} token request failed {r.status_code}: {r.text[:300]}")
    j = r.json()
    _tok[env] = (j["access_token"], datetime.now().timestamp() + int(j.get("expires_in", 3600)) - 60)
    print(f"{env}: got OAuth token, expires in {j.get('expires_in')}s")
    return _tok[env][0]

def basic(t): return t if t.startswith("Basic ") else "Basic " + t

def headers_for(env, c, path):
    h = {"Content-Type": "application/json"}
    if path == "service":
        if c.get("service_auth") == "oauth":
            h["Authorization"] = "Bearer " + oauth_token(env, c)
        else:
            h["Authorization"] = basic(c.get("service_token", ""))
    else:
        h["Authorization"] = basic(c.get("opensearch_token", ""))
    return h

CRITERIA_FIELDS = [("FIRSTNAME", "first"), ("MIDDLENAME", "middle"), ("LASTNAME", "last"),
                   ("DOB", "dob"), ("ANUMBER", "alien_nbr"), ("RECEIPT", "receipt_nbr"),
                   ("COB", "country_of_birth"), ("COC", "country_of_citizenship")]

def describe_criteria(f):
    return ", ".join(label for key, label in CRITERIA_FIELDS if (f.get(key) or "").strip())

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

def person(src):
    nm = (src.get("biographicInfo", {}) or {}).get("name", {}) or {}
    sd = src.get("_search", {})
    return {"id": str(src.get("identityId") or ""), "first": nm.get("first") or "",
            "middle": nm.get("middle") or "", "last": nm.get("last") or "",
            "dob": norm_dob(sd.get("dateOfBirth") if isinstance(sd, dict) else sd)}
def api_person(x):
    if "biographicInfo" in x or "_search" in x: return person(x)
    nm = x.get("name") if isinstance(x.get("name"), dict) else {}
    return {"id": str(x.get("identityId") or x.get("id") or ""), "first": nm.get("first") or "",
            "middle": nm.get("middle") or "", "last": nm.get("last") or "",
            "dob": norm_dob(x.get("dateOfBirth"))}

retries_used = [0]

def post_with_retry(url, headers, body):
    """OpenSearch answers 429 when it is saturated and the search service turns
       that into a 500. Retrying with a pause fulfils the request instead of
       recording a false miss, and backing off reduces the load this test adds."""
    last = None
    for attempt in range(MAX_RETRIES + 1):
        r = SESSION.post(url, headers=headers, json=body, timeout=TIMEOUT)
        if r.status_code not in RETRY_STATUSES:
            return r, attempt
        last = r
        if attempt < MAX_RETRIES:
            retries_used[0] += 1
            time.sleep(RETRY_BACKOFF_S * (2 ** attempt))
    return last, MAX_RETRIES

def call(run, f, tpl, scal):
    try:
        if run["path"] == "service":
            body, method = build_service(f, run["cid"])
            r, _ = post_with_retry(run["url"], run["h"], body)
            if r.status_code >= 400: return [], 0, r.status_code, r.text[:300], None, method, (0, 0)
            j = r.json()
            ex = (j.get("exactMatches") or {}).get("content") or []
            sim = (j.get("similarMatches") or {}).get("content") or []
            ppl = [api_person(x) for x in list(ex) + list(sim)]
            counts = (len(ex), len(sim))
            tot = sum((j.get(k) or {}).get("totalElements", 0) or 0 for k in ("exactMatches", "similarMatches"))
            return ppl, tot, r.status_code, "", j.get("clientId"), method, counts
        body = build_dsl(tpl, f, scal)
        qh = hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        r, _ = post_with_retry(run["url"], run["h"], body)
        if r.status_code >= 400: return [], 0, r.status_code, r.text[:300], None, qh, (0, 0)
        j = r.json()
        return ([person(h.get("_source", {})) for h in j["hits"]["hits"]],
                (j["hits"]["total"] or {}).get("value", 0), r.status_code, "", None, qh, (0, 0))
    except Exception as e:
        return [], 0, None, f"{type(e).__name__}: {e}"[:300], None, "", (0, 0)

if not REUSE_RAW_CSV:
    tpls = {}
    chk = []
    for path in TEMPLATE_FILES:
        name = os.path.basename(path)
        label = re.sub(r"^search-|\.(yaml|yml|txt)$", "", name)
        tpl, scal, note = load_template(path)
        tpls[label] = {"tpl": tpl, "scal": scal,
                       "cid": label if name.lower().endswith((".yaml", ".yml")) else None}
        chk.append({"template": label, "file": name, "loaded": tpl is not None,
                    "tiers": len(re.findall(r'"_name"', tpl or "")), "note": note})
    check = pd.DataFrame(chk)
    print(check.to_string(index=False))

    runs = []
    for env, c in ENVS.items():
        svc_ready = c.get("service") and (
            (c.get("service_auth") == "oauth" and not str(c.get("oauth_secret", "")).startswith("PASTE"))
            or (c.get("service_auth") != "oauth" and c.get("service_token") and not str(c["service_token"]).startswith("PASTE")))
        os_ready = c.get("opensearch") and not str(c.get("opensearch_token", "")).startswith("PASTE")
        svc_h = os_h = None
        if svc_ready:
            try:
                svc_h = headers_for(env, c, "service")
            except Exception as e:
                print(f"{env} service auth FAILED: {e}")
                svc_ready = False
        if os_ready:
            os_h = headers_for(env, c, "direct")
        print(f"{env}: service={'yes' if svc_ready else 'no'} opensearch={'yes' if os_ready else 'no'}")
        if svc_ready and not os_ready:
            print(f"  {env} is service only. A template that is not deployed there cannot be tested.")
        for label, t in tpls.items():
            if svc_ready and t["cid"]:
                runs.append({"key": f"{label}|{env}|service", "tpl": label, "env": env, "path": "service",
                             "url": c["service"], "h": svc_h, "cid": t["cid"]})
            if os_ready and t["tpl"]:
                runs.append({"key": f"{label}|{env}|direct", "tpl": label, "env": env, "path": "direct",
                             "url": c["opensearch"], "h": os_h, "cid": None})
    print(f"{len(runs)} runs")

    cases = load_logs()
    if not cases:
        raise SystemExit("No searches were loaded from the log files. Nothing to run.")

    probe = cases[0]["f"]
    print(f"\nHealth check uses the first search from {cases[0]['source_file']}. No search terms are invented.")
    pf = []
    for r in runs:
        t = tpls[r["tpl"]]
        res, tot, st, err, cid_back, _, _ = call(r, probe, t["tpl"], t["scal"])
        note = ""
        if err:
            note = "CALL FAILED"
        elif r["path"] == "service" and cid_back and cid_back != r["cid"]:
            note = f"template not deployed, service used '{cid_back}'"
        pf.append({"run": r["key"], "template": r["tpl"], "environment": r["env"], "path": r["path"],
                   "url": r["url"], "status": st, "results": len(res), "error": err[:200], "note": note})
    preflight = pd.DataFrame(pf)
    print("\nHEALTH CHECK")
    print(preflight[["run", "status", "results", "error", "note"]].to_string(index=False))

    failed = preflight[preflight["error"] != ""]
    if len(failed) == len(preflight):
        raise SystemExit("Every endpoint failed the health check. Nothing was run. Fix the errors above.")
    if len(failed):
        print(f"\n{len(failed)} of {len(preflight)} runs failed the health check and are dropped.")
        runs = [r for r in runs if r["key"] not in set(failed["run"])]
    if not runs:
        raise SystemExit("No healthy runs remain.")
    nd = preflight[preflight["note"].str.startswith("template not deployed")]
    if len(nd):
        print(f"\n{len(nd)} runs ask for a template that is not deployed in that environment. They still run, "
              f"but describe the default template, not the one named.")
    print(f"\n{len(runs)} runs passed the health check\n")
    key_files = {}
    for c in cases:
        k = (c["consumer"],) + tuple(c["f"].values())
        key_files.setdefault(k, set()).add(c["source_file"])
    shared = {k: v for k, v in key_files.items() if len(v) > 1}
    print(f"\n{len(key_files):,} distinct searches across all files. "
          f"{len(shared):,} of them appear in more than one file.")
    if shared:
        pairs = {}
        for v in shared.values():
            pairs[" + ".join(sorted(v))] = pairs.get(" + ".join(sorted(v)), 0) + 1
        for combo, cnt in sorted(pairs.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:,} searches appear in: {combo}")
        print("Searches appearing in two files are run once per file, so both sets can be compared "
              "side by side on the By file tab.")

    if SAMPLE_PER_CONSUMER:
        by_c = {}
        for c in cases: by_c.setdefault(c["consumer"], []).append(c)
        cases = [x for v in by_c.values() for x in v[:SAMPLE_PER_CONSUMER]]
        print(f"Sampling {SAMPLE_PER_CONSUMER} per consumer -> {len(cases)} searches")

    tasks = [(c, run) for c in cases for run in runs]
    total = len(tasks)
    print(f"{len(cases)} searches x {len(runs)} runs = {total} calls, {CONCURRENCY} at a time")

    done = [0]
    t0 = time.time()

    def do(task):
        c, run = task
        f, pid = c["f"], c["pid"]
        t = tpls[run["tpl"]]
        res, tot, st, err, cid_back, method, counts = call(run, f, t["tpl"], t["scal"])
        top = res[0] if res else None
        ids = [r["id"] for r in res if r["id"]]
        unique_ids = sorted(set(ids))
        done[0] += 1
        if PROGRESS_EVERY and done[0] % PROGRESS_EVERY == 0:
            el = time.time() - t0
            rate = done[0] / el if el else 0
            left = (total - done[0]) / rate if rate else 0
            print(f"  {done[0]}/{total} calls, {rate:.0f}/sec, about {left/60:.0f} min left")
        return {"source_file": c["source_file"], "consumer": c["consumer"], "dup": c["dup"],
                "input_name": f"{f['FIRSTNAME']} {f['MIDDLENAME']} {f['LASTNAME']}".strip(),
                "input_dob": f["DOB"], "input_anumber": f["ANUMBER"], "input_receipt": f["RECEIPT"],
                "input_cob": f["COB"], "input_coc": f["COC"],
                "search_criteria": describe_criteria(f),
                "criteria_count": sum(1 for k, _ in CRITERIA_FIELDS if (f.get(k) or "").strip()),
                "identifier_count": sum(1 for k in ("ANUMBER", "RECEIPT") if (f.get(k) or "").strip()),
                "search_method": method if run["path"] == "service" else "direct query",
                "search_key": "|".join([c["source_file"], c["consumer"]] + list(f.values())),
                "log_returned": c["pname"], "log_identity_id": pid,
                "log_dob": norm_dob(c.get("log_dob")),
                "log_total_identities": c.get("log_total_identities"),
                "log_results_listed": c.get("log_results_listed"),
                "log_search_method": c.get("log_search_method"),
                "template": run["tpl"], "environment": run["env"], "path": run["path"],
                "template_used": cid_back if run["path"] == "service" else run["tpl"],
                "identities_returned": len(ids),
                "unique_identities_returned": len(unique_ids),
                "duplicate_ids_in_result": len(ids) - len(unique_ids),
                "returned_one_only": len(unique_ids) == 1,
                "returned_none": len(unique_ids) == 0,
                "total_hits": tot,
                "exact_matches": counts[0], "similar_matches": counts[1],
                "top_returned": " ".join(x for x in [top["first"], top["middle"], top["last"]] if x) if top else
                                ("(call failed)" if err else "(no result)"),
                "top_id": top["id"] if top else "",
                "top_dob": top["dob"] if top else "",
                "top_score": top.get("score") if top else None,
                "top_tiers_matched": top.get("tiers", "") if top else "",
                "returned_ids": ",".join(unique_ids[:50]),
                "status": st, "error": err}

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        rows = list(pool.map(do, tasks))
    el = time.time() - t0
    print(f"{total} calls in {el/60:.1f} min ({total/el if el else 0:.0f}/sec)")
    if retries_used[0]:
        print(f"{retries_used[0]} calls were retried after an overload response ({RETRY_STATUSES}). "
              f"The endpoint was saturated at this request rate. Lower CONCURRENCY if this is large.")
    long = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    raw_path = os.path.join(RESULTS, f"raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    long.to_csv(raw_path, index=False)
    print(f"Raw results saved to {raw_path} before any analysis, so a failure below does not lose the run.")
else:
    long = pd.read_csv(REUSE_RAW_CSV)
    for c in ("returned_one_only", "returned_none"):
        if c in long.columns:
            long[c] = long[c].map({True: True, False: False, "True": True, "False": False,
                                   "TRUE": True, "FALSE": False})
    for c in ("error", "search_criteria", "template_used", "outcome", "top_returned"):
        if c in long.columns: long[c] = long[c].fillna("")
    tpls = {t: {} for t in sorted(long["template"].dropna().unique())}
    print(f"Reusing {len(long):,} rows from {REUSE_RAW_CSV}. No calls are made.")
    print("Templates", list(tpls), "environments", sorted(long["environment"].dropna().unique()))


ok_rows = long[long["error"] == ""].copy()
for c in ("unique_identities_returned", "identities_returned", "total_hits",
          "exact_matches", "similar_matches", "duplicate_ids_in_result"):
    if c in ok_rows: ok_rows[c] = pd.to_numeric(ok_rows[c], errors="coerce")
for c in ("returned_one_only", "returned_none"):
    if c in ok_rows:
        ok_rows[c] = ok_rows[c].map({True: True, False: False, "True": True, "False": False,
                                     "TRUE": True, "FALSE": False}).fillna(False)

def pct_col(num, den): return (100 * pd.to_numeric(num, errors="coerce") /
                               pd.to_numeric(den, errors="coerce")).round(1)

score = []
for (e, t, p), s in long.groupby(["environment", "template", "path"]):
    failed = int((s["error"] != "").sum())
    s = ok_rows[(ok_rows["environment"] == e) & (ok_rows["template"] == t) & (ok_rows["path"] == p)]
    if not len(s): continue
    for crit, g in list(s.groupby("search_criteria")) + [("ALL CRITERIA", s)]:
        n = len(g)
        u = g["unique_identities_returned"]
        pool = set()
        for v in g["returned_ids"].dropna():
            pool.update(x for x in str(v).split(",") if x)
        score.append({"environment": e, "template": t, "path": p,
                      "search_criteria": crit,
                      "searches": n,
                      "identities_per_search": round(u.sum() / n, 1),
                      "median_identities_per_search": float(u.median()),
                      "min_identities_returned": int(u.min()),
                      "max_identities_returned": int(u.max()),
                      "total_identities_returned": int(u.sum()),
                      "distinct_identities_across_searches": len(pool),
                      "searches_returning_none": int((u == 0).sum()),
                      "searches_returning_one_only": int((u == 1).sum()),
                      "searches_returning_one_only_pct": round(100 * (u == 1).sum() / n, 1),
                      "searches_returning_under_10": int((u < 10).sum()),
                      "searches_returning_under_10_pct": round(100 * (u < 10).sum() / n, 1),
                      "median_total_hits": float(g["total_hits"].median()),
                      "failed_calls_excluded": failed if crit == "ALL CRITERIA" else None})
scorecard = pd.DataFrame(score)
scorecard["is_total"] = scorecard["search_criteria"] == "ALL CRITERIA"
scorecard = scorecard.sort_values(["environment", "template", "is_total", "searches"],
                                  ascending=[True, True, False, False]).drop(columns=["is_total"])

pairs = []
for (env, path), g in ok_rows.groupby(["environment", "path"]):
    sets = {}
    for _, r in g.iterrows():
        sets.setdefault(r["search_key"], {})[r["template"]] = set(
            x for x in str(r["returned_ids"] or "").split(",") if x)
    tmpls = sorted(g["template"].unique())
    for i in range(len(tmpls)):
        for j in range(i + 1, len(tmpls)):
            a, b = tmpls[i], tmpls[j]
            na = nb = shared = only_a = only_b = fewer_a = fewer_b = same = n = 0
            for d in sets.values():
                if a not in d or b not in d: continue
                n += 1
                sa, sb = d[a], d[b]
                na += len(sa); nb += len(sb)
                shared += len(sa & sb); only_a += len(sa - sb); only_b += len(sb - sa)
                if len(sa) < len(sb): fewer_a += 1
                elif len(sb) < len(sa): fewer_b += 1
                else: same += 1
            if not n: continue
            pairs.append({"environment": env, "path": path, "template_a": a, "template_b": b,
                          "searches_compared": n,
                          "identities_per_search_a": round(na / n, 1),
                          "identities_per_search_b": round(nb / n, 1),
                          "difference_per_search": round((na - nb) / n, 1),
                          "shared_identities": shared, "only_in_a": only_a, "only_in_b": only_b,
                          "overlap_pct": round(100 * shared / max(shared + only_a + only_b, 1), 1),
                          "searches_where_a_returned_fewer": fewer_a,
                          "searches_where_b_returned_fewer": fewer_b,
                          "searches_returning_the_same_count": same})
template_overlap = pd.DataFrame(pairs)

by_file = (ok_rows.groupby(["source_file", "consumer", "environment", "template", "path"])
             .agg(searches=("unique_identities_returned", "size"),
                  total_identities=("unique_identities_returned", "sum"),
                  median_identities=("unique_identities_returned", "median"),
                  min_identities=("unique_identities_returned", "min"),
                  max_identities=("unique_identities_returned", "max"),
                  searches_returning_one_only=("returned_one_only", "sum"),
                  searches_returning_none=("returned_none", "sum"))
             .reset_index())
by_file["identities_per_search"] = (by_file["total_identities"] / by_file["searches"]).round(1)

file_grid = by_file.pivot_table(index=["source_file", "consumer", "environment", "searches"],
                                columns="template", values="median_identities").reset_index()

criteria_grid = (ok_rows.groupby(["search_criteria", "environment", "template"])
                   ["unique_identities_returned"].median().reset_index()
                   .pivot_table(index=["search_criteria", "environment"], columns="template",
                                values="unique_identities_returned").reset_index())

criteria_mix = (ok_rows.drop_duplicates("search_key").groupby("search_criteria")
                  .agg(searches=("search_key", "size")).reset_index()
                  .sort_values("searches", ascending=False))
criteria_mix["share_of_searches_pct"] = pct_col(criteria_mix["searches"], criteria_mix["searches"].sum())

print("\nIDENTITIES RETURNED, BY SEARCH CRITERIA")
print("Counts unique identities returned per search. No comparison is made against the identity in the log.")
display(scorecard)
print("\nSAME SEARCH, DIFFERENT TEMPLATE")
display(template_overlap)
print("\nMEDIAN IDENTITIES BY FILE AND TEMPLATE")
display(file_grid)
print("\nMEDIAN IDENTITIES BY SEARCH CRITERIA AND TEMPLATE")
display(criteria_grid)

thin = scorecard[(scorecard["search_criteria"] == "ALL CRITERIA") &
                 (scorecard["searches_returning_one_only_pct"] > 10)]
if len(thin):
    print("\nTEMPLATES RETURNING A SINGLE IDENTITY ON MORE THAN ONE SEARCH IN TEN")
    print(thin[["environment", "template", "searches", "searches_returning_one_only",
                "searches_returning_one_only_pct"]].to_string(index=False))

codes = long[long["error"] != ""]
if len(codes):
    print(f"\n{len(codes)} calls failed and are excluded from every figure above.")
    print(codes.groupby(["environment", "path", "status"]).size().reset_index(name="calls").to_string(index=False))

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, f"Template_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xl:
    scorecard.to_excel(xl, sheet_name="Identities by criteria", index=False)
    template_overlap.to_excel(xl, sheet_name="Template overlap", index=False)
    file_grid.to_excel(xl, sheet_name="File x template", index=False)
    by_file.to_excel(xl, sheet_name="By file", index=False)
    criteria_grid.to_excel(xl, sheet_name="By search criteria", index=False)
    criteria_mix.to_excel(xl, sheet_name="Search criteria mix", index=False)
    cols = ["source_file", "consumer",
            "input_name", "input_dob", "input_anumber", "input_receipt", "input_cob", "input_coc",
            "search_criteria", "criteria_count", "identifier_count", "search_method",
            "log_returned", "log_identity_id", "log_dob", "log_search_method",
            "log_total_identities", "log_results_listed",
            "environment", "path", "template_used",
            "identities_returned", "unique_identities_returned", "duplicate_ids_in_result",
            "returned_one_only", "returned_none", "total_hits", "exact_matches", "similar_matches",
            "top_returned", "top_id", "top_dob", "top_score", "top_tiers_matched",
            "returned_ids", "status", "error"]
    cols = [c for c in cols if c in long.columns]
    for label in tpls:
        d = long[long["template"] == label]
        if not len(d): continue
        d = d.sort_values(["environment", "source_file", "unique_identities_returned"])[cols]
        if len(d) > EXCEL_ROW_LIMIT:
            csv_path = os.path.join(RESULTS, f"detail_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            d.to_csv(csv_path, index=False)
            print(f"{label}: {len(d):,} rows written to {csv_path}")
            d.head(EXCEL_ROW_LIMIT).to_excel(xl, sheet_name=label[:31], index=False)
        else:
            d.to_excel(xl, sheet_name=label[:31], index=False)
    e = long[long["error"] != ""]
    if len(e): e.to_excel(xl, sheet_name="Errors", index=False)
print(f"\nSaved: {out}")
