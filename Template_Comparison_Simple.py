# Databricks notebook source
import requests, json, re, os, glob, time, hashlib
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
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-first.yaml",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-max-clause-test.yaml",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-reduced-tiers.yaml",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-ui.yaml",
]
LOG_FILES = [
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/BHUB.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/CRIS.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/ELIS.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/FIRST.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/GLOBAL.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/UIPATH.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/BHUB 1.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/CRIS 1.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/ELIS 1.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/FIRST 1.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/GLOBAL 1.csv",
    "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/UIPATH 1.csv",
]
RESULTS = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"

SIZE, TIMEOUT = 100, 120
SERVICE_SIZE_FIELD = "size"
DEDUPE = True

CONCURRENCY = 16
SAMPLE_PER_CONSUMER = 0
PROGRESS_EVERY = 2000
EXCEL_ROW_LIMIT = 200000
STABILITY_SAMPLE = 25

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
            if not pid and not any(f[k] for k in ("FIRSTNAME", "LASTNAME", "ANUMBER", "RECEIPT")): continue
            cases.append({"source_file": name, "consumer": consumer, "f": f, "pid": pid,
                          "pname": f"{nm.group(1)} {nm.group(2)}" if nm else ""}); n += 1
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

def person(src):
    nm = (src.get("biographicInfo", {}) or {}).get("name", {}) or {}
    sd = src.get("_search", {})
    return {"id": str(src.get("identityId", "")), "first": nm.get("first", ""), "middle": nm.get("middle", ""),
            "last": nm.get("last", ""), "dob": str((sd.get("dateOfBirth", "") if isinstance(sd, dict) else "") or "")}
def api_person(x):
    if "biographicInfo" in x or "_search" in x: return person(x)
    nm = x.get("name") if isinstance(x.get("name"), dict) else {}
    return {"id": str(x.get("identityId", "") or x.get("id", "")), "first": nm.get("first", ""),
            "middle": nm.get("middle", ""), "last": nm.get("last", ""),
            "dob": str(x.get("dateOfBirth", "") or "").replace("-", "")}

def call(run, f, tpl, scal):
    try:
        if run["path"] == "service":
            body, method = build_service(f, run["cid"])
            r = SESSION.post(run["url"], headers=run["h"], json=body, timeout=TIMEOUT)
            if r.status_code >= 400: return [], 0, r.status_code, r.text[:300], None, method
            j = r.json()
            ppl = [api_person(x) for k in ("exactMatches", "similarMatches")
                   for x in ((j.get(k) or {}).get("content") or [])]
            tot = sum((j.get(k) or {}).get("totalElements", 0) or 0 for k in ("exactMatches", "similarMatches"))
            return ppl, tot, r.status_code, "", j.get("clientId"), method
        body = build_dsl(tpl, f, scal)
        qh = hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        r = SESSION.post(run["url"], headers=run["h"], json=body, timeout=TIMEOUT)
        if r.status_code >= 400: return [], 0, r.status_code, r.text[:300], None, qh
        j = r.json()
        return ([person(h.get("_source", {})) for h in j["hits"]["hits"]],
                (j["hits"]["total"] or {}).get("value", 0), r.status_code, "", None, qh)
    except Exception as e:
        return [], 0, None, f"{type(e).__name__}: {e}"[:300], None, ""

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
    res, tot, st, err, cid_back, _ = call(r, probe, t["tpl"], t["scal"])
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
    res, tot, st, err, cid_back, method = call(run, f, t["tpl"], t["scal"])
    top = res[0] if res else None
    ids = [r["id"] for r in res]
    rank = ids.index(pid) + 1 if pid and pid in ids else None
    done[0] += 1
    if PROGRESS_EVERY and done[0] % PROGRESS_EVERY == 0:
        el = time.time() - t0
        rate = done[0] / el if el else 0
        left = (total - done[0]) / rate if rate else 0
        print(f"  {done[0]}/{total} calls, {rate:.0f}/sec, about {left/60:.0f} min left")
    return {"source_file": c["source_file"], "consumer": c["consumer"], "dup": c["dup"],
            "input_name": f"{f['FIRSTNAME']} {f['MIDDLENAME']} {f['LASTNAME']}".strip(),
            "input_dob": f["DOB"], "input_anumber": f["ANUMBER"], "input_receipt": f["RECEIPT"],
            "log_returned": c["pname"], "log_identity_id": pid,
            "template": run["tpl"], "environment": run["env"], "path": run["path"],
            "template_used": cid_back if run["path"] == "service" else run["tpl"],
            "searchable": bool(pid),
            "matched": rank is not None, "rank": rank,
            "top_returned": f"{top['first']} {top['last']}".strip() if top else
            ("(call failed)" if err else "(no result)"),
            "returned_count": len(res), "total_hits": tot,
            "search_key": "|".join([c["source_file"], c["consumer"]] + list(f.values())),
            "top_id": top["id"] if top else "",
            "status": st, "error": err,
            "query_id": method if run["path"] == "direct" else ""}

with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
    rows = list(pool.map(do, tasks))
el = time.time() - t0
print(f"{total} calls in {el/60:.1f} min ({total/el if el else 0:.0f}/sec)")
long = pd.DataFrame(rows)

score = []
for (e, t, p), s in long.groupby(["environment", "template", "path"]):
    s = s[s["searchable"]]
    n = len(s)
    if not n: continue
    m = s[s["matched"]]
    r = m["rank"].dropna()
    pct = lambda x: round(100 * x / n, 1)
    score.append({"environment": e, "template": t, "path": p,
                  "searches": n, "matched": len(m), "not_matched": n - len(m),
                  "match_rate_pct": pct(len(m)),
                  "rank_1": int((r == 1).sum()), "rank_1_pct": pct((r == 1).sum()),
                  "rank_2_10": int(((r >= 2) & (r <= 10)).sum()),
                  "rank_2_10_pct": pct(((r >= 2) & (r <= 10)).sum()),
                  "top_10_pct": pct((r <= 10).sum()),
                  "rank_11_plus": int((r > 10).sum()), "rank_11_plus_pct": pct((r > 10).sum()),
                  "not_matched_pct": pct(n - len(m)),
                  "median_rank": float(r.median()) if len(r) else None,
                  "worst_rank": int(r.max()) if len(r) else None,
                  "max_results_returned": int(s["returned_count"].max()),
                  "ran_requested_template": "yes" if p == "direct" else
                  ("unknown, service did not echo clientId" if not set(s["template_used"].dropna())
                   else ("yes" if set(s["template_used"].dropna()) == {t} else
                         "no, service used " + ", ".join(sorted(set(s["template_used"].dropna())))))})
if not score:
    raise SystemExit(
        "No search has an identity recorded in the production log, so there is nothing to match against. "
        "Check that the log rows contain a result block with an identityId.")
scorecard = pd.DataFrame(score).sort_values(["environment", "path", "match_rate_pct"], ascending=[1, 1, 0])

capped = scorecard[scorecard["max_results_returned"] < SIZE]
if len(capped):
    for _, r in capped.iterrows():
        print(f"NOTE {r['environment']} {r['template']} {r['path']}: never returned more than "
              f"{r['max_results_returned']} results although {SIZE} were requested. If that number is the "
              f"same on every search, the endpoint is capping the result set and ranks beyond it cannot "
              f"be seen.")
bad = scorecard[scorecard["ran_requested_template"] != "yes"]
if len(bad):
    print(f"\n{len(bad)} rows cannot be confirmed as running the template requested:")
    print(bad[["environment", "template", "path", "ran_requested_template"]].to_string(index=False))
    print("Where the service does not echo clientId back there is no way to tell from the response whether "
          "the named config was applied or the default was used. Identical numbers across templates on that "
          "path are expected if the default was used for all of them.")

sr = long[long["searchable"]].copy()
diff = []
for (t, p), s in sr.groupby(["template", "path"]):
    d = {e: round(100 * g["matched"].sum() / len(g), 1) for e, g in s.groupby("environment")}
    if "staging" in d and "prod" in d:
        diff.append({"template": t, "path": p, "staging_match_rate_pct": d["staging"],
                     "prod_match_rate_pct": d["prod"],
                     "staging_minus_prod": round(d["staging"] - d["prod"], 1)})
env_diff = pd.DataFrame(diff)

by_file = (sr.groupby(["source_file", "environment", "template", "path"])
             .agg(searches=("matched", "size"), matched=("matched", "sum"))
             .reset_index())
by_file["match_rate_pct"] = (100 * by_file["matched"] / by_file["searches"]).round(1)
by_file["not_matched"] = by_file["searches"] - by_file["matched"]
by_file = by_file[["source_file", "environment", "template", "path",
                   "searches", "matched", "not_matched", "match_rate_pct"]]

sr["run"] = sr["environment"] + " | " + sr["template"] + " | " + sr["path"]
by_consumer = sr.pivot_table(index=["source_file", "consumer"], columns="run", values="matched",
                             aggfunc=lambda x: round(100 * sum(x) / len(x), 1)).reset_index()
by_consumer.insert(2, "searches", sr.groupby(["source_file", "consumer"])["matched"].size().values //
                   max(sr["run"].nunique(), 1))

no_id = long[~long["searchable"]]["input_name"].nunique() if len(long) else 0
if no_id:
    tot_searches = long.groupby(["environment", "template", "path"]).size().max()
    print(f"\n{no_id} of {tot_searches} searches have no identity recorded in the production log. "
          f"There is nothing to match against, so they are excluded from every match rate below. "
          f"Match rates are calculated over the {tot_searches - no_id} searches that can be checked.")

if "query_id" in long:
    qsum = (long[long["path"] == "direct"].groupby("template")["query_id"]
            .agg(searches="size", distinct_queries="nunique").reset_index())
    if len(qsum):
        print("\nQUERIES BUILT PER TEMPLATE (direct path)")
        print(qsum.to_string(index=False))
        allq = long[long["path"] == "direct"].pivot(index="search_key", columns="template", values="query_id")
        if len(allq.columns) > 1:
            same_everywhere = int((allq.nunique(axis=1) == 1).sum())
            print(f"{same_everywhere} of {len(allq)} searches produced the SAME query under every template. "
                  f"For those searches the templates cannot differ, whatever their tier counts say.")

stability = pd.DataFrame()
if STABILITY_SAMPLE:
    sample = [t for t in tasks[:STABILITY_SAMPLE * len(runs)]][:STABILITY_SAMPLE]
    srows = []
    for c, run in sample:
        t = tpls[run["tpl"]]
        r1 = call(run, c["f"], t["tpl"], t["scal"])
        r2 = call(run, c["f"], t["tpl"], t["scal"])
        top1 = r1[0][0]["id"] if r1[0] else ""
        top2 = r2[0][0]["id"] if r2[0] else ""
        srows.append({"run": run["key"], "environment": run["env"], "path": run["path"],
                      "template": run["tpl"], "same_query_same_top": top1 == top2,
                      "first_call_top_id": top1, "second_call_top_id": top2})
    stability = pd.DataFrame(srows)
    agree = stability["same_query_same_top"].mean() * 100
    print(f"\nSTABILITY CHECK: the same query sent twice returned the same top result on "
          f"{agree:.1f} percent of {len(stability)} repeats.")
    if agree < 100:
        print("The index does not return a stable top result for an identical query. Production is a live "
              "index being updated while this runs, and a query with no deterministic tie break can order "
              "equally scored records differently between calls. Template differences smaller than this "
              "noise level cannot be trusted.")
        print(stability[~stability["same_query_same_top"]]
              [["run", "first_call_top_id", "second_call_top_id"]].head(10).to_string(index=False))

tdiff = []
for (env, path), g in sr.groupby(["environment", "path"]):
    dup = g.duplicated(subset=["search_key", "template"]).sum()
    if dup:
        print(f"WARNING {env} {path}: {dup} duplicate search and template rows. Comparison may be wrong.")
    piv = g.pivot(index="search_key", columns="template", values="top_id")
    tmpls = list(piv.columns)
    for i in range(len(tmpls)):
        for j in range(i + 1, len(tmpls)):
            a, b = tmpls[i], tmpls[j]
            both = piv[[a, b]].dropna()
            same_top = int((both[a] == both[b]).sum())
            row = {"environment": env, "path": path, "template_a": a, "template_b": b,
                   "searches_compared": len(both), "same_top_result": same_top,
                   "different_top_result": len(both) - same_top,
                   "same_top_pct": round(100 * same_top / len(both), 1) if len(both) else None,
                   "consistent": ""}
            if path == "direct" and "query_id" in g:
                qp = g.pivot(index="search_key", columns="template", values="query_id")
                if a in qp and b in qp:
                    qb = qp[[a, b]].dropna()
                    idq = int((qb[a] == qb[b]).sum())
                    row["identical_query_built"] = idq
                    row["identical_query_pct"] = round(100 * idq / len(qb), 1) if len(qb) else None
                    if row["identical_query_pct"] == 100 and row["same_top_pct"] != 100:
                        row["consistent"] = ("IMPOSSIBLE: identical queries cannot return different top "
                                             "results. Investigate before using these numbers.")
            tdiff.append(row)
template_diff = pd.DataFrame(tdiff)

if len(template_diff):
    ident = template_diff[template_diff.get("identical_query_pct", pd.Series(dtype=float)) == 100]
    if len(ident):
        print(f"\n{len(ident)} template pairs built a BYTE IDENTICAL query on every search. Those templates "
              f"cannot produce different results, so equal match rates are the correct outcome, not a bug.")
        print(ident[["environment", "path", "template_a", "template_b", "searches_compared"]].to_string(index=False))
    diff_q = template_diff[template_diff.get("identical_query_pct", pd.Series(dtype=float)) < 100]
    if len(diff_q):
        print(f"\nTemplate pairs that built different queries but returned the same top result:")
        print(diff_q[["environment", "path", "template_a", "template_b", "searches_compared",
                      "same_top_pct", "identical_query_pct"]].to_string(index=False))

print("\nSCORECARD"); display(scorecard)
if len(env_diff): print("\nSTAGING MINUS PROD"); display(env_diff)
print("\nBY CONSUMER"); display(by_consumer)
os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, f"Template_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xl:
    scorecard.to_excel(xl, sheet_name="Scorecard", index=False)
    if len(env_diff): env_diff.to_excel(xl, sheet_name="Staging vs prod", index=False)
    by_consumer.to_excel(xl, sheet_name="By consumer", index=False)
    by_file.to_excel(xl, sheet_name="By file", index=False)
    if len(template_diff): template_diff.to_excel(xl, sheet_name="Template differences", index=False)
    if len(stability): stability.to_excel(xl, sheet_name="Stability check", index=False)
    check.to_excel(xl, sheet_name="Template check", index=False)
    preflight.to_excel(xl, sheet_name="Health check", index=False)
    cols = ["source_file", "consumer", "input_name", "input_dob", "input_anumber", "input_receipt",
            "log_returned", "log_identity_id", "environment", "path", "template_used",
            "matched", "rank", "top_returned", "top_id", "returned_count", "total_hits", "status", "error"]
    for label in tpls:
        s = long[long["template"] == label]
        if not len(s): continue
        s = s.sort_values(["environment", "source_file", "rank"], na_position="last")[cols]
        if len(s) > EXCEL_ROW_LIMIT:
            csv_path = os.path.join(RESULTS, f"detail_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            s.to_csv(csv_path, index=False)
            print(f"{label}: {len(s)} rows, too many for a sheet, written to {csv_path}")
            s.head(EXCEL_ROW_LIMIT).to_excel(xl, sheet_name=label[:31], index=False)
        else:
            s.to_excel(xl, sheet_name=label[:31], index=False)
    e = long[long["error"] != ""]
    if len(e): e.to_excel(xl, sheet_name="Errors", index=False)
print(f"\nSaved: {out}")
