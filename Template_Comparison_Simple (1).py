# Databricks notebook source
import requests, json, re, os, glob
from datetime import datetime
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

TPL_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates"
TEMPLATES = ["search-default.yaml", "search-first.yaml", "search-max-clause-test.yaml",
             "search-reduced-tiers.yaml", "search-ui.yaml"]
BASELINE = "default"

LOGS_DIR  = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs"
LOG_FILES = ["BHUB.csv", "CRIS.csv", "ELIS.csv", "FIRST.csv", "GLOBAL.csv", "UIPATH.csv"]
RESULTS   = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"

SIZE, TIMEOUT = 100, 120

PH = re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")
NESTED = re.compile(r'("([^"\n]+)"\s*:\s*\{)(\s*)"\2"\s*:\s*\{')

START = re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')
def grab(pat, s, d=""):
    m = re.search(pat, s); return m.group(1) if m else d

def load_logs():
    cases = []
    for name in LOG_FILES:
        path = os.path.join(LOGS_DIR, name)
        if not os.path.exists(path):
            print(f"MISSING {name}"); continue
        txt = open(path, encoding="utf-8", errors="replace").read()
        st = [m.start() for m in START.finditer(txt)]
        recs = [txt[s:(st[i+1] if i+1 < len(st) else len(txt))] for i, s in enumerate(st)]
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
            cases.append({"consumer": consumer, "f": f, "pid": pid,
                          "pname": f"{nm.group(1)} {nm.group(2)}" if nm else ""}); n += 1
        print(f"{name}: {n} searches")
    seen, out = {}, []
    for c in cases:
        k = (c["consumer"],) + tuple(c["f"].values())
        if k in seen: seen[k]["dup"] += 1
        else: c["dup"] = 1; seen[k] = c; out.append(c)
    print(f"{len(cases)} rows -> {len(out)} distinct searches")
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

def load_template(name):
    path = os.path.join(TPL_DIR, name)
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
    b = {"page": 0, "clientId": client_id, "searchMethodType": method}
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
            r = requests.post(run["url"], headers=run["h"], json=body, timeout=TIMEOUT)
            if r.status_code >= 400: return [], 0, r.status_code, r.text[:300], None, method
            j = r.json()
            ppl = [api_person(x) for k in ("exactMatches", "similarMatches")
                   for x in ((j.get(k) or {}).get("content") or [])]
            tot = sum((j.get(k) or {}).get("totalElements", 0) or 0 for k in ("exactMatches", "similarMatches"))
            return ppl, tot, r.status_code, "", j.get("clientId"), method
        body = build_dsl(tpl, f, scal)
        r = requests.post(run["url"], headers=run["h"], json=body, timeout=TIMEOUT)
        if r.status_code >= 400: return [], 0, r.status_code, r.text[:300], None, ""
        j = r.json()
        return ([person(h.get("_source", {})) for h in j["hits"]["hits"]],
                (j["hits"]["total"] or {}).get("value", 0), r.status_code, "", None, "")
    except Exception as e:
        return [], 0, None, f"{type(e).__name__}: {e}"[:300], None, ""

tpls = {}
chk = []
for name in TEMPLATES:
    label = re.sub(r"^search-|\.(yaml|yml|txt)$", "", name)
    tpl, scal, note = load_template(name)
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

PROBE = {"FIRSTNAME": "MARIA", "MIDDLENAME": "", "LASTNAME": "GARCIA", "ANUMBER": "", "RECEIPT": "",
         "DOB": "19800101", "COB": "", "COC": ""}
pf = []
for r in runs:
    t = tpls[r["tpl"]]
    res, tot, st, err, cid_back, _ = call(r, PROBE, t["tpl"], t["scal"])
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
          f"but describe the default template and are excluded from the {BASELINE} comparison.")
print(f"\n{len(runs)} runs passed the health check\n")

cases = load_logs()
rows = []
for c in cases:
    f, pid = c["f"], c["pid"]
    for run in runs:
        t = tpls[run["tpl"]]
        res, tot, st, err, cid_back, method = call(run, f, t["tpl"], t["scal"])
        top = res[0] if res else None
        ids = [r["id"] for r in res]
        rank = ids.index(pid) + 1 if pid and pid in ids else None
        rows.append({"consumer": c["consumer"], "dup": c["dup"],
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
                     "status": st, "error": err})
long = pd.DataFrame(rows)

score = []
for (e, t, p), s in long.groupby(["environment", "template", "path"]):
    s = s[s["searchable"]]
    n = len(s)
    if not n: continue
    m = s[s["matched"]]
    r = m["rank"].dropna()
    score.append({"environment": e, "template": t, "path": p,
                  "searches": n, "matched": len(m), "not_matched": n - len(m),
                  "match_rate": round(100 * len(m) / n, 1),
                  "rank_1": int((r == 1).sum()),
                  "rank_2_10": int(((r >= 2) & (r <= 10)).sum()),
                  "rank_11_100": int((r > 10).sum()),
                  "best_rank": int(r.min()) if len(r) else None,
                  "median_rank": float(r.median()) if len(r) else None,
                  "worst_rank": int(r.max()) if len(r) else None,
                  "ran_requested_template": p == "direct" or
                  set(s["template_used"].dropna()) <= {t},
                  "errors": int((s["error"] != "").sum())})
scorecard = pd.DataFrame(score).sort_values(["environment", "path", "match_rate"], ascending=[1, 1, 0])
base = {(r["environment"], r["path"]): r["match_rate"]
        for _, r in scorecard.iterrows() if r["template"] == BASELINE and r["ran_requested_template"]}
scorecard["vs_" + BASELINE] = [
    None if r["template"] == BASELINE or not r["ran_requested_template"]
    or (r["environment"], r["path"]) not in base
    else round(r["match_rate"] - base[(r["environment"], r["path"])], 1)
    for _, r in scorecard.iterrows()]
if not scorecard["ran_requested_template"].all():
    bad = scorecard[~scorecard["ran_requested_template"]]
    print(f"\n{len(bad)} rows did not run the template requested. The service did not recognise the clientId "
          f"and used its default config, so those numbers describe the default, not the named template. "
          f"They are excluded from the {BASELINE} comparison.")
    print(bad[["environment", "template", "path"]].to_string(index=False))

sr = long[long["searchable"]].copy()
diff = []
for (t, p), s in sr.groupby(["template", "path"]):
    d = {e: round(100 * g["matched"].sum() / len(g), 1) for e, g in s.groupby("environment")}
    if "staging" in d and "prod" in d:
        diff.append({"template": t, "path": p, "staging_match_rate": d["staging"],
                     "prod_match_rate": d["prod"],
                     "staging_minus_prod": round(d["staging"] - d["prod"], 1)})
env_diff = pd.DataFrame(diff)

sr["run"] = sr["environment"] + " | " + sr["template"] + " | " + sr["path"]
by_consumer = sr.pivot_table(index="consumer", columns="run", values="matched",
                             aggfunc=lambda x: round(100 * sum(x) / len(x), 1)).reset_index()

no_id = long[~long["searchable"]]["input_name"].nunique() if len(long) else 0
if no_id:
    tot_searches = long.groupby(["environment", "template", "path"]).size().max()
    print(f"\n{no_id} of {tot_searches} searches have no identity recorded in the production log. "
          f"There is nothing to match against, so they are excluded from every match rate below. "
          f"Match rates are calculated over the {tot_searches - no_id} searches that can be checked.")

print("\nSCORECARD"); display(scorecard)
if len(env_diff): print("\nSTAGING MINUS PROD"); display(env_diff)
print("\nBY CONSUMER"); display(by_consumer)
os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, f"Template_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xl:
    scorecard.to_excel(xl, sheet_name="Scorecard", index=False)
    if len(env_diff): env_diff.to_excel(xl, sheet_name="Staging vs prod", index=False)
    by_consumer.to_excel(xl, sheet_name="By consumer", index=False)
    check.to_excel(xl, sheet_name="Template check", index=False)
    preflight.to_excel(xl, sheet_name="Health check", index=False)
    cols = ["consumer", "input_name", "input_dob", "input_anumber", "input_receipt",
            "log_returned", "log_identity_id", "environment", "path", "template_used",
            "matched", "rank", "top_returned", "returned_count", "total_hits", "status", "error"]
    for label in tpls:
        s = long[long["template"] == label]
        if len(s):
            s.sort_values(["environment", "consumer", "rank"], na_position="last")[cols] \
                .to_excel(xl, sheet_name=label[:31], index=False)
    e = long[long["error"] != ""]
    if len(e): e.to_excel(xl, sheet_name="Errors", index=False)
print(f"\nSaved: {out}")
