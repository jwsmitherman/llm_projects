# Databricks notebook source
import socket, ssl, http.client, time, json, re, os, glob, random, base64, itertools
from urllib.parse import urlparse, urlencode
from datetime import datetime
import requests
import pandas as pd

OAUTH_URL = "https://oauth-preprod.uscis.dhs.gov/uaa/oauth/token"
OAUTH_CLIENT_ID = "svc-oauth-pcisui-preprod"
OAUTH_CLIENT_SECRET = "PASTE_OAUTH_SECRET"
OAUTH_SCOPE = ""
OAUTH_CLIENT_AUTH = "auto"

BASIC_AUTH_TOKEN = ""
OPENSEARCH_AUTH_TOKEN = ""

PSS_URL = "https://pcis-search-service-staging.apps.k8s.uscis.dhs.gov/search"
OPENSEARCH_SEARCH_URL = ""
APIGEE_URL = ""
APIGEE_KEY_HEADER = "x-apigee-apikey"
APIGEE_KEY = ""

CLIENT_ID = "default"
TEMPLATE_YAML = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/updated_20260903/search-default.yaml"
PROD_LOGS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs"
RESULTS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"
REUSE_RAW_CSV = ""

EXTRA_NONPROD_HOSTS = []

GAPS_S = [5, 20, 35, 60, 120, 300]
REPS = 2
WARMUP_CALLS = 3
WARM_CALLS = 3
WARM_SPACING_S = 1.0
RESULT_SIZE = 10
TIMEOUT_S = 120
VERIFY_TLS = True
MATERIAL_MS = 250
RANDOM_SEED = 7

show = globals().get("display", print)
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_START = datetime.now().isoformat()

def is_set(v):
    v = (v or "").strip()
    return bool(v) and "PASTE" not in v and "<" not in v

def basic(tok):
    tok = (tok or "").strip()
    return tok if tok.lower().startswith("basic ") else "Basic " + tok

NONPROD_TOKENS = {"staging","stage","stg","dev","develop","development","preprod","nonprod","test","qa","uat","sit","sandbox","perf","lower"}

def env_verdict(text):
    s = (text or "").lower()
    s = re.sub(r"pre[-_.]?prod", "preprod", s)
    s = re.sub(r"non[-_.]?prod", "nonprod", s)
    toks = set(t for t in re.split(r"[^a-z0-9]+", s) if t)
    if "prod" in s.replace("preprod", "").replace("nonprod", "") or "prd" in toks:
        return False, "production marker present"
    host = (urlparse(text).hostname or "").lower() if "://" in (text or "") else ""
    if host and host in [h.lower() for h in EXTRA_NONPROD_HOSTS]:
        return True, "host allow-listed in EXTRA_NONPROD_HOSTS"
    if toks & NONPROD_TOKENS or any(re.fullmatch(r"(dev|stg|stage|staging|test|qa|uat|sit)\d+", t) for t in toks):
        return True, "non-production marker present"
    return False, "no non-production marker; add the host to EXTRA_NONPROD_HOSTS only after confirming it is non-production"

TOKEN_ENABLED = is_set(OAUTH_URL) and is_set(OAUTH_CLIENT_ID) and is_set(OAUTH_CLIENT_SECRET)
if not TOKEN_ENABLED and not is_set(BASIC_AUTH_TOKEN):
    raise RuntimeError("Set the OAuth client (OAUTH_URL, OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET) or BASIC_AUTH_TOKEN.")

LAYER_URLS = {"opensearch_direct": OPENSEARCH_SEARCH_URL, "pss_direct": PSS_URL, "apigee_internal": APIGEE_URL}
active, skipped, env_rows = [], [], []
for name, url in LAYER_URLS.items():
    if not is_set(url):
        skipped.append((name, "URL not set"))
        continue
    ok, why = env_verdict(url)
    env_rows.append({"layer": name, "url": url, "allowed": ok, "reason": why})
    if not ok:
        raise RuntimeError(f"REFUSED {name}: {url} ({why}). This notebook runs against non-production only.")
    active.append(name)

if TOKEN_ENABLED:
    ok, why = env_verdict(OAUTH_URL)
    env_rows.append({"layer": "oauth_token", "url": OAUTH_URL, "allowed": ok, "reason": why})
    if not ok:
        raise RuntimeError(f"REFUSED token endpoint: {OAUTH_URL} ({why}).")
    ok, why = env_verdict(OAUTH_CLIENT_ID)
    if why == "production marker present":
        raise RuntimeError(f"REFUSED OAuth client '{OAUTH_CLIENT_ID}': it carries a production marker.")

if "opensearch_direct" in active and not OPENSEARCH_SEARCH_URL.rstrip("/").endswith("_search"):
    raise RuntimeError("OPENSEARCH_SEARCH_URL must end in _search. A PSS URL in this slot returns empty 200s.")
for name in ("pss_direct", "apigee_internal"):
    if name in active and LAYER_URLS[name].rstrip("/").endswith("_search"):
        raise RuntimeError(f"{name} URL points at an OpenSearch _search endpoint; it must be the PSS /search contract.")
if "apigee_internal" in active and not is_set(APIGEE_KEY) and not TOKEN_ENABLED:
    active.remove("apigee_internal")
    skipped.append(("apigee_internal", "no Apigee key or OAuth client set"))
if "opensearch_direct" in active and not is_set(OPENSEARCH_AUTH_TOKEN) and not is_set(BASIC_AUTH_TOKEN):
    active.remove("opensearch_direct")
    skipped.append(("opensearch_direct", "no OpenSearch token set"))

if not active and not REUSE_RAW_CSV:
    raise RuntimeError(f"No layer is runnable: {skipped}")

PROXIED = any(os.environ.get(k) for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"))

def _ssl_ctx():
    if not VERIFY_TLS:
        c = ssl.create_default_context()
        c.check_hostname = False
        c.verify_mode = ssl.CERT_NONE
        return c, "verification disabled"
    ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if not ca:
        try:
            import certifi
            ca = certifi.where()
        except Exception:
            ca = None
    return (ssl.create_default_context(cafile=ca), ca) if ca else (ssl.create_default_context(), "system store")

SSL_CTX, CA_SOURCE = _ssl_ctx()
PHASES = ["dns_ms", "tcp_ms", "tls_ms", "send_ms", "ttfb_ms", "download_ms", "total_ms"]

def _ms(a, b):
    return round((b - a) * 1000, 2)

def timed_request(url, method, headers, body):
    r = {k: None for k in PHASES}
    r.update({"status": None, "error": "", "text": ""})
    data = None if body is None else (body if isinstance(body, bytes) else body.encode("utf-8"))
    if PROXIED:
        t0 = time.perf_counter()
        try:
            resp = requests.request(method, url, headers=headers, data=data, verify=VERIFY_TLS, timeout=TIMEOUT_S)
            r["ttfb_ms"] = round(resp.elapsed.total_seconds() * 1000, 2)
            r["status"] = resp.status_code
            r["text"] = resp.text
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:400]
        r["total_ms"] = _ms(t0, time.perf_counter())
    else:
        u = urlparse(url)
        https = u.scheme == "https"
        port = u.port or (443 if https else 80)
        path = (u.path or "/") + ("?" + u.query if u.query else "")
        sock = None
        t0 = time.perf_counter()
        try:
            info = socket.getaddrinfo(u.hostname, port, type=socket.SOCK_STREAM)[0]
            t1 = time.perf_counter(); r["dns_ms"] = _ms(t0, t1)
            sock = socket.socket(info[0], info[1], info[2])
            sock.settimeout(TIMEOUT_S)
            sock.connect(info[4])
            t2 = time.perf_counter(); r["tcp_ms"] = _ms(t1, t2)
            if https:
                sock = SSL_CTX.wrap_socket(sock, server_hostname=u.hostname)
            t3 = time.perf_counter(); r["tls_ms"] = _ms(t2, t3) if https else 0.0
            conn = http.client.HTTPConnection(u.hostname, port, timeout=TIMEOUT_S)
            conn.sock = sock
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", u.netloc.split("@")[-1])
            for k, v in headers.items():
                conn.putheader(k, v)
            if data is not None:
                conn.putheader("Content-Length", str(len(data)))
            conn.endheaders(data)
            t4 = time.perf_counter(); r["send_ms"] = _ms(t3, t4)
            resp = conn.getresponse()
            t5 = time.perf_counter(); r["ttfb_ms"] = _ms(t4, t5)
            raw = resp.read()
            t6 = time.perf_counter(); r["download_ms"] = _ms(t5, t6)
            r["status"] = resp.status
            r["text"] = raw.decode("utf-8", "replace")
            conn.close()
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:400]
        finally:
            try:
                if sock: sock.close()
            except Exception:
                pass
        r["total_ms"] = _ms(t0, time.perf_counter())
    if r["status"] and r["status"] >= 400 and not r["error"]:
        r["error"] = f"HTTP {r['status']}: {r['text'][:300]}"
    return r

JSON_HEADERS = {"Content-Type": "application/json", "Accept": "application/json", "Connection": "close"}
PSS_HEADERS = dict(JSON_HEADERS)
if is_set(BASIC_AUTH_TOKEN):
    PSS_HEADERS["Authorization"] = basic(BASIC_AUTH_TOKEN)
OS_HEADERS = dict(JSON_HEADERS, Authorization=basic(OPENSEARCH_AUTH_TOKEN if is_set(OPENSEARCH_AUTH_TOKEN) else BASIC_AUTH_TOKEN))
APIGEE_HEADERS = dict(PSS_HEADERS)
if is_set(APIGEE_KEY):
    APIGEE_HEADERS[APIGEE_KEY_HEADER] = APIGEE_KEY

OS_BASE, OS_ALIAS = "", ""
if is_set(OPENSEARCH_SEARCH_URL):
    _u = urlparse(OPENSEARCH_SEARCH_URL)
    _parts = [p for p in _u.path.split("/") if p]
    OS_BASE = f"{_u.scheme}://{_u.netloc}"
    OS_ALIAS = _parts[-2] if len(_parts) >= 2 and _parts[-1] == "_search" else ""

def os_get(path, params=None):
    resp = requests.get(OS_BASE + path, params=params, headers=OS_HEADERS, verify=VERIFY_TLS, timeout=30)
    resp.raise_for_status()
    return resp.json()

CLUSTER = {}
if OS_BASE:
    try:
        j = os_get("/")
        CLUSTER = {"cluster_name": j.get("cluster_name"), "version": (j.get("version") or {}).get("number"),
                   "distribution": (j.get("version") or {}).get("distribution")}
        ok, why = env_verdict(str(CLUSTER.get("cluster_name") or ""))
        if why == "production marker present":
            raise RuntimeError(f"REFUSED: OpenSearch cluster_name '{CLUSTER['cluster_name']}' carries a production marker.")
    except RuntimeError:
        raise
    except Exception as e:
        CLUSTER = {"cluster_name": None, "version": None, "error": f"{type(e).__name__}: {e}"[:200]}

def os_snapshot():
    if not (OS_BASE and OS_ALIAS):
        return None
    try:
        t = os_get(f"/{OS_ALIAS}/_stats/search,refresh,indexing")["_all"]["total"]
        return {"query_total": t["search"]["query_total"], "refresh_total": t["refresh"]["total"],
                "index_total": t["indexing"]["index_total"]}
    except Exception:
        return None

def parse_secs(v):
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(ms|s|m|h|d)\s*", str(v or ""))
    if not m:
        return None
    return float(m.group(1)) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]

def os_settings():
    if not (OS_BASE and OS_ALIAS):
        return pd.DataFrame(columns=["index"])
    try:
        j = os_get(f"/{OS_ALIAS}/_settings", {"include_defaults": "true", "flat_settings": "true"})
    except Exception as e:
        return pd.DataFrame([{"index": OS_ALIAS, "error": f"{type(e).__name__}: {e}"[:200]}])
    rows = []
    for idx, v in j.items():
        s = v.get("settings", {}) or {}
        d = v.get("defaults", {}) or {}
        rows.append({"index": idx,
                     "search_idle_after": s.get("index.search.idle.after", d.get("index.search.idle.after")),
                     "search_idle_after_source": "explicit" if "index.search.idle.after" in s else "default",
                     "refresh_interval": s.get("index.refresh_interval", d.get("index.refresh_interval")),
                     "refresh_interval_source": "explicit" if "index.refresh_interval" in s else "default",
                     "number_of_shards": s.get("index.number_of_shards"),
                     "number_of_replicas": s.get("index.number_of_replicas")})
    return pd.DataFrame(rows)

_START = re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')

def _g(pat, s, d=""):
    m = re.search(pat, s)
    return m.group(1) if m else d

def parse_fields(rec):
    consumer = _g(r'^([A-Z0-9_]+),', rec)
    rec = rec.replace('""', '"')
    i = rec.find('"result":')
    t = rec[:i] if i >= 0 else rec
    mid = _g(r'"personMiddleName":(null|"[^"]*")', t, "null")
    return consumer, {"FIRSTNAME": _g(r'"personGivenName":"([^"]*)"', t),
                      "MIDDLENAME": "" if mid in ("null", "") else mid.strip('"'),
                      "LASTNAME": _g(r'"personSurName":"([^"]*)"', t),
                      "ANUMBER": _g(r'"type":"ALIEN_NBR","value":"([^"]*)"', t),
                      "RECEIPT": _g(r'"type":"RECEIPT_NBR","value":"([^"]*)"', t),
                      "DOB": (_g(r'"dob":"(\d{4}-\d{2}-\d{2})"', t) or "").replace("-", ""),
                      "COB": _g(r'"cobs":\["([^"]*)"\]', t),
                      "COC": _g(r'"cocs":\["([^"]*)"\]', t)}

def load_cases(folder):
    out, seen = [], set()
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        txt = open(path, encoding="utf-8", errors="replace").read()
        starts = [m.start() for m in _START.finditer(txt)]
        for k, s in enumerate(starts):
            consumer, f = parse_fields(txt[s:(starts[k + 1] if k + 1 < len(starts) else len(txt))])
            if not (f["FIRSTNAME"] and f["LASTNAME"]):
                continue
            key = tuple(f.values())
            if key in seen:
                continue
            seen.add(key)
            out.append({"consumer": consumer, "fields": f})
    return out

PH = re.compile(r"\{\{\s*([A-Z_0-9]+)\s*\}\}")

def quote_bare(s):
    out, i, n, instr, esc = [], 0, len(s), False, False
    while i < n:
        c = s[i]
        if instr:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': instr = False
            out.append(c); i += 1; continue
        if c == '"':
            instr = True; out.append(c); i += 1; continue
        m = PH.match(s, i)
        if m:
            out.append('"' + m.group(0) + '"'); i = m.end(); continue
        out.append(c); i += 1
    return "".join(out)

def extract_template(txt):
    try:
        import yaml
        tpl = yaml.safe_load(txt)["search-config"]["similar-query-template"]
        if tpl:
            return tpl
    except Exception:
        pass
    key = re.search(r'similar-query-template\s*:\s*[|>]?', txt)
    if not key:
        raise ValueError("similar-query-template not found under search-config")
    return txt[txt.index("{", key.end()):]

def config_scalars(txt):
    out = {}
    try:
        import yaml
        cfg = (yaml.safe_load(txt) or {}).get("search-config", {}) or {}
        for k, v in cfg.items():
            if k != "similar-query-template" and isinstance(v, (str, int, float, bool)):
                out["{{" + k.upper().replace("-", "_") + "}}"] = str(v)
    except Exception:
        head = txt.split("similar-query-template")[0]
        for m in re.finditer(r"^\s{2,}([A-Za-z0-9\-]+)\s*:\s*([^\n|>#]+)$", head, re.MULTILINE):
            out["{{" + m.group(1).upper().replace("-", "_") + "}}"] = m.group(2).strip()
    out.setdefault("{{SIMILAR_SIZE}}", str(RESULT_SIZE))
    return out

def _has_ph(n):
    if isinstance(n, str): return bool(PH.search(n))
    if isinstance(n, list): return any(_has_ph(x) for x in n)
    if isinstance(n, dict): return any(_has_ph(v) for v in n.values())
    return False

def _empty_bool(n):
    b = n.get("bool") if isinstance(n, dict) else None
    return isinstance(b, dict) and not any(isinstance(b.get(k), list) and b[k] for k in ("must", "should", "filter", "must_not"))

def _prune(n):
    if isinstance(n, dict) and isinstance(n.get("bool"), dict):
        b = n["bool"]
        for key in ("must", "should"):
            if isinstance(b.get(key), list):
                for c in b[key]: _prune(c)
                b[key] = [c for c in b[key] if not _has_ph(c) and not _empty_bool(c)]
                if not b[key]: del b[key]
    for v in (n.values() if isinstance(n, dict) else n if isinstance(n, list) else []):
        if isinstance(v, (dict, list)): _prune(v)

def _strip(n):
    if isinstance(n, dict):
        for k, v in list(n.items()):
            if isinstance(v, str): n[k] = PH.sub("", v)
            elif isinstance(v, (dict, list)): _strip(v)
    elif isinstance(n, list):
        for i, v in enumerate(n):
            if isinstance(v, str): n[i] = PH.sub("", v)
            elif isinstance(v, (dict, list)): _strip(v)

def _has_real(n):
    if isinstance(n, dict):
        if isinstance(n.get("bool"), dict) and any(isinstance(n["bool"].get(k), list) and n["bool"][k] for k in ("must", "should", "filter")):
            return True
        if any(k in n for k in ("match", "term", "match_phrase", "multi_match", "prefix", "fuzzy", "range")):
            return True
        return any(_has_real(v) for v in n.values())
    if isinstance(n, list): return any(_has_real(v) for v in n)
    return False

def render(tpl, f, scalars):
    s = tpl
    for ph, val in scalars.items():
        s = s.replace(ph, val)
    p = {k: ("" if v is None else str(v)) for k, v in f.items()}
    ids = []
    if f.get("ANUMBER"): ids.append(("ALIEN_NBR", f["ANUMBER"]))
    if f.get("RECEIPT"): ids.append(("RECEIPT_NBR", f["RECEIPT"]))
    for i, (nm, val) in enumerate(ids[:2], start=1):
        p[f"IDENTIFIER_NAME_{i}"] = nm
        p[f"IDENTIFIER_VALUE_{i}"] = val
    s = PH.sub(lambda m: (p[m.group(1)] if p.get(m.group(1)) else m.group(0)), s)
    s = re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
    s = quote_bare(s)
    if s.lstrip()[:1] != "{":
        s = "{" + s + "}"
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    q = json.loads(s)
    _prune(q); _strip(q)
    q["size"] = RESULT_SIZE
    return q

TEMPLATE = {"tpl": None, "scalars": {}, "status": ""}
try:
    _txt = open(TEMPLATE_YAML).read()
    TEMPLATE.update(tpl=extract_template(_txt), scalars=config_scalars(_txt), status="loaded")
except Exception as e:
    TEMPLATE["status"] = f"not loaded: {type(e).__name__}: {e}"[:200]

def os_body(f):
    if TEMPLATE["tpl"]:
        try:
            q = render(TEMPLATE["tpl"], f, TEMPLATE["scalars"])
            if q.get("query") and _has_real(q["query"]):
                return q, "template"
        except Exception:
            pass
    return {"size": RESULT_SIZE, "query": {"bool": {"should": [
        {"match": {"biographicInfo.name.first": f["FIRSTNAME"]}},
        {"match": {"biographicInfo.name.last": f["LASTNAME"]}}], "minimum_should_match": 1}}}, "fallback name match"

def pss_body(f):
    b = {"page": 0, "clientId": CLIENT_ID, "searchMethodType": "advancedSearch"}
    nm = {k2: f[k1] for k1, k2 in (("FIRSTNAME", "first"), ("MIDDLENAME", "middle"), ("LASTNAME", "last")) if f.get(k1)}
    if nm: b["names"] = [nm]
    d = (f.get("DOB") or "").replace("-", "")
    if len(d) == 8: b["dobs"] = [{"dob": f"{d[:4]}-{d[4:6]}-{d[6:]}"}]
    if f.get("COB"): b["cobs"] = [f["COB"]]
    if f.get("COC"): b["cocs"] = [f["COC"]]
    ids = []
    if f.get("ANUMBER"): ids.append({"type": "ALIEN_NBR", "value": f["ANUMBER"]})
    if f.get("RECEIPT"): ids.append({"type": "RECEIPT_NBR", "value": f["RECEIPT"]})
    if ids: b["identifiers"] = ids
    return b

OAUTH_MODE = {"mode": OAUTH_CLIENT_AUTH}

def _token_call(mode):
    data = {"grant_type": "client_credentials"}
    if is_set(OAUTH_SCOPE):
        data["scope"] = OAUTH_SCOPE
    h = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json", "Connection": "close"}
    if mode == "basic":
        h["Authorization"] = "Basic " + base64.b64encode(f"{OAUTH_CLIENT_ID}:{OAUTH_CLIENT_SECRET}".encode()).decode()
    else:
        data["client_id"] = OAUTH_CLIENT_ID
        data["client_secret"] = OAUTH_CLIENT_SECRET
    r = timed_request(OAUTH_URL, "POST", h, urlencode(data))
    tok = None
    if not r["error"]:
        try:
            tok = json.loads(r["text"]).get("access_token")
            if not tok: r["error"] = "no access_token in token response"
        except Exception:
            r["error"] = "token response was not JSON"
    return r, tok

def get_token():
    return _token_call(OAUTH_MODE["mode"])

if TOKEN_ENABLED and not REUSE_RAW_CSV and OAUTH_MODE["mode"] not in ("basic", "body"):
    errs = {}
    for mode in ("basic", "body"):
        r, tok = _token_call(mode)
        if tok:
            OAUTH_MODE["mode"] = mode
            break
        errs[mode] = r["error"][:200]
    else:
        raise RuntimeError(f"OAuth token call failed in both client-auth modes: {errs}")
    print(f"OAuth token call works with client credentials sent as: {OAUTH_MODE['mode']}")

def _conn(r):
    v = [r.get(k) for k in ("dns_ms", "tcp_ms", "tls_ms") if r.get(k) is not None]
    return round(sum(v), 2) if v else None

def call_layer(layer, f):
    row = {"query_source": "", "took_ms": None, "result_count": None, "client_id_returned": None}
    auth = None
    if layer == "opensearch_direct":
        body, row["query_source"] = os_body(f)
        r = timed_request(OPENSEARCH_SEARCH_URL, "POST", OS_HEADERS, json.dumps(body))
        if not r["error"]:
            try:
                j = json.loads(r["text"])
                row["took_ms"] = j.get("took")
                row["result_count"] = len((j.get("hits") or {}).get("hits") or [])
            except Exception:
                r["error"] = "response was not JSON"
    else:
        url = PSS_URL if layer == "pss_direct" else APIGEE_URL
        h = dict(PSS_HEADERS if layer == "pss_direct" else APIGEE_HEADERS)
        row["query_source"] = f"PSS contract, clientId={CLIENT_ID}"
        if TOKEN_ENABLED:
            auth, tok = get_token()
            if tok:
                h["Authorization"] = "Bearer " + tok
        if auth is not None and auth["error"]:
            r = {k: None for k in PHASES}
            r.update({"status": None, "error": "token call failed: " + auth["error"], "text": ""})
        else:
            r = timed_request(url, "POST", h, json.dumps(pss_body(f)))
            if not r["error"]:
                try:
                    j = json.loads(r["text"])
                    row["client_id_returned"] = j.get("clientId")
                    cnt = 0
                    for k in ("exactMatches", "similarMatches"):
                        c = j.get(k)
                        content = c.get("content") if isinstance(c, dict) else None
                        cnt += len(content) if isinstance(content, list) else 0
                    row["result_count"] = cnt
                except Exception:
                    r["error"] = "response was not JSON"
    for k in PHASES:
        row[k] = r.get(k)
    row["status"] = r.get("status")
    row["error"] = r.get("error") or ""
    row["conn_ms"] = _conn(r)
    if auth is not None:
        for k in PHASES:
            row["auth_" + k] = auth.get(k)
        row["auth_status"] = auth.get("status")
        row["auth_error"] = auth.get("error") or ""
        row["auth_conn_ms"] = _conn(auth)
    row["e2e_ms"] = None if row["total_ms"] is None else round(row["total_ms"] + ((auth or {}).get("total_ms") or 0), 2)
    return row

def _delta(a, b, k):
    return (b[k] - a[k]) if a and b and a.get(k) is not None and b.get(k) is not None else None

os.makedirs(RESULTS_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESULTS_DIR, f"ColdStart_raw_{RUN_TS}.csv")
calls, layer_failed, cases = [], {}, []

if REUSE_RAW_CSV:
    calls = pd.read_csv(REUSE_RAW_CSV).to_dict("records")
    print(f"Reusing {len(calls)} calls from {REUSE_RAW_CSV}")
else:
    cases = load_cases(PROD_LOGS_DIR)
    if not cases:
        raise RuntimeError(f"No name-bearing searches found in {PROD_LOGS_DIR}. Inputs come only from logged searches.")
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(cases)
    pool = itertools.cycle(cases)
    per_layer_s = REPS * (sum(GAPS_S) + len(GAPS_S) * WARM_CALLS * (WARM_SPACING_S + 1))
    print(f"Layers: {active}  Skipped: {skipped}")
    print(f"Template: {TEMPLATE['status']}  Cases: {len(cases)}  Timing mode: {'requests (proxy present)' if PROXIED else 'socket phases'}")
    print(f"OpenSearch cluster: {CLUSTER or 'not configured'}")
    print(f"Estimated runtime: about {per_layer_s * len(active) / 60:.0f} minutes\n")
    state = {"last_end": time.perf_counter()}

    def record(layer, tid, rep, gap, kind, seq, idle=None):
        case = next(pool)
        started = datetime.now().isoformat()
        row = call_layer(layer, case["fields"])
        state["last_end"] = time.perf_counter()
        row.update({"layer": layer, "trial_id": tid, "rep": rep, "gap_s": gap, "kind": kind, "seq": seq,
                    "consumer": case["consumer"], "started_local": started, "idle_actual_s": idle})
        calls.append(row)
        return row

    for layer in active:
        good = 0
        for i in range(WARMUP_CALLS):
            good += 0 if record(layer, f"{layer}|warmup", None, None, "warmup", i)["error"] else 1
        if WARMUP_CALLS and good == 0:
            layer_failed[layer] = calls[-1]["error"]
            print(f"[{layer}] every warmup call failed, layer skipped: {calls[-1]['error'][:200]}")
            continue
        trials = [(rep, g) for rep in range(REPS) for g in GAPS_S]
        rng.shuffle(trials)
        for n, (rep, gap) in enumerate(trials, 1):
            tid = f"{layer}|r{rep}|g{gap}"
            s0 = os_snapshot()
            time.sleep(gap)
            s1 = os_snapshot()
            idle = round(time.perf_counter() - state["last_end"], 2)
            cold = record(layer, tid, rep, gap, "cold", 0, idle)
            s2 = os_snapshot()
            cold.update({"query_delta_idle": _delta(s0, s1, "query_total"),
                         "refresh_delta_idle": _delta(s0, s1, "refresh_total"),
                         "index_delta_idle": _delta(s0, s1, "index_total"),
                         "refresh_delta_cold": _delta(s1, s2, "refresh_total")})
            warm = []
            for w in range(WARM_CALLS):
                time.sleep(WARM_SPACING_S)
                warm.append(record(layer, tid, rep, gap, "warm", w + 1))
            wt = [x["ttfb_ms"] for x in warm if not x["error"] and x["ttfb_ms"] is not None]
            print(f"[{layer}] {n}/{len(trials)} gap={gap}s idle={idle}s cold_ttfb={cold['ttfb_ms']} "
                  f"warm_ttfb_med={sorted(wt)[len(wt)//2] if wt else None} e2e={cold['e2e_ms']} "
                  f"refresh_on_cold={cold['refresh_delta_cold']} other_searches_during_idle={cold['query_delta_idle']} "
                  f"{('ERROR ' + cold['error'][:120]) if cold['error'] else ''}")
            pd.DataFrame(calls).to_csv(RAW_CSV, index=False)
    pd.DataFrame(calls).to_csv(RAW_CSV, index=False)

num = lambda s: pd.to_numeric(s, errors="coerce")
cd = pd.DataFrame(calls)
for c in ["error", "auth_total_ms", "took_ms", "conn_ms", "e2e_ms", "ttfb_ms", "total_ms", "dns_ms", "tcp_ms", "tls_ms",
          "query_delta_idle", "refresh_delta_idle", "index_delta_idle", "refresh_delta_cold", "idle_actual_s"]:
    if c not in cd:
        cd[c] = None
cd["error"] = cd["error"].fillna("").astype(str)
cd["ok"] = cd["error"].eq("")

tr = []
for tid, g in cd[cd["kind"].isin(["cold", "warm"])].groupby("trial_id", sort=False):
    c = g[g["kind"] == "cold"]
    if c.empty:
        continue
    c = c.iloc[0]
    w = g[(g["kind"] == "warm") & g["ok"]]
    cv = lambda col: (None if pd.isna(num(pd.Series([c[col]])).iloc[0]) else float(num(pd.Series([c[col]])).iloc[0]))
    wm = lambda col: (float(num(w[col]).dropna().median()) if num(w[col]).notna().any() else None)
    pen = lambda col: (None if cv(col) is None or wm(col) is None else round(cv(col) - wm(col), 1))
    qd = cv("query_delta_idle")
    rec = {"layer": c["layer"], "trial_id": tid, "rep": c["rep"], "gap_s": c["gap_s"], "idle_actual_s": cv("idle_actual_s"),
           "cold_ok": bool(c["ok"]), "warm_ok_calls": int(len(w)), "contaminated": bool(qd is not None and qd > 0),
           "cold_ttfb_ms": cv("ttfb_ms"), "warm_ttfb_ms": wm("ttfb_ms"), "ttfb_penalty_ms": pen("ttfb_ms"),
           "cold_conn_ms": cv("conn_ms"), "warm_conn_ms": wm("conn_ms"), "conn_penalty_ms": pen("conn_ms"),
           "cold_auth_ms": cv("auth_total_ms"), "warm_auth_ms": wm("auth_total_ms"), "auth_penalty_ms": pen("auth_total_ms"),
           "cold_e2e_ms": cv("e2e_ms"), "warm_e2e_ms": wm("e2e_ms"), "e2e_penalty_ms": pen("e2e_ms"),
           "cold_took_ms": cv("took_ms"), "warm_took_ms": wm("took_ms"), "took_penalty_ms": pen("took_ms"),
           "other_searches_during_idle": qd, "docs_indexed_during_idle": cv("index_delta_idle"),
           "refreshes_during_idle": cv("refresh_delta_idle"), "refreshes_on_cold_call": cv("refresh_delta_cold"),
           "cold_error": c["error"]}
    rec["clean"] = rec["cold_ok"] and rec["warm_ok_calls"] > 0 and not rec["contaminated"]
    tr.append(rec)
trials = pd.DataFrame(tr)

def _agg(g):
    m = lambda col: (round(float(num(g[col]).dropna().median()), 1) if num(g[col]).notna().any() else None)
    mx = lambda col: (round(float(num(g[col]).dropna().max()), 1) if num(g[col]).notna().any() else None)
    rc = num(g["refreshes_on_cold_call"]).dropna()
    return pd.Series({"clean_trials": len(g), "cold_ttfb_med": m("cold_ttfb_ms"), "warm_ttfb_med": m("warm_ttfb_ms"),
                      "ttfb_penalty_med": m("ttfb_penalty_ms"), "ttfb_penalty_max": mx("ttfb_penalty_ms"),
                      "conn_penalty_med": m("conn_penalty_ms"), "auth_penalty_med": m("auth_penalty_ms"),
                      "cold_e2e_med": m("cold_e2e_ms"), "cold_e2e_max": mx("cold_e2e_ms"), "e2e_penalty_med": m("e2e_penalty_ms"),
                      "cold_took_med": m("cold_took_ms"), "warm_took_med": m("warm_took_ms"),
                      "refresh_on_cold_rate": round(float((rc > 0).mean()), 2) if len(rc) else None,
                      "docs_indexed_during_idle_med": m("docs_indexed_during_idle")})

if len(trials):
    clean = trials[trials["clean"]]
    by_gap = trials.groupby(["layer", "gap_s"]).agg(all_trials=("trial_id", "count"),
                                                    contaminated=("contaminated", "sum"),
                                                    cold_errors=("cold_ok", lambda s: int((~s.astype(bool)).sum()))).reset_index()
    if len(clean):
        by_gap = by_gap.merge(clean.groupby(["layer", "gap_s"]).apply(_agg).reset_index(), on=["layer", "gap_s"], how="left")
else:
    clean, by_gap = pd.DataFrame(), pd.DataFrame(columns=["layer", "gap_s"])

def _lg(layer, col):
    if "layer" not in by_gap or col not in by_gap or not (by_gap["layer"] == layer).any():
        return {}
    return by_gap[by_gap["layer"] == layer].set_index("gap_s")[col].to_dict()

attr = pd.DataFrame({"gap_s": sorted(by_gap["gap_s"].dropna().unique())}) if len(by_gap) else pd.DataFrame(columns=["gap_s"])
for l in ("opensearch_direct", "pss_direct", "apigee_internal"):
    attr[f"{l}_ttfb_penalty_med"] = attr["gap_s"].map(_lg(l, "ttfb_penalty_med"))
attr["pss_direct_token_penalty_med"] = attr["gap_s"].map(_lg("pss_direct", "auth_penalty_med"))
attr["pss_direct_e2e_penalty_med"] = attr["gap_s"].map(_lg("pss_direct", "e2e_penalty_med"))
attr["apigee_internal_auth_penalty_med"] =attr["gap_s"].map(_lg("apigee_internal", "auth_penalty_med"))
attr["apigee_internal_e2e_penalty_med"] = attr["gap_s"].map(_lg("apigee_internal", "e2e_penalty_med"))
attr["pss_over_opensearch_ms"] = num(attr["pss_direct_ttfb_penalty_med"]) - num(attr["opensearch_direct_ttfb_penalty_med"])
attr["apigee_e2e_over_pss_ms"] = num(attr["apigee_internal_e2e_penalty_med"]) - num(attr["pss_direct_ttfb_penalty_med"])

warm_calls = cd[(cd["kind"] == "warm") & cd["ok"]]
steady = []
for l, g in warm_calls.groupby("layer"):
    q = lambda col, p=0.5: (round(float(num(g[col]).dropna().quantile(p)), 1) if num(g[col]).notna().any() else None)
    steady.append({"layer": l, "warm_calls": len(g), "ttfb_med": q("ttfb_ms"), "ttfb_p90": q("ttfb_ms", 0.9),
                   "total_med": q("total_ms"), "dns_med": q("dns_ms"), "tcp_med": q("tcp_ms"), "tls_med": q("tls_ms"),
                   "conn_med": q("conn_ms"), "auth_total_med": q("auth_total_ms"), "e2e_med": q("e2e_ms"), "took_med": q("took_ms")})
steady = pd.DataFrame(steady)

settings_df = os_settings()
idle_vals = [parse_secs(v) for v in settings_df.get("search_idle_after", pd.Series(dtype=object)).tolist()]
idle_vals = [v for v in idle_vals if v is not None]
IDLE_AFTER_S = min(idle_vals) if idle_vals else 30.0
IDLE_SRC = "read from index settings" if idle_vals else "assumed OpenSearch default of 30s (settings not readable)"
explicit_refresh = bool(len(settings_df)) and "refresh_interval_source" in settings_df and (settings_df["refresh_interval_source"] == "explicit").any()

def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.0f} ms"

def med_where(layer, col, cond):
    if not len(clean):
        return None
    s = num(clean[(clean["layer"] == layer) & cond(num(clean["gap_s"]))][col]).dropna()
    return float(s.median()) if len(s) else None

def steady_val(layer, col):
    if not len(steady) or not (steady["layer"] == layer).any():
        return None
    v = steady.loc[steady["layer"] == layer, col].iloc[0]
    return None if v is None or pd.isna(v) else float(v)

short = lambda s: s < IDLE_AFTER_S
long_ = lambda s: s > IDLE_AFTER_S
findings = []
add = lambda area, text: findings.append({"area": area, "finding": text})

add("Scope", f"Layers measured: {', '.join(l for l in active if l not in layer_failed) or 'none'}. "
             f"Skipped: {'; '.join(f'{a} ({b})' for a, b in skipped) or 'none'}. "
             f"Failed at warmup: {'; '.join(f'{a} ({b[:120]})' for a, b in layer_failed.items()) or 'none'}. "
             f"All endpoints passed the non-production check; no production endpoint was called.")
if OS_BASE:
    add("OpenSearch settings", f"Cluster {CLUSTER.get('cluster_name')} version {CLUSTER.get('version')}. "
        f"index.search.idle.after = {IDLE_AFTER_S:g}s ({IDLE_SRC}). "
        + ("refresh_interval is set explicitly on at least one index behind the alias, so shards there do not go search-idle and the search-idle cold start should not apply to them."
           if explicit_refresh else
           "refresh_interval is not set explicitly, so shards that see no search for longer than search.idle.after stop refreshing and the first search after that waits for a refresh."))
else:
    add("OpenSearch settings", f"OpenSearch endpoint not configured; search.idle.after {IDLE_SRC}. Settings, refresh counts and contamination checks were not available.")

if "opensearch_direct" in active:
    s, l = med_where("opensearch_direct", "ttfb_penalty_ms", short), med_where("opensearch_direct", "ttfb_penalty_ms", long_)
    rs = med_where("opensearch_direct", "refreshes_on_cold_call", long_)
    ix = med_where("opensearch_direct", "docs_indexed_during_idle", long_)
    verdict = ("consistent with the search-idle cold start: the penalty appears only once the idle gap exceeds search.idle.after"
               if s is not None and l is not None and l - s >= MATERIAL_MS else
               f"no material cold penalty at the OpenSearch layer at the tested gaps (threshold {MATERIAL_MS} ms)")
    add("OpenSearch direct", f"Median server-side cold penalty (cold minus warm time to first byte): {fmt(s)} for gaps under "
        f"{IDLE_AFTER_S:g}s, {fmt(l)} for gaps over. This is {verdict}. Median refreshes triggered by the cold call after a long gap: "
        f"{'n/a' if rs is None else f'{rs:g}'}. Median documents indexed during long idle gaps: {'n/a' if ix is None else f'{ix:g}'}. "
        f"The refresh cost scales with writes during the idle period, so a quiet staging index understates what an index with steady writes would show.")
if "pss_direct" in active:
    l = med_where("pss_direct", "ttfb_penalty_ms", long_)
    s = med_where("pss_direct", "ttfb_penalty_ms", short)
    o = med_where("opensearch_direct", "ttfb_penalty_ms", long_) if "opensearch_direct" in active else None
    add("PSS direct", f"Median cold penalty: {fmt(s)} for gaps under {IDLE_AFTER_S:g}s, {fmt(l)} for gaps over. "
        + (f"Above the OpenSearch layer's own penalty by {fmt(None if l is None or o is None else l - o)}, which is the share attributable to the service itself (JVM, connection pool to OpenSearch, pod idle behavior)."
           if o is not None else "OpenSearch direct was not measured, so the PSS penalty cannot be split from the OpenSearch penalty."))
    if TOKEN_ENABLED:
        tc, tw = med_where("pss_direct", "cold_auth_ms", long_), steady_val("pss_direct", "auth_total_med")
        tp_s, tp_l = med_where("pss_direct", "auth_penalty_ms", short), med_where("pss_direct", "auth_penalty_ms", long_)
        ec, ew = med_where("pss_direct", "cold_e2e_ms", long_), steady_val("pss_direct", "e2e_med")
        share = None if tw is None or ew in (None, 0) else round(100 * tw / ew)
        add("ICAM OAuth token", f"Token call before each search ({OAUTH_URL}, client credentials sent as {OAUTH_MODE['mode']}): "
            f"{fmt(tc)} after gaps over {IDLE_AFTER_S:g}s versus {fmt(tw)} warm. Token cold penalty: {fmt(tp_s)} for short gaps, {fmt(tp_l)} for long gaps. "
            f"Warm, the token call is {'n/a' if share is None else f'{share}%'} of the token-plus-search time. "
            f"Token plus search end to end: {fmt(ec)} cold after long gaps versus {fmt(ew)} warm. "
            f"This is the authentication line item the architecture diagram estimates at about 2,000 ms, measured without Apigee.")
if "apigee_internal" in active:
    l = med_where("apigee_internal", "e2e_penalty_ms", long_)
    p = med_where("pss_direct", "ttfb_penalty_ms", long_) if "pss_direct" in active else None
    a_c, a_w = med_where("apigee_internal", "cold_auth_ms", long_), steady_val("apigee_internal", "auth_total_med")
    ss = None if steady_val("apigee_internal", "total_med") is None or steady_val("pss_direct", "total_med") is None else steady_val("apigee_internal", "total_med") - steady_val("pss_direct", "total_med")
    add("Apigee internal", f"Median end-to-end cold penalty after gaps over {IDLE_AFTER_S:g}s: {fmt(l)}. "
        f"Above the PSS direct penalty by {fmt(None if l is None or p is None else l - p)}; a material value here answers yes to whether Apigee has its own cold start. "
        f"Steady-state proxy overhead (warm Apigee call minus warm PSS call, both on fresh connections): {fmt(ss)}.")
conn_txt = "; ".join(f"{r['layer']} {fmt(r['conn_med'])} (DNS {fmt(r['dns_med'])}, TCP {fmt(r['tcp_med'])}, TLS {fmt(r['tls_med'])})" for _, r in steady.iterrows()) if len(steady) else "n/a"
add("Connection setup", f"Fresh-connection cost per call, warm median: {conn_txt}. A caller that opens a new connection per request, as a cold Lambda does, pays this on every call.")
if len(clean):
    worst = clean.groupby("layer")["cold_e2e_ms"].max().to_dict()
    add("Worst observed", "Largest cold end-to-end time per layer: " + "; ".join(f"{k} {fmt(v)}" for k, v in worst.items())
        + ". Reported by the consuming team: about 8,000 ms; diagram estimate for ICAM plus Apigee: about 2,000 ms.")
if len(trials):
    add("Data quality", f"{int(trials['contaminated'].sum())} trials excluded because other searches hit the index during the idle gap; "
        f"{int((~trials['cold_ok']).sum())} cold calls failed. Coordinate so no load test runs on the same environment during this notebook.")
add("Not covered", "The consuming team's Lambda start-up, the external Apigee endpoint they call, and any production behavior are outside this run.")
findings = pd.DataFrame(findings)

print("\nFINDINGS")
for _, r in findings.iterrows():
    print(f"- [{r['area']}] {r['finding']}")
print("\nLAYER BY IDLE GAP (clean trials)")
show(by_gap)
print("\nATTRIBUTION BY IDLE GAP")
show(attr)
print("\nSTEADY STATE (warm calls)")
show(steady)
if len(settings_df):
    print("\nOPENSEARCH SETTINGS")
    show(settings_df)

runinfo = pd.DataFrame([{"field": k, "value": str(v)} for k, v in [
    ("run_start_local", RUN_START), ("run_end_local", datetime.now().isoformat()),
    ("layers_active", ", ".join(active)), ("layers_skipped", "; ".join(f"{a} ({b})" for a, b in skipped) or "none"),
    ("layers_failed_warmup", "; ".join(f"{a} ({b[:150]})" for a, b in layer_failed.items()) or "none"),
    ("gaps_s", GAPS_S), ("reps", REPS), ("warmup_calls", WARMUP_CALLS), ("warm_calls", WARM_CALLS),
    ("warm_spacing_s", WARM_SPACING_S), ("material_ms", MATERIAL_MS), ("client_id", CLIENT_ID),
    ("template_yaml", TEMPLATE_YAML), ("template_status", TEMPLATE["status"]), ("prod_logs_dir", PROD_LOGS_DIR),
    ("cases_available", len(cases)), ("timing_mode", "requests (proxy present)" if PROXIED else "socket phases"),
    ("ca_source", CA_SOURCE), ("opensearch_cluster", CLUSTER), ("search_idle_after_s", f"{IDLE_AFTER_S:g} ({IDLE_SRC})"),
    ("token_step", f"enabled, {OAUTH_URL}, client {OAUTH_CLIENT_ID}, credentials as {OAUTH_MODE['mode']}" if TOKEN_ENABLED else "not configured; Basic auth"), ("raw_csv", REUSE_RAW_CSV or RAW_CSV)]]
    + [{"field": f"url_{r['layer']}", "value": f"{r['url']} ({r['reason']})"} for r in env_rows])

out = os.path.join(RESULTS_DIR, f"ColdStart_layers_{RUN_TS}.xlsx")
drop_cols = [c for c in ("ok",) if c in cd]
with pd.ExcelWriter(out, engine="openpyxl") as xl:
    findings.to_excel(xl, sheet_name="Findings", index=False)
    by_gap.to_excel(xl, sheet_name="Layer by gap", index=False)
    attr.to_excel(xl, sheet_name="Attribution", index=False)
    steady.to_excel(xl, sheet_name="Steady state", index=False)
    trials.to_excel(xl, sheet_name="Trials", index=False)
    cd.drop(columns=drop_cols).to_excel(xl, sheet_name="Calls", index=False)
    settings_df.to_excel(xl, sheet_name="OpenSearch settings", index=False)
    runinfo.to_excel(xl, sheet_name="Run info", index=False)
print(f"\nExcel written: {out}")
print(f"Raw calls: {REUSE_RAW_CSV or RAW_CSV}")
