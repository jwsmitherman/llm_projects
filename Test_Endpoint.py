# Databricks notebook source
import requests, json, socket, base64
from urllib.parse import urlparse

OPENSEARCH_URL = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search"
OPENSEARCH_TOKEN = "PASTE_PROD_BASIC"

SERVICE_URL = "https://pcis-search-service-staging.apps.k8s.uscis.dhs.gov/search"
SERVICE_CLIENT_ID = "default"
OAUTH_URL = "https://oauth-preprod.uscis.dhs.gov/uaa/oauth/token"
OAUTH_ID = "svc-oauth-pcisui-preprod"
OAUTH_SECRET = "PASTE_CLIENT_SECRET"

TIMEOUT = 30

def line(): print("-" * 78)

def basic(t):
    t = (t or "").strip()
    if not t: return ""
    return t if t.startswith("Basic ") else "Basic " + t

def show_token(t):
    t = (t or "").strip()
    if not t: return "EMPTY"
    body = t[6:] if t.startswith("Basic ") else t
    try:
        decoded = base64.b64decode(body + "=" * (-len(body) % 4)).decode("utf-8", "replace")
        user = decoded.split(":")[0] if ":" in decoded else "no colon found"
        return f"{len(t)} chars, decodes to user '{user}'"
    except Exception:
        return f"{len(t)} chars, does not decode as base64, may not be a Basic token"

def reachable(url):
    host = urlparse(url).hostname
    port = urlparse(url).port or (443 if url.startswith("https") else 80)
    try:
        socket.gethostbyname(host)
    except Exception as e:
        return f"DNS lookup failed for {host}: {e}"
    try:
        s = socket.create_connection((host, port), timeout=10); s.close()
        return f"{host}:{port} reachable"
    except Exception as e:
        return f"cannot open a connection to {host}:{port}: {e}"

def explain(status, body):
    if status == 401: return "401 unauthorized, the credentials were rejected"
    if status == 403: return "403 forbidden, the credentials are valid but not permitted on this index"
    if status == 404: return "404 not found, the index or path in the URL does not exist"
    if status == 405: return "405 method not allowed, the URL may be missing /_search"
    if status == 429: return "429 too many requests, the cluster is rate limiting"
    if status and status >= 500: return f"{status} server error, the cluster returned a failure"
    if status == 200: return "200 OK"
    return f"status {status}"

print("OPENSEARCH DIRECT PATH")
line()
print("url  ", OPENSEARCH_URL)
print("token", show_token(OPENSEARCH_TOKEN))
print("net  ", reachable(OPENSEARCH_URL))
if OPENSEARCH_TOKEN.strip() and not OPENSEARCH_TOKEN.startswith("PASTE"):
    h = {"Content-Type": "application/json", "Authorization": basic(OPENSEARCH_TOKEN)}
    root = OPENSEARCH_URL.split("/_search")[0]
    for label, method, url, body in [
        ("cluster reachable", "GET", urlparse(OPENSEARCH_URL).scheme + "://" + urlparse(OPENSEARCH_URL).netloc, None),
        ("index exists", "GET", root, None),
        ("count documents", "POST", root + "/_count", {}),
        ("match all, 1 row", "POST", OPENSEARCH_URL, {"size": 1, "query": {"match_all": {}}}),
    ]:
        try:
            r = requests.request(method, url, headers=h, json=body, timeout=TIMEOUT)
            print(f"\n{label}: {explain(r.status_code, r.text)}")
            print("  ", url)
            print("  ", r.text[:400].replace("\n", " "))
        except Exception as e:
            print(f"\n{label}: call raised {type(e).__name__}: {e}")
            print("  ", url)
else:
    print("\ntoken not set, paste the value from Vault into OPENSEARCH_TOKEN above")

print("\n")
print("SEARCH SERVICE PATH")
line()
if not SERVICE_URL.strip():
    print("no service url set, skipping")
else:
    print("url  ", SERVICE_URL)
    print("net  ", reachable(SERVICE_URL))
    token = None
    if OAUTH_SECRET.strip() and not OAUTH_SECRET.startswith("PASTE"):
        try:
            r = requests.post(OAUTH_URL, data={"grant_type": "client_credentials"},
                              auth=(OAUTH_ID, OAUTH_SECRET),
                              headers={"Content-Type": "application/x-www-form-urlencoded"},
                              timeout=TIMEOUT)
            print(f"\noauth token: {explain(r.status_code, r.text)}")
            if r.status_code < 400:
                token = r.json().get("access_token")
                print("   token received, expires in", r.json().get("expires_in"), "seconds")
            else:
                print("  ", r.text[:400])
        except Exception as e:
            print(f"\noauth token: call raised {type(e).__name__}: {e}")
    else:
        print("\noauth secret not set, paste the value from Vault into OAUTH_SECRET above")
    if token:
        body = {"page": 0, "size": 1, "clientId": SERVICE_CLIENT_ID,
                "searchMethodType": "identifierSearch",
                "identifiers": [{"type": "RECEIPT_NBR", "value": "TEST0000000000"}]}
        try:
            r = requests.post(SERVICE_URL, headers={"Content-Type": "application/json",
                                                    "Authorization": "Bearer " + token},
                              json=body, timeout=TIMEOUT)
            print(f"\nservice search: {explain(r.status_code, r.text)}")
            print("  ", r.text[:400].replace("\n", " "))
            if r.status_code < 400:
                j = r.json()
                print("   clientId echoed back:", j.get("clientId", "NOT RETURNED"))
                for k in ("exactMatches", "similarMatches"):
                    print(f"   {k}: {len((j.get(k) or {}).get('content') or [])} returned, "
                          f"{(j.get(k) or {}).get('totalElements')} total")
        except Exception as e:
            print(f"\nservice search: call raised {type(e).__name__}: {e}")

print("\n")
print("WHAT TO CHECK")
line()
print("Token empty            paste the value from Vault into the variable at the top of this cell")
print("DNS lookup failed      the host name is wrong, or this cluster has no route to it")
print("Connection refused     the host is right but the port is closed, check firewall or VPN")
print("401                    the credentials are wrong or expired, pull a fresh value from Vault")
print("403                    the credentials work but have no access to this index")
print("404                    the index name in the url is wrong, check it against the alias")
print("Timeout                the cluster is unreachable from Databricks, raise with infrastructure")
