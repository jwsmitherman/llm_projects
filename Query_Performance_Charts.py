# Databricks notebook source
# # Identity Search Query Test: v6 vs v7 (Charts)
#
# ## What we tried
# - Ran the old query (v6) and the new query (v7) against the same index with the same search inputs.
# - Measured how fast each one is, how many records each one matches, and how the work spreads across shards.
#
# ## Why
# - The new search is slower than the old system and we need to know if the cause is the query or the servers.
# - Charts make it easy to show the team and to support the request for bigger staging servers.

# COMMAND ----------

# ## 1. Setup
#
# - Fill in the endpoint and token below, same as the test UI.
# - Point at staging when testing staging. Do not load production.

# COMMAND ----------

import requests, json, time, os, re, statistics
import pandas as pd
import matplotlib.pyplot as plt

ENDPOINT   = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search"
AUTH_TOKEN = "PASTE_BASE64_TOKEN_HERE"

auth_header = AUTH_TOKEN if AUTH_TOKEN.startswith("Basic ") else "Basic " + AUTH_TOKEN
HEADERS = {"Content-Type": "application/json", "Authorization": auth_header}
VERIFY_TLS = True

def os_search(query_body, profile=False):
    body = dict(query_body)
    if profile:
        body["profile"] = True
    t0 = time.perf_counter()
    resp = requests.post(ENDPOINT, headers=HEADERS, json=body, verify=VERIFY_TLS, timeout=120)
    client_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code >= 400:
        print("STATUS", resp.status_code, "BODY:", resp.text[:1500])
        resp.raise_for_status()
    return resp.json(), client_ms

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
print("Setup done. Endpoint:", ENDPOINT)

# COMMAND ----------

# ## 2. Load the queries
#
# - Files can be a template with placeholders or an already built request.
# - Both use the same inputs so the test is fair.

# COMMAND ----------

PARAMS = {
    "SIMILAR_SIZE": "25",
    "FIRSTNAME": "JOSE", "MIDDLENAME": "F", "LASTNAME": "MENDOZA",
    "ANUMBER": "", "DOB": "", "COB": "", "COC": "",
    "IDENTIFIER_NAME": "", "IDENTIFIER_VALUE": "",
}

PH = re.compile(r"\{\{\s*([A-Z_]+)\s*\}\}")

def _has_ph(n):
    if isinstance(n, str):  return bool(PH.search(n))
    if isinstance(n, list): return any(_has_ph(x) for x in n)
    if isinstance(n, dict): return any(_has_ph(v) for v in n.values())
    return False

def _empty_bool(n):
    if not isinstance(n, dict): return False
    b = n.get("bool")
    return isinstance(b, dict) and not any(isinstance(b.get(k), list) and b[k]
                                           for k in ("must", "should", "filter", "must_not"))

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

def render_template(text, params):
    s = PH.sub(lambda m: (str(params[m.group(1)])
                          if params.get(m.group(1)) not in (None, "") else m.group(0)), text)
    s = re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
    if s.lstrip()[:1] != "{": s = "{" + s + "}"
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    obj = json.loads(s)
    _prune(obj); _strip(obj)
    return obj

def load_query(path, params=PARAMS):
    if not os.path.exists(path):
        print("NOT FOUND:", path); return {}
    text = open(path).read()
    if "{{" not in text:
        try: return json.loads(text)
        except json.JSONDecodeError as e:
            print("Not valid JSON:", path, e); return {}
    try:
        q = render_template(text, params)
        print("Rendered:", path)
        return q
    except json.JSONDecodeError as e:
        print("Template is broken, cannot render:", path, e); return {}

V6_FILE = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-templatev6.txt"
V7_FILE = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-templatev7.txt"

V6_QUERY = load_query(V6_FILE)
V7_QUERY = load_query(V7_FILE)

def ready(q): return bool(q) and "query" in q
print("v6 ready:", ready(V6_QUERY), " v7 ready:", ready(V7_QUERY))

# COMMAND ----------

# ## 3. Collect the data
#
# - Runs each query once with profiling on, then repeats it to get a spread of response times.
# - Nothing is charted yet, this cell only gathers numbers.

# COMMAND ----------

RUNS = 20   # repeats per query for the latency spread

def profile_data(query):
    data, client_ms = os_search(query, profile=True)
    shards = data.get("profile", {}).get("shards", [])
    per_shard, comps = [], {}
    for sh in shards:
        total = 0
        for s in sh.get("searches", []):
            for q in s.get("query", []):
                total += q.get("time_in_nanos", 0)
                def walk(n):
                    if not n.get("children"):
                        t = n.get("type", "other")
                        comps[t] = comps.get(t, 0) + n.get("time_in_nanos", 0)
                    for c in n.get("children", []): walk(c)
                walk(q)
        per_shard.append(total / 1e6)
    return {
        "took": data.get("took", 0),
        "hits": data.get("hits", {}).get("total", {}).get("value", 0),
        "per_shard_ms": per_shard,
        "components": {k: v / 1e6 for k, v in comps.items()},
    }

def latency_runs(query, n=RUNS):
    out = []
    for _ in range(n):
        data, client_ms = os_search(query)
        out.append({"server_ms": data.get("took", 0), "client_ms": client_ms})
    return pd.DataFrame(out)

results = {}
for name, q in [("v6", V6_QUERY), ("v7", V7_QUERY)]:
    if not ready(q):
        print(name, "skipped, query not loaded"); continue
    results[name] = profile_data(q)
    results[name]["latency"] = latency_runs(q)
    print(name, "done. hits:", f"{results[name]['hits']:,}", " took:", results[name]["took"], "ms")

# COMMAND ----------

# ## 4. Summary table
#
# - One row per query version with the headline numbers.

# COMMAND ----------

def pct(s, p): return float(s.quantile(p / 100.0))

rows = []
for name, r in results.items():
    lat = r["latency"]["server_ms"]
    shards = r["per_shard_ms"]
    rows.append({
        "version": name,
        "hits": r["hits"],
        "hits_pct_of_index": round(100.0 * r["hits"] / 68_000_000, 1),
        "p50_ms": round(pct(lat, 50)),
        "p95_ms": round(pct(lat, 95)),
        "p99_ms": round(pct(lat, 99)),
        "slowest_shard_ms": round(max(shards), 1) if shards else None,
        "fastest_shard_ms": round(min(shards), 1) if shards else None,
        "shard_skew": round(max(shards) / min(shards), 2) if shards and min(shards) > 0 else None,
    })

summary = pd.DataFrame(rows)
display(summary)

# COMMAND ----------

# ## 5. Chart: how many records each query matches
#
# - Taller bar means the query is casting a wider net.
# - A very wide net means more work per search and slower responses.

# COMMAND ----------

if results:
    names = list(results.keys())
    hits = [results[n]["hits"] for n in names]

    fig, ax = plt.subplots()
    bars = ax.bar(names, hits, color=["#4C78A8", "#F58518"][:len(names)])
    ax.set_title("Records matched per query version")
    ax.set_ylabel("records matched")
    ax.set_xlabel("query version")
    for b, h in zip(bars, hits):
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:,.0f}", ha="center", va="bottom")
    ax.axhline(68_000_000, color="red", linestyle="--", linewidth=1)
    ax.text(0, 68_000_000, " whole index (68M)", color="red", va="bottom")
    plt.tight_layout(); plt.show()

# COMMAND ----------

# ## 6. Chart: response time spread
#
# - Bars show typical (p50) and worst case (p95, p99) response times.
# - Compare versions only when both match a similar number of records.

# COMMAND ----------

if results:
    names = list(results.keys())
    metrics = ["p50", "p95", "p99"]
    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    for i, n in enumerate(names):
        lat = results[n]["latency"]["server_ms"]
        vals = [pct(lat, 50), pct(lat, 95), pct(lat, 99)]
        pos = [p + (i - (len(names) - 1) / 2) * width for p in x]
        bars = ax.bar(pos, vals, width, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_title("Server response time: typical and worst case")
    ax.set_ylabel("milliseconds")
    ax.legend()
    plt.tight_layout(); plt.show()

# COMMAND ----------

# ## 7. Chart: time spent on each shard
#
# - Each bar is one shard doing part of the search at the same time.
# - The search is only as fast as the slowest bar, so a tall outlier holds everything up.

# COMMAND ----------

for name, r in results.items():
    shards = r["per_shard_ms"]
    if not shards: continue
    fig, ax = plt.subplots()
    colors = ["#E45756" if v == max(shards) else "#4C78A8" for v in shards]
    ax.bar(range(len(shards)), shards, color=colors)
    ax.axhline(sum(shards) / len(shards), color="green", linestyle="--", linewidth=1,
               label=f"average {sum(shards)/len(shards):.0f} ms")
    ax.set_title(f"{name}: time spent on each shard (slowest in red)")
    ax.set_xlabel("shard number"); ax.set_ylabel("milliseconds")
    ax.legend()
    plt.tight_layout(); plt.show()

# COMMAND ----------

# ## 8. Chart: what the query spends its time on
#
# - Shows which parts of the query cost the most time.
# - The biggest bar is where tuning would pay off most.

# COMMAND ----------

for name, r in results.items():
    comps = r["components"]
    if not comps: continue
    top = sorted(comps.items(), key=lambda kv: kv[1], reverse=True)[:6]
    labels = [k for k, v in top][::-1]
    vals   = [v for k, v in top][::-1]

    fig, ax = plt.subplots()
    bars = ax.barh(labels, vals, color="#72B7B2")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:,.0f}", va="center", fontsize=9)
    ax.set_title(f"{name}: time by query part (all shards added up)")
    ax.set_xlabel("milliseconds")
    plt.tight_layout(); plt.show()

# COMMAND ----------

# ## 9. Chart: response time over repeated runs
#
# - Flat lines mean steady performance, spikes mean the cluster is under pressure.
# - Use this to tell a real problem apart from a one off slow run.

# COMMAND ----------

if results:
    fig, ax = plt.subplots()
    for name, r in results.items():
        lat = r["latency"]["server_ms"]
        ax.plot(range(1, len(lat) + 1), lat, marker="o", label=name)
    ax.set_title("Response time across repeated runs")
    ax.set_xlabel("run number"); ax.set_ylabel("milliseconds")
    ax.legend()
    plt.tight_layout(); plt.show()

# COMMAND ----------

# ## 10. Shard data check
#
# - Confirms every shard holds about the same amount of data.
# - If doc counts are even, uneven timing is not a data problem.

# COMMAND ----------

BASE = ENDPOINT.split("/iis-identity-api-alias")[0]
INDEX = "iis-identity-api-alias"

try:
    r = requests.get(f"{BASE}/_cat/shards/{INDEX}?format=json&h=shard,prirep,state,docs,store,node",
                     headers=HEADERS, verify=VERIFY_TLS, timeout=30)
    if r.status_code >= 400:
        print("STATUS", r.status_code, r.text[:500])
    else:
        sh = pd.DataFrame(r.json())
        sh["docs"] = pd.to_numeric(sh["docs"], errors="coerce")
        primaries = sh[sh["prirep"] == "p"].sort_values("shard")
        display(primaries)

        fig, ax = plt.subplots()
        ax.bar(primaries["shard"].astype(str), primaries["docs"], color="#54A24B")
        ax.set_title("Documents per primary shard")
        ax.set_xlabel("shard number"); ax.set_ylabel("documents")
        plt.tight_layout(); plt.show()

        spread = 100.0 * (primaries["docs"].max() - primaries["docs"].min()) / primaries["docs"].mean()
        print(f"Total docs: {primaries['docs'].sum():,.0f}")
        print(f"Spread between biggest and smallest shard: {spread:.2f} percent")
except Exception as e:
    print("Could not read shard info:", e)

# COMMAND ----------

# ## 11. What the results mean
#
# - Fill this in after the run using the charts above.
# - Keep it to the three points that matter for the decision.

# COMMAND ----------

if results and len(results) == 2:
    a, b = list(results.keys())
    ha, hb = results[a]["hits"], results[b]["hits"]
    la = pct(results[a]["latency"]["server_ms"], 50)
    lb = pct(results[b]["latency"]["server_ms"], 50)

    print("FINDINGS")
    print(f"1. Records matched: {a} = {ha:,}   {b} = {hb:,}")
    if max(ha, hb) > 2 * min(ha, hb):
        wide = a if ha > hb else b
        print(f"   {wide} matches far more records, so the speed comparison is not fair yet.")
    else:
        print("   Both match a similar amount, so the speed comparison is fair.")
    print(f"2. Typical response time: {a} = {la:.0f} ms   {b} = {lb:.0f} ms")
    print(f"3. Shard spread: {a} = {rows[0]['shard_skew']}x   {b} = {rows[1]['shard_skew']}x")
    print("   A value near 1 is even. A high value means one shard is holding up the search.")

# COMMAND ----------

# ## 12. Next steps
#
# - Narrow the query that matches too many records, then run this notebook again.
# - Run against staging, not production, and repeat the test to rule out a one off slow run.
