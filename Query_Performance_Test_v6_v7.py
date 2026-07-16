# Databricks notebook source
# # Identity Search — Query Performance Test (v6 vs v7)
#
# **Goal (from the perf meetings):** figure out *why* the new search is CPU-bound and ~2-5x slower,
# and get hard numbers to compare **query v6** (older/shorter) vs **query v7** (current) and to justify
# the staging infrastructure changes.
#
# This notebook does three things against the cluster:
# 1. **Profiles** a single query (`profile: true`) and shows **per-shard time** + **which query
#    components eat the time** — the isolation test everyone asked for.
# 2. **Benchmarks** latency over N runs (p50 / p95 / p99), cache disabled, so the comparison is real.
# 3. **Compares v6 vs v7** side by side.
#
# ---
# ## What we already know from the two templates (static diff)
#
# | Construct | v6 (older) | v7 (current) | Why it matters for CPU |
# |---|---|---|---|
# | `prefix` clauses | **0** | **74** | Prefix queries scan the term dictionary for every term with that prefix — CPU-heavy. This is the single biggest change. |
# | `fuzziness` usages | ~21 | ~61 | Fuzzy queries expand each term into all terms within an edit distance (Levenshtein automaton) — expensive term expansion. |
# | `match` clauses | 19 | 86 | ~4.5x more scored clauses per query. |
# | `bool` blocks | 12 | 24 | ~2x the boolean nesting / clause evaluation. |
# | `generatedFullNames` | 7 | 24 | ~3.4x more cross-field full-name matching. |
# | `cross_fields` multi_match | 2 | 0 | v7 replaced the cross_fields approach with the prefix+fuzzy tier explosion. |
# | size (lines) | 810 | 1883 | v7 is ~3x larger; 22 tiers, boosts up to 3,000,000. |
#
# **Leading hypothesis:** v7's **74 prefix clauses + 3x fuzzy matching** are the CPU cost. The
# middle-initial query is slowest because middle-name tiers add another set of prefix/fuzzy expansions.
# Each query fans out to **21 shards on 3 data nodes (7 shards/node)**, so a CPU-heavy query runs 7
# shard-searches per node at once — that is how one query spikes a node to 90-98% CPU.
#
# **This notebook turns that hypothesis into measured numbers.** If the profile shows PrefixQuery /
# FuzzyQuery dominating per-shard time, the fix is the query (prune tiers / drop prefix clauses),
# not just bigger boxes.

# COMMAND ----------

# ## 1. Connection
#
# Endpoint + Basic token, same style as the HTML test UI. **Point `ENDPOINT` at the cluster you're
# testing** — use **staging** for the CPU investigation (the meetings were about staging), or prod to
# capture a baseline.
#
# > The token is hardcoded here for quick testing, like the UI. It's a real credential — don't commit
# > this notebook with it filled in; move it to `dbutils.secrets` (or rotate it) once you're done.

# COMMAND ----------

import requests, json, time, statistics
from concurrent.futures import ThreadPoolExecutor

# --- Fill these in (values from the KT doc; swap host for staging when testing staging) ---
ENDPOINT   = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search"
AUTH_TOKEN = "aW5kZXgtc2VydmljZS11c2VyOlFCNiVmRDQ3NXchMA=="   # base64("user:pass"), value after "Basic "

auth_header = AUTH_TOKEN if AUTH_TOKEN.startswith("Basic ") else "Basic " + AUTH_TOKEN
HEADERS = {"Content-Type": "application/json", "Authorization": auth_header}
VERIFY_TLS = True

# request_cache=false so we measure real work, not cache hits (matches how the team ran Gatling)
def os_search(query_body, profile=False, request_cache=False):
    url = ENDPOINT + ("?request_cache=false" if not request_cache else "")
    body = dict(query_body)
    if profile:
        body["profile"] = True
    t0 = time.perf_counter()
    resp = requests.post(url, headers=HEADERS, json=body, verify=VERIFY_TLS, timeout=120)
    client_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    return data, client_ms

print("Connection configured. Endpoint:", ENDPOINT)

# COMMAND ----------

# ## 2. Load the two queries from files
#
# Each query is read from a **text file containing a valid, final query body** (JSON) and parsed with
# `json.load`. Get a valid body from the **HTML test UI**: fill in the slow case (**middle-initial**,
# e.g. first=`JOSE`, middle=`F`, last=`MENDOZA`), copy the **"Final OpenSearch Request (Pretty-Printed)"**
# box, and save it to a file — one for v6, one for v7. Use the **same inputs** for both.
#
# > ⚠️ The raw `search_query_v6/v7.txt` **template** files won't work here — they're fragments with
# > `{{PLACEHOLDER}}` tokens and are not valid JSON on their own (they rely on the API's
# > substitute-and-prune step; v7 alone has ~20 unclosed braces). Always save the **rendered** request.
#
# Put the files anywhere the driver can read (DBFS `/dbfs/FileStore/...`, a workspace path, or `/tmp`).

# COMMAND ----------

import os

V6_QUERY_FILE = "/dbfs/FileStore/identity_perf/v6_query.txt"   # valid RENDERED request (not the template)
V7_QUERY_FILE = "/dbfs/FileStore/identity_perf/v7_query.txt"

def load_query(path):
    """Read a text file and parse it as a JSON query body. Tolerates a leading console verb line."""
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path} — save the rendered request there first.")
        return {}
    text = open(path).read()
    lines = text.strip().splitlines()
    if lines and lines[0].strip().upper().startswith(("GET ", "POST ")):  # strip Dev Tools verb line
        text = "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "{{" in text:
            print(f"  {path} looks like a raw TEMPLATE ({{{{...}}}} placeholders), not valid JSON.\n"
                  f"  Save the UI's 'Final OpenSearch Request' (rendered) instead.")
        else:
            print(f"  {path} is not valid JSON: {e}")
        return {}

V6_QUERY = load_query(V6_QUERY_FILE)
V7_QUERY = load_query(V7_QUERY_FILE)

def _looks_empty(q):
    return not q or "query" not in q
for name, q in [("v6", V6_QUERY), ("v7", V7_QUERY)]:
    status = "ready" if not _looks_empty(q) else "not loaded / no query block"
    print(f"{name}: {status}  ({len(json.dumps(q))} chars)")

# COMMAND ----------

# ## 3. Profile a single query (the isolation test)
#
# Run each query once with `profile: true` on an **idle-ish cluster** (no load test running). This
# answers the core question: *is one query expensive by itself, and which construct is the cost?*
#
# - **`per_shard_ms`** — time each of the 21 shards spent. Even spread = genuinely expensive query;
#   one hot shard = data skew.
# - **`top_components`** — the leaf query types (PrefixQuery, FuzzyQuery, TermQuery, …) ranked by time.
#   If PrefixQuery / FuzzyQuery dominate, that confirms the static-diff hypothesis.

# COMMAND ----------

def summarize_profile(resp, top_n=6):
    prof = resp.get("profile", {}); shards = prof.get("shards", [])
    per_shard, comp_times = [], {}
    for sh in shards:
        shard_total = 0
        for search in sh.get("searches", []):
            for q in search.get("query", []):
                shard_total += q.get("time_in_nanos", 0)
                def walk(node):
                    if not node.get("children"):                       # leaf = actual work unit
                        comp_times[node.get("type", "?")] = comp_times.get(node.get("type", "?"), 0) + node.get("time_in_nanos", 0)
                    for c in node.get("children", []):
                        walk(c)
                walk(q)
        per_shard.append(round(shard_total / 1e6, 3))
    top = sorted(({"type": k, "total_ms": round(v / 1e6, 3)} for k, v in comp_times.items()),
                 key=lambda x: x["total_ms"], reverse=True)[:top_n]
    out = {"num_shards": len(shards), "per_shard_ms": per_shard, "top_components": top}
    if per_shard:
        out["slowest_shard_ms"] = max(per_shard)
        out["shard_skew_ratio"] = round(max(per_shard) / max(min(per_shard), 0.001), 2)
    return out

def profile_query(name, query):
    if _looks_empty(query):
        print(f"{name}: not set — paste the rendered request in section 2."); return
    data, client_ms = os_search(query, profile=True)
    took = data.get("took", "?")
    hits = data.get("hits", {}).get("total", {})
    s = summarize_profile(data)
    print(f"=== {name} ===")
    print(f"  took (server): {took} ms   |  client round-trip: {client_ms:.0f} ms   |  hits: {hits}")
    print(f"  shards profiled: {s['num_shards']}   slowest shard: {s.get('slowest_shard_ms')} ms   skew: {s.get('shard_skew_ratio')}")
    print(f"  per-shard ms: {s['per_shard_ms']}")
    print(f"  top components (leaf query types by time):")
    for c in s["top_components"]:
        print(f"      {c['type']:<24} {c['total_ms']} ms")
    print()
    return s

profile_query("v6", V6_QUERY)
profile_query("v7", V7_QUERY)

# COMMAND ----------

# ## 4. Latency benchmark (p50 / p95 / p99)
#
# Fire each query N times sequentially with cache off, and report the distribution — not just the
# average, which hides the slow tail. Run on an idle cluster first for a clean per-query cost.

# COMMAND ----------

def pct(values, p):
    if not values: return None
    s = sorted(values); k = (len(s) - 1) * (p / 100.0); f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)

def benchmark(name, query, n=30, warmup=3):
    if _looks_empty(query):
        print(f"{name}: not set."); return
    for _ in range(warmup):                       # warm up (JIT, caches) but we still send cache=false
        os_search(query)
    server, client = [], []
    for _ in range(n):
        data, client_ms = os_search(query)
        server.append(data.get("took", 0)); client.append(client_ms)
    print(f"=== {name}  (n={n}) ===")
    for label, arr in [("server took", server), ("client round-trip", client)]:
        print(f"  {label:<18} p50={pct(arr,50):.0f}  p95={pct(arr,95):.0f}  p99={pct(arr,99):.0f}  "
              f"min={min(arr):.0f}  max={max(arr):.0f}  mean={statistics.mean(arr):.0f}  (ms)")
    print()
    return {"server": server, "client": client}

b6 = benchmark("v6", V6_QUERY, n=30)
b7 = benchmark("v7", V7_QUERY, n=30)

if b6 and b7:
    r = statistics.median(b7["server"]) / max(statistics.median(b6["server"]), 0.001)
    print(f">> v7 median server time is {r:.1f}x v6.  "
          f"(If ~1x, the query isn't the difference — look at infra. If >1.5x, v7's tiers are the cost.)")

# COMMAND ----------

# ## 5. Optional: light concurrency probe
#
# Sequential timing shows per-query cost. This fires a few queries at once to see how latency degrades
# under a little pressure — a small-scale echo of the Gatling finding. **Keep concurrency low on a
# shared cluster.** This is not a replacement for Gatling; it's a quick smell test.

# COMMAND ----------

def concurrency_probe(name, query, workers=5, total=30):
    if _looks_empty(query):
        print(f"{name}: not set."); return
    def one(_):
        _, c = os_search(query); return c
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lat = list(ex.map(one, range(total)))
    wall = time.perf_counter() - t0
    print(f"=== {name}  ({workers} concurrent, {total} total) ===")
    print(f"  wall: {wall:.1f}s   throughput: {total/wall:.1f} req/s   "
          f"client p50={pct(lat,50):.0f}  p95={pct(lat,95):.0f}  p99={pct(lat,99):.0f} ms")
    print()

# concurrency_probe("v7", V7_QUERY, workers=5, total=30)   # uncomment to run
print("Uncomment the line above to run a small concurrency probe. Start low (workers=5).")

# COMMAND ----------

# ## 6. How to read the results
#
# - **v7 slow when profiled alone, PrefixQuery/FuzzyQuery on top** → it's the **query**. The fix is
#   pruning tiers / dropping prefix clauses (and comparing v6's accuracy — the meeting's other action
#   item). Bigger boxes help, but you'd be paying to run an over-heavy query.
# - **v6 ≈ v7 alone, but both slow only under concurrency** → it's **capacity** (the 49-thread cap,
#   shards-per-node, CPU). That's the staging-upgrade case (data nodes / instance type).
# - **One shard much slower than the rest (high skew)** → data/shard imbalance, a separate issue.
# - **Bring to the deep-dive:** the per-shard profile for v6 and v7, the p95/p99 table, and the v7/v6
#   ratio. That single ratio tells the room how much of the slowdown is the query vs. the box.
#
# Reminder from the meetings: `master` nodes don't run searches — **data** nodes do. For a CPU-bound
# *query*, sizing/adding data nodes (and revisiting shard count, which needs a reindex) is the lever,
# not the master node.
