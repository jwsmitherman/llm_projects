# Databricks notebook source
# # OpenSearch, end to end — the PCIS Identity Search demo
#
# **Audience:** you, becoming the team's OpenSearch expert.
# **What this notebook does:** explains every core OpenSearch concept *in OpenSearch terms*, tied to
# the identity-search relevance solution you now own, and lets you run the mechanics yourself.
#
# It has three parts:
#
# | Part | What | Needs a cluster? |
# |---|---|---|
# | **1 — Concepts, made concrete** | cluster, node, index, **shard**, replica, document, field, mapping, analyzer, inverted index, BM25 scoring, **boosting**, the Query DSL — each explained and **simulated in pure Python** | No — runs on any Databricks cluster as-is |
# | **2 — The real solution** | your exact field model, the tiered relevance template + boost ladder, the **faithful pruning port**, then a **live** read-only connection to the cluster | Yes (Part 2 only) |
# | **3 — Expert runbook** | the mental model, the relevance-change workflow, the gotchas, the `_cat`/explain cheat-sheet | No |
#
# > **PII discipline:** the identity index holds real A-numbers, SSNs, DOBs, and names. Part 1 uses
# > **synthetic** data only. In Part 2, keep results in-memory for inspection — **do not** write real
# > identity records into DBFS, Delta tables, or notebook output you'll share. Pull auth from
# > `dbutils.secrets`, never hardcode it.

# COMMAND ----------

# # Part 1 — OpenSearch concepts, made concrete
#
# OpenSearch is a distributed search engine: you put JSON **documents** into an **index**, and it
# answers ranked queries over them fast. It's built on Apache Lucene and is the open-source fork of
# Elasticsearch 7.10. Everything below is a piece of that machine.
#
# ### The vocabulary, mapped to things you already know
#
# | OpenSearch term | What it is | Rough analogy |
# |---|---|---|
# | **Cluster** | One or more nodes working as a unit, sharing data | A database server / farm |
# | **Node** | A single OpenSearch process (usually one per VM) | A server instance |
# | **Index** | A named collection of documents you search over | A table |
# | **Document** | One JSON record — here, one identity | A row |
# | **Field** | A key inside a document | A column |
# | **Mapping** | The schema: each field's type + how it's analyzed | Table DDL |
# | **Analyzer** | Rules that turn text into searchable **tokens** | — (no SQL analog) |
# | **Shard** | A horizontal slice of an index; a self-contained Lucene index | A table partition |
# | **Replica** | A copy of a shard for resilience + read throughput | A read replica |
# | **`_score`** | How relevant a doc is to *this* query (BM25) | `ORDER BY relevance` |
#
# Run the cells in order. Part 1 has no external dependencies.

# COMMAND ----------

import math, hashlib, json, re
from collections import defaultdict
print("Part 1 uses only the Python standard library — no cluster, no pip install.")

# COMMAND ----------

# ## 1. Documents & fields — what an "identity" looks like
#
# A document is just JSON. Your identity documents are **nested** — names live under
# `biographicInfo.name`, search-optimized values under `_search`. A field is addressed by its dotted
# path, e.g. `biographicInfo.name.last` or `_search.identifiers.ALIEN_NBR`. That nesting is why the
# UI references fields the way it does.

# COMMAND ----------

sample_identity = {
    "biographicInfo": {
        "name":        {"first": "Jose", "middle": "Franklin", "last": "Mendoza"},
        "birthInfo":   {"country": {"isoCode": "SLV"}},
        "citizenship": {"country": {"isoCode": "SLV"}},
    },
    "_search": {
        "generatedFullNames": "Jose Mendoza | Jose Franklin Mendoza | Mendoza Jose",
        "dateOfBirth": "20060905",                         # YYYYMMDD, one token
        "identifiers": {"ALIEN_NBR": "A200000001", "SOCIAL_SECURITY_NBR": "900000001"},
    },
}

def get_field(doc, dotted_path):
    cur = doc
    for part in dotted_path.split("."):
        if not isinstance(cur, dict): return None
        cur = cur.get(part)
    return cur

for path in ["biographicInfo.name.last", "_search.identifiers.ALIEN_NBR", "_search.dateOfBirth"]:
    print(f"{path:40s} -> {get_field(sample_identity, path)!r}")

# COMMAND ----------

# ## 2. Mapping & analyzers — `text` vs `keyword` (the distinction that drives this whole project)
#
# When you index a field, the **analyzer** decides how its value becomes searchable **tokens**:
#
# - A **`text`** field is *analyzed*: lowercased and split into tokens. `"Mendoza Aguilar"` → `["mendoza", "aguilar"]`. Great for fuzzy / partial / full-text matching. Useless for exact matching.
# - A **`keyword`** field is *not analyzed*: the whole value stays one token. `"Mendoza Aguilar"` → `["Mendoza Aguilar"]`. Great for exact match, sorting, filtering, aggregations.
#
# Your name fields are mapped **both ways** — `biographicInfo.name.last` (text) *and*
# `biographicInfo.name.last.keyword` (keyword) — so each query clause can pick the right behavior.
# The top relevance tiers use `.keyword` for **exact** name matching; lower tiers use the analyzed
# `text` fields and `generatedFullNames` for fuzzy/partial recall.
#
# This is also exactly why **DOB as text is broken**: `"2011-01-02"` analyzed becomes `["2011","01","02"]`,
# so `"01"` and `"02"` match a huge fraction of records. Indexing DOB as a single `keyword` token
# `"20110102"` (with `fuzziness=1`) is the proposed fix.

# COMMAND ----------

def standard_analyzer(text):
    """Approximates OpenSearch's 'standard'/'text' analyzer: lowercase + split on non-alphanumerics."""
    cleaned = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in (text or ""))
    return [t for t in cleaned.split() if t]

def keyword_analyzer(text):
    """A 'keyword' field is stored verbatim as a single token."""
    return [text] if text else []

examples = ["Mendoza Aguilar", "Jose Franklin Mendoza", "2011-01-02"]
for ex in examples:
    print(f"{ex!r}")
    print(f"   text   -> {standard_analyzer(ex)}")
    print(f"   keyword-> {keyword_analyzer(ex)}")

# COMMAND ----------

# ## 3. The inverted index & BM25 — how `_score` is actually computed
#
# OpenSearch doesn't scan every document. For each **token**, it keeps a **posting list**: which
# documents contain it and how often. That inverted structure is what makes search fast.
#
# Relevance (`_score`) defaults to **BM25**, which rewards three things per query term:
#
# - **TF** (term frequency): more occurrences in a doc → higher, with diminishing returns.
# - **IDF** (inverse document frequency): rare terms across the corpus → worth more. (`"Mendoza"` discriminates more than a common token.)
# - **Field-length normalization**: a hit in a short field counts more than the same hit buried in a long field.
#
# `boost` is just a multiplier layered on top — that's the lever your tiers pull. The cell below
# builds a tiny inverted index over synthetic full names and computes BM25 by hand so you can see the
# ranking emerge.

# COMMAND ----------

class MiniField:
    """A single analyzed field with an inverted index + BM25 scoring (teaching-grade)."""
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.postings = defaultdict(dict)   # token -> {doc_id: term_freq}
        self.doc_len  = {}                   # doc_id -> length in tokens
        self.total_len = 0

    def add(self, doc_id, text):
        toks = self.analyzer(text)
        self.doc_len[doc_id] = len(toks)
        self.total_len += len(toks)
        tf = defaultdict(int)
        for t in toks: tf[t] += 1
        for t, f in tf.items(): self.postings[t][doc_id] = f

    @property
    def N(self): return len(self.doc_len)
    @property
    def avgdl(self): return self.total_len / self.N if self.N else 0.0

    def bm25(self, doc_id, query, k1=1.2, b=0.75):
        score = 0.0
        for qt in set(self.analyzer(query)):
            postings = self.postings.get(qt, {})
            n = len(postings)                       # docs containing this term
            if doc_id not in postings: continue
            idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            f   = postings[doc_id]
            dl  = self.doc_len[doc_id]
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / self.avgdl))
        return score

corpus = {
    0: "Jose Franklin Mendoza",
    1: "Jose Franklin Mendoza Aguilar",
    2: "Franklin Quiroz Mendoza",
    3: "Maria Elena Mendoza",
    4: "John Robert Smith",
}
field = MiniField(standard_analyzer)
for doc_id, name in corpus.items(): field.add(doc_id, name)

query = "Jose Mendoza"
ranked = sorted(corpus, key=lambda d: field.bm25(d, query), reverse=True)
print(f"Query: {query!r}   (avg field length = {field.avgdl:.2f} tokens)\n")
for d in ranked:
    s = field.bm25(d, query)
    bar = "#" * int(s * 20)
    print(f"  _score={s:5.3f} {bar:22s} doc {d}: {corpus[d]}")

print("\nNote: doc 4 scores 0 (no shared terms). Doc 0 beats doc 1 — same terms, but doc 0 is")
print("shorter, so field-length normalization ranks it higher. THIS is BM25 doing its job.")

# COMMAND ----------

# > **The score-comparability rule — internalize this.** `_score` is only meaningful **within one
# > result set**. A 1.08 here and a 1.08 from a different search mean nothing to each other: IDF
# > depends on the corpus and boosts compound differently per query. You compare *ranking* and
# > *relative* scores inside a single response — never thresholds across searches. This is the single
# > most common misunderstanding stakeholders will have.

# COMMAND ----------

# ## 3b. Boosting — the lever that turns BM25 into your ranking *policy*
#
# BM25 tells you how well text matches. **Boosting** is how you impose *business* priorities on top of
# that — it is a **multiplier** applied to a clause's (or a whole tier's) score contribution. Two kinds:
#
# - **Query-time boost** (what you use): put `"boost": N` on any clause or `bool` tier; it multiplies
#   that clause's score. Dynamic — you tune it per query without touching the index.
# - **Index-time boost** (avoid): baking a boost into the mapping. Deprecated and inflexible; ignore it.
#
# Your whole relevance design is built on query-time boosting. Three things to know cold:
#
# 1. **Boost is multiplicative, not additive.** A clause contributing BM25 `s` with `"boost": 3`
#    contributes `3 × s`.
# 2. **The 10000 / 1000 / 100 … ladder is spaced by orders of magnitude on purpose.** Within a tier,
#    BM25 only ranges ~0–3, so a single ×10 tier step dwarfs any within-tier BM25 difference. Net
#    effect: the **tier that fires sets the order of magnitude** of the score, and BM25 only orders
#    records *inside* the same tier. That is *why* "exact full name" always beats "fuzzy partial,"
#    regardless of text-match noise.
# 3. **The tiny `should` boosts (A-number `0.05`, DOB/COB/COC `0.02`) are deliberate tie-breakers.**
#    Against a 10000 tier they are rounding error — they only separate records that are otherwise equal
#    within a tier. That is the design intent from the relevance journey: identifiers refine, names rank.
#
# For needs beyond a multiplier — decay by recency, score from a numeric field, *demote* rather than
# exclude — OpenSearch has `function_score` and the `boosting` query. You don't use them today, but
# knowing they exist makes you the person who can answer "can we down-rank instead of filter it out?"

# COMMAND ----------

# Reuses `field`, `corpus`, and the analyzer from ?3. Boost is a multiplier on a clause's score.
query = "Jose Mendoza"

print("1) Per-clause boost multiplies the BM25 contribution:")
for boost in (1, 3, 10):
    contribs = {d: round(boost * field.bm25(d, query), 3) for d in corpus}
    print(f"   boost={boost:<3} -> {contribs}")

print("\n2) Why the ladder is spaced by orders of magnitude (10000/1000/100):")
raw = [field.bm25(d, query) for d in corpus]
lo, hi = min(r for r in raw if r > 0), max(raw)
print(f"   within-tier BM25 spans only ~{lo:.2f}..{hi:.2f}")
for boost in (10000, 1000, 100):
    print(f"   tier boost={boost:<6} -> {{",
          ", ".join(f'{d}:{round(boost*field.bm25(d,query),1)}' for d in corpus), "}")
print("   A single tier step (x10) dwarfs the whole within-tier BM25 swing, so the FIRING TIER")
print("   strictly dominates ranking and BM25 only breaks ties inside it.")

print("\n3) Identifier should-boosts are tie-breakers, not rank-changers:")
tier_contrib = 10000 * field.bm25(0, query)
anumber_bonus = 0.05 * 1.0
print(f"   exact-name tier ~{tier_contrib:.1f}  vs  A-number should-bonus ~{anumber_bonus}")
print(f"   ratio {anumber_bonus / tier_contrib:.1e}  ->  only separates otherwise-equal records.")

# COMMAND ----------

# ## 4. Shards & replicas — how OpenSearch scales (the part you asked about)
#
# An index is split into **primary shards**. Each shard is a *complete, self-contained Lucene index*
# holding a subset of the documents. Two reasons this exists:
#
# 1. **Capacity** — one index can hold more data than fits on a single node, by spreading shards across nodes.
# 2. **Parallelism** — a query runs on all shards *at the same time*, so latency stays low as data grows.
#
# **Which shard does a document go to?** OpenSearch routes deterministically:
#
# ```
# shard_number = hash(_routing) % number_of_primary_shards      # _routing defaults to the document _id
# ```
#
# (OpenSearch uses a Murmur3 hash; the cell below uses a stand-in hash to show the *principle*.)
#
# **Two consequences that matter operationally:**
#
# - The **primary shard count is fixed at index creation.** Changing it means **reindexing** into a new
#   index — because the modulo in that routing formula would otherwise send reads to the wrong shard.
#   (This is why the DOB-reindex and any mapping change are "make a new index" conversations.)
# - A **replica** is a full copy of a primary shard on a *different* node. Replicas give you
#   **high availability** (lose a node, a replica is promoted) and **extra read throughput** (searches
#   can hit replicas too). They do **not** increase write capacity.

# COMMAND ----------

def route(doc_id, num_primary_shards):
    """Illustrative routing: hash(_id) % primary_shard_count. Real OpenSearch uses Murmur3."""
    h = int(hashlib.md5(str(doc_id).encode()).hexdigest(), 16)
    return h % num_primary_shards

NUM_PRIMARY_SHARDS = 3
distribution = defaultdict(list)
for doc_id in range(15):
    distribution[route(doc_id, NUM_PRIMARY_SHARDS)].append(doc_id)

print(f"Routing 15 documents across {NUM_PRIMARY_SHARDS} primary shards:\n")
for shard in range(NUM_PRIMARY_SHARDS):
    docs = distribution[shard]
    print(f"  shard {shard}:  {len(docs)} docs  {docs}")

print(f"\nSame _id always routes to the same shard (deterministic): "
      f"doc 7 -> shard {route(7, NUM_PRIMARY_SHARDS)} every time.")
print("Change NUM_PRIMARY_SHARDS to 5 and re-run: the whole distribution shifts — which is exactly")
print("why you can't resize primaries in place; you reindex into a new index instead.")

# COMMAND ----------

# ### Scatter-gather: how a search uses all those shards
#
# A search you send doesn't go to one place. A **coordinating node**:
#
# 1. **Scatters** the query to every relevant shard (one primary or one of its replicas).
# 2. Each shard runs the query locally and returns its top-K hits **with local BM25 scores**.
# 3. The coordinator **gathers** and merges them into one globally-ranked result set, then fetches the
#    full documents for the winners.
#
# Because each shard scores against *its own* slice, this is another reason scores are a within-result
# construct, and why very small shards can produce slightly different IDF than one big shard would.

# COMMAND ----------

# A toy scatter-gather: spread the corpus across shards, score locally, merge globally.
shards = defaultdict(dict)
for doc_id, name in corpus.items():
    shards[route(doc_id, NUM_PRIMARY_SHARDS)][doc_id] = name

def shard_local_search(docs, query, top_k=10):
    f = MiniField(standard_analyzer)
    for d, name in docs.items(): f.add(d, name)
    scored = [(d, f.bm25(d, query)) for d in docs if f.bm25(d, query) > 0]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

query = "Jose Mendoza"
gathered = []
for shard, docs in shards.items():
    hits = shard_local_search(docs, query)
    print(f"shard {shard} local hits: {[(d, round(s,3)) for d,s in hits]}")
    gathered.extend(hits)

merged = sorted(gathered, key=lambda x: x[1], reverse=True)
print(f"\nGathered & globally ranked for {query!r}:")
for d, s in merged:
    print(f"  _score={s:5.3f}  doc {d}: {corpus[d]}")

# COMMAND ----------

# ## 5. Nodes & cluster roles (one-screen version)
#
# A node can wear several hats; in a managed cluster these are assigned for you:
#
# - **Cluster-manager** (formerly "master"): tracks cluster state — which shards live where, mapping changes. One active at a time, elected.
# - **Data node**: holds shards, does the indexing and the per-shard search work.
# - **Ingest node**: runs pre-index transformation pipelines.
# - **Coordinating node**: the scatter-gather router from §4. *Every* node can coordinate.
#
# When you hit the search endpoint, whichever node receives it coordinates the request. You rarely
# think about roles day to day, but you'll see them in `_cat/nodes` (Part 2) and they explain phrases
# like "the cluster-manager is doing a shard reallocation."

# COMMAND ----------

# ## 6. The Query DSL — the grammar your template is written in
#
# Searches are JSON trees. The container is **`bool`**, which combines clause lists:
#
# | Clause | Meaning | Affects score? |
# |---|---|---|
# | **`must`** | all must match (AND) | yes |
# | **`should`** | optional; matching adds to score (OR-ish) | yes |
# | **`filter`** | must match, but yes/no only | **no** (just includes/excludes) |
# | **`must_not`** | must not match (exclude) | no |
# | **`minimum_should_match`** | how many `should` clauses are required | — |
#
# Leaf clauses you'll see throughout the template:
#
# - **`term`** — exact match on a single token; used on `.keyword` fields (exact name, ISO codes, DOB).
# - **`match`** — analyzed match on a `text` field; supports `fuzziness` (e.g. A-number `fuzziness: 2`) and `operator`.
# - **`multi_match`** with `type: "cross_fields"` — treats several fields as one combined field, so
#   `"Jose Franklin Mendoza"` can match with first/middle/last spread across `generatedFullNames`,
#   `name.first`, etc., each with its own `^boost` weight.
# - **`boost`** — the per-clause/per-tier multiplier that encodes your ranking priorities.
#
# The cell shows a single tier as a Python dict so the shape is concrete. Part 2 assembles the full
# stack of tiers.

# COMMAND ----------

one_tier = {
    "bool": {
        "boost": 10000,                                  # this tier dominates when it fires
        "must": [                                        # exact full-name match (keyword fields)
            {"term": {"biographicInfo.name.last.keyword":   "{{LASTNAME}}"}},
            {"term": {"biographicInfo.name.first.keyword":  "{{FIRSTNAME}}"}},
            {"term": {"biographicInfo.name.middle.keyword": "{{MIDDLENAME}}"}},
        ],
        "should": [                                      # identifiers nudge ties, tiny boosts
            {"match": {"_search.identifiers.ALIEN_NBR": {"query": "{{ANUMBER}}", "fuzziness": 2, "boost": 0.05}}},
            {"term":  {"_search.dateOfBirth":           {"value": "{{DOB}}",     "boost": 0.02}}},
        ],
    }
}
print(json.dumps(one_tier, indent=2))

# COMMAND ----------

# # Part 2 — The real identity relevance solution
#
# Now we connect the concepts to **your** artifacts: the field model, the tiered template, the pruning
# algorithm from the UI, and a live read-only look at the cluster.

# COMMAND ----------

# ## 7. The field model (your exact fields)
#
# These are the dotted paths the template and UI actually use:
#
# | Purpose | Field | Type used |
# |---|---|---|
# | First / middle / last (exact) | `biographicInfo.name.{first,middle,last}.keyword` | keyword |
# | First / middle / last (fuzzy) | `biographicInfo.name.{first,middle,last}` | text |
# | Full-name permutations | `_search.generatedFullNames` | text |
# | A-number | `_search.identifiers.ALIEN_NBR` | (match, `fuzziness: 2`) |
# | Date of birth | `_search.dateOfBirth` | keyword (`YYYYMMDD`) |
# | Country of birth (COB) | `biographicInfo.birthInfo.country.isoCode` | keyword |
# | Citizenship country (COC) | `biographicInfo.citizenship.country.isoCode` | keyword |
#
# Template placeholders: `{{FIRSTNAME}} {{MIDDLENAME}} {{LASTNAME}} {{ANUMBER}} {{DOB}} {{COB}} {{COC}}`.

# COMMAND ----------

# ## 8. The tiered template & boost ladder
#
# The production query is one big `bool` with `minimum_should_match: 1` and a **stack of tiers** in its
# `should`, ordered from most-specific (huge boost) to least-specific (small boost). Representative
# ladder from your default UI template:
#
# | Tier (most → least specific) | `must` (the defining combination) | boost |
# |---|---|---|
# | Exact first + middle + last | `name.{first,middle,last}.keyword` | **10000** |
# | Exact first + last | `name.{first,last}.keyword` | 1000 |
# | Exact first + last + full-name cross-field | + `multi_match` on `generatedFullNames` | 1000 |
# | Full-name cross-field only | `multi_match` cross_fields | ~100–500 |
# | …partial / fuzzy fallbacks… | one analyzed name field | 20–100 |
#
# Every tier also carries small `should` boosts (A-number `0.05`, DOB/COB/COC `0.02`) that act as
# **tie-breakers** within a tier. So ranking is dominated by *which tier fires*, and fine-tuned by
# *which identifiers also matched* — the design intent from the relevance journey.
#
# > ⚠️ **Source-of-truth gotcha.** The default template embedded in the HTML UI is **stale and not
# > even valid JSON** (it has unbalanced braces; `JSON.parse` would throw on a fresh load). The UI runs
# > off whatever's saved in `localStorage`, and the real template lives in the **API repo**. Per Chris:
# > **always pull the latest template from the API** and iterate from that — never from the HTML default.

# COMMAND ----------

# ## 9. The clause-pruning algorithm — faithful Python port of the UI
#
# Searches rarely supply every field, so before running, the UI **prunes** the template. Ported
# exactly from the JavaScript in `simple_identity_search_ui_7.html`:
#
# 1. Substitute every **non-blank** input: `{{LASTNAME}}` → `Mendoza`. Blank inputs are left as-is.
# 2. **Prune** `must`/`should` arrays: drop any child that **still contains a `{{...}}` placeholder** or
#    is an **empty bool**. If an array empties out, delete it.
# 3. **Strip** any remaining placeholders from leftover strings.
#
# **Two behaviors worth knowing cold — you'll be asked about both:**
#
# - Pruning is **leaf-level, not tier-level.** Blank middle name removes just the middle `term` from a
#   tier's `must`; the tier survives on its remaining `must` clauses. (This differs from a stricter
#   "drop the whole tier" rule — the production UI does *not* do that.)
# - Because pruning runs **before** stripping, any clause that bundles names —
#   `"{{FIRSTNAME}} {{MIDDLENAME}} {{LASTNAME}}"` — is **dropped entirely when middle name is blank**,
#   since the leftover `{{MIDDLENAME}}` makes the whole clause look unfilled. **Net effect: the
#   `generatedFullNames` cross-field tiers silently disappear for any search without a middle name.**
#   That's a real relevancy footgun to keep in mind when triaging "why didn't this match" tickets.

# COMMAND ----------

PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")

def contains_placeholder(node):
    if node is None: return False
    if isinstance(node, str):  return bool(PLACEHOLDER_RE.search(node))
    if isinstance(node, list): return any(contains_placeholder(x) for x in node)
    if isinstance(node, dict): return any(contains_placeholder(v) for v in node.values())
    return False

def is_empty_bool(node):
    if not isinstance(node, dict): return False
    b = node.get("bool")
    if not isinstance(b, dict): return False
    return not any(isinstance(b.get(k), list) and b[k] for k in ("must", "should", "filter", "must_not"))

def prune_bool_clauses(node):
    if isinstance(node, dict) and isinstance(node.get("bool"), dict):
        b = node["bool"]
        for key in ("must", "should"):
            if not isinstance(b.get(key), list): continue
            for child in b[key]:
                prune_bool_clauses(child)
            b[key] = [c for c in b[key] if not contains_placeholder(c) and not is_empty_bool(c)]
            if not b[key]:
                del b[key]
    for v in (node.values() if isinstance(node, dict) else node if isinstance(node, list) else []):
        if isinstance(v, (dict, list)):
            prune_bool_clauses(v)

def strip_remaining_placeholders(node):
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, str): node[k] = PLACEHOLDER_RE.sub("", v)
            elif isinstance(v, (dict, list)): strip_remaining_placeholders(v)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str): node[i] = PLACEHOLDER_RE.sub("", v)
            elif isinstance(v, (dict, list)): strip_remaining_placeholders(v)

def add_param_if_not_blank(params, key, value):
    if value is not None and str(value).strip() != "":
        params[key] = str(value)

def build_final_query(template_text, params):
    """Mirrors buildFinalQuery() in the UI: substitute -> prune -> strip."""
    txt = json.dumps(json.loads(template_text))
    for key, value in params.items():
        txt = txt.replace("{{" + key + "}}", str(value))
    obj = json.loads(txt)
    prune_bool_clauses(obj)
    strip_remaining_placeholders(obj)
    return obj

print("Pruning port loaded: build_final_query(template_text, params)")

# COMMAND ----------

# A compact, VALID template modeled on the real boost ladder (the real one lives in the API repo).
DEMO_TEMPLATE = json.dumps({
  "size": 10,
  "query": {"bool": {"minimum_should_match": 1, "should": [
    {"bool": {"_name": "t1_exact_first_middle_last", "boost": 10000,
      "must": [
        {"term": {"biographicInfo.name.last.keyword":   "{{LASTNAME}}"}},
        {"term": {"biographicInfo.name.first.keyword":  "{{FIRSTNAME}}"}},
        {"term": {"biographicInfo.name.middle.keyword": "{{MIDDLENAME}}"}}],
      "should": [
        {"match": {"_search.identifiers.ALIEN_NBR": {"query": "{{ANUMBER}}", "fuzziness": 2, "boost": 0.05}}},
        {"term":  {"_search.dateOfBirth":           {"value": "{{DOB}}", "boost": 0.02}}}]}},
    {"bool": {"_name": "t2_exact_first_last", "boost": 1000,
      "must": [
        {"term": {"biographicInfo.name.last.keyword":  "{{LASTNAME}}"}},
        {"term": {"biographicInfo.name.first.keyword": "{{FIRSTNAME}}"}}],
      "should": [
        {"term": {"biographicInfo.name.middle.keyword": "{{MIDDLENAME}}"}},
        {"term": {"_search.dateOfBirth": {"value": "{{DOB}}", "boost": 0.02}}}]}},
    {"bool": {"_name": "t3_fullname_crossfields", "boost": 100,
      "must": [
        {"multi_match": {"query": "{{FIRSTNAME}} {{MIDDLENAME}} {{LASTNAME}}",
          "fields": ["_search.generatedFullNames^10", "biographicInfo.name.last^6",
                     "biographicInfo.name.first^4", "biographicInfo.name.middle^2"],
          "type": "cross_fields", "operator": "and"}}]}}
  ]}}}
)

def surviving_tiers(q):
    return [(t["bool"].get("_name"), t["bool"].get("boost")) for t in q["query"]["bool"]["should"]]

print("=== Full input (first + middle + last + A-number + DOB) ===")
p = {}
add_param_if_not_blank(p, "FIRSTNAME", "Jose"); add_param_if_not_blank(p, "MIDDLENAME", "Franklin")
add_param_if_not_blank(p, "LASTNAME", "Mendoza"); add_param_if_not_blank(p, "ANUMBER", "A200000001")
add_param_if_not_blank(p, "DOB", "20060905")
for name, boost in surviving_tiers(build_final_query(DEMO_TEMPLATE, p)):
    print(f"   tier {name:32s} boost={boost}")

print("\n=== No middle name (watch the cross-field tier disappear) ===")
p = {}
add_param_if_not_blank(p, "FIRSTNAME", "Jose"); add_param_if_not_blank(p, "LASTNAME", "Mendoza")
add_param_if_not_blank(p, "ANUMBER", "A200000001")
for name, boost in surviving_tiers(build_final_query(DEMO_TEMPLATE, p)):
    print(f"   tier {name:32s} boost={boost}")
print("   -> t3_fullname_crossfields is GONE: blank {{MIDDLENAME}} pruned the whole multi_match clause.")

# COMMAND ----------

# ## 10. Connect to the live cluster (read-only)
#
# Now that you have access. **Auth comes from a Databricks secret scope — never hardcode it.** Set one
# up once (CLI): `databricks secrets create-scope identity-search`, then put the basic-auth token
# (or username/password) in it.
#
# The basic-auth token is just `base64("username:password")` — the helper below builds one if you only
# have a username/password (matching the KT discussion).

# COMMAND ----------

%pip install opensearch-py

# COMMAND ----------

import base64
from opensearchpy import OpenSearch

def basic_token(username, password):
    """KT note: a basic-auth token is base64(user:pass). Use this if Vault gives you user/pass."""
    return base64.b64encode(f"{username}:{password}".encode()).decode()

# --- Pull secrets; do NOT inline credentials ---
# SCOPE = "identity-search"
# OS_HOST  = dbutils.secrets.get(SCOPE, "opensearch_host")     # e.g. opensearch-identity-prod...dhs.gov
# OS_USER  = dbutils.secrets.get(SCOPE, "opensearch_user")
# OS_PASS  = dbutils.secrets.get(SCOPE, "opensearch_pass")
# INDEX    = dbutils.secrets.get(SCOPE, "opensearch_index")    # e.g. identity

# client = OpenSearch(
#     hosts=[{"host": OS_HOST, "port": 443}],
#     http_auth=(OS_USER, OS_PASS),
#     use_ssl=True, verify_certs=True,
#     http_compress=True,
# )
# print(client.info()["version"])

print("Fill in the secret-scope lines above, then uncomment. Left commented so the notebook")
print("runs end-to-end in Part 1 without credentials.")

# COMMAND ----------

# ## 11. See the real shards, nodes, and mapping
#
# These `_cat` and mapping calls are the fastest way to *see* the concepts from Part 1 on the actual
# cluster. All read-only.

# COMMAND ----------

# --- Uncomment once `client` and `INDEX` are set ---

# # Cluster & node health (roles from Part 1 ?5 show up here)
# print(client.cat.health(v=True))
# print(client.cat.nodes(v=True, h="name,node.role,master,heap.percent,cpu,disk.used_percent"))

# # The shards of YOUR index: primaries (p) vs replicas (r), which node, doc count, size
# print(client.cat.shards(index=INDEX, v=True, h="index,shard,prirep,state,docs,store,node"))

# # How many primary shards does this index have? (fixed at creation — see ?4)
# settings = client.indices.get_settings(index=INDEX)
# print(json.dumps(settings, indent=2)[:1500])

# # The mapping: confirm which fields are text vs keyword (the ?2 distinction, live)
# mapping = client.indices.get_mapping(index=INDEX)
# print(json.dumps(mapping, indent=2)[:2000])

# COMMAND ----------

# ## 12. Run the real relevance query against the index
#
# Build the final query with the pruning port, then execute it. Adding `_name` to each tier (your
# template already does) makes OpenSearch return **`matched_queries`** per hit — so you can see exactly
# *which tier fired* for each result. That's your #1 debugging tool for relevance.
#
# > Use **test inputs** here. Inspect results in-memory; don't persist real identity records.

# COMMAND ----------

# --- Uncomment once connected. Replace DEMO_TEMPLATE with the LATEST template pulled from the API repo. ---

# test_inputs = {}
# add_param_if_not_blank(test_inputs, "FIRSTNAME", "Jose")
# add_param_if_not_blank(test_inputs, "MIDDLENAME", "Franklin")
# add_param_if_not_blank(test_inputs, "LASTNAME",  "Mendoza")
# add_param_if_not_blank(test_inputs, "DOB",       "20060905")

# final_query = build_final_query(API_TEMPLATE, test_inputs)   # API_TEMPLATE = current template text
# resp = client.search(index=INDEX, body=final_query)

# print(f"hits: {resp['hits']['total']['value']}  max_score: {resp['hits']['max_score']}")
# for h in resp["hits"]["hits"][:10]:
#     n = get_field(h["_source"], "biographicInfo.name") or {}
#     name = f"{n.get('first','')} {n.get('middle','')} {n.get('last','')}".strip()
#     print(f"  _score={h['_score']:>10.2f}  tiers={h.get('matched_queries', [])}  {name}")

# COMMAND ----------

# ## 13. Expert tools: explain, profile, validate, analyze
#
# Four APIs that turn "the ranking looks wrong" into a precise answer:
#
# - **`_explain`** — for *one* document + query, a full breakdown of how its `_score` was built (which clauses, TF/IDF/boost). Use it to answer "why did this record rank here?"
# - **`profile: true`** — add to a search body to get per-shard timing and the query tree that actually executed (after your pruning). Confirms which tiers really ran.
# - **`_validate/query?explain=true`** — checks a query is well-formed and shows the rewritten Lucene form, without running it. Great after editing the template.
# - **`_analyze`** — feed it text + an analyzer and see the exact tokens produced. The live version of Part 1 ?2; use it to debug "why didn't this name match?"

# COMMAND ----------

# --- Handy, read-only. Uncomment once connected. ---

# # Why did one specific doc score what it scored?
# print(json.dumps(client.explain(index=INDEX, id="<doc_id>", body=final_query), indent=2)[:2500])

# # What tokens does the analyzer actually produce for a value?
# print(client.indices.analyze(index=INDEX, body={"field": "biographicInfo.name.last", "text": "Mendoza Aguilar"}))

# # Profile a search (per-shard timing + executed query tree)
# profiled = client.search(index=INDEX, body={**final_query, "profile": True})
# print(list(profiled.get("profile", {}).keys()))

# COMMAND ----------

# # Part 3 — Expert runbook
#
# ### The mental model, in one paragraph
# Identity search is one big `bool` query made of **tiers** ordered by specificity. The tier that fires
# sets the order of magnitude of the score (exact full name ≫ exact first+last ≫ fuzzy/partial), and
# small per-field `should` boosts (A-number, DOB, COB, COC) break ties within a tier. A **template**
# with `{{placeholders}}` is **pruned** at query time so only clauses whose inputs were supplied
# survive. Scores rank within a single result set and nowhere else.
#
# ### The relevance-change workflow (don't deviate)
# 1. **Pull the latest template from the API repo** — never the HTML default (it's stale + invalid JSON).
# 2. Edit the template (or have DHSChat edit it from the natural-language rules + index mapping).
# 3. Test in the UI / this notebook with representative inputs; eyeball `matched_queries` and `_score`.
# 4. Run your **regression test cases** so you don't fix one edge case while degrading a more important
#    search ("whack-a-mole").
# 5. Hand the finished template back to the API team to drop in.
#
# ### Questions you'll get, and the crisp answers
# - *"Can we set a score cutoff?"* → No across searches; `_score` is within-result only.
# - *"Why didn't record X match?"* → `_analyze` the field + `_explain` the doc; usually a text-vs-keyword
#   or a pruned-clause issue.
# - *"Why did a no-middle-name search miss the full-name tier?"* → The `{{MIDDLENAME}}` placeholder
#   pruned that bundled clause (the §9 gotcha).
# - *"Can we add a shard / change the field type?"* → Not in place; that's a reindex into a new index.
# - *"How do we prove it's better?"* → The batch old-vs-new comparison (separate workstream).
#
# ### The gotchas, collected
# - HTML default template = stale & invalid JSON → source of truth is the API repo.
# - Pruning is leaf-level; bundled-name clauses vanish without a middle name.
# - DOB-as-text tokenizes into year/month/day → matches ~everything; keyword `YYYYMMDD` + `fuzziness:1` is the fix (needs reindex).
# - Primary shard count and field types are fixed at index creation.
# - A-number uses `fuzziness: 2` — generous; worth reviewing if false positives surface.
#
# ### Where to go deeper
# Query DSL, analyzers, and the `_cat`/explain APIs: <https://docs.opensearch.org/latest/>
