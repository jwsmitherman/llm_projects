# Databricks notebook source
# =============================================================================
# PCIS Identity Search - Performance and Stress Test Harness
# =============================================================================
#
# PURPOSE
#
# Characterize the behavior of the identity search stack under load, isolate
# which layer degrades first, and produce a comparable result set that can be
# re-run on every promotion or code change.
#
# This harness is built to answer the specific questions raised in the
# performance testing and stress test sessions:
#
#   1. What is the behavior when the stack is pushed past its limit -
#      timeouts, 4xx, 5xx, connection resets, or silent degradation.
#   2. When the load is removed, does the stack recover gracefully, and
#      how long does recovery take.
#   3. Which layer is the actual constraint - OpenSearch, the PSS pods and
#      their Kubernetes liveness probe, or the Apigee proxy quota.
#   4. How does the front door (Apigee) compare to the direct OpenSearch
#      single-search endpoint, since the front door is what consumers use.
#   5. How does a given run compare to the prior run, so that each
#      promotional layer and code change can be monitored for drift.
#
# WHAT THIS ADDS OVER THE EXISTING GATLING RUNS
#
#   - Server-side search time (OpenSearch "took") is captured alongside
#     client-observed latency. The gap between the two is queue time, pod
#     time, and proxy time. This separates "search is slow" from
#     "something in front of search is slow", which was the open question
#     when the pods bounced at roughly 55 requests per second.
#   - OpenSearch thread pool rejection counters are sampled during the run.
#     A rising search queue with rejections is the definitive signal that
#     OpenSearch itself is the constraint. An absence of rejections while
#     pods restart points at the liveness probe instead.
#   - Errors are classified into a taxonomy rather than counted as one
#     bucket, so connection refused (pod restart) is distinguished from
#     429 (proxy quota) and from 504 (gateway timeout).
#   - Queries are drawn from production audit logs rather than a single
#     repeated payload, so the load profile reflects the real mix of name
#     only, name plus date of birth, and identifier searches.
#   - A dedicated recovery phase measures time to return to baseline after
#     the overload is lifted.
#   - Every run writes a summary record and is compared against the stored
#     baseline, producing a pass or fail verdict suitable for a pipeline.
#
# SCOPE AND HONEST LIMITS
#
#   - This harness is a characterization and regression tool. It is not a
#     replacement for Gatling or Artillery at very high offered load. A
#     single driver node is limited by client-side threads and sockets.
#     Set DISTRIBUTED_LOAD to True to spread generation across the cluster
#     when higher offered rates are needed.
#   - Offered rate and achieved rate are reported separately. When the
#     client cannot sustain the offered rate, the achieved rate is the
#     number that matters and the shortfall is reported explicitly.
#   - Reference figures recorded below come from prior sessions and are
#     used as context only, not as assertions about current behavior.
#
# =============================================================================

# COMMAND ----------

# =============================================================================
# CELL 1 - CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Run identity. RUN_LABEL is what shows up in the comparison output, so it
# should name the thing being tested, not the date.
# -----------------------------------------------------------------------------
RUN_LABEL = "staging_v7_baseline"
RUN_NOTES = "Baseline characterization run against staging."

# -----------------------------------------------------------------------------
# Environment. ENV must be set explicitly. There is no default, because an
# accidental production stress run is the one outcome this harness must not
# produce.
# -----------------------------------------------------------------------------
ENV = "staging"                      # "staging" or "prod"

# Production overload guard. Phases that intentionally push past the limit
# are blocked in production unless this is deliberately set to True.
ALLOW_PROD_OVERLOAD = False

# -----------------------------------------------------------------------------
# Targets. Two paths are exercised so the front door and the direct search
# endpoint can be compared on the same query pool in the same window.
#
#   direct  - the OpenSearch single-search endpoint. This is what the
#             existing Gatling runs hit. It excludes the proxy and the
#             search service.
#   gateway - the Apigee proxy in front of the search service. This is the
#             path consumers actually use, and the only path where proxy
#             quota can appear.
#
# Fill in the host values for the environment being tested. Leaving a target
# blank disables that path rather than failing the run.
# -----------------------------------------------------------------------------
TARGETS = {
    "staging": {
        "direct": {
            "enabled": True,
            "url": "https://<staging-opensearch-host>/<index>/_search",
            "auth_mode": "basic",
        },
        "gateway": {
            "enabled": False,
            "url": "https://<staging-apigee-host>/<basepath>/search",
            "auth_mode": "bearer",
        },
    },
    "prod": {
        "direct": {
            "enabled": True,
            "url": "https://<prod-opensearch-host>/<index>/_search",
            "auth_mode": "basic",
        },
        "gateway": {
            "enabled": False,
            "url": "https://<prod-apigee-host>/<basepath>/search",
            "auth_mode": "bearer",
        },
    },
}

# Cluster telemetry endpoint. Used for thread pool and health sampling.
# Leave blank to skip telemetry sampling.
CLUSTER_BASE_URL = {
    "staging": "https://<staging-opensearch-host>",
    "prod": "https://<prod-opensearch-host>",
}

# -----------------------------------------------------------------------------
# Credentials. Same pattern as the existing test notebooks - a token pasted
# in for the duration of the run. The 'Basic ' prefix is added if absent.
# -----------------------------------------------------------------------------
AUTH_TOKEN = ""                      # base64 of user:pass for the direct path
GATEWAY_TOKEN = ""                   # bearer token or key for the Apigee path

# -----------------------------------------------------------------------------
# Query template. The source of truth is the latest template in the API repo.
# The template embedded in the HTML test UI is stale and is not used.
# -----------------------------------------------------------------------------
TEMPLATE_PATH = "/dbfs/FileStore/pcis/search-template-v7.txt"

# Optional second template for side by side comparison under load. Leaving
# this blank runs a single template.
TEMPLATE_PATH_B = ""                 # for example the v7-improved template
TEMPLATE_LABEL_A = "v7"
TEMPLATE_LABEL_B = "v7_improved"

# -----------------------------------------------------------------------------
# Query pool source. Production audit logs give a realistic mix of search
# shapes. A synthetic pool is generated only if the log table is unavailable,
# and the run is flagged accordingly.
# -----------------------------------------------------------------------------
AUDIT_LOG_TABLE = "<catalog>.<schema>.<audit_log_table>"
AUDIT_LOG_LOOKBACK_DAYS = 7
QUERY_POOL_SIZE = 2000

# PHI masking in the written output. The team does not require masking.
MASK_PHI = False

# -----------------------------------------------------------------------------
# Load profile. Each phase is a named stage with an offered request rate and
# a duration. The default profile mirrors the shape discussed in the
# sessions: a steady baseline, a stepped ramp to find the knee, a deliberate
# overload, and a recovery window.
#
# Reference points from prior sessions, for context when reading results:
#   - A 100 user, 10 minute staging run averaged roughly 3.2 seconds.
#   - Pods in staging began restarting at roughly 55 requests per second.
#   - The staging Apigee proxy quota was 3600 requests per minute, which is
#     60 requests per second. The production proxy quota was left unset.
# -----------------------------------------------------------------------------
PHASES = [
    # name,        offered_rps, duration_s, purpose
    ("warmup",           5,        60,  "Prime connections and caches. Excluded from scoring."),
    ("baseline",        25,       300,  "Steady state well below the known knee."),
    ("step_40",         40,       180,  "First step toward the knee."),
    ("step_55",         55,       180,  "The rate at which pods previously restarted."),
    ("step_70",         70,       180,  "Past the staging proxy quota equivalent."),
    ("overload",       150,       120,  "Deliberate overload to observe failure behavior."),
    ("recovery",        25,       300,  "Load returned to baseline to measure recovery."),
]

# Phases that are considered overload and are gated in production.
OVERLOAD_PHASES = {"overload", "step_70"}

# Phases excluded from the scored summary.
EXCLUDED_FROM_SCORE = {"warmup"}

# -----------------------------------------------------------------------------
# Client behavior.
# -----------------------------------------------------------------------------
REQUEST_TIMEOUT_S = 30               # hard client timeout per request
CONNECT_TIMEOUT_S = 5
MAX_WORKERS = 200                    # driver-side thread ceiling
CLIENT_RETRIES = 0                   # kept at zero; retries mask failure behavior

# Distributed generation. Set to True when the offered rate exceeds what the
# driver node can sustain. NUM_WORKER_SLOTS should not exceed the number of
# available executor cores.
DISTRIBUTED_LOAD = False
NUM_WORKER_SLOTS = 8

# -----------------------------------------------------------------------------
# Latency buckets. These mirror the bucketing used in the Gatling reports so
# results can be read side by side. The band above 1200 milliseconds is the
# one flagged for reduction in the performance session.
# -----------------------------------------------------------------------------
BUCKET_FAST_MS = 800
BUCKET_SLOW_MS = 1200

# -----------------------------------------------------------------------------
# Pass criteria for the automated verdict. These are starting values and
# should be confirmed with the business before being treated as commitments.
# -----------------------------------------------------------------------------
PASS_CRITERIA = {
    "baseline_p95_ms_max": 1200,
    "baseline_error_rate_max": 0.005,
    "recovery_seconds_max": 120,
    "regression_p95_pct_max": 0.20,      # p95 may not worsen by more than 20 percent
    "regression_error_rate_pct_max": 0.50,
}

# -----------------------------------------------------------------------------
# Output locations.
# -----------------------------------------------------------------------------
OUTPUT_DIR = "/dbfs/FileStore/pcis/perf"
BASELINE_STORE = f"{OUTPUT_DIR}/baseline_store.json"
TELEMETRY_INTERVAL_S = 5

# -----------------------------------------------------------------------------
# Pipeline mode. When True, the notebook exits with a JSON verdict that a
# calling job can gate on, and suppresses interactive display calls.
# -----------------------------------------------------------------------------
PIPELINE_MODE = False

# COMMAND ----------

# =============================================================================
# CELL 2 - IMPORTS AND SETUP
# =============================================================================

import json
import math
import os
import random
import re
import statistics
import string
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 250)

RUN_ID = f"{RUN_LABEL}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
RUN_START_UTC = datetime.now(timezone.utc)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}")


log(f"Run id: {RUN_ID}")
log(f"Environment: {ENV}")

if ENV not in ("staging", "prod"):
    raise ValueError("ENV must be set to 'staging' or 'prod'.")

if ENV == "prod" and not ALLOW_PROD_OVERLOAD:
    blocked = [p[0] for p in PHASES if p[0] in OVERLOAD_PHASES]
    if blocked:
        log(f"Production run. Overload phases are blocked and will be skipped: {blocked}")

# COMMAND ----------

# =============================================================================
# CELL 3 - AUTHENTICATION AND SESSION CONSTRUCTION
# =============================================================================
#
# One pooled session per target. The pool is sized to the worker ceiling so
# that connection setup is not itself the bottleneck being measured. Retries
# are disabled at the adapter level for the same reason a client retry count
# of zero is used - a retry hides the failure behavior this run is trying to
# observe.
# =============================================================================


def normalize_basic(token):
    token = (token or "").strip()
    if not token:
        return ""
    if token.lower().startswith("basic "):
        return token
    return f"Basic {token}"


def normalize_bearer(token):
    token = (token or "").strip()
    if not token:
        return ""
    lowered = token.lower()
    if lowered.startswith("bearer ") or lowered.startswith("apikey "):
        return token
    return f"Bearer {token}"


def build_headers(auth_mode):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if auth_mode == "basic":
        value = normalize_basic(AUTH_TOKEN)
    else:
        value = normalize_bearer(GATEWAY_TOKEN)
    if value:
        headers["Authorization"] = value
    return headers


def build_session(pool_size):
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=CLIENT_RETRIES,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


ACTIVE_TARGETS = {}
for name, cfg in TARGETS[ENV].items():
    if not cfg.get("enabled"):
        continue
    if "<" in cfg["url"]:
        log(f"Target '{name}' still contains a placeholder host and is skipped.")
        continue
    ACTIVE_TARGETS[name] = {
        "url": cfg["url"],
        "headers": build_headers(cfg["auth_mode"]),
        "session": build_session(MAX_WORKERS),
    }

if not ACTIVE_TARGETS:
    raise ValueError("No target is configured. Set at least one target URL for this environment.")

log(f"Active targets: {list(ACTIVE_TARGETS.keys())}")

if "gateway" not in ACTIVE_TARGETS:
    log(
        "Gateway path is not active. Results describe the search tier only and exclude the proxy "
        "and the search service. Proxy quota and gateway timeouts cannot be observed in this run."
    )

# COMMAND ----------

# =============================================================================
# CELL 4 - QUERY TEMPLATE LOADING, SUBSTITUTION, AND PRUNING
# =============================================================================
#
# The template is a single bool query with minimum_should_match of one. The
# should array holds tiered clauses ordered by specificity, with boosts
# spaced by orders of magnitude so the firing tier dominates ranking.
#
# Pruning behavior replicated here matches the search service:
#   1. Substitute every non-blank parameter.
#   2. Drop any must or should child that still contains a placeholder, and
#      drop any bool that has been emptied as a result.
#   3. Strip any leftover placeholder text.
#
# Pruning is leaf level. A blank field removes only the clauses that
# reference it; the surrounding tier survives on its remaining clauses.
#
# Known behavior worth watching in load results: clauses that bundle first,
# middle, and last name together are removed for any search with no middle
# name, because the leftover middle name placeholder makes the whole clause
# appear unfilled. Searches without a middle name therefore execute a
# structurally different and generally cheaper query. Cost per request is
# not uniform across the pool, and the summary reports latency split by
# query shape for this reason.
# =============================================================================

PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_0-9]+\}\}")

PLACEHOLDER_FIELDS = {
    "{{FIRSTNAME}}": "first_name",
    "{{MIDDLENAME}}": "middle_name",
    "{{LASTNAME}}": "last_name",
    "{{ANUMBER}}": "a_number",
    "{{DOB}}": "dob",
    "{{COB}}": "cob",
    "{{COC}}": "coc",
}


def load_template(path):
    if not path:
        return None
    read_path = path
    if read_path.startswith("dbfs:/"):
        read_path = "/dbfs/" + read_path[len("dbfs:/"):].lstrip("/")
    with open(read_path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Template at {path} is not valid JSON: {exc}. "
            "Confirm the template was pulled from the API repo and not the HTML test UI."
        )
    return raw


def substitute(raw_template, params):
    filled = raw_template
    for placeholder, field in PLACEHOLDER_FIELDS.items():
        value = params.get(field)
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        escaped = json.dumps(value)[1:-1]
        filled = filled.replace(placeholder, escaped)
    return filled


def contains_placeholder(node):
    if isinstance(node, str):
        return bool(PLACEHOLDER_RE.search(node))
    if isinstance(node, dict):
        return any(contains_placeholder(v) for v in node.values())
    if isinstance(node, list):
        return any(contains_placeholder(v) for v in node)
    return False


def prune(node):
    if isinstance(node, dict):
        pruned = {}
        for key, value in node.items():
            if key in ("must", "should", "filter", "must_not") and isinstance(value, list):
                kept = [c for c in value if not contains_placeholder(c)]
                kept = [prune(c) for c in kept]
                kept = [c for c in kept if c not in ({}, [], None)]
                if kept:
                    pruned[key] = kept
            else:
                child = prune(value)
                if child in ({}, [], None) and isinstance(value, (dict, list)):
                    continue
                pruned[key] = child
        if "bool" in pruned and not pruned["bool"]:
            return {}
        return pruned
    if isinstance(node, list):
        items = [prune(v) for v in node]
        return [v for v in items if v not in ({}, [], None)]
    return node


def strip_leftovers(node):
    if isinstance(node, str):
        return PLACEHOLDER_RE.sub("", node)
    if isinstance(node, dict):
        return {k: strip_leftovers(v) for k, v in node.items()}
    if isinstance(node, list):
        return [strip_leftovers(v) for v in node]
    return node


def build_query(raw_template, params):
    filled = substitute(raw_template, params)
    parsed = json.loads(filled)
    parsed = prune(parsed)
    parsed = strip_leftovers(parsed)
    return parsed


def query_shape(params):
    parts = []
    if params.get("first_name"):
        parts.append("F")
    if params.get("middle_name"):
        parts.append("M")
    if params.get("last_name"):
        parts.append("L")
    if params.get("a_number"):
        parts.append("A")
    if params.get("dob"):
        parts.append("D")
    if params.get("cob") or params.get("coc"):
        parts.append("C")
    return "".join(parts) or "EMPTY"


TEMPLATE_A = load_template(TEMPLATE_PATH)
TEMPLATE_B = load_template(TEMPLATE_PATH_B) if TEMPLATE_PATH_B else None

TEMPLATES = {TEMPLATE_LABEL_A: TEMPLATE_A}
if TEMPLATE_B:
    TEMPLATES[TEMPLATE_LABEL_B] = TEMPLATE_B

log(f"Templates loaded: {list(TEMPLATES.keys())}")

# COMMAND ----------

# =============================================================================
# CELL 5 - QUERY POOL FROM PRODUCTION AUDIT LOGS
# =============================================================================
#
# A load test that repeats one payload measures the cache behavior of one
# payload. The pool here is drawn from production audit logs so that the
# distribution of search shapes under test matches the distribution the
# system actually serves.
#
# If the audit log table is unavailable, a synthetic pool is generated and
# the run is flagged as synthetic in the summary. A synthetic run is still
# useful for comparing one iteration against another, but it should not be
# presented as representative of production traffic.
# =============================================================================


def load_pool_from_audit_logs():
    cutoff = (datetime.now(timezone.utc) - timedelta(days=AUDIT_LOG_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    query = f"""
        SELECT
            first_name,
            middle_name,
            last_name,
            a_number,
            dob,
            cob,
            coc,
            consumer
        FROM {AUDIT_LOG_TABLE}
        WHERE request_date >= '{cutoff}'
        LIMIT {QUERY_POOL_SIZE * 3}
    """
    pdf = spark.sql(query).toPandas()
    records = []
    seen = set()
    for row in pdf.to_dict(orient="records"):
        cleaned = {k: ("" if pd.isna(v) else str(v).strip()) for k, v in row.items()}
        key = "|".join(
            cleaned.get(f, "") for f in
            ["first_name", "middle_name", "last_name", "a_number", "dob", "cob", "coc"]
        )
        if key in seen or key == "||||||":
            continue
        seen.add(key)
        records.append(cleaned)
        if len(records) >= QUERY_POOL_SIZE:
            break
    return records


def build_synthetic_pool(size):
    firsts = ["MARIA", "JOSE", "AHMED", "LI", "ANNA", "CARLOS", "FATIMA", "JOHN", "PRIYA", "OMAR"]
    middles = ["", "", "", "A", "LUIS", "MARIE", "KUMAR", ""]
    lasts = ["GARCIA", "NGUYEN", "PATEL", "SMITH", "HERNANDEZ", "KIM", "ALI", "SANTOS", "WANG"]
    records = []
    rng = random.Random(1729)
    for _ in range(size):
        has_dob = rng.random() < 0.55
        has_anum = rng.random() < 0.20
        records.append({
            "first_name": rng.choice(firsts),
            "middle_name": rng.choice(middles),
            "last_name": rng.choice(lasts),
            "a_number": f"{rng.randint(100000000, 999999999)}" if has_anum else "",
            "dob": f"{rng.randint(1955, 2005)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}" if has_dob else "",
            "cob": rng.choice(["", "MX", "IN", "PH", "CN", "SV"]),
            "coc": "",
            "consumer": rng.choice(["EVERIFY", "PCIS_UI", "BHUB", "CRIS", "ELIS", "GLOBAL", "UIPATH"]),
        })
    return records


POOL_IS_SYNTHETIC = False
try:
    QUERY_POOL = load_pool_from_audit_logs()
    if not QUERY_POOL:
        raise ValueError("Audit log query returned no usable rows.")
    log(f"Query pool loaded from audit logs: {len(QUERY_POOL)} distinct searches.")
except Exception as exc:
    POOL_IS_SYNTHETIC = True
    QUERY_POOL = build_synthetic_pool(QUERY_POOL_SIZE)
    log(f"Audit log pool unavailable ({exc}). Falling back to a synthetic pool of {len(QUERY_POOL)}.")

SHAPE_MIX = Counter(query_shape(p) for p in QUERY_POOL)
log(f"Query shape mix: {dict(SHAPE_MIX.most_common(10))}")

# Pre-build the request bodies once so that template rendering time is not
# counted as request latency.
PREBUILT = defaultdict(list)
for template_label, raw in TEMPLATES.items():
    for params in QUERY_POOL:
        try:
            body = build_query(raw, params)
        except Exception:
            continue
        PREBUILT[template_label].append({
            "body": json.dumps(body),
            "shape": query_shape(params),
            "consumer": params.get("consumer", ""),
            "first_name": params.get("first_name", ""),
            "last_name": params.get("last_name", ""),
            "dob": params.get("dob", ""),
            "a_number": params.get("a_number", ""),
        })
    log(f"Prebuilt {len(PREBUILT[template_label])} request bodies for template '{template_label}'.")

# COMMAND ----------

# =============================================================================
# CELL 6 - ERROR TAXONOMY
# =============================================================================
#
# The stress session showed several distinct failure modes being reported as
# one number. Separating them is what makes the result actionable:
#
#   connection_refused / connection_reset
#       The pod is gone. In the observed staging failure this was the
#       Kubernetes liveness probe killing containers after the health
#       endpoint failed to answer within its timeout, not OpenSearch
#       rejecting work.
#   client_timeout
#       The request exceeded the client timeout. Pair with server_took_ms
#       to determine whether search was slow or the request never reached
#       search.
#   http_429
#       Proxy quota. Only reachable on the gateway path.
#   http_502 / http_503 / http_504
#       Gateway or upstream unavailable. Typically follows pod restarts.
#   os_rejected
#       OpenSearch thread pool rejection. This is the signal that search
#       itself is saturated.
#   os_task_cancelled / os_channel_closed
#       Search work abandoned. Observed in cluster logs during the prior
#       run and consistent with clients disconnecting rather than with
#       search failing.
# =============================================================================

OS_ERROR_MARKERS = [
    ("os_rejected", "es_rejected_execution_exception"),
    ("os_rejected", "rejected_execution_exception"),
    ("os_task_cancelled", "task_cancelled"),
    ("os_task_cancelled", "TaskCancelledException"),
    ("os_channel_closed", "channel closed"),
    ("os_circuit_breaker", "circuit_breaking_exception"),
    ("os_search_phase", "search_phase_execution_exception"),
    ("os_too_many_clauses", "too_many_clauses"),
]


def classify_exception(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    if "connectionreset" in text or "connection reset" in text:
        return "connection_reset"
    if "connection refused" in text or "newconnectionerror" in text:
        return "connection_refused"
    if "readtimeout" in text or "read timed out" in text:
        return "client_timeout"
    if "connecttimeout" in text or "connection timed out" in text:
        return "connect_timeout"
    if "toomanyredirects" in text:
        return "redirect_error"
    if "sslerror" in text:
        return "tls_error"
    if "protocolerror" in text or "remotedisconnected" in text:
        return "connection_reset"
    return "client_error_other"


def classify_response(status_code, body_text):
    if status_code == 200:
        lowered = (body_text or "")[:4000].lower()
        for label, marker in OS_ERROR_MARKERS:
            if marker.lower() in lowered:
                return label
        return "ok"
    if status_code == 429:
        return "http_429"
    if status_code in (502, 503, 504):
        return f"http_{status_code}"
    if 400 <= status_code < 500:
        lowered = (body_text or "")[:4000].lower()
        for label, marker in OS_ERROR_MARKERS:
            if marker.lower() in lowered:
                return label
        return f"http_{status_code}"
    if status_code >= 500:
        lowered = (body_text or "")[:4000].lower()
        for label, marker in OS_ERROR_MARKERS:
            if marker.lower() in lowered:
                return label
        return f"http_{status_code}"
    return f"http_{status_code}"


SUCCESS_LABELS = {"ok"}

# COMMAND ----------

# =============================================================================
# CELL 7 - CLUSTER TELEMETRY SAMPLER
# =============================================================================
#
# Sampled in the background for the duration of the run. The two fields that
# matter most for attributing a bottleneck:
#
#   search queue and rejected counters - a rising queue with a rising
#   rejected count means OpenSearch is the constraint.
#
#   an unchanged rejected count while client errors climb means the
#   constraint is upstream of search, in the service or the proxy.
# =============================================================================

TELEMETRY_ROWS = []
_telemetry_stop = threading.Event()


def sample_telemetry():
    base = CLUSTER_BASE_URL.get(ENV, "")
    if not base or "<" in base:
        log("Cluster telemetry endpoint not configured. Telemetry sampling is disabled.")
        return

    headers = build_headers("basic")
    session = build_session(4)
    last_rejected = {}

    while not _telemetry_stop.is_set():
        stamp = time.time()
        row = {"ts": stamp, "iso": datetime.fromtimestamp(stamp, timezone.utc).isoformat()}
        try:
            health = session.get(
                f"{base}/_cluster/health",
                headers=headers,
                timeout=(CONNECT_TIMEOUT_S, 10),
            ).json()
            row["cluster_status"] = health.get("status")
            row["active_shards"] = health.get("active_shards")
            row["relocating_shards"] = health.get("relocating_shards")
            row["number_of_nodes"] = health.get("number_of_nodes")
            row["task_max_waiting_ms"] = health.get("task_max_waiting_in_queue_millis")
        except Exception as exc:
            row["health_error"] = classify_exception(exc)

        try:
            stats = session.get(
                f"{base}/_nodes/stats/thread_pool,jvm,os",
                headers=headers,
                timeout=(CONNECT_TIMEOUT_S, 10),
            ).json()
            queue_total = 0
            rejected_total = 0
            active_total = 0
            heap_pcts = []
            cpu_pcts = []
            for node_id, node in (stats.get("nodes") or {}).items():
                search_pool = ((node.get("thread_pool") or {}).get("search") or {})
                queue_total += search_pool.get("queue", 0) or 0
                rejected_total += search_pool.get("rejected", 0) or 0
                active_total += search_pool.get("active", 0) or 0
                heap = ((node.get("jvm") or {}).get("mem") or {}).get("heap_used_percent")
                if heap is not None:
                    heap_pcts.append(heap)
                cpu = ((node.get("os") or {}).get("cpu") or {}).get("percent")
                if cpu is not None:
                    cpu_pcts.append(cpu)
            row["search_queue"] = queue_total
            row["search_active"] = active_total
            row["search_rejected_cumulative"] = rejected_total
            row["search_rejected_delta"] = rejected_total - last_rejected.get("v", rejected_total)
            last_rejected["v"] = rejected_total
            row["heap_used_pct_max"] = max(heap_pcts) if heap_pcts else None
            row["cpu_pct_max"] = max(cpu_pcts) if cpu_pcts else None
        except Exception as exc:
            row["stats_error"] = classify_exception(exc)

        TELEMETRY_ROWS.append(row)
        _telemetry_stop.wait(TELEMETRY_INTERVAL_S)


def start_telemetry():
    thread = threading.Thread(target=sample_telemetry, daemon=True)
    thread.start()
    return thread


def stop_telemetry(thread):
    _telemetry_stop.set()
    if thread is not None:
        thread.join(timeout=TELEMETRY_INTERVAL_S + 5)

# COMMAND ----------

# =============================================================================
# CELL 8 - LOAD ENGINE
# =============================================================================
#
# Open model generation. Requests are issued on a fixed schedule derived
# from the offered rate rather than waiting for the previous response. A
# closed model, where a fixed number of virtual users each wait for their
# response, hides queueing: as the system slows, a closed model quietly
# reduces the offered rate and the system appears to cope.
#
# Scheduling slip is recorded. When the client cannot keep up with the
# schedule, the shortfall is reported rather than silently absorbed.
# =============================================================================


def issue_request(target_name, target, entry, template_label, phase_name, scheduled_at):
    started = time.time()
    record = {
        "phase": phase_name,
        "target": target_name,
        "template": template_label,
        "shape": entry["shape"],
        "consumer": entry["consumer"],
        "scheduled_at": scheduled_at,
        "started_at": started,
        "schedule_slip_ms": (started - scheduled_at) * 1000.0,
    }
    if not MASK_PHI:
        record["first_name"] = entry["first_name"]
        record["last_name"] = entry["last_name"]
        record["dob"] = entry["dob"]
        record["a_number"] = entry["a_number"]

    try:
        response = target["session"].post(
            target["url"],
            headers=target["headers"],
            data=entry["body"],
            timeout=(CONNECT_TIMEOUT_S, REQUEST_TIMEOUT_S),
        )
        elapsed = (time.time() - started) * 1000.0
        text = response.text
        label = classify_response(response.status_code, text)
        record.update({
            "status_code": response.status_code,
            "outcome": label,
            "success": label in SUCCESS_LABELS,
            "client_latency_ms": elapsed,
        })
        if label in SUCCESS_LABELS:
            try:
                payload = response.json()
                record["server_took_ms"] = payload.get("took")
                hits = (payload.get("hits") or {})
                total = hits.get("total")
                if isinstance(total, dict):
                    record["hit_count"] = total.get("value")
                else:
                    record["hit_count"] = total
                hit_list = hits.get("hits") or []
                record["returned_count"] = len(hit_list)
                record["top_score"] = hit_list[0].get("_score") if hit_list else None
                record["timed_out_flag"] = payload.get("timed_out")
                shards = payload.get("_shards") or {}
                record["shards_failed"] = shards.get("failed")
            except Exception:
                record["parse_error"] = True
        else:
            record["error_snippet"] = (text or "")[:300]
    except Exception as exc:
        elapsed = (time.time() - started) * 1000.0
        label = classify_exception(exc)
        record.update({
            "status_code": None,
            "outcome": label,
            "success": False,
            "client_latency_ms": elapsed,
            "error_snippet": f"{type(exc).__name__}: {exc}"[:300],
        })

    record["completed_at"] = time.time()
    return record


def run_phase(phase_name, offered_rps, duration_s, target_name, target, template_label):
    entries = PREBUILT[template_label]
    if not entries:
        raise ValueError(f"No prebuilt request bodies for template '{template_label}'.")

    total_requests = int(offered_rps * duration_s)
    interval = 1.0 / float(offered_rps) if offered_rps > 0 else 0.0
    workers = min(MAX_WORKERS, max(8, int(offered_rps * 4)))
    rng = random.Random(hash(f"{phase_name}{target_name}{template_label}") & 0xFFFFFFFF)

    log(
        f"Phase '{phase_name}' | target={target_name} | template={template_label} | "
        f"offered={offered_rps} rps | duration={duration_s}s | planned={total_requests} requests"
    )

    results = []
    phase_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for i in range(total_requests):
            scheduled_at = phase_start + (i * interval)
            wait = scheduled_at - time.time()
            if wait > 0:
                time.sleep(wait)
            entry = entries[rng.randrange(len(entries))]
            futures.append(
                pool.submit(
                    issue_request,
                    target_name, target, entry, template_label, phase_name, scheduled_at,
                )
            )
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({
                    "phase": phase_name,
                    "target": target_name,
                    "template": template_label,
                    "outcome": "harness_error",
                    "success": False,
                    "error_snippet": str(exc)[:300],
                })

    phase_end = time.time()
    wall = phase_end - phase_start
    achieved = len(results) / wall if wall > 0 else 0.0
    log(
        f"Phase '{phase_name}' complete | wall={wall:.1f}s | "
        f"achieved={achieved:.1f} rps | offered={offered_rps} rps"
    )
    for record in results:
        record["phase_start"] = phase_start
        record["phase_end"] = phase_end
        record["phase_offered_rps"] = offered_rps
        record["phase_achieved_rps"] = achieved
    return results

# COMMAND ----------

# =============================================================================
# CELL 9 - DISTRIBUTED GENERATION (OPTIONAL)
# =============================================================================
#
# Enabled by setting DISTRIBUTED_LOAD to True. Each executor slot generates
# a share of the offered rate. Use this when the offered rate exceeds what a
# single driver node can sustain, which is the point at which driver-only
# results stop being trustworthy.
#
# The worker function is deliberately self contained so that nothing from
# the driver scope is captured.
# =============================================================================


def distributed_phase(phase_name, offered_rps, duration_s, target_name, target_cfg,
                      template_label, entries, slots):
    payload = {
        "phase_name": phase_name,
        "duration_s": duration_s,
        "target_name": target_name,
        "url": target_cfg["url"],
        "headers": target_cfg["headers"],
        "template_label": template_label,
        "connect_timeout": CONNECT_TIMEOUT_S,
        "request_timeout": REQUEST_TIMEOUT_S,
        "mask_phi": MASK_PHI,
    }
    per_slot_rps = offered_rps / float(slots)
    sample = entries[:500]
    broadcast = spark.sparkContext.broadcast({"cfg": payload, "entries": sample,
                                              "per_slot_rps": per_slot_rps})

    def worker(slot_index):
        import json as _json
        import random as _random
        import time as _time
        from concurrent.futures import ThreadPoolExecutor as _Pool
        import requests as _requests

        state = broadcast.value
        cfg = state["cfg"]
        pool_entries = state["entries"]
        rps = state["per_slot_rps"]
        session = _requests.Session()
        rows = []
        interval = 1.0 / rps if rps > 0 else 0.0
        total = int(rps * cfg["duration_s"])
        start = _time.time()
        rng = _random.Random(slot_index * 7919)

        def one(entry, scheduled_at):
            began = _time.time()
            row = {
                "phase": cfg["phase_name"],
                "target": cfg["target_name"],
                "template": cfg["template_label"],
                "shape": entry["shape"],
                "consumer": entry["consumer"],
                "scheduled_at": scheduled_at,
                "started_at": began,
                "schedule_slip_ms": (began - scheduled_at) * 1000.0,
                "slot": slot_index,
            }
            try:
                resp = session.post(
                    cfg["url"], headers=cfg["headers"], data=entry["body"],
                    timeout=(cfg["connect_timeout"], cfg["request_timeout"]),
                )
                row["client_latency_ms"] = (_time.time() - began) * 1000.0
                row["status_code"] = resp.status_code
                if resp.status_code == 200:
                    row["outcome"] = "ok"
                    row["success"] = True
                    try:
                        body = resp.json()
                        row["server_took_ms"] = body.get("took")
                        hits = body.get("hits") or {}
                        total_hits = hits.get("total")
                        row["hit_count"] = (total_hits or {}).get("value") if isinstance(total_hits, dict) else total_hits
                        row["returned_count"] = len(hits.get("hits") or [])
                    except Exception:
                        row["parse_error"] = True
                else:
                    row["outcome"] = f"http_{resp.status_code}"
                    row["success"] = False
                    row["error_snippet"] = (resp.text or "")[:300]
            except Exception as exc:
                row["client_latency_ms"] = (_time.time() - began) * 1000.0
                row["outcome"] = "worker_exception"
                row["success"] = False
                row["error_snippet"] = f"{type(exc).__name__}: {exc}"[:300]
            row["completed_at"] = _time.time()
            return row

        with _Pool(max_workers=max(4, int(rps * 4))) as pool:
            futures = []
            for i in range(total):
                scheduled = start + i * interval
                delay = scheduled - _time.time()
                if delay > 0:
                    _time.sleep(delay)
                futures.append(pool.submit(one, pool_entries[rng.randrange(len(pool_entries))], scheduled))
            for fut in futures:
                rows.append(fut.result())
        return rows

    rdd = spark.sparkContext.parallelize(list(range(slots)), slots)
    collected = rdd.flatMap(worker).collect()
    log(f"Distributed phase '{phase_name}' collected {len(collected)} records from {slots} slots.")
    return collected

# COMMAND ----------

# =============================================================================
# CELL 10 - RUN ORCHESTRATION
# =============================================================================

telemetry_thread = start_telemetry()
ALL_RECORDS = []

try:
    for phase_name, offered_rps, duration_s, purpose in PHASES:
        if ENV == "prod" and phase_name in OVERLOAD_PHASES and not ALLOW_PROD_OVERLOAD:
            log(f"Skipping overload phase '{phase_name}' in production.")
            continue

        for template_label in TEMPLATES.keys():
            for target_name, target in ACTIVE_TARGETS.items():
                if DISTRIBUTED_LOAD:
                    records = distributed_phase(
                        phase_name, offered_rps, duration_s, target_name,
                        TARGETS[ENV][target_name] | {"headers": target["headers"]},
                        template_label, PREBUILT[template_label], NUM_WORKER_SLOTS,
                    )
                else:
                    records = run_phase(
                        phase_name, offered_rps, duration_s,
                        target_name, target, template_label,
                    )
                ALL_RECORDS.extend(records)

        # Settle window between phases so that the next phase does not
        # inherit the queue depth of the previous one.
        time.sleep(10)
finally:
    stop_telemetry(telemetry_thread)

results_df = pd.DataFrame(ALL_RECORDS)
telemetry_df = pd.DataFrame(TELEMETRY_ROWS)

log(f"Total requests recorded: {len(results_df)}")
log(f"Telemetry samples recorded: {len(telemetry_df)}")

# COMMAND ----------

# =============================================================================
# CELL 11 - METRICS
# =============================================================================


def percentile(values, pct):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    clean = sorted(clean)
    index = min(len(clean) - 1, max(0, int(round((pct / 100.0) * len(clean) + 0.5)) - 1))
    return clean[index]


def summarize(frame, group_cols):
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        latencies = group["client_latency_ms"].tolist()
        successes = group[group["success"] == True]           # noqa: E712
        success_lat = successes["client_latency_ms"].tolist()
        took = successes["server_took_ms"].dropna().tolist() if "server_took_ms" in successes else []

        total = len(group)
        ok = len(successes)
        fast = sum(1 for v in success_lat if v is not None and v < BUCKET_FAST_MS)
        mid = sum(1 for v in success_lat if v is not None and BUCKET_FAST_MS <= v <= BUCKET_SLOW_MS)
        slow = sum(1 for v in success_lat if v is not None and v > BUCKET_SLOW_MS)

        wall = 0.0
        if "phase_start" in group and "phase_end" in group and not group["phase_start"].isna().all():
            wall = float(group["phase_end"].max() - group["phase_start"].min())

        row = dict(zip(group_cols, keys))
        row.update({
            "requests": total,
            "ok": ok,
            "failed": total - ok,
            "error_rate": round((total - ok) / total, 5) if total else None,
            "offered_rps": round(float(group["phase_offered_rps"].mean()), 2) if "phase_offered_rps" in group else None,
            "achieved_rps": round(total / wall, 2) if wall > 0 else None,
            "p50_ms": round(percentile(success_lat, 50), 1) if success_lat else None,
            "p75_ms": round(percentile(success_lat, 75), 1) if success_lat else None,
            "p90_ms": round(percentile(success_lat, 90), 1) if success_lat else None,
            "p95_ms": round(percentile(success_lat, 95), 1) if success_lat else None,
            "p99_ms": round(percentile(success_lat, 99), 1) if success_lat else None,
            "max_ms": round(max(success_lat), 1) if success_lat else None,
            "mean_ms": round(statistics.fmean(success_lat), 1) if success_lat else None,
            "pct_under_800ms": round(fast / ok, 4) if ok else None,
            "pct_800_to_1200ms": round(mid / ok, 4) if ok else None,
            "pct_over_1200ms": round(slow / ok, 4) if ok else None,
            "server_took_p50_ms": round(percentile(took, 50), 1) if took else None,
            "server_took_p95_ms": round(percentile(took, 95), 1) if took else None,
            "overhead_p50_ms": None,
            "schedule_slip_p95_ms": round(percentile(group["schedule_slip_ms"].tolist(), 95), 1)
                                    if "schedule_slip_ms" in group else None,
        })
        if row["p50_ms"] is not None and row["server_took_p50_ms"] is not None:
            row["overhead_p50_ms"] = round(row["p50_ms"] - row["server_took_p50_ms"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


PHASE_ORDER = {name: i for i, (name, _, _, _) in enumerate(PHASES)}

phase_summary = summarize(results_df, ["phase", "target", "template"])
phase_summary["phase_order"] = phase_summary["phase"].map(PHASE_ORDER).fillna(999)
phase_summary = phase_summary.sort_values(["phase_order", "target", "template"]).drop(columns=["phase_order"])

shape_summary = summarize(results_df, ["phase", "shape"])
consumer_summary = summarize(results_df, ["phase", "consumer"]) if "consumer" in results_df else pd.DataFrame()

# Error taxonomy by phase and target.
error_rows = []
failures = results_df[results_df["success"] != True]          # noqa: E712
if len(failures):
    for keys, group in failures.groupby(["phase", "target", "outcome"], dropna=False):
        error_rows.append({
            "phase": keys[0],
            "target": keys[1],
            "outcome": keys[2],
            "count": len(group),
            "example": (group["error_snippet"].dropna().iloc[0]
                        if "error_snippet" in group and group["error_snippet"].notna().any() else ""),
        })
error_summary = pd.DataFrame(error_rows).sort_values(["phase", "count"], ascending=[True, False]) \
    if error_rows else pd.DataFrame(columns=["phase", "target", "outcome", "count", "example"])

if not PIPELINE_MODE:
    print("\nPHASE SUMMARY")
    print(phase_summary.to_string(index=False))
    print("\nERROR TAXONOMY")
    print(error_summary.to_string(index=False) if len(error_summary) else "No failures recorded.")

# COMMAND ----------

# =============================================================================
# CELL 12 - KNEE DETECTION AND BOTTLENECK ATTRIBUTION
# =============================================================================
#
# The knee is the highest offered rate at which the error rate and the p95
# both remain within the pass criteria. Everything above it is the region
# where the stack degrades.
#
# Attribution compares three signals at the point of degradation:
#
#   - OpenSearch rejection deltas from telemetry. Present means search is
#     the constraint.
#   - The gap between client latency and server took time. A large and
#     growing gap with flat server time means the constraint is in front of
#     search - the service, the pods, or the proxy.
#   - Connection level failures. These indicate the process serving the
#     request went away, which in the observed staging case traced to the
#     liveness probe restarting containers rather than to search failing.
# =============================================================================

scored = phase_summary[~phase_summary["phase"].isin(EXCLUDED_FROM_SCORE)].copy()

knee_rows = []
for target_name in scored["target"].unique():
    subset = scored[scored["target"] == target_name].copy()
    subset["phase_order"] = subset["phase"].map(PHASE_ORDER).fillna(999)
    subset = subset.sort_values("phase_order")
    knee = None
    first_break = None
    for _, row in subset.iterrows():
        within = (
            (row["error_rate"] is not None and row["error_rate"] <= PASS_CRITERIA["baseline_error_rate_max"])
            and (row["p95_ms"] is not None and row["p95_ms"] <= PASS_CRITERIA["baseline_p95_ms_max"])
        )
        if within:
            knee = row
        elif first_break is None:
            first_break = row
    knee_rows.append({
        "target": target_name,
        "knee_phase": knee["phase"] if knee is not None else "none",
        "knee_achieved_rps": knee["achieved_rps"] if knee is not None else None,
        "knee_p95_ms": knee["p95_ms"] if knee is not None else None,
        "first_breaking_phase": first_break["phase"] if first_break is not None else "none",
        "first_breaking_rps": first_break["achieved_rps"] if first_break is not None else None,
        "first_breaking_error_rate": first_break["error_rate"] if first_break is not None else None,
        "first_breaking_p95_ms": first_break["p95_ms"] if first_break is not None else None,
    })
knee_df = pd.DataFrame(knee_rows)


def attribute_bottleneck():
    notes = []

    rejected = 0
    if len(telemetry_df) and "search_rejected_delta" in telemetry_df:
        rejected = int(telemetry_df["search_rejected_delta"].fillna(0).clip(lower=0).sum())
    if rejected > 0:
        notes.append(
            f"OpenSearch recorded {rejected} search thread pool rejections during the run. "
            "Search capacity is a real constraint at the rates tested."
        )
    elif len(telemetry_df):
        notes.append(
            "OpenSearch recorded no search thread pool rejections during the run. "
            "Degradation observed at the client is therefore upstream of search."
        )
    else:
        notes.append("Cluster telemetry was not collected, so search-side saturation cannot be confirmed.")

    degraded = scored[scored["p95_ms"].notna() & (scored["p95_ms"] > PASS_CRITERIA["baseline_p95_ms_max"])]
    if len(degraded):
        overheads = degraded["overhead_p50_ms"].dropna()
        server_p95 = degraded["server_took_p95_ms"].dropna()
        if len(overheads) and len(server_p95):
            if overheads.mean() > server_p95.mean():
                notes.append(
                    "In degraded phases the time spent outside search exceeded the reported search time. "
                    "The constraint sits in the request path in front of search."
                )
            else:
                notes.append(
                    "In degraded phases the reported search time was the dominant component of total latency."
                )

    conn_failures = 0
    if len(error_summary):
        conn_labels = {"connection_refused", "connection_reset", "connect_timeout",
                       "http_502", "http_503", "http_504"}
        conn_failures = int(error_summary[error_summary["outcome"].isin(conn_labels)]["count"].sum())
    if conn_failures:
        notes.append(
            f"{conn_failures} connection level failures were recorded. These are consistent with the "
            "serving process being restarted or unavailable rather than with search returning an error. "
            "Confirm against pod restart counts and liveness probe events for the same window."
        )

    quota_failures = int(error_summary[error_summary["outcome"] == "http_429"]["count"].sum()) \
        if len(error_summary) else 0
    if quota_failures:
        notes.append(
            f"{quota_failures} responses were proxy quota rejections. The gateway quota is being reached "
            "before the search tier is. Confirm the configured quota for the proxy under test."
        )

    slip = scored["schedule_slip_p95_ms"].dropna()
    if len(slip) and slip.max() > 2000:
        notes.append(
            f"Client scheduling slip reached {round(float(slip.max()), 0)} milliseconds at p95. "
            "The load generator did not sustain the offered rate. Treat achieved rate as the valid figure "
            "and consider enabling distributed generation."
        )

    return notes


ATTRIBUTION_NOTES = attribute_bottleneck()

if not PIPELINE_MODE:
    print("\nKNEE DETECTION")
    print(knee_df.to_string(index=False))
    print("\nBOTTLENECK ATTRIBUTION")
    for note in ATTRIBUTION_NOTES:
        print(f"  - {note}")

# COMMAND ----------

# =============================================================================
# CELL 13 - RECOVERY MEASUREMENT
# =============================================================================
#
# The recovery phase answers the second question raised directly in the
# performance session: when the stress is removed, does the stack recover,
# and how long does it take.
#
# Recovery is defined as the first ten second window inside the recovery
# phase where the error rate returns to zero and the p95 returns to within
# the pass threshold, and where the two subsequent windows also hold. The
# requirement for consecutive windows prevents a single quiet window from
# being read as recovery.
# =============================================================================

RECOVERY_WINDOW_S = 10
RECOVERY_CONSECUTIVE = 3


def measure_recovery():
    if "recovery" not in set(results_df.get("phase", [])):
        return {"measured": False, "reason": "No recovery phase was executed."}

    phase = results_df[results_df["phase"] == "recovery"].copy()
    if not len(phase):
        return {"measured": False, "reason": "Recovery phase produced no records."}

    start = float(phase["phase_start"].min())
    phase["window"] = ((phase["completed_at"] - start) // RECOVERY_WINDOW_S).astype(int)

    windows = []
    for window_index, group in phase.groupby("window"):
        ok = group[group["success"] == True]                  # noqa: E712
        lat = ok["client_latency_ms"].tolist()
        windows.append({
            "window": int(window_index),
            "seconds_from_release": int(window_index * RECOVERY_WINDOW_S),
            "requests": len(group),
            "error_rate": round((len(group) - len(ok)) / len(group), 4) if len(group) else None,
            "p95_ms": round(percentile(lat, 95), 1) if lat else None,
        })
    windows_df = pd.DataFrame(windows).sort_values("window")

    healthy_streak = 0
    recovered_at = None
    for _, row in windows_df.iterrows():
        healthy = (
            row["error_rate"] is not None and row["error_rate"] <= PASS_CRITERIA["baseline_error_rate_max"]
            and row["p95_ms"] is not None and row["p95_ms"] <= PASS_CRITERIA["baseline_p95_ms_max"]
        )
        if healthy:
            healthy_streak += 1
            if healthy_streak == RECOVERY_CONSECUTIVE and recovered_at is None:
                recovered_at = int(row["seconds_from_release"]) - (RECOVERY_CONSECUTIVE - 1) * RECOVERY_WINDOW_S
        else:
            healthy_streak = 0

    return {
        "measured": True,
        "recovered": recovered_at is not None,
        "recovery_seconds": recovered_at,
        "windows": windows_df,
    }


RECOVERY = measure_recovery()

if not PIPELINE_MODE:
    print("\nRECOVERY")
    if not RECOVERY.get("measured"):
        print(f"  {RECOVERY.get('reason')}")
    elif RECOVERY["recovered"]:
        print(f"  Recovered to baseline {RECOVERY['recovery_seconds']} seconds after load was reduced.")
    else:
        print("  Did not return to baseline within the recovery window. Extend the recovery phase and re-run.")
    if RECOVERY.get("measured"):
        print(RECOVERY["windows"].head(20).to_string(index=False))

# COMMAND ----------

# =============================================================================
# CELL 14 - BASELINE COMPARISON AND VERDICT
# =============================================================================
#
# This is the mechanism for monitoring how the system changes between
# promotional layers and code changes. Each run writes a compact summary
# record. The current run is compared against the most recent stored run
# with the same comparison key, and the differences are reported as changed,
# improved, or regressed.
#
# The comparison is deliberately narrow. It reports how throughput, latency
# and error behavior moved between iterations. It does not make any claim
# about result quality or correctness, which is handled separately by the
# query comparison tooling.
# =============================================================================

COMPARISON_KEY = f"{ENV}|{'+'.join(sorted(ACTIVE_TARGETS.keys()))}|{'+'.join(sorted(TEMPLATES.keys()))}"


def build_run_record():
    baseline_rows = scored[scored["phase"] == "baseline"]
    baseline = baseline_rows.iloc[0].to_dict() if len(baseline_rows) else {}
    return {
        "run_id": RUN_ID,
        "run_label": RUN_LABEL,
        "run_notes": RUN_NOTES,
        "comparison_key": COMPARISON_KEY,
        "started_utc": RUN_START_UTC.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "env": ENV,
        "targets": sorted(ACTIVE_TARGETS.keys()),
        "templates": sorted(TEMPLATES.keys()),
        "pool_synthetic": POOL_IS_SYNTHETIC,
        "pool_size": len(QUERY_POOL),
        "total_requests": int(len(results_df)),
        "baseline_p50_ms": baseline.get("p50_ms"),
        "baseline_p95_ms": baseline.get("p95_ms"),
        "baseline_p99_ms": baseline.get("p99_ms"),
        "baseline_mean_ms": baseline.get("mean_ms"),
        "baseline_error_rate": baseline.get("error_rate"),
        "baseline_pct_over_1200ms": baseline.get("pct_over_1200ms"),
        "baseline_achieved_rps": baseline.get("achieved_rps"),
        "knee_rps": float(knee_df["knee_achieved_rps"].max()) if len(knee_df) and knee_df["knee_achieved_rps"].notna().any() else None,
        "first_breaking_rps": float(knee_df["first_breaking_rps"].min()) if len(knee_df) and knee_df["first_breaking_rps"].notna().any() else None,
        "recovered": RECOVERY.get("recovered"),
        "recovery_seconds": RECOVERY.get("recovery_seconds"),
        "search_rejections": int(telemetry_df["search_rejected_delta"].fillna(0).clip(lower=0).sum())
                             if len(telemetry_df) and "search_rejected_delta" in telemetry_df else None,
    }


def load_baseline_store():
    try:
        with open(BASELINE_STORE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return []


def save_baseline_store(store):
    with open(BASELINE_STORE, "w", encoding="utf-8") as handle:
        json.dump(store[-100:], handle, indent=2)


RUN_RECORD = build_run_record()
STORE = load_baseline_store()
PRIOR = [r for r in STORE if r.get("comparison_key") == COMPARISON_KEY]
PREVIOUS = PRIOR[-1] if PRIOR else None


def pct_change(current, previous):
    if current is None or previous in (None, 0):
        return None
    return round((current - previous) / previous, 4)


comparison_rows = []
findings = []
verdict = "pass"

if PREVIOUS is None:
    findings.append("No prior run exists for this configuration. This run is stored as the first baseline.")
else:
    tracked = [
        ("baseline_p50_ms", "lower_is_better"),
        ("baseline_p95_ms", "lower_is_better"),
        ("baseline_p99_ms", "lower_is_better"),
        ("baseline_error_rate", "lower_is_better"),
        ("baseline_pct_over_1200ms", "lower_is_better"),
        ("baseline_achieved_rps", "higher_is_better"),
        ("knee_rps", "higher_is_better"),
        ("recovery_seconds", "lower_is_better"),
    ]
    for metric, direction in tracked:
        current = RUN_RECORD.get(metric)
        previous = PREVIOUS.get(metric)
        change = pct_change(current, previous)
        if change is None:
            movement = "not comparable"
        elif abs(change) < 0.05:
            movement = "unchanged"
        elif (change > 0) == (direction == "higher_is_better"):
            movement = "improved"
        else:
            movement = "regressed"
        comparison_rows.append({
            "metric": metric,
            "previous": previous,
            "current": current,
            "pct_change": change,
            "movement": movement,
        })

    p95_row = next((r for r in comparison_rows if r["metric"] == "baseline_p95_ms"), None)
    if p95_row and p95_row["pct_change"] is not None and p95_row["pct_change"] > PASS_CRITERIA["regression_p95_pct_max"]:
        verdict = "fail"
        findings.append(
            f"Baseline p95 increased by {round(p95_row['pct_change'] * 100, 1)} percent against the prior run, "
            f"which exceeds the allowed {round(PASS_CRITERIA['regression_p95_pct_max'] * 100)} percent."
        )

    err_row = next((r for r in comparison_rows if r["metric"] == "baseline_error_rate"), None)
    if err_row and err_row["pct_change"] is not None and err_row["pct_change"] > PASS_CRITERIA["regression_error_rate_pct_max"]:
        verdict = "fail"
        findings.append("Baseline error rate increased beyond the allowed regression threshold.")

if RUN_RECORD.get("baseline_p95_ms") is not None and RUN_RECORD["baseline_p95_ms"] > PASS_CRITERIA["baseline_p95_ms_max"]:
    verdict = "fail"
    findings.append(
        f"Baseline p95 of {RUN_RECORD['baseline_p95_ms']} milliseconds exceeds the "
        f"{PASS_CRITERIA['baseline_p95_ms_max']} millisecond threshold."
    )

if RUN_RECORD.get("baseline_error_rate") is not None and RUN_RECORD["baseline_error_rate"] > PASS_CRITERIA["baseline_error_rate_max"]:
    verdict = "fail"
    findings.append("Baseline error rate exceeds the configured threshold.")

if RECOVERY.get("measured") and not RECOVERY.get("recovered"):
    verdict = "fail"
    findings.append("The stack did not return to baseline behavior within the recovery window.")
elif RECOVERY.get("recovery_seconds") is not None and RECOVERY["recovery_seconds"] > PASS_CRITERIA["recovery_seconds_max"]:
    verdict = "fail"
    findings.append(
        f"Recovery took {RECOVERY['recovery_seconds']} seconds, which exceeds the "
        f"{PASS_CRITERIA['recovery_seconds_max']} second threshold."
    )

if POOL_IS_SYNTHETIC:
    findings.append(
        "The query pool was synthetic. Latency figures are usable for iteration to iteration comparison "
        "but do not represent the production query mix."
    )

if "gateway" not in ACTIVE_TARGETS:
    findings.append(
        "The gateway path was not exercised. Every figure in this run is measured against the search "
        "endpoint directly and excludes the proxy and the search service. These numbers describe the "
        "search tier only and are not the limits a consuming system would experience."
    )

comparison_df = pd.DataFrame(comparison_rows)

STORE.append(RUN_RECORD)
save_baseline_store(STORE)

if not PIPELINE_MODE:
    print(f"\nVERDICT: {verdict.upper()}")
    for finding in findings:
        print(f"  - {finding}")
    if len(comparison_df):
        print("\nCOMPARISON AGAINST PRIOR RUN")
        print(comparison_df.to_string(index=False))

# COMMAND ----------

# =============================================================================
# CELL 15 - WORKBOOK OUTPUT
# =============================================================================

run_info = pd.DataFrame([
    {"field": "run_id", "value": RUN_ID},
    {"field": "run_label", "value": RUN_LABEL},
    {"field": "run_notes", "value": RUN_NOTES},
    {"field": "environment", "value": ENV},
    {"field": "targets", "value": ", ".join(sorted(ACTIVE_TARGETS.keys()))},
    {"field": "templates", "value": ", ".join(sorted(TEMPLATES.keys()))},
    {"field": "template_source", "value": TEMPLATE_PATH},
    {"field": "query_pool_source", "value": "synthetic" if POOL_IS_SYNTHETIC else AUDIT_LOG_TABLE},
    {"field": "query_pool_size", "value": len(QUERY_POOL)},
    {"field": "total_requests", "value": len(results_df)},
    {"field": "distributed_load", "value": DISTRIBUTED_LOAD},
    {"field": "client_retries", "value": CLIENT_RETRIES},
    {"field": "request_timeout_s", "value": REQUEST_TIMEOUT_S},
    {"field": "started_utc", "value": RUN_START_UTC.isoformat()},
    {"field": "finished_utc", "value": datetime.now(timezone.utc).isoformat()},
    {"field": "verdict", "value": verdict},
])

findings_df = pd.DataFrame({"finding": findings + ATTRIBUTION_NOTES}) if (findings or ATTRIBUTION_NOTES) \
    else pd.DataFrame({"finding": ["No findings recorded."]})

shape_mix_df = pd.DataFrame(
    [{"shape": k, "count": v, "share": round(v / len(QUERY_POOL), 4)} for k, v in SHAPE_MIX.most_common()]
)

output_path = f"{OUTPUT_DIR}/{RUN_ID}_performance.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    run_info.to_excel(writer, sheet_name="run_info", index=False)
    findings_df.to_excel(writer, sheet_name="findings", index=False)
    phase_summary.to_excel(writer, sheet_name="phase_summary", index=False)
    knee_df.to_excel(writer, sheet_name="knee", index=False)
    error_summary.to_excel(writer, sheet_name="error_taxonomy", index=False)
    shape_summary.to_excel(writer, sheet_name="by_query_shape", index=False)
    if len(consumer_summary):
        consumer_summary.to_excel(writer, sheet_name="by_consumer", index=False)
    if RECOVERY.get("measured"):
        RECOVERY["windows"].to_excel(writer, sheet_name="recovery", index=False)
    if len(comparison_df):
        comparison_df.to_excel(writer, sheet_name="vs_prior_run", index=False)
    if len(telemetry_df):
        telemetry_df.to_excel(writer, sheet_name="cluster_telemetry", index=False)
    shape_mix_df.to_excel(writer, sheet_name="query_pool_mix", index=False)
    slow_tail = results_df.sort_values("client_latency_ms", ascending=False).head(500)
    slow_tail.to_excel(writer, sheet_name="slowest_500", index=False)
    if len(failures):
        failures.head(2000).to_excel(writer, sheet_name="failure_detail", index=False)

log(f"Workbook written to {output_path}")

# COMMAND ----------

# =============================================================================
# CELL 16 - PIPELINE EXIT
# =============================================================================
#
# When run from a job, the notebook exits with a JSON payload so the calling
# pipeline can gate on the verdict. This is the hook that supports running
# the comparison automatically on every code change rather than on request.
# =============================================================================

exit_payload = {
    "run_id": RUN_ID,
    "verdict": verdict,
    "env": ENV,
    "baseline_p95_ms": RUN_RECORD.get("baseline_p95_ms"),
    "baseline_error_rate": RUN_RECORD.get("baseline_error_rate"),
    "baseline_pct_over_1200ms": RUN_RECORD.get("baseline_pct_over_1200ms"),
    "knee_rps": RUN_RECORD.get("knee_rps"),
    "first_breaking_rps": RUN_RECORD.get("first_breaking_rps"),
    "recovered": RUN_RECORD.get("recovered"),
    "recovery_seconds": RUN_RECORD.get("recovery_seconds"),
    "search_rejections": RUN_RECORD.get("search_rejections"),
    "workbook": output_path,
    "findings": findings,
}

print(json.dumps(exit_payload, indent=2, default=str))

if PIPELINE_MODE:
    dbutils.notebook.exit(json.dumps(exit_payload, default=str))

# COMMAND ----------

# =============================================================================
# RECOMMENDED IMPROVEMENTS
# =============================================================================
#
# These follow from the observations in the three sessions. They are ordered
# by how much they change the interpretation of results, not by effort.
#
# 1. RESOLVE THE LIVENESS PROBE BEFORE RESOLVING ANYTHING ELSE
#    The staging failure at roughly 55 requests per second was the search
#    service health endpoint failing to answer within the probe timeout,
#    which caused Kubernetes to restart the containers. Until that is
#    addressed, every stress result is measuring the probe, not the search
#    tier. Raising the timeout and failure threshold, and confirming the
#    health endpoint does no meaningful work, should precede any further
#    capacity discussion. Re-run this harness before and after the change
#    with the same RUN_LABEL prefix so the two runs compare directly.
#
# 2. SEPARATE THE HEALTH ENDPOINT FROM THE REQUEST THREAD POOL
#    A health check that shares a thread pool with request handling will
#    always fail first under load, regardless of the timeout value. If the
#    health endpoint is served from the same pool, moving it to a dedicated
#    path removes an entire class of false failure.
#
# 3. TEST THE FRONT DOOR, NOT ONLY THE SEARCH ENDPOINT
#    Current load tests hit the OpenSearch single-search endpoint directly.
#    Consumers reach the service through the proxy. A limit found against
#    the direct endpoint is not the limit consumers experience, and any
#    number reported externally should come from the front door path.
#    Enable the gateway target and run both in the same window.
#
# 4. RESOLVE AND DOCUMENT THE PROXY QUOTA IN BOTH ENVIRONMENTS
#    Staging carried a quota of 3600 requests per minute, which is 60 per
#    second. Production had no quota set, and the effective default is not
#    yet confirmed. Until the production default is documented, the
#    production ceiling is unknown rather than unlimited, and should be
#    described that way. Any deviation between staging and production
#    quotas belongs in the test plan before testing, not after.
#
# 5. REPORT AT PERCENTILES RATHER THAN AVERAGES
#    A 3.2 second average across 100 users describes almost nothing about
#    the experience of the slowest requests. The band above 1200
#    milliseconds that was flagged for reduction is a percentile question.
#    This harness reports p50 through p99 and the three latency bands so
#    the tail is visible.
#
# 6. TREAT OFFERED RATE AND ACHIEVED RATE AS DIFFERENT NUMBERS
#    A load generator that cannot sustain its own schedule reports a system
#    as healthier than it is. Schedule slip is captured here for that
#    reason. When slip is material, the achieved rate is the only valid
#    figure to report.
#
# 7. LOAD FROM PRODUCTION QUERY SHAPES, NOT A SINGLE PAYLOAD
#    Query cost is not uniform. Searches without a middle name execute a
#    structurally different query, because the clauses that bundle first,
#    middle and last name together are pruned when the middle name is
#    blank. A pool drawn from audit logs reflects that mix. A single
#    repeated payload does not, and will also benefit from caching that
#    production traffic does not receive.
#
# 8. CAPTURE SEARCH TIME ALONGSIDE CLIENT TIME
#    The reported search time and the client observed latency together
#    localize the delay. This is the difference between a conversation
#    about cluster sizing and a conversation about pod counts, and it is
#    cheap to collect.
#
# 9. WATCH THREAD POOL REJECTIONS AS THE SEARCH SATURATION SIGNAL
#    Rising search queue depth with rejections is the definitive marker
#    that the search tier is the constraint. Its absence during a failure
#    is equally informative, and was likely the case in the observed
#    staging failure.
#
# 10. RUN A RECOVERY PHASE EVERY TIME
#     Behavior after the load is removed was raised as an open question and
#     is not answered by a test that stops at the peak. Recovery time is
#     also the number that matters most operationally, since sustained
#     overload is rare and transient spikes are not.
#
# 11. DISABLE RETRIES DURING CHARACTERIZATION
#     Client and proxy retries convert one visible failure into several
#     invisible ones and inflate load at exactly the moment the system is
#     struggling. Whether a retry exists at the proxy layer is still open
#     and should be confirmed, since it changes how the error counts are
#     read.
#
# 12. SCALE THE CLUSTER AS A TESTED VARIABLE, NOT A REMEDY
#     Scaling staging took roughly thirty minutes, which makes it practical
#     to treat cluster class as one more variable in a comparison run
#     rather than as a change made in response to a failure. Run the same
#     profile at the current class and the candidate class and compare the
#     knee.
#
# 13. RUN THIS AS A PIPELINE ON EVERY PROMOTION
#     Set PIPELINE_MODE to True and schedule the notebook on code change
#     and on promotion. Each run stores a summary and compares against the
#     prior one, producing the change monitoring mechanism discussed rather
#     than a one time measurement.
#
# 14. KEEP THE SCOPE OF THIS OUTPUT NARROW
#     This harness reports throughput, latency, error behavior and
#     recovery. It makes no claim about result quality. Confidence grade
#     language about results should not be attached to these numbers
#     without business approval of that framing.
#
# =============================================================================
