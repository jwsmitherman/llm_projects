# Databricks notebook source

# ============================================================================
# SEARCH ACCURACY TEST  —  driven by PRODUCTION LOGS  (one cell)
#
# HOW ACCURACY IS APPLIED (the key idea)
# --------------------------------------
# The six production-log files are SELF-LABELING. Every row contains BOTH:
#   - searchTerms  : the exact inputs a real consumer sent (name, A-number, DOB, ...)
#   - searchResult : the ranked identities production actually returned (with identityId + score)
# So the TOP-RANKED identity from production becomes the "expected identity" (a ground-truth
# silver label) pulled straight from real traffic -- no waiting on the Mars team to hand-label.
#
# The test replays each real input against the v7 query and checks whether that expected identity
# comes back:
#   - most consumers            -> HIT if it is in the TOP 10
#   - single-result consumers   -> HIT only if it is the #1 result (e.g. Crystal First / FIRST)
# Agreement = v7 reproduces the known-good production result (true positive).
# Disagreement = a regression to investigate (false negative).
#
# CONFIDENCE: a label is only as good as the search that produced it.
#   - HIGH  : a unique identifier was used AND production returned a small set (totalIdentities small)
#             -> the rank-1 identity is almost certainly the right person.
#   - LOW   : name-only search, or production returned the 10000 cap -> rank-1 is just "what prod
#             returned", not verified. These are reported separately and excluded from the headline.
# Use the HIGH-confidence hit rate as the headline number; treat LOW-confidence as directional until
# the Mars team confirms expected results.
# ============================================================================
import requests, json, re, os, glob
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- CONFIG -------------------------------------------
ENDPOINT   = "https://opensearch-identity-prod.pcis.uscis.dhs.gov/iis-identity-api-alias/_search"
AUTH_TOKEN = "PASTE_BASE64_TOKEN_HERE"
TEMPLATE_FILE = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/query_templates/search-templatev7.txt"
PROD_LOGS_DIR = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs"
RESULTS_DIR   = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/results"  # Excel written here
ID_FIELD   = "identityId"
TOP_N      = 10
SINGLE_RESULT_CONSUMERS = {"FIRST"}     # take only the #1 result (Crystal First, etc.)
HIGH_CONF_MAX_TOTAL = 5                 # "small result set" threshold for a high-confidence label
VERIFY_TLS = True

auth = AUTH_TOKEN if AUTH_TOKEN.startswith("Basic ") else "Basic " + AUTH_TOKEN
HEADERS = {"Content-Type": "application/json", "Authorization": auth}
CONFIGURED = AUTH_TOKEN != "PASTE_BASE64_TOKEN_HERE"

# ------------------------- LOAD + PARSE PRODUCTION LOGS ---------------------
# Robust to the CSV corruption in these files (embedded newlines, split columns, doubled quotes):
# split on each record header, normalize quotes, then regex-extract only the fields we need.
_START = re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')
def _g(pat, s, d="", grp=1):
    m = re.search(pat, s); return m.group(grp) if m else d

def parse_record(rec):
    consumer = _g(r'^([A-Z0-9_]+),', rec)
    rec = rec.replace('""', '"')                       # undo doubled-quote escaping
    i = rec.find('"result":')                          # split input vs output
    terms  = rec[:i] if i >= 0 else rec
    result = rec[i:] if i >= 0 else ""
    mid = _g(r'"personMiddleName":(null|"[^"]*")', terms, "null")
    fields = {
        "FIRSTNAME": _g(r'"personGivenName":"([^"]*)"', terms),
        "MIDDLENAME": "" if mid in ("null","") else mid.strip('"'),
        "LASTNAME":  _g(r'"personSurName":"([^"]*)"', terms),
        "ANUMBER":   _g(r'"type":"ALIEN_NBR","value":"([^"]*)"', terms),
        "RECEIPT":   _g(r'"type":"RECEIPT_NBR","value":"([^"]*)"', terms),
        "DOB":      (_g(r'"dob":"(\d{4}-\d{2}-\d{2})"', terms) or "").replace("-",""),
        "COB":       _g(r'"cobs":\["([^"]*)"\]', terms),
        "COC":       _g(r'"cocs":\["([^"]*)"\]', terms),
    }
    total    = _g(r'"totalIdentities":(\d+)', result, None)
    total    = int(total) if total else None
    expected = _g(r'"identityId":"([0-9a-fA-F]{16,})"', result, None)   # first hit = rank 1
    strong   = bool(fields["ANUMBER"] or fields["RECEIPT"])
    conf = ("high" if (strong and total is not None and total <= HIGH_CONF_MAX_TOTAL)
            else "low" if (total == 10000 or not strong) else "medium")
    # reproducible with the v7 name/ID template? (receipt-only queries the template can't build)
    reproducible = any(fields[k] for k in ("FIRSTNAME","LASTNAME","ANUMBER","DOB","COB","COC"))
    mode = "top1" if consumer in SINGLE_RESULT_CONSUMERS else "top10"
    return {"consumer":consumer, "mode":mode, "expected_id":expected, "fields":fields,
            "total_identities":total, "confidence":conf, "reproducible":reproducible}

def load_cases_from_prod_logs(folder):
    cases=[]
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    for path in files:
        txt = open(path, encoding="utf-8", errors="replace").read()
        starts = [m.start() for m in _START.finditer(txt)]
        recs = [txt[s:(starts[i+1] if i+1 < len(starts) else len(txt))] for i,s in enumerate(starts)]
        for r in recs:
            c = parse_record(r)
            if c["expected_id"]:            # keep only rows that have a production label
                cases.append(c)
        print(f"  {os.path.basename(path)}: {len(recs)} record(s)")
    return cases

# ------------------------- QUERY BUILD (v7 template) ------------------------
PH = re.compile(r"\{\{\s*([A-Z_]+)\s*\}\}")
def _has_ph(n):
    if isinstance(n,str): return bool(PH.search(n))
    if isinstance(n,list): return any(_has_ph(x) for x in n)
    if isinstance(n,dict): return any(_has_ph(v) for v in n.values())
    return False
def _empty_bool(n):
    b=n.get("bool") if isinstance(n,dict) else None
    return isinstance(b,dict) and not any(isinstance(b.get(k),list) and b[k] for k in ("must","should","filter","must_not"))
def _prune(n):
    if isinstance(n,dict) and isinstance(n.get("bool"),dict):
        b=n["bool"]
        for key in ("must","should"):
            if isinstance(b.get(key),list):
                for c in b[key]: _prune(c)
                b[key]=[c for c in b[key] if not _has_ph(c) and not _empty_bool(c)]
                if not b[key]: del b[key]
    for v in (n.values() if isinstance(n,dict) else n if isinstance(n,list) else []):
        if isinstance(v,(dict,list)): _prune(v)
def _strip(n):
    if isinstance(n,dict):
        for k,v in list(n.items()):
            if isinstance(v,str): n[k]=PH.sub("",v)
            elif isinstance(v,(dict,list)): _strip(v)
    elif isinstance(n,list):
        for i,v in enumerate(n):
            if isinstance(v,str): n[i]=PH.sub("",v)
            elif isinstance(v,(dict,list)): _strip(v)

try:
    TEMPLATE_TEXT = open(TEMPLATE_FILE).read()
except Exception as e:
    TEMPLATE_TEXT = None
    print(f"Template not readable ({e}); using a simple fallback query. Point TEMPLATE_FILE at the real v7 template for true numbers.")

def build_query(fields):
    p = {k:("" if v is None else str(v)) for k,v in fields.items()}
    if TEMPLATE_TEXT:
        s = PH.sub(lambda m: (p[m.group(1)] if p.get(m.group(1)) else m.group(0)), TEMPLATE_TEXT)
        s = re.sub(r'"size"\s*:\s*"(\d+)"', r'"size": \1', s)
        if s.lstrip()[:1] != "{": s = "{"+s+"}"
        s = re.sub(r",(\s*[}\]])", r"\1", s)
        q = json.loads(s); _prune(q); _strip(q); q["size"]=max(TOP_N,10); return q
    should=[]
    if p.get("FIRSTNAME"): should.append({"match":{"biographicInfo.name.first":p["FIRSTNAME"]}})
    if p.get("LASTNAME"):  should.append({"match":{"biographicInfo.name.last":p["LASTNAME"]}})
    if p.get("DOB"):       should.append({"term":{"_search.dateOfBirth":p["DOB"]}})
    if p.get("ANUMBER"):   should.append({"match":{"_search.identifiers.ALIEN_NBR":{"query":p["ANUMBER"],"fuzziness":2}}})
    if p.get("RECEIPT"):   should.append({"match":{"_search.identifiers.RECEIPT_NBR":p["RECEIPT"]}})
    return {"size":max(TOP_N,10),"query":{"bool":{"should":should or [{"match_all":{}}],"minimum_should_match":1}}}

def search_ids(fields):
    r = requests.post(ENDPOINT, headers=HEADERS, json=build_query(fields), verify=VERIFY_TLS, timeout=120)
    if r.status_code>=400:
        print("STATUS",r.status_code,r.text[:300]); r.raise_for_status()
    out=[]
    for h in r.json().get("hits",{}).get("hits",[]):
        src=h.get("_source",{}); v=src.get(ID_FIELD)
        if v is None: v=(src.get("biographicInfo",{}) or {}).get(ID_FIELD)
        out.append(str(v) if v is not None else h.get("_id"))
    return out

# ------------------------- EVALUATE -----------------------------------------
def evaluate(cases):
    rows=[]
    for i,c in enumerate(cases):
        try: top = search_ids(c["fields"])
        except Exception as e: top=[]; print(f"case {i} ({c.get('consumer')}) error: {e}")
        exp=str(c["expected_id"]); k=1 if c["mode"]=="top1" else TOP_N
        rank=(top.index(exp)+1) if exp in top else None
        hit=(rank is not None and rank<=k)
        rows.append({"consumer":c["consumer"],"mode":c["mode"],"confidence":c["confidence"],
                     "reproducible":c["reproducible"],"total_identities":c["total_identities"],
                     "expected_id":exp,"found_rank":rank,"hit":hit,
                     "outcome":("TP" if hit else "FN")})
    return pd.DataFrame(rows)

def summarize(df):
    rows=[]
    for grp,sub in list(df.groupby("consumer"))+[("OVERALL",df)]:
        n=len(sub); hits=int(sub["hit"].sum())
        ranks=sub.loc[sub["hit"],"found_rank"].dropna()
        rows.append({"consumer":grp,"queries":n,"hits":hits,"misses":n-hits,
                     "hit_rate_pct":round(100*hits/n,1) if n else None,
                     "rank1_pct":round(100*(ranks<=1).sum()/n,1) if n else None,
                     "avg_rank_when_found":round(ranks.mean(),1) if len(ranks) else None})
    return pd.DataFrame(rows)

def charts(df, summary):
    sc=summary[summary["consumer"]!="OVERALL"]; overall=summary.set_index("consumer").loc["OVERALL","hit_rate_pct"]
    plt.rcParams["figure.figsize"]=(9,4.5); plt.rcParams["axes.grid"]=True; plt.rcParams["grid.alpha"]=0.3
    fig,ax=plt.subplots()
    b=ax.bar(sc["consumer"],sc["hit_rate_pct"],color="#4C78A8")
    ax.axhline(overall,color="#C0392B",ls="--",label=f"overall {overall} pct")
    for r,v in zip(b,sc["hit_rate_pct"]): ax.text(r.get_x()+r.get_width()/2,v,f"{v}",ha="center",va="bottom")
    ax.set_ylim(0,100); ax.set_ylabel("hit rate (percent)"); ax.set_title("v7 accuracy by consumer (vs production label)"); ax.legend()
    plt.tight_layout(); plt.show()
    fig,ax=plt.subplots()
    ax.bar(sc["consumer"],sc["hits"],label="hits",color="#54A24B")
    ax.bar(sc["consumer"],sc["misses"],bottom=sc["hits"],label="misses",color="#E45756")
    ax.set_ylabel("queries"); ax.set_title("Hits and misses by consumer"); ax.legend()
    plt.tight_layout(); plt.show()
    ranks=df.loc[df["hit"],"found_rank"].dropna().astype(int)
    if len(ranks):
        fig,ax=plt.subplots()
        ax.hist(ranks,bins=range(1,TOP_N+2),color="#72B7B2",align="left",rwidth=0.8)
        ax.set_xticks(range(1,TOP_N+1)); ax.set_xlabel("rank of the correct identity"); ax.set_ylabel("queries")
        ax.set_title("Where the correct identity landed"); plt.tight_layout(); plt.show()

def write_excel(results, summary, conf_table, out_dir, cases):
    """Write the ACTUAL run results (accuracy measured against production labels) to an .xlsx."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "PCIS_Search_Accuracy_Results.xlsx")

    # per-log findings: what each log row is (input, prod result, confidence, why) — the context
    def qtype(f):
        parts=[]
        if f["FIRSTNAME"] or f["LASTNAME"]: parts.append("name")
        if f["ANUMBER"]: parts.append("A-number")
        if f["RECEIPT"]: parts.append("receipt")
        if f["DOB"]: parts.append("DOB")
        if f["COB"]: parts.append("COB")
        return " + ".join(parts) if parts else "none"
    def why(c):
        if c["reproducible"] is False: return "Receipt-only search; v7 name/ID template has no receipt clause"
        if c["total_identities"]==10000: return "Name search hit the 10000 result cap; label is low-confidence"
        if c["total_identities"]==1: return "Unique result; high-confidence label"
        return "Small result set"
    findings=[{"consumer":c["consumer"],"query_type":qtype(c["fields"]),
               "name":" ".join(x for x in [c["fields"]["FIRSTNAME"],c["fields"]["MIDDLENAME"],c["fields"]["LASTNAME"]] if x),
               "A-number":c["fields"]["ANUMBER"],"receipt":c["fields"]["RECEIPT"],
               "DOB":c["fields"]["DOB"],"COB":c["fields"]["COB"],
               "prod_result_count":c["total_identities"],
               "expected_identity_rank1":c["expected_id"],
               "confidence":c["confidence"],"reproducible":("Yes" if c["reproducible"] else "No"),
               "finding":why(c)} for c in cases]
    findings_df=pd.DataFrame(findings)

    # rollup summary of the logs (aggregated, built from the parsed cases)
    overall=pd.DataFrame({"metric":["Labeled cases","Consumers represented","Reproducible with v7",
                                     "Not reproducible (receipt-only, etc.)","High confidence",
                                     "Medium confidence","Low confidence","Hit the 10000 result cap"],
                          "value":[len(findings_df),
                                   findings_df["consumer"].nunique(),
                                   int((findings_df["reproducible"]=="Yes").sum()),
                                   int((findings_df["reproducible"]=="No").sum()),
                                   int((findings_df["confidence"]=="high").sum()),
                                   int((findings_df["confidence"]=="medium").sum()),
                                   int((findings_df["confidence"]=="low").sum()),
                                   int((findings_df["prod_result_count"]==10000).sum())]})
    by_consumer=(findings_df.groupby("consumer").size().reset_index(name="cases").sort_values("cases",ascending=False))
    by_conf=(findings_df.groupby("confidence").size().reset_index(name="cases"))
    by_qtype=(findings_df.groupby("query_type").size().reset_index(name="cases").sort_values("cases",ascending=False))

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        # 0) LOGS SUMMARY: aggregated rollup of the logs (stacked sections on one sheet)
        overall.to_excel(xl, sheet_name="Logs summary", index=False, startrow=1)
        sh=xl.sheets["Logs summary"]; sh["A1"]="Overall"
        r=len(overall)+4; sh.cell(row=r,column=1,value="By consumer")
        by_consumer.to_excel(xl, sheet_name="Logs summary", index=False, startrow=r)
        r=r+len(by_consumer)+3; sh.cell(row=r,column=1,value="By confidence")
        by_conf.to_excel(xl, sheet_name="Logs summary", index=False, startrow=r)
        r=r+len(by_conf)+3; sh.cell(row=r,column=1,value="By query type")
        by_qtype.to_excel(xl, sheet_name="Logs summary", index=False, startrow=r)
        # 1) what each log row is: input, prod result, confidence, and the finding
        findings_df.to_excel(xl, sheet_name="Findings by log", index=False)
        # 2) accuracy by consumer (measured this run)
        summary.to_excel(xl, sheet_name="Accuracy by consumer", index=False)
        # 3) accuracy by confidence (trust the high row)
        conf_table.to_excel(xl, sheet_name="Accuracy by confidence", index=False)
        # 4) confusion counts per consumer + overall
        rowsm=[]
        for grp,sub in list(results.groupby("consumer"))+[("OVERALL",results)]:
            v=sub["outcome"].value_counts().to_dict()
            rowsm.append({"consumer":grp,"TP (hit)":v.get("TP",0),"FN (miss)":v.get("FN",0),
                          "queries":len(sub),
                          "hit_rate_pct":round(100*v.get("TP",0)/len(sub),1) if len(sub) else None})
        pd.DataFrame(rowsm).to_excel(xl, sheet_name="Confusion", index=False)
        # 5) full per-query detail (every replayed search and whether v7 returned the label)
        results.to_excel(xl, sheet_name="Per-query detail", index=False)
        # 6) run metadata so a reader knows what produced these numbers
        meta=pd.DataFrame({"field":["endpoint","template_file","prod_logs_dir","top_n",
                                    "single_result_consumers","cases_run","overall_hit_rate_pct"],
                           "value":[ENDPOINT, TEMPLATE_FILE, PROD_LOGS_DIR, TOP_N,
                                    ", ".join(sorted(SINGLE_RESULT_CONSUMERS)), len(results),
                                    round(100*results["hit"].mean(),1) if len(results) else None]})
        meta.to_excel(xl, sheet_name="Run info", index=False)
    print(f"\nExcel written: {path}")
    return path

# ------------------------- RUN ----------------------------------------------
if not CONFIGURED:
    print("Set AUTH_TOKEN (and TEMPLATE_FILE / PROD_LOGS_DIR) in CONFIG, then re-run.")
else:
    print("Loading production-log cases:")
    cases = load_cases_from_prod_logs(PROD_LOGS_DIR)
    total = len(cases)
    repro = [c for c in cases if c["reproducible"]]
    skipped = total - len(repro)
    print(f"\n{total} labeled cases; {len(repro)} reproducible with the v7 template; "
          f"{skipped} skipped (e.g. receipt-only, which the name/ID template can't build).\n")

    results = evaluate(repro)

    print("SUMMARY: v7 accuracy by consumer (measured against the production label)")
    summary = summarize(results)
    display(summary)

    print("\nHEADLINE by confidence (use HIGH as the trustworthy number)")
    conf = (results.groupby("confidence")
                   .agg(queries=("hit","size"), hits=("hit","sum"))
                   .assign(hit_rate_pct=lambda d: (100*d["hits"]/d["queries"]).round(1))
                   .reset_index())
    display(conf)

    print("\nPER-QUERY DETAIL")
    display(results)

    charts(results, summary)

    # write the ACTUAL results (only exists now, after running against the cluster) to Excel
    write_excel(results, summary, conf, RESULTS_DIR, repro)

    print("\nNOTE: labels come from what production returned, not a verified source of truth. "
          "HIGH-confidence rows (unique-ID search, small result set) are reliable; LOW-confidence "
          "rows (name-only or 10000-cap) are directional until the Mars team confirms expected results.")
