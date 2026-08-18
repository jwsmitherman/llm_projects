# Databricks notebook source
import re, os, hashlib
import pandas as pd

FILE_A = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/BHUB.csv"
FILE_B = "/Workspace/Users/joshua.w.smitherman@uscis.dhs.gov/open_search/prod_logs/BHUB 1.csv"

START = re.compile(r'(?m)^[A-Z0-9_]+,[A-Z0-9_]+,CORE_SEARCH,')

def grab(pat, s, d=""):
    m = re.search(pat, s)
    return m.group(1) if m else d

def read(path):
    if not os.path.exists(path):
        print(f"MISSING {path}")
        return None
    raw = open(path, "rb").read()
    txt = raw.decode("utf-8", errors="replace")
    st = [m.start() for m in START.finditer(txt)]
    recs = [txt[s:(st[i+1] if i+1 < len(st) else len(txt))] for i, s in enumerate(st)]
    return {"path": path, "name": os.path.basename(path), "raw": raw, "txt": txt, "recs": recs,
            "bytes": len(raw), "lines": txt.count("\n") + 1,
            "core_search_mentions": txt.count("CORE_SEARCH"),
            "md5": hashlib.md5(raw).hexdigest()}

def parse(rec):
    consumer = grab(r'^([A-Z0-9_]+),', rec)
    rec = rec.replace('""', '"')
    i = rec.find('"result":')
    terms, result = (rec[:i], rec[i:]) if i >= 0 else (rec, "")
    mid = grab(r'"personMiddleName":(null|"[^"]*")', terms, "null")
    f = {"FIRST": grab(r'"personGivenName":"([^"]*)"', terms),
         "MIDDLE": "" if mid in ("null", "") else mid.strip('"'),
         "LAST": grab(r'"personSurName":"([^"]*)"', terms),
         "ANUM": grab(r'"type":"ALIEN_NBR","value":"([^"]*)"', terms),
         "RECEIPT": grab(r'"type":"RECEIPT_NBR","value":"([^"]*)"', terms),
         "DOB": grab(r'"dob":"(\d{4}-\d{2}-\d{2})"', terms).replace("-", "")}
    pid = grab(r'"identityId":"([0-9a-fA-F]{16,})"', result)
    return consumer, f, pid

A, B = read(FILE_A), read(FILE_B)
if A and B:
    print("FILE FACTS")
    facts = pd.DataFrame([
        {"file": A["name"], "bytes": A["bytes"], "lines": A["lines"],
         "records_matched": len(A["recs"]), "CORE_SEARCH_mentions": A["core_search_mentions"],
         "bytes_per_record": A["bytes"] // max(len(A["recs"]), 1), "md5": A["md5"]},
        {"file": B["name"], "bytes": B["bytes"], "lines": B["lines"],
         "records_matched": len(B["recs"]), "CORE_SEARCH_mentions": B["core_search_mentions"],
         "bytes_per_record": B["bytes"] // max(len(B["recs"]), 1), "md5": B["md5"]},
    ])
    print(facts.to_string(index=False))

    for d in (A, B):
        if d["core_search_mentions"] > len(d["recs"]):
            print(f"\nCHECK {d['name']}: CORE_SEARCH appears {d['core_search_mentions']} times but only "
                  f"{len(d['recs'])} records were captured. The extra mentions are either inside a record "
                  f"payload or are records that do not begin at the start of a line. Inspect before trusting "
                  f"the counts.")
        else:
            print(f"\nCHECK {d['name']}: every CORE_SEARCH occurrence was captured as a record "
                  f"({len(d['recs'])} of {d['core_search_mentions']}). No records are being skipped.")

    if A["md5"] == B["md5"]:
        print("\nTHE TWO FILES ARE BYTE FOR BYTE IDENTICAL. Same export saved under two names.")
    else:
        print("\nThe files differ in content.")

    for d in (A, B):
        if len(d["recs"]) < d["lines"] / 2:
            print(f"\nWARNING {d['name']}: {d['lines']:,} lines but only {len(d['recs']):,} records matched "
                  f"the CONSUMER,APP,CORE_SEARCH pattern. Most of this file is not being read, so any "
                  f"comparison below covers only the part that parsed.")

    print("\nFIRST 2 LINES OF EACH")
    for d in (A, B):
        print(f"\n--- {d['name']} ---")
        for line in d["txt"].split("\n")[:2]:
            print(line[:300])

    def keyset(d):
        out = {}
        for rec in d["recs"]:
            consumer, f, pid = parse(rec)
            k = (consumer, f["FIRST"], f["MIDDLE"], f["LAST"], f["ANUM"], f["RECEIPT"], f["DOB"])
            out.setdefault(k, {"count": 0, "pid": pid})
            out[k]["count"] += 1
        return out

    ka, kb = keyset(A), keyset(B)
    both = set(ka) & set(kb)
    only_a = set(ka) - set(kb)
    only_b = set(kb) - set(ka)

    print("\nSEARCH OVERLAP")
    print(pd.DataFrame([
        {"measure": f"records parsed in {A['name']}", "value": len(A["recs"])},
        {"measure": f"records parsed in {B['name']}", "value": len(B["recs"])},
        {"measure": f"distinct searches in {A['name']}", "value": len(ka)},
        {"measure": f"distinct searches in {B['name']}", "value": len(kb)},
        {"measure": "in both files", "value": len(both)},
        {"measure": f"only in {A['name']}", "value": len(only_a)},
        {"measure": f"only in {B['name']}", "value": len(only_b)},
    ]).to_string(index=False))

    if not only_a and not only_b:
        print("\nEVERY SEARCH IS IN BOTH FILES. The two files contain the same set of searches, "
              "so running both produces the same result twice.")
    else:
        cols = ["consumer", "first", "middle", "last", "anumber", "receipt", "dob", "times_in_file"]
        if only_a:
            print(f"\nSAMPLE OF SEARCHES ONLY IN {A['name']}")
            print(pd.DataFrame([list(k) + [ka[k]["count"]] for k in list(only_a)[:15]],
                               columns=cols).to_string(index=False))
        if only_b:
            print(f"\nSAMPLE OF SEARCHES ONLY IN {B['name']}")
            print(pd.DataFrame([list(k) + [kb[k]["count"]] for k in list(only_b)[:15]],
                               columns=cols).to_string(index=False))

    pa = {k: v["pid"] for k, v in ka.items() if v["pid"]}
    pb = {k: v["pid"] for k, v in kb.items() if v["pid"]}
    print(f"\nsearches with a logged identity: {A['name']} {len(pa)} of {len(ka)}, "
          f"{B['name']} {len(pb)} of {len(kb)}")
    conflict = [k for k in both if ka[k]["pid"] and kb[k]["pid"] and ka[k]["pid"] != kb[k]["pid"]]
    if conflict:
        print(f"{len(conflict)} searches appear in both files but with a different logged identity. "
              f"The two files were captured at different times or from different sources.")
