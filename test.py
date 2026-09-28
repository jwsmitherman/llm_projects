import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
from IPython.display import display

REPO_ROOT = Path("/Users/jwsmithe/Documents/updated_pcs_repo/python312_version/pcs-model-training")
MODEL_PATH = REPO_ROOT / "pcs_models/models/dedup_model/dedup_model.onnx"
THRESHOLD = 0.5
FEATURIZER_NAME = "get_features"
OUTPUT_DIR = Path("iden_45868_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pcs_models.manager.schemas.dedup import DedupCandidate, DedupRow


def main():
    print("\n=== Model ===")

    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    FEATURES = list(DedupRow.get_type_mappings().keys())
    FIELDS = list(DedupCandidate.model_fields)

    for i in sess.get_inputs():
        print("input ", i.name, i.type, i.shape)
    for o in sess.get_outputs():
        print("output", o.name, o.type, o.shape)

    print([m for m in vars(DedupRow) if not m.startswith("_")])

    def featurize(a, b):
        out = getattr(DedupRow(declared=a, candidate=b), FEATURIZER_NAME)()
        if isinstance(out, pd.DataFrame):
            out = out.iloc[0]
        return dict(out)

    def get_proba(outputs):
        probs = outputs[-1]
        if isinstance(probs, list):
            key = 1 if 1 in probs[0] else "1"
            return np.array([p[key] for p in probs])
        probs = np.asarray(probs)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.ravel()

    def score(pairs):
        if not pairs:
            return np.array([])
        df = pd.DataFrame([featurize(a, b) for a, b in pairs])
        inputs = sess.get_inputs()
        if len(inputs) == 1:
            feed = {inputs[0].name: df[FEATURES].astype(np.float32).to_numpy()}
        else:
            feed = {i.name: df[i.name].astype(np.float32).to_numpy().reshape(-1, 1) for i in inputs}
        return get_proba(sess.run(None, feed))

    print("\n=== Data ===")

    data = pd.DataFrame(DATA["rows"], columns=DATA["columns"])

    SPONSOR_MAP = {
        "sponsor_record_id": "record_id",
        "sponsor_first_name": "firstName",
        "sponsor_middle_name": "middleName",
        "sponsor_last_name": "lastName",
        "sponsor_full_name": "fullName",
        "sponsor_dob": "dateOfBirth",
        "sponsor_gender": "genderCd",
        "sponsor_a_number": "aNumber",
        "sponsor_ssn": "ssn",
        "sponsor_country_of_birth": "countryOfBirth",
        "sponsor_country_of_citizenship": "countryOfCitizenship",
        "sponsor_phone": "phoneNumber",
        "sponsor_email": "emailAddress",
        "sponsor_receipt_number": "receiptNumber",
        "sponsor_status": "status",
    }

    beneficiaries = data.drop(columns=list(SPONSOR_MAP)).rename(columns={"beneficiary_id": "record_id"})
    beneficiaries["role"] = "beneficiary"

    sponsors = data[list(SPONSOR_MAP)].drop_duplicates("sponsor_record_id").rename(columns=SPONSOR_MAP)
    sponsors["role"] = "sponsor"

    records = pd.concat([beneficiaries, sponsors], ignore_index=True)

    links = data[["sponsor_record_id", "beneficiary_id", "sponsor_relationship"]]
    links = links.rename(columns={"sponsor_record_id": "sponsor_id", "sponsor_relationship": "relationship"})

    print(records["role"].value_counts())
    print(len(links), "links")

    print("\n=== Sponsor input check: does the model accept sponsor fields? ===")

    SPONSOR_TERMS = ["sponsor", "relationship", "petitioner", "affidavit"]

    def sponsor_names(names):
        return [n for n in names if any(t in n.lower() for t in SPONSOR_TERMS)]

    onnx_inputs = [i.name for i in sess.get_inputs()]
    onnx_width = sess.get_inputs()[0].shape[-1] if len(onnx_inputs) == 1 else len(onnx_inputs)

    input_check = pd.DataFrame([
        {"source": "ONNX input names", "count": len(onnx_inputs), "sponsor_related": sponsor_names(onnx_inputs)},
        {"source": "Model features (get_type_mappings)", "count": len(FEATURES), "sponsor_related": sponsor_names(FEATURES)},
        {"source": "DedupCandidate fields", "count": len(FIELDS), "sponsor_related": sponsor_names(FIELDS)},
        {"source": "DedupRow fields", "count": len(DedupRow.model_fields), "sponsor_related": sponsor_names(DedupRow.model_fields)},
    ])
    print("ONNX input width:", onnx_width, "| feature count:", len(FEATURES))
    display(input_check)

    extra_setting = DedupCandidate.model_config.get("extra", "ignore")
    try:
        test = DedupCandidate(firstName="TEST", sponsor_a_number="A012345678", sponsor_relationship="CHILD")
        kept = test.model_extra or {}
        extra_result = "kept but unused" if kept else "silently dropped"
    except Exception:
        extra_result = "rejected"

    print("DedupCandidate extra setting:", extra_setting)
    print("Sponsor fields passed to DedupCandidate are:", extra_result)

    rows = data.head(300).to_dict("records")
    shuffled = data.sample(len(rows), random_state=1).to_dict("records")

    def beneficiary(r):
        return {k: r[k] for k in FIELDS if k in r and pd.notna(r[k])}

    def sponsor_extras(r):
        return {k: r[k] for k in SPONSOR_MAP if pd.notna(r[k])}

    plain, attached, swapped = [], [], []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        try:
            plain.append((DedupCandidate(**beneficiary(a)), DedupCandidate(**beneficiary(b))))
            attached.append((DedupCandidate(**beneficiary(a), **sponsor_extras(a)),
                             DedupCandidate(**beneficiary(b), **sponsor_extras(b))))
            swapped.append((DedupCandidate(**beneficiary(a), **sponsor_extras(shuffled[i])),
                            DedupCandidate(**beneficiary(b), **sponsor_extras(shuffled[i + 1]))))
        except Exception:
            continue

    if attached:
        s_plain = score(plain)
        s_attached = score(attached)
        s_swapped = score(swapped)
        max_diff = max(np.abs(s_plain - s_attached).max(), np.abs(s_plain - s_swapped).max())
    else:
        max_diff = 0.0

    print("pairs tested:", len(attached))
    print("max score change from adding/swapping sponsor fields:", max_diff)

    sponsor_in_inputs = sum(len(x) for x in input_check["sponsor_related"])
    confirmed = sponsor_in_inputs == 0 and max_diff == 0

    print("Sponsor-related model inputs:", sponsor_in_inputs)
    print("Sponsor fields on DedupCandidate:", extra_result)
    print("Score change from sponsor fields:", max_diff)
    print()
    print("CONFIRMED: model does not accept sponsor information" if confirmed
          else "NOT CONFIRMED: review the rows above")

    print("\n=== Test 1: Schema validation and fill rates ===")

    cands = {}
    errors = []
    for rec in records.to_dict("records"):
        data = {k: rec[k] for k in FIELDS if k in rec and pd.notna(rec[k])}
        try:
            cands[rec["record_id"]] = DedupCandidate(**data)
        except Exception as e:
            for err in e.errors():
                errors.append({"record_id": rec["record_id"], "role": rec["role"], "field": err["loc"][0]})

    records["valid"] = records["record_id"].isin(cands)
    display(records.groupby("role")["valid"].mean())
    display(records[records.role == "sponsor"].groupby("status")["valid"].mean())

    errors = pd.DataFrame(errors, columns=["record_id", "role", "field"])
    display(errors.groupby(["role", "field"]).size())

    cols = [f for f in FIELDS if f in records]
    fill = records.groupby("role")[cols].apply(lambda d: d.notna().mean()).T
    fill.to_csv(OUTPUT_DIR / "t1_fill_rates.csv")
    display(fill)

    print("\n=== Sanity check ===")

    sample = list(cands.values())[:100]
    same = score([(c, c) for c in sample])
    diff = score([(sample[i], sample[i - 1]) for i in range(len(sample))])
    print("identical:", same.mean().round(3))
    print("different:", diff.mean().round(3))

    print("\n=== Test 2: Sponsor vs own beneficiaries (should not match) ===")

    def same(a, b, field):
        x = getattr(a, field)
        y = getattr(b, field)
        return bool(x) and bool(y) and str(x).upper() == str(y).upper()

    t2 = links[links["sponsor_id"].isin(cands) & links["beneficiary_id"].isin(cands)].copy()
    pairs = [(cands[s], cands[b]) for s, b in zip(t2["sponsor_id"], t2["beneficiary_id"])]

    SHARED = ["lastName", "firstName", "dateOfBirth", "phoneNumber", "emailAddress", "receiptNumber"]
    for f in SHARED:
        t2["same_" + f] = [same(a, b, f) for a, b in pairs]

    t2["score"] = score(pairs)
    t2["match"] = t2["score"] >= THRESHOLD

    print("false merge rate:", round(t2["match"].mean(), 4))
    t2.to_csv(OUTPUT_DIR / "t2_sponsor_beneficiary.csv", index=False)

    rows = []
    for f in SHARED:
        g = t2[t2["same_" + f]]
        rows.append({"shared_field": f, "n": len(g), "mean_score": g["score"].mean(), "false_merge_rate": g["match"].mean()})
    display(pd.DataFrame(rows))

    display(t2.groupby("relationship")[["score", "match"]].mean())

    display(t2.sort_values("score", ascending=False)[["sponsor_id", "beneficiary_id", "relationship", "score"]].head(25))

    print("\n=== Test 3: Same sponsor across records (should match) ===")

    sponsors = records[(records["role"] == "sponsor") & records["valid"]]

    pos = set()
    for key in ["ssn", "aNumber"]:
        for _, g in sponsors.dropna(subset=[key]).groupby(key):
            ids = sorted(g["record_id"])
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pos.add((ids[i], ids[j]))

    neg = []
    for _, g in sponsors.groupby("lastName"):
        ids = list(g["record_id"])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = cands[ids[i]], cands[ids[j]]
                if not same(a, b, "ssn") and not same(a, b, "aNumber"):
                    neg.append((ids[i], ids[j]))
    neg = neg[:5000]

    t3 = pd.DataFrame(list(pos) + neg, columns=["id_a", "id_b"])
    t3["label"] = [1] * len(pos) + [0] * len(neg)
    pairs3 = [(cands[a], cands[b]) for a, b in zip(t3["id_a"], t3["id_b"])]

    t3["score"] = score(pairs3)
    no_ids = lambda c: c.model_copy(update={"ssn": None, "aNumber": None})
    t3["score_no_ids"] = score([(no_ids(a), no_ids(b)) for a, b in pairs3])

    for col in ["score", "score_no_ids"]:
        print(col)
        print("  recall:", (t3[t3.label == 1][col] >= THRESHOLD).mean())
        print("  false match rate:", (t3[t3.label == 0][col] >= THRESHOLD).mean())

    t3.to_csv(OUTPUT_DIR / "t3_sponsor_duplicates.csv", index=False)

    print("\n=== Test 4: Copy sponsor field onto beneficiary ===")

    sample_pairs = pairs[:500]
    base = score(sample_pairs)

    rows = []
    for f in ["lastName", "firstName", "dateOfBirth", "phoneNumber", "emailAddress", "receiptNumber", "ssn", "aNumber"]:
        new_pairs = []
        for s, b in sample_pairs:
            b2 = b.model_copy(update={f: getattr(s, f)})
            b2 = b2.model_copy(update={"fullName": " ".join(x for x in [b2.firstName, b2.middleName, b2.lastName] if x)})
            new_pairs.append((s, b2))
        new = score(new_pairs)
        rows.append({
            "field": f,
            "mean_change": (new - base).mean(),
            "pushed_over_threshold": ((base < THRESHOLD) & (new >= THRESHOLD)).mean(),
        })

    t4 = pd.DataFrame(rows).sort_values("mean_change", ascending=False)
    t4.to_csv(OUTPUT_DIR / "t4_field_copy.csv", index=False)
    display(t4)

    print("\n=== Test 5: Sponsor linking beneficiaries together ===")

    def groups(nodes, edges):
        parent = {n: n for n in nodes}
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for a, b in edges:
            parent[find(a)] = find(b)
        return {n: find(n) for n in nodes}

    t2_scores = dict(zip(zip(t2["sponsor_id"], t2["beneficiary_id"]), t2["score"]))
    households = t2.groupby("sponsor_id")["beneficiary_id"].apply(list)
    households = households[households.map(len) >= 2].head(300)

    rows = []
    for sid, bens in households.items():
        bens = bens[:20]
        bb = [(bens[i], bens[j]) for i in range(len(bens)) for j in range(i + 1, len(bens))]
        bb_scores = score([(cands[a], cands[b]) for a, b in bb])
        bb_edges = [p for p, s in zip(bb, bb_scores) if s >= THRESHOLD]
        sponsor_edges = [(sid, b) for b in bens if t2_scores[(sid, b)] >= THRESHOLD]

        without = groups(bens, bb_edges)
        with_sponsor = groups(bens + [sid], bb_edges + sponsor_edges)
        chained = sum(1 for a, b in bb if with_sponsor[a] == with_sponsor[b] and without[a] != without[b])

        rows.append({"sponsor_id": sid, "n_beneficiaries": len(bens),
                     "sponsor_matches": len(sponsor_edges), "chained_pairs": chained})

    t5 = pd.DataFrame(rows)
    print("households with sponsor match:", (t5["sponsor_matches"] > 0).mean())
    print("households with chained pairs:", (t5["chained_pairs"] > 0).mean())
    t5.to_csv(OUTPUT_DIR / "t5_chaining.csv", index=False)

    print("\n=== Summary ===")

    summary = pd.DataFrame([
        ["sponsor-related model inputs", sponsor_in_inputs],
        ["score change from sponsor fields", max_diff],
        ["sponsor valid rate", records[records.role == "sponsor"]["valid"].mean()],
        ["beneficiary valid rate", records[records.role == "beneficiary"]["valid"].mean()],
        ["sponsor-beneficiary false merge rate", t2["match"].mean()],
        ["sponsor duplicate recall", (t3[t3.label == 1]["score"] >= THRESHOLD).mean()],
        ["sponsor duplicate recall without ids", (t3[t3.label == 1]["score_no_ids"] >= THRESHOLD).mean()],
        ["same surname false match rate", (t3[t3.label == 0]["score"] >= THRESHOLD).mean()],
        ["households with chained pairs", (t5["chained_pairs"] > 0).mean()],
    ], columns=["metric", "value"])

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    display(summary)

    return {"summary": summary, "records": records, "t2": t2, "t3": t3, "t4": t4, "t5": t5}


DATA = json.loads(r"""
{
    "columns": ["beneficiary_id", "receiptNumber", "form_type", "firstName", "middleName", "lastName", "fullName", "dateOfBirth", "genderCd", "aNumber", "countryOfBirth", "countryOfCitizenship", "ssn", "fin", "phoneNumber", "emailAddress", "parentName", "sponsor_record_id", "sponsor_relationship", "sponsor_receipt_number", "sponsor_first_name", "sponsor_middle_name", "sponsor_last_name", "sponsor_full_name", "sponsor_dob", "sponsor_gender", "sponsor_status", "sponsor_a_number", "sponsor_ssn", "sponsor_country_of_birth", "sponsor_country_of_citizenship", "sponsor_phone", "sponsor_email", "sponsor_address", "sponsor_city", "sponsor_state", "sponsor_zip"],
    "rows": [
        ["B00001_00", "IOE2905302051", "I-134", "FATIMA", null, "LOPEZ", "FATIMA LOPEZ", "2011-12-13", "F", "A023350813", "VNM", "VNM", null, "32090335", "4347141113", null, "HUGO m lopez", "S00001", "CHILD", "IOE2905302051", "HUGO", "m", "lopez", "HUGO m lopez", "1962-08-21", "F", "LPR", "A072926275", "521-41-1025", "CUB", "CUB", "4347141113", "hugo.lopez11@example.com", "881 OAK AVE", "LOS ANGELES", "CA", "90018"],
        ["B00001_01", "IOE2905302051", "I-134", "hugo", "m", "lopez", "hugo m lopez", "2007-08-06", "M", "A053017486", "GTM", "GTM", null, null, "2295542744", "hugo.lopez11@example.com", null, "S00001", "CHILD", "IOE2905302051", "HUGO", "m", "lopez", "HUGO m lopez", "1962-08-21", "F", "LPR", "A072926275", "521-41-1025", "CUB", "CUB", "4347141113", "hugo.lopez11@example.com", "881 OAK AVE", "LOS ANGELES", "CA", "90018"],
        ["B00001_02", "IOE5726575454", "I-134", "wei", "m", "lopez", "wei m lopez", "2016-06-28", "M", "A072120248", "CUB", "CUB", null, "76514404", "4347141113", null, "HUGO m lopez", "S00001", "CHILD", "IOE2905302051", "HUGO", "m", "lopez", "HUGO m lopez", "1962-08-21", "F", "LPR", "A072926275", "521-41-1025", "CUB", "CUB", "4347141113", "hugo.lopez11@example.com", "881 OAK AVE", "LOS ANGELES", "CA", "90018"],
        ["B00002_00", "IOE6234632757", "I-864A", "ENRIQUE", "m", "Lopez", "ENRIQUE m Lopez", "2010-04-14", "M", "A073257776", "NGA", "NGA", null, "00568566", "0182149274", null, "huog LOPEZ", "S00002", "CHILD", "IOE6234632757", "huog", null, "LOPEZ", "huog LOPEZ", "1962-08-21", "F", "LPR", "A072926275", "521-41-1025", "CUB", "CUB", "0182149274", "hugo.lopez11@example.com", "881 OAK AVE", "LOS ANGELES", "CA", "90018"],
        ["B00002_01", "IOE6234632757", "I-864A", "THIAGO", null, "lopez", "THIAGO lopez", "1976-11-28", "F", "A011445456", "SLV", "SLV", null, null, "9644454738", "hugo.lopez11@example.com", null, "S00002", "SPOUSE", "IOE6234632757", "huog", null, "LOPEZ", "huog LOPEZ", "1962-08-21", "F", "LPR", "A072926275", "521-41-1025", "CUB", "CUB", "0182149274", "hugo.lopez11@example.com", "881 OAK AVE", "LOS ANGELES", "CA", "90018"],
        ["B00002_02", "IOE0264466813", "I-864A", "linh", "m", "LOPEZ", "linh m LOPEZ", "2004-11-09", "F", "A078258024", "PUR", "PUR", null, "40727320", "0182149274", null, "huog LOPEZ", "S00002", "CHILD", "IOE6234632757", "huog", null, "LOPEZ", "huog LOPEZ", "1962-08-21", "F", "LPR", "A072926275", "521-41-1025", "CUB", "CUB", "0182149274", "hugo.lopez11@example.com", "881 OAK AVE", "LOS ANGELES", "CA", "90018"],
        ["B00003_00", "IOE9189849060", "I-134", "CARLOS", "m", "Lopez", "CARLOS m Lopez", "1986-02-27", "F", "A060378870", "UKR", "UKR", null, "57936549", "3738703864", null, null, "S00003", "SPOUSE", "IOE9189849060", "BROOKS", null, "lopez", "BROOKS lopez", "1991-01-16", "F", "LPR", "A044867237", "270-97-6537", "GTM", "GTM", "3738703864", "brooks.singh23@example.com", "8900 MAIN ST", "CHICAGO", "IL", "60653"],
        ["B00003_01", "IOE9189849060", "I-134", "Thiago", "m", "LOPEZ", "Thiago m LOPEZ", "2012-04-07", "F", "A040418264", "HND", "HND", null, null, "7985420930", "brooks.singh23@example.com", null, "S00003", "CHILD", "IOE9189849060", "BROOKS", null, "lopez", "BROOKS lopez", "1991-01-16", "F", "LPR", "A044867237", "270-97-6537", "GTM", "GTM", "3738703864", "brooks.singh23@example.com", "8900 MAIN ST", "CHICAGO", "IL", "60653"],
        ["B00004_00", "IOE3714894481", "I-864A", "brooks", null, "Lopez", "brooks Lopez", "1998-12-27", "M", "A002394997", "GTM", "GTM", null, "04963155", "3738703864", null, null, "S00004", "SPOUSE", "IOE3714894481", "brooks", null, "Lopez", "brooks Lopez", "1991-01-16", "F", "LPR", "A044867237", "270-97-6537", "GTM", "GTM", "3738703864", "brooks.singh23@example.com", "8900 MAIN ST", "CHICAGO", "IL", "60653"],
        ["B00004_01", "IOE3714894481", "I-864A", "Enrique", null, "lopez", "Enrique lopez", "2006-04-18", "F", "A088675473", "PUR", "PUR", null, null, "8170603961", "brooks.singh23@example.com", null, "S00004", "CHILD", "IOE3714894481", "brooks", null, "Lopez", "brooks Lopez", "1991-01-16", "F", "LPR", "A044867237", "270-97-6537", "GTM", "GTM", "3738703864", "brooks.singh23@example.com", "8900 MAIN ST", "CHICAGO", "IL", "60653"],
        ["B00005_00", "IOE8121516218", "I-134", "Enrique", null, "erickson", "Enrique erickson", "2017-07-07", "M", "A053224120", "UKR", "UKR", "462-82-7141", "39175515", "5945861633", null, "ELYSE Erickson", "S00005", "CHILD", "IOE8121516218", "ELYSE", null, "Erickson", "ELYSE Erickson", "1979-08-19", "F", "USC", null, "462-82-7141", "USA", "USA", "5945861633", "elyse.erickson29@example.com", "9366 OAK AVE", "MIAMI", "FL", "33180"],
        ["B00005_01", "IOE8121516218", "I-134", "David", null, "Erickson", "David Erickson", "1962-12-14", "F", "A059214643", "CHN", "CHN", null, null, "1573544415", "elyse.erickson29@example.com", null, "S00005", "SIBLING", "IOE8121516218", "ELYSE", null, "Erickson", "ELYSE Erickson", "1979-08-19", "F", "USC", null, "462-82-7141", "USA", "USA", "5945861633", "elyse.erickson29@example.com", "9366 OAK AVE", "MIAMI", "FL", "33180"],
        ["B00005_02", "IOE7434337300", "I-134", "EVANGELINE", null, "ERICKSON", "EVANGELINE ERICKSON", "1961-03-15", "M", "A086641988", "HND", "HND", null, "74388449", "5945861633", null, null, "S00005", "PARENT", "IOE8121516218", "ELYSE", null, "Erickson", "ELYSE Erickson", "1979-08-19", "F", "USC", null, "462-82-7141", "USA", "USA", "5945861633", "elyse.erickson29@example.com", "9366 OAK AVE", "MIAMI", "FL", "33180"],
        ["B00006_00", "IOE4745416689", "I-134", "Brooks", null, "CHERRY", "Brooks CHERRY", "1998-06-02", "F", "A096898441", "NGA", "NGA", null, "68932335", "3996282395", null, null, "S00006", "SIBLING", "IOE4745416689", "Enrique", "m", "Cherry", "Enrique m Cherry", "1991-10-03", "M", "LPR", "A010070991", "228-96-0113", "UKR", "UKR", "3996282395", "enrique.cherry75@example.com", "5160 OAK AVE", "NEW YORK", "NY", "10082"],
        ["B00006_01", "IOE4745416689", "I-134", "KENNETH", "m", "CHERRY", "KENNETH m CHERRY", "1962-10-11", "M", "A029253270", "VNM", "VNM", null, null, "2718105306", "enrique.cherry75@example.com", null, "S00006", "PARENT", "IOE4745416689", "Enrique", "m", "Cherry", "Enrique m Cherry", "1991-10-03", "M", "LPR", "A010070991", "228-96-0113", "UKR", "UKR", "3996282395", "enrique.cherry75@example.com", "5160 OAK AVE", "NEW YORK", "NY", "10082"],
        ["B00007_00", "IOE9001875923", "I-864", "SANTANA", "m", "pope", "SANTANA m pope", "1945-02-18", "F", "A060293016", "UKR", "UKR", null, "48653292", "5790513312", null, null, "S00007", "PARENT", "IOE9001875923", "Ahmed", "m", "pope", "Ahmed m pope", "1995-04-01", "M", "USC", null, "417-38-9304", "USA", "USA", "5790513312", "ahmed.pope92@example.com", "2472 MAPLE DR", "DALLAS", "TX", "75282"],
        ["B00007_01", "IOE9001875923", "I-864", "Miguel", null, "martinez", "Miguel martinez", "1950-06-11", "M", "A048730603", "UKR", "UKR", null, null, "3743111347", "ahmed.pope92@example.com", null, "S00007", "OTHER", "IOE9001875923", "Ahmed", "m", "pope", "Ahmed m pope", "1995-04-01", "M", "USC", null, "417-38-9304", "USA", "USA", "5790513312", "ahmed.pope92@example.com", "2472 MAPLE DR", "DALLAS", "TX", "75282"],
        ["B00007_02", "IOE5624771422", "I-864", "brooks", "m", "Pope", "brooks m Pope", "2012-02-17", "M", "A021760681", "UKR", "UKR", null, "23928113", "5790513312", null, "Ahmed m pope", "S00007", "CHILD", "IOE9001875923", "Ahmed", "m", "pope", "Ahmed m pope", "1995-04-01", "M", "USC", null, "417-38-9304", "USA", "USA", "5790513312", "ahmed.pope92@example.com", "2472 MAPLE DR", "DALLAS", "TX", "75282"],
        ["B00008_00", "IOE3920600894", "I-864A", "Thiago", null, "FISCHER", "Thiago FISCHER", "1945-12-19", "F", "A096190438", "MEX", "MEX", null, "19817685", "1893193092", null, null, "S00008", "OTHER", "IOE3920600894", "maria", "m", "Kim", "maria m Kim", "1990-06-18", "F", "USC", null, "152-12-7790", "PUR", "USA", "1893193092", "maria.kim30@example.com", "4648 CEDAR LN", "HOUSTON", "TX", "77089"],
        ["B00008_01", "IOE3920600894", "I-864A", "nancy", null, "KIM", "nancy KIM", "2020-02-05", "M", "A098479965", "UKR", "UKR", null, null, "5734777815", "maria.kim30@example.com", null, "S00008", "CHILD", "IOE3920600894", "maria", "m", "Kim", "maria m Kim", "1990-06-18", "F", "USC", null, "152-12-7790", "PUR", "USA", "1893193092", "maria.kim30@example.com", "4648 CEDAR LN", "HOUSTON", "TX", "77089"],
        ["B00009_00", "IOE5572589410", "I-864", "BROOKS", "m", "mensah", "BROOKS m mensah", "2011-11-19", "M", "A051958359", "PUR", "PUR", null, "73826163", "5245878928", null, "evangeline m Mensah", "S00009", "CHILD", "IOE5572589410", "evangeline", "m", "Mensah", "evangeline m Mensah", "1985-07-25", "F", "USC", null, "676-46-0980", "USA", "USA", "5245878928", "evangeline.mensah40@example.com", "7402 LAKE BLVD", "MIAMI", "FL", "33140"],
        ["B00009_01", "IOE5572589410", "I-864", "Linh", null, "MENSAH", "Linh MENSAH", "1973-08-06", "F", "A016294232", "PUR", "PUR", null, null, "6145769608", "evangeline.mensah40@example.com", null, "S00009", "SPOUSE", "IOE5572589410", "evangeline", "m", "Mensah", "evangeline m Mensah", "1985-07-25", "F", "USC", null, "676-46-0980", "USA", "USA", "5245878928", "evangeline.mensah40@example.com", "7402 LAKE BLVD", "MIAMI", "FL", "33140"],
        ["B00009_02", "IOE3296339148", "I-864", "hugo", "m", "Mensah", "hugo m Mensah", "2017-01-28", "F", "A045446637", "CUB", "CUB", null, "68307191", "5245878928", null, "evangeline m Mensah", "S00009", "CHILD", "IOE5572589410", "evangeline", "m", "Mensah", "evangeline m Mensah", "1985-07-25", "F", "USC", null, "676-46-0980", "USA", "USA", "5245878928", "evangeline.mensah40@example.com", "7402 LAKE BLVD", "MIAMI", "FL", "33140"],
        ["B00010_00", "IOE8153543990", "I-134", "CARLOS", null, "PATEL", "CARLOS PATEL", "1956-01-15", "M", "A029990368", "UKR", "UKR", null, "47241513", "1712548707", null, null, "S00010", "SPOUSE", "IOE8153543990", "HUGO", "m", "patel", "HUGO m patel", "1952-03-26", "F", "USC", null, "453-95-4899", "USA", "USA", "1712548707", "hugo.patel72@example.com", "4712 MAPLE DR", "DALLAS", "TX", "75215"],
        ["B00010_01", "IOE8153543990", "I-134", "Raj", null, "Patel", "Raj Patel", "2019-10-27", "F", "A041782753", "VNM", "VNM", null, null, "4023745619", "hugo.patel72@example.com", null, "S00010", "CHILD", "IOE8153543990", "HUGO", "m", "patel", "HUGO m patel", "1952-03-26", "F", "USC", null, "453-95-4899", "USA", "USA", "1712548707", "hugo.patel72@example.com", "4712 MAPLE DR", "DALLAS", "TX", "75215"]
    ]
}
""")


results = main()
