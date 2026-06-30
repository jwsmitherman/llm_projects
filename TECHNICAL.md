# Section B (Splink) — technical line-by-line

Detailed technical walkthrough of the Splink section of `IDEN-43831_anm_probabilistic.py`. Splink is an implementation of the Fellegi-Sunter probabilistic record-linkage model. This section runs it on **record-level** data (raw `_1`/`_2` fields) rather than the precomputed pcs pairwise features used in Section A.

---

## Imports

```python
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
```

- `comparison_library` (`cl`) — factory functions for field comparisons (`ForenameSurnameComparison`, `DateOfBirthComparison`, `ExactMatch`, `LevenshteinAtThresholds`). Each returns a `Comparison` object describing the agreement levels and fuzzy thresholds for one column.
- `DuckDBAPI` — the execution backend. Splink compiles its linkage logic to SQL; `DuckDBAPI` runs that SQL in an in-process DuckDB engine on the driver node. Chosen over the Spark backend because the sample is small enough to fit on the driver and this avoids a known Splink≥4.0.4 Spark clustering-cleanup issue.
- `Linker` — the central object; holds the record table, settings, and exposes `.training` and `.inference`.
- `SettingsCreator` — builds the settings object (link type, unique-id column, comparisons, blocking rules, prior).
- `block_on` — helper that constructs a blocking rule from one or more column names or SQL expressions.

## Reshape pairs → records

```python
ATTRS = ["firstName", "middleName", "lastName", "dob", "cob", "anumber", "ssn", "fin",
         "driversLicense", "passport", "i94", "travel_doc"]
RID = "party_record_receipt"
```

- `ATTRS` — the attribute columns carried into the record table for comparison/blocking.
- `RID` — prefix of the per-record identity key, stored as `party_record_receipt_1` / `_2`.

```python
if f"{RID}_1" in src.columns and "firstName_1" in src.columns:
    pairs = src
else:
    print(...); df = spark.table(FULLSET)
    n_pos = df.filter("label = 1").count(); n_neg = df.filter("label = 0").count()
    fr = {1: min(1.0, SAMPLE_PER_CLASS / max(n_pos, 1)), 0: min(1.0, SAMPLE_PER_CLASS / max(n_neg, 1))}
    pairs = df.sampleBy("label", fractions=fr, seed=SEED).toPandas()
```

- Splink needs raw record fields, not the pcs `f_*` features. If the featurized training table still carries the raw `_1`/`_2` columns, reuse it so Section B scores the **same rows** as Section A.
- Otherwise fall back to `FULLSET` and draw a class-balanced sample: `sampleBy` performs stratified Bernoulli sampling per `label` at the given fractions; `fr` targets `SAMPLE_PER_CLASS` per class and is capped at `1.0`. Same `SEED` for reproducibility. **Caveat:** this path yields different rows than Section A, so cross-section comparison is weakened.

```python
def side(n):
    ren = {f"{RID}_{n}": "record_id"}; ren.update({f"{a}_{n}": a for a in ATTRS})
    return pairs[list(ren)].rename(columns=ren)
records = (pd.concat([side("1"), side("2")], ignore_index=True)
             .dropna(subset=["record_id"]).drop_duplicates("record_id").reset_index(drop=True))
```

- `side(n)` projects one side of every pair and strips the `_n` suffix, renaming `party_record_receipt_n → record_id` and `attr_n → attr`.
- Concatenating side 1 and side 2 vertically yields a record-level table. `dropna` removes rows with no id; `drop_duplicates("record_id")` collapses records that appear in multiple pairs so each physical record is represented once. This is the table Splink dedupes over.

## Truth table and the no-strong-id flag

```python
def _usable(c1, c2):
    a, b = pairs[c1].astype("string"), pairs[c2].astype("string")
    return a.notna() & b.notna() & (a.str.strip() != "") & (b.str.strip() != "")
lab = pairs[[f"{RID}_1", f"{RID}_2", "label"]].copy()
lab["no_strong_id"] = (~(_usable("ssn_1", "ssn_2") | _usable("anumber_1", "anumber_2")
                         | _usable("fin_1", "fin_2"))).values
```

- `_usable(c1, c2)` returns a boolean Series: `True` where **both** sides of a field are present and non-blank. Cast to pandas `string` dtype so `.str` operations and null handling behave consistently.
- `lab` is the evaluation truth table: the two record ids plus the gold `label`.
- `no_strong_id` flags pairs where none of SSN / A-Number / FIN is usable on both sides — the population the spike targets. The leading `~` negates the union, so `True` means "no strong identifier available."

```python
lab = lab.dropna(subset=[f"{RID}_1", f"{RID}_2"])
lab["key"] = lab.apply(lambda r: tuple(sorted([r[f"{RID}_1"], r[f"{RID}_2"]])), axis=1)
lab = lab.drop_duplicates("key")
```

- Drop rows missing a record id.
- `key` is an **order-independent** pair identifier: sorting the two ids makes `(A,B)` and `(B,A)` collapse to one tuple. This is essential because Splink's predicted pairs come back unordered (`record_id_l`, `record_id_r`), so scoring must match on a canonical key.
- `drop_duplicates("key")` keeps one row per unordered pair.

## Prior probability

```python
_npairs = len(records) * (len(records) - 1) / 2
P_MATCH = max(1e-6, min(0.5, int((lab["label"] == 1).sum()) / _npairs))
print("[B] records:", len(records), "labeled pairs:", len(lab), "prior P_MATCH:", round(P_MATCH, 8))
```

- `_npairs` = C(N, 2), the number of distinct record pairs.
- `P_MATCH` is Fellegi-Sunter's prior — `probability_two_random_records_match`, i.e. P(two randomly chosen records are the same entity). Estimated as true-match pairs ÷ all possible pairs, clamped to `[1e-6, 0.5]`.
- **Why computed directly:** the prior is normally derived via `estimate_probability_two_random_records_match(deterministic_rules, recall)`. On the real data the strict deterministic rules matched more pairs than `recall=0.7` allows, raising `ValueError: recall must be at least 0.95`. Supplying a fixed prior removes that dependency. Because AUC is rank-based, the exact prior does not affect AUC; it shifts the absolute probability scale (which is why Splink F1 at a fixed threshold can read low).

## Comparisons

```python
NAME_COMPS = [cl.ForenameSurnameComparison("firstName", "lastName"),
              cl.DateOfBirthComparison("dob", input_is_string=True),
              cl.ExactMatch("cob")]
def id_comp(col):
    return cl.LevenshteinAtThresholds(col, [1, 2]).configure(term_frequency_adjustments=True)
```

- `ForenameSurnameComparison` — a composite comparison with graded agreement levels for first+last together, including the reversed-columns level (first/last transposition) and term-frequency-aware levels.
- `DateOfBirthComparison(input_is_string=True)` — graded levels by date closeness (exact, off-by-character, etc.); `input_is_string=True` tells Splink the column is a string, not a date type.
- `ExactMatch("cob")` — binary agree/disagree on country of birth.
- `id_comp(col)` — for identifier fields: agreement levels at Levenshtein distance 1 and 2 (tolerate 1–2 character edits). `.configure(term_frequency_adjustments=True)` makes a rare shared value (e.g. an uncommon ID) weigh more than a common one, because the `u` probability of a chance match is lower for rare values.

## Scenario definitions

```python
ID_COL = {"anum": "anumber", "ssn": "ssn", "fin": "fin", "dl": "driversLicense",
          "passport": "passport", "i94": "i94", "travel_doc": "travel_doc"}
SCN_IDS = {
    "baseline":             ["anum", "ssn", "fin"],
    "no_ssn":               ["anum", "fin"],
    "no_strong_id":         [],
    "with_dl":              ["anum", "ssn", "fin", "dl"],
    "with_dl_no_strong_id": ["dl"],
    "with_other_ids":       ["anum", "ssn", "fin", "dl", "passport", "i94", "travel_doc"],
    "other_ids_no_strong":  ["dl", "passport", "i94", "travel_doc"],
    "all_ids":              ["anum", "ssn", "fin", "dl", "passport", "i94", "travel_doc"],
}
```

- `ID_COL` maps a short scenario key to the real column name.
- `SCN_IDS` enumerates the 8 ablation scenarios — which identifier columns each scenario is allowed to use (as comparisons and as blocking keys). Name/DOB/COB are always present; these lists control only the identifiers. This mirrors Section A's `SCN` so the two sections answer the same questions.

## Blocking (the fix)

```python
NAME_DOB_BLOCKS = [
    block_on("firstName", "dob"),
    block_on("lastName", "dob"),
    block_on("firstName", "lastName"),
    block_on("substr(lastName,1,3)", "dob"),
    block_on("lastName", "substr(dob,1,4)"),
    block_on("firstName", "substr(dob,1,4)"),
]
```

- **Blocking** = candidate generation. Splink only scores record pairs that satisfy at least one blocking rule; all other pairs are never compared. This is what makes linkage tractable (you avoid the full N² comparison), but it caps recall: any true match that no rule captures is never seen.
- Each `block_on(...)` requires exact equality on its argument(s) for a pair to become a candidate. Multiple columns mean "equal on all of them"; SQL expressions like `substr(lastName,1,3)` block on a derived key.
- **Why this set:** the previous version blocked only on `(firstName,dob)`, `(lastName,dob)`, and the per-scenario id columns. For the no-strong-id population the id blocks generate nothing (ids absent), and strict exact name+dob misses any pair with a name or DOB typo, so ~45% of true matches were never generated and scored `p=0`, collapsing AUC to ~0.60. This set adds name-only and prefix/year-relaxed blocks to recover typo'd matches, and applies to **every** scenario.
- `substr(...)` are crude fuzzy stand-ins. Diagnostics showed they only lift no-strong-id blocking recall to ~0.55 because the remaining misses disagree on the name/DOB *strings themselves*; the production fix is phonetic keys (`soundex`/`metaphone`), which DuckDB/Spark provide and `block_on` accepts as SQL expressions.

## Per-scenario train + predict

```python
def splink_scenario(ids):
    comps = NAME_COMPS + [id_comp(ID_COL[k]) for k in ids]
    blocking = NAME_DOB_BLOCKS + [block_on(ID_COL[k]) for k in ids]
    settings = SettingsCreator(link_type="dedupe_only",
                               unique_id_column_name="record_id",
                               comparisons=comps,
                               blocking_rules_to_generate_predictions=blocking,
                               probability_two_random_records_match=P_MATCH)
    linker = Linker(records, settings, db_api=DuckDBAPI())
```

- `comps` — name/DOB/COB comparisons plus one id comparison per scenario identifier.
- `blocking` — the robust name/DOB set plus an exact-id block per scenario identifier. Id blocks raise recall cheaply on populations that have ids; the name/DOB set covers the rest.
- `SettingsCreator`: `dedupe_only` links a table to itself (one table, find duplicates) rather than linking two tables; `unique_id_column_name` names the record key; the comparisons and blocking rules define the model and candidate set; `probability_two_random_records_match` injects the fixed prior.
- `Linker(records, settings, db_api=DuckDBAPI())` instantiates the linker over the record table with the DuckDB backend.

```python
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("firstName", "lastName"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
```

- `estimate_u_using_random_sampling` — estimates the **u** parameters (probability of agreement at each level *given the pair is a non-match*) by sampling random record pairs, which are almost all non-matches at scale. `max_pairs=1e6` bounds the sample size/runtime.
- The two `estimate_parameters_using_expectation_maximisation` calls run **EM** to estimate the **m** parameters (agreement probabilities *given a match*) and refine the model. Each pass is trained within a blocking pass (`firstName+lastName`, then `dob`) so EM sees enough true-match-rich pairs to converge. EM is unsupervised — it does not use the `label` column. Warnings like "u/m values not fully trained for passport/i94/travel_doc" mean those comparison levels were rarely observed during EM and fall back to defaults; harmless for ranking but they leave those identifiers weakly calibrated.

```python
    preds = linker.inference.predict().as_pandas_dataframe()
    prob = {tuple(sorted([a, b])): p for a, b, p in
            zip(preds["record_id_l"], preds["record_id_r"], preds["match_probability"])}
    scored = lab.copy()
    scored["p"] = scored["key"].map(prob)
    scored["blocked"] = scored["p"].notna()
    scored["p"] = scored["p"].fillna(0.0)
    return scored
```

- `predict()` scores every candidate pair and returns `match_probability` (plus a `match_weight` log2-odds). `.as_pandas_dataframe()` materializes it.
- `prob` is a lookup from canonical (sorted) record-id pair to predicted probability.
- `scored["p"]` joins Splink's probability onto the truth table by `key`. Pairs Splink never generated have no entry → `NaN`.
- `blocked` records whether the pair was a candidate at all (`p` not null) — this is the coverage signal.
- `fillna(0.0)` treats unblocked pairs as confident non-matches (probability 0). This is the line that makes blocking recall directly limit AUC.

## Evaluation loop

```python
rows = []
for scn, ids in SCN_IDS.items():
    s = splink_scenario(ids)
    for slice_name, sub in [("all", s), ("no_strong_id", s[s["no_strong_id"]])]:
        if len(sub) == 0 or sub["label"].nunique() < 2:
            continue
        bf = sub[sub["blocked"]]
        m = sub["label"] == 1
        rows.append(dict(scenario=scn, model="splink", eval=slice_name, n=len(sub),
                         auc=round(roc_auc_score(sub["label"], sub["p"]), 4),
                         ap=round(average_precision_score(sub["label"], sub["p"]), 4),
                         f1=round(f1_score(sub["label"], (sub["p"] >= THRESH).astype(int), zero_division=0), 4),
                         blocking_recall=round(sub.loc[m, "blocked"].mean(), 4),
                         auc_blocked_only=(round(roc_auc_score(bf["label"], bf["p"]), 4)
                                           if bf["label"].nunique() > 1 else None)))
```

- For each scenario, score on the full evaluation set and on the no-strong-id subset.
- Skip a slice that is empty or single-class (AUC/AP undefined without both labels).
- `bf` = only the pairs Splink actually compared (blocked). `m` = true matches in the slice.
- Metrics recorded:
  - `auc` / `ap` — ranking quality and precision-recall summary, computed with `p=0` for unblocked pairs (so they reflect the *deployed* behaviour including coverage loss).
  - `f1` — at the fixed `THRESH`; reads low for Splink because of its calibrated probability scale, not its ranking.
  - `blocking_recall` = `mean(blocked | label==1)` — the fraction of true matches that blocking captured. This is the coverage metric; `1 − blocking_recall` is the fraction auto-scored 0.
  - `auc_blocked_only` — AUC over blocked pairs only. Comparing it to `auc` isolates model quality from coverage: a high `auc_blocked_only` with a low `auc` proves the loss is candidate generation, not the model. `None` when the blocked set is single-class.

```python
splink_results = pd.DataFrame(rows)
all_results.append(splink_results)
print("\n[B] splink results (blocking_recall = coverage; auc vs auc_blocked_only shows the coverage gap):")
print(splink_results.sort_values(["eval", "auc"], ascending=[True, False]).to_string(index=False))
```

- Assemble Section B's results, add them to the combined bucket (concatenated and written to `anm_scenario_results` in the final block), and print best-AUC-first.

---

## How to read a real run

On the real data:
- `no_strong_id` rows: `blocking_recall ≈ 0.55` while `auc_blocked_only ≈ 0.99`. The model is near-perfect on compared pairs; the ~45% of true matches that blocking misses (scored `p=0`) cap `auc` at ~0.60. **This is a coverage problem, now measured rather than inferred.**
- `all` rows: `blocking_recall ≈ 0.96`, so `auc ≈ 0.97` — close to `auc_blocked_only`, confirming the gap is coverage-driven.
- The remaining misses disagree on name/DOB strings, so further substr/year blocks won't help. The next lever is phonetic/token blocking (Soundex/Metaphone, sorted name tokens). Adding blocks raises recall and candidate volume together; tune with the blocking sweep cell.
