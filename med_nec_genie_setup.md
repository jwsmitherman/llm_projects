# Medical Necessity — Databricks Genie Setup

How to stand up an AI/BI Genie space so the business can explore the scored orders in plain English.

## 1. Create the table

In the scoring notebook (`medical_necessity_scoring.py`), set `WRITE_GENIE_TABLE = True` and confirm `GENIE_TABLE` points to a schema you can write to (default `prod-sandbox.vivekkumar_patel.med_nec_genie`). Run the notebook. It writes one row per order with the scores, labels, concept flags, and column comments Genie reads.

## 2. Create the Genie space

1. In Databricks: **Genie → New**.
2. Add the table `med_nec_genie` as the data source.
3. Paste the instructions below into the space **Instructions**.
4. Add the example questions below as **Sample questions** (curated SQL optional but recommended for the trusted ones).

## 3. Space instructions (paste in)

This data is non-emergent ground ambulance transport orders, one row per order, scored for whether the free-text reason recorded at order time documents medical necessity under CMS criteria. Labels describe the documentation, not the trip, and are not a billing determination.

- `necessity_class`: necessary / not_necessary / indeterminate.
- `total_score` = mobility_score + monitoring_score. total_score 0 = not_necessary; total_score 3 or more with a named concept = necessary; otherwise indeterminate.
- A "named" concept is one CMS names explicitly (bed_confined, mobility_deficit, cannot_sit, ventilator, suctioning, iv_medication, cardiac). `has_named_concept` = 1 when any is present.
- The 15 concept columns (bed_confined ... nonclinical) are 1/0 flags for whether that reason appears in the text.
- `unmatched_text` = 1 means text was entered but no rule recognized it — the set a language model would review.
- `gy_disposition` relates the label to the GY billing process; not_necessary orders are GY candidates.
- This reflects order-time documentation only; the crew PCR is not in this data.

## 4. Column glossary

| Column | Meaning |
|---|---|
| necessity_class | necessary / not_necessary / indeterminate |
| gy_disposition | GY-process relation for the label |
| total_score | mobility_score + monitoring_score |
| mobility_score | matched mobility-axis concept weights (why not a cheaper option) |
| monitoring_score | matched monitoring-axis concept weights (why this level of service) |
| named_score | points from named CMS concepts only |
| has_named_concept | 1 if any named CMS concept matched |
| unmatched_text | 1 if text present but nothing scored (language-model target) |
| confidence | high / medium / low |
| recommended_los | level of service implied by monitoring concepts (descriptive) |
| why_labeled | concepts matched and the exact triggering text |
| clinical_text | the order-time free-text reason |
| bed_confined … nonclinical | 1/0 flag per concept |

## 5. Curated example questions + SQL

Replace `T` with the full table name.

**Overall split**
```sql
SELECT necessity_class, COUNT(*) AS orders,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM T GROUP BY necessity_class ORDER BY orders DESC;
```

**By level of service**
```sql
SELECT LevelOfService, necessity_class, COUNT(*) AS orders
FROM T GROUP BY LevelOfService, necessity_class ORDER BY LevelOfService, orders DESC;
```

**Not-necessary rate by level of service**
```sql
SELECT LevelOfService,
       ROUND(100.0 * SUM(CASE WHEN necessity_class='not_necessary' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_not_necessary,
       COUNT(*) AS orders
FROM T GROUP BY LevelOfService ORDER BY orders DESC;
```

**GY candidates (no documented reason)**
```sql
SELECT COUNT(*) AS gy_candidates
FROM T WHERE necessity_class = 'not_necessary';
```

**How many BLS orders mention cardiac monitoring**
```sql
SELECT COUNT(*) AS bls_cardiac
FROM T WHERE LevelOfService = 'BLS' AND cardiac = 1;
```

**Most common reasons for mobility**
```sql
SELECT 'bed_confined' AS reason, SUM(bed_confined) AS orders FROM T
UNION ALL SELECT 'mobility_deficit', SUM(mobility_deficit) FROM T
UNION ALL SELECT 'cannot_sit', SUM(cannot_sit) FROM T
UNION ALL SELECT 'bariatric', SUM(bariatric) FROM T
UNION ALL SELECT 'wound_ostomy', SUM(wound_ostomy) FROM T
UNION ALL SELECT 'behavioral', SUM(behavioral) FROM T
ORDER BY orders DESC;
```

**Orders the rules could not read (language-model target)**
```sql
SELECT COUNT(*) AS unmatched
FROM T WHERE unmatched_text = 1;
```

**Score distribution**
```sql
SELECT total_score, COUNT(*) AS orders
FROM T GROUP BY total_score ORDER BY total_score;
```

**Necessary orders resting on a single concept (borderline)**
```sql
SELECT COUNT(*) AS borderline
FROM T WHERE necessity_class = 'necessary' AND total_score = 3;
```

**Sample the not-necessary orders to read**
```sql
SELECT OrderId, LevelOfService, clinical_text
FROM T WHERE necessity_class = 'not_necessary' LIMIT 50;
```

**Confidence breakdown**
```sql
SELECT confidence, COUNT(*) AS orders FROM T GROUP BY confidence ORDER BY orders DESC;
```

**Requested vs recommended level of service**
```sql
SELECT LevelOfService AS requested, recommended_los, COUNT(*) AS orders
FROM T GROUP BY LevelOfService, recommended_los ORDER BY orders DESC;
```

## 6. Notes

- Genie generates SQL from the comments, so keep column comments current if the schema changes.
- Add only trusted SQL as curated answers; leave exploratory questions for Genie to generate.
- Re-run the notebook to refresh the table after any scoring change.
