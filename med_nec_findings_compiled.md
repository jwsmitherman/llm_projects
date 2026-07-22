# Medical Necessity Analysis - Findings

**Source:** `` `prod-sandbox`.vivekkumar_patel.temp_tnet_tripmaster `` (read-only snapshot from
`prod.silver_transbroker.triprequest` + `tripleg`)
**Period:** CY2025 - *not yet confirmed*
**Customers:** Texas Health Resources (74,378), MUSC (30,256). 52 facilities.
**Rows:** 104,634 trip legs
**Workbook:** `med_nec_qa_summary.xlsx`

---

## Goal

An AI check that runs while the nurse places the order and answers:
**Does what you just typed justify the ambulance you're requesting?**

If not, tell them what's missing before submit - not 45 days later at denial.

---

## Scope - read first

104,634 includes transport types that need no medical necessity documentation:

| Excluded | Orders |
|---|---|
| Rideshare / Taxi / Lyft | ~9,743 |
| Rotor / Fixed Wing (air) | ~2,096 |
| Emergent | ~2,322 |
| **Total** | **~14,161** |

**In-scope: ~90,500 non-emergent ground orders.**

Rideshare alone is 9,292 of the 24,303 "nothing typed" orders. It needs no justification.
An earlier draft said 26.5% at-risk. **Correct figure is ~19%.** Use scoped numbers.

---

## Headline

**~1 in 5 non-emergent ground orders cannot support medical necessity from the written text.**

| Finding | Orders | Share |
|---|---|---|
| Nothing typed | ~13,500 | ~15% |
| Vague filler only | ~3,300 | ~3.6% |
| **At-risk** | **~16,800** | **~18.6%** |
| Text present, nothing recognized | ~23,900 | ~26% |
| Specific clinical reason | ~44,200 | ~49% |
| Specific + some filler | ~8,700 | ~9.6% |

*Approximate - scope filter not yet re-run. Raw counts: 24,303 / 3,413 / 23,979 / 44,233 / 8,706.*

**Four things to know:**

1. **MUSC Ground leaves the box empty on 49% of orders. Texas Health on 13%.** Same software.
   This is workflow, not technology.
2. **BLS is the exposure.** 9,606 Basic Life Support orders have no documentation or filler only.
3. **"General weakness" appears in 8,568 orders (8.2%).** With fall-risk language, 13,316.
4. **348 distinct clinical questions found** - the index nobody had.

---

## Definitions

### Clinical reason
A specific medical fact explaining why the patient can't use a wheelchair van or car.

Two tests, both required:
- **Verifiable?** Could a payer check it against the record?
- **Rules out the cheaper option?** Does it explain the contraindication?

| Passes | Fails |
|---|---|
| post-CVA hemiparesis | patient is elderly |
| unable to bear weight | diabetic |
| requires supine transport | going to rehab |
| on 3L nasal cannula | doctor ordered ambulance |
| ventilator dependent | |

"Diabetic" is verifiable but doesn't stop someone riding in a car. Fails test 2.

*Source: Benefit Policy Manual 10.2.1 - medical necessity exists when use of any other method of
transportation is contraindicated.*

### Vague
Describes a general state without the cause or a specific functional limit.

Test: **can you picture what's physically wrong?**
- "General weakness" - no. Can they sit? Stand? Walk?
- "Post-CVA hemiparesis, cannot bear weight" - yes.

Not false. Incomplete. CMS needs cause *and* functional deficit.

| Vague | Specific |
|---|---|
| "General weakness" | "Post-CVA left hemiparesis, unable to bear weight or sit upright" |

Same patient. One gets paid.

*Source: MAC guidance - "Vague and general information is of little or no value."*

### Filler
Same as vague. "Filler" = the behavior, "vague" = the language. Three types:

| Type | Examples | Why it fails |
|---|---|---|
| Vague condition | general weakness, generally weak | No cause, no functional limit |
| Risk label | fall risk, unsteady gait, deconditioning | A risk is not a contraindication |
| Non-clinical | per protocol, convenience, no other transport, family request | Logistics, not medicine |

*Source: Manual 10.2.1 - a physician's order does not prove necessity, and other transport
disqualifies "whether or not such other transportation is actually available."*

Why it happens: a nurse booking 12 transports before lunch types the shortest thing that gets the
ambulance dispatched. The consequence lands 45 days later in a billing office they never see.

---

## Method

Searched one field only: **`tripleg.ClinicalData`** - the nurse's free-text box.

Excluded `LosQuestions` (it's the full form template, ~49,000 chars - contaminated v1) and
`SpecialNeeds` (equipment flags, not justification).

### Group A - supports necessity (12)

| Concept | Search terms | CMS source |
|---|---|---|
| mobility_deficit | hemiparesis, paralysis, non-ambulatory, unable to bear weight, fracture | Manual 10.2.3 prongs 1-2 |
| cannot_sit | cannot sit, special positioning, must lie flat, supine, stretcher | Manual 10.2.3 prong 3 |
| bed_confined | bed bound, bed confined, unable to get up | Manual 10.2.3 (all three) |
| oxygen | oxygen, O2, LPM, nasal cannula, BiPAP, CPAP | inferred - 10.2.1 |
| cardiac | cardiac, telemetry, EKG, NSTEMI, arrhythmia, afib | 42 CFR 414.605 (ALS) |
| behavioral | dementia, Alzheimer's, combative, agitated, altered mental status, flight risk | inferred - **weakest** |
| isolation | isolation, MRSA, C. diff, precautions | inferred - 10.2.1 |
| wound_ostomy | wound, ostomy, ulcer, decubitus, drain | inferred - 10.2.1 |
| ventilator | ventilator, vent, trach, intubated | 42 CFR 414.605 (ALS2/SCT) |
| iv_medication | IV, infusion, drip, heparin, antibiotic, TPN | 42 CFR 414.605 (ALS2) |
| suctioning | suction | 42 CFR 414.605 (SCT) |
| bariatric | bariatric, morbid obesity | inferred - 10.2.1 |

### Group B - quality problems (3)

| Concept | Search terms | CMS source |
|---|---|---|
| weakness_only | general weakness, generally weak, weak (no cause) | MAC: "vague and general information is of little or no value" |
| fall_risk_only | fall risk, unsteady, deconditioning | MAC (same) |
| nonclinical | per protocol, convenience, no other transport, family request | Manual 10.2.1 |

"Inferred" means the concept is **not named** in CMS ambulance guidance - it rests on the general
contraindication test in 10.2.1. Those five need SME review.

### Labels

| Group A hits | Group B hits | Bucket |
|---|---|---|
| box empty | - | no_documentation |
| 0 | 1+ | filler_only |
| 0 | 0 | unrecognized |
| 1+ | 1+ | reason_plus_filler |
| 1+ | 0 | clear_reason |

Every order gets exactly one bucket. These are the names used in the workbook.

### CMS references

| # | Source | Used for |
|---|---|---|
| 1 | [42 CFR 410.40 - Coverage of ambulance services](https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-B/part-410/subpart-B/section-410.40) | A signed PCS alone does not prove necessity - the clinical text is what counts |
| 2 | [Benefit Policy Manual Ch. 10 (Pub 100-02)](https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c10.pdf) | 10.2.1 necessity test, 10.2.3 bed-confined three prongs, 10.2.4 documentation |
| 3 | [42 CFR 414.605 - Definitions](https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-B/part-414/subpart-H/section-414.605) | BLS / ALS1 / ALS2 / Specialty Care Transport definitions |
| 4 | [CMS Prior Auth Operational Guide (RSNAT)](https://www.cms.gov/research-statistics-data-and-systems/monitoring-programs/medicare-ffs-compliance-programs/prior-authorization-initiatives/downloads/ambulancepriorauth_operationalguide_123115.pdf) | How CMS itself evaluates medical necessity documentation |

Key quotes:

- **Manual 10.2.1** - the foundation of our two-part test:
  *"Medical necessity is established when the patient's condition is such that use of any other
  method of transportation is contraindicated. In any case in which some means of transportation
  other than an ambulance could be used without endangering the individual's health, whether or
  not such other transportation is actually available, no payment may be made."*
  The last clause is what defeats "no other transport available" as a justification.

- **Manual 10.2.1** - why "doctor ordered it" is not enough:
  *"The presence (or absence) of a physician's order for a transport by ambulance does not
  necessarily prove (or disprove) whether the transport was medically necessary."*

- **Manual 10.2.3** - bed-confined requires ALL THREE: unable to get up from bed without
  assistance, unable to ambulate, unable to sit in a chair or wheelchair. And:
  *"Bed-confinement, by itself, is neither sufficient nor is it necessary."*
  If even bed-confinement is not sufficient alone, a vague term certainly is not.

- **MAC guidance** - the direct justification for the filler bucket:
  *"Vague and general information is of little or no value."*

- **42 CFR 410.40** - why we read the text instead of just checking a PCS exists:
  *"The presence of the physician or non-physician certification statement or signed return
  receipt does not alone demonstrate that the ambulance transport was medically necessary."*

Notes:
- The federal term is **Specialty Care Transport (SCT)**. GMR's "CCT" maps to SCT for billing.
- ALS2 at 414.605 means 3+ IV medications, or one of seven procedures (intubation, surgical
  airway, central line, intraosseous line, defibrillation/cardioversion, cardiac pacing, chest
  decompression).

Supporting context for the business case - CMS 2020 Medicare FFS improper payment data, as
summarized in industry reporting (verify against the CMS source before quoting externally):
ambulance improper payment rate ~7.2% (~$349M), with **insufficient documentation accounting for
~62.5% of it**, medical necessity ~23.5%, coding ~10.8%. If accurate, documentation - not clinical
judgment - is the largest driver nationally, which is exactly what a front-end prompt addresses.

The word lists were assembled from these sources plus the Transport.net strategy deck
("general weakness" vs "post-CVA hemiparesis") and the medical necessity working sessions.

**Not validated by Jen Jones or Michelle's team. Directional until they review.**

### Judgment calls

- **"Behavioral" is in Group A. CMS may disagree** - they want a *physical* limitation, and a
  confused but ambulatory patient can ride with an escort. Most likely reclassification.
- **"Fall risk" is in Group B.** A risk is not a contraindication. Wheelchair vans carry
  fall-risk patients daily. Defensible but confirm.
- **"Weak" in context lands in `mixed`, not `weak_only`** - e.g. "weak from post-op anemia,
  unable to stand" has a real reason. Intended behavior.

### Limitation

This is keyword search, not AI. It can't tell that "patient cannot support trunk in vehicle"
means "unable to sit upright." That gap is what the LLM fills - and `unrecognized` (~26%) is the
measurement of it.

---

## Workbook tabs

Output file: `med_nec_buckets.xlsx`. Five tabs, plain formatting.

**1. Definitions**
- Why we did this, what we did, the source table, and the scope.
- Both word groups with their tests and CMS sources.
- The five buckets, how each is assigned, and the four limits.
- Read this first if you have not seen the analysis before.

**2. Summary**
- Bucket counts with a plain-English meaning for each.
- A Status column marking each bucket AT RISK / Unknown / Supported.

**3. Where**
- At-risk rate by customer and by level of service in one table.
- Columns: Orders, NothingTyped, FillerOnly, AtRisk, PctAtRisk.
- Answers "where is the problem" directly. MUSC Ground and BLS are the two answers.

**4. Examples**
- Real orders from each bucket with what the nurse actually typed.
- Makes the categories concrete. This is the review material.
- Check for patient information before sharing outside the team.

**5. Concepts**
- Which clinical reasons and filler terms actually appear, with counts.
- Includes the **search terms used** and the **CMS source** for each, so anyone can audit
  or challenge a category.

---

## Data provenance

| | |
|---|---|
| Table | `` `prod-sandbox`.vivekkumar_patel.temp_tnet_tripmaster `` (read-only snapshot) |
| Built from | `prod.silver_transbroker.triprequest` INNER JOIN `prod.silver_transbroker.tripleg` |
| Built by | Vivekkumar Patel, notebook `TripMaster_V2` |
| Grain | One row per **trip leg**, not per request |
| Field analyzed | `tripleg.ClinicalData` - the nurse's free-text box. **This one field produced the finding.** |
| Fields excluded | `LosQuestions` (form template, ~49k chars, not answers), `SpecialNeeds` (equipment flags) |

**Unconfirmed - verify before publishing:**

1. The date range. The query file says CY2025, but the snapshot may have been run with different
   parameters, and the source has date logic that switches between `TripLog.LastModifiedDate` and
   `RequestDateTime`. Run:
   ```sql
   SELECT min(RequestDateTime), max(RequestDateTime), count(*)
   FROM `prod-sandbox`.vivekkumar_patel.temp_tnet_tripmaster;
   ```
2. When the snapshot was taken. If before year-end, late-2025 orders are missing. Ask Vivekkumar.
3. The scope filter has not yet been re-run through the notebook - headline figures were derived
   by subtracting out-of-scope orders from reported totals.

**Confidence by claim:**

| Claim | Confidence |
|---|---|
| Row counts, customers, facilities | High - direct counts |
| Empty free-text box counts | High - null test, no judgment |
| MUSC 49% vs THR 13% | High - same test both sides |
| `filler_only` and `clear_reason` counts | Medium - depends on unvalidated word lists |
| `unrecognized` count | Low as a finding - measures our vocabulary gap |
| Time period | Unverified |

---

## Next steps

1. Confirm the date range and snapshot date (queries above).
2. Re-run with the scope filter applied and replace the approximate percentages.
3. Read 50-100 `unrecognized` samples, extend the term lists, re-run.
4. Jen Jones / Michelle's team review Group A and Group B terms - especially `behavioral`.
5. Build the golden set from the Examples tab.

**Baseline to beat: ~19% of non-emergent ground orders at risk. 49% at MUSC Ground.**
