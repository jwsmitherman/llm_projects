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

### Filler
Same as vague. "Filler" = the behavior, "vague" = the language. Three types:

| Type | Examples | Why it fails |
|---|---|---|
| Vague condition | general weakness, generally weak | No cause, no functional limit |
| Risk label | fall risk, unsteady gait, deconditioning | A risk is not a contraindication |
| Non-clinical | per protocol, convenience, no other transport, family request | Logistics, not medicine |

Why it happens: a nurse booking 12 transports before lunch types the shortest thing that gets the
ambulance dispatched. The consequence lands 45 days later in a billing office they never see.

---

## Method

Searched one field only: **`tripleg.ClinicalData`** - the nurse's free-text box.

Excluded `LosQuestions` (it's the full form template, ~49,000 chars - contaminated v1) and
`SpecialNeeds` (equipment flags, not justification).

### Group A - supports necessity (12)

| Concept | Search terms |
|---|---|
| mobility_deficit | hemiparesis, paralysis, non-ambulatory, unable to bear weight, fracture |
| cannot_sit | cannot sit, special positioning, must lie flat, supine, stretcher |
| bed_confined | bed bound, bed confined, unable to get up |
| oxygen | oxygen, O2, LPM, nasal cannula, BiPAP, CPAP |
| cardiac | cardiac, telemetry, EKG, NSTEMI, arrhythmia, afib |
| behavioral | dementia, Alzheimer's, combative, agitated, altered mental status, flight risk |
| isolation | isolation, MRSA, C. diff, precautions |
| wound_ostomy | wound, ostomy, ulcer, decubitus, drain |
| ventilator | ventilator, vent, trach, intubated |
| iv_medication | IV, infusion, drip, heparin, antibiotic, TPN |
| suctioning | suction |
| bariatric | bariatric, morbid obesity |

### Group B - quality problems (3)

| Concept | Search terms |
|---|---|
| weakness_only | general weakness, generally weak, weak (no cause) |
| fall_risk_only | fall risk, unsteady, deconditioning |
| nonclinical | per protocol, convenience, no other transport, family request |

### Labels

| Group A hits | Group B hits | Label |
|---|---|---|
| box empty | - | no_documentation |
| 0 | 1+ | weak_only |
| 0 | 0 | unclassified |
| 1+ | 1+ | mixed |
| 1+ | 0 | specific |

Every order gets exactly one label.

### Where the terms came from

CMS ambulance rules (incl. the 3-part bed-confined test), the Transport.net strategy deck
("general weakness" vs "post-CVA hemiparesis"), and the medical necessity working sessions.

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
means "unable to sit upright." That gap is what the LLM fills - and `unclassified` (26%) is the
measurement of it.

---

## Workbook tabs

**README**
- Source, scope, and what each tab contains.
- Trust: n/a

**Overview**
- Population and how full each field is. 77% have clinical text, 77% have a questionnaire.
- Key number: 23% of orders have nothing typed.
- Trust: yes

**QA Extraction Health**
- Proves the questionnaire parsing worked. 7.9M question/answer pairs from 80,293 orders.
- Key number: **348 distinct questions** - the index nobody had. ~98 questions per order.
- Trust: yes

**Question Catalog**
- Every clinical question the forms ask, both customers.
- Maps straight to CMS criteria: *bed confined - all three criteria*, *unable to sit in a chair
  or wheelchair*, *requires special positioning*, *deep tracheal suctioning*, *requires CPAP*.
- This becomes the AI's rulebook. Each question is a testable condition.
- Trust: yes as a catalog. Ignore the percentages (parser bug).

**Answer Values**
- What nurses actually type or pick.
- Three formats: coded (`2L`, `3L NC`), short phrases (`Dementia`, `bedbound`), full sentences
  (`Patient cannot support trunk in vehicle`).
- Messy: `bedbound` / `bed bound` / `Bed bound` are three entries for one condition.
- **This is the argument for an LLM over a rules engine.**
- Trust: yes

**Concept Responses**
- Answered questions grouped by clinical concept.
- Trust: **no** - parser bug. `other` swamps at 5.5M, `bed_confined` shows 0 confirmed.

**Needs Per Order**
- How many clinical needs were confirmed per patient.
- Purpose: orders with zero confirmed needs are the risk cases.
- Trust: **no** - shows patients with 83 needs.

**Needs By LOS**
- Average confirmed needs by service level.
- Purpose: a critical care transport should show more need than a wheelchair van.
- Trust: **no** on the numbers, but ranking is right: CCT 84 > ALS 79 > BLS 75 > WC 29 > Recliner 5.

**Doc Quality**
- Quality of the nurse's free text. The corrected view.
- specific 42% | no_documentation 23% | unclassified 23% | mixed 8% | weak_only 3%.
- Trust: yes

**Doc Quality x Contract**
- Same labels split by customer.
- Key number: **MUSC Ground 49% empty. Texas Health 13%.** Four-fold gap on the same software.
- Tells you where to pilot.
- Trust: yes

**Doc Quality x LOS**
- Quality by transport type - the denial-risk view.
- Key number: **BLS has 6,713 empty + 2,893 filler = 9,606 exposed orders.**
- Rideshare shows 9,292 empty, which is fine - rideshare needs no justification. Exclude it.
- Trust: yes

**Text Concepts**
- Which clinical reasons actually appear in nurses' writing.
- mobility deficit 21% | cannot sit 18% | bed confined 16% | oxygen 15% | cardiac 11%.
- Quality problems: weakness-only 8.2% (8,568 orders), fall-risk-only 4.5%.
- Trust: yes

**Sample QA**
- Real question/answer pairs from live orders.
- **This is literally what the AI will read.** Use it to write and test prompts.
- Check for patient information before sharing outside the team.
- Trust: yes

**Sample Specific**
- Examples of good documentation. Positive training examples for the model.
- Trust: yes

**Sample Weak**
- Examples of thin documentation. These are what the front-end prompt should catch.
- Pair with Sample Specific for the before/after story.
- Trust: yes

---

## Known issue

Three tabs are broken. The questionnaire parser grabbed a form property instead of the answer:
**"True" appears 4,062,545 times out of 4,117,796 answers (98.7%).**

Tell: Needs Per Order shows patients with 83 confirmed clinical needs.

Note: the *ranking* is still right - CCT 84 > ALS 79 > BLS 75 > Wheelchair 29 > Reclining 5.
Higher service = more documented need. Scale is wrong, direction is correct.

**Fix:** need the Part A key-path output to point the parser at the real answer field. One line.

---

## Next steps

1. Confirm the date range: `SELECT min(RequestDateTime), max(RequestDateTime) FROM ...`
2. Ask Vivekkumar when the snapshot was built and what the WHERE clause was.
3. Apply the scope filter, re-run, use corrected percentages.
4. Read 50-100 `unclassified` samples, extend the term lists.
5. Fix the parser.
6. Jen Jones / Michelle's team review Group A and Group B terms.
7. Build the golden set from Sample Specific + Sample Weak.

**Baseline to beat: ~19% of non-emergent ground orders at risk. 49% at MUSC Ground.**
