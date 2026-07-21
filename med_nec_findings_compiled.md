# Medical Necessity Analysis - Findings & Method

**Data:** 104,634 non-emergency transport orders placed through Transport.net in 2025 by
Texas Health Resources and MUSC, across 52 facilities.
**Workbook:** `med_nec_qa_summary.xlsx`

---

## The Goal

Build an AI check that runs **while the nurse is placing the order** and answers one question:

> *Does what this person just typed actually justify the ambulance they are requesting?*

If it does not, the system tells them what is missing before they hit submit - instead of us
finding out 45 days later when the claim is denied.

To build that, we needed three things, and this analysis produced all three:

1. **The rulebook** - what questions does the order form actually ask, and which ones map to
   Medicare's medical-necessity criteria?
2. **The input** - what do nurses actually write, and in what format?
3. **The baseline** - how bad is the problem today, so we can prove the AI improved it?

---

## Executive Summary

**About 1 in 4 orders cannot support medical necessity on the paperwork alone.**

| What we found | Orders | Share |
|---|---|---|
| Nothing typed at all | 24,303 | 23.2% |
| Only vague filler ("general weakness") | 3,413 | 3.3% |
| **At-risk subtotal** | **27,716** | **26.5%** |
| Text present but no recognized clinical reason | 23,979 | 22.9% |
| Genuinely specific clinical justification | 44,233 | 42.3% |
| Specific reason plus some filler | 8,706 | 8.3% |

Four findings worth taking to the business:

- **The two customers are not the same problem.** MUSC Ground leaves the justification box empty
  on **49%** of orders. Texas Health Resources does it on **13%**. Same software, very different
  behavior - which means this is fixable through workflow, not just technology.
- **BLS ambulance is where the exposure sits.** Of roughly 40,000 Basic Life Support orders,
  **9,606 have either no documentation or vague filler only**. BLS is the most-denied ground
  service, and this is the population the front-end prompt would catch.
- **"General weakness" is real and measurable.** It appears in **8,568 orders (8.2%)**. Add
  fall-risk-only language and it is **13,316 orders**. This is the exact pattern the strategy
  deck called out, now quantified.
- **We found the question index nobody had.** 348 distinct clinical questions extracted from the
  order forms, mapping directly onto CMS criteria.

---

## How We Defined Things

This section matters more than the numbers, because every figure above depends on these choices.

### What is a "concept"?

A **concept** is a single clinical idea we search for in the nurse's text. There are 15, split
into two groups.

**Group 1 - supports medical necessity.** A legitimate reason the patient cannot travel by
wheelchair van or private car:

| Concept | What we search for |
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

**Group 2 - documentation quality problems.** Phrases that do *not* justify an ambulance on
their own:

| Concept | What we search for |
|---|---|
| weakness_only | "general weakness", "generally weak", "weak" with no cause |
| fall_risk_only | "fall risk", "unsteady", "deconditioning" |
| nonclinical | "per protocol", "convenience", "no other transport", "family request" |

### Where the concept list came from

Not from the data - that would be circular. It was assembled from three sources:

1. **Medicare/CMS ambulance rules**, including the bed-confined test (cannot get up unassisted,
   cannot walk, cannot sit in a chair - all three required).
2. **The Transport.net strategy deck**, which used "general weakness" as the textbook example of
   insufficient documentation, and "post-CVA hemiparesis, unable to bear weight" as the same
   patient documented correctly.
3. **The medical necessity working sessions**, which identified BLS-versus-wheelchair as the
   judgment call that drives denials.

**This is the caveat that matters: these word lists are our best reading of policy. They have not
been reviewed by Jen Jones or Michelle's team, and they have not been checked against how nurses
at these specific hospitals write.** Treat the percentages as directional until they are.

### What does "specific" mean, exactly?

Every order gets exactly one quality label. The logic runs in this order:

```
Is the free-text box empty?              -> no_documentation
Only quality-problem words, no clinical? -> weak_only
No recognized words at all?              -> unclassified
Clinical reason AND filler both present? -> mixed
Clinical reason only?                    -> specific
```

In plain terms:

- **specific** - the nurse named a real clinical reason and did not lean on filler.
  *"Post-CVA left hemiparesis, unable to bear weight, requires supine transport."*
- **mixed** - a real reason is there, but so is vague language.
  *"General weakness, patient on 2L oxygen and cannot sit upright."*
- **weak_only** - only filler. *"General weakness, fall risk."*
- **unclassified** - something was written, but nothing we look for appeared. **This is a to-do
  list, not a verdict** - 22.9% of orders landed here, which almost certainly means our word
  lists are missing vocabulary nurses actually use.
- **no_documentation** - the box is empty.

### One important limitation

This is a **keyword search, not artificial intelligence.** It cannot tell that "patient cannot
support trunk in vehicle" means the same thing as "unable to sit upright." That is precisely the
gap the LLM fills - and the size of the `unclassified` bucket is the measurement of that gap.

---

## Tab-by-Tab

### Overview
- Population and how completely each field is filled in.
- 77% of orders have clinical free text; 77% have a questionnaire.
- The 23% with nothing typed is the first headline.

### QA Extraction Health
- Proves the questionnaire parsing worked: 7.9M question/answer pairs from 80,293 orders.
- **348 distinct questions** - the master list nobody had indexed.
- Roughly 98 questions per order, which tells you how long these forms have become.

### Question Catalog
- **The most valuable tab in the file.** Every clinical question asked, across both customers.
- The questions map straight onto CMS criteria: *"bed confined before and after transport - all
  three criteria must be met"*, *"unable to sit in a chair or wheelchair"*, *"requires special
  positioning or handling"*, *"deep tracheal suctioning"*, *"requires CPAP"*, *"is intubated"*.
- This becomes the backbone of the AI's rulebook - each question is a testable condition.
- Read it as a **catalog**, not a frequency count (see the known issue below).

### Answer Values
- What nurses actually type or select.
- Three formats the AI must handle: **coded** (`2L`, `3L NC`, `2 LPM`), **short phrases**
  (`Dementia`, `bedbound`, `fall risk`), and **full sentences** (`Patient cannot support trunk in
  vehicle`).
- Wildly inconsistent - `bedbound` / `bed bound` / `Bed bound` are three separate entries for one
  condition. Same for `fall risk` / `Fall Risk` / `high fall risk`.
- **That inconsistency is the single best argument for an LLM over a rules engine.**

### Doc Quality *(trustworthy)*
- The corrected picture, based only on what the nurse typed.
- specific 42.3% | no_documentation 23.2% | unclassified 22.9% | mixed 8.3% | weak_only 3.3%.

### Doc Quality x Contract *(trustworthy)*
- **MUSC Ground: 14,803 of 29,995 orders (49%) have no documentation.**
- **Texas Health Resources: 9,363 of 74,378 (13%).**
- Same platform, four-fold difference. This is a workflow and training story as much as a
  technology one, and it tells you where to pilot.

### Doc Quality x LOS *(trustworthy)*
- Documentation quality by transport type - the denial-risk view.
- **Basic Life Support: 6,713 no documentation + 2,893 weak-only = 9,606 exposed orders.**
- Advanced Life Support: 4,207 no documentation.
- Rideshare shows 9,292 with no documentation, which is expected - a rideshare does not need a
  medical justification. Worth excluding from the risk math.

### Text Concepts *(trustworthy)*
- What clinical reasons actually appear in nurses' writing:
  mobility deficit 21.4% | cannot sit 18.1% | bed confined 15.5% | oxygen 14.9% | cardiac 11.3%.
- Quality problems: **weakness-only 8.2% (8,568 orders)**, fall-risk-only 4.5% (4,748).
- These are believable, well-spread numbers - the honest replacement for the first version's
  broken chart.

### Sample QA / Sample Specific / Sample Weak
- Real examples pulled straight from live orders.
- **Sample QA is literally what the AI will read** - use it to write and test prompts.
- Sample Specific and Sample Weak become the positive and negative training examples.
- Review for patient information before sharing outside the team.

---

## Known Issue: three tabs are not yet usable

**Concept Responses, Needs Per Order, and Needs By LOS** depend on the questionnaire parser,
which has a bug.

- The word **"True" appears 4,062,545 times** out of 4,117,796 total answers - 98.7% of everything.
- No nurse answered "true" four million times. The parser grabbed a behind-the-scenes form
  property (something like *is this question visible*) instead of the patient's actual answer.
- The giveaway: **Needs Per Order shows patients with 83 confirmed clinical needs.** No patient
  has 83 needs.

One detail worth noting anyway: even with inflated numbers, the **ranking** in Needs By LOS is
directionally right - Critical Care (84) > ALS (79) > BLS (75) > Wheelchair (29) > Reclining
Chair (5). Higher service levels do carry more documented need. Once the parser is fixed, this
becomes a clean measure of whether the vehicle sent matches the patient's condition.

**Fix required:** the key-path output from Part A of the notebook, so we can point the parser at
the real answer field. One-line change.

---

## What Happens Next

1. **Fix the parser** - unlocks the three broken tabs and gives a clean per-patient needs count.
2. **Attack the `unclassified` bucket** - 23,979 orders have text we could not categorize. Read a
   sample, extend the word lists, and that number should drop sharply. Whatever remains is the
   genuine case for the LLM.
3. **Validate with the experts** - take the Question Catalog and the concept word lists to Jen
   Jones and Michelle's team. Their sign-off is what turns these from directional to defensible.
4. **Build the golden set** - use Sample Specific and Sample Weak as the seed for a few hundred
   human-labeled orders to test the AI against.
5. **Set the baseline** - 26.5% of orders at risk today, 49% at MUSC Ground. Those are the numbers
   the AI has to move.
