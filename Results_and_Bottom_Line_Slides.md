# Results
## No-strong-ID population (the hard case)

**Read:** Accuracy looks weak (~0.60) — but accuracy on the pairs the model could actually decide is ~0.99. The gap is entirely **coverage**: only ~56% of true-match pairs had enough information to be decided, and the rest defaulted to misses. Adding more identifier features (DL, passport, etc.) did **not** help, because those identifiers are largely missing for this population.

### What this measures (not a search — a scored set of pairs)
- A fixed set of **labeled record-pairs** (99,863 total → 75/25 split → **9,480 no-strong-ID test pairs**, the scope of this slide).
- Each pair carries 11 precomputed **similarity features** (`f_dob`, `f_cob`, `f_anum`, `f_ssn`, `f_dl`, `f_passport`, …) plus a true match / non-match label.
- A **scenario** = the *same* pairs, with the model allowed to use a different subset of features (an ablation). Name features (`f_name`, `f_dob`, `f_cob`, `f_cartos`) are always on.

### What the three metrics mean (within these pairs)
- **Coverage** — of the true-match pairs, how many had enough present information to be **decided** (vs. undecidable → counted as a miss).
- **Accuracy on compared pairs** — of the pairs that *could* be decided, how often the call was correct.
- **Accuracy** — end-to-end over all in-scope pairs, undecidable ones counted as misses → **Accuracy ≈ Coverage × Accuracy-on-compared-pairs** (~0.56 × ~0.99 ≈ ~0.60).

### Why coverage caps at ~56%
- This population has **no strong ID** (no SSN / A-number), so the strong-ID features are null and the model leans on name + DOB.
- Pairs where those weak signals don't resolve **can't be confidently decided**, so they default to non-match — that's the ~44% gap.
- Extra ID features (DL, passport, I-94, travel doc) are **mostly blank** here, so turning them on adds no signal — every `*_ids` scenario stays flat.
- **Takeaway:** the limit is **information availability for this population**, not the model's decisions — which are ~99% correct when it has something to go on.

| Scenario | Features added (beyond name) | Accuracy | Coverage | Accuracy on compared pairs |
|---|---|---|---|---|
| baseline | anum, ssn, fin | 0.61 | 0.56 | 1.00 |
| no_ssn | anum, fin | 0.62 | 0.56 | 1.00 |
| no_strong_id | (name only) | 0.77 | 0.55 | — |
| with_dl | anum, ssn, fin, dl | 0.61 | 0.56 | 0.99 |
| with_dl_no_strong_id | dl | 0.61 | 0.56 | 0.99 |
| with_other_ids | anum, ssn, fin, dl, passport, … | 0.60 | 0.56 | 0.99 |
| other_ids_no_strong | dl, passport, i94, travel | 0.61 | 0.56 | 0.99 |
| all_ids | all identifiers | 0.60 | 0.56 | 0.99 |

---

# Slide 10 — Bottom line

**The method works.** When it has enough information to make a call, it gets the match right about **99% of the time** — even for people with no strong ID.

**The problem is missing information, not the method.** People with a strong ID (SSN, A-number) already match well. For people without one, we often don't have enough to go on, so many matches get missed. Adding other IDs (driver's license, passport) doesn't help, because those are usually blank for this group too. And one model built for everyone leans on strong IDs, so it under-uses the name and birth-date clues these people depend on.

**The fix is three choices, not a smarter algorithm:**
- **Better candidate matching** — match on how names *sound* and on name pieces, not just exact spelling.
- **A separate setup for the no-ID group** — tuned to lean on name and birth date.
- **Capture more usable IDs upstream** — so the data has more to match on in the first place.

---

## Speaker notes (not for the slide)

*(Add talk track here.)*
