# CPMS Non-Identity Records: Should We Lower the Match Threshold?

*Jira: IDEN-43344 (Spike) | PCIS Identity ART*

---

## Slide 1 — The Problem

- 1.1 million CPMS party records cannot currently form an identity.
- They are matched against existing people by a scoring model. Today a score of **0.98 or higher** is required to call it a match.
- **85%** of these records score as a "near match" — just below the cutoff.
- That means we are likely missing real matches (false negatives).
- **The question:** should the threshold be lowered to 0.90, and what breaks if we do?

> **Speaker notes:** CPMS records are created in response to other filings or encounters, so they are blocked from forming a standalone identity today. The business wants them to form one when they don't match an existing person.

---

## Slide 2 — Why the Records Get Stuck

The model needs strong ID numbers to be confident. Those fields are mostly empty:

| Field | Filled in |
|---|---|
| Name | 100% |
| Date of birth | 100% |
| SSN | 25% |
| Fingerprint ID (FIN) | 7% |
| A-number | 6% |

**Result:** two records for the same person can agree on name and date of birth, but with no A-number or SSN to confirm it, the score lands just under 0.98 and we miss the match.

**But lowering the bar also lets wrong matches through.** Example: a father and son sharing an A-number were merged into one identity. Merging two real people is more damaging than missing a match.

> **Speaker notes:** Sparse strong identifiers are the root cause of the false negatives — pairs cannot accumulate enough match weight to clear 0.98, so they pile up in the 0.90–0.98 band.

---

## Slide 3 — What the Analysis Will Deliver

- **Who is affected** — which records are stuck and which fields they are missing. Confirms the cause is missing ID numbers, not a broken model.
- **Gain vs. risk at each cutoff** — how many true people we correctly reunite, versus how many wrong merges we create.
- **The hidden risk** — one bad link can fuse two real identities together. We count those merges directly; a simple error rate hides them.
- **A recommended rule** — accept a lower score *when an ID number backs it up*; hold the rest for review. Recovers the matches without inheriting the risk.

**Bottom line:** the goal is not simply a lower threshold, but a defensible one, with a guardrail on the merges that would do real harm.

> **Speaker notes:** Maps to the EDA notebook — fill-rate and sentinel analysis; precision/recall vs threshold; cluster-level over-merge and identity net gain; tiered rule accepting 0.90 only when a strong identifier corroborates.
