# ANM — Does Splink work with limited or no IDs?
### Business review

---

## Slide 1 — The Question

**Does Splink correctly match people when we have limited identifiers — or none at all?**

- Some records have **strong IDs** (A-Number, SSN, FIN). Matching those is straightforward.
- Many records **don't**. For those, matching has to rely on **name and date of birth** alone.
- We tested Splink across **8 scenarios**, each simulating which identifiers are available, and measured results for **everyone** and for the **no-strong-ID group** specifically.

> The question is not "which identifiers are best." It is: **does this tool work when the identifiers aren't there?**

**Status:** This is an **experiment**. Splink is approved for experimentation only — not production. These are early findings intended to build understanding of the tool.

---

## Slide 2 — What Splink Is (and how it decides)

**Splink is an open-source record-matching tool that runs in our Databricks notebooks. It answers one question: are these two records the same person?**

It works in **two steps**:

### Step 1 — Blocking (the shortlist)
**What it does:** Decides which record pairs are even worth comparing.

- Comparing every record to every other record would be billions of comparisons — too slow to run.
- Instead, Splink only compares records that already share something in common (for example, same last name + birthdate).
- We use **several overlapping shortlist rules**, so a match is still caught when one field is wrong:

| Shortlist rule | What it catches |
|---|---|
| First name + birthdate | Clean, obvious matches |
| Last name + birthdate | Matches where the first name differs (e.g., "Bob" vs "Robert") |
| First + last name (no birthdate) | Matches where the **birthdate has a typo** |
| Partial last name + birthdate | Matches where the **last name is misspelled** |
| Last name + birth **year** only | Matches where the **month/day is wrong** |
| First name + birth **year** only | Same, from the other direction |

**Why it matters:** If a pair doesn't make the shortlist, it is **never compared** — and counts as a miss. Blocking sets the ceiling on how many real matches we can possibly find.

### Step 2 — Scoring (the decision)
**What it does:** For each shortlisted pair, weighs how much each field agreeing or disagreeing counts as evidence, then produces a **match score**.

- Rare agreements count for more (a shared uncommon surname is stronger evidence than a shared "Smith").
- Every decision is **explainable** — you can see exactly which fields drove the match.

> **Blocking decides what gets looked at. Scoring decides the answer.**

---

## Slide 3 — The Results

*Three numbers matter:*
- **Coverage** — of the true matches, how many had enough information to be shortlisted and compared at all.
- **Accuracy when decidable** — of the pairs it compared, how often it was right.
- **End-to-end accuracy** — the real-world number: pairs it never compared count as misses.

### Headline

| Population | End-to-end accuracy | Coverage | Accuracy when decidable |
|---|---|---|---|
| **Everyone** | 0.97 | 0.97 | **0.99** |
| **No strong ID** | 0.63 | 0.56 | **0.99** |

### What this says

- **With strong IDs:** matching works end to end — **97%**.
- **With no strong ID:** when Splink could compare the pair, it was right **99% of the time**. The matching itself does **not** degrade when the IDs disappear.
- The lower end-to-end number (0.63) is **entirely** explained by coverage: only **56%** of those true matches were shortlisted. The other 44% were never compared, so they count as misses.

> **The scoring works without IDs. The shortlist is what falls short.**

---

## Slide 4 — The Answer

**Does Splink work with limited or no IDs? — Yes, the matching works.**

- On pairs it can compare, Splink is **~99% accurate even with no strong ID** — using name and date of birth alone.
- Where we lose accuracy, it is because the pair **never made the shortlist**, not because Splink judged it wrong.
- **The constraint is coverage, not capability.**

**To close the gap:**
1. **Better shortlist rules** — phonetic and nickname matching (match on how a name *sounds*, not just how it's spelled) so misspellings and nicknames still get compared.
2. **More complete source data** — the more usable information captured upstream, the more pairs become decidable.

> **Bottom line: Splink can match people without strong IDs when it sees the right candidates. Our next gain comes from candidate selection — not from a different matching tool.**

*This is an early experimental finding. It is not a recommendation to deploy — see the next slide for what production would require.*

---

## Slide 5 — What This Would Look Like in Production (and its limits)

### The gap between the experiment and production

Splink is a **batch tool**. It is built to take a **large, static set of records**, shortlist candidate pairs across the whole corpus, score them, and group matches into identities. That is not how our production matching works — **PCIS matches one incoming record against the existing population, in real time.**

Splink *can* do pairwise matching if we feed pairs in, but that is not what the package is designed around.

### The proposed production approach

| Step | What we would do |
|---|---|
| **1. Train once** | Build a **baseline model** trained on a large sample of records from party index — not retrained per request. |
| **2. Save the weights** | Persist the learned field weights. The trained model is the asset; we don't recompute it at call time. |
| **3. Use our own candidate selection** | **Do not use Splink's blocking.** Use the candidate selection already in **MLS**, and send in only those records. |
| **4. Force full comparison** | In Splink, block on a **constant column**, so every candidate we send is always compared against the incoming record. |
| **5. Score pairwise** | Send pairwise records in, get a match response back. |

> In short: **our blocking, Splink's scoring.** The trained weights are what we carry to production — not Splink's batch workflow.

### Limitations and open questions

- **Coverage moves upstream, it does not disappear.** In production the shortlist becomes **MLS candidate selection**. If MLS doesn't surface the right candidate, the model never sees it — the same coverage constraint we measured, in a different place.
- **This experiment ignores the production stack.** It answers one narrow question: *given these records, can Splink find the match when IDs are swapped out?* It does not test integration, latency, or serving.
- **Our scenarios retrain per scenario.** Each of the 8 scenarios trains a fresh model — closer to "training the model 10 different times" than to asking how **one production model** behaves when identifiers are missing. The pairwise, single-baseline-model setup above is the more production-realistic test, and it is **more work than what we have done so far**.
- **Not yet validated for production use.** Compatibility with our stack, stress testing, and validation all remain.

### Where we actually are

- **Splink is approved for experiments only. It is not approved for production.**
- We are **early** — we have just gotten hold of the tool and are learning its capabilities.
- Realistically, **months** of testing and validation stand between here and any production decision.

> **The encouraging finding:** when Splink sees the right candidates, it matches very well — including without strong IDs. **The work ahead is the blocking / candidate selection, and proving the tool out against our production stack.**
