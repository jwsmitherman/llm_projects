# Identity Search Relevance on OpenSearch
### Finding the right person — fast, even with imperfect information

*[Your name] · PCIS · [Date]*

*Talk track: This is what we've built to make identity search return the right match at the top, and where it's headed.*

---

## The Problem

- We match a person against millions of identity records
- Real-world inputs are messy: misspellings, missing middle name, transposed dates, different name orders
- The old search scored each field **on its own** — so a record matching just one field could outrank the right person
- Analysts didn't always see the best match at the top

*Talk track: The core issue wasn't bad data — it was that the old ranking rewarded single-field matches instead of the whole picture.*

---

## What OpenSearch Is

- The open-source **search engine** that holds the identity index and ranks results
- We send it a person's details; it returns the most relevant records, scored
- Our work lives in **how we ask it to rank** — not in the engine itself

*Talk track: Think of OpenSearch as the engine; the value we add is the ranking logic we feed it.*

---

## What We Built

- A smarter ranking approach that scores **combinations** of fields together, not fields in isolation
- The more complete and exact the match, the higher it ranks
- Net effect: **the right person rises to the top**

*Talk track: One sentence — we reward matching the whole identity, not just a piece of it.*

---

## How the Ranking Works (plain English)

- A ladder of match-quality tiers:
  - Exact full name (first + middle + last) → ranked highest
  - Exact first + last → next
  - Partial / fuzzy name matches → lower
- Identifiers — A-number, date of birth, country — act as **tie-breakers** between otherwise-equal matches
- **Names decide the rank; identifiers refine it**

*Talk track: The tiers set the order of magnitude; identifiers just separate near-ties.*

---

## How We Test & Improve It

- A lightweight **testing tool** runs real searches and shows exactly why each record ranked where it did
- The ranking logic is a reusable **template** — we adjust it, test it, and hand it to the API team
- Changes are checked against known cases, so improving one search doesn't quietly break others

*Talk track: This is how we avoid "whack-a-mole" — fixing one case while degrading ten others.*

---

## Where It Stands

- The new relevance is **live and stable**; stakeholders are seeing materially better matches
- Known limitation: date-of-birth fuzzy matching — a planned future improvement
- Knowledge transfer underway so the team can own and extend it

*Talk track: It's working well today; we're documenting it so it's not a one-person dependency.*

---

## What's Next

- **Quantify the win** — large-scale old-vs-new comparison to put a real number on the improvement
- **Rollout & cleanup** — consolidate to a single search endpoint and retire the old ones
- **Future enhancements** — nicknames/variants, phonetic matching, regional name rules, better date-of-birth indexing

*Talk track: Prove the improvement with data, simplify the footprint, then keep raising match quality.*

---

# — Deeper Dive (Appendix) —

*Optional slides for a technical audience — skip in a leadership readout.*

---

## How We Boost Certain Fields

- Ranking is a ladder of match tiers; each tier carries a **boost** (a score multiplier)
- Approximate ladder:
  - Exact first + middle + last → boost **~10,000**
  - Exact first + last → **~1,000**
  - Full-name cross-field / partial / fuzzy → **~20–100**
- Boosts are spaced by **orders of magnitude**, so the tier that matches sets the score's magnitude
- Identifiers — A-number, DOB, country — carry **tiny** boosts (~0.02–0.05): pure tie-breakers
- **Names decide the rank; identifiers refine it**

*Talk track: The gaps between tiers are intentional — they guarantee a better match type always wins.*

---

## The Query at a Glance

- One big **boolean** query; at least one tier must match (`minimum_should_match: 1`)
- Each tier targets specific fields:
  - Exact names → first / middle / last (keyword)
  - Fuzzy recall → generated full-name permutations + analyzed name fields
  - A-number (fuzzy), date of birth, country of birth / citizenship
- It's a reusable **template** with placeholders — `{{FIRSTNAME}}`, `{{LASTNAME}}`, `{{ANUMBER}}`, …
- Fields the search doesn't supply are **pruned out** before the query runs

*Talk track: One template, parameterized — we fill in what we have and the rest is trimmed automatically.*

---

## What We Learned in KT

- **Pull the template from the API repo** — the testing UI's built-in default is stale and invalid
- A search **score only means something within one result set** — never compare scores across searches
- **No middle name?** The combined full-name tiers silently drop out — a pruning quirk to watch for
- **DOB matching is limited today** (text indexing); the real fix needs a reindex — future work
- **Field types and shard count are locked at index creation** — changing them means a reindex

*Talk track: These are the non-obvious lessons that keep us from breaking relevance while improving it.*
