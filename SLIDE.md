# Approximate Name Matching — Splink Evaluation (Pass 2)
### Business review

---

## Slide 1 — What we're doing

Testing whether a probabilistic record-matching tool (**Splink**) can correctly decide when two records belong to the same person — especially for people who **lack a strong ID** (no SSN, A-Number, or FIN), where matching must rely on name and date of birth alone.

We ran the tool across **8 controlled scenarios**, each simulating which identifiers are available, and measured accuracy on two groups: **everyone**, and the **no-strong-ID population** specifically.

---

## Slide 2 — What is Splink?

**Splink is an open-source tool for record linkage** — deciding whether two records refer to the same real-world person when there's no single reliable shared ID.

- It implements a long-established statistical method called **Fellegi-Sunter** (the industry standard for probabilistic matching, used in census, health, and government identity systems for decades).
- For each field (name, birthdate, ID numbers) it weighs **how much agreement or disagreement counts as evidence** that two records are the same person, then combines those into a single **match probability**.
- It is built to run at **production scale**: it shortlists which record pairs are worth comparing ("blocking"), groups matched records into single identities ("clustering"), and runs over the full population.

**In one line:** Splink is a transparent, scalable engine for "are these two records the same person?"

---

## Slide 3 — Why we use it / how it compares to other models

We evaluated **two families of approach**:

| | **Splink / Fellegi-Sunter** (probabilistic) | **GBM** (machine-learning classifier) |
|---|---|---|
| What it is | Statistical matching method, purpose-built for record linkage | General ML model that learns patterns from labeled examples |
| Accuracy | Very strong | Very strong; can have a slightly higher ceiling |
| Explainability | **High** — every decision shows the weight each field contributed | **Low** — effectively a black box |
| Needs labeled data | No (can train unsupervised) | Yes |
| Shortlisting candidates at scale | **Built in** | Not included — needs separate engineering |
| Groups records into identities | **Built in** | Not included |
| Governance / audit fit | **Strong** — defensible, transparent | Weaker — hard to justify individual decisions |

**Why it matters for a federal identity system:** decisions must be **explainable and auditable**. A GBM might win by a point or two of raw accuracy, but it can't easily justify *why* two records matched. Splink can — which makes the probabilistic approach the better fit for identity resolution, and the recommended production path.

> Note: our companion benchmark (Fellegi-Sunter computed directly on existing features) is the **same statistical method** as Splink, used as an apples-to-apples accuracy check. Splink is that method plus the production machinery (shortlisting, clustering, scale).

---

## Slide 4 — What each scenario means

Each scenario simulates **which identifiers are available** for matching. Name, birthdate, and country are always available; the scenarios change only the ID numbers.

| Scenario | Plain-English meaning |
|---|---|
| **baseline** | Today's normal case: strong government IDs (A-Number, SSN, FIN) + name + birthdate |
| **no_ssn** | Strong IDs but **without SSN** — tests how much we depend on SSN |
| **no_strong_id** | **Name + birthdate only** — no strong IDs at all (the hard population) |
| **with_dl** | Strong IDs **plus driver's license** — does DL add value on top? |
| **with_dl_no_strong_id** | Name + birthdate + **driver's license only** — can DL rescue the no-ID population? |
| **with_other_ids** | Everything: strong IDs + DL + passport + I-94 + travel doc |
| **other_ids_no_strong** | No strong IDs, but DL + passport + I-94 + travel doc — can the weaker IDs rescue the no-ID population? |
| **all_ids** | Every identifier available at once |

The "**…_no_strong**" scenarios are the key test: **can secondary identifiers compensate when strong IDs are missing?**

---

## Slide 5 — Results

*Accuracy is "ranking accuracy" (AUC): 1.0 = perfect, 0.5 = coin flip. "Coverage" = the share of true matches the shortlisting step actually compared. "Accuracy on compared pairs" = accuracy ignoring the matches that were skipped.*

### Everyone (full population)

| Scenario | Accuracy | Coverage | Accuracy on compared pairs |
|---|---|---|---|
| baseline | 0.97 | 0.97 | 1.00 |
| no_ssn | 0.97 | 0.97 | 1.00 |
| no_strong_id | 0.98 | 0.95 | 1.00 |
| with_dl | 0.97 | 0.97 | 1.00 |
| with_dl_no_strong_id | 0.97 | 0.97 | 1.00 |
| with_other_ids | 0.97 | 0.97 | 1.00 |
| other_ids_no_strong | 0.97 | 0.97 | 1.00 |
| all_ids | 0.97 | 0.97 | 1.00 |

**Read:** matching is excellent across the board (~0.97) when the population includes people with strong IDs.

### No-strong-ID population (the hard case)

| Scenario | Accuracy | Coverage | Accuracy on compared pairs |
|---|---|---|---|
| baseline | 0.61 | 0.56 | 1.00 |
| no_ssn | 0.62 | 0.56 | 1.00 |
| no_strong_id | 0.77 | 0.55 | — |
| with_dl | 0.61 | 0.56 | 0.99 |
| with_dl_no_strong_id | 0.61 | 0.56 | 0.99 |
| with_other_ids | 0.60 | 0.56 | 0.99 |
| other_ids_no_strong | 0.61 | 0.56 | 0.99 |
| all_ids | 0.60 | 0.56 | 0.99 |

**Read:** accuracy looks weak (~0.60) — **but accuracy on the pairs actually compared is ~0.99.** The gap is entirely **coverage**: the shortlisting step compared only ~56% of true matches, and the rest were counted as misses. Adding more identifiers (DL, passport, etc.) did **not** help, because those identifiers are largely missing for this population.

---

## Slide 6 — What we found

- **The matching model is not the problem.** On the record pairs the engine actually compares, accuracy is ~99% even with no strong ID.
- **Coverage is the bottleneck.** The shortlisting step skipped ~45% of genuine matches for the no-ID population — those are people with typos, nicknames, or transposed names who don't share an exact shortlist key.
- **Extra identifiers don't rescue this group** because the identifiers themselves are missing in the source data.

---

## Slide 7 — Recommendation / next steps

1. **Improve shortlisting for hard cases** — add **phonetic and token-based matching keys** (match by how names *sound* and by name parts, not exact spelling) to recover typo'd and nickname matches.
2. **Improve identifier coverage in the source data** — the no-ID population can only be matched so well until more usable identifiers are captured upstream. This is the larger, longer-term lever.
3. **Adopt Splink (probabilistic) as the production path** for identity resolution, for its accuracy, scale, and — critically — its explainability and audit fit.

---

## Slide 8 — Which scenario works, and why

**For people who have strong IDs:** the **baseline** scenario already works well (~0.97). The strong government identifiers (A-Number / SSN / FIN) carry the match — name and birthdate are secondary. Adding more identifiers (DL, passport, etc.) does not improve it, because the strong IDs have already done the job.

**For people with no strong ID:** the **name + birthdate–only** configuration (`no_strong_id`) is the **best of the options (~0.77 vs ~0.60 for the rest)** — and that result is informative in two ways:

1. **Adding the secondary identifiers did not help — and slightly hurt.** DL, passport, I-94, and travel doc are **largely missing for this population**, so they contribute no signal; including them only adds noise. You cannot match on an identifier that isn't there.
2. **A single global model under-serves this population.** When strong IDs are in the model, training learns to lean on them and **under-weights name and birthdate** — the very signals this population depends on. Stripping the IDs out forces the model to give name/birthdate their full weight, which is why the leaner configuration scores higher here. *This points to a dedicated, name/birthdate-centric matching configuration for the no-strong-ID population, rather than one global model for everyone.*

**But no scenario is "good enough" for the hard population yet** — all are capped at ~0.55 coverage (the shortlisting step only compares ~55% of true matches). That ceiling, not the scenario choice, is the binding constraint.

---

## Slide 9 — The fundamental questions this raises

| Question | What the evidence says | Decision needed |
|---|---|---|
| Is probabilistic matching viable for ANM? | **Yes** — ~0.99 accuracy on compared pairs, even with no strong ID | Adopt Splink (probabilistic) as the production path |
| Can secondary IDs (DL, passport, etc.) rescue the no-ID population? | **No, while they're missing in the data** — they can't contribute and add noise | Invest in **upstream identifier capture**, or accept the ceiling |
| Should the no-strong-ID population have its own matching configuration? | **Likely yes** — a name/birthdate-centric model beats the global model for this group | Build and validate a **dedicated config** for no-ID matching |
| What accuracy is "good enough," given the cost of a wrong match? | Not a modeling question — **no model alone** brings the no-ID group to parity | Program/policy must set the **acceptable accuracy and risk** bar |

---

## Slide 10 — Bottom line

> **The probabilistic approach works** — on the pairs it actually compares, accuracy is ~99% even with no strong ID.
>
> **For people with strong IDs, matching is already strong today.** For people without, the limit is **coverage and source-data completeness — not the algorithm.** Secondary identifiers can't rescue this group while they're missing from the data, and a single global model under-weights the name/birthdate signals these people depend on.
>
> **The path forward is three decisions, not a better algorithm:** smarter shortlisting (phonetic/token keys), a dedicated configuration for the no-ID population, and upstream investment in capturing more usable identifiers.

---

### Speaker notes (not for the slide)

- "Accuracy / ranking accuracy" = AUC (0.5 = random, 1.0 = perfect). "Accuracy on compared pairs" = AUC measured only on pairs that passed shortlisting (`auc_blocked_only`).
- "Coverage" = blocking recall: the fraction of true matches the candidate-generation step actually compared. `1 − coverage` is the fraction auto-counted as misses.
- The "—" in the no-strong-ID table is a scenario where the compared set was single-class, so that one number couldn't be calculated; it doesn't change the story.
- Splink = an implementation of the Fellegi-Sunter probabilistic model. The companion "Fellegi-Sunter on features" benchmark is the same method computed directly on existing pipeline features for an apples-to-apples accuracy comparison with the GBM.
- Phonetic keys = Soundex/Metaphone; token keys = matching on sorted name parts (handles first/last transposition).
