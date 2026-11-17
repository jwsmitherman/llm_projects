# Medical necessity — how to read the file

Dave,

The score is a documentation score of 0 to 6 — mobility, level of consciousness and medical monitoring, each 0 to 2 — built from the model's own verdict on each answer. Level of service, modifiers and isolation are columns on the sheet but are not added into the score, because they are on every order regardless of what the facility documented. The score and the determination are both built from the same per-question findings, and a new column names the rows where they conflict.

## The score

- **`Score: documentation` is 0 to 6.** It is mobility, level of consciousness and medical monitoring added together.
- **Each of those three is 0 to 2**, taken from what the model recorded for each answer:
  - **2** — the answer supports necessity
  - **1** — the answer is addressed but not sufficient on its own
  - **0** — administrative, not covered, unclear, or nothing documented in that category
- **A 0 means nothing was documented.**
- The three category columns are colour shaded, so a row reads as a shape before you read a number.

## The score does not decide the determination

- Both the score and `LLM: meets` are built from the same per-question findings.
- The determination then applies what the score cannot: the criteria weights, the exclusions, and the rules that require elements together — bed confinement needs all three.
- A trip can carry three "not sufficient alone" findings, score 3, and correctly not meet.
- **`LLM: basis`** is the model's own line on what drove the determination. That is the link between the score and the label.
- **`Score vs determination`** flags the rows where the two conflict sharply: meets with nothing documented, or does not meet on a score of 4 or more. Those are the rows worth your time.

## Columns reported but not scored

- **`Score: LOS`** — every order has a level of service. It describes the trip, not the documentation.
- **`Score: modifiers`** — every order has an origin and destination code. Same reason.
- **`Score: isolation`** — always 0. No question on the order form asks about isolation precautions.

## Failed calls

- A trip the model never assessed has **blank** scores, not zeros. The reason is in `Flags`.

## The determination

- **`LLM: meets`** — yes or no. This is the answer.
- **`LLM: confidence`** — the model's confidence in that answer, High, Medium or Low. It is not derived from the score.
- **`LLM: summary`** — two sentences on the patient's condition.
- **`LLM: clarifying question`** — what we would ask the facility. Blank when nothing is needed.
- **`LLM: assessment per question`** — per answer: what the model recorded, the reason, the answer quoted, the citation. Look here when you disagree with a determination.
- **`AUDIT` columns** — flag anything the model asserted that is not in the order.

## The order as entered

- **`Questions and answers`** — every question asked and what the facility answered.
- **`Other form info`** — diagnosis, weight, Medicare question 10, special needs.
- **`Duplicate payloads`** — whether the order was submitted more than once.

## The patient care record

- **`ePCR` columns** — what the crew documented.
- **`Agreement`** — whether the order and the record reached the same conclusion.

## Your columns

- **`SME: meets`, `SME: notes`** — blank, for your determination.
- **`Flags`** — failed calls, trips with no matching record.

## Layout

- Row 1 is the group banner and the scale. Row 2 is the headers. Every column filters.

## Decisions for you

- **Scale.** 0 to 6 is three categories at 0 to 2, so any score decomposes by eye. Say the word if you want 0 to 10.
- **Isolation.** Should the order form ask about isolation precautions? The criteria recognise it; the form does not ask.
- **Category weights.** The criteria file weights concepts at 0.3, 1.0 and 2.0 — a stretcher-only finding counts for more than a bariatric one. The score treats them equally. We can carry the weights through if you want the score to rank strength rather than coverage.

## Open

- **Billing link.** When Vic has the RCM data we need to know what it joins on. If it keys on claim or encounter rather than trip request id, that field has to be in the extract before the join is possible.
- **Corti APIs.** Will review ahead of the 28th.

Josh
