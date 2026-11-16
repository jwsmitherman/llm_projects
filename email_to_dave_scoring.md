# Medical necessity — how to read the file

Dave,

The score is now a documentation score of 0 to 6 — mobility, level of consciousness and medical monitoring, each 0 to 2 — so a 0 means nothing was documented. Level of service, modifiers and isolation are still columns on the sheet but are no longer added into the score, because they are on every order regardless of what the facility documented. Trips the model never assessed show blank scores rather than zeros, with the reason in the Flags column.

## The score

- **`Score: documentation` is 0 to 6.** It is mobility, level of consciousness and medical monitoring added together.
- **Each of those three is 0 to 2.** 2 = documented and supports necessity. 1 = addressed but not sufficient alone. 0 = not documented, or asked and answered no.
- **A 0 means nothing was documented.**
- The three category columns are colour shaded, so a row reads as a shape before you read a number.

## Columns that are reported but not scored

- **`Score: LOS`** — every order has a level of service. It describes the trip, not the documentation.
- **`Score: modifiers`** — every order has an origin and destination code. Same reason.
- **`Score: isolation`** — always 0. No question on the order form asks about isolation precautions.

## Failed calls

- A trip the model never assessed has **blank** scores, not zeros. The reason is in `Flags`.

## The determination

- **`LLM: meets`** — yes or no. This is the answer. The score is the working behind it.
- **`LLM: summary`** — two sentences on the patient's condition.
- **`LLM: clarifying question`** — what we would ask the facility. Blank when nothing is needed.
- **`LLM: assessment per question`** — per answer: supports necessity or not, the reason, the answer quoted, the citation. Look here when you disagree with a determination.
- **`AUDIT` columns** — flag anything the model asserted that is not in the order.

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

## Open

- **Billing link.** When Vic has the RCM data we need to know what it joins on. If it keys on claim or encounter rather than trip request id, that field has to be in the extract before the join is possible.
- **Corti APIs.** Will review ahead of the 28th.

Josh
