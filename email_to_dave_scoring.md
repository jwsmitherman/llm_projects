# Medical necessity — scoring change and how to read the file

Dave,

Updates from Friday's call below, then how to read the new file.

## From the call

**"Nothing higher than a two in the individual columns."**
Correct. Each category is scored 0 to 2: 2 = documented and supports necessity, 1 = addressed but not sufficient alone, 0 = not documented. That is unchanged.

**"The total only goes up to eight."**
This was a defect. We were adding six columns, and two of them could never be 0:

- Level of service is on every order — scored 1 for BLS, 2 for ALS or SCT
- Modifiers (origin and destination) are on every order — scored 1 or 2

Every trip collected 2 to 4 points before any clinical documentation was read. A trip with no documentation scored 2 or 3 out of 12.

**Fixed:** the score is now `Score: documentation`, 0 to 6 — mobility, level of consciousness and medical monitoring only. Level of service and modifiers are still columns on the sheet but are not added in.

**"It's just a straight add."**
It was. It no longer is.

**Billing link.**
No code change. Waiting on the RCM data. One request: when Vic has it, we need to know what it joins on. If it keys on claim or encounter rather than trip request id, we need that field added to the extract now or the join will not be possible.

**Corti APIs.**
Nothing yet. Will review ahead of the 28th.

## Two things we found while making the change

**Isolation scored 0 on every trip.** No question on the order form asks about isolation precautions, so the category can never match. It is still a column but is no longer added into the score. Whether isolation should be asked at order entry is a separate decision.

**One trip was a technical failure, not a bad order.** Trip 10130063 never reached the model because of an API error. It scored 2 and looked like a poorly documented trip. Failed calls now show blank scores with the reason in the Flags column.

## Effect on the trips you reviewed

| Trip | Documented | Old /12 | New /6 |
|---|---|---|---|
| 10129725 | Cardiac monitoring only | 5 | 2 |
| 10131394 | One of three bed confinement elements | 3 | 1 |
| 10131626 | Mobility, consciousness, monitoring | 8 | 6 |
| 10130063 | Technical failure | 2 | blank |

## How to read the file

Row 1 is the group banner and the scale. Row 2 is the headers. Every column filters.

**The score**

- `Score: mobility`, `Score: LOC`, `Score: monitoring` — each 0 to 2, colour shaded
- `Score: documentation` — the three added, 0 to 6

**Reported, not scored**

- `Score: LOS`, `Score: modifiers` — on every order, describe the trip not the documentation
- `Score: isolation` — always 0, no form question feeds it

**The determination**

- `LLM: meets` — yes or no. This is the answer; the score is the working behind it.
- `LLM: summary` — two sentences on the patient's condition
- `LLM: clarifying question` — what we would ask the facility. Blank when nothing is needed.
- `LLM: assessment per question` — per answer: supports necessity or not, the reason, the answer quoted, the citation. Look here when you disagree.
- `AUDIT` columns — flag anything the model asserted that is not in the order

**The record**

- `ePCR` columns — what the crew documented
- `Agreement` — whether the order and the record reached the same conclusion

**Yours**

- `SME: meets`, `SME: notes` — blank, for your determination
- `Flags` — failed calls, no matching record

## Two decisions for you

1. **Scale.** You said it should run to 10. It is 0 to 6 because it is three categories at 0 to 2, so any score decomposes by eye. Say the word if you want 0 to 10.
2. **Isolation.** Should the order form ask about isolation precautions? The criteria recognise it; the form does not ask.

Josh
