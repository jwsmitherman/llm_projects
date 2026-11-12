# Medical necessity — AI assessment of October orders

Dave,

Attached is the AI medical necessity assessment of real ambulance orders submitted through Transport.net on 25 October.

## What we did

We took the actual orders facility staff submitted, sent the form answers and trip details to the AI model, and it returned a medical necessity determination against the CMS criteria. Where a patient care record existed for the same trip, we ran a second assessment against what the crew documented and compared the two.

## The columns

### The order, as entered

- **Trip request id, Ordered, LOS, Trip type** — which trip, when, level of service, origin and destination
- **Duplicate payloads** — whether the order was submitted more than once
- **Questions and answers** — every question asked and what the facility answered
- **Other form info** — diagnosis, weight, Medicare question 10, special needs

### The AI assessment

- **Score: mobility / LOC / monitoring / isolation / LOS / modifiers** — each category scored on its own. 2 = supports necessity, 1 = not sufficient alone, 0 = not documented
- **Score: total** — out of 12
- **LLM: meets** — yes or no
- **LLM: confidence** — High, Medium, Low
- **LLM: summary** — two sentences on the patient's condition
- **LLM: clarifying question** — what we would ask the facility. Blank when nothing is needed.
- **LLM: assessment per question** — for each answer: supports necessity or not, the reason, the answer quoted, and the criteria citation
- **AUDIT columns** — automated checks that flag anything the model said that is not in the order

### The patient care record

- **ePCR columns** — what the crew documented
- **LLM ePCR: meets, summary, documentation gaps** — the determination from the record
- **Agreement** — whether the order and the record reached the same conclusion

### For you

- **SME: meets, SME: notes** — blank, for your determination next to the model's
- **Flags** — processing notes, including where no patient care record matched

## Coverage

Every order is assessed. ePCR coverage is partial — those records are pulled manually by trip ID, so we have the ones that were pulled. Where none matched, the Flags column says so.

Where you disagree with a determination, the per-question assessment shows the exact answer text and citation the model used.

Josh
