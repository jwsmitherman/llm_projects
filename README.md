# Batch processing

The same engine as the review screen, run over a file of trip requests instead of
one at a time. This process does no thinking of its own: it reads the export,
de-identifies every payload, posts it to the service, and writes the workbook.

The service and the review screen are untouched and keep working as they do now.

```
med_nec_api\
  app.py  llm.py  prompts\  static\      the service and the review screen
  batch_processing\
    actual_payloads\                      drop the export here
    batch_results\                        workbooks land here
    batch_process.py                      the entry point
    deidentify.py
    workbook.py
```

## Two commands, two windows

**First**, start the service and leave it running. Same command as always,
from `med_nec_api`:

```powershell
python -m uvicorn app:app --port 8000
```

That is the same service the review screen uses, on `http://127.0.0.1:8000`.
`http://127.0.0.1:8000/ui` still works while the batch runs.

**Then**, in a second window:

```
batch_processing\run_batch.cmd
```

or directly:

```powershell
python batch_processing\batch_process.py
```

Put the export from the Transport.net query in `actual_payloads\` and run with no
arguments. It picks up the newest `.csv` or `.xlsx` it finds. Workbooks land in
`batch_results\`.

It uses the same `.venv` as the service, so there is nothing extra to install.

**Note there are two batch runners and they do different jobs.**
`med_nec_api\run_batch.py` posts a folder of JSON files from `inbox\`.
`batch_processing\batch_process.py` is this one: it reads the production export,
de-identifies it, and writes the review workbook.

## Run the dry run first

```
run_batch.cmd --dry-run
```

De-identifies, writes the workbook with no determinations, and sends nothing. On
a new extract this is the cheapest way to see what the scrubber found before a
line of clinical text leaves the machine.

## What the input looks like

One row per trip, from the query against `O_RequestEventLogs`:

| Column | |
|---|---|
| `RequestPayload` | the raw JSON, `EventJson` from the event log |
| `TripRequestId` | the trip |
| `EventTimeStamp` | optional, but see below |

Other columns are ignored. `--payload-column` / `--trip-column` /
`--timestamp-column` if yours are named differently.

**Include the timestamp.** `O_RequestEventLogs` holds an `EventId = 3` row per
submission, so a trip that was edited and resubmitted appears twice. With a
timestamp the latest submission is kept, which is the order as it finally stood.
Without one the row kept is arbitrary, and the run says so.

## What the model is sent

Everything on the form that bears on medical necessity, not only the question
and answer pairs. Josh, 10 November: *"we need to send all the form info related
to med nec to the llm."*

Before this the prompt carried four lines of context and the pairs. The
diagnosis, the patient's weight, the Medicare questionnaire and every drop-down
on the form were de-identified, kept in the payload, written to the workbook,
and then never shown to the model that had to judge the trip.

The prompt now has three blocks before the pairs:

- **Order details** — diagnosis, patient weight and height, over-max-weight,
  level of service and any override with its reason, origin and destination
  modifiers, `medicareQuestionNums` and `medicareQuestionTen`, special needs,
  requester comment, facility pay reason, distance, duration, one-way /
  recurring / will-call.
- **Other form fields the requester completed** — `additionalFields`, grouped by
  the form's own sections (level of service questions, patient information,
  scheduling, payer). This is where the drop-downs and toggles live: level of
  care requested, reason for transfer, chief complaint, clinical summary notes,
  intubated, number of drips, special equipment, transport following inpatient
  stay, residence and stairs questions.
- **Question and answer pairs** — as before.

A field the form left blank is left out rather than sent as "not stated", so the
prompt carries answers, not a list of empty boxes. `false` is an answer to a
yes/no question and is sent as `no`.

`Form as entered` in the workbook shows the same fields, so the cell and the
prompt hold the same thing.

## De-identification

`deidentify.py`. The rule is ALLOW, not DENY: every field is dropped unless it is
named, because a deny list is only as good as its author's imagination and nobody
on this project had seen a production payload until this week. A payload goes
from roughly 13KB to 2.3KB. What survives is the question and answer pairs, the
level of service, the origin and destination codes, the trip reason, and the free
text boxes, scrubbed.

Three guards, in order:

1. **An unrecognised field stops the run.** It is dropped and reported, and the
   batch refuses to send until someone has classified it in `deidentify.py`.
   `--allow-unknown-fields` proceeds once the list has been read.
2. **Free text is scrubbed against the patient's own name**, read out of
   `patientInfo` before that block is blanked. A requester who typed the name
   into an answer box is caught, which no pattern would do.
3. **Residual scan.** The cleaned payload is searched again for anything that
   looks like an identifier. A hit holds that trip back rather than sending it.

`additionalFields` is filtered entry by entry rather than dropped whole, which
is what used to happen and is why the form's drop-downs never reached the model.
Each entry is kept by `name` against an allow list, reduced to its label,
section, value and selected label, and the free-text ones are scrubbed. An
unrecognised name is dropped **and listed in the Flags column**, so a field
added to the form shows up for classification instead of silently going to the
model or silently disappearing.

`patientInfo` is still blanked, with five clinical exceptions: `diagnosis`
(scrubbed, it is typed by a human), `patientWeight`, `patientHeight`,
`overMaxWeight` and `patientOver300Lbs`. Those are clinical content, not
identity, and dropping them was costing the model the two things a person reads
first. `admitDate` is **not** kept: `inpatStay` is what the Part A question turns
on, and a bare admit date is both a HIPAA identifier and something the residual
scan would hold the trip back over.

## The workbook

**One sheet, one row per trip, 33 columns, in five banner groups** so that what
the nurse entered and what the model concluded are visibly different parts of
the sheet.

| Group | Columns |
|---|---|
| **FROM THE ORDER, as entered** | `Trip request id`, `Ordered`, `LOS`, `Trip type`, `Duplicate payloads`, `Questions and answers`, `Other form info`, `Form chars` |
| **LLM ASSESSMENT OF THE ORDER** | the six `Score:` columns, `Score: total`, `LLM: meets`, `LLM: confidence`, `LLM: summary`, `LLM: clarifying question`, `LLM: assessment per question`, `AUDIT: summary not in order`, `AUDIT: quotes not in order` |
| **ePCR, as documented** | `level of care provided`, `primary impression`, `narrative`, `other fields` |
| **LLM ASSESSMENT OF THE ePCR** | `meets`, `summary`, `documentation gaps`, `Agreement` |
| **REVIEWER** | `SME: meets`, `SME: notes`, `Flags` |

Long columns **wrap**, and every row is the same fixed height — Mukund,
10 November: *"give a fixed distance here, so that it opens up, and then you can
make wrap text, so it's easy to read."* Wrapping on its own made each row as
tall as its longest cell; a fixed height keeps the sheet a grid and shows the
first eight lines or so. Expand a row, or read the cell in the formula bar, for
the rest.

`Questions and answers` and `Other form info` do not overlap. Looking at the old
`Form as entered` beside the Q&A, Josh and Mukund found the same content twice:
*"the same information is here ... we can delete that column."* True of that
payload, where the parser recovered the whole form. So the duplication is gone
rather than the column: `Other form info` now carries only what is **not** in
any question and answer pair — diagnosis, patient weight, Medicare question 10,
the level of service reason, and the drop-downs from `additionalFields`.

### The two AUDIT columns

Mukund, 10 November, on a summary that read *"transported after a suicide
attempt"* where the order said only suicidal ideation: *"ideation is not an
attempt. So I'm just thinking, how is it saying it's an attempt?"* And:
*"let's kind of complete this check that the question and answers align with the
LLM summaries, so that you know we know."*

- **`AUDIT: quotes not in order`** — the prompt requires evidence verbatim, so a
  quote that is not in what the model was sent is checkable with no judgement at
  all. Any quote listed here is a fabrication.
- **`AUDIT: summary not in order`** — clinical **events and interventions**
  asserted in the summary that the order never mentions, with the weaker term it
  was likely escalated from: `suicide attempt (order says suicidal ideation)`.
  Descriptive paraphrase is deliberately left alone — a summary saying "altered
  mental status" where the order says "A&O x1-2" is doing its job. Only things
  that either happened or did not are checked: arrests, intubation, fractures,
  overdose, stroke, seizure, sepsis, haemorrhage, restraints and the like.

Sort on either column and the trips worth auditing come to the top. **Neither
column changes a determination** — they report on one.

### The score

**`Score: documentation` is 0 to 6** — mobility, level of consciousness and
medical monitoring, each 0 to 2. These three measure what the facility actually
documented, so the score has a real zero.

**LOS, modifiers and isolation are reported but not added in.** Dave,
13 November: *"I don't see anything higher than a two in the individual coloured
ones ... and the total score does go up to eight. I'm just guessing you're
adding the scores."*

He was, and that was the defect. Level of service and the two modifiers are read
off the payload, and a production order always carries them — level of service
is never blank, origin and destination codes are never missing — so they scored
1 or 2 on every trip and put a **floor of 2 under every total**. A trip with no
clinical documentation at all scored 2 or 3 out of 12, which reads as partly
documented when it is not documented at all. They stay on the sheet as their own
columns because they describe the trip; they are not a measure of documentation.

Isolation is carried and not scored because **no question on the order form asks
about isolation precautions**. It was 0 on every trip of the 25 October run. It
starts counting the day a form question feeds it. Whether isolation *should* be
asked at order entry is a separate question worth putting to Dave.

Only the three scored categories are colour-scaled.

**A failed call leaves the score cells blank, not zero.** If the model never
answered, the row cannot be read as a weakly documented order; the reason is in
`Flags`. The 25 October run had one of these — a `BadRequest` on the
`temperature` parameter — which scored 2 and was indistinguishable from a thin
order.

Row 1 is the group banner, row 2 the headers, trip id frozen, filter on every
column.

Four categories are derived **here, by keyword** from the words on the order,
not decided by the model, and their scores are rolled up from the findings the
model already returned per question.

**Level of service** and **modifiers** are read off the payload instead: the
order states them. ALS and SCT score 2, BLS scores 1. Modifiers are scored by
the rule already in `prompts\med_nec_knowledge.json` — origin lower on the scale
than destination means the higher level of care tree governs, so an upward
transport scores 2 and a discharge direction scores 1. The scale is read from
that file, not restated. Those two score even on a `--dry-run`.

If the categories are going to carry weight with Anna and Ed they should come
from the model against the criteria instead of from keywords here.

## Patient care records

Put the ePCR extract in `actual_pcr\`. It is picked up automatically, matched to
trips on `TNET_TripRequestId`, and scored by the same model against the record
prompt. A matched record fills the ePCR block of that trip's row and gets an
agreement verdict against its order.

Identifier patterns in the crew's free text, `EPCR_Narrative` above all, are
masked before anything is sent. Which columns the model is allowed to read is
decided by the service, in `prompts\epcr_fields.json`: 50 of them, with payment
method, insurance, addresses and agency names withheld. The file on disk keeps
every column.

A record with `MDM_GoldenRecord` other than `T`, or with no clinical content, is
reported as unmatched and gets no verdict. An absent record is not evidence
against a trip.

### Other sources

```
run_batch.cmd --records ..\..\med_nec_payload_builder\json_payloads3
```

Takes either the folder-per-trip tree the builder notebook writes, or a single
wide csv with a `TNET_TripRequestId` column. Trips with a record get a second
determination and an agreement verdict; trips without are scored on the order
alone and say so.

## Other arguments

```
--limit 5              first five trips, for a quick look
--workers 3            parallel requests. Mind the TPM quota
--base http://...      a service somewhere other than localhost:8000
--timeout 300          seconds; the record pass is the slow one
```
