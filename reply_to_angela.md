# RE: Transport.Net – UI Payload Response Confirmation & Second Return to AI Med Necessity App

Hi Angela,

Answers below, and the response contract is attached.

## 1. Recheck — how we know, and what to send

The service does care. On a recheck the model is given the question it asked and the requester's answer, and it will not ask a second question. So we need two fields:

```
"clarifyingQuestion": "the question we returned on the first call"
"clarifyingAnswer":   "what the requester typed in response"
```

Both populated is how we know it is a recheck. Send the rest of the payload unchanged.

If you prefer a separate route, `POST /v1/medical-necessity/reevaluate` takes:

```json
{
  "tripRequest": { "...original payload, unchanged..." },
  "clarification": { "question": "...", "answer": "..." }
}
```

## 1a. `previousResponse` — not needed

We do not need the first response back. The question and the answer are the only parts the model uses.

One thing that would help if it is easy on your side: echo back the `evaluationId` we returned on the first call, as `priorEvaluationId`. We would use it to link the two calls in our audit log so a reviewer can see the first determination, the question, the answer and the second determination as one record. Not required — the recheck works without it.

## 2. Response payload

Attached. The three fields the UI needs:

- `meetsMedicalNecessity` — true or false
- `clinicalSummary` — display text
- `clarifyingQuestion` — string, or null. Null means show nothing and let the submission continue.

`detail.questionFindings[]` has one entry per question with the evidence, reason and citation if you want to show the reasoning. `meta` is for logging.

## 3. One question back on `questionIds`

You mentioned `questionIds` will be blank on a full submission check and populated for specific question checks. Today the service evaluates the whole trip and does not read `questionIds`.

Medical necessity is a trip-level determination — it depends on all the answers together, not any one in isolation. So for the specific-question mode, what do you expect back? If it is a trip-level `meetsMedicalNecessity`, we cannot compute that from a subset of questions. If it is per-question findings only for the listed ids, we can do that, but it would be a different response shape and we should define it before your developer builds to it.

Can you describe the user flow where that mode is used? That will tell us which it should be.

Thanks,
Josh
