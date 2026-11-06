# 120155 — Perform Testing with Production Payloads

## Description

The service has only been tested against the UAT sample payload and hand-written scenarios. This
spike runs real production trip request payloads through it in batch, plus the matching crew
records (ePCR) joined on TNET_TripRequestId, to confirm it works on real data before SME
validation.

Payloads are de-identified before any call is made. Output is a review workbook and written
findings.

**Testing in scope:**

- Schema — the fields the determination depends on are present in production payloads
- Functional — every payload returns a determination, summary, justification and confidence
- Data handling — no patient identifier reaches the model
- Integration — ePCR records match to trips and are scored, producing an agreement rate

**Testing out of scope:**

- Clinical accuracy — needs SME review against a labelled sample. Separate story.
- Performance and load — needs the deployed App Service. Separate story.
- Transport.Net UI integration

**Dependency:** the ePCR and payload extracts currently cover different trips. One needs re-pulling
over matching trip ids before the agreement rate can be produced.

## Acceptance Criteria

1. At least 50 production payloads processed end to end through the service.
2. De-identification report shows no unclassified fields and no identifier reaching the model.
3. Each trip reports determination, summary, justification, confidence, per-question findings,
   tokens and latency.
4. Payload field coverage documented — which required fields are present, which are null in
   production.
5. ePCR matched on TNET_TripRequestId, scored, agreement rate reported, disagreements listed.
6. Failures logged with reason; defects raised where the cause is a service fault.
7. Findings written up with a recommendation on readiness for SME validation.
