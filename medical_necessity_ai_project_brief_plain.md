# Medical Necessity AI - Project Brief

Consolidated from the workshop transcript and whiteboard photos. Purpose: get everyone aligned on the objective, the process and data landscape, and concrete next steps so we can start building.

## 1. The Objective

Build AI capabilities into transport.net (T-Net), GMR's digital platform for ordering and tracking medical transport, to determine medical necessity at the time a non-emergency interfacility transport (IFT) is ordered, so we send the right level of service and produce a defensible PCS (Physician Certification Statement) up front.

The thesis: catching medical necessity problems at the point of order reduces downstream denials, eliminates the proof of mailing busy-work, and improves margin. Estimated value of the medical necessity piece alone is roughly 20 to 25 million dollars per year (prior auth is believed to be even larger). Treat that number as internal and unverified. Confirming it is part of the work.

Scope guardrails for now:

- Ground only. Air is explicitly out. We do not want to add friction to "I want a helicopter" requests, and we will not have enriched data there yet.
- Non-emergency IFT only. Medical necessity matters far less on the 911 or scene side.
- transport.net customers only for the first data pull and pilot. Start with MUSC and Texas Health (both on T-Net). HCA Virginia is the cautionary paper-based example, not a build target.
- Do not loop in legal or compliance prematurely on the idea of hiding irrelevant questions, but stay compliant. There is a real requirement that users be given the option to select clinical items.

## 2. What Medical Necessity Actually Is Here

The PCS (Physician Certification Statement, recently renamed MNC or Medical Necessity Certification, called EPCS when electronic) is a CMS-required document, signed at time of order by a physician or designee (nurse, case manager, discharge planner, but not a clerk), documenting the clinical condition that justifies the ambulance level of service (wheelchair, BLS, ALS, or CCT).

The pain it creates: If a PCS is missing or does not meet medical necessity, billing cannot bill a third party. They trigger proof of mailing (POM or PLM), which means snail-mailing a blank PCS to the hospital, waiting about 21 to 30 days (hospitals almost never respond), then billing anyway. It is pure delay and busy-work, and the whole thing is audit-after-the-fact policing. CMS audits a market every couple of years and we have to show the proof of mailing trail.

The scale of the problem: A roughly two-years-ago audit at HCA Virginia (paper-based) found 40 percent of PCS documents were invalid, a mix of medical necessity failures and demographic errors. Billing only records one rejection reason even when several apply, so denial data is noisy.

The core human behavior driving errors. Nurses and case managers in transfer centers transcribe from the EHR and take the path of least resistance, clicking the minimum to get the ambulance they want:

- They type "general weakness" in free text, which does not meet CMS medical necessity. But "post-CVA hemiparesis" would. This single pattern may touch about 20 percent of trips.
- They hit the "easy button" (for example, the severe dementia or Alzheimer's checkbox) to guarantee an ambulance even when inappropriate.
- The single most important field is "patient requires special positioning" (cannot sit in a wheelchair, must lie on a stretcher). This is the key BLS-ambulance versus wheelchair discrimination.

Central tension: improve accuracy without adding friction that pushes customers off the platform back to phone and paper. Getting paid for every trip is worthless if people refuse to use the system.

Why it is hard: scope of practice varies by market. Level-of-service rules are market and state specific and configured per contract. Example: deep tracheal suction is ALS in Texas but CCT in Tennessee. So the medical necessity determination depends on the market's business rules. Those rules are an input, not a constant.

## 3. The Process and Data Landscape

The value stream is Call to Care to Claim:

- Call. System: transport.net. Order intake. Only about 20 percent of IFT is ordered online; the rest is phone. Voice AI is a separate parallel effort to digitize that.
- Dispatch. Systems: CAD, which is Logis for ground and TCS for air. Standardizing onto Logis. Markets and regions are compartmentalized.
- Care. Systems: ePCR, which is mostly ImageTrend, plus ESO and legacy Meds. Crew clinical documentation. Must be associated to a CAD record before submit.
- Claim. Systems: Billing, which is Integra for ground and Sweet plus Billing for air, with RevQ as the data store. Integra is a separate company using our tech. Coders assign ALS, BLS, and mileage codes with geography-based rates.

Key data-flow facts:

- The "happy path" is one request to one assignment to one transport to one clinical doc. Reassignments break ID continuity (the data attribution problem). The initial T-Net request ID often is not the final run ID downstream.
- CODS is a proprietary on-prem warehouse (about 15 years old) aggregating all CAD data, feeding OPAP, the analytics platform. Normalization is poor and inconsistent across markets (for example, the priority field means different things in different markets). Be very cautious using non-normalized fields for prediction.
- Checkpoint (now renamed) is the reconciliation matching CAD trips to PCRs so every billable trip has clinical documentation.
- The MDM mapping tool (Michael Abner, on Doug's team) links OPAP to Meds to RevQ IDs. It is claimed to be about 99 percent match where present, but with missing matches. It was built about two years ago and needs a "what has changed" refresh. This is the key to pulling end-to-end data by ID.
- There are 29 EHR connections across Epic and Cerner via HL7 FHIR, a genuine competitive asset. When integrated, payer and demographics auto-fill.

The transport.net clinical form (where we will intervene):

- Fully manual, heavily configurable per customer and contract (sales sold "configure however you want"). Moving toward standardization, but still very custom.
- Questions are not indexed or standardized. The same question has different IDs across customers. Standardizing and indexing the questions is a prerequisite step.
- Business rules (which answer leads to which level of service or special equipment) are configured in the database by developer AJ, from Smartsheet configs maintained by Philip's team after sitting with local operations. The Smartsheet does not link back to the system rules and has no update process, which is a known gap.
- Data is stored in two columns:
  - A giant JSON, which holds rendering and visualization config plus which data point is stored (can be hundreds of thousands of characters). It describes the form setup, not cleanly the answers.
  - A "blurb", which is the unstructured narrative that lands on the PCS.
- Critical limitation: the stored data does not cleanly associate question to answer, and does not indicate whether a level of service was rule-derived or user-overridden. Past work required parsing narratives for keywords (for example, "contains oxygen"). We may need to change how output is stored.
- The form today generates a verbose, low-quality PCS narrative ("patient requires X, patient requires Y") that is about 10 years old.

Technical environment (good news):

- transport.net is a .NET app hosted in Azure (cloud, not on-prem), so latency and data-sharing for an LLM are workable.
- Azure OpenAI is available. The LLM would be included in the Git repo and CI/CD, with resources built in Azure.
- Note: Ernie's Voice AI also feeds into transport.net, so architecture needs to stay consistent across both.

## 4. The Build: MVP and Approach

Three MVP features (from the whiteboard):

1. Medical necessity assessment (textfield plus submit) for MUSC, SC, and Texas Health. Inputs: patient records (insurance and clinical), payer and CMS knowledge base, GMR rev-cycle rules, and denial patterns (to be determined).
2. Guidance flow: recommendation, reasoning, and confirmation, plus highlighting problem text fields. Priority discrimination is BLS versus wheelchair versus stretcher.
3. PCS summary modification: turn the messy JSON and blurb into a crisp, succinct summary on the PCS. Seen as the easiest early win.

How the assessment should work (the design debate, resolved direction):

- Stream the free-text field to an LLM on field-blur or off-focus, not keystroke by keystroke.
- If it fails medical necessity, block submit and return succinct, actionable, conversational feedback right there. For example: "general weakness does not meet CMS's definition; why are they generally weak?" with suggested options and CMS-definition links. Avoid turning fields red and forcing scroll-backs, which is too much friction.
- Alternative or fallback intervention point: intercept at submit (when the trip is written to the database), run the check, then either pass through or pop a review. Easier to implement first; may not be where we go live.
- Leadership preference: validate earlier in the process, but only on free-text fields for now.
- Start narrow. Even just catching "general weakness" would move the needle on about 20 percent of trips.

What the AI needs:

- A RAG knowledge base of the medical necessity rules and policies, so it can both flag problems and recommend the fix. Sources: CMS criteria, GMR and Integra internal rules, and market business rules.
- A golden or labeled data set of roughly a couple hundred to about 1,000 human-labeled trips, categorized medically necessary or not, used to test and measure the model, not to train it.
- Careful prompt engineering plus domain-specific knowledge bases per focus area.

Longer-term vision (not MVP, but the direction): Auto-generate the PCS from EHR data (pull PT notes, flow sheets for oxygen and vent settings, recent nursing and physician notes). Pass medical necessity context into the crew's PCR so they see what the hospital said. Real-time clinical nudges to crews. Propensity-to-pay modeling. HCC-style risk scoring as a differentiator. Eventually becoming a trusted medical necessity determination platform that payers rely on. Adjacent efforts: electronic prior authorization (real-time clearinghouse via TevixMD, RPA or API into payer portals, e-prior-auth and Da Vinci, the January 1 HL7 FHIR mandate) and facility billing (a separate team).

## 5. The Data Strategy and EDA

Two parallel tracks:

1. EDA to size the prize. Quantify the denial bucket, split controllable versus uncontrollable, then do feature analysis and profiling. Caveat: denial data is dirty. Only one reason is captured even when several apply, and a denial does not equal a medical necessity failure. Do not over-trust it; corroborate with the experts' rules.
2. Intelligence and rules. Capture CMS criteria and GMR's internal criteria. The real source of truth for the rules is the medical necessity and audit team (they are the ones who send PCSs back), supplemented by training manuals and a lot of tribal knowledge.

Recommended data pull: filter to a couple of T-Net markets and customers (MUSC, Texas Health), pull PCS, PCR, and denial info directly from rev cycle, using MDM IDs (via Michael Abner) to get the raw structured data rather than OCR-ing PDFs. Then human-label a ground-truth set.

## 6. Key People and Stakeholders

Project and build team:

- Mukund Sridhar. Leading the engagement and EDA approach.
- Patrick Boone. Data analytics specialist. Worked the original Medical Necessity Project (about 2024, roughly six months). Owns the data pipeline knowledge. Chicago.
- Josh Minerman. New. Healthcare AI background (home health, hospice, rehab). Dallas.
- Philip Archer. Implementation manager for transport.net. Builds and configures the clinical forms with health systems. East Tennessee.
- Vic Patel. Data product manager. Building reference data sets. Florida.
- Jaron Polson. Data analytics, working with Patrick. Southeast Idaho.
- Dave. Product and vision lead driving much of the long-term direction.
- Alliance lead for digital products for GMR transport (transport.net, Voice AI, and others). Name to confirm from the intro round.

Business and domain owners:

- Suzie. Owns the overall org (medical necessity, prior auth, facility billing).
- Jen Jones. Heads in-house medical necessity. Reports under her: Michelle Hightop (medical necessity and PCS manager), Robin (denials), plus a prior-auth manager (name to be determined).
- Sean Notary. Facility billing.
- Dr. Costello. Air denial process (separate; uses Appian plus Macedon consultant for appeal letters).

Technical:

- Angela. transport.net dev and product.
- AJ. Developer who configures the clinical rules in the database.
- Michael Abner (Doug's team). Built the MDM ID-mapping. First stop for the data pull.
- John. Databricks.
- Ernie. Voice AI feeding into transport.net.
- Integra. Separate billing company (ground) running on our tech.

Customers: MUSC and Texas Health (pilot targets). HCA Virginia (the 40 percent audit example). Barnes-Jewish St. Louis (went to RFP, competitive pressure).

Vendors, competitors, and tools: Cordy (coding-LLM vendor, being dropped). Movi or Moby and Round Trip (competitors). Milliman (air market data). TevixMD (clearinghouse). Appian and Macedon (air denial letters).

## 7. Next Steps

Scheduled (from the whiteboard timeline):

- June 17. Ernie plus DL to connect Jen, Patrick, and Phil.
- June 24. Rev Cycle plus transport.net (Angela).
- July 1. Next checkpoint.
- July. Multi-day workshop with key leaders from medical necessity, prior auth, and facility billing.

Immediate actions:

1. Meet the medical necessity and PCS team (Jen Jones and Michelle Hightop, possibly Robin for denials, and Integra). Get their training manuals, policies, and the documented rules. Distinguish two rule sets: PCS-at-order review (billing and coding flavor) versus PCR medical necessity (clinical flavor, crew-written).
2. Reconnect with Michael Abner and Doug to learn what has changed in the MDM mapping over two years, then pull an end-to-end dataset for a couple of T-Net markets (MUSC, Texas Health): PCS, PCR, and denial info by ID.
3. Get a full transport.net product demo and a session with the dev team, Angela, and product on how the clinical form, JSON, and database rules are built, and get direct contacts.
4. Pull the raw JSON and blurb for about 100 to 1,000 trips, analyze the structure, and assess what changes are needed (especially standardizing and indexing the clinical questions and storing answers in a usable, question-linked form).
5. Build the golden labeled set. Human-label trips medically necessary or not for testing.
6. Stand up the RAG from CMS plus GMR and Integra rules. Draft the assessment prompt. Run offline against the labeled set.
7. Run the EDA to size controllable versus uncontrollable denials (acknowledging the data is dirty).
8. Architecture alignment with the Azure and .NET and CI/CD setup and Ernie's Voice AI path.

Success metrics (from the whiteboard):

1. Reduction in proof of mailing.
2. Front-end BLS-assessment tracking.
3. Text-field updates (medical necessity corrections made at order time).

## 8. Open Questions and Watch-outs

- Denial data is noisy. Single captured reason; a denial does not equal a medical necessity failure. Do not anchor the EDA to it alone.
- No question-to-answer association and no derived-versus-override flag in stored data today. May require a storage change before the model is reliable.
- Market-specific scope of practice must be an explicit input to any determination.
- Friction versus accuracy is the make-or-break product tension. Design the guidance to be succinct and in-place.
- The Smartsheet rules are not linked to system rules and are not maintained. This needs fixing.
- EHR data does not equal transport needs. What the patient receives in-hospital (drips, oxygen) may be discontinued before transport, and the crew's PCR is ultimately what gets billed, so order-time medical necessity only goes so far until the crew side is connected.
- Confirm the 20 to 25 million dollar figure rather than treating it as given.
