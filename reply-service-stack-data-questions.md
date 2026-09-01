**Subject:** RE: Data questions for Service Stack options

Shelli,

Thanks for writing these out.

## Volume of 911 calls flagged as BH or mental health, by state and county

There's no single field that identifies behavioral health, so the definition has to be chosen: dispatch code, clinician impression, or diagnosis code. Each will give a different number.

Rather than write our own, NEMSIS — the national EMS database — publishes a Behavioral Health Case Definition Public Use Dataset built on standardized definitions. It's free and the 2025 release is available by request. Two uses: adopt their definition so our internal numbers are defensible and comparable, and use their national figures as a benchmark for what share of 911 volume is BH-related. State-level is available; county generally isn't. NEMSIS is not population-based and has known missing-data issues, so it's a benchmark rather than a census.

## 1. BH-related 911 calls transferred to Nurse Navigation

I can move on the Nurse Nav side of this now. The extract includes the nurse's notes and 99% are long enough to read. Two layers:

First, the main reason for the call as it's already captured in the data. If that field carries psychiatric, behavioral, overdose or similar categories, that's a clean count with no interpretation required. I'll check what values it actually holds before assuming either way.

Second, the notes — which is your suggestion, and it's the right one. Behavioral health can appear as a secondary factor on a call logged under something else: chest pain with an anxiety component, a fall where intoxication is documented, a welfare check. I'd run a full-population keyword pass plus an LLM read on a sample to measure its accuracy, then compare both against the coded count so we can see how much the notes add. That produces a number with a known error range rather than an estimate. I've built and run both methods on this data already.

One question. Since these calls originate as county 911 transfers, there should be an identifier or originating-agency field tying a Nurse Nav record back to the 911 call. It isn't in the extract I have. Do you know who owns that link, or whether it's available in the source system? With it we can report transfers directly rather than estimating, and depending on what the 911 record carries it may also give us the originating county — my current extract has market rather than state and county.

## 2. National ED discharge volume with a primary BH diagnosis

HCUP NEDS from AHRQ — the largest publicly available all-payer ED database, about a 20% stratified sample of US EDs. Two constraints. It's purchased through the HCUP Central Distributor under a data use agreement, so there's cost and lead time. And several states are legally barred from reporting behavioral health conditions to HCUP, which means national BH counts run low and need a footnote. State-level detail exists but that restriction hits it hardest. HCUPnet is free if a directional number is enough. SAMHSA covers prevalence and treatment capacity.

## 3. Uninsured percentage of 911 calls where we have payer data

From claims data, with two definitions to pick between: self-pay at time of service, or self-pay after insurance discovery. Worth noting the denominator would be transports rather than all calls, since non-transports don't generate a claim. NEMSIS carries payment method and can serve as an external comparison.

## 4. Repeat encounters within 30, 60, 90 days

Requires a consistent patient identifier across encounters. I don't see one in the Nurse Nav extract, so I need to check what exists at the source. If there's a usable ID this is a direct analysis. If not, the alternative is probabilistic matching on name, date of birth and address, which carries real error rates and would need to be scoped separately.

## 5. High-utilizer cohorts at 2+, 5+, 10+ calls per year

Same dependency as #4 — it stands or falls on the patient identifier. If we can't get there internally, published EMS high-utilizer research can give us a benchmark range in the meantime.

## What I need from you

1. Time period — CY2025 or trailing 24 months
2. Definition of 911 calls — dispatches, responses, or transports
3. Who to go to for the 911-to-Nurse Nav link and the originating county
4. Whether we adopt the NEMSIS behavioral health definition as our standard

I'll start the NEMSIS request in parallel since it has lead time, and begin the Nurse Nav BH pass while the rest gets sorted.

Thanks,
Josh
