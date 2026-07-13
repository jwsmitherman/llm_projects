# Nurse Navigation Data Analysis — Business Notes

## Overall Goal

Use AI/NLP on historical nurse navigation data — primarily unstructured free-text nurse notes in **Logis** — to explain the *why* behind navigation decisions. Today the system captures the outcome (where the patient was sent) but not the reasoning, which lives only in nurse notes.

The objective is to establish a **historical baseline** ahead of the new operating system going live (target: **before end of year**), so the business can explain the expected shift in metrics and prepare clients for change management.

## Business Context

- Patients are triaged to an **"Antara" level 0–6**, which produces a *menu* of care options (ER, urgent care, virtual care, self-care). Nurses choose from the menu; the rationale is never captured in a structured field.
- The new system separates triage into three distinct decisions — **how fast**, **where**, and **how they get there** — and forces a single recommended option with a structured **override reason code**.
- Because current outcomes blend transport and care setting, metrics will move significantly at go-live. Clients (often local government) use this data to justify funding, so an unexplained "cliff" is a real risk.
- **Primary KPIs:**
  - Ambulance diversion rate — ~40%
  - ED diversion rate — ~30%
  - % of 911 calls referred into NurseNav — mid–single digits average; 10–20% in mature markets

## Phase 1 Scope — Three Buckets to Decode

| # | Bucket | Question to Answer |
|---|--------|--------------------|
| 1 | **Self-care** (~18–20% of navigations) | Known to be mislabeled — often patients transporting *themselves* to urgent care/ED. Recast history into the new three-question framework. |
| 2 | **Antara 6** ("triage not completed") | Assumed to be largely patient refusal, but unverified. Why did triage fail? |
| 3 | **Antara 1–5 ambulance overrides** | Triage didn't call for an ambulance, but the nurse sent one anyway. Why? (patient refusal, no after-hours alternative, mobility/clinical constraints — e.g., 14% of Riverside calls are catheter issues with no viable alternative.) |

All three are required for a complete change-management story; a single bucket won't suffice.

## Data & Approach

- **Source:** ~200K Logis free-text call notes. From **Cordy**, only the protocol / chief complaint is needed. Call transcripts deemed low-value for Phase 1 — the "why" often precedes the patient conversation and exists only in the note.
- **Method:** Agentic framework with specialized agents per dimension (modeled on prior hospice-notes work).
- **Key challenge:** Notes mix *operational* content (what the nurse did) with *clinical* content (patient need). Separating the two is non-trivial.
- **Data quality:** Notes have improved over the past 1–2 years. Expect ~25% unusable; 50–75% coverage is sufficient for directional accuracy.
- **Watch-out — work set ID:** A "work set ID" ≠ one patient episode. A single patient may generate multiple work set IDs (some operational only, patient not on the phone). Need rules to avoid double-counting.
- **Watch-out — training data drift:** Historical data reflects the old workflow and homegrown triage questions; the new system moves to the **Schmitt Thompson** standard. AI trained on history may conflict with the new framework.
- **Output needed at both national and client-specific levels.**

## Next Steps / Tasks

- [ ] Connect Josh with **Rich** (consultant on Anisa's team who builds the Power BI reporting) to understand data lineage across Logis and Cordy.
- [ ] Josh to run **EDA** on nurse notes + protocol data, then propose an **MVP scope and 4–8 week timeline**.
- [ ] Josh to get a look at the **new Logis/Bingley build** and attend the Logis on-site (**July 29, Louisville**) — early findings may surface structured fields worth adding to the new system *now*.
- [ ] Schedule Josh on-site at **Dallas HQ** to whiteboard with Anisa and Nathan during EDA.
- [ ] Clinical setup of new single-option protocols owned internally (**Dr. Stites**, with nurse and Dr. Troutman input).
- [ ] Shelly to add any further data asks after her session with Noah next week.

## Later Phases (Context Only)

- **Phase 2:** Analyze upside in referral volume (including field/EPCR referrals from EMS crews that never reach NurseNav) and resolution rate; layer in external ED-avoidance benchmarks by market.
- **Phase 3 — "911 Value Stack":** Evaluate net-new capabilities (72-hour telehealth follow-up, behavioral health, social needs assessment) and new data frameworks (e.g., RAF scores, a proprietary 911 patient score). Owned by Shelly and Noah.
