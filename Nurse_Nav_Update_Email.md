**Subject:** Nurse Nav analysis update — self-care deep dive and other changes

Team,

The Nurse Navigation analysis has been updated based on the protocol review discussion. Summary of what changed and the key self-care findings below.

## What changed

- Added a self-care deep dive that breaks the self-care bucket into what patients actually did next, using the nurse notes.
- Added a breakdown of which conditions are resolved as self-care.
- Updated the override driver analysis to handle negation (for example, "denies chest pain" is no longer counted as a clinical driver) and added an "unclear from notes" category for calls with no clear driver.
- Added a call-drop rate per market alongside the raw count, so markets are compared on their own call volume rather than absolute numbers.

## Self-care highlights

Self-care is 25,174 calls, about 17% of all volume. Today the label only means no ambulance, transport, or referral was arranged — it does not describe the outcome. Reading the notes separates it into:

| Sub-type | Calls | Share |
|---|---|---|
| Stayed home on nurse advice | 14,939 | 59% |
| Refused or declined further care | 7,541 | 30% |
| Self-transported to urgent care | 2,001 | 8% |
| Self-transported to the ED | 1,341 | 5% |
| Waiting for an existing appointment | 455 | 2% |
| Could not be determined from the notes | 6,332 | 25% |

**Main takeaway:** most self-care is a patient staying home, but roughly one in eight self-transported to the ED or urgent care — meaning the current "self-care" label includes patients who did seek care on their own. By condition, the largest self-care group is calls where triage did not complete (3,343), followed by abdominal pain, anxiety, hypertension, and vomiting.

These figures are directional. They come from scanning the notes for keywords, so the categories can overlap and a note-sample audit is the recommended next step before the numbers are treated as final. This breakdown matters most for the operating-system transition, since it shows how the self-care bucket would need to be re-categorized under the new protocols.

The updated notebook and slides are in the shared folder. Happy to walk through any of it.

Thanks,
Josh
