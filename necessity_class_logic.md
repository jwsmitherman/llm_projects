# `necessity_class` — Determination

Named Group A concepts (explicit in CMS text): `bed_confined`, `mobility_deficit`, `cannot_sit`, `ventilator`, `suctioning`, `iv_medication`, `cardiac`.

Evaluated in order; first match wins:

**`necessary`** — if all three bed-confined prongs matched (`bed_confined` + `mobility_deficit` + `cannot_sit`), OR ≥1 named concept matched AND (`mobility_score ≥ 3` OR `monitoring_score ≥ 3`).

**`not_necessary`** — if no Group A concept matched AND the field was empty or held only filler (`weakness_only`, `fall_risk_only`, `nonclinical`).

**`indeterminate`** — everything else. Split by `indeterminate_reason`:
- `weak_or_inferred_only` — a Group A concept matched but below threshold.
- `text_unmatched` — text present, no concept matched.

| Condition | Result |
|---|---|
| All three bed-confined prongs | `necessary` |
| ≥1 named concept and (mobility ≥ 3 or monitoring ≥ 3) | `necessary` |
| No Group A concept; field empty or filler-only | `not_necessary` |
| Group A signal below threshold | `indeterminate` (weak_or_inferred_only) |
| Text present, no concept matched | `indeterminate` (text_unmatched) |

Thresholds: `MOBILITY_CLEAR = 3`, `MONITORING_CLEAR = 3`. Set in config; not CMS-mandated.
