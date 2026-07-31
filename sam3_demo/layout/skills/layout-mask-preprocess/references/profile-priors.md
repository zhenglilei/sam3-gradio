# Profile priors

These values are candidate centers, not model-editable parameters.

## ACT

- Baseline: `threshold=12`, `close=15`, `morph=0`.
- Search close values `11/13/15/17/19` for many small concavities.
- Prefer close over dilation so the entire device does not become thicker.
- Reject accidental bridges between separate devices.

## GE1

- Baseline: `threshold=12`, `close=0`, `morph=+1`.
- When lines are thin, compare `morph=0/+1/+2`.
- Offer `close=3/5` only when the user explicitly wants to repair gaps without thickening.

## GE2

- Baseline: `threshold=12`, `close=0`, `morph=0`.
- Preserve square holes and separation gaps.
- Never inherit ACT close values by default.

## Unknown

- Stay near current controls.
- Prefer candidates changing one parameter at a time.
- Lower confidence and request manual review when geometry is ambiguous.
