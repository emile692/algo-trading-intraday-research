# MNQ Prod Config Detected

confidence: medium

## Most Probable Candidate

The most probable live/prod ORB reference found in this repo is the audited MNQ ORB export used by the client notebook:

- Source export: `data/exports/mnq_orb_vix_vvix_validation_20260327_run`
- Variant referenced by client notebook: `baseline_fixed_nominal_atr`
- Client notebook reference: `notebooks/mnq_orb_pullback_equal_weight_client.ipynb`
- Validation campaign source: `src/analytics/mnq_orb_vix_vvix_validation_campaign.py`

### Extracted parameters

- symbol: `MNQ`
- session timezone: `America/New_York`
- session window: `09:30` -> `16:00`
- opening range window: `30 minutes`
- trigger buffer ticks: `2`
- stop buffer ticks: `2`
- target multiple: `2.0`
- vwap confirmation: `true`
- vwap column: `continuous_session_vwap`
- entry timing: `entry_on_next_open = true`
- one trade per day: `true`
- account size: `50,000 USD`
- risk sizing: `1.5%` per trade in the notebook/campaign baseline
- slippage: `1 tick`
- commission: `0.62 USD per side`
- structural filter active: ATR ensemble day-selection `majority_50`
- selected sessions in audited export: `1134` total, `339` OOS according to `run_metadata.json`

### Ambiguity to note

There is a real ambiguity on direction:

1. `data/exports/mnq_orb_vix_vvix_validation_20260327_run/run_metadata.json`
   - baseline direction: `both`
2. `notebooks/orb_MNQ_final_ensemble_validation.ipynb`
   - current active parameter block `OR30LONG`
   - baseline direction: `long`

Because your question is specifically about invalidating the **long setup** after a downside first breakout, the practical interpretation for this study is:

- use the audited MNQ OR30 baseline assumptions
- keep the selected-session logic from the audited ATR ensemble export
- analyze the **long leg only** for the invalidation decision

## Other plausible candidates found

### Candidate A: audited client-linked baseline

- file: `data/exports/mnq_orb_vix_vvix_validation_20260327_run/run_metadata.json`
- direction: `both`
- OR window: `30m`
- VWAP confirmation: `true`
- risk sizing: `1.5%`
- costs: `0.62 / side`, `1 tick`
- ATR ensemble filter: `majority_50`
- confidence: `medium-high`

### Candidate B: current notebook parameter block

- file: `notebooks/orb_MNQ_final_ensemble_validation.ipynb`
- active ensemble header: `OR30LONG`
- direction: `long`
- OR window: `30m`
- entry buffer: `2`
- stop buffer: `2`
- VWAP confirmation: `true`
- risk sizing: `1.5%`
- costs: `0.62 / side`, `1 tick`
- confidence: `medium`

### Candidate C: later business overlays

Several later exports reference:

- `baseline_3state`
- `baseline_vvix_modulator`
- `baseline_vvix_3state`

These appear to be downstream overlays on top of the validated ORB baseline rather than a different ORB entry engine. They matter for funded/business sizing, but the underlying ORB signal family still traces back to the OR30 baseline above.

## Decision for this analysis

For the invalidation study, the chosen working interpretation is:

- prod_mnq_signal_config: `OR30 long, entry_buffer=2, stop_buffer=2, VWAP confirmation on`
- prod_mnq_costs: `commission 0.62 / side, slippage 1 tick`
- prod_mnq_session_filter: use the audited ATR ensemble selected sessions from `baseline_fixed_nominal_atr/controls.csv`
- confidence for this exact reconstruction: `medium`

## Why the existing opposite-breakout export is not exact

The existing export `export/orb_opposite_breakout_invalidation_20260528_231811` does **not** appear to match the prod baseline exactly because it uses a different baseline family:

- OR window there: `15m`
- direction there: `long`
- account risk there: `0.5%`
- repo default costs there: `1.25 / side`
- no explicit audited ATR ensemble selected-session filter

So it is useful as a directional clue, but not enough for a final prod-only deployment verdict.
