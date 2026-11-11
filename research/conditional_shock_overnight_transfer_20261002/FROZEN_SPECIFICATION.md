# Frozen specification — shock recovery and overnight/RTH transfer

Frozen on 2026-10-02 before campaign calculations. The six existing sleeves are read-only references and are never recalibrated.

## Data and common conventions

- Local one-minute continuous futures only: `MNQ_c_0_1m_20260321_094501.parquet` and `MGC_v_0_1m_20150101_20260923.parquet`.
- Common research interval: 2019-05-06 through 2026-03-19. IS: 2019-05-06–2022-12-31; validation: 2023-01-01–2023-12-31; untouched OOS: 2024-01-01–2026-03-19.
- America/New_York civil time. RTH is 09:30–16:00 for MNQ and 08:20–13:30 for MGC. Tz-aware timestamps handle DST. Actual observed RTH last bars identify shortened sessions; sessions with no usable opening/closing observation are rejected and reported.
- Overnight is the interval strictly after the preceding observed RTH close and strictly before the current RTH open. Events spanning a continuous-contract `instrument_id` change are rejected.
- Ex-ante volatility is the previous completed value of Wilder RTH daily ATR(14). Five-minute signal bars are left-closed/right-open. A condition using a completed bar can fill only at the next one-minute open.
- One contract, at most one open position per instrument and one entry per session for session strategies. Conservative stop-before-target ordering inside an ambiguous one-minute bar.
- Costs: MNQ tick 0.25, point value USD 2; MGC tick 0.10, point value USD 10; commission USD 1.25 per side; nominal adverse slippage one tick per side.

## Conditional shock recovery

- Five-minute RTH observations; shock lookbacks 30, 60, 120 minutes and thresholds 1.5, 2.0, 2.5 prior ATR units.
- Down shock: completed-bar close excursion below the prior rolling high. Up shock is symmetric from the prior rolling low. First crossing per session/side/cell is retained; overlapping event counts are explicitly reported.
- Canonical reclaim within 30 minutes: no new extreme since shock, then close above the previous five-minute high after a down shock (below previous low after an up shock). Robustness alternative uses the previous-bar midpoint. Entry is the next exact one-minute open.
- Event endpoints: 5, 15, 30, 60, 120 minutes; MFE/MAE and recovery of shock midpoint/origin. Immediate and reclaim-conditioned paths are published separately.
- Strategy (only actionable when the directional event gate passes): stop one tick beyond shock extreme, target shock midpoint, maximum holding 120 minutes, mandatory observed RTH-close exit, one entry per shock/session.

## Overnight to RTH information transfer

- Pre-open variables: overnight return and range divided by prior ATR, RTH-open location in overnight range, distance to prior RTH close, overnight close-location, overnight VWAP, and ex-ante ATR regime tercile fitted on IS only.
- Ordered overnight-move bins use fixed signed ATR edges: below -1.5, -1.5…-0.5, -0.5…0.5, 0.5…1.5, above 1.5. Forward returns are open→15m, 15→30m, 30m→noon (or close if earlier), and open→RTH close.
- `H1_CONTINUATION` and `H2_REJECTION_FADE` are separate families and sides. Absolute move thresholds: 0.5, 1.0, 1.5 ATR; confirmation windows: 15 or 30 minutes.
- Acceptance requires the confirmation close to retain the overnight sign and remain beyond the overnight midpoint. Rejection requires the RTH open in the outer quartile in the overnight direction and the confirmation close to reintegrate beyond the overnight midpoint. Entry is the next exact one-minute open.
- Continuation exits at RTH close. Rejection uses the first reached structural target (overnight midpoint, then prior RTH close if crossed first in the same direction), stop one tick beyond the overnight extreme, and otherwise exits at RTH close.
- Placebos: opposite signal sign and one-session-lagged overnight sign.

## Frozen inference and decision gates

- Session-cluster bootstrap: 1,000 deterministic resamples (seed 20261002); 95% intervals. Family-wise p-values use Bonferroni correction across cells within mechanism/instrument/side.
- Minimum sample: 80 full-period trades and 25 OOS trades. A low-sample otherwise-consistent result is `WATCHLIST_LOW_SAMPLE`.
- Candidate requires positive net expectancy in IS, validation and OOS; validation and OOS PF above 1; positive event-study direction with corrected p≤0.05; at least half of contiguous neighboring cells positive; positive expectancy with +1 tick/side, doubled commission, and one-bar delayed entry; no single year contributing over 50% of positive PnL; and superiority to both placebos.
- Failure labels are applied in order: no directional/event edge, instability across splits/neighbors/years, cost sensitivity, redundancy with frozen sleeves. Passing standalone and portfolio gates yields only `LOCKED_SLEEVE_CANDIDATE`, never an automatic lock.
- Marginal portfolio gate: absolute daily correlation below 0.50, non-negative PnL on baseline negative days, and improvement in at least two of Sharpe, MaxDD, or 95% expected shortfall under equal-sleeve and inverse-volatility diagnostic allocations. Frozen sleeve trades and weights are not altered.
- Full Topstep PASS/MLL/EXPIRY simulation is run only for a standalone survivor that also passes the marginal portfolio gate.

