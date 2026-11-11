# MNQ close hedging flow — Phase 1/2

The one-minute MNQ continuous series covers 5 May 2019 through 20 March 2026. The implementation is explicitly look-ahead safe: ROD ends at the close of the 15:29 bar, the signal is known at 15:30 ET, and the event return begins at the 15:30 bar open and ends at the 15:59 bar close.

## Canonical event study

- ROD: previous relevant 16:00 cash close proxy (15:59 bar close) to today's 15:29 bar close.
- Outcome: 15:30 bar open to 15:59 bar close.
- 1,667 eligible sessions; 1,126 IS and 541 OOS from 1 January 2024.
- No threshold, gamma conditioner, or alternate timing was tested.

## Evidence

- Full-sample regression slope: −0.0125, ordinary t = −2.07, HAC/Newey-West t = −0.77.
- OOS slope: −0.0131 with HAC t = −0.77; IS-fitted OOS predictive R² = 0.38%.
- Decile monotonicity (Spearman): 0.018, effectively zero.
- Positive-ROD days: final-30m mean +0.03 bps (p = 0.973).
- Negative-ROD days: final-30m mean +0.30 bps (p = 0.847).
- From 2021 onward, ROD/outcome correlation is −0.001; the conditional sign pattern is opposite to the hypothesized momentum direction.

The published positive conditional relationship is not visible in this MNQ sample. Ordinary significance is erased by HAC correction and the decile plot is non-monotone.

## Phase gate

`DO_NOT_ADVANCE`. The canonical mechanism fails before costs and before strategy construction. Running timing variants would be post-hoc search, so Phase 4 is not run.

Files: `event_level.csv`, `conditional_returns_by_decile.csv`, `conditional_returns_by_sign.csv`, `yearly_metrics.csv`, `aggregate_metrics.json`, and `conditional_last30_by_rod_decile.png`.
