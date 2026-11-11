# Cross-asset intraday time-series momentum — phase 1 closure

**Status:** `REJECTED` (IS gate failed: effect too small after costs, not replicated on the pre-sample holdout)
**Date:** 2026-11-11
**Spec:** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `6b4e7eee…df6e61` ([SPEC_LOCK.json](SPEC_LOCK.json)), frozen before any return of this strategy was computed.
**OOS (2024-01 → 2026-03):** not computed, still unseen for this hypothesis.

## Hypothesis

The sign of the move from the session open to open+h predicts the return from open+h to the session close. One common rule on every locally available market: follow the move at open+h if it is at least k times its typical size, exit at the close, no stop.

- Universe: MNQ, MES, M2K (one equity-index cluster), MGC (gold), MBT (bitcoin). 6E, ZN and MCL were not tested because their one-minute files are no longer on disk.
- Grid: h ∈ {30, 60, 90, 120, 180} minutes, k ∈ {0, 0.5, 1.0}; 15 cells, identical on every instrument.
- Costs: 1 tick slippage per side plus commission. Unit: R = net return divided by the trailing 20-session open-to-close volatility.
- Pooling: cluster-balanced daily series, so the three equity indices count once.
- Selection: highest neighbourhood-smoothed Sharpe on IS (2019-05 → 2023-12), one cell for all instruments.

## Gate results

Selected cell: **h = 90 minutes, k = 1.0** (about 30 % of sessions trade).

| Gate | Requirement | Result | |
|---|---|---|---|
| G1 | pooled IS mean net R > 0, HAC t ≥ 2.0 | +0,008 R per day, t 0,96, Sharpe 0,43 | **fail** |
| G2 | max-stat permutation p ≤ 0,05 | p 0,046 | pass |
| G3 | CSCV PBO ≤ 0,35 | PBO 0,17 | pass |
| G4 | positive in ≥ 2 of 3 clusters and ≥ 3 of 5 instruments | 2 clusters, 4 instruments | pass |
| G5 | more than half of IS years positive | 3 of 5 (2019 and 2021 negative) | pass |
| G6 | PRE holdout mean net R > 0, HAC t ≥ 1.0 | +0,003 R, t 0,29 | **fail** |

PRE holdout is NQ 2010-06 → 2019-05 (NQ costs) and MGC 2016-02 → 2019-05, with the cell fixed from IS.

## Selected cell by instrument (IS)

| Instrument | Trades | Mean net R per day | HAC t | Gross USD per trade | Net USD per trade |
|---|---:|---:|---:|---:|---:|
| MNQ | 337 | +0,002 | 0,12 | 9,08 | 5,58 |
| MES | 333 | −0,007 | −0,44 | 1,38 | −3,62 |
| M2K | 377 | +0,001 | 0,06 | 1,57 | −1,93 |
| MGC | 305 | +0,007 | 0,50 | 8,96 | 4,46 |
| MBT | 96 | +0,052 | 2,29 | 21,30 | 17,48 |

Robustness of the selected cell on IS:
- With costs doubled, the pooled mean turns negative (−0,003 R, t −0,31).
- With entry delayed by one more bar, it barely changes (+0,007 R, t 0,86).
- On PRE, the entry delay alone makes it negative.

## Reading

- **The gross effect exists but is tiny.** All 15 cells have a positive pooled gross mean (+0,005 to +0,032 R per day). After costs, 11 of 15 are negative. Trading every day (k = 0) is negative at every horizon; only the k = 1.0 filter gets near zero or slightly positive.
- **It is not a cross-asset effect.** The equity-index cluster is flat (−0,002 R, t −0,16) and gold is flat (t 0,50). The pooled result is carried by MBT.
- **The permutation and PBO gates pass, but they only say the grid ranking is stable**, not that the level is tradable. The level tests (G1 on IS, G6 on nine independent years of NQ) both fail.
- **MBT is positive in all 15 cells** (t 2,29 on the selected one). This is a post-hoc observation on 96 trades over 359 thin sessions, and the book already holds three MBT trend/breakout sleeves, so most of it is probably the same bet.

## What may and may not be reopened

- No rescue variant on this data: no other window, threshold, reference price, exit, stop, or single-instrument pick.
- The OOS window stays unread for this hypothesis and must not be consulted to "check" MBT.
- An MBT-only version would be a new pre-registered campaign. It would have to show incremental value over the three frozen MBT sleeves (correlation and marginal contribution), not standalone performance.
- 6E, ZN and MCL can be tested with this same frozen spec if their one-minute data is restored; that would be a replication, not a reopening.

## Files

- [run_phase1.py](run_phase1.py): `selftest` (synthetic data), `is`, `oos` (refuses to run unless the IS gate passed).
- `results/is_results.json`: selected cell, gates, PRE holdout, per-cluster and per-instrument detail.
- `results/is_cell_table.csv`: the 15 cells, pooled and per instrument.
