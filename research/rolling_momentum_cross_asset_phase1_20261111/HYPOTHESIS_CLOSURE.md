# Cross-asset rolling intraday momentum — phase 1 closure

**Status:** `REJECTED` (IS gate failed; every one of the 32 cells is negative after costs)
**Date:** 2026-11-11
**Spec:** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `c59cf11f…b7468f` ([SPEC_LOCK.json](SPEC_LOCK.json)), frozen before any return of these rules was computed.
**OOS (2024-01 → 2026-03):** not computed, still unseen for this hypothesis.

## Hypothesis

Two of the simplest momentum rules, with no filter, held and flipped bar by bar during the RTH session and flat overnight:
- **diff:** position = sign of `close.diff(n)` on tf-minute bars, tf ∈ {5, 15, 30, 60}, n ∈ {1, 2, 4, 8, 16};
- **ema:** position = sign of EMA_fast − EMA_slow on tf-minute bars, pairs (5, 20), (10, 40), (20, 80).

Other settings:
- Decision at each bar close, fill at the next bar open, 1 tick of slippage plus commission per side.
- The signal may look back across sessions; it is zeroed across a roll.
- Same 32 cells on MNQ, MES, M2K, MGC and MBT, pooled by cluster.
- Selection on IS only (2019-05 → 2023-12) by neighbourhood-smoothed Sharpe. The pre-sample holdout is NQ 2010 → 2019 plus MGC 2016 → 2019.

## Gate results

Selected cell: **ema, tf = 30 min, EMA 5/20**, which is the least negative cell.

| Gate | Result | |
|---|---|---|
| G1 IS level (HAC t ≥ 2,0) | −0,030 R per day, t −1,79 | **fail** |
| G2 max-stat permutation (p ≤ 0,05) | p 0,54 | **fail** |
| G3 CSCV PBO (≤ 0,35) | 0,16, but the chosen cell's test Sharpe is positive in 0,2 % of splits | pass, not meaningful |
| G4 breadth | 0 of 3 clusters, 1 of 5 instruments | **fail** |
| G5 years | 0 of 5 positive | **fail** |
| G6 PRE holdout (t ≥ 1,0) | −0,039 R, t −1,84 | **fail** |

## What the 32 cells show

- **Net:** all 32 cells are negative pooled.
  - Fast cells are crushed by costs: diff 5 min n=1 trades about 79 sides a day, for −1,41 R/day.
  - The slowest cells (60 min, or EMA 20/80) still lose 0,03 to 0,05 R/day.
- **Gross:** before costs the pooled mean is small and mostly positive (about +0,02 to +0,06 R/day, comparable to the open-anchored campaign). So the momentum is there, but it is much smaller than the cost of following it bar by bar.
- **Per instrument:**
  - **MNQ is the only instrument positive on about half the cells**, for example +0,044 R on diff 60 min n=4 and +13,7 USD/day on the selected cell (t 1,45). It also has the best cost-to-notional ratio of the five.
  - On NQ 2010–2019 the selected cell is negative net (t −0,89).
  - MES, M2K and MGC are negative almost everywhere.
  - MBT is gross positive on average (+0,08 R) but net flat to negative.
- **Robustness of the selected cell:** delaying the signal by one bar makes it worse (t −2,64), and doubling costs gives t −4,83.

## Reading

Together with [the open-anchored campaign](../intraday_momentum_cross_asset_phase1_20261111/HYPOTHESIS_CLOSURE.md), the picture is consistent:
- Intraday price momentum on these markets has a small positive gross drift.
- No simple sign rule captures it above costs on a common specification.
- The more often the rule trades, the worse it gets.
- The MNQ positives are a post-hoc single-instrument observation. They are close to the existing MNQ trend sleeve (`atr_trend_flip`, now shadow only) and do not replicate on NQ before 2019.

## What may and may not be reopened

- No rescue variant on this data: no dead band, other lookback, filter, stop, holding rule or single-instrument pick.
- The OOS window stays unread for this hypothesis.
- An MNQ-only trend rule would be a new pre-registered campaign judged on marginal value against the MNQ book, not standalone.

## Files

- [run_phase1.py](run_phase1.py): `selftest` (synthetic data: manual one-day reconstruction, causality, roll handling, power), `is`, `oos` (refuses unless the IS gate passed).
- `results/is_results.json`, `results/is_cell_table.csv` (32 cells, gross and net, per instrument, sides per day), `results/results_is_stdout.txt`.
