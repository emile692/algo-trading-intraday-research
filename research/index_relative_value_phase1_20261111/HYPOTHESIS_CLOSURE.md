# Intraday relative value between equity index futures — phase 1 closure

**Status:** `REJECTED` (IS gate failed: all 30 cells negative after costs)
**Date:** 2026-11-11
**Spec:** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `25baa963…eb6102` ([SPEC_LOCK.json](SPEC_LOCK.json)), frozen before any return of this strategy was computed.
**OOS (2024-01 → 2026-03):** not computed, still unseen for this hypothesis.

## Hypothesis

The relative move of two US equity indices from the RTH open to open+h predicts their relative return from open+h to the close. A market-neutral position (long one index, short the other, equal notional) is taken when the relative move is at least k times its typical size, and closed at the session close.

- Pairs: MNQ−MES, MNQ−M2K, MES−M2K, pooled as one bet.
- Grid: h ∈ {30, 60, 90, 120, 180} minutes, k ∈ {0,5, 1,0, 1,5}, and both signs (reversion and continuation); 30 cells, identical on the three pairs.
- Costs: each leg pays 1 tick slippage per side plus commission. Unit: R = net spread return divided by the trailing 20-session open-to-close volatility of the spread.
- Selection: highest neighbourhood-smoothed Sharpe on IS (2019-05 → 2023-12), one cell for all pairs.
- No pre-sample holdout exists (no ES or RTY data before 2019); two extra gates replace it.

Cost measured before the freeze, without any direction-dependent quantity: a two-leg round trip is 3,5 to 5,8 bp against a daily spread volatility of 41 to 82 bp, that is 0,056 to 0,088 R per trade.

## Gate results

Selected cell: **continuation, h = 30 minutes, k = 1,5** (about 15 % of sessions trade). It is the least negative cell, not a positive one.

| Gate | Requirement | Result | |
|---|---|---|---|
| G1 | pooled IS mean net R > 0, HAC t ≥ 2,0 | −0,001 R per day, t −0,16 | **fail** |
| G2 | max-stat permutation p ≤ 0,05 | p 0,14 | **fail** |
| G3 | CSCV PBO ≤ 0,35 | PBO 0,07 | pass |
| G4 | positive on ≥ 2 of 3 pairs | 1 (MNQ−M2K) | **fail** |
| G5 | more than half of IS years positive | 2 of 5 (2020, 2021) | **fail** |
| G6 | positive in both halves of IS | +0,005 R then −0,007 R | **fail** |
| G7 | still positive with costs × 1,5 | negative | **fail** |

G3 passes because the ranking of the cells is stable (continuation at short h is always the least bad); the chosen cell has a positive test-half Sharpe in 26 % of the CSCV splits.

## Selected cell by pair (IS)

| Pair | Trades | Gross bp per trade | Cost bp per trade | Net R per trade | Win rate |
|---|---:|---:|---:|---:|---:|
| MNQ−MES | 174 | 5,7 | 4,2 | −0,018 | 53 % |
| MNQ−M2K | 159 | 11,5 | 5,3 | +0,065 | 53 % |
| MES−M2K | 166 | 4,7 | 6,6 | −0,043 | 49 % |

Pooled gross mean of the selected cell: +0,013 R per day (t 1,68). With entry delayed by one bar on both legs the net mean stays about the same (−0,002 R).

## Reading

- **There is no intraday reversion between indices.** The reversion mode is negative before costs in 14 of 15 cells. A large relative move in the first part of the session does not come back by the close.
- **The relative move continues, but only from the first 30 to 60 minutes, and by less than the cost.** Continuation is positive gross at h = 30 (+0,013 to +0,028 R per day) and h = 60, and about zero from h = 90 onwards. The best per-trade gross (5 to 11 bp) is of the same size as the two-leg cost (4 to 7 bp).
- **Same pattern as the single-instrument momentum campaigns of 2026-11-11:** a small positive gross continuation effect that the cost removes. Hedging the market factor out did not make it larger in risk units, and a second leg doubled the cost.
- MNQ−M2K is the only positive pair (t 0,93). This is a post-hoc observation inside a failed pooled test.

## What may and may not be reopened

- No rescue variant on this data: no other window, threshold, exit, stop, hedge ratio or single-pair pick.
- The OOS window stays unread for this hypothesis.
- A version on mini contracts (NQ, ES, RTY), where the tick is a smaller share of notional, would be a new pre-registered campaign and needs ES and RTY data that are not on disk. It would also not fit a 50K Topstep account.

## Files

- [run_phase1.py](run_phase1.py): `selftest` (synthetic data), `costcheck` (volatility and cost only), `is`, `oos` (refuses to run unless the IS gate passed).
- `results/is_results.json`: selected cell, gates, per-pair detail.
- `results/is_cell_table.csv`: the 30 cells, pooled and per pair, with gross and cost means.
