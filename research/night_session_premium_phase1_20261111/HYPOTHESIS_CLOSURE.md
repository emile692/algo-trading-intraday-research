# Night-session long premium, cross-asset — phase 1 closure

**Status:** `REJECTED` (IS gate failed: all 12 cells negative after costs)
**Date:** 2026-11-11
**Spec:** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `3c81cde9…412fca` ([SPEC_LOCK.json](SPEC_LOCK.json)), frozen before any return of this strategy was computed.
**OOS (2024-01 → 2026-03):** not computed, still unseen for this hypothesis.

## Hypothesis

Holding a long futures position during the Globex night session earns a positive premium after costs. Priors: the overnight drift of US equity index futures around the European open, stronger after a down US session, and the positive Asian-hours return of gold. Crude oil and bitcoin had no documented prior and were included as a breadth test.

- Universe: MNQ, MES, M2K (one equity-index cluster), MGC (gold), MCL (crude oil, from 2021-07-12), MBT (bitcoin).
- Grid: 6 windows between the nodes 18:00, 02:00, 03:00 and 08:00 New York, each traded every night or only after a down RTH session; 12 cells, identical on every instrument. Flat at 08:00, before every RTH sleeve of the book.
- Costs: 1 tick slippage per side plus commission. Unit: R = net return divided by the trailing 20-night volatility of the 18:00 → 08:00 return.
- Pooling: cluster-balanced daily series. Selection: highest pooled Sharpe on IS (2019-05 → 2023-12), one cell for all instruments.
- Node prices use the first bar within 15 minutes of the node. This tolerance was set before the freeze from session counts only (at 5 minutes MBT lost about a third of its 2022–2023 nights).

## Gate results

Selected cell: **18:00 → 08:00, every night**. It is the least negative cell, not a positive one.

| Gate | Requirement | Result | |
|---|---|---|---|
| G1 | pooled IS mean net R > 0, HAC t ≥ 2.0 | −0,017 R per night, t −0,85, Sharpe −0,36 | **fail** |
| G2 | max-stat permutation p ≤ 0,05 | p 0,19 | **fail** |
| G3 | CSCV PBO ≤ 0,35 | PBO 0,20 | pass |
| G4 | positive in ≥ 3 of 4 clusters and ≥ 4 of 6 instruments | 0 clusters, 1 instrument (MNQ) | **fail** |
| G5 | more than half of IS years positive | 2 of 5 (2019, 2020) | **fail** |
| G6 | PRE holdout mean net R > 0, HAC t ≥ 1.0 | +0,002 R, t 0,11 | **fail** |
| G7 | above the costless RTH long benchmark on IS and PRE | below on both | **fail** |

G3 passes only because the ranking of the cells is stable; the chosen cell has a positive test-half Sharpe in 6 % of the CSCV splits.

PRE holdout is NQ 2010-06 → 2019-05, MGC 2015 → 2019-05 and CL 2015-01 → 2021-07 with MCL contract economics, with the cell fixed from IS.

## Selected cell by instrument

| Period | Instrument | Trades | Mean net R per night | HAC t | Gross USD per trade | Net USD per trade |
|---|---|---:|---:|---:|---:|---:|
| IS | MNQ | 1 185 | +0,020 | 0,73 | 7,59 | 4,09 |
| IS | MES | 1 187 | −0,032 | −1,08 | 5,29 | 0,29 |
| IS | M2K | 1 175 | −0,024 | −0,78 | 4,26 | 0,76 |
| IS | MGC | 1 168 | −0,040 | −1,18 | −0,52 | −5,02 |
| IS | MCL | 602 | −0,027 | −0,61 | 1,86 | −2,64 |
| IS | MBT | 612 | −0,030 | −0,68 | 0,17 | −3,65 |
| PRE | NQ | 2 249 | +0,047 | 2,12 | 35,45 | 22,95 |
| PRE | MGC | 998 | −0,055 | −1,68 | 1,61 | −2,89 |
| PRE | CL as MCL | 1 603 | −0,076 | −2,75 | −0,08 | −4,58 |

The mean net USD can be slightly positive while the mean R is negative (MES, M2K): the gains sit in high-volatility nights, which weigh less in risk units.

## Reading

- **Costs are not small against a night.** One tick per side plus commission is about 0,054 R per trade on the pooled series, more than the pooled gross premium of the best window (+0,037 R per night). The assumption that a single long hold dilutes the cost was wrong: the night's volatility is small too.
- **The European-open hour holds nothing tradable.** 02:00 → 03:00 has a pooled gross mean of +0,004 R and a net mean of −0,050 R (t −7,6). Trading only after a down RTH session does not help: it lowers the gross mean in 5 of 6 windows.
- **Gold has no night premium here.** MGC is negative gross on IS and negative net on both IS and PRE. Crude oil is negative on IS and significantly negative on PRE (t −2,75).
- **The equity index is the only place with a gross premium**, and it is market beta: on IS and on PRE the costless RTH long benchmark earns more per day than the night position (NQ on PRE: +0,074 R for RTH against +0,047 R for the night).
- NQ 2010–2019 is positive on its own (t 2,12, +22,95 USD per trade with NQ costs). This is a post-hoc single-instrument observation, it did not continue on MNQ 2019–2023 (t 0,73), and the worst night costs −3 842 USD on NQ (−808 USD on MNQ), above the Topstep 50K daily room.

## What may and may not be reopened

- No rescue variant on this data: no other node, window, condition, stop, short leg or single-instrument pick.
- The OOS window stays unread for this hypothesis.
- A short night leg on gold or crude oil would be a new pre-registered campaign, and the −2,75 t on CL is now known: it cannot serve as its own evidence.

## Files

- [run_phase1.py](run_phase1.py): `selftest` (synthetic data), `coverage` (session counts only), `is`, `oos` (refuses to run unless the IS gate passed).
- `results/is_results.json`: selected cell, gates, PRE holdout, per-cluster and per-instrument detail.
- `results/is_cell_table.csv`: the 12 cells, pooled and per instrument.
