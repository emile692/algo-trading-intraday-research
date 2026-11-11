# MCL — parameter landscape of the intraday EMA rule and the intraday Donchian rule — closure

**Status:** `REJECTED` (no parameter region is positive on both MCL IS and CL 2015–2021; OOS negative almost everywhere)
**Date:** 2026-11-11
**Nature:** exploratory landscape requested by the user after the [Unger campaign](../mcl_unger_trend_following_20261111/HYPOTHESIS_CLOSURE.md) was closed. It is not a pre-registered test and it has no gate.
**OOS (2024-01 → 2026-03): read here, on the whole grid.** It is no longer an unseen verdict window for these two rule families.
**Extension (2026-03-20 → 2026-11-10): not computed.** It is the only untouched MCL sample left for these families.

## What was computed

Annualised Sharpe of the daily net PnL (one contract, 4,50 USD per round trip), on three windows: CL 2015-01 → 2021-07 with MCL size and costs, MCL IS 2021-07 → 2023-12, MCL OOS 2024-01 → 2026-03. Same data, roll adjustment and excluded days as the earlier campaigns.

- **S7 intraday**, the multi-day EMA cross adapted to a daily flat requirement. Position = sign(fast EMA − slow EMA) on closed 15-minute bars (EMAs run on the full Globex session), held only while the holding window is open, reversed on a cross.
  - `s7_core`: held 09:00–14:30 New York.
  - `s7_day`: held 18:00 → 16:00 New York, flat before the Topstep 16:10 cut-off and over weekends.
  - Grid: fast ∈ {5, 10, 20, 30, 50, 80} × slow ∈ {50, 100, 150, 200, 300, 400}, fast < slow, 34 cells.
- **S8**, intraday Donchian breakout as in the Unger campaign. Grid: channel length ∈ {4, 8, 12, 16, 20, 30, 40, 60, 90} bars × first entry time ∈ {09:00, 09:15, 09:30, 10:00, 10:30}, 45 cells. The cell (20, 09:00) reproduces the Unger campaign exactly (IS +3 603,50 USD, CL −6 515,50 USD).

Heatmaps: [heatmap_s7_core.png](heatmap_s7_core.png), [heatmap_s7_day.png](heatmap_s7_day.png), [heatmap_s8.png](heatmap_s8.png). Numbers: [landscape.csv](landscape.csv).

## Results

| Rule | Window | Cells > 0 | Median Sharpe | Best Sharpe |
|---|---|---:|---:|---:|
| S7 core 09:00–14:30 | CL 2015–2021 | 0 / 34 | −0,65 | −0,34 |
| S7 core 09:00–14:30 | MCL IS | 7 / 34 | −0,28 | +0,28 |
| S7 core 09:00–14:30 | MCL OOS | 0 / 34 | −0,92 | −0,57 |
| S7 day 18:00 → 16:00 | CL 2015–2021 | 0 / 34 | −0,72 | −0,32 |
| S7 day 18:00 → 16:00 | MCL IS | 15 / 34 | −0,06 | +0,71 |
| S7 day 18:00 → 16:00 | MCL OOS | 0 / 34 | −1,17 | −0,81 |
| S8 Donchian | CL 2015–2021 | 0 / 45 | −0,45 | −0,02 |
| S8 Donchian | MCL IS | 30 / 45 | +0,21 | +0,77 |
| S8 Donchian | MCL OOS | 2 / 45 | −0,66 | +0,01 |

Cells of the initial campaign (EMA 50/200; Donchian 20 from 09:00):

| Rule | Window | Days | Trades | Net USD | Gross USD | Sharpe | t |
|---|---|---:|---:|---:|---:|---:|---:|
| S7 core | CL 2015–2021 | 1 681 | 1 968 | −6 687 | +2 170 | −0,73 | −1,88 |
| S7 core | MCL IS | 642 | 744 | +228 | +3 576 | 0,03 | 0,05 |
| S7 core | MCL OOS | 572 | 674 | −3 271 | −238 | −0,85 | −1,28 |
| S7 day | CL 2015–2021 | 1 672 | 2 449 | −9 351 | +1 670 | −0,76 | −1,97 |
| S7 day | MCL IS | 639 | 890 | +861 | +4 866 | 0,11 | 0,17 |
| S7 day | MCL OOS | 567 | 833 | −6 613 | −2 865 | −1,24 | −1,86 |
| S8 | CL 2015–2021 | 1 620 | 1 591 | −6 516 | +644 | −0,95 | −2,40 |
| S8 | MCL IS | 609 | 593 | +3 604 | +6 272 | 0,72 | 1,11 |
| S8 | MCL OOS | 544 | 537 | +16 | +2 432 | 0,01 | 0,01 |

## Reading

- **The problem is the sign, not the t-stat threshold.** Out of 113 cells, none is positive on both MCL IS and CL 2015–2021, and 2 are positive on OOS (both at +0,01). Lowering the gate would not have changed any decision.
- **The positive zone is confined to MCL 2021–2023**, which contains 2022. It appears on neither the 6,5 years before nor the 2,2 years after. The rank correlation of cells between IS and CL is 0,02 (S7 core), 0,13 (S7 day) and −0,40 (S8): the IS map does not predict the others.
- **Forcing S7 flat every day removes what little it had.** The multi-day version made +4 001 USD on IS with 309 trades. The 18:00 → 16:00 version makes +861 USD with 890 trades, since each daily flat adds a round trip, and it is negative on the other two windows gross included on OOS.
- **Gross PnL is close to zero outside IS**: a few hundred to about two thousand USD over 1 600 to 2 400 trades on CL, against 9 000 to 11 000 USD of costs.
- **S8 depends on the 09:00 entry on IS and the opposite on CL**: the best IS column (09:00) is the worst on CL, where the least bad cells start at 10:00. That is the pattern of a period effect, not of a parameter plateau.

## On the significance threshold

The remark that t ≥ 2 is out of reach on 2,4 years is correct: it requires a Sharpe of about 1,3. The remedy is a longer sample, not a lower bar: CL 2015–2021 plus MCL 2021–2023 is about 9 years, where t ≥ 2 needs a Sharpe of about 0,67. No cell here is even positive on that 9-year sample.

## What may and may not be reopened

- No cell of these grids may be picked as a candidate: OOS has been seen for all of them.
- No further variant of EMA-sign or Donchian breakout on MCL (other lengths, timeframes, holding windows, entry times).
- A filtered version (volatility, time or trend filter, as the Unger article hints) is a new hypothesis. It would need a mechanism stated in advance, a pre-registered spec, selection on CL 2015–2021 plus MCL 2021–2023, and the untouched 2026-03 → 2026-11 window as the only verdict.
