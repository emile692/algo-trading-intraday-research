# MCL — five generic crude-oil strategies (Mudrex article) — closure

**Status:** `REJECTED` (all five fail the IS gate; none replicates on the CL 2015–2021 holdout)
**Date:** 2026-11-11
**Spec:** [mcl_mudrex_config.json](mcl_mudrex_config.json), sha256 `1b08ea54…db456800`, fixed before any return on the Databento data was computed.
**OOS (2024-01 → 2026-03) and extension (2026-03 → 2026-11):** not computed, still unseen for these rules.

## Hypothesis

The five strategies of <https://mudrex.com/learn/best-crude-oil-trading-strategy/> make money on MCL when transposed to intraday. The page is mostly daily and qualitative, so each strategy got one canonical intraday version: signals on closed 15-minute bars (indicators on the full Globex session), trading 09:00–14:30 New York, flat at the last core bar, one contract, no grid.

| Id | Page rule | Frozen intraday version |
|---|---|---|
| S1 | SMA 50/200 regime, enter on pullback to the SMA 50, exit below it or on a MACD cross | Same on 15m bars, stop 1,5 ATR(14), no target |
| S2 | Break and hold of VWAP with RSI > 50 after the US open, enter on the first pullback to VWAP, stop 1 ATR, target 2:1 | As written, limit at VWAP, entries 09:30–12:00, short side mirrored |
| S3 | Close through support/resistance on above-average volume, enter the retest | Level = prior trading-day high/low, 15m volume > 20-bar average, limit at the level, stop 1 ATR, target 2R |
| S4 | RSI < 30 / > 70 at range boundaries with Bollinger extremes, not in trends | Close outside Bollinger(20, 2) with RSI extreme, only if abs(SMA50 − SMA200) ≤ 1 ATR; target = band middle, stop 1 ATR |
| S5 | After the EIA report, wait for the spike to fade and follow the prevailing trend | EIA Wednesdays, entry 10:45, direction = sign(SMA50 − SMA200), stop 1,5 ATR, target 2R |

Choices the page does not make and that were fixed here: stops and targets of S1, S3 and S5 (taken from the page's general "1–1,5 ATR" rule and the 2:1 of S2), the range filter of S4, the short side of S2. Left out: Bollinger compression (S3), Stochastic (S4), pre-event positioning (S5, needs consensus data). S2–S4 take no entry 10:00–11:00 on EIA Wednesdays.

## Protocol

- Data: `MCL.v.0` 1-minute, Databento, 2021-07-12 → 2026-11-10; `CL.v.0` 1-minute, 2015-01-01 → 2021-07-11. Volume-based continuous contract, additively back-adjusted at each `instrument_id` change. The 26 days Databento flags as degraded are excluded.
- Costs: 1,25 USD commission and 1 tick slippage per side, 4,50 USD per round trip, on every fill including limits.
- IS gate (set before the run): at least 100 trades, net PnL > 0, daily t-stat ≥ 2,0. The t-stat is mean ÷ sample standard deviation × √n of the daily net PnL over all valid sessions, days without a trade counted as zero.
- PRE_CL: CL prices simulated with MCL size and costs. Informative holdout, not part of the gate. CL and MCL were never spliced.
- A pipeline smoke test on 48 sessions of Yahoo 5-minute bars was run before the Databento files were restored. Only data paths, windows, roll adjustment and day exclusions were changed after it; no strategy rule or parameter.

## Results

| Strategy | Window | Sessions | Trades | Net USD | Gross USD | PF | Sharpe | t | Max DD USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 trend SMA pullback | IS | 609 | 691 | +334 | +3 443 | 1,02 | 0,10 | 0,16 | −1 552 |
| S1 trend SMA pullback | PRE_CL | 1 620 | 1 725 | −4 138 | +3 625 | 0,85 | −0,89 | −2,25 | −5 216 |
| S2 VWAP + RSI pullback | IS | 609 | 284 | −49 | +1 229 | 0,99 | −0,03 | −0,05 | −1 156 |
| S2 VWAP + RSI pullback | PRE_CL | 1 620 | 829 | −4 773 | −1 042 | 0,65 | −2,17 | −5,51 | −4 947 |
| S3 breakout retest | IS | 609 | 211 | +526 | +1 475 | 1,09 | 0,36 | 0,56 | −638 |
| S3 breakout retest | PRE_CL | 1 620 | 556 | −3 487 | −985 | 0,63 | −1,99 | −5,05 | −3 632 |
| S4 range RSI/Bollinger | IS | 609 | 175 | −1 074 | −286 | 0,80 | −0,78 | −1,20 | −1 339 |
| S4 range RSI/Bollinger | PRE_CL | 1 620 | 477 | −1 635 | +512 | 0,80 | −0,78 | −1,97 | −2 433 |
| S5 EIA post-event trend | IS | 609 | 107 | +1 392 | +1 873 | 1,39 | 0,94 | 1,47 | −613 |
| S5 EIA post-event trend | PRE_CL | 1 620 | 289 | −610 | +691 | 0,91 | −0,30 | −0,76 | −1 165 |

All five: IS gate `FAIL`.

## Reading

- **No rule clears the gate.** The best IS t-stat is 1,47 (S5). With five rules tested, the best of five reaching a Sharpe near 1 over 2,4 years is expected by chance.
- **S5 (EIA) does not replicate.** +1 392 USD on 107 trades on MCL, then −610 USD on 289 trades on CL 2015–2021. The IS Sharpe of 0,94 has a standard error of about 0,65 on 2,4 years.
- **S1 has a positive gross on both samples** (+3 443 and +3 625 USD) that costs remove entirely: about 2 to 5 USD gross per trade against 4,50 USD of costs.
- **S2 and S3 are strongly negative on CL** (t −5,5 and −5,1), gross included. Their limit entries at VWAP or at the broken level are filled mostly when the move fails.
- **S4 is negative on both samples.**
- Consistent with the September 2026 campaign, where 16 frozen mechanisms (MA crossover, breakouts, VWAP and z-score reversion among them) were not retained on MCL.

## What may and may not be reopened

- No rescue variant: no other period, threshold, timeframe, stop, target, entry window or single-strategy pick on this data.
- OOS and the 2026 extension stay unread for these rules.
- S5 is not a candidate. An EIA idea would need a new pre-registered campaign with information this one did not use (consensus versus actual inventory surprise), judged on MCL IS and on CL 2015–2021 together.
- Files: [mcl_mudrex_results.csv](mcl_mudrex_results.csv), [run_mcl_mudrex.py](run_mcl_mudrex.py), `trades/`.
