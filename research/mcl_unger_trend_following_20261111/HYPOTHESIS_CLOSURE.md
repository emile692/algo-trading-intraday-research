# MCL — two Unger Academy CL trend-following strategies — closure

**Status:** `REJECTED` (both fail the IS gate; neither replicates on the CL 2015–2021 holdout)
**Date:** 2026-11-11
**Spec:** [config.json](config.json), sha256 `e9d26596…da5ffce2`, fixed before any return of these rules was computed.
**OOS (2024-01 → 2026-03) and extension (2026-03 → 2026-11):** not computed, still unseen for these rules.

## Hypothesis

The two strategies of <https://ungeracademy.com/blog/crude-oil-trading-cl-trend-following-strategies> make money on MCL.

**The page cannot be reproduced as published.** It gives no EMA lengths, no Donchian length, no stop, no target, no costs, and its filters ("specific price patterns", "volatility, time-based and trend filters") are not defined. It reports 333 900 USD since 2010 for the first strategy and about 170 000 USD for the second, on CL, and states that its figures are illustrative. What was tested here is the unfiltered core of each idea with conventional defaults, fixed before the run and not fitted to those figures.

| Id | Page rule | Frozen version |
|---|---|---|
| S7 | 15m bars, full session 18:00–17:00, long when the fast EMA is above the slow EMA, short when below, held across sessions | EMA 50 / 200 on 15m bars, always in the market, reversal at the first 1-minute open after the 15m close showing the cross; no stop, no target, no filter |
| S8 | 15m bars, Donchian channel breakout, held a few hours | Channel = high/low of the last 20 closed 15m bars; stop order one tick beyond it, 09:00–14:00 New York, first breakout of the day only; trailing stop on the opposite channel side; flat at 14:30; no filter |

## Protocol

Same data, roll adjustment, costs (4,50 USD per round trip), windows and IS gate as [the Mudrex campaign](../mcl_mudrex_strategies_20261111/HYPOTHESIS_CLOSURE.md): at least 100 trades, net PnL > 0, daily t-stat ≥ 2,0. One spec per strategy, no grid.

Specific to S7:
- PnL is marked to market by trading day (18:00 New York roll), because positions span sessions.
- One extra round trip is charged at each contract roll while in position.
- Degraded-quality days are not removed, since positions span them (639 IS days against 609 for the intraday rules).
- It holds overnight and over weekends, which Topstep does not allow. Even a passing result would not have been usable in the Topstep book as is.

## Results

| Strategy | Window | Days | Trades | Net USD | Gross USD | PF | Win rate | Sharpe | t | Max DD USD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S7 multiday EMA cross | IS | 639 | 309 | +4 001 | +5 254 | 1,14 | 33 % | 0,48 | 0,77 | −2 353 |
| S7 multiday EMA cross | PRE_CL | 1 682 | 883 | −428 | +3 752 | 0,99 | 29 % | −0,03 | −0,09 | −3 704 |
| S8 intraday Donchian | IS | 609 | 593 | +3 604 | +6 272 | 1,13 | 47 % | 0,72 | 1,11 | −2 185 |
| S8 intraday Donchian | PRE_CL | 1 620 | 1 591 | −6 516 | +644 | 0,86 | 42 % | −0,95 | −2,40 | −8 006 |

Both: IS gate `FAIL`.

## Reading

- **Both are positive on MCL IS but far from significant** (t 0,77 and 1,11). The IS window contains 2022, a strong trending year for crude.
- **Neither holds on CL 2015–2021.** S7 is flat after costs (−428 USD on 883 trades); S8 is clearly negative (t −2,40) and its gross is close to zero (+644 USD on 1 591 trades).
- **S8 trades almost every session** (593 trades on 609 days). With a 20-bar channel that includes the night, price is often already outside it at 09:00: 101 of the IS entries are filled on the 09:00 bar. It behaves like a daily bet on the overnight direction, not like a selective breakout.
- **The published figures are not contradicted, but not confirmed either.** They rely on filters the page does not disclose and on a sample starting in 2010; the unfiltered core has no edge after costs on 2015–2021.
- Close to mechanisms already rejected on MCL: `ma_crossover_trend`, `rolling_range_breakout` and `session_range_breakout` (September 2026).

## What may and may not be reopened

- No rescue variant on this data: no other EMA pair, channel length, entry window, stop, or added volatility / time / trend filter chosen after seeing these results.
- OOS and the 2026 extension stay unread for these rules.
- A test of the actual Unger rules would need their full published specification (lengths, filters, stops), frozen as a new campaign.
- Files: [results.csv](results.csv), [run.py](run.py), `trades/`.

## Addendum, 2026-11-11

At the user's request, a parameter landscape of an intraday version of S7 and of S8 was computed afterwards, **including the OOS window 2024-01 → 2026-03**. The statements above that OOS is unseen no longer hold for these two rule families; only 2026-03-20 → 2026-11-10 remains untouched. The landscape confirms the rejection: see [its closure](../mcl_trend_landscape_20261111/HYPOTHESIS_CLOSURE.md).
