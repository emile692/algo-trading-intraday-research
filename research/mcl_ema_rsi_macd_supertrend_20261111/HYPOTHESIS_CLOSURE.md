# MCL — EMA / RSI / MACD / Supertrend intraday plan — closure

**Status:** `REJECTED` (IS gate failed; negative on the CL 2015–2021 holdout)
**Date:** 2026-11-11
**Spec:** [config.json](config.json), sha256 `5307db50…914c4b44`, fixed before any return of this rule was computed.
**OOS (2024-01 → 2026-03) and extension (2026-03 → 2026-11):** not computed, still unseen for this rule.

## Hypothesis

A discretionary "Crude Oil Option Intraday Trading Plan" written for MCX (India) makes money on MCL when its indicator set is turned into a mechanical rule.

**The source text was truncated**: it stops at "Buy Setup:", with no entry, exit, stop or target rules. The rule below was reconstructed from its indicator section and is an interpretation.

- Trend, last closed 15-minute bar: close > EMA 20 > EMA 50 > EMA 200 for longs, mirrored for shorts.
- Entry, on a closed 5-minute bar: RSI(14) > 60 (shorts: < 40), MACD(12, 26, 9) line crossing its signal in the trade direction on that bar, 5m volume above its 20-bar average, Supertrend(10, 3) already on the trade side. Market order at the next 1-minute open.
- Exit: trailing stop on the 5m Supertrend line, ratcheting only in favour of the trade; otherwise flat at the last core bar. No target.
- Time rules: entries 09:30–14:00 New York (the plan starts at 6:30 PM IST = 09:00 New York and waits 15–30 minutes); no entry 10:00–11:00 on EIA Wednesdays.
- Left out: hand-drawn trendlines and price-action zones, DXY and equity-index sentiment, OBV, execution through options (the future is traded).

## Protocol

Same data, roll adjustment, excluded days, costs (4,50 USD per round trip), windows and IS gate as [the Mudrex campaign](../mcl_mudrex_strategies_20261111/HYPOTHESIS_CLOSURE.md): at least 100 trades, net PnL > 0, daily t-stat ≥ 2,0. One spec, no grid.

## Results

| Window | Sessions | Trades | Net USD | Gross USD | PF | Win rate | Sharpe | t | Max DD USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS (MCL 2021-07 → 2023-12) | 609 | 137 | +198 | +814 | 1,05 | 39 % | 0,12 | 0,19 | −831 |
| PRE_CL (CL 2015 → 2021-07) | 1 620 | 363 | −548 | +1 086 | 0,92 | 41 % | −0,24 | −0,60 | −1 361 |

IS gate: `FAIL`. On IS, 81 of 137 trades end on the Supertrend stop and 56 at the end of the session; median risk per trade is about 85 USD.

## Reading

- **Indistinguishable from zero.** Gross is slightly positive on both samples (about 3 to 6 USD per trade) and does not cover 4,50 USD of costs.
- **The filters remove trades, not noise.** Four simultaneous conditions leave about one trade every four to five sessions, and the average trade is no better than the simpler trend rules already tested.
- Same family as the trend and breakout mechanisms rejected on MCL in September 2026 and in the Mudrex campaign of the same day.

## What may and may not be reopened

- No rescue variant on this data: no other RSI threshold, EMA set, Supertrend setting, timeframe or entry window.
- OOS and the 2026 extension stay unread for this rule.
- If the missing part of the plan (Buy and Sell setups, stops, targets) is provided, it may be tested once as a new frozen spec with a new campaign id; this result stays as is.
- Files: [results.csv](results.csv), [run.py](run.py), `trades/`.
