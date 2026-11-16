# Event/flow mechanisms — Phase 1/2 comparison

This campaign froze the canonical definitions before performance inspection. The Treasury branch was subsequently completed with paid ZF data and the mandated controls. No strategy backtest or cost stress was run because none of the three mechanisms passed its event-study gate.

| Mechanism | Instrument | Economic expectation | Events | IS/OOS | Canonical evidence | Recent evidence | Phase gate |
|---|---|---|---:|---:|---|---|---|
| Treasury auction pressure | ZF primary, ZN comparison | Futures down pre-auction, up post-result | 159 5Y; 171 7Y; 162 10Y | Fixed subsamples | 5Y→ZF pre −1.88 bps significant; post +0.40 insignificant; other mappings insignificant | 5Y→ZF 2024+: pre +0.35, post −0.41 bps; wrong signs | **REJECTED** |
| Close hedging flow | MNQ | ROD positively predicts final 30m | 1,667 | 1,126 / 541 | Slope −0.0125; HAC t −0.77; no decile monotonicity | 2021+ correlation −0.001 | Reject mechanism |
| London-fix inventory reversal | 6E | 6E down before 16:00 London, up after | 2,959 | 2,262 / 697 | Pre +0.44 bps (wrong sign); post +0.22 bps, insignificant | 2021+: pre +0.78, post −0.56 (both wrong as a pair) | Reject mechanism |

## Data audit

- MNQ: 2,401,697 one-minute rows, 2019-05-05 to 2026-03-20, calendar-front continuous contract, 27 instrument transitions, no duplicate timestamps or missing OHLCV values.
- 6E: 4,060,880 one-minute rows, 2015-01-01 to 2026-09-22, volume-ranked continuous contract, no duplicate timestamps or missing OHLCV values. The processed file does not retain `instrument_id`, so roll-transition diagnostics cannot be reconstructed from it.
- ZN: 4,314,095 one-minute rows, 2013-05-17 to 2026-09-18, volume-ranked continuous contract, 54 instrument transitions, median 400 bars in the 08:20–15:00 ET audit session.
- ZF: 4,069,036 one-minute rows from 2013-05-17 through 2026-09-18; volume-ranked `v.0`, 56 roll transitions, no structural quality failure.
- Treasury calendar: official FiscalData API, 1,115 nominal coupon auctions; exact bid deadlines retained, unavailable result times left null.

## Interpretation

The key separation is preserved: these are mechanism results, not trading results. MNQ and 6E fail their canonical mechanism gates. Treasury now also fails after adding ZF: its historical 5Y pre-leg is visible, but the recovery leg is absent, 2024+ has the wrong signs, T−60 is as strong as T=0, and the best 10 events contribute 79% of signed effect. Exact result timestamps remain unavailable, but the deadline evidence is already sufficient for final rejection.

Strategy columns requested for the final comparison (IS/OOS Sharpe, PF, PnL, MaxDD, cost stress, and sleeve correlation) are intentionally `N/R` at this phase because no canonical strategy was authorized.

Detailed outputs live in `research/treasury_auction_pressure/`, `research/mnq_close_hedging_flow/`, `research/six_e_fx_fix_reversal/`, and `research/event_flow_mechanisms_comparison/`.
