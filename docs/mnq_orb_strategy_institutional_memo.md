# MNQ ORB Strategy - Institutional Research Memo

**2026-09-18 audit note:** The archived performance figures below predate a correction to the backtester's 16:00 exit ordering. The dedicated generated audit was not recovered; the corrected reference figures are preserved in the signal sheet and the archived figures below must not be used as the current performance reference.

## Decision for execution (2026-09-16)

The retained MNQ strategy for execution is `orb_vwap_atr`: OR15, long-only,
breakout confirmed at signal-bar close, entry at the next bar open,
`continuous_session_vwap` confirmation, and the ATR(14) ensemble with a 0.50
vote threshold. The stop remains below the opening range with a two-tick
buffer, the target is 2R, and the time exit is 16:00 New York time.

Compression (`weak_close`), the dynamic `noise_area_gate`, and three-state
sizing are historical research variants, not part of the retained execution
strategy. The three-state sizing configuration is definitively rejected for
this MNQ execution reference; use standard risk sizing only.
The added complexity of the full stack is not justified by its incremental
performance for the chosen execution scope. The IS/OOS figures below belong
to that historical full-stack experiment; they must not be quoted as the
performance of `orb_vwap_atr`. The dedicated backtest report for the retained
research configuration is in
`../export/mnq_orb_vwap_atr_retained/performance_report.md`. Its OOS result is
241 trades, 6,343 USD net PnL, Sharpe 1.115, and max drawdown -1,177.50 USD.
The execution repository uses a different nominal sizing base, so this is not
a live PnL forecast.
The direct comparison on the simple ORB + VWAP + ATR sleeve is recorded in
`../export/mnq_orb_vwap_atr_3state_comparison/comparison_report.md`.

## Decision for execution (2026-09-16)

The retained MNQ strategy for execution is `orb_vwap_atr`: OR15, long-only,
breakout confirmed at signal-bar close, entry at the next bar open,
`continuous_session_vwap` confirmation, and the ATR(14) ensemble with a 0.50
vote threshold. The stop remains below the opening range with a two-tick
buffer, the target is 2R, and the time exit is 16:00 New York time.

Compression (`weak_close`), the dynamic `noise_area_gate`, and three-state
sizing are historical research variants, not part of the retained execution
strategy. The three-state sizing configuration is definitively rejected for
this MNQ execution reference; use standard risk sizing only.
The added complexity of the full stack is not justified by its incremental
performance for the chosen execution scope. The IS/OOS figures below belong
to that historical full-stack experiment; they must not be quoted as the
performance of `orb_vwap_atr`. The dedicated backtest report for the retained
research configuration is in
`../export/mnq_orb_vwap_atr_retained/performance_report.md`. Its OOS result is
241 trades, 6,343 USD net PnL, Sharpe 1.115, and max drawdown -1,177.50 USD.
The execution repository uses a different nominal sizing base, so this is not
a live PnL forecast.
The direct comparison on the simple ORB + VWAP + ATR sleeve is recorded in
`../export/mnq_orb_vwap_atr_3state_comparison/comparison_report.md`.

## Decision for execution (2026-09-16)

The retained MNQ strategy for execution is `orb_vwap_atr`: OR15, long-only,
breakout confirmed at signal-bar close, entry at the next bar open,
`continuous_session_vwap` confirmation, and the ATR(14) ensemble with a 0.50
vote threshold. The stop remains below the opening range with a two-tick
buffer, the target is 2R, and the time exit is 16:00 New York time.

Compression (`weak_close`), the dynamic `noise_area_gate`, and three-state
sizing are historical research variants, not part of the retained execution
strategy. The three-state sizing configuration is definitively rejected for
this MNQ execution reference; use standard risk sizing only.
The added complexity of the full stack is not justified by its incremental
performance for the chosen execution scope. The IS/OOS figures below belong
to that historical full-stack experiment; they must not be quoted as the
performance of `orb_vwap_atr`. The dedicated backtest report for the retained
research configuration is in
`../export/mnq_orb_vwap_atr_retained/performance_report.md`. Its OOS result is
241 trades, 6,343 USD net PnL, Sharpe 1.115, and max drawdown -1,177.50 USD.
The execution repository uses a different nominal sizing base, so this is not
a live PnL forecast.
The direct comparison on the simple ORB + VWAP + ATR sleeve is recorded in
`../export/mnq_orb_vwap_atr_3state_comparison/comparison_report.md`.

## Decision for execution (2026-09-16)

The retained MNQ strategy for execution is `orb_vwap_atr`: OR15, long-only,
breakout confirmed at signal-bar close, entry at the next bar open,
`continuous_session_vwap` confirmation, and the ATR(14) ensemble with a 0.50
vote threshold. The stop remains below the opening range with a two-tick
buffer, the target is 2R, and the time exit is 16:00 New York time.

Compression (`weak_close`), the dynamic `noise_area_gate`, and three-state
sizing are historical research variants, not part of the retained execution
strategy. The three-state sizing configuration is definitively rejected for
this MNQ execution reference; use standard risk sizing only.
The added complexity of the full stack is not justified by its incremental
performance for the chosen execution scope. The IS/OOS figures below belong
to that historical full-stack experiment; they must not be quoted as the
performance of `orb_vwap_atr`. The dedicated backtest report for the retained
research configuration is in
`../export/mnq_orb_vwap_atr_retained/performance_report.md`. Its OOS result is
241 trades, 6,343 USD net PnL, Sharpe 1.115, and max drawdown -1,177.50 USD.
The execution repository uses a different nominal sizing base, so this is not
a live PnL forecast.
The direct comparison on the simple ORB + VWAP + ATR sleeve is recorded in
`../export/mnq_orb_vwap_atr_3state_comparison/comparison_report.md`.

## Decision for execution (2026-09-16)

The retained MNQ strategy for execution is `orb_vwap_atr`: OR15, long-only,
breakout confirmed at signal-bar close, entry at the next bar open,
`continuous_session_vwap` confirmation, and the ATR(14) ensemble with a 0.50
vote threshold. The stop remains below the opening range with a two-tick
buffer, the target is 2R, and the time exit is 16:00 New York time.

Compression (`weak_close`), the dynamic `noise_area_gate`, and three-state
sizing are historical research variants, not part of the retained execution
strategy. The three-state sizing configuration is definitively rejected for
this MNQ execution reference; use standard risk sizing only.
The added complexity of the full stack is not justified by its incremental
performance for the chosen execution scope. The IS/OOS figures below belong
to that historical full-stack experiment; they must not be quoted as the
performance of `orb_vwap_atr`. The dedicated backtest report for the retained
research configuration is in
`../export/mnq_orb_vwap_atr_retained/performance_report.md`. Its OOS result is
241 trades, 6,343 USD net PnL, Sharpe 1.115, and max drawdown -1,177.50 USD.
The execution repository uses a different nominal sizing base, so this is not
a live PnL forecast.
The direct comparison on the simple ORB + VWAP + ATR sleeve is recorded in
`../export/mnq_orb_vwap_atr_3state_comparison/comparison_report.md`.

## 1. Executive Summary

This memo now follows the real retained stack in the same order as the research logic:

1. base signal,
2. VWAP filter,
3. ATR ensemble,
4. compression and dynamic gating,
5. 3-state overlay on top of the retained-final sleeve.

The historical full-stack configuration studied below is `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate` with `OR15`, `long`, VWAP confirmation, ATR vote, `weak_close`, and `noise_area_gate`. The later 3-state campaign was rerun directly on that full-stack sleeve. Within that experiment, `realized_vol_ratio_15_60` is **not** the best overlay: the strongest OOS sizing candidate was `sizing_3state_atr_ratio_10_30`. Neither overlay is retained for execution.

## 2. Layer 1 - Base Signal

The retained signal is an ORB continuation entry:

- OR window: `15` minutes
- Direction: `long`
- Entry confirmation: breakout bar closes beyond the OR boundary
- Execution: next bar open
- Entry buffer / stop buffer: `2` / `2` ticks
- Target: `2.0R`
- Time exit: `16:00:00`

Current retained-final IS/OOS metrics:

| Scope | Net PnL | Sharpe | Max DD | Trades |
| --- | --- | --- | --- | --- |
| IS | 4,620.5 USD | 0.604 | -1,913.5 USD | 222 |
| OOS | 5,155.5 USD | 1.559 | -590.5 USD | 76 |

*Figure historique « ORB mechanics » non récupérée.*

*Figure historique « OR minutes × target multiple » non récupérée.*

Readout:

- The retained point `OR15 / target 2.0R` sits inside a usable zone rather than on an isolated spike.
- The memo now shows the IS and OOS surfaces side by side instead of only describing the selected point.

## 3. Layer 2 - VWAP Filter

VWAP belongs to the retained signal stack itself. The retained point is:

- VWAP confirmation: `enabled`
- VWAP column: `continuous_session_vwap`
- Current time exit used with that filter: `16:00:00`

Retained-final IS/OOS metrics are the same trade set shown above because VWAP is already embedded in the retained sleeve:

| Scope | Net PnL | Sharpe | Max DD | Trades |
| --- | --- | --- | --- | --- |
| IS | 4,620.5 USD | 0.604 | -1,913.5 USD | 222 |
| OOS | 5,155.5 USD | 1.559 | -590.5 USD | 76 |

*Figure historique « VWAP confirmation × time exit » non récupérée.*

Readout:

- The memo now makes VWAP explicit instead of burying it under the old sizing narration.
- The selected point is `vwap=True / 16:00 exit`.

## 4. Layer 3 - ATR Ensemble

ATR also belongs to the retained-final sleeve itself. The selected retained point is:

- ATR window: `14`
- Vote threshold: `0.50`
- Quantile lows: `(20, 25, 30)`
- Quantile highs: `(90, 95)`

Current retained-final IS/OOS metrics:

| Scope | Net PnL | Sharpe | Max DD | Trades |
| --- | --- | --- | --- | --- |
| IS | 4,620.5 USD | 0.604 | -1,913.5 USD | 222 |
| OOS | 5,155.5 USD | 1.559 | -590.5 USD | 76 |

*Figure historique « ATR window × vote threshold » non récupérée.*

Readout:

- This is the layer where ATR actually lives in the retained-final architecture.
- The retained point `ATR(14) / vote 0.50` is now shown directly against nearby alternatives in IS and OOS.

## 5. Layer 4 - Compression And Dynamic Gate

The retained-final sleeve adds a pattern overlay and a dynamic noise gate:

- Compression mode: `weak_close`
- Compression usage: `soft_vote_bonus`
- Dynamic mode: `noise_area_gate`
- Noise lookback / VM: `30` / `1.00`
- Dynamic schedule: `continuous_on_bar_close`
- Threshold style: `max_or_high_noise`

Current retained-final IS/OOS metrics:

| Scope | Net PnL | Sharpe | Max DD | Trades |
| --- | --- | --- | --- | --- |
| IS | 4,620.5 USD | 0.604 | -1,913.5 USD | 222 |
| OOS | 5,155.5 USD | 1.559 | -590.5 USD | 76 |

*Figure historique « compression mode × usage » non récupérée.*

*Figure historique « noise lookback × noise VM » non récupérée.*

Readout:

- `weak_close / soft_vote_bonus` and `noise_area_gate` are no longer treated as footnotes.
- The memo now shows where the selected gate sits relative to nearby settings.

## 6. Layer 5 - 3-State Overlay On Retained Final

This layer is now run on the retained-final sleeve itself, not on the old nominal ORB branch.

Top retained-final overlay candidates by feature:

| feature_name | family | feature_selection_score | best_bucket_is | worst_bucket_is | valid_for_overlay |
| --- | --- | --- | --- | --- | --- |
| overnight_range_pts | volatility | 4.940 | low | mid | 1 |
| realized_vol_ratio_15_60 | volatility | 4.363 | high | mid | 1 |
| gap_abs_atr20 | extension | 4.358 | high | low | 1 |
| signal_extension_over_or | extension | 4.275 | high | mid | 1 |
| opening_range_width_pts | volatility | 4.098 | mid | high | 1 |
| atr_ratio_10_30 | volatility | 4.057 | high | low | 1 |

*Figure historique IS « fast/slow realized-volatility ratios » non récupérée.*

*Figure historique OOS « fast/slow realized-volatility ratios » non récupérée.*

The retained-final campaign compared the nominal sleeve against multiple 3-state overlays:

| variant_name | feature_name | oos_net_pnl | oos_sharpe | oos_max_drawdown | oos_net_pnl_retention_vs_nominal | oos_sharpe_delta_vs_nominal | oos_max_drawdown_improvement_vs_nominal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | nominal | 5155.500 | 1.559 | -590.500 | 1.000 | 0.000 | 0.000 |
| sizing_3state_atr_ratio_10_30 | atr_ratio_10_30 | 4022.500 | 1.799 | -328.000 | 0.780 | 0.240 | 0.445 |
| sizing_3state_overnight_range_pts | overnight_range_pts | 2179.500 | 1.253 | -517.500 | 0.423 | -0.305 | 0.124 |
| sizing_3state_realized_vol_ratio_15_60 | realized_vol_ratio_15_60 | 2374.500 | 0.972 | -887.500 | 0.461 | -0.587 | -0.503 |

*Figure historique « OOS Sharpe by 3-state overlay » non récupérée.*

Specific readout for `realized_vol_ratio_15_60` on retained final:

- Variant: `sizing_3state_realized_vol_ratio_15_60`
- IS net / Sharpe / maxDD: `2,207.0 USD` / `0.405` / `-1,632.0 USD`
- OOS net / Sharpe / maxDD: `2,374.5 USD` / `0.972` / `-887.5 USD`
- OOS retention vs nominal: `0.461` 
- OOS Sharpe delta vs nominal: `-0.587` 

Bucket map used for `realized_vol_ratio_15_60` in the retained-final campaign:

| bucket_label | lower_bound | upper_bound | risk_multiplier |
| --- | --- | --- | --- |
| low | 0.519 | 1.085 | 0.750 |
| mid | 1.085 | 1.375 | 0.500 |
| high | 1.375 | 1.874 | 1.000 |

Readout:

- `realized_vol_ratio_15_60` still ranks near the top in IS feature selection, but it does **not** hold up as the best retained-final overlay in OOS.
- In the latest retained-final campaign, the strongest OOS 3-state candidate is `sizing_3state_atr_ratio_10_30` with net `4,022.5 USD`, Sharpe `1.799`, and maxDD `-328.0 USD`.
- The retained-final nominal sleeve remains stronger in raw PnL than the `15/60` overlay.

## 7. Recommendation

The logical reading of the stack is now:

1. the retained edge starts with the `OR15 / long / next-open` base signal;
2. VWAP and ATR are structural parts of that retained signal stack;
3. compression and dynamic gating belonged to the historical full-stack filtering logic;
4. the 3-state overlay was a **separate last layer** evaluated on that full-stack sleeve.

Historical readout of the full-stack experiment:

- keep the full-stack nominal sleeve clearly identified as the reference for its own overlay comparisons;
- do **not** describe `realized_vol_ratio_15_60` as the retained-final 3-state winner;
- if a 3-state overlay is revisited in research on that sleeve, inspect `sizing_3state_atr_ratio_10_30` first, rather than `realized_vol_ratio_15_60`.

Execution decision: retain `orb_vwap_atr` as specified above. Re-evaluate the
full stack only if a controlled, like-for-like comparison demonstrates enough
incremental benefit to justify its operational complexity.
