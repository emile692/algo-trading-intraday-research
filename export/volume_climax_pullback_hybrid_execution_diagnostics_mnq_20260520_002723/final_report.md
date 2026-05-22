# Volume Climax Pullback Hybrid Execution Diagnostics

## 1. Executive Summary
- Baseline 1H/1H net PnL: `7535.12` USD.
- Hybrid after-entry-fill net PnL: `-1675.73` USD.
- Hybrid next-execution-bar net PnL: `-1567.28` USD.
- Matching confidence: `high`. Baseline-only `48`, hybrid-only `8`, uncertain `0`.
- Baseline 1H/1H appears biased favorably relative to the intrabar-aware replay.
- Main driver of alpha loss: `winner_to_loser`.
- Recalibration finds a positive ex-post zone without changing the alpha.

## 2. Input Runs
- Comparison dir: `export\volume_climax_pullback_hybrid_execution_validation_mnq_20260519_220109`.
- Minute dataset: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet`.
- Symbol: `MNQ`.
- Variant: `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2`.

## 3. Baseline vs Hybrid Metrics
| scenario | trades | net_pnl_usd | profit_factor | sharpe | max_drawdown_usd | expectancy_usd | hit_rate | raw_signal_count | avg_minutes_held |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_1h | 315 | 7535.1250 | 1.7279 | 1.3106 | -790.0000 | 23.9210 | 0.3556 | 324 |  |
| hybrid_after_entry_fill | 275 | -1675.7250 | 0.8508 | -0.3672 | -2251.8250 | -6.0935 | 0.2400 | 324 | 32.9927 |
| hybrid_next_execution_bar | 275 | -1567.2750 | 0.8604 | -0.3428 | -2143.3750 | -5.6992 | 0.2436 | 324 | 33.4036 |

## 4. PnL Bridge
| bridge_component | amount |
| --- | --- |
| baseline_net_pnl | 7535.1250 |
| removed_or_missing_trades_effect | -1936.4500 |
| entry_price_effect | -609.6500 |
| stop_before_target_effect | -6187.4000 |
| target_not_reached_effect | 89.8750 |
| time_stop_effect | -323.3250 |
| eod_effect | -116.4000 |
| ambiguous_stop_first_effect | 0.0000 |
| residual_unexplained | -127.5000 |
| hybrid_net_pnl | -1675.7250 |

## 5. Divergence Taxonomy
| divergence_type | count | pct_of_setups | baseline_pnl_sum | hybrid_after_pnl_sum | hybrid_next_pnl_sum | delta_pnl_after_sum | delta_pnl_next_sum | avg_delta_pnl_after | median_delta_pnl_after | winrate_baseline | winrate_hybrid_after | winrate_hybrid_next |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| winner_to_loser | 37 | 0.1146 | 5756.6000 | -1588.5000 | -1480.0500 | -7345.1000 | -7236.6500 | -198.5162 | -212.1750 | 1.0000 | 0.0000 | 0.0270 |
| baseline_trade_missing_in_hybrid | 48 | 0.1486 | 1603.0500 | 0.0000 | 0.0000 | -1603.0500 | -1603.0500 | -33.3969 | 13.2500 | 0.2917 | 0.0000 | 0.0000 |
| hybrid_trade_missing_in_baseline | 8 | 0.0248 | 0.0000 | -333.4000 | -333.4000 | -333.4000 | -333.4000 | -41.6750 | -52.7500 | 0.0000 | 0.1250 | 0.1250 |
| time_stop_changed | 11 | 0.0341 | 677.8250 | 354.5000 | 354.5000 | -323.3250 | -323.3250 | -29.3932 | -16.2500 | 0.5455 | 0.5455 | 0.5455 |
| exit_reason_changed | 214 | 0.6625 | -503.7500 | -655.2500 | -655.2500 | -151.5000 | -151.5000 | -0.7079 | 0.0000 | 0.2523 | 0.2523 | 0.2523 |

## 6. Exit Reason Transition
| baseline_exit_reason | hybrid_exit_reason | count | delta_pnl_sum |
| --- | --- | --- | --- |
| stop | stop_1m | 148 | 132.0000 |
| target | target_1m | 44 | 0.0000 |
| target | stop_1m | 24 | -5356.9000 |
| time_stop | time_stop_1m | 16 | -451.0000 |
| target | time_stop_1m | 9 | -1213.4750 |
| time_stop | stop_1m | 9 | -903.0000 |
| stop_ambiguous_first | stop_1m | 6 | 7.0000 |
| stop | time_stop_1m | 5 | 280.5000 |
| stop | target_1m | 3 | 637.9250 |
| target | eod_flat_1m | 2 | -392.4500 |
| eod_flat | stop_1m | 1 | -15.0000 |
| eod_flat | eod_flat_1m | 0 | 0.0000 |

## 7. Baseline Winners Destroyed
- Count: `37`.
- Aggregate baseline PnL of destroyed winners: `5756.60` USD.
- Aggregate hybrid PnL on the same setups: `-1588.50` USD.

## 8. 1min Path Reconstruction
- Reconstructed paths: `275`.
- First touch = stop: `194`.
- First touch = target: `48`.
- First touch = both same minute: `0`.

## 9. Execution Convention Sensitivity
| config_name | execution_timeframe | entry_timing | protective_orders_active_from | ambiguous_policy | entry_delay_minutes | trades | net_pnl | winrate | avg_trade | profit_factor | max_drawdown | sharpe_if_available | pnl_vs_baseline | pnl_vs_hybrid_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline_1h | 1h | n/a | n/a | n/a | 0 | 315 | 7535.1250 | 0.3556 | 23.9210 | 1.7279 | -790.0000 |  | 0.0000 | 9210.8500 |
| H_hybrid_delayed_entry_5min | 1min | next_execution_bar_open | after_entry_fill | stop_first | 5 | 241 | -463.4500 | 0.2946 | -1.9230 | 0.9535 | -1685.5500 |  | -7998.5750 | 1212.2750 |
| I_hybrid_delayed_entry_15min | 1min | next_execution_bar_open | after_entry_fill | stop_first | 15 | 221 | -619.3000 | 0.3258 | -2.8023 | 0.9407 | -1571.3750 |  | -8154.4250 | 1056.4250 |
| C_hybrid_next_execution_bar | 1min | next_execution_bar_open | next_execution_bar | stop_first | 0 | 275 | -1567.2750 | 0.2436 | -5.6992 | 0.8604 | -2143.3750 |  | -9102.4000 | 108.4500 |
| B_hybrid_after_entry_fill | 1min | next_execution_bar_open | after_entry_fill | stop_first | 0 | 275 | -1675.7250 | 0.2400 | -6.0935 | 0.8508 | -2251.8250 |  | -9210.8500 | 0.0000 |
| E_hybrid_same_timestamp_next_execution_bar | 1min | same_timestamp_execution_open | next_execution_bar | stop_first | 0 | 324 | -2161.7750 | 0.2191 | -6.6721 | 0.8246 | -2536.5000 |  | -9696.9000 | -486.0500 |
| D_hybrid_same_timestamp_after_entry_fill | 1min | same_timestamp_execution_open | after_entry_fill | stop_first | 0 | 324 | -2601.0750 | 0.2099 | -8.0280 | 0.7894 | -2975.8000 |  | -10136.2000 | -925.3500 |
- Not executed: `F_hybrid_next_execution_bar_open_but_neutral_ambiguous_skip, G_hybrid_next_execution_bar_open_but_target_first_ambiguous`.

## 10. Stop/Target Recalibration Diagnostic
| stop_multiplier | target_multiplier | trades | net_pnl | winrate | avg_trade | profit_factor | max_drawdown_if_reconstructable | median_holding_minutes | comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7500 | 2.0000 | 275 | 510.5250 | 0.1964 | 1.8565 | 1.0535 | -1556.0000 | 9.0000 | Ex-post 1min replay with fixed alpha and stop-first ambiguity convention. |

## 11. Verdict
- `Alpha partially survives but requires intrabar-aware risk geometry.`

## 12. Next Actions
- Abandonner le baseline 1H comme reference de performance brute pour toute strategie intraday path-dependent.
- Exiger un diagnostic intrabar 1min avant de publier une strategie research avec stops/targets/time-stop intraday.
- Recalibrer la geometrie stop/target a partir des distributions MAE/MFE 1min avant tout nouveau jugement sur l'alpha.
- Tester des conventions d'entree decalees ou confirmees post-signal pour verifier si l'immediatete degrade structurellement l'execution.
- Ajouter un guardrail CI qui bloque les campagnes intraday si aucun audit baseline-vs-path n'est genere.
