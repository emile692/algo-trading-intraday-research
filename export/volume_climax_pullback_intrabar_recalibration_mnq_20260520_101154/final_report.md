# Volume Climax Pullback Intrabar Recalibration Campaign

## 1. Executive Summary
- A tradable 1min-executed edge does not clearly survive after simple recalibration.
- The main improvement family is `none` with delay `30` minutes.
- The gain comes primarily from `entry delay and tighter stop / wider target geometry`.
- Verdict: `Reject: alpha does not survive realistic execution.`.

## 2. Reminder: Why 1H Baseline Is Invalidated
- Previous diagnostics flagged the hourly baseline as biased, with dominant divergence cause `high` confidence matching and top driver `winner_to_loser`.
- The best full-sample ex-post recalibration from phase 1 was stop x0.75 / target x2.00 for 510.52 USD, but this campaign does not use that full-sample result for selection.
- The hourly baseline remains historical context only and is not used for model selection in this phase.

## 3. Research Design
- Signal 1H unchanged: `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2`.
- Execution path: 1min only.
- Calibration and ranking: IS only, with OOS reported afterwards as blind evaluation.
- Base grid is exhaustive on stop/target/delay for no-filter configs and compact on filter families.

## 4. Current Hybrid Benchmark
| scenario | trades | net_pnl_usd | profit_factor | sharpe | max_drawdown_usd | expectancy_usd | hit_rate | raw_signal_count | avg_minutes_held |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_1h | 315 | 7535.1250 | 1.7279 | 1.3106 | -790.0000 | 23.9210 | 0.3556 | 324 |  |
| hybrid_after_entry_fill | 275 | -1675.7250 | 0.8508 | -0.3672 | -2251.8250 | -6.0935 | 0.2400 | 324 | 32.9927 |
| hybrid_next_execution_bar | 275 | -1567.2750 | 0.8604 | -0.3428 | -2143.3750 | -5.6992 | 0.2436 | 324 | 33.4036 |

## 5. Stop/Target and Entry Delay Grid
| config_id | stop_multiplier | target_multiplier | entry_delay_minutes | filter_family | net_pnl | profit_factor | trades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none_sm1p25_tm2p00_d30 | 1.2500 | 2.0000 | 30 | none | 2065.0500 | 1.4179 | 128 |
| none_sm1p25_tm2p50_d30 | 1.2500 | 2.5000 | 30 | none | 2045.0625 | 1.4138 | 128 |
| none_sm2p00_tm2p00_d30 | 2.0000 | 2.0000 | 30 | none | 2026.3000 | 1.3844 | 127 |
| none_sm2p00_tm2p50_d30 | 2.0000 | 2.5000 | 30 | none | 2006.3125 | 1.3806 | 127 |
| none_sm1p00_tm2p00_d30 | 1.0000 | 2.0000 | 30 | none | 1901.8000 | 1.3891 | 128 |
| none_sm1p25_tm3p00_d30 | 1.2500 | 3.0000 | 30 | none | 1900.5000 | 1.3846 | 128 |
| none_sm1p00_tm2p50_d30 | 1.0000 | 2.5000 | 30 | none | 1881.8125 | 1.3850 | 128 |
| none_sm2p00_tm3p00_d30 | 2.0000 | 3.0000 | 30 | none | 1861.7500 | 1.3532 | 127 |
| none_sm1p25_tm1p50_d30 | 1.2500 | 1.5000 | 30 | none | 1771.1125 | 1.3584 | 128 |
| none_sm1p50_tm2p00_d30 | 1.5000 | 2.0000 | 30 | none | 1763.8000 | 1.3332 | 127 |

## 6. Filter Families
| filter_family | configs | median_net_pnl | median_profit_factor |
| --- | --- | --- | --- |
| avoid_immediate_adverse_move | 96 | -101.1562 | 0.9343 |
| avoid_high_noise_first_5min | 72 | -141.9688 | 0.9352 |
| require_micro_momentum_confirmation | 48 | -180.0625 | 0.9549 |
| none | 150 | -202.0813 | 0.9693 |
| require_no_stop_zone_touch_before_entry | 54 | -256.7437 | 0.8701 |

## 7. IS Robustness Ranking
| rank_is | config_id | filter_family | stop_multiplier | target_multiplier | entry_delay_minutes | robust_score_is | net_pnl | profit_factor | trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | none_sm1p00_tm3p00_d30 | none | 1.0000 | 3.0000 | 30 | 0.9180 | 1737.2500 | 1.3554 | 128 |
| 2 | none_sm1p25_tm3p00_d30 | none | 1.2500 | 3.0000 | 30 | 0.9180 | 1900.5000 | 1.3846 | 128 |
| 3 | none_sm1p00_tm2p00_d30 | none | 1.0000 | 2.0000 | 30 | 0.9129 | 1901.8000 | 1.3891 | 128 |
| 4 | none_sm1p25_tm2p00_d30 | none | 1.2500 | 2.0000 | 30 | 0.9127 | 2065.0500 | 1.4179 | 128 |
| 5 | require_no_stop_zone_touch_before_entry_sm0p75_tm2p50_d5_stop_zone_1p00 | require_no_stop_zone_touch_before_entry | 0.7500 | 2.5000 | 5 | 0.8377 | 1646.6250 | 1.4230 | 110 |
| 6 | require_no_stop_zone_touch_before_entry_sm0p75_tm2p00_d5_stop_zone_1p00 | require_no_stop_zone_touch_before_entry | 0.7500 | 2.0000 | 5 | 0.7791 | 1375.7500 | 1.3534 | 110 |
| 7 | require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75 | require_no_stop_zone_touch_before_entry | 1.0000 | 2.5000 | 5 | 0.7043 | 950.3750 | 1.2035 | 110 |
| 8 | require_micro_momentum_confirmation_sm0p75_tm2p50_d5_momentum_w5_close_vs_window_open | require_micro_momentum_confirmation | 0.7500 | 2.5000 | 5 | 0.6929 | 766.4375 | 1.2016 | 98 |
| 9 | require_no_stop_zone_touch_before_entry_sm0p75_tm1p50_d5_stop_zone_1p00 | require_no_stop_zone_touch_before_entry | 0.7500 | 1.5000 | 5 | 0.6734 | 784.1000 | 1.2014 | 110 |
| 10 | require_micro_momentum_confirmation_sm0p75_tm2p00_d5_momentum_w5_close_vs_window_open | require_micro_momentum_confirmation | 0.7500 | 2.0000 | 5 | 0.6660 | 733.9000 | 1.1930 | 98 |

## 8. OOS Results of IS-Selected Configs
| rank_is | config_id | net_pnl | profit_factor | trades | net_pnl_oos | profit_factor_oos | trades_oos | degradation_ratio | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | none_sm1p00_tm3p00_d30 | 1737.2500 | 1.3554 | 128 | -183.5000 | 0.9509 | 65 | -0.1056 | overfit_suspect |
| 2 | none_sm1p25_tm3p00_d30 | 1900.5000 | 1.3846 | 128 | -416.3750 | 0.8952 | 65 | -0.2191 | overfit_suspect |
| 3 | none_sm1p00_tm2p00_d30 | 1901.8000 | 1.3891 | 128 | -183.5000 | 0.9509 | 65 | -0.0965 | overfit_suspect |
| 4 | none_sm1p25_tm2p00_d30 | 2065.0500 | 1.4179 | 128 | -416.3750 | 0.8952 | 65 | -0.2016 | overfit_suspect |
| 5 | require_no_stop_zone_touch_before_entry_sm0p75_tm2p50_d5_stop_zone_1p00 | 1646.6250 | 1.4230 | 110 | -183.1250 | 0.9335 | 50 | -0.1112 | overfit_suspect |
| 6 | require_no_stop_zone_touch_before_entry_sm0p75_tm2p00_d5_stop_zone_1p00 | 1375.7500 | 1.3534 | 110 | -235.1750 | 0.9147 | 50 | -0.1709 | overfit_suspect |
| 7 | require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75 | 950.3750 | 1.2035 | 110 | 1028.5000 | 1.4056 | 50 | 1.0822 | robust_candidate |
| 8 | require_micro_momentum_confirmation_sm0p75_tm2p50_d5_momentum_w5_close_vs_window_open | 766.4375 | 1.2016 | 98 | 192.7500 | 1.0754 | 48 | 0.2515 | weak_oos |
| 9 | require_no_stop_zone_touch_before_entry_sm0p75_tm1p50_d5_stop_zone_1p00 | 784.1000 | 1.2014 | 110 | -210.4750 | 0.9236 | 50 | -0.2684 | overfit_suspect |
| 10 | require_micro_momentum_confirmation_sm0p75_tm2p00_d5_momentum_w5_close_vs_window_open | 733.9000 | 1.1930 | 98 | 140.7000 | 1.0550 | 48 | 0.1917 | weak_oos |

## 9. Best Candidate Audit
| trades | net_pnl | gross_profit | gross_loss | winrate | avg_trade | median_trade | profit_factor | max_drawdown | sharpe_daily | sortino_daily | pnl_to_maxdd | avg_holding_minutes | median_holding_minutes | stop_exit_rate | target_exit_rate | time_stop_exit_rate | eod_exit_rate | skipped_trades | skip_rate | blocked_setups | turnover_proxy | estimated_cost_per_trade | config_id | symbol | execution_timeframe | entry_timing | protective_orders_active_from | ambiguous_policy | stop_multiplier | target_multiplier | entry_delay_minutes | filter_family | filter_label | filter_params | normalized_net_pnl_is | normalized_profit_factor_is | normalized_pnl_to_maxdd_is | trade_count_score_x | positive_years_is | years_available_is | max_year_contribution_pct | temporal_stability_score | parameter_neighborhood_score | trade_count_score_y | one_year_dependency | penalties | robust_score_is | admissible_is | rank_is |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 1737.2500 | 6624.7500 | 4887.5000 | 0.3906 | 13.5723 | -17.5000 | 1.3554 | -558.5000 | 1.7929 | 4.4188 | 3.1106 | 76.3359 | 119.5000 | 0.9609 | 0.0078 | 0.5000 | 0.0312 | 0 | 0.0000 | 2 | 27.6967 | 2.5000 | none_sm1p00_tm3p00_d30 | MNQ | 1min | next_execution_bar_open | next_execution_bar | stop_first | 1.0000 | 3.0000 | 30 | none | none | {} | 0.9318 | 0.9294 | 0.9420 | 1.0000 | 4 | 4 | 0.5618 | 0.7191 | 1.0000 | 1.0000 | False | 0.0000 | 0.9180 | True | 1 |

## 10. Robustness and Failure Modes
- Main failure modes remain too few trades after filtering, sensitivity to stop/target geometry, and OOS drawdown control.
- Some gains come from delaying entry rather than from complex filters, which is preferable from a robustness standpoint.

## 11. Verdict
- `Reject: alpha does not survive realistic execution.`

## 12. Next Actions
- Extend the same intrabar-aware protocol to MES, M2K and MGC.
- Integrate the surviving candidate family into the multi-asset research layer only after cross-asset validation.
- Add prop-firm style constraints and daily loss overlays on top of the best candidate.
- Run a walk-forward selection version of the same campaign.
- Add a CI guardrail that blocks any intraday alpha publication without intrabar validation.
