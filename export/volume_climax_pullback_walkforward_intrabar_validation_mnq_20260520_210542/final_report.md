# Volume Climax Pullback Walk-Forward Intrabar Validation

## 1. Executive Summary
- The delay + stop-zone family does not show a reliable walk-forward edge under strict train-only selection.
- Positive folds: `2` / `5`.
- Stitched OOS net PnL: `-785.53 USD`.
- Improvement vs current hybrid over the same stitched test windows: `-423.50 USD`.
- Verdict: `Reject: no reliable walk-forward edge.`.

## 2. Context
- The 1H baseline remains invalidated and is never used for selection.
- Current hybrid benchmark from phase 1 remained negative at `-1567.28 USD` when executed realistically.
- Phase 2 best IS-only config failed global OOS, while one delay + stop-zone config looked informative but non-authoritative.

## 3. Walk-Forward Design
- Signal timeframe stays 1H, execution timeframe stays 1min.
- Each fold selects on train only using a deterministic robustness score, then applies the winner to the strictly future test window.
| fold_id | train_start | train_end | test_start | test_end | train_days | test_days |
| --- | --- | --- | --- | --- | --- | --- |
| fold_1 | 2020-01-01 | 2021-12-31 | 2022-01-01 | 2022-12-31 | 731 | 365 |
| fold_2 | 2020-01-01 | 2022-12-31 | 2023-01-01 | 2023-12-31 | 1096 | 365 |
| fold_3 | 2020-01-01 | 2023-12-31 | 2024-01-01 | 2024-12-31 | 1461 | 366 |
| fold_4 | 2020-01-01 | 2024-12-31 | 2025-01-01 | 2025-12-31 | 1827 | 365 |
| fold_5 | 2020-01-01 | 2025-12-31 | 2026-01-01 | 2026-03-19 | 2192 | 78 |

## 4. Config Universe
| family | configs |
| --- | --- |
| benchmark_current_hybrid | 1 |
| delay_only | 36 |
| delay_stop_zone | 104 |
| sanity_anti_overfit | 4 |

## 5. Fold-by-Fold Selection
| fold_id | selected_config_id | selected_family | train_robust_score | train_net_pnl | train_profit_factor | test_net_pnl | test_profit_factor | test_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_1 | none_sm1p25_tm2p00_d30 | delay_only | 0.7515 | 1706.3000 | 1.9839 | 532.3750 | 1.3741 | True |
| fold_2 | none_sm1p25_tm2p00_d30 | delay_only | 0.9569 | 2238.6750 | 1.7090 | -112.0250 | 0.9174 | False |
| fold_3 | none_sm1p00_tm2p50_d30 | delay_only | 0.9313 | 1892.3750 | 1.4219 | -22.5000 | 0.9839 | False |
| fold_4 | require_no_stop_zone_touch_before_entry_sm0p75_tm2p50_d5_stop_zone_1p00 | delay_stop_zone | 0.9028 | 2036.4375 | 1.4478 | -1258.3750 | 0.2658 | False |
| fold_5 | none_sm1p00_tm2p50_d30 | delay_only | 0.9224 | 1633.8750 | 1.2142 | 75.0000 | 1.1269 | False |

## 6. Stitched OOS Performance
| total_test_trades | total_test_net_pnl | test_profit_factor | test_winrate | avg_trade | max_drawdown | pnl_to_maxdd | number_of_folds | positive_folds | negative_folds | pass_rate | selected_config_diversity | selected_family_counts | benchmark_current_hybrid_net_pnl_over_same_test_windows | improvement_vs_current_hybrid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 116 | -785.5250 | 0.8788 | 0.3534 | -6.7718 | -2299.4000 | -0.3416 | 5 | 2 | 3 | 0.2000 | 3 | {"delay_only": 4, "delay_stop_zone": 1} | -362.0250 | -423.5000 |

## 7. Fixed Candidate Tracking
| fold_id | config_id | test_trades | test_net_pnl | test_profit_factor | test_winrate | test_avg_trade | test_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold_1 | require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75 | 21 | 78.1250 | 1.0539 | 0.3333 | 3.7202 | -490.0000 |
| fold_2 | require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75 | 18 | 358.3125 | 1.5992 | 0.2222 | 19.9062 | -343.5000 |
| fold_3 | require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75 | 22 | 665.0000 | 1.6952 | 0.4545 | 30.2273 | -441.0000 |
| fold_4 | require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75 | 25 | -206.5000 | 0.8659 | 0.3600 | -8.2600 | -815.5000 |
| fold_5 | require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d5_stop_zone_0p75 | 3 | 570.0000 | 15.4304 | 0.6667 | 190.0000 | -39.5000 |

## 8. Family-Level Diagnostics
| family | avg_rank_train | times_selected | stitched_test_net_pnl_if_selected | fixed_oracle_warning | median_test_pnl_across_configs | pct_configs_positive_test | pct_configs_pf_above_1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| delay_only | 39.3556 | 4 | 472.8500 | descriptive_only_test_stats | -18.4375 | 0.4722 | 0.4722 |
| delay_stop_zone | 84.1923 | 1 | -1258.3750 | descriptive_only_test_stats | -66.3750 | 0.3442 | 0.4077 |
| sanity_anti_overfit | 70.1000 | 0 | 0.0000 | descriptive_only_test_stats | 145.5000 | 0.6500 | 0.6500 |
| benchmark_current_hybrid | 131.8000 | 0 | 0.0000 | descriptive_only_test_stats | -148.5250 | 0.4000 | 0.4000 |

## 9. Train Score vs Test Reality
- Correlation proxy between train score and test PnL is `-0.4323` across selected folds.

## 10. Failure Modes
- Typical failure modes remain fold concentration, low trade density in some windows, and parameter instability between adjacent delay / stop-zone variants.
- If train winners repeatedly fail future tests, the phase 2 positive pocket should be treated as cherry-pick risk rather than surviving alpha.

## 11. Verdict
- `Reject: no reliable walk-forward edge.`

## 12. Next Actions
- Stop treating MNQ pullback standalone as a deployable alpha if the verdict remains reject.
- Keep the signal only as a candidate feature or overlay inside broader intraday frameworks.
- Focus forward effort on ORB / TopstepX-ready paths before spending more on standalone MNQ pullback.
- If the verdict improves later, validate the same family on MES, M2K and MGC with the same walk-forward discipline.
- Add a CI guardrail that blocks intraday research publication without walk-forward intrabar validation.
