# Volume Climax Pullback Survivor Audit

## 1. Executive Summary
- M2K 1H strict survivor status: `weak_watchlist`.
- MGC 1H instability diagnosis: `reject` with `2` positive folds.
- Strict train-only portfolio status: `m2k_only -> weak_watchlist`.
- Deployable conclusion: `watchlist_only_or_reject`.

## 2. Strict Train-Only WFA Summary
| symbol | signal_timeframe | total_test_trades | total_test_net_pnl | gross_profit | gross_loss | test_profit_factor | test_win_rate | avg_trade | median_trade | max_drawdown | max_daily_drawdown | positive_folds | fold_count | fold_sharpe | active_days | exposure_ratio | avg_holding_minutes | long_trades | short_trades | long_pnl | short_pnl | monthly_positive_ratio | cluster_positive_ratio | selected_family_counts | selected_cluster_counts | train_score_test_corr | benchmark_raw_hybrid_net_pnl_same_windows | improvement_vs_raw_hybrid | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | 1H | 39 | 474.7750 | 994.5250 | 519.7500 | 1.9135 | 0.3846 | 12.1737 | -11.6250 | -133.1250 | -52.5000 | 4 | 5 | 1.2931 | 39 | 0.1940 | 74.3590 | 25 | 14 | 380.7500 | 94.0250 | 0.4400 | 1.0000 | {"delay_adverse_filter": 4, "delay_stop_zone_filter": 1} | {"m2k_1h_adverse_core": 4, "m2k_1h_stop_zone_diag": 1} | 0.6845 | 47.0750 | 427.7000 | weak_watchlist |
| MGC | 1H | 23 | 1217.2000 | 1706.2000 | 489.0000 | 3.4892 | 0.4783 | 52.9217 | -11.5000 | -265.0000 | -81.5000 | 2 | 5 | 1.8880 | 23 | 0.6571 | 118.6522 | 19 | 4 | 1032.7000 | 184.5000 | 0.6250 | 1.0000 | {"delay_stop_zone_filter": 2, "none_baseline": 2, "raw_hybrid": 1} | {"mgc_1h_none_core": 2, "mgc_1h_raw_baseline": 1, "mgc_1h_stop_zone_core": 2} | 0.7000 | 671.1000 | 546.1000 | reject |
| MNQ | 1H | 116 | -878.1250 | 5616.2500 | 6494.3750 | 0.8648 | 0.3276 | -7.5700 | -38.0000 | -2260.7750 | -269.5000 | 2 | 5 | -0.5846 | 114 | 0.5787 | 76.7500 | 72 | 44 | -579.8750 | -298.2500 | 0.3673 | 1.0000 | {"delay_stop_zone_filter": 1, "none_baseline": 4} | {"mnq_1h_negative_control_none": 4, "mnq_1h_negative_control_stop_zone": 1} | -0.0270 | -362.0250 | -516.1000 | reject |

## 3. Fold Breakdown
| symbol | signal_timeframe | fold_id | selected_config_id | selected_family | selected_cluster_id | train_robust_score | train_net_pnl | train_profit_factor | train_trades | test_net_pnl | test_profit_factor | test_trades | test_win_rate | test_avg_trade | test_median_trade | test_max_drawdown | test_max_daily_drawdown | test_fold_sharpe | test_active_days | test_exposure_ratio | test_avg_holding_minutes | raw_hybrid_test_net_pnl | strict_train_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | 1H | fold_1 | m2k_1h_delay_stop_zone_filter_require_no_stop_zone_touch_before_entry_sm0p75_tm2p00_d15_ts4_sz0p50 | delay_stop_zone_filter | m2k_1h_stop_zone_diag | 0.2746 | 32.0000 | 1.2092 | 10 | -42.3750 | 0.0000 | 2 | 0.0000 | -21.1875 | -21.1875 | -30.0000 | -30.0000 | -3.4001 | 2 | 0.0426 | 16.5000 | -45.8000 | True |
| M2K | 1H | fold_2 | m2k_1h_delay_adverse_filter_avoid_immediate_adverse_move_sm0p75_tm2p50_d5_ts3_aw5_mt12 | delay_adverse_filter | m2k_1h_adverse_core | 0.4568 | 6.1250 | 1.0082 | 37 | 130.1250 | 1.6831 | 15 | 0.4667 | 8.6750 | -7.8750 | -119.6250 | -52.5000 | 0.7929 | 15 | 0.2885 | 77.7333 | -164.4000 | True |
| M2K | 1H | fold_3 | m2k_1h_delay_adverse_filter_avoid_immediate_adverse_move_sm0p75_tm2p00_d15_ts3_aw5_mt12 | delay_adverse_filter | m2k_1h_adverse_core | 0.8090 | 330.5000 | 1.4180 | 45 | 140.4000 | 1.9132 | 14 | 0.4286 | 10.0286 | -7.3125 | -88.5000 | -35.2500 | 0.7878 | 14 | 0.3182 | 90.7143 | 365.0250 | True |
| M2K | 1H | fold_4 | m2k_1h_delay_adverse_filter_avoid_immediate_adverse_move_sm0p75_tm2p00_d15_ts3_aw5_mt12 | delay_adverse_filter | m2k_1h_adverse_core | 0.8590 | 470.9000 | 1.4986 | 59 | 69.6250 | 1.8179 | 5 | 0.2000 | 13.9250 | -18.3750 | -85.1250 | -33.7500 | 0.4380 | 5 | 0.1163 | 32.6000 | 13.2000 | True |
| M2K | 1H | fold_5 | m2k_1h_delay_adverse_filter_avoid_immediate_adverse_move_sm0p75_tm2p00_d15_ts3_aw5_mt8 | delay_adverse_filter | m2k_1h_adverse_core | 0.8484 | 455.9000 | 1.8058 | 43 | 177.0000 | 4.6875 | 3 | 0.3333 | 59.0000 | -12.0000 | -12.0000 | -36.0000 | 0.8676 | 3 | 0.2000 | 89.3333 | -120.9500 | True |
| MGC | 1H | fold_1 | mgc_1h_delay_stop_zone_filter_require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d15_ts3_sz0p50 | delay_stop_zone_filter | mgc_1h_stop_zone_core | 0.4275 | 213.0000 | 10.6818 | 4 | 0.0000 | inf | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  | 0 | 0.0000 | 0.0000 | 39.6000 | True |
| MGC | 1H | fold_2 | mgc_1h_delay_stop_zone_filter_require_no_stop_zone_touch_before_entry_sm1p00_tm2p50_d15_ts3_sz0p50 | delay_stop_zone_filter | mgc_1h_stop_zone_core | 0.4008 | 213.0000 | 10.6818 | 4 | -48.5000 | 0.4948 | 3 | 0.3333 | -16.1667 | -22.5000 | -96.0000 | -73.5000 | -0.5645 | 3 | 0.5000 | 126.6667 | 88.1500 | True |
| MGC | 1H | fold_3 | mgc_1h_raw_hybrid_none_sm1p00_tm1p00_d0_ts4 | raw_hybrid | mgc_1h_raw_baseline | 0.7040 | 121.5500 | 1.2280 | 30 | -5.3000 | 0.9674 | 7 | 0.2857 | -0.7571 | -19.5000 | -130.5000 | -81.5000 | -0.0351 | 7 | 1.0000 | 56.0000 | -55.5000 | True |
| MGC | 1H | fold_4 | mgc_1h_none_baseline_none_sm1p00_tm2p50_d5_ts4 | none_baseline | mgc_1h_none_core | 0.6310 | 251.6250 | 1.4438 | 35 | 562.0000 | 3.4382 | 11 | 0.5455 | 51.0909 | 16.5000 | -96.0000 | -68.5000 | 1.5535 | 11 | 0.8462 | 134.2727 | 200.9500 | True |
| MGC | 1H | fold_5 | mgc_1h_none_baseline_none_sm1p00_tm2p50_d5_ts4 | none_baseline | mgc_1h_none_core | 0.7943 | 813.6250 | 2.0202 | 46 | 709.0000 | inf | 2 | 1.0000 | 354.5000 | 354.5000 | 0.0000 | 216.5000 | 3.6329 | 2 | 0.6667 | 240.0000 | 397.9000 | True |
| MNQ | 1H | fold_1 | mnq_1h_none_baseline_none_sm1p00_tm2p00_d30_ts2 | none_baseline | mnq_1h_negative_control_none | 0.7452 | 1621.0500 | 1.9231 | 52 | 321.5000 | 1.2172 | 29 | 0.3103 | 11.0862 | -31.5000 | -483.0000 | -191.0000 | 0.4217 | 28 | 0.6087 | 75.5172 | -570.1750 | True |
| MNQ | 1H | fold_2 | mnq_1h_none_baseline_none_sm1p00_tm2p00_d30_ts2 | none_baseline | mnq_1h_negative_control_none | 0.9445 | 1942.5500 | 1.6003 | 81 | -4.9000 | 0.9961 | 25 | 0.4000 | -0.1960 | -36.5000 | -359.0000 | -184.5000 | -0.0078 | 25 | 0.5556 | 66.2400 | -148.5250 | True |
| MNQ | 1H | fold_3 | mnq_1h_none_baseline_none_sm1p00_tm2p00_d30_ts3 | none_baseline | mnq_1h_negative_control_none | 0.9628 | 2006.7000 | 1.3786 | 106 | -251.0000 | 0.8441 | 31 | 0.3548 | -8.0968 | -26.5000 | -579.5000 | -186.0000 | -0.3634 | 31 | 0.5849 | 107.3226 | 507.7750 | True |
| MNQ | 1H | fold_4 | mnq_1h_delay_stop_zone_filter_require_no_stop_zone_touch_before_entry_sm0p75_tm2p50_d5_ts2_sz1p00 | delay_stop_zone_filter | mnq_1h_negative_control_stop_zone | 0.8630 | 2036.4375 | 1.4478 | 111 | -1258.3750 | 0.2658 | 25 | 0.2400 | -50.3350 | -52.8750 | -1211.6250 | -236.2500 | -2.7896 | 24 | 0.5455 | 46.4000 | -559.3250 | True |
| MNQ | 1H | fold_5 | mnq_1h_none_baseline_none_sm1p00_tm2p00_d30_ts3 | none_baseline | mnq_1h_negative_control_none | 0.9562 | 2314.2000 | 1.2670 | 165 | 314.6500 | 1.7127 | 6 | 0.3333 | 52.4417 | -46.2500 | -362.0000 | -269.5000 | 0.4360 | 6 | 0.6667 | 95.0000 | 408.2250 | True |

## 4. Cluster Stability
| symbol | cluster_id | family | configs | median_is_net_pnl | median_oos_net_pnl | median_fixed_wfa_net_pnl | pct_configs_positive_oos | pct_configs_positive_fixed_wfa | median_neighbor_oos_pnl | selected_in_any_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | m2k_1h_adverse_core | delay_adverse_filter | 64 | -80.3687 | 262.9500 | 388.0000 | 1.0000 | 1.0000 | 262.9500 | 3 |
| M2K | m2k_1h_none_local | none_baseline | 36 | -1002.7937 | 743.7188 | 410.0125 | 1.0000 | 0.9444 | 743.7188 | 0 |
| M2K | m2k_1h_raw_baseline | raw_hybrid | 2 | -1336.1875 | 286.4750 | 97.5125 | 1.0000 | 1.0000 | 286.4750 | 0 |
| M2K | m2k_1h_stop_zone_diag | delay_stop_zone_filter | 64 | -774.9750 | 557.8125 | 221.5313 | 0.8906 | 0.7812 | 557.8125 | 1 |
| MGC | mgc_1h_adverse_diag | delay_adverse_filter | 64 | -1.1250 | -61.7500 | -32.3750 | 0.0938 | 0.4062 | -61.7500 | 0 |
| MGC | mgc_1h_none_core | none_baseline | 54 | 215.3750 | 797.5000 | 733.5000 | 0.8889 | 0.8889 | 797.5000 | 1 |
| MGC | mgc_1h_raw_baseline | raw_hybrid | 3 | -316.3000 | 753.5500 | 862.6000 | 1.0000 | 1.0000 | 753.5500 | 1 |
| MGC | mgc_1h_stop_zone_core | delay_stop_zone_filter | 64 | -158.6500 | 329.6250 | 409.8750 | 0.8750 | 0.8750 | 329.6250 | 1 |
| MNQ | mnq_1h_negative_control_adverse | delay_adverse_filter | 32 | 73.3000 | 503.5750 | 108.5000 | 1.0000 | 0.8125 | 503.5750 | 0 |
| MNQ | mnq_1h_negative_control_none | none_baseline | 8 | 1287.1625 | 380.4312 | 606.6125 | 0.7500 | 1.0000 | 380.4312 | 2 |
| MNQ | mnq_1h_negative_control_raw | raw_hybrid | 2 | -1764.1250 | 756.9875 | 168.2875 | 1.0000 | 0.5000 | 756.9875 | 0 |
| MNQ | mnq_1h_negative_control_stop_zone | delay_stop_zone_filter | 32 | -459.9937 | 314.9437 | 138.1187 | 0.5625 | 0.5312 | 314.9437 | 1 |

## 5. Strict Portfolio
| portfolio_name | selection_basis | deployable | net_pnl | gross_profit | gross_loss | profit_factor | positive_folds | fold_count | fold_sharpe | max_drawdown | monthly_positive_ratio | weights_last_json | improvement_vs_m2k_only | cluster_positive_ratio | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m2k_mgc_capped_equal | strict_train_only | True | 845.9875 | 1350.3625 | 504.3750 | 2.6773 | 4 | 5 | 2.1181 | -107.8125 | 0.5312 | {"M2K": 0.5, "MGC": 0.5} | 371.2125 | 1.0000 | watchlist |
| m2k_mgc_equal_weight | strict_train_only | True | 845.9875 | 1350.3625 | 504.3750 | 2.6773 | 4 | 5 | 2.1181 | -107.8125 | 0.5312 | {"M2K": 0.5, "MGC": 0.5} | 371.2125 | 1.0000 | watchlist |
| m2k_mgc_inverse_vol | strict_train_only | True | 793.8981 | 1320.1664 | 526.2682 | 2.5085 | 4 | 5 | 2.1435 | -119.0455 | 0.5312 | {"M2K": 0.5972821822876963, "MGC": 0.40271781771230364} | 319.1231 | 1.0000 | watchlist |
| m2k_only | strict_train_only | True | 474.7750 | 994.5250 | 519.7500 | 1.9135 | 4 | 5 | 2.7626 | -133.1250 | 0.4400 | {"M2K": 1.0} | 0.0000 | 1.0000 | weak_watchlist |
| mgc_only | strict_train_only | True | 1217.2000 | 1706.2000 | 489.0000 | 3.4892 | 2 | 4 | 1.8135 | -265.0000 | 0.6250 | {"MGC": 1.0} | 742.4250 | 1.0000 | reject |

## 6. Negative Control
| symbol | signal_timeframe | total_test_trades | total_test_net_pnl | gross_profit | gross_loss | test_profit_factor | test_win_rate | avg_trade | median_trade | max_drawdown | max_daily_drawdown | positive_folds | fold_count | fold_sharpe | active_days | exposure_ratio | avg_holding_minutes | long_trades | short_trades | long_pnl | short_pnl | monthly_positive_ratio | cluster_positive_ratio | selected_family_counts | selected_cluster_counts | train_score_test_corr | benchmark_raw_hybrid_net_pnl_same_windows | improvement_vs_raw_hybrid | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNQ | 1H | 116 | -878.1250 | 5616.2500 | 6494.3750 | 0.8648 | 0.3276 | -7.5700 | -38.0000 | -2260.7750 | -269.5000 | 2 | 5 | -0.5846 | 114 | 0.5787 | 76.7500 | 72 | 44 | -579.8750 | -298.2500 | 0.3673 | 1.0000 | {"delay_stop_zone_filter": 1, "none_baseline": 4} | {"mnq_1h_negative_control_none": 4, "mnq_1h_negative_control_stop_zone": 1} | -0.0270 | -362.0250 | -516.1000 | reject |

## 7. Diagnostic Posthoc Results
| result_type | name | deployable | reason | net_pnl | profit_factor | positive_folds |
| --- | --- | --- | --- | --- | --- | --- |
| posthoc_positive_sleeve | M2K_1H | False | selected_after_full_walkforward_observation | 474.7750 | 1.9135 | 4 |
| posthoc_positive_sleeve | MGC_1H | False | selected_after_full_walkforward_observation | 1217.2000 | 3.4892 | 2 |

## 8. Verdict
- Final strict verdict: `watchlist`
- Diagnostic posthoc rows, if present, are explicitly non-deployable and excluded from the promotion decision.

## 9. Next Actions
- Keep M2K 1H as watchlist only unless a strict portfolio earns a candidate verdict.
- Treat MGC 1H as regime-fragile until fold dispersion improves materially.
- Keep MNQ 1H as a negative control rather than a promotion target.
- If no strict candidate emerges, stop standalone pullback promotion and keep only as overlay/feature research.
- Stress any future watchlist sleeve with slippage shocks and live shadow monitoring before escalation.
