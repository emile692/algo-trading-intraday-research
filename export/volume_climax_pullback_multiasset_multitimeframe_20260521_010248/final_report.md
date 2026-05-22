# Volume Climax Pullback Multi-Asset Multi-Timeframe Campaign

## 1. Executive Summary
- Pullback survival beyond MNQ is mixed but not dead under realistic 1min execution.
- Best asset/timeframe in walk-forward: `MGC 1H`.
- Robust OOS support remains limited: `5` selected configs passed OOS filters.
- Global verdict: `Candidate as diversified portfolio only.`

## 2. Context: MNQ 1H Was Rejected
- The 1H hourly baseline was invalidated once intrabar stop/target sequencing was enforced on the 1min path.
- MNQ hybrid 1H/1min stayed negative and its standalone walk-forward remained reject.

## 3. Research Design
- Assets: MNQ, MES, M2K, MGC.
- Signal timeframes: 15min, 30min, 1H.
- Execution timeframe: 1min only, with strict intrabar-aware exits.
- Selection uses IS/train only; OOS and walk-forward are reporting and validation only.

## 4. Data Audit
| symbol | signal_timeframe | number_of_1min_rows | number_of_signal_rows | rth_rows | first_timestamp | last_timestamp | split_mode | variant_name |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MES | 15min | 2399361 | 46346 | 671391 | 2019-05-05 18:00:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5 |
| MNQ | 1H | 2401697 | 12035 | 671037 | 2019-05-05 18:03:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2 |
| MNQ | 30min | 2401697 | 24023 | 671037 | 2019-05-05 18:03:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2 |
| MES | 30min | 2399361 | 24023 | 671391 | 2019-05-05 18:00:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5 |
| MES | 1H | 2399361 | 12035 | 671391 | 2019-05-05 18:00:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5 |
| MNQ | 15min | 2401697 | 46346 | 671037 | 2019-05-05 18:03:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2 |
| M2K | 1H | 2232652 | 12035 | 667935 | 2019-05-05 18:01:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2 |
| M2K | 30min | 2232652 | 24021 | 667935 | 2019-05-05 18:01:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2 |
| MGC | 15min | 1648698 | 55733 | 523029 | 2010-10-03 19:26:00-04:00 | 2026-03-20 16:59:00-04:00 | fixed_calendar | regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2 |
| M2K | 15min | 2232652 | 46342 | 667935 | 2019-05-05 18:01:00-04:00 | 2026-03-20 09:29:00-04:00 | fixed_calendar | dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2 |
| MGC | 1H | 1648698 | 16737 | 523029 | 2010-10-03 19:26:00-04:00 | 2026-03-20 16:59:00-04:00 | fixed_calendar | regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2 |
| MGC | 30min | 1648698 | 30559 | 523029 | 2010-10-03 19:26:00-04:00 | 2026-03-20 16:59:00-04:00 | fixed_calendar | regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2 |

## 5. Config Universe
| family | configs |
| --- | --- |
| delay_adverse_filter | 1620 |
| delay_only | 36 |
| delay_stop_target | 576 |
| delay_stop_zone_filter | 972 |
| raw_hybrid | 12 |
| stop_target_recalibration | 192 |

## 6. IS/OOS Results by Asset and Timeframe
| symbol | signal_timeframe | rank_is | config_id | family | net_pnl | profit_factor | net_pnl_oos | profit_factor_oos | oos_pass | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MES | 15min | 1 | mes_15min_none_sm1p00_tm1p00_d0 | raw_hybrid | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MES | 15min | 2 | mes_15min_none_sm1p00_tm1p00_d5 | delay_only | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MES | 15min | 3 | mes_15min_none_sm1p00_tm1p00_d15 | delay_only | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MNQ | 1H | 1 | mnq_1h_none_sm1p00_tm2p50_d30 | delay_stop_target | 1881.8125 | 1.3850 | -183.5000 | 0.9509 | False | reject |
| MNQ | 1H | 2 | mnq_1h_require_no_stop_zone_touch_before_entry_sm0p75_tm2p50_d5_sz1p00 | delay_stop_zone_filter | 1646.6250 | 1.4230 | -183.1250 | 0.9335 | False | reject |
| MNQ | 1H | 3 | mnq_1h_none_sm1p00_tm1p00_d30 | delay_only | 1032.9250 | 1.2113 | -206.3250 | 0.9447 | False | reject |
| MNQ | 30min | 1 | mnq_30min_avoid_immediate_adverse_move_sm0p75_tm2p00_d5_aw5_mt16 | delay_adverse_filter | 376.0250 | 1.2260 | 190.2750 | 1.2033 | True | strong_candidate |
| MES | 30min | 1 | mes_30min_none_sm1p00_tm1p00_d0 | raw_hybrid | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MES | 30min | 2 | mes_30min_none_sm1p00_tm1p00_d5 | delay_only | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MES | 30min | 3 | mes_30min_none_sm1p00_tm1p00_d15 | delay_only | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MES | 1H | 1 | mes_1h_none_sm1p00_tm1p00_d0 | raw_hybrid | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MES | 1H | 2 | mes_1h_none_sm1p00_tm1p00_d5 | delay_only | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MES | 1H | 3 | mes_1h_none_sm1p00_tm1p00_d15 | delay_only | 0.0000 | inf | 0.0000 | inf | False | weak_watchlist |
| MNQ | 15min | 1 | mnq_15min_avoid_immediate_adverse_move_sm1p25_tm3p00_d5_aw5_mt16 | delay_adverse_filter | 374.4000 | 1.0849 | -658.1250 | 0.7788 | False | reject |
| M2K | 1H | 1 | m2k_1h_avoid_immediate_adverse_move_sm0p75_tm2p00_d15_aw5_mt8 | delay_adverse_filter | 176.2500 | 1.3013 | 361.4000 | 2.9275 | True | strong_candidate |
| M2K | 30min | 1 | m2k_30min_require_no_stop_zone_touch_before_entry_sm1p25_tm2p00_d30_sz0p50 | delay_stop_zone_filter | 211.0500 | 1.4450 | 175.2000 | 2.1898 | False | weak_watchlist |
| M2K | 30min | 2 | m2k_30min_require_no_stop_zone_touch_before_entry_sm0p75_tm2p00_d15_sz0p75 | delay_stop_zone_filter | 235.2750 | 1.3139 | -96.3250 | 0.7816 | False | reject |
| M2K | 30min | 3 | m2k_30min_require_no_stop_zone_touch_before_entry_sm1p00_tm2p00_d15_sz0p50 | delay_stop_zone_filter | 314.1000 | 1.5690 | -289.0000 | 0.2093 | False | reject |
| MGC | 15min | 1 | mgc_15min_avoid_immediate_adverse_move_sm0p75_tm3p00_d30_aw5_mt8 | delay_adverse_filter | 109.0000 | 1.0860 | -78.0000 | 0.7500 | False | reject |
| M2K | 15min | 1 | m2k_15min_avoid_immediate_adverse_move_sm1p25_tm3p00_d5_aw5_mt8 | delay_adverse_filter | 673.2250 | 1.2923 | -567.0750 | 0.5515 | False | reject |
| M2K | 15min | 2 | m2k_15min_require_no_stop_zone_touch_before_entry_sm1p00_tm2p00_d5_sz1p00 | delay_stop_zone_filter | 437.3500 | 1.0888 | -452.3500 | 0.8503 | False | reject |
| MGC | 1H | 1 | mgc_1h_require_no_stop_zone_touch_before_entry_sm1p25_tm3p00_d15_sz0p50 | delay_stop_zone_filter | 436.6000 | 1.5379 | 1161.7500 | 10.3879 | True | strong_candidate |
| MGC | 1H | 2 | mgc_1h_none_sm1p50_tm3p00_d15 | delay_stop_target | 419.3500 | 1.2907 | 988.5000 | 3.4051 | True | strong_candidate |
| MGC | 1H | 3 | mgc_1h_avoid_immediate_adverse_move_sm1p25_tm3p00_d30_aw5_mt12 | delay_adverse_filter | 339.2500 | 1.4887 | 1.7500 | 1.0091 | False | weak_watchlist |
| MGC | 30min | 1 | mgc_30min_avoid_immediate_adverse_move_sm0p75_tm3p00_d15_aw5_mt8 | delay_adverse_filter | 643.7500 | 1.6755 | -64.5000 | 0.5114 | False | reject |
| MGC | 30min | 2 | mgc_30min_require_no_stop_zone_touch_before_entry_sm0p75_tm3p00_d5_sz0p50 | delay_stop_zone_filter | 313.8500 | 1.3402 | 436.0000 | 15.2951 | False | weak_watchlist |
| MGC | 30min | 3 | mgc_30min_none_sm0p75_tm3p00_d15 | delay_stop_target | 283.2500 | 1.1516 | 187.7500 | 1.3716 | True | strong_candidate |

## 7. Walk-Forward Results
| symbol | signal_timeframe | total_test_net_pnl | test_profit_factor | positive_folds | pass_rate | improvement_vs_raw_hybrid | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MES | 15min | 0.0000 | inf | 0 | 0.0000 | 0.0000 | reject |
| MNQ | 1H | -886.1500 | 0.8653 | 1 | 0.2000 | -524.1250 | reject |
| MNQ | 30min | -482.5250 | 0.7503 | 1 | 0.0000 | 1087.2000 | reject |
| MES | 30min | 0.0000 | inf | 0 | 0.0000 | 0.0000 | reject |
| MES | 1H | 0.0000 | inf | 0 | 0.0000 | 0.0000 | reject |
| MNQ | 15min | -2148.1750 | 0.6298 | 1 | 0.2000 | -2203.4000 | reject |
| M2K | 1H | 314.2000 | 1.3335 | 4 | 0.2000 | 166.2500 | weak_watchlist |
| M2K | 30min | -345.1750 | 0.6916 | 2 | 0.0000 | 1269.7500 | reject |
| MGC | 15min | -318.0000 | 0.5225 | 2 | 0.0000 | 377.1000 | reject |
| M2K | 15min | -825.9750 | 0.6646 | 2 | 0.0000 | 1487.1750 | reject |
| MGC | 1H | 748.0000 | 3.8387 | 2 | 0.0000 | -114.6000 | reject |
| MGC | 30min | 45.1500 | 1.0884 | 2 | 0.0000 | 609.7500 | reject |

## 8. Timeframe Analysis
| signal_timeframe | total_configs | median_is_pf | median_oos_pf | pct_configs_is_positive | pct_configs_oos_positive | pct_selected_configs_oos_pass | wfa_stitched_median_pnl | best_symbol | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15min | 1136 | 0.9186 | 0.8350 | 0.1224 | 0.1232 | 0.0000 | -571.9875 | MES | reject |
| 1H | 1136 | 1.0380 | 1.8954 | 0.3019 | 0.5467 | 0.3000 | 157.1000 | MGC | weak_watchlist |
| 30min | 1136 | 0.9148 | 1.0067 | 0.1831 | 0.2526 | 0.2000 | -172.5875 | MGC | reject |

## 9. Asset Analysis
| symbol | best_timeframe_is_only | best_timeframe_wfa | selected_configs_count | oos_pass_count | wfa_verdict | net_pnl_stitched | pf_stitched | positive_folds | main_failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | 15min | 1H | 6 | 1 | weak_watchlist | 314.2000 | 1.3335 | 4 | unstable_params |
| MES | 15min | 15min | 9 | 0 | reject | 0.0000 | inf | 0 | negative_raw_edge |
| MGC | 30min | 1H | 7 | 3 | reject | 748.0000 | 3.8387 | 2 | unstable_params |
| MNQ | 1H | 30min | 5 | 1 | reject | -482.5250 | 0.7503 | 1 | negative_raw_edge |

## 10. Family Analysis
| family | selected | oos_pass_rate | median_oos_pnl |
| --- | --- | --- | --- |
| delay_stop_target | 3 | 0.6667 | 187.7500 |
| delay_only | 7 | 0.0000 | 0.0000 |
| raw_hybrid | 3 | 0.0000 | 0.0000 |
| delay_adverse_filter | 7 | 0.2857 | -64.5000 |
| delay_stop_zone_filter | 7 | 0.1429 | -96.3250 |

## 11. Portfolio Test
| portfolio_name | selection_basis | diagnostic_post_oos_filter | net_pnl | annualized_pnl | volatility_daily_pnl | sharpe_like_daily_pnl | max_drawdown | pnl_to_maxdd | positive_days_pct | worst_day | worst_month | asset_contribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight_all_candidates | train_only_preselected | False | 40.2389 | 37.9783 | 14.6329 | 0.1635 | -274.0861 | 0.1468 | 0.3745 | -53.3750 | -72.2306 | {"M2K": -48.230555555541564, "MGC": 146.2916666666673, "MNQ": -57.82222222222278} |
| inverse_vol_daily_pnl | train_only_preselected | False | 37.4209 | 35.3186 | 9.6990 | 0.2294 | -184.4620 | 0.2029 | 0.3745 | -22.8060 | -39.3682 | {"M2K": -58.83421219973156, "MGC": 126.95791716255488, "MNQ": -30.70280007402996} |
| capped_equal_weight | train_only_preselected | False | 7.7756 | 7.3388 | 15.9939 | 0.0289 | -325.9046 | 0.0239 | 0.3708 | -64.0500 | -89.9198 | {"M2K": -48.230555555541564, "MGC": 125.39285714285771, "MNQ": -69.38666666666734} |
| conservative_watchlist_book | diagnostic_post_oos_filter | True | 206.0368 | 674.3021 | 13.3147 | 3.1902 | -32.5515 | 6.3296 | 0.4545 | -13.0294 | -22.0294 | {"M2K": 31.564705882353792, "MGC": 163.27941176470577, "MNQ": 11.192647058823617} |

## 12. Best Candidate Audit
| symbol | signal_timeframe | total_test_trades | total_test_net_pnl | test_profit_factor | test_winrate | avg_trade | max_drawdown | pnl_to_maxdd | number_of_folds | positive_folds | pass_rate | selected_family_counts | train_score_test_corr | benchmark_raw_hybrid_net_pnl_same_windows | improvement_vs_raw_hybrid | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGC | 1H | 16 | 748.0000 | 3.8387 | 0.4375 | 46.7500 | -130.2500 | 5.7428 | 5 | 2 | 0.0000 | {"delay_stop_target": 1, "delay_stop_zone_filter": 3, "stop_target_recalibration": 1} | 0.4798 | 862.6000 | -114.6000 | reject |

## 13. Failure Modes
- The dominant failure modes remain poor OOS transfer, unstable parameter rankings, intrabar stop-before-target damage, and sparse trade counts on some sleeves.

## 14. Verdict
- `Candidate as diversified portfolio only.`

## 15. Next Actions
- Archive pullback standalone if the verdict stays reject across assets and timeframes.
- Keep surviving sleeves only as watchlist overlays or feature candidates inside broader intraday books.
- Move more effort toward ORB and execution-ready strategies if no cross-asset candidate emerges.
- If one sleeve survives, stress it with prop-firm constraints, slippage shocks, and live shadow monitoring.
- Add CI guardrails requiring realistic execution and walk-forward validation before publishing intraday alpha claims.
