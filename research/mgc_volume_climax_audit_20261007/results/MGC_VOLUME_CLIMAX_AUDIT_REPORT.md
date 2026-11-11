# MGC Volume Climax Reversal — Frozen-Region Audit

**Final decision: `REAL_EFFECT_NOT_YET_EXECUTABLE`**

No new parameter search was run, no best cell was selected, and the frozen nine sleeves were not modified. The canonical cell is the inherited geometric centre used only to make the requested diagnostics concrete.

## Executive conclusion

The confirmed-fade OOS effect is economically coherent and cost-robust, but it is not temporally stable enough to execute. All 18 inherited cells lose money IS; only 44.4% are positive OOS and 0.0% are positive in both. The centre cell changes from IS Sharpe -2.96 / PnL $-6,763 to OOS Sharpe 1.98 / PnL $3,165. The apparent `BROAD_REGION` is therefore an OOS confirmed-mode plateau, not a history-wide region.

The price-only benchmark is already profitable OOS. The volume-conditioned Sharpe sits at the 92.6th percentile of 1,000 trade-count-matched random thinnings, which is suggestive but does not clear a 95th-percentile falsification gate. Most positive PnL is recent: 2025–2026 contribute 87.3% of all positive-year PnL.

## 1. Inherited specification

| item | value |
| --- | --- |
| inherited_grid | move={1.0,1.5,2.0} ATR; RVOL={1.5,2.0,2.5}; mode={immediate_fade,confirmed_fade}; 18 cells |
| price_normalization | abs(close-open) / causal session ATR; ATR is median true range of prior 20 same-session bars, min 10 |
| volume_normalization | bar volume / median volume at same ET slot over prior 20 sessions |
| immediate_entry | climax completed at t; fade signal t; fill open t+1 |
| confirmed_entry | climax completed at t-1; opposite-direction bar t; fade signal t; fill open t+1 |
| exit | close of sixth held 5-minute bar, truncated at session end; non-overlapping trades |
| session | MGC weekdays 08:20 <= ET < 13:30; no overnight carry |
| nominal_cost | $4.50 round turn = one tick/side slippage ($2.00) + $1.25/side commission ($2.50) |
| validation_split | chronological 70/30 by session; OOS begins 2023-03-17 |
| canonical_diagnostic_cell | B__confirmed_fade__n1__m1.5__rv2.0__p0__c1; inherited geometric center, fixed without metric selection |

## 2. Every inherited cell

| variant_id | mode | move | rvol | split | trades | net_pnl | sharpe | pf | expectancy | win_rate | avg_winner | avg_loser | maxdd | avg_holding | turnover_trades_per_year | long_pnl | short_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B__immediate_fade__n1__m1.0__rv1.5__p0__c1 | immediate_fade | 1 | 1.5 | IS | 4797 | -2.221e+04 | -3.972 | 0.6159 | -4.63 | 0.4165 | 17.83 | -20.66 | -2.239e+04 | 5.651 | 591.1 | -1.107e+04 | -1.114e+04 |
| B__immediate_fade__n1__m1.0__rv1.5__p0__c1 | immediate_fade | 1 | 1.5 | OOS | 1611 | -4174 | -0.5886 | 0.9188 | -2.591 | 0.4792 | 61.19 | -61.28 | -8419 | 5.716 | 457.9 | -3588 | -586.5 |
| B__confirmed_fade__n1__m1.0__rv1.5__p0__c1 | confirmed_fade | 1 | 1.5 | IS | 3079 | -1.342e+04 | -2.982 | 0.6322 | -4.359 | 0.4232 | 17.71 | -20.55 | -1.403e+04 | 5.693 | 379.4 | -7213 | -6208 |
| B__confirmed_fade__n1__m1.0__rv1.5__p0__c1 | confirmed_fade | 1 | 1.5 | OOS | 999 | 1819 | 0.4067 | 1.066 | 1.821 | 0.4895 | 59.77 | -53.74 | -5141 | 5.683 | 284 | -796 | 2616 |
| B__immediate_fade__n1__m1.0__rv2.0__p0__c1 | immediate_fade | 1 | 2 | IS | 3857 | -1.676e+04 | -3.355 | 0.6414 | -4.346 | 0.426 | 18.25 | -21.11 | -1.746e+04 | 5.67 | 475.3 | -8216 | -8545 |
| B__immediate_fade__n1__m1.0__rv2.0__p0__c1 | immediate_fade | 1 | 2 | OOS | 1112 | -5307 | -1.021 | 0.862 | -4.772 | 0.4667 | 63.88 | -64.86 | -6673 | 5.695 | 316.1 | -3744 | -1563 |
| B__confirmed_fade__n1__m1.0__rv2.0__p0__c1 | confirmed_fade | 1 | 2 | IS | 2480 | -1.142e+04 | -2.84 | 0.6232 | -4.604 | 0.4278 | 17.8 | -21.35 | -1.196e+04 | 5.704 | 305.6 | -5661 | -5759 |
| B__confirmed_fade__n1__m1.0__rv2.0__p0__c1 | confirmed_fade | 1 | 2 | OOS | 666 | -17 | -0.005271 | 0.9991 | -0.02553 | 0.497 | 58.79 | -58.14 | -3739 | 5.656 | 189.3 | -1467 | 1449 |
| B__immediate_fade__n1__m1.0__rv2.5__p0__c1 | immediate_fade | 1 | 2.5 | IS | 3118 | -1.361e+04 | -3.103 | 0.6404 | -4.366 | 0.4304 | 18.06 | -21.31 | -1.412e+04 | 5.687 | 384.4 | -7338 | -6275 |
| B__immediate_fade__n1__m1.0__rv2.5__p0__c1 | immediate_fade | 1 | 2.5 | OOS | 814 | -807 | -0.1966 | 0.9702 | -0.9914 | 0.4803 | 67.28 | -64.1 | -4434 | 5.743 | 231.4 | 391.5 | -1198 |
| B__confirmed_fade__n1__m1.0__rv2.5__p0__c1 | confirmed_fade | 1 | 2.5 | IS | 1991 | -9000 | -2.604 | 0.6345 | -4.52 | 0.4314 | 18.19 | -21.75 | -9489 | 5.719 | 245.4 | -4625 | -4374 |
| B__confirmed_fade__n1__m1.0__rv2.5__p0__c1 | confirmed_fade | 1 | 2.5 | OOS | 498 | 615 | 0.252 | 1.043 | 1.235 | 0.5 | 60.17 | -57.7 | -2567 | 5.685 | 141.6 | 176.5 | 438.5 |
| B__immediate_fade__n1__m1.5__rv1.5__p0__c1 | immediate_fade | 1.5 | 1.5 | IS | 2459 | -1.212e+04 | -3.165 | 0.6118 | -4.928 | 0.4213 | 18.43 | -21.94 | -1.237e+04 | 5.629 | 303.1 | -6232 | -5886 |
| B__immediate_fade__n1__m1.5__rv1.5__p0__c1 | immediate_fade | 1.5 | 1.5 | OOS | 701 | -848.5 | -0.1913 | 0.9651 | -1.21 | 0.5193 | 64.51 | -72.19 | -3277 | 5.746 | 199.3 | -268 | -580.5 |
| B__confirmed_fade__n1__m1.5__rv1.5__p0__c1 | confirmed_fade | 1.5 | 1.5 | IS | 1528 | -7401 | -2.93 | 0.6031 | -4.844 | 0.4241 | 17.35 | -21.19 | -7630 | 5.664 | 188.4 | -4092 | -3309 |
| B__confirmed_fade__n1__m1.5__rv1.5__p0__c1 | confirmed_fade | 1.5 | 1.5 | OOS | 418 | 3923 | 2.033 | 1.442 | 9.385 | 0.5287 | 57.94 | -45.09 | -1098 | 5.72 | 118.8 | 2882 | 1041 |
| B__immediate_fade__n1__m1.5__rv2.0__p0__c1 | immediate_fade | 1.5 | 2 | IS | 2133 | -1.066e+04 | -3.13 | 0.6115 | -4.998 | 0.4257 | 18.48 | -22.4 | -1.113e+04 | 5.65 | 262.9 | -5562 | -5098 |
| B__immediate_fade__n1__m1.5__rv2.0__p0__c1 | immediate_fade | 1.5 | 2 | OOS | 541 | -2212 | -0.6111 | 0.8904 | -4.09 | 0.5009 | 66.3 | -74.74 | -3465 | 5.73 | 153.8 | -1453 | -759.5 |
| B__confirmed_fade__n1__m1.5__rv2.0__p0__c1 | confirmed_fade | 1.5 | 2 | IS | 1333 | -6763 | -2.962 | 0.5936 | -5.073 | 0.4269 | 17.36 | -21.78 | -7025 | 5.678 | 164.3 | -3716 | -3047 |
| B__confirmed_fade__n1__m1.5__rv2.0__p0__c1 | confirmed_fade | 1.5 | 2 | OOS | 318 | 3165 | 1.984 | 1.461 | 9.953 | 0.5283 | 59.68 | -45.75 | -841 | 5.679 | 90.39 | 1949 | 1215 |
| B__immediate_fade__n1__m1.5__rv2.5__p0__c1 | immediate_fade | 1.5 | 2.5 | IS | 1832 | -8942 | -2.973 | 0.6182 | -4.881 | 0.435 | 18.17 | -22.63 | -9322 | 5.666 | 225.8 | -5149 | -3793 |
| B__immediate_fade__n1__m1.5__rv2.5__p0__c1 | immediate_fade | 1.5 | 2.5 | OOS | 446 | -936 | -0.3558 | 0.9421 | -2.099 | 0.4978 | 68.55 | -72.11 | -3023 | 5.72 | 126.8 | -10 | -926 |
| B__confirmed_fade__n1__m1.5__rv2.5__p0__c1 | confirmed_fade | 1.5 | 2.5 | IS | 1142 | -6024 | -2.998 | 0.5847 | -5.275 | 0.4299 | 17.27 | -22.28 | -6235 | 5.705 | 140.8 | -3519 | -2505 |
| B__confirmed_fade__n1__m1.5__rv2.5__p0__c1 | confirmed_fade | 1.5 | 2.5 | OOS | 266 | 1892 | 1.438 | 1.31 | 7.113 | 0.5 | 60.05 | -45.82 | -1125 | 5.688 | 75.61 | 1688 | 204.5 |
| B__immediate_fade__n1__m2.0__rv1.5__p0__c1 | immediate_fade | 2 | 1.5 | IS | 1358 | -6904 | -2.714 | 0.6169 | -5.084 | 0.4352 | 18.81 | -23.49 | -7265 | 5.635 | 167.5 | -3986 | -2917 |
| B__immediate_fade__n1__m2.0__rv1.5__p0__c1 | immediate_fade | 2 | 1.5 | OOS | 334 | -737 | -0.3197 | 0.9408 | -2.207 | 0.5419 | 64.67 | -81.32 | -2148 | 5.763 | 95.31 | 735.5 | -1473 |
| B__confirmed_fade__n1__m2.0__rv1.5__p0__c1 | confirmed_fade | 2 | 1.5 | IS | 811 | -3935 | -2.629 | 0.6101 | -4.851 | 0.4377 | 17.35 | -22.13 | -4044 | 5.656 | 100 | -1772 | -2163 |
| B__confirmed_fade__n1__m2.0__rv1.5__p0__c1 | confirmed_fade | 2 | 1.5 | OOS | 197 | 1951 | 2.037 | 1.452 | 9.906 | 0.5178 | 61.5 | -45.49 | -567 | 5.69 | 56.48 | 1391 | 560 |
| B__immediate_fade__n1__m2.0__rv2.0__p0__c1 | immediate_fade | 2 | 2 | IS | 1230 | -5732 | -2.446 | 0.6445 | -4.66 | 0.4415 | 19.14 | -23.47 | -6105 | 5.656 | 151.7 | -3230 | -2502 |
| B__immediate_fade__n1__m2.0__rv2.0__p0__c1 | immediate_fade | 2 | 2 | OOS | 285 | -1527 | -0.7614 | 0.8647 | -5.36 | 0.5333 | 64.2 | -84.86 | -2423 | 5.74 | 81.33 | -467 | -1060 |
| B__confirmed_fade__n1__m2.0__rv2.0__p0__c1 | confirmed_fade | 2 | 2 | IS | 736 | -3668 | -2.653 | 0.6032 | -4.984 | 0.4402 | 17.21 | -22.43 | -3777 | 5.659 | 90.76 | -1626 | -2042 |
| B__confirmed_fade__n1__m2.0__rv2.0__p0__c1 | confirmed_fade | 2 | 2 | OOS | 171 | 1583 | 1.905 | 1.417 | 9.26 | 0.5322 | 59.14 | -47.48 | -554.5 | 5.643 | 49.02 | 1034 | 549.5 |
| B__immediate_fade__n1__m2.0__rv2.5__p0__c1 | immediate_fade | 2 | 2.5 | IS | 1106 | -5069 | -2.381 | 0.6459 | -4.583 | 0.4494 | 18.6 | -23.51 | -5339 | 5.665 | 136.4 | -3046 | -2023 |
| B__immediate_fade__n1__m2.0__rv2.5__p0__c1 | immediate_fade | 2 | 2.5 | OOS | 253 | -236.5 | -0.1435 | 0.9732 | -0.9348 | 0.5257 | 64.63 | -73.6 | -1580 | 5.711 | 72.19 | 448.5 | -685 |
| B__confirmed_fade__n1__m2.0__rv2.5__p0__c1 | confirmed_fade | 2 | 2.5 | IS | 655 | -3384 | -2.689 | 0.5909 | -5.166 | 0.4489 | 16.63 | -22.91 | -3445 | 5.684 | 80.77 | -1541 | -1843 |
| B__confirmed_fade__n1__m2.0__rv2.5__p0__c1 | confirmed_fade | 2 | 2.5 | OOS | 149 | 1341 | 1.824 | 1.398 | 9.003 | 0.5101 | 61.96 | -46.13 | -554.5 | 5.597 | 42.72 | 970 | 371.5 |

## 3. Parameter-region audit

| scope | cells | fraction_positive_is | fraction_positive_oos | fraction_positive_both | mean_local_oos_continuity | edge_only |
| --- | --- | --- | --- | --- | --- | --- |
| all | 18 | 0 | 0.4444 | 0 | 0.4306 | False |
| confirmed_fade | 9 | 0 | 0.8889 | 0 | 0.8611 | False |
| immediate_fade | 9 | 0 | 0 | 0 | 0 | False |

Continuity is real only inside the post-2023 confirmed-fade surface. Immediate fades remain negative, every IS cell is negative, and the sign change is mode- and era-dependent rather than an edge-threshold accident.

## 4. Temporal stability

| period | removed_year | trades | net_pnl | sharpe | pf | expectancy | maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IS |  | 1333 | -6763 | -2.962 | 0.5936 | -5.073 | -7025 |
| OOS |  | 318 | 3165 | 1.984 | 1.461 | 9.953 | -841 |
| first_half |  | 1071 | -6460 | -3.911 | 0.4941 | -6.031 | -6695 |
| second_half |  | 580 | 2862 | 1.239 | 1.267 | 4.934 | -1016 |
| pre_2020 |  | 960 | -4644 | -4.058 | 0.5206 | -4.838 | -4684 |
| 2020_2022 |  | 343 | -2253 | -2.386 | 0.6622 | -6.567 | -2275 |
| 2023_plus |  | 348 | 3299 | 1.969 | 1.462 | 9.48 | -841 |
| remove_best_year | 2026 | 1579 | -6223 | -1.872 | 0.7087 | -3.941 | -7083 |
| remove_worst_year | 2020 | 1530 | -1953 | -0.5002 | 0.9043 | -1.276 | -5439 |

| year | trades | net_pnl | sharpe | pf | expectancy | maxdd |
| --- | --- | --- | --- | --- | --- | --- |
| 2015 | 193 | -963.5 | -4.513 | 0.4802 | -4.992 | -1016 |
| 2016 | 197 | -946.5 | -3.606 | 0.5853 | -4.805 | -1164 |
| 2017 | 217 | -765.5 | -3.351 | 0.5822 | -3.528 | -758 |
| 2018 | 175 | -558.5 | -3.789 | 0.5829 | -3.191 | -704.5 |
| 2019 | 178 | -1410 | -5.248 | 0.4078 | -7.921 | -1414 |
| 2020 | 121 | -1645 | -4.074 | 0.4669 | -13.59 | -2034 |
| 2021 | 114 | -175 | -0.7001 | 0.8886 | -1.535 | -552 |
| 2022 | 108 | -433 | -1.56 | 0.7848 | -4.009 | -811 |
| 2023 | 99 | 495.5 | 2.263 | 1.484 | 5.005 | -223.5 |
| 2024 | 95 | -606.5 | -2.425 | 0.6854 | -6.384 | -834 |
| 2025 | 82 | 785 | 1.79 | 1.382 | 9.573 | -398 |
| 2026 | 72 | 2625 | 4.71 | 2.228 | 36.46 | -466.5 |

The mechanism structurally changes after 2023: 2015–2022 are negative, 2024 is negative, and 2025–2026 dominate the OOS gain. It does not exist throughout the history.

## 5. Long/short decomposition

| scope | side | trades | net_pnl | sharpe | pf | expectancy | maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IS | negative_climax_long | 723 | -3716 | -2.759 | 0.6015 | -5.139 | -4073 |
| IS | positive_climax_short | 610 | -3047 | -2.917 | 0.5834 | -4.995 | -3105 |
| OOS | negative_climax_long | 185 | 1949 | 1.948 | 1.473 | 10.54 | -769 |
| OOS | positive_climax_short | 133 | 1215 | 2.051 | 1.444 | 9.139 | -531.5 |
| OOS_plus_2_ticks | negative_climax_long | 185 | 1209 | 1.209 | 1.272 | 6.538 | -1097 |
| OOS_plus_2_ticks | positive_climax_short | 133 | 683.5 | 1.152 | 1.226 | 5.139 | -559.5 |

The OOS effect is approximately symmetric: both sides have Sharpe near 2 and both survive +2 ticks/side. The long side contributes more dollars because it trades more often and captures the large 2026 contribution. No directional filter is proposed; the file retains every year and both sides.

## 6. Event study and dose response

| events | probability_reversal | probability_25pct_retracement | probability_50pct_retracement | probability_full_retracement | mean_mae_atr | mean_mfe_atr | median_time_to_reversal_bars | median_time_to_50pct_retracement_bars | dimension | bucket | reversal_probability | retracement_50_probability | full_retracement_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 239 | 0.8619 | 0.9833 | 0.8828 | 0.6653 | 2.608 | 2.703 | 1 | 1 | overall | all |  |  |  |
| 48 |  |  |  |  | 2.117 | 2.999 | 1 |  | volume_abnormality | low | 0.8542 | 0.8958 | 0.7292 |
| 74 |  |  |  |  | 2.721 | 2.579 | 1 |  | volume_abnormality | mid | 0.8784 | 0.8919 | 0.6757 |
| 117 |  |  |  |  | 2.737 | 2.66 | 1 |  | volume_abnormality | high | 0.8547 | 0.8718 | 0.6325 |
| 110 |  |  |  |  | 2.6 | 2.369 | 1 |  | price_displacement | low | 0.8091 | 0.9182 | 0.6909 |
| 61 |  |  |  |  | 2.236 | 2.569 | 1 |  | price_displacement | mid | 0.9344 | 0.9344 | 0.7049 |
| 68 |  |  |  |  | 2.953 | 3.364 | 1 |  | price_displacement | high | 0.8824 | 0.7794 | 0.5882 |
| 80 |  |  |  |  | 2.909 | 2.654 | 1 |  | candle_body_range | low | 0.85 | 0.925 | 0.7 |
| 79 |  |  |  |  | 2.826 | 2.515 | 1 |  | candle_body_range | mid | 0.8481 | 0.8354 | 0.6329 |
| 80 |  |  |  |  | 2.09 | 2.938 | 1 |  | candle_body_range | high | 0.8875 | 0.8875 | 0.6625 |

Paths are normalized by signal-bar ex-ante ATR. A true reversal signature is present after confirmed climaxes, but the bucket table does not establish a clean monotone response across every volume, displacement, and candle-structure axis.

## 7. Does volume add information?

| candidate | scenario | positive_year_fraction | trades | net_pnl | sharpe | pf | expectancy | win_rate | avg_winner | avg_loser | maxdd | avg_holding | turnover_trades_per_year | long_trades | long_pnl | long_sharpe | short_trades | short_pnl | short_sharpe | avg_mae | avg_mfe | empirical_candidate_sharpe_percentile | empirical_candidate_expectancy_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRICE_EXTREME_PLUS_VOLUME_CLIMAX | nominal | 0.75 | 318 | 3165 | 1.984 | 1.461 | 9.953 | 0.5283 | 59.68 | -45.75 | -841 | 5.679 | 90.39 | 185 | 1949 | 1.948 | 133 | 1215 | 2.051 | 55 | 68 |  |  |
| PRICE_EXTREME_ONLY | nominal | 0.75 | 547 | 2599 | 0.938 | 1.186 | 4.752 | 0.5155 | 58.69 | -52.64 | -1060 | 5.726 | 155.5 | 308 | 2466 | 1.654 | 239 | 133.5 | 0.09645 | 58.4 | 65.92 |  |  |
| DELTA_VOLUME_MINUS_PRICE | nominal | 0 | -229 | 565.5 | 1.046 | 0.2749 | 5.201 |  |  |  | 219.5 |  |  |  |  |  |  |  |  |  |  |  |  |
| PRICE_EXTREME_PLUS_VOLUME_CLIMAX | plus_1_tick_side |  | 318 | 2529 | 1.586 | 1.353 | 7.953 | 0.5189 | 58.75 | -46.83 | -1090 | 5.679 | 90.39 | 185 | 1579 | 1.578 | 133 | 949.5 | 1.602 | 55 | 68 |  |  |
| PRICE_EXTREME_ONLY | plus_1_tick_side |  | 547 | 1505 | 0.5432 | 1.104 | 2.752 | 0.4991 | 58.59 | -52.88 | -1382 | 5.726 | 155.5 | 308 | 1850 | 1.241 | 239 | -344.5 | -0.2487 | 58.4 | 65.92 |  |  |
| DELTA_VOLUME_MINUS_PRICE | plus_1_tick_side |  | -229 | 1024 | 1.042 | 0.249 | 5.201 |  |  |  | 292.5 |  |  |  |  |  |  |  |  |  |  |  |  |
| PRICE_EXTREME_PLUS_VOLUME_CLIMAX | plus_2_ticks_side |  | 318 | 1893 | 1.187 | 1.253 | 5.953 | 0.5 | 58.93 | -47.02 | -1355 | 5.679 | 90.39 | 185 | 1209 | 1.209 | 133 | 683.5 | 1.152 | 55 | 68 |  |  |
| PRICE_EXTREME_ONLY | plus_2_ticks_side |  | 547 | 411.5 | 0.1484 | 1.027 | 0.7523 | 0.4753 | 59.46 | -52.43 | -1980 | 5.726 | 155.5 | 308 | 1234 | 0.8276 | 239 | -822.5 | -0.5932 | 58.4 | 65.92 |  |  |
| DELTA_VOLUME_MINUS_PRICE | plus_2_ticks_side |  | -229 | 1482 | 1.038 | 0.2258 | 5.201 |  |  |  | 625 |  |  |  |  |  |  |  |  |  |  |  |  |
| PRICE_EXTREME_PLUS_VOLUME_CLIMAX | double_commission |  | 318 | 2370 | 1.486 | 1.327 | 7.453 | 0.5157 | 58.61 | -47.03 | -1155 | 5.679 | 90.39 | 185 | 1487 | 1.486 | 133 | 883 | 1.489 | 55 | 68 |  |  |
| PRICE_EXTREME_ONLY | double_commission |  | 547 | 1232 | 0.4445 | 1.084 | 2.252 | 0.4973 | 58.3 | -53.19 | -1506 | 5.726 | 155.5 | 308 | 1696 | 1.137 | 239 | -464 | -0.3349 | 58.4 | 65.92 |  |  |
| DELTA_VOLUME_MINUS_PRICE | double_commission |  | -229 | 1138 | 1.041 | 0.243 | 5.201 |  |  |  | 351 |  |  |  |  |  |  |  |  |  |  |  |  |
| RANDOM_THINNING_CONTROL | nominal |  | 318 |  | 0.8885 | 1.201 | 4.741 |  |  |  | -1175 |  |  |  |  |  |  |  |  |  |  | 0.926 | 0.913 |

Volume improves the point estimate over the exact price-only event, but the 1,000-draw thinning percentile is below the required 95th percentile. The evidence cannot distinguish incremental volume information from favorable thinning strongly enough for execution.

## 8. Candidate-specific falsification

| scenario | note | trades | net_pnl | sharpe | pf | expectancy | maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal |  | 318 | 3165 | 1.984 | 1.461 | 9.953 | -841 |
| plus_1_tick_side |  | 318 | 2529 | 1.586 | 1.353 | 7.953 | -1090 |
| plus_2_ticks_side |  | 318 | 1893 | 1.187 | 1.253 | 5.953 | -1355 |
| double_commission |  | 318 | 2370 | 1.486 | 1.327 | 7.453 | -1155 |
| entry_delayed_1_bar |  | 309 | 1790 | 1.288 | 1.261 | 5.791 | -798 |
| entry_delayed_2_bar |  | 302 | 853 | 0.5619 | 1.108 | 2.825 | -991.5 |
| remove_best_5_trades |  | 313 | 1444 | 1.09 | 1.211 | 4.615 | -841 |
| remove_best_10_trades |  | 308 | 460 | 0.3739 | 1.067 | 1.494 | -1039 |
| remove_best_5_trading_days |  | 311 | 1293 | 0.9946 | 1.189 | 4.159 | -841 |
| remove_best_year | removed 2026 | 246 | 540 | 0.5928 | 1.114 | 2.195 | -841 |
| opposite_direction |  | 318 | -6027 | -3.776 | 0.4827 | -18.95 | -6075 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m1.0__rv1.5__p0__c1 | 999 | 1819 | 0.4067 | 1.066 | 1.821 | -5141 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m1.0__rv2.0__p0__c1 | 666 | -17 | -0.005271 | 0.9991 | -0.02553 | -3739 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m1.0__rv2.5__p0__c1 | 498 | 615 | 0.252 | 1.043 | 1.235 | -2567 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m1.5__rv1.5__p0__c1 | 418 | 3923 | 2.033 | 1.442 | 9.385 | -1098 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m1.5__rv2.0__p0__c1 | 318 | 3165 | 1.984 | 1.461 | 9.953 | -841 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m1.5__rv2.5__p0__c1 | 266 | 1892 | 1.438 | 1.31 | 7.113 | -1125 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m2.0__rv1.5__p0__c1 | 197 | 1951 | 2.037 | 1.452 | 9.906 | -567 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m2.0__rv2.0__p0__c1 | 171 | 1583 | 1.905 | 1.417 | 9.26 | -554.5 |
| nearby_inherited_threshold | B__confirmed_fade__n1__m2.0__rv2.5__p0__c1 | 149 | 1341 | 1.824 | 1.398 | 9.003 | -554.5 |
| random_same_frequency_labels | 100 draws; mean Sharpe 0.695; p95 1.674 | 378.2 | 1429 | 0.6945 | 1.158 | 3.783 | -1474 |
| shuffle_volume_within_time_of_day | 100 draws; mean Sharpe 0.187; p95 2.442 | 108.5 | 62.27 | 0.1866 | 1.076 | 0.4342 | -988.5 |

The candidate survives +2 ticks/side, one- and two-bar delays, and removal of the best 10 trades, but fails the full-history temporal falsification and does not decisively beat matched thinning.

## 9. Regime diagnostics

| dimension | state | trades | expectancy | sharpe | reversal_probability | mean_mae | mean_mfe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| time_of_day | early | 81 | 12.6 | 2.398 | 0.5802 | 59.81 | 83.21 |
| time_of_day | middle | 138 | 7.5 | 1.234 | 0.5 | 62.79 | 72.68 |
| time_of_day | late | 99 | 11.21 | 3.627 | 0.6364 | 40.21 | 49.04 |
| gap_day | gap | 216 | 11.19 | 2.157 | 0.5556 | 57.27 | 69.53 |
| gap_day | no_gap | 102 | 7.343 | 1.578 | 0.5784 | 50.2 | 64.77 |
| prior_rv_pct | low | 106 | -2.896 | -0.7531 | 0.5 | 46.38 | 48.12 |
| prior_rv_pct | mid | 107 | 18.22 | 4.242 | 0.5888 | 47.64 | 65.06 |
| prior_rv_pct | high | 105 | 14.5 | 2.239 | 0.6 | 71.22 | 91.08 |
| atr_ratio | low | 106 | -1.151 | -0.5267 | 0.5283 | 32.11 | 30.12 |
| atr_ratio | mid | 106 | 10.47 | 2.287 | 0.5755 | 46.92 | 62.14 |
| atr_ratio | high | 106 | 20.54 | 2.901 | 0.5849 | 85.98 | 111.7 |
| cum_rvol_20 | low | 106 | 9.509 | 1.827 | 0.5755 | 46.02 | 59.3 |
| cum_rvol_20 | mid | 106 | 5.83 | 1.672 | 0.5377 | 56.61 | 65.58 |
| cum_rvol_20 | high | 106 | 14.52 | 2.36 | 0.5755 | 62.38 | 79.13 |
| prior_efficiency | low | 102 | 12.85 | 2.323 | 0.5882 | 51.78 | 71.75 |
| prior_efficiency | mid | 102 | 4.275 | 1.09 | 0.5392 | 54.15 | 58.03 |
| prior_efficiency | high | 103 | 10.45 | 1.914 | 0.5728 | 60.07 | 69.23 |
| prior_range_pct | low | 107 | 12.1 | 3.386 | 0.5888 | 38.57 | 56.97 |
| prior_range_pct | mid | 105 | -2.205 | -0.3897 | 0.4857 | 65.48 | 61.64 |
| prior_range_pct | high | 106 | 19.83 | 3.564 | 0.6132 | 61.22 | 85.44 |

The lowest prior-RV and ATR buckets lose after costs while the middle and high buckets are profitable. Reversal probability also rises only modestly, so the weakest-volatility failure is primarily disappearance of net reversion/cost coverage, not a pure MAE expansion. Time-of-day concentration is reported but not converted into a filter.

## 10. Existing MGC block and frozen nine sleeves

| block | days | candidate_correlation | correlation_on_negative_days | active_day_overlap | simultaneous_loss_rate | candidate_mean_worst_5pct | candidate_mean_worst_10pct | candidate_mean_worst_20pct | base_sharpe | added_sharpe | delta_sharpe | base_maxdd | added_maxdd | base_downside_deviation | added_downside_deviation | marginal_variance | marginal_risk_adjusted_contribution | intraday_overlap_rate | direction_overlap_when_simultaneous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_2_MGC_sleeves | 564 | 0.3647 | -0.3084 | 0.1082 | 0.02482 | 8 | 2.482 | 3.979 | 0.763 | 0.9529 | 0.1899 | -1890 | -1902 | 1497 | 1491 | 1.26e+04 | 0.2608 | 0.1824 | 0.3448 |
| frozen_9_sleeves | 564 | 0.1415 | -0.03085 | 0.3103 | 0.07447 | 13.28 | 6.86 | 2.265 | 1.806 | 1.884 | 0.07804 | -6014 | -5944 | 4550 | 4551 | 1.299e+04 | 0.1053 | 0.1824 | 0.3448 |

One MGC contract is added without weight optimization. Intraday direction overlap is computed from the two frozen MGC trade ledgers; worst-day details are in the supporting CSV. Diversification is useful descriptively but cannot repair the standalone temporal failure.

## 11. Topstep

| status | reason |
| --- | --- |
| NOT_RUN_FAILED_PRIOR_GATES | All 18 cells are negative IS and the canonical OOS result is recent/regime-concentrated; Phase 11 gate was not reached. |

The protocol says to run Topstep only after prior phases survive. They did not, so no point estimate or confidence interval is manufactured.

## Gate decision

- Broad inherited stability: **FAIL** — 0% of cells are positive in both IS and OOS.
- Temporal stability: **FAIL** — the profitable regime is post-2023 and strongest in 2025–2026.
- Incremental volume information: **INCONCLUSIVE** — improved point estimates, below the 95th percentile of matched thinning.
- Cost and top-trade robustness: **PASS OOS**.
- Causal/event-study interpretation: **PLAUSIBLE**, but regime-dependent.
- Diversification: **POSITIVE DESCRIPTIVELY**, not sufficient for promotion.

**Decision: `REAL_EFFECT_NOT_YET_EXECUTABLE`.** Freeze the inherited region and mechanism for prospective observation only. Do not freeze a best cell, add a directional/regime filter, alter the nine sleeves, or deploy the candidate.