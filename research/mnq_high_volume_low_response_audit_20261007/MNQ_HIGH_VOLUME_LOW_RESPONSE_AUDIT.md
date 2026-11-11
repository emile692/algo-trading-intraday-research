# MNQ High-Volume / Low-Response Proxy Audit

**Final verdict: `REJECTED`**

This is an OHLCV proxy for low price impact under abnormal volume. It is not true absorption. No new parameter values were tested and no existing sleeve was changed.

## Executive conclusion

The inherited representative reproduces exactly (maximum source/replay PnL difference: $0.00), but it is a retrospective local anomaly rather than a repeatable executable mechanism. Its OOS result is positive (Sharpe 1.09, PF 1.15, 179 trades), while the same cell is negative IS and the inherited landscape is not jointly stable.

Across the 12 inherited cells, 33.3% are positive OOS, 8.3% are positive IS, and 0.0% are positive in both. The representative's immediate positive-neighbour fraction is 40.0%; `LOCAL_REGION` is best described as **one threshold boundary**, not a broad plateau.

Concentration is material: after removing the best 5 trades, PnL is $-634 and Sharpe -0.60; after removing the best 10, PnL is $-1,874 and Sharpe -1.95.

## Phase 1 — exact inherited mechanism

- Bars: MNQ regular session, repository 5-minute bars resampled from Databento OHLCV-1m.
- Expected volume: median volume of the prior 20 sessions at the identical five-minute time-of-day slot; the current session is shifted out.
- RVOL: current bar volume / causal expected median volume. Volume surprise also exists in the source as (volume − median) / causal IQR-equivalent scale, but is not used by this signal.
- Normalized response: body = |close − open| / causal 20-bar within-session median true range; range = (high − low) / the same ATR.
- Low response: body ≤ 0.25 ATR and range ≤ 1.0 ATR for the canonical cell.
- Preceding move: sign(close[t] − close[t−3]) on the signal bar.
- Reversal direction: negative of that preceding three-bar move.
- Confirmation: none beyond the completed signal bar (`confirm=1` is an inherited identifier field, not an extra bar).
- Execution: signal at bar close; enter next bar open. Non-overlapping trades only.
- Exit: close of the sixth held bar, clipped to the session close.
- Nominal costs: one tick per side plus $1.25 commission per side = $3.50 round trip.

### Every inherited cell — IS and OOS

| variant_id | scope | trades | net_pnl | sharpe | pf | expectancy | win_rate | maxdd | annualized_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D__preceding_move_reversal__n3__m0.25__rv1.5__p0__c1 | IS | 1501 | -5846 | -1.378 | 0.8744 | -3.894 | 0.465 | -6306 | -1242 |
| D__preceding_move_reversal__n3__m0.25__rv1.5__p0__c1 | OOS | 610 | 1458 | 0.4491 | 1.052 | 2.391 | 0.4885 | -3474 | 712.2 |
| D__bar_breakout__n3__m0.25__rv1.5__p0__c1 | IS | 1505 | -8598 | -1.787 | 0.8211 | -5.713 | 0.4545 | -8900 | -1827 |
| D__bar_breakout__n3__m0.25__rv1.5__p0__c1 | OOS | 612 | -7907 | -2.364 | 0.7604 | -12.92 | 0.4493 | -9172 | -3861 |
| D__preceding_move_reversal__n3__m0.5__rv1.5__p0__c1 | IS | 2123 | -1.409e+04 | -2.441 | 0.7978 | -6.638 | 0.4508 | -1.403e+04 | -2987 |
| D__preceding_move_reversal__n3__m0.5__rv1.5__p0__c1 | OOS | 864 | -3424 | -0.8627 | 0.9204 | -3.963 | 0.4769 | -4834 | -1670 |
| D__bar_breakout__n3__m0.5__rv1.5__p0__c1 | IS | 2125 | -9744 | -1.657 | 0.8557 | -4.585 | 0.4673 | -1e+04 | -2066 |
| D__bar_breakout__n3__m0.5__rv1.5__p0__c1 | OOS | 865 | -9015 | -2.317 | 0.8036 | -10.42 | 0.459 | -9789 | -4396 |
| D__preceding_move_reversal__n3__m0.25__rv2.0__p0__c1 | IS | 570 | -2774 | -1.128 | 0.8677 | -4.866 | 0.4825 | -3546 | -590.3 |
| D__preceding_move_reversal__n3__m0.25__rv2.0__p0__c1 | OOS | 179 | 1316 | 1.094 | 1.154 | 7.349 | 0.5028 | -1584 | 678.7 |
| D__bar_breakout__n3__m0.25__rv2.0__p0__c1 | IS | 572 | -6755 | -2.951 | 0.7068 | -11.81 | 0.458 | -7092 | -1438 |
| D__bar_breakout__n3__m0.25__rv2.0__p0__c1 | OOS | 179 | 43.5 | 0.03829 | 1.005 | 0.243 | 0.5028 | -1100 | 22.44 |
| D__preceding_move_reversal__n3__m0.5__rv2.0__p0__c1 | IS | 843 | -4058 | -1.33 | 0.8653 | -4.813 | 0.4733 | -4232 | -863.6 |
| D__preceding_move_reversal__n3__m0.5__rv2.0__p0__c1 | OOS | 259 | 2507 | 1.698 | 1.204 | 9.68 | 0.5174 | -1364 | 1293 |
| D__bar_breakout__n3__m0.5__rv2.0__p0__c1 | IS | 844 | -8210 | -3.047 | 0.7454 | -9.727 | 0.4609 | -9056 | -1747 |
| D__bar_breakout__n3__m0.5__rv2.0__p0__c1 | OOS | 259 | -407 | -0.2661 | 0.9704 | -1.571 | 0.5019 | -1770 | -210 |
| D__preceding_move_reversal__n3__m0.25__rv2.5__p0__c1 | IS | 270 | 146.5 | 0.1486 | 1.015 | 0.5426 | 0.5148 | -1828 | 31.42 |
| D__preceding_move_reversal__n3__m0.25__rv2.5__p0__c1 | OOS | 51 | -425 | -0.9918 | 0.8616 | -8.333 | 0.4118 | -1124 | -222.4 |
| D__bar_breakout__n3__m0.25__rv2.5__p0__c1 | IS | 271 | -4245 | -3.914 | 0.6458 | -15.66 | 0.4354 | -4246 | -910.4 |
| D__bar_breakout__n3__m0.25__rv2.5__p0__c1 | OOS | 51 | -1059 | -2.699 | 0.6867 | -20.76 | 0.451 | -1907 | -554.2 |
| D__preceding_move_reversal__n3__m0.5__rv2.5__p0__c1 | IS | 389 | -40 | -0.03323 | 0.997 | -0.1028 | 0.5141 | -1300 | -8.579 |
| D__preceding_move_reversal__n3__m0.5__rv2.5__p0__c1 | OOS | 82 | -1214 | -1.744 | 0.7774 | -14.8 | 0.4268 | -2213 | -628.1 |
| D__bar_breakout__n3__m0.5__rv2.5__p0__c1 | IS | 389 | -5191 | -3.245 | 0.6796 | -13.34 | 0.4524 | -5336 | -1113 |
| D__bar_breakout__n3__m0.5__rv2.5__p0__c1 | OOS | 82 | -2227 | -4.06 | 0.6261 | -27.16 | 0.439 | -2922 | -1152 |

Every cell's calendar-year performance is preserved in `results/inherited_cell_annual_results.csv`.

## Phase 3 — decomposition and incremental interaction

| control | trades | net_pnl | sharpe | pf | expectancy | win_rate | avg_winner | avg_loser | maxdd | avg_holding | reversal_probability | both_minus_control_sharpe | both_minus_control_expectancy | both_minus_control_pf | both_minus_control_reversal_probability | both_minus_control_maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price_response_only | 3658 | 1988 | 0.2461 | 1.017 | 0.5433 | 0.4803 | 68.92 | -63.08 | -4290 | 5.885 | 0.5038 | 0.8476 | 6.806 | 0.1375 | 0.004553 | 2706 |
| high_volume_only | 1068 | -3822 | -0.7378 | 0.9237 | -3.578 | 0.4841 | 89.52 | -91.93 | -4858 | 5.862 | 0.5084 | 1.831 | 10.93 | 0.2305 | -4.708e-05 | 3274 |
| both_conditions | 179 | 1316 | 1.094 | 1.154 | 7.349 | 0.5028 | 109.4 | -96.93 | -1584 | 5.933 | 0.5084 |  |  |  |  |  |
| random_trade_count_matched_mean | 179 | 111.9 | 0.1028 | 1.042 | 0.6253 | 0.4797 |  |  | -1349 |  | 0.5041 | 0.9909 | 6.724 | 0.1125 | 0.004302 | -234 |
| shuffled_volume_same_tod | 803 | -3646 | -0.9458 | 0.864 | -4.541 | 0.4558 | 63.32 | -61.94 | -4732 | 5.899 | 0.4757 | 2.039 | 11.89 | 0.2902 | 0.03266 | 3148 |

The interaction is judged against price geometry, volume alone, count-matched random thinning, and same-time-of-day shuffled volume. A positive representative alone is insufficient when these controls and the IS replay do not jointly establish repeatability.

## Phase 4 — mandatory economic 2×2 event study

| category | events | before_cumulative_normalized_return | rvol | normalized_range | candle_body | close_location | return_1_bar_atr | return_2_bar_atr | return_3_bar_atr | return_6_bar_atr | return_12_bar_atr | return_24_bar_atr | reversal_probability | mae_atr | mfe_atr | time_to_reversal_bars | subsequent_realized_volatility_atr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_volume_high_response | 21370 | 0.02677 | 1.033 | 1.07 | 0.597 | 0.5323 | 0.0115 | 0.01024 | 0.00245 | 0.0001968 | 0.01376 | -0.002622 | 0.8579 | 2.136 | 2.139 | 1 | 2.618 |
| normal_volume_low_response | 10032 | 0.1163 | 0.8751 | 0.6777 | 0.1212 | 0.511 | 0.001892 | 0.001612 | 0.002233 | 0.0054 | 0.0116 | 0.05079 | 0.8781 | 1.965 | 2.049 | 1 | 2.456 |
| high_volume_high_response | 2649 | -0.6491 | 2.714 | 1.729 | 0.9836 | 0.4756 | -0.01578 | -0.03441 | -0.04749 | -0.09925 | -0.261 | -0.3872 | 0.8456 | 2.703 | 2.272 | 1 | 3.065 |
| high_volume_low_response | 261 | -0.3175 | 2.414 | 0.7931 | 0.1256 | 0.5236 | 0.04082 | 0.1041 | 0.1382 | 0.1125 | 0.06212 | -0.1725 | 0.8429 | 1.893 | 1.892 | 1 | 2.344 |

Full event rows: `event_level_2x2.csv`. Plot: `event_study_2x2.png`.

## Phase 5 — continuation, reversal, or volatility only

| outcome | trades | net_pnl | sharpe | pf | expectancy | win_rate | avg_winner | avg_loser | maxdd | avg_holding | annualized_pnl | reversal_probability | mean_future_6bar_range_atr | median_future_6bar_range_atr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reversal | 179 | 1316 | 1.094 | 1.154 | 7.349 | 0.5028 | 109.4 | -96.93 | -1584 | 5.933 | 678.7 | 0.5084 |  |  |
| continuation | 179 | -2568 | -2.133 | 0.755 | -14.35 | 0.486 | 90.99 | -114 | -3566 | 5.933 | -1325 | 0.4916 |  |  |
| volatility_only | 262 |  |  |  |  |  |  |  |  |  |  |  | 2.251 | 2.055 |

## Phase 6 — temporal stability

| period | trades | net_pnl | sharpe | pf | expectancy | win_rate | maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IS | 570 | -2774 | -1.128 | 0.8677 | -4.866 | 0.4825 | -3546 |
| OOS | 179 | 1316 | 1.094 | 1.154 | 7.349 | 0.5028 | -1584 |
| first_half | 512 | -2998 | -1.411 | 0.8439 | -5.856 | 0.4805 | -3205 |
| second_half | 237 | 1540 | 0.9973 | 1.15 | 6.5 | 0.5021 | -1584 |
| pre_2020 | 96 | -53 | -0.3827 | 0.9572 | -0.5521 | 0.5208 | -364.5 |
| 2020_2022 | 421 | -3182 | -1.656 | 0.8265 | -7.559 | 0.4679 | -3375 |
| 2023_plus | 232 | 1778 | 1.183 | 1.179 | 7.662 | 0.5086 | -1584 |
| 2019 | 96 | -53 | -0.3827 | 0.9572 | -0.5521 | 0.5208 | -364.5 |
| 2020 | 197 | -1808 | -3.042 | 0.7649 | -9.175 | 0.4518 | -1926 |
| 2021 | 138 | 1006 | 2.05 | 1.218 | 7.293 | 0.529 | -530.5 |
| 2022 | 86 | -2382 | -3.524 | 0.6056 | -27.69 | 0.407 | -3364 |
| 2023 | 43 | 652.5 | 2.722 | 1.657 | 15.17 | 0.5814 | -487.5 |
| 2024 | 55 | -856 | -3.624 | 0.614 | -15.56 | 0.4 | -1098 |
| 2025 | 94 | 1216 | 1.775 | 1.245 | 12.94 | 0.4681 | -1584 |
| 2026 | 40 | 765 | 3.071 | 1.442 | 19.12 | 0.675 | -562 |
| remove_best_year | 655 | -2674 | -0.9002 | 0.891 | -4.082 | 0.4901 | -3935 |

Signal counts and rolling 12-month Sharpe are in `rolling_12m_sharpe_and_frequency.csv` and `rolling_12m_sharpe.png`. The full-history sign reversal from negative IS to positive OOS prevents a stable-effect interpretation.

## Phase 7 — costs, concentration, and falsification

| scenario | trades | net_pnl | sharpe | pf | expectancy | win_rate | maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | 179 | 1316 | 1.094 | 1.154 | 7.349 | 0.5028 | -1584 |
| plus_1_tick_side | 179 | 1136 | 0.9449 | 1.132 | 6.349 | 0.5028 | -1620 |
| plus_2_ticks_side | 179 | 957.5 | 0.7961 | 1.11 | 5.349 | 0.5028 | -1656 |
| double_commissions | 179 | 868 | 0.7217 | 1.099 | 4.849 | 0.5028 | -1674 |
| delay_entry_1_bar | 175 | -2340 | -1.966 | 0.7626 | -13.37 | 0.4114 | -2406 |
| delay_entry_2_bar | 166 | -2186 | -1.818 | 0.7707 | -13.17 | 0.4337 | -2766 |
| remove_best_5_trades | 174 | -633.5 | -0.6039 | 0.9257 | -3.641 | 0.4885 | -2008 |
| remove_best_10_trades | 169 | -1874 | -1.955 | 0.7804 | -11.09 | 0.4734 | -2394 |
| remove_best_5_days | 164 | -1304 | -1.453 | 0.8455 | -7.948 | 0.4634 | -1736 |
| remove_best_year | 85 | 99.5 | 0.1931 | 1.028 | 1.171 | 0.5412 | -1098 |
| opposite_direction | 179 | -2568 | -2.133 | 0.755 | -14.35 | 0.486 | -3566 |
| shuffled_volume_labels | 797 | -1112 | -0.3563 | 0.9532 | -1.395 | 0.478 | -3178 |

The requested random matched controls are in `random_trade_count_controls.csv`; opposite direction, shuffled labels, delays, costs, best trades/days/year removals are above and in the supporting CSV.

## Phase 8 — time of day / volume seasonality

| segment | signal_frequency | mean_rvol | mean_volume | mean_expected_volume | net_pnl | sharpe | expectancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| opening_segment | 0 |  |  |  |  |  |  |
| mid_morning | 16 | 2.288 | 3.357e+04 | 1.479e+04 | 378 | 2.263 | 23.62 |
| midday | 77 | 2.371 | 2.229e+04 | 9459 | -31 | -0.05305 | -0.4026 |
| afternoon | 81 | 2.394 | 1.619e+04 | 6846 | 1038 | 1.639 | 12.82 |
| closing_segment | 5 | 2.428 | 1.752e+04 | 7165 | -70 | -1.237 | -14 |

These are diagnostics only; no time filter is selected.

## Phase 9 — relation to frozen MNQ sleeves

| sleeve | pnl_correlation | negative_day_correlation | signal_day_overlap | direction_overlap | simultaneous_loss_rate | candidate_pnl_worst_mnq_cluster_days | candidate_pnl_strong_trend_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mnq_orb_vwap_atr | -0.008159 | -0.04661 | 0.05142 | 0.6071 | 0.02128 | -4 | -95.5 |
| mnq_volatility_breakout | 0.08639 | 0.04919 | 0.1489 | 0.3514 | 0.03191 | -369.5 | -329.5 |
| mnq_compression_breakout | -0.004641 | -0.004882 | 0.0195 | 0.7273 | 0.007092 | -165 | -244 |
| mnq_atr_trend_flip | 0.06009 | -0.06254 | 0.1596 | 0.2727 | 0.04078 | 270.5 | 946 |
| mnq_trend_cluster_total | 0.07037 | -0.02846 | 0.1826 | 0.3 | 0.04255 | -164 | 134.5 |

## Phase 10 — portfolio and Topstep

| scenario | pnl | sharpe | maxdd | downside_correlation | marginal_variance | worst_day |
| --- | --- | --- | --- | --- | --- | --- |
| 9_sleeves | 4.323e+04 | 1.806 | -6014 |  | 0 | -1378 |
| 9_plus_candidate | 4.454e+04 | 1.816 | -5667 | 0.08044 | 2.201e+04 | -1499 |

| metric | paths | estimate | ci95_low | ci95_high | baseline_probability | candidate_probability |
| --- | --- | --- | --- | --- | --- | --- |
| delta_pass | 8000 | -0.0075 | -0.01275 | -0.002375 | 0.4524 | 0.4449 |
| delta_mll_fail | 8000 | 0.01375 | 0.009 | 0.01862 | 0.4811 | 0.4949 |
| delta_expiry | 8000 | -0.00625 | -0.01013 | -0.002375 | 0.0665 | 0.06025 |

Topstep uses the frozen nine-sleeve sizing unchanged and a paired 8,000-path moving-block bootstrap at the 60-traded-day horizon. The candidate's observed five-minute MAE is added conservatively to the unchanged minute-level nine-sleeve adverse path. A delta is not treated as improvement when its 95% interval includes zero.

## Decision

**`REJECTED`**

The high-volume + low-response proxy has an interesting OOS pocket, but it fails the required standard of an incremental, temporally repeatable interaction: the canonical cell loses IS, the profitable area is threshold-sensitive, only a minority of inherited cells are positive, and concentration removal materially weakens the result. Do not freeze it prospectively as an executable region and do not add it to the nine-sleeve portfolio.