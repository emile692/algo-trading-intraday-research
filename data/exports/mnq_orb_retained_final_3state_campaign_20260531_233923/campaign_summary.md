# MNQ ORB retained-final 3-state campaign

## Baseline

- This campaign is run on the **retained final** sleeve, not on the earlier nominal ORB branch.
- Retained config: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate`
- OR window: `15` minutes
- Direction: `long`
- VWAP confirmation: `enabled`
- ATR ensemble: `ATR(14)`, vote threshold `0.50`
- Compression / dynamic gate: `weak_close` / `noise_area_gate`
- Base risk per trade: `0.50%`
- Dataset: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet`

## Nominal retained-final OOS

- Net PnL: `5155.5`
- Sharpe: `1.559`
- Max DD: `-590.5`
- Trades: `76`

## Top feature candidates

| feature_name | family | bucket_kind | bucket_count | min_bucket_obs_is | balance_is | is_score_spread | feature_selection_score | best_bucket_is | worst_bucket_is | skip_coverage_is | valid_for_overlay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overnight_range_pts | volatility | quantile | 3 | 74 | 1.000 | 4.940 | 4.940 | low | mid | 0.667 | yes |
| weekday_name | structural | categorical | 5 | 37 | 0.755 | 5.124 | 4.496 | friday | tuesday | 0.793 | no |
| realized_vol_ratio_15_60 | volatility | quantile | 3 | 74 | 1.000 | 4.363 | 4.363 | high | mid | 0.667 | yes |
| gap_abs_atr20 | extension | quantile | 3 | 74 | 1.000 | 4.358 | 4.358 | high | low | 0.667 | yes |
| signal_extension_over_or | extension | quantile | 3 | 74 | 1.000 | 4.275 | 4.275 | high | mid | 0.667 | yes |

## Best 3-state overlay

- Variant: `sizing_3state_atr_ratio_10_30`
- Feature: `atr_ratio_10_30`
- OOS net PnL: `4022.5`
- OOS Sharpe: `1.799`
- OOS max DD: `-328.0`
- Net-PnL retention vs nominal: `0.780`
- Sharpe delta vs nominal: `0.240`
- Max-DD improvement vs nominal: `0.445`

## Full summary

| variant_name | family | feature_name | bucketing | description | calibration_scope | parameters_json | note | verdict | overall_net_pnl | overall_sharpe | overall_sortino | overall_profit_factor | overall_expectancy | overall_max_drawdown | overall_n_trades | overall_n_days_traded | overall_pct_days_traded | overall_worst_day | overall_longest_losing_streak_daily | overall_median_recovery_days | overall_max_recovery_days | is_net_pnl | is_sharpe | is_sortino | is_profit_factor | is_expectancy | is_max_drawdown | is_n_trades | is_n_days_traded | is_pct_days_traded | is_worst_day | is_longest_losing_streak_daily | is_median_recovery_days | is_max_recovery_days | oos_net_pnl | oos_sharpe | oos_sortino | oos_profit_factor | oos_expectancy | oos_max_drawdown | oos_n_trades | oos_n_days_traded | oos_pct_days_traded | oos_worst_day | oos_longest_losing_streak_daily | oos_median_recovery_days | oos_max_recovery_days | overall_trade_coverage_vs_nominal | overall_day_coverage_vs_nominal | is_trade_coverage_vs_nominal | is_day_coverage_vs_nominal | oos_trade_coverage_vs_nominal | oos_day_coverage_vs_nominal | oos_net_pnl_retention_vs_nominal | oos_sharpe_delta_vs_nominal | oos_max_drawdown_improvement_vs_nominal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | baseline | nominal | none | Retained-final original sleeve without extra 3-state overlay. | none | {} | Reference retained-final sleeve for all coverage and improvement comparisons. | baseline_reference | 9776.000 | 0.892 | 0.404 | 1.407 | 32.805 | -1913.500 | 298 | 298 | 0.139 | -249.000 | 2 | 47.500 | 227.000 | 4620.500 | 0.604 | 0.270 | 1.249 | 20.813 | -1913.500 | 222 | 222 | 0.148 | -249.000 | 2 | 84.000 | 224.000 | 5155.500 | 1.559 | 0.731 | 1.951 | 67.836 | -590.500 | 76 | 76 | 0.118 | -247.500 | 2 | 17.000 | 77.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| filter_skip_worst_gap_abs_atr20 | regime_filter | gap_abs_atr20 | quantile_3 | Skip the weakest retained-final IS bucket for feature gap_abs_atr20. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 0.0, "mid": 1.0}} | Retained-final campaign: bucket low removed using IS-only ranking. | worse_than_baseline | 6847.500 | 0.766 | 0.285 | 1.435 | 35.479 | -1325.500 | 193 | 193 | 0.090 | -243.000 | 2 | 45.000 | 336.000 | 4128.000 | 0.644 | 0.243 | 1.334 | 27.892 | -1325.500 | 148 | 148 | 0.099 | -243.000 | 2 | 67.500 | 333.000 | 2719.500 | 1.078 | 0.387 | 1.802 | 60.433 | -759.000 | 45 | 45 | 0.070 | -239.000 | 1 | 37.000 | 149.000 | 0.648 | 0.648 | 0.667 | 0.667 | 0.592 | 0.592 | 0.527 | -0.481 | -0.285 |
| filter_skip_worst_overnight_range_pts | regime_filter | overnight_range_pts | quantile_3 | Skip the weakest retained-final IS bucket for feature overnight_range_pts. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 1.0, "mid": 0.0}} | Retained-final campaign: bucket mid removed using IS-only ranking. | mixed | 8760.500 | 0.996 | 0.366 | 1.622 | 45.391 | -1319.500 | 193 | 193 | 0.090 | -249.000 | 2 | 40.000 | 300.000 | 4756.000 | 0.762 | 0.283 | 1.415 | 32.135 | -1319.500 | 148 | 148 | 0.099 | -249.000 | 2 | 72.000 | 300.000 | 4004.500 | 1.570 | 0.567 | 2.520 | 88.989 | -385.000 | 45 | 45 | 0.070 | -247.500 | 1 | 32.000 | 71.000 | 0.648 | 0.648 | 0.667 | 0.667 | 0.592 | 0.592 | 0.777 | 0.011 | 0.348 |
| sizing_3state_overnight_range_pts | dynamic_sizing | overnight_range_pts | quantile_3 | Three-state discrete sizing on retained-final feature overnight_range_pts. | is_only | {"bucket_multipliers": {"high": 0.75, "low": 1.0, "mid": 0.5}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | cuts_too_much_exposure | 4976.500 | 0.707 | 0.238 | 1.446 | 30.345 | -1542.000 | 164 | 164 | 0.077 | -249.000 | 2 | 83.000 | 502.000 | 2797.000 | 0.533 | 0.193 | 1.294 | 20.873 | -1542.000 | 134 | 134 | 0.089 | -249.000 | 2 | 84.000 | 502.000 | 2179.500 | 1.253 | 0.326 | 2.325 | 72.650 | -517.500 | 30 | 30 | 0.047 | -186.000 | 1 | 54.500 | 160.000 | 0.550 | 0.550 | 0.604 | 0.604 | 0.395 | 0.395 | 0.423 | -0.305 | 0.124 |
| sizing_3state_realized_vol_ratio_15_60 | dynamic_sizing | realized_vol_ratio_15_60 | quantile_3 | Three-state discrete sizing on retained-final feature realized_vol_ratio_15_60. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 0.75, "mid": 0.5}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | cuts_too_much_exposure | 4581.500 | 0.580 | 0.209 | 1.351 | 27.936 | -1632.000 | 164 | 164 | 0.077 | -247.500 | 2 | 39.000 | 694.000 | 2207.000 | 0.405 | 0.149 | 1.224 | 17.516 | -1632.000 | 126 | 126 | 0.084 | -243.000 | 2 | 49.000 | 694.000 | 2374.500 | 0.972 | 0.323 | 1.738 | 62.487 | -887.500 | 38 | 38 | 0.059 | -247.500 | 2 | 24.500 | 243.000 | 0.550 | 0.550 | 0.568 | 0.568 | 0.500 | 0.500 | 0.461 | -0.587 | -0.503 |
| sizing_3state_gap_abs_atr20 | dynamic_sizing | gap_abs_atr20 | quantile_3 | Three-state discrete sizing on retained-final feature gap_abs_atr20. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 0.5, "mid": 0.75}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | mixed | 5375.500 | 0.675 | 0.255 | 1.390 | 29.057 | -1185.500 | 185 | 185 | 0.086 | -243.000 | 2 | 64.500 | 341.000 | 2438.000 | 0.439 | 0.169 | 1.226 | 17.169 | -1185.500 | 142 | 142 | 0.095 | -243.000 | 2 | 101.000 | 341.000 | 2937.500 | 1.219 | 0.426 | 1.983 | 68.314 | -533.500 | 43 | 43 | 0.067 | -239.000 | 1 | 37.000 | 149.000 | 0.621 | 0.621 | 0.640 | 0.640 | 0.566 | 0.566 | 0.570 | -0.340 | 0.097 |
| sizing_3state_signal_extension_over_or | dynamic_sizing | signal_extension_over_or | quantile_3 | Three-state discrete sizing on retained-final feature signal_extension_over_or. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 0.75, "mid": 0.5}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | cuts_too_much_exposure | 5542.500 | 0.728 | 0.268 | 1.430 | 30.287 | -973.500 | 183 | 183 | 0.085 | -243.000 | 2 | 24.000 | 331.000 | 3786.500 | 0.686 | 0.265 | 1.380 | 26.665 | -973.500 | 142 | 142 | 0.095 | -243.000 | 2 | 17.000 | 331.000 | 1756.000 | 0.845 | 0.269 | 1.599 | 42.829 | -647.000 | 41 | 41 | 0.064 | -208.000 | 2 | 63.000 | 200.000 | 0.614 | 0.614 | 0.640 | 0.640 | 0.539 | 0.539 | 0.341 | -0.714 | -0.096 |
| sizing_3state_opening_range_width_pts | dynamic_sizing | opening_range_width_pts | quantile_3 | Three-state discrete sizing on retained-final feature opening_range_width_pts. | is_only | {"bucket_multipliers": {"high": 0.5, "low": 0.75, "mid": 1.0}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | cuts_too_much_exposure | 5247.000 | 0.732 | 0.264 | 1.428 | 29.150 | -883.500 | 180 | 180 | 0.084 | -243.000 | 2 | 49.000 | 413.000 | 3544.000 | 0.685 | 0.259 | 1.368 | 24.611 | -883.500 | 144 | 144 | 0.096 | -243.000 | 2 | 29.000 | 413.000 | 1703.000 | 0.858 | 0.266 | 1.645 | 47.306 | -691.500 | 36 | 36 | 0.056 | -208.000 | 2 | 97.000 | 143.000 | 0.604 | 0.604 | 0.649 | 0.649 | 0.474 | 0.474 | 0.330 | -0.701 | -0.171 |
| sizing_3state_atr_ratio_10_30 | dynamic_sizing | atr_ratio_10_30 | quantile_3 | Three-state discrete sizing on retained-final feature atr_ratio_10_30. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 0.5, "mid": 0.75}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | cuts_too_much_exposure | 7353.000 | 0.935 | 0.344 | 1.599 | 41.778 | -1875.000 | 176 | 176 | 0.082 | -249.000 | 2 | 35.500 | 335.000 | 3330.500 | 0.592 | 0.222 | 1.325 | 24.310 | -1875.000 | 137 | 137 | 0.091 | -249.000 | 2 | 55.000 | 228.000 | 4022.500 | 1.799 | 0.639 | 2.967 | 103.141 | -328.000 | 39 | 39 | 0.061 | -228.000 | 1 | 21.000 | 165.000 | 0.591 | 0.591 | 0.617 | 0.617 | 0.513 | 0.513 | 0.780 | 0.240 | 0.445 |
| sizing_3state_signal_vwap_distance_atr20 | dynamic_sizing | signal_vwap_distance_atr20 | quantile_3 | Three-state discrete sizing on retained-final feature signal_vwap_distance_atr20. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 0.5, "mid": 0.75}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | mixed | 5973.000 | 0.761 | 0.269 | 1.450 | 32.286 | -1148.000 | 185 | 185 | 0.086 | -247.500 | 2 | 45.000 | 319.000 | 3616.000 | 0.665 | 0.241 | 1.358 | 25.287 | -1148.000 | 143 | 143 | 0.095 | -240.500 | 2 | 49.000 | 319.000 | 2357.000 | 0.978 | 0.315 | 1.741 | 56.119 | -525.000 | 42 | 42 | 0.065 | -247.500 | 1 | 41.000 | 113.000 | 0.621 | 0.621 | 0.644 | 0.644 | 0.553 | 0.553 | 0.457 | -0.580 | 0.111 |