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

- Variant: `sizing_3state_overnight_range_pts`
- Feature: `overnight_range_pts`
- OOS net PnL: `2179.5`
- OOS Sharpe: `1.253`
- OOS max DD: `-517.5`
- Net-PnL retention vs nominal: `0.423`
- Sharpe delta vs nominal: `-0.305`
- Max-DD improvement vs nominal: `0.124`

## Full summary

| variant_name | family | feature_name | bucketing | description | calibration_scope | parameters_json | note | verdict | overall_net_pnl | overall_sharpe | overall_sortino | overall_profit_factor | overall_expectancy | overall_max_drawdown | overall_n_trades | overall_n_days_traded | overall_pct_days_traded | overall_worst_day | overall_longest_losing_streak_daily | overall_median_recovery_days | overall_max_recovery_days | is_net_pnl | is_sharpe | is_sortino | is_profit_factor | is_expectancy | is_max_drawdown | is_n_trades | is_n_days_traded | is_pct_days_traded | is_worst_day | is_longest_losing_streak_daily | is_median_recovery_days | is_max_recovery_days | oos_net_pnl | oos_sharpe | oos_sortino | oos_profit_factor | oos_expectancy | oos_max_drawdown | oos_n_trades | oos_n_days_traded | oos_pct_days_traded | oos_worst_day | oos_longest_losing_streak_daily | oos_median_recovery_days | oos_max_recovery_days | overall_trade_coverage_vs_nominal | overall_day_coverage_vs_nominal | is_trade_coverage_vs_nominal | is_day_coverage_vs_nominal | oos_trade_coverage_vs_nominal | oos_day_coverage_vs_nominal | oos_net_pnl_retention_vs_nominal | oos_sharpe_delta_vs_nominal | oos_max_drawdown_improvement_vs_nominal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | baseline | nominal | none | Retained-final original sleeve without extra 3-state overlay. | none | {} | Reference retained-final sleeve for all coverage and improvement comparisons. | baseline_reference | 9776.000 | 0.892 | 0.404 | 1.407 | 32.805 | -1913.500 | 298 | 298 | 0.139 | -249.000 | 2 | 47.500 | 227.000 | 4620.500 | 0.604 | 0.270 | 1.249 | 20.813 | -1913.500 | 222 | 222 | 0.148 | -249.000 | 2 | 84.000 | 224.000 | 5155.500 | 1.559 | 0.731 | 1.951 | 67.836 | -590.500 | 76 | 76 | 0.118 | -247.500 | 2 | 17.000 | 77.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| filter_skip_worst_gap_abs_atr20 | regime_filter | gap_abs_atr20 | quantile_3 | Skip the weakest retained-final IS bucket for feature gap_abs_atr20. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 0.0, "mid": 1.0}} | Retained-final campaign: bucket low removed using IS-only ranking. | worse_than_baseline | 6847.500 | 0.766 | 0.285 | 1.435 | 35.479 | -1325.500 | 193 | 193 | 0.090 | -243.000 | 2 | 45.000 | 336.000 | 4128.000 | 0.644 | 0.243 | 1.334 | 27.892 | -1325.500 | 148 | 148 | 0.099 | -243.000 | 2 | 67.500 | 333.000 | 2719.500 | 1.078 | 0.387 | 1.802 | 60.433 | -759.000 | 45 | 45 | 0.070 | -239.000 | 1 | 37.000 | 149.000 | 0.648 | 0.648 | 0.667 | 0.667 | 0.592 | 0.592 | 0.527 | -0.481 | -0.285 |
| filter_skip_worst_overnight_range_pts | regime_filter | overnight_range_pts | quantile_3 | Skip the weakest retained-final IS bucket for feature overnight_range_pts. | is_only | {"bucket_multipliers": {"high": 1.0, "low": 1.0, "mid": 0.0}} | Retained-final campaign: bucket mid removed using IS-only ranking. | mixed | 8760.500 | 0.996 | 0.366 | 1.622 | 45.391 | -1319.500 | 193 | 193 | 0.090 | -249.000 | 2 | 40.000 | 300.000 | 4756.000 | 0.762 | 0.283 | 1.415 | 32.135 | -1319.500 | 148 | 148 | 0.099 | -249.000 | 2 | 72.000 | 300.000 | 4004.500 | 1.570 | 0.567 | 2.520 | 88.989 | -385.000 | 45 | 45 | 0.070 | -247.500 | 1 | 32.000 | 71.000 | 0.648 | 0.648 | 0.667 | 0.667 | 0.592 | 0.592 | 0.777 | 0.011 | 0.348 |
| sizing_3state_overnight_range_pts | dynamic_sizing | overnight_range_pts | quantile_3 | Three-state discrete sizing on retained-final feature overnight_range_pts. | is_only | {"bucket_multipliers": {"high": 0.75, "low": 1.0, "mid": 0.5}} | Retained-final original trade set, with risk scaled by the calibrated 3-state map. | cuts_too_much_exposure | 4976.500 | 0.707 | 0.238 | 1.446 | 30.345 | -1542.000 | 164 | 164 | 0.077 | -249.000 | 2 | 83.000 | 502.000 | 2797.000 | 0.533 | 0.193 | 1.294 | 20.873 | -1542.000 | 134 | 134 | 0.089 | -249.000 | 2 | 84.000 | 502.000 | 2179.500 | 1.253 | 0.326 | 2.325 | 72.650 | -517.500 | 30 | 30 | 0.047 | -186.000 | 1 | 54.500 | 160.000 | 0.550 | 0.550 | 0.604 | 0.604 | 0.395 | 0.395 | 0.423 | -0.305 | 0.124 |