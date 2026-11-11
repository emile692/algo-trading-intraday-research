# Equity Index Gap Reversion — Frozen Region Audit

**Final decision: `REAL_EFFECT_NOT_YET_EXECUTABLE`**

No new parameter was tested or selected. The economic mechanism and four inherited threshold cells were audited as frozen.

## Frozen region — every cell

| asset | threshold | exit_kind | is_trades | is_net_pnl | is_sharpe | is_profit_factor | oos_trades | oos_net_pnl | oos_sharpe | oos_profit_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | 0.25 | fair_value | 678 | -673.1 | -0.1899 | 0.9582 | 293 | 1542 | 0.9623 | 1.242 |
| M2K | 0.5 | fair_value | 334 | 448.3 | 0.1563 | 1.051 | 168 | 2072 | 1.533 | 1.599 |
| M2K | 0.75 | fair_value | 163 | 88.89 | 0.04213 | 1.02 | 93 | 1865 | 1.615 | 2.012 |
| M2K | 1 | fair_value | 80 | -579.5 | -0.4039 | 0.7685 | 52 | 837.5 | 1.021 | 1.744 |
| MES | 0.25 | fair_value | 746 | -3181 | -0.6037 | 0.8737 | 297 | 2178 | 0.6812 | 1.174 |
| MES | 0.5 | fair_value | 432 | -2022 | -0.4507 | 0.8767 | 169 | 4059 | 1.459 | 1.602 |
| MES | 0.75 | fair_value | 250 | -577.7 | -0.1555 | 0.9415 | 96 | 3160 | 1.36 | 1.869 |
| MES | 1 | fair_value | 137 | -947.9 | -0.3649 | 0.8256 | 54 | 1799 | 1.121 | 1.845 |
| MNQ | 0.25 | fair_value | 711 | -6870 | -0.7858 | 0.838 | 304 | 6844 | 1.208 | 1.31 |
| MNQ | 0.5 | fair_value | 392 | -1906 | -0.2664 | 0.9218 | 177 | 8215 | 1.663 | 1.643 |
| MNQ | 0.75 | fair_value | 201 | 1841 | 0.3465 | 1.162 | 94 | 6359 | 1.533 | 1.886 |
| MNQ | 1 | fair_value | 113 | -828 | -0.2018 | 0.8937 | 56 | 2820 | 0.9639 | 1.655 |
| NQ | 0.25 | fair_value | 1586 | -5.068e+04 | -0.5323 | 0.8594 | 672 | 4.524e+04 | 0.4132 | 1.098 |
| NQ | 0.5 | fair_value | 850 | -3.317e+04 | -0.41 | 0.8511 | 366 | 9.193e+04 | 0.996 | 1.362 |
| NQ | 0.75 | fair_value | 450 | 2774 | 0.04544 | 1.025 | 188 | 8.117e+04 | 1.084 | 1.615 |
| NQ | 1 | fair_value | 228 | -4115 | -0.09061 | 0.9344 | 110 | 3.472e+04 | 0.6321 | 1.422 |

## Region ensemble IS/OOS

| asset | scope | trades | net_pnl | sharpe | profit_factor | expectancy | win_rate | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | IS | 678 | -178.8 | -0.08289 | 0.9776 | -0.2638 | 0.4956 | -1272 |
| M2K | OOS | 293 | 1579 | 1.431 | 1.493 | 5.389 | 0.5461 | -418.9 |
| M2K | FULL | 971 | 1400 | 0.4273 | 1.125 | 1.442 | 0.5108 | -1272 |
| MES | IS | 740 | -1682 | -0.47 | 0.8818 | -2.273 | 0.4838 | -2585 |
| MES | OOS | 296 | 2799 | 1.251 | 1.447 | 9.456 | 0.5304 | -1140 |
| MES | FULL | 1036 | 1117 | 0.1887 | 1.055 | 1.078 | 0.4971 | -3021 |
| MNQ | IS | 711 | -1941 | -0.3506 | 0.9097 | -2.73 | 0.4655 | -4492 |
| MNQ | OOS | 304 | 6060 | 1.517 | 1.523 | 19.93 | 0.5724 | -1797 |
| MNQ | FULL | 1015 | 4119 | 0.4178 | 1.125 | 4.058 | 0.4975 | -4492 |
| NQ | IS | 1586 | -2.13e+04 | -0.3423 | 0.8875 | -13.43 | 0.4855 | -2.603e+04 |
| NQ | OOS | 672 | 6.327e+04 | 0.8571 | 1.272 | 94.15 | 0.5164 | -2.739e+04 |
| NQ | FULL | 2258 | 4.197e+04 | 0.2721 | 1.099 | 18.59 | 0.4947 | -4.402e+04 |

## Executive findings

- The OOS effect is strong across all four indices, but IS is negative for most cells. This is temporal instability, not four independent confirmations.
- Signal dates and directions are highly shared; the evidence is primarily one common US equity-gap event process.
- Larger normalized gaps are diagnosed without threshold retuning; see the dose-response table below.
- Cost and concentration tests are candidate-specific. The exact rows are preserved in CSV, including delayed entries, best-day and best-year removal.
- NQ is excluded from executable Topstep comparisons. No tenth sleeve is proposed.

## Temporal stability

| asset | period | trades | net_pnl | sharpe | profit_factor | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- |
| M2K | IS | 334 | 448.3 | 0.1563 | 1.051 | -1427 |
| M2K | OOS | 168 | 2072 | 1.533 | 1.599 | -501 |
| M2K | pre_2020 | 45 | 169.5 | 0.7699 | 1.257 | -91 |
| M2K | 2020_2022 | 219 | -378.5 | -0.1833 | 0.9455 | -1427 |
| M2K | 2023_plus | 238 | 2729 | 1.456 | 1.589 | -501 |
| MES | IS | 432 | -2022 | -0.4507 | 0.8767 | -3265 |
| MES | OOS | 169 | 4059 | 1.459 | 1.602 | -1353 |
| MES | pre_2020 | 64 | 1.25 | 0.004133 | 1.001 | -218.8 |
| MES | 2020_2022 | 282 | -2445 | -0.7481 | 0.8085 | -3265 |
| MES | 2023_plus | 255 | 4481 | 1.215 | 1.485 | -1487 |
| MNQ | IS | 392 | -1906 | -0.2664 | 0.9218 | -5948 |
| MNQ | OOS | 177 | 8215 | 1.663 | 1.643 | -1977 |
| MNQ | pre_2020 | 62 | 131.5 | 0.3388 | 1.1 | -324 |
| MNQ | 2020_2022 | 250 | -3338 | -0.6542 | 0.8239 | -5948 |
| MNQ | 2023_plus | 257 | 9515 | 1.44 | 1.564 | -1977 |
| NQ | IS | 850 | -3.317e+04 | -0.41 | 0.8511 | -3.655e+04 |
| NQ | OOS | 366 | 9.193e+04 | 0.996 | 1.362 | -3.036e+04 |
| NQ | pre_2020 | 710 | -1.65e+04 | -0.4156 | 0.8684 | -2.293e+04 |
| NQ | 2020_2022 | 250 | -2.724e+04 | -0.5335 | 0.8541 | -5.443e+04 |
| NQ | 2023_plus | 256 | 1.025e+05 | 1.552 | 1.624 | -1.861e+04 |

## Gap-size dose response

| asset | gap_bucket | events | prob_fill_50 | prob_fill_100 | median_time_fill_100_min | avg_mae_usd | avg_return_120m_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | 0.25_0.50 | 470 | 0.6809 | 0.5277 | 25 | 77.66 | -4.226 |
| M2K | 0.50_0.75 | 246 | 0.6667 | 0.4106 | 60 | 74.78 | 8.301 |
| M2K | 0.75_1.00 | 124 | 0.6129 | 0.2984 | 60 | 65.65 | 12.91 |
| M2K | 1.00_plus | 132 | 0.447 | 0.1818 | 80 | 75.8 | 9.761 |
| MES | 0.25_0.50 | 442 | 0.7308 | 0.5724 | 35 | 110.4 | 3.75 |
| MES | 0.50_0.75 | 255 | 0.6196 | 0.4118 | 70 | 109.7 | 2.775 |
| MES | 0.75_1.00 | 155 | 0.6 | 0.2903 | 125 | 118.6 | 12.99 |
| MES | 1.00_plus | 191 | 0.4764 | 0.1832 | 115 | 126.8 | 11.45 |
| MNQ | 0.25_0.50 | 446 | 0.7108 | 0.5179 | 20 | 203.2 | -12.71 |
| MNQ | 0.50_0.75 | 274 | 0.6168 | 0.4307 | 62.5 | 188.9 | 4.969 |
| MNQ | 0.75_1.00 | 126 | 0.5556 | 0.2937 | 65 | 197.7 | 52.23 |
| MNQ | 1.00_plus | 169 | 0.497 | 0.2367 | 97.5 | 218.7 | 15.7 |
| NQ | 0.25_0.50 | 1046 | 0.7285 | 0.5545 | 20 | 1127 | -41.63 |
| NQ | 0.50_0.75 | 578 | 0.628 | 0.4343 | 55 | 1154 | 3.183 |
| NQ | 0.75_1.00 | 300 | 0.5 | 0.2967 | 70 | 1136 | 185.5 |
| NQ | 1.00_plus | 338 | 0.5178 | 0.2426 | 110 | 1318 | 108.6 |

## Long-gap versus short-gap

| asset | scope | gap_side | trades | net_pnl | sharpe | profit_factor | expectancy | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | IS | negative_gap_long | 157 | 364.6 | 0.1733 | 1.081 | 2.323 | 0.5223 |
| M2K | IS | positive_gap_short | 177 | 83.68 | 0.04311 | 1.019 | 0.4728 | 0.4802 |
| M2K | OOS | negative_gap_long | 78 | 1619 | 1.601 | 2.178 | 20.75 | 0.6154 |
| M2K | OOS | positive_gap_short | 90 | 452.8 | 0.5051 | 1.217 | 5.031 | 0.5222 |
| M2K | FULL | negative_gap_long | 235 | 1983 | 0.6347 | 1.339 | 8.44 | 0.5532 |
| M2K | FULL | positive_gap_short | 267 | 536.5 | 0.1886 | 1.084 | 2.009 | 0.4944 |
| MES | IS | negative_gap_long | 201 | -1018 | -0.3137 | 0.8772 | -5.064 | 0.5174 |
| MES | IS | positive_gap_short | 231 | -1004 | -0.3255 | 0.8762 | -4.346 | 0.4329 |
| MES | OOS | negative_gap_long | 78 | 2524 | 1.121 | 1.712 | 32.36 | 0.6538 |
| MES | OOS | positive_gap_short | 91 | 1535 | 0.9376 | 1.479 | 16.87 | 0.5165 |
| MES | FULL | negative_gap_long | 279 | 1506 | 0.2658 | 1.127 | 5.397 | 0.5556 |
| MES | FULL | positive_gap_short | 322 | 531.4 | 0.1118 | 1.047 | 1.65 | 0.4565 |
| MNQ | IS | negative_gap_long | 178 | -2434 | -0.4757 | 0.8058 | -13.68 | 0.4831 |
| MNQ | IS | positive_gap_short | 214 | 528.2 | 0.1062 | 1.045 | 2.468 | 0.4533 |
| MNQ | OOS | negative_gap_long | 77 | 4079 | 1.095 | 1.637 | 52.97 | 0.6364 |
| MNQ | OOS | positive_gap_short | 100 | 4136 | 1.272 | 1.649 | 41.36 | 0.52 |
| MNQ | FULL | negative_gap_long | 255 | 1644 | 0.1793 | 1.087 | 6.448 | 0.5294 |
| MNQ | FULL | positive_gap_short | 314 | 4664 | 0.5542 | 1.256 | 14.85 | 0.4745 |
| NQ | IS | negative_gap_long | 365 | -1.07e+04 | -0.1911 | 0.8897 | -29.3 | 0.5233 |
| NQ | IS | positive_gap_short | 485 | -2.247e+04 | -0.386 | 0.8213 | -46.33 | 0.433 |
| NQ | OOS | negative_gap_long | 177 | 3.257e+04 | 0.4705 | 1.232 | 184 | 0.5424 |
| NQ | OOS | positive_gap_short | 189 | 5.936e+04 | 0.9732 | 1.524 | 314.1 | 0.4974 |
| NQ | FULL | negative_gap_long | 542 | 2.188e+04 | 0.1525 | 1.092 | 40.36 | 0.5295 |
| NQ | FULL | positive_gap_short | 674 | 3.689e+04 | 0.2803 | 1.154 | 54.73 | 0.451 |

## Cross-asset dependence

| asset_a | asset_b | signal_date_correlation | common_signal_dates | jaccard_signal_dates | overlap_vs_smaller_signal_set | direction_agreement | daily_trade_pnl_correlation | large_gap_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | MES | 0.5521 | 397 | 0.5623 | 0.7908 | 0.995 | 0.5912 | 0.4719 |
| M2K | MNQ | 0.4348 | 345 | 0.4752 | 0.6873 | 0.9884 | 0.4639 | 0.3741 |
| M2K | NQ | -0.3363 | 345 | 0.2513 | 0.6873 | 0.9884 | 0.452 | 0.2049 |
| MES | MNQ | 0.7029 | 482 | 0.7006 | 0.8471 | 1 | 0.8171 | 0.5866 |
| MES | NQ | -0.1154 | 482 | 0.361 | 0.802 | 1 | 0.7928 | 0.319 |
| MNQ | NQ | 0.333 | 567 | 0.4655 | 0.9965 | 1 | 0.9682 | 0.4624 |

| unique_event_dates | asset_level_signals | mean_assets_per_event | four_asset_event_fraction | effective_observation_ratio |
| --- | --- | --- | --- | --- |
| 1430 | 2888 | 2.02 | 0.2329 | 0.4952 |

## Event-level pooled analysis

| relative_gap_bucket | observations | prob_fill_50 | prob_fill_100 | avg_return_120m_usd | avg_gap_norm |
| --- | --- | --- | --- | --- | --- |
| small_relative | 482 | 0.612 | 0.3672 | 10.93 | 0.8292 |
| middle_relative | 639 | 0.5587 | 0.3412 | 13.87 | 0.9198 |
| large_relative | 1767 | 0.5659 | 0.322 | 56.87 | 1.043 |

## Pre-entry regime diagnosis

| asset | variable | bucket | events | bucket_min | bucket_max | prob_full_reversion | prob_50_reversion | avg_mae_usd | median_time_to_mean_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | prior_rv_pct | low | 162 | 0.003968 | 0.2222 | 0.3951 | 0.6852 | 60.38 | 30 |
| M2K | prior_rv_pct | mid | 164 | 0.2302 | 0.5516 | 0.2988 | 0.5549 | 74.46 | 70 |
| M2K | prior_rv_pct | high | 160 | 0.5556 | 1 | 0.2625 | 0.5563 | 86.7 | 90 |
| M2K | intraday_vol_ratio | low | 168 | 0.1362 | 0.407 | 0.2083 | 0.5417 | 74.76 | 90 |
| M2K | intraday_vol_ratio | mid | 166 | 0.4072 | 0.6174 | 0.3675 | 0.5843 | 71.52 | 75 |
| M2K | intraday_vol_ratio | high | 168 | 0.6234 | 6.977 | 0.3929 | 0.6607 | 72.07 | 30 |
| M2K | efficiency | low | 167 | 0 | 0.4604 | 0.2635 | 0.5389 | 78.7 | 90 |
| M2K | efficiency | mid | 168 | 0.4615 | 1.012 | 0.3333 | 0.631 | 75.04 | 70 |
| M2K | efficiency | high | 167 | 1.017 | 12.4 | 0.3713 | 0.6168 | 64.62 | 20 |
| MES | prior_rv_pct | low | 192 | 0.003968 | 0.2112 | 0.3958 | 0.6979 | 83.52 | 67.5 |
| MES | prior_rv_pct | mid | 192 | 0.2121 | 0.5635 | 0.3281 | 0.5521 | 101.3 | 105 |
| MES | prior_rv_pct | high | 192 | 0.5675 | 1 | 0.2083 | 0.4479 | 176.1 | 120 |
| MES | intraday_vol_ratio | low | 200 | 0.1181 | 0.3267 | 0.18 | 0.42 | 112.6 | 142.5 |
| MES | intraday_vol_ratio | mid | 200 | 0.3276 | 0.4776 | 0.335 | 0.605 | 130.1 | 100 |
| MES | intraday_vol_ratio | high | 201 | 0.4792 | 5.909 | 0.408 | 0.6816 | 109.5 | 67.5 |
| MES | efficiency | low | 202 | 0 | 0.4667 | 0.2673 | 0.5149 | 126.2 | 87.5 |
| MES | efficiency | mid | 204 | 0.4694 | 1 | 0.2892 | 0.5882 | 116.4 | 105 |
| MES | efficiency | high | 194 | 1.014 | 32 | 0.3711 | 0.6031 | 109.5 | 62.5 |
| MNQ | prior_rv_pct | low | 182 | 0.003968 | 0.2222 | 0.4615 | 0.6978 | 161.4 | 65 |
| MNQ | prior_rv_pct | mid | 180 | 0.2245 | 0.5635 | 0.3278 | 0.5278 | 184.5 | 65 |
| MNQ | prior_rv_pct | high | 182 | 0.5668 | 1 | 0.2363 | 0.467 | 272.3 | 115 |
| MNQ | intraday_vol_ratio | low | 190 | 0.1263 | 0.3754 | 0.2158 | 0.4895 | 205.1 | 140 |
| MNQ | intraday_vol_ratio | mid | 189 | 0.3778 | 0.5706 | 0.3598 | 0.5767 | 196.1 | 77.5 |
| MNQ | intraday_vol_ratio | high | 190 | 0.5714 | 5.471 | 0.4526 | 0.6368 | 197.8 | 42.5 |
| MNQ | efficiency | low | 190 | 0.002882 | 0.461 | 0.2684 | 0.5158 | 226.4 | 85 |
| MNQ | efficiency | mid | 189 | 0.462 | 1.091 | 0.381 | 0.619 | 184.2 | 62.5 |
| MNQ | efficiency | high | 190 | 1.099 | 5.833 | 0.3789 | 0.5684 | 188.4 | 67.5 |
| NQ | prior_rv_pct | low | 400 | 0.003968 | 0.1984 | 0.425 | 0.6475 | 889 | 47.5 |
| NQ | prior_rv_pct | mid | 405 | 0.2 | 0.5159 | 0.3481 | 0.563 | 998.4 | 70 |
| NQ | prior_rv_pct | high | 396 | 0.5198 | 1 | 0.2626 | 0.4823 | 1741 | 107.5 |
| NQ | intraday_vol_ratio | low | 406 | 0.08696 | 0.3781 | 0.2438 | 0.4828 | 1264 | 125 |
| NQ | intraday_vol_ratio | mid | 405 | 0.3781 | 0.573 | 0.358 | 0.5728 | 1177 | 70 |
| NQ | intraday_vol_ratio | high | 405 | 0.5732 | 5.471 | 0.4395 | 0.642 | 1145 | 30 |
| NQ | efficiency | low | 405 | 0 | 0.4574 | 0.2815 | 0.5111 | 1321 | 85 |
| NQ | efficiency | mid | 405 | 0.4615 | 1.06 | 0.3778 | 0.6222 | 1127 | 60 |
| NQ | efficiency | high | 406 | 1.061 | 5.516 | 0.3818 | 0.564 | 1138 | 55 |

Interpretation rule: falling fill probability indicates a genuine regime problem; stable fill probability with rising MAE indicates primarily a volatility/stop-width problem. No filter is created.

## Exit decomposition

| asset | exit_kind | trades | net_pnl | sharpe | profit_factor | expectancy | max_drawdown | avg_holding_minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | fair_value | 168 | 2072 | 1.533 | 1.599 | 12.33 | -501 | 99.7 |
| M2K | partial_50 | 168 | 1798 | 1.582 | 1.59 | 10.7 | -419.3 | 75.03 |
| M2K | time_30 | 168 | 978 | 0.9512 | 1.355 | 5.821 | -771.5 | 35 |
| M2K | slow_120 | 168 | 2384 | 1.59 | 1.603 | 14.19 | -786 | 125 |
| MES | fair_value | 169 | 4059 | 1.459 | 1.602 | 24.02 | -1353 | 102.4 |
| MES | partial_50 | 169 | 3152 | 1.429 | 1.522 | 18.65 | -1348 | 76.51 |
| MES | time_30 | 169 | 667.5 | 0.3762 | 1.133 | 3.95 | -1092 | 35 |
| MES | slow_120 | 169 | 4190 | 1.5 | 1.585 | 24.79 | -1064 | 125 |
| MNQ | fair_value | 177 | 8215 | 1.663 | 1.643 | 46.41 | -1977 | 103.7 |
| MNQ | partial_50 | 177 | 6110 | 1.533 | 1.529 | 34.52 | -1840 | 76.84 |
| MNQ | time_30 | 177 | 4630 | 1.378 | 1.554 | 26.16 | -1262 | 35 |
| MNQ | slow_120 | 177 | 8782 | 1.678 | 1.641 | 49.62 | -2036 | 125 |
| NQ | fair_value | 366 | 9.193e+04 | 0.996 | 1.362 | 251.2 | -3.036e+04 | 101.5 |
| NQ | partial_50 | 366 | 5.242e+04 | 0.6871 | 1.222 | 143.2 | -2.924e+04 | 78.57 |
| NQ | time_30 | 366 | 3.147e+04 | 0.4922 | 1.168 | 85.98 | -2.853e+04 | 35 |
| NQ | slow_120 | 366 | 1.101e+05 | 1.09 | 1.403 | 300.7 | -2.43e+04 | 125 |

## Diversification

| asset | corr_9_sleeve | corr_negative_days | mean_candidate_worst_5pct | mean_candidate_worst_10pct | base_sharpe | added_sharpe | base_maxdd | added_maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | 0.1024 | -0.1341 | 41.65 | 32.45 | 0.8764 | 1.048 | -4913 | -4558 |
| MES | 0.05161 | -0.3244 | 142.9 | 83.55 | 1.279 | 1.523 | -3705 | -4002 |
| MNQ | 0.01396 | -0.2914 | 223.3 | 150.6 | 0.8003 | 1.326 | -4614 | -4137 |
| NQ | 0.0137 | -0.2887 | 2255 | 1529 | 0.8025 | 1.808 | -4587 | -1.548e+04 |

## Paired Topstep bootstrap — executable micros only

| asset | metric | paths | estimate | ci95_low | ci95_high | baseline_probability | candidate_probability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | delta_pass | 8000 | 0.00575 | 0.002 | 0.009375 | 0.4524 | 0.4581 |
| M2K | delta_mll | 8000 | -0.000625 | -0.00425 | 0.003125 | 0.4811 | 0.4805 |
| MES | delta_pass | 8000 | 0.00225 | -0.003 | 0.0075 | 0.4524 | 0.4546 |
| MES | delta_mll | 8000 | 0.000125 | -0.00475 | 0.00525 | 0.4811 | 0.4813 |
| MNQ | delta_pass | 8000 | -0.00125 | -0.008875 | 0.006375 | 0.4524 | 0.4511 |
| MNQ | delta_mll | 8000 | 0.01613 | 0.00825 | 0.02375 | 0.4811 | 0.4973 |

A point estimate is not treated as improvement unless the paired 95% interval excludes zero. NQ is excluded by construction.

## Final assessment

The effect is economically plausible and repeatedly visible OOS, but the historical split reveals a substantial post-2023 concentration, while the four assets largely duplicate the same event dates. The correct unit of evidence is the pooled equity-gap event, not four contracts. The mechanism therefore remains real enough for prospective observation but is not yet justified as an executable tenth sleeve.

**Decision: `REAL_EFFECT_NOT_YET_EXECUTABLE`.** Freeze the mechanism and regions for prospective monitoring only; do not select an asset or exact threshold by Sharpe.