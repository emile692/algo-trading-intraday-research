# Price Move Without Volume Confirmation Audit

**Global economic-mechanism classification: `REJECTED`**

This is a frozen-region audit. No new parameter, asset, threshold, time filter, weight, or contract count was selected. The frozen nine sleeves were read only.

## 1. Exact inherited mechanism

- Five-minute regular-session OHLCV bars in America/New_York time.
- Price move: `abs((close-open)/ATR)`, with ATR equal to the median true range of the prior 10-20 completed bars within that session (10-bar minimum).
- Large-move region: 1.0, 1.5, and 2.0 ATR; inherited representatives use 1.5 ATR.
- Expected volume: median volume of the prior 20 completed sessions at the same five-minute clock slot; `RVOL = volume / expected volume`.
- No confirmation: RVOL <= 0.75 or <= 1.0; the representative threshold is asset-specific below.
- Signal at completed-bar close; entry at next bar open; exit at the close of the sixth holding bar, truncated at session end; trades do not overlap.
- Nominal cost: one adverse tick per side plus commission. Stress adds one or two further ticks per side or doubles commission.

| asset | variant_id | mode | move | rvol |
| --- | --- | --- | --- | --- |
| M2K | E__fade__n1__m1.5__rv1.0__p0__c1 | fade | 1.5 | 1 |
| MES | E__fade__n1__m1.5__rv0.75__p0__c1 | fade | 1.5 | 0.75 |
| MNQ | E__continuation_control__n1__m1.5__rv1.0__p0__c1 | continuation_control | 1.5 | 1 |
| NQ | E__fade__n1__m1.5__rv1.0__p0__c1 | fade | 1.5 | 1 |
| MGC | E__continuation_control__n1__m1.5__rv1.0__p0__c1 | continuation_control | 1.5 | 1 |
| MBT | E__continuation_control__n1__m1.5__rv1.0__p0__c1 | continuation_control | 1.5 | 1 |

All 144 inherited IS/OOS rows are preserved in `inherited_region_cells.csv`: 12 exact variants per asset (3 move thresholds x 2 RVOL thresholds x fade/continuation control), each with IS and OOS rows. No cell was added.

## 2. Price move x volume 2x2 (OOS)

| asset | group | events | forward_return_usd | reversal_probability | continuation_probability | mae_usd | mfe_usd | forward_realized_vol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | A_large_high | 748 | -1.396 | 0.5348 | 0.4572 | 30.55 | 30.24 | 0.003401 |
| M2K | B_large_low_normal | 169 | -5.426 | 0.5621 | 0.426 | 23.86 | 18.36 | 0.002345 |
| MES | A_large_high | 1039 | 0.4066 | 0.5014 | 0.487 | 50.08 | 51.46 | 0.002054 |
| MES | B_large_low_normal | 89 | -5.73 | 0.5281 | 0.4494 | 49.76 | 41.66 | 0.001793 |
| MNQ | A_large_high | 831 | 3.666 | 0.5018 | 0.4922 | 89.77 | 97.14 | 0.002634 |
| MNQ | B_large_low_normal | 135 | -2.63 | 0.5556 | 0.4444 | 91.19 | 82.19 | 0.0024 |
| NQ | A_large_high | 1933 | 54.54 | 0.4982 | 0.5003 | 763.7 | 855.3 | 0.002862 |
| NQ | B_large_low_normal | 333 | -76.38 | 0.5315 | 0.4685 | 805.9 | 722.2 | 0.002863 |
| MGC | A_large_high | 1153 | -1.829 | 0.5325 | 0.4605 | 69.15 | 76.62 | 0.002522 |
| MGC | B_large_low_normal | 160 | 64.89 | 0.4375 | 0.5437 | 77.83 | 156.2 | 0.003141 |
| MBT | A_large_high | 801 | -0.3059 | 0.5318 | 0.4594 | 32.61 | 34.54 | 0.004602 |
| MBT | B_large_low_normal | 136 | 1.316 | 0.4632 | 0.5294 | 20.53 | 25.24 | 0.003272 |

The forward return is signed in the direction of the triggering price bar: negative means reversal and positive means continuation. Thus the A/B comparison is independent of each representative's inherited trade direction.

## 3. Matched price-only benchmark

| asset | draws | observed_trades | benchmark_trades | observed_sharpe | sharpe_percentile | observed_expectancy | expectancy_percentile | observed_pf | pf_percentile | prob_random_matches_exceeds_sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | 1000 | 135 | 638 | 1.353 | 95.9 | 2.396 | 95 | 1.305 | 95.7 | 0.041 |
| MES | 1000 | 66 | 773 | 1.582 | 94.1 | 5.436 | 93.5 | 1.351 | 94.7 | 0.059 |
| MNQ | 1000 | 102 | 682 | 0.4737 | 63.5 | 2.598 | 61 | 1.096 | 63.8 | 0.365 |
| NQ | 1000 | 253 | 1631 | 0.7959 | 91.1 | 45.84 | 89.5 | 1.179 | 92.1 | 0.089 |
| MGC | 1000 | 132 | 947 | 1.768 | 97.6 | 29.86 | 98.5 | 1.996 | 98.6 | 0.024 |
| MBT | 1000 | 118 | 675 | -0.5911 | 52.5 | -0.9853 | 59 | 0.9134 | 54.9 | 0.475 |

Random thinning is OOS, without replacement, and exactly matched to the volume-filtered trade count for 1,000 draws per asset.

## 4. Cross-asset replication and classifications

| asset | variant_id | trades | sharpe | pf | expectancy | maxdd | region_stability | classification | beats_thinning | plus2_cost_positive | positive_oos_year_fraction | remove_best_10_positive | large_low_minus_high_forward_usd | meaningful_2x2_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | E__fade__n1__m1.5__rv1.0__p0__c1 | 135 | 1.353 | 1.305 | 2.396 | -251.5 | LOCAL_REGION | WEAK_SUPPORT | True | True | 0.6667 | False | -4.03 | True |
| MES | E__fade__n1__m1.5__rv0.75__p0__c1 | 66 | 1.582 | 1.351 | 5.436 | -243.8 | LOCAL_REGION | NO_EFFECT | False | True | 0.3333 | False | -6.137 | True |
| MNQ | E__continuation_control__n1__m1.5__rv1.0__p0__c1 | 102 | 0.4737 | 1.096 | 2.598 | -710.5 | LOCAL_REGION | NO_EFFECT | False | True | 0.3333 | False | -6.296 | True |
| NQ | E__fade__n1__m1.5__rv1.0__p0__c1 | 253 | 0.7959 | 1.179 | 45.84 | -1.191e+04 | BROAD_REGION | NO_EFFECT | False | True | 0.5 | False | -130.9 | True |
| MGC | E__continuation_control__n1__m1.5__rv1.0__p0__c1 | 132 | 1.768 | 1.996 | 29.86 | -1105 | BROAD_REGION | OPPOSITE_EFFECT | True | True | 0.5 | False | 66.72 | True |
| MBT | E__continuation_control__n1__m1.5__rv1.0__p0__c1 | 118 | -0.5911 | 0.9134 | -0.9853 | -318.4 | FRAGILE | OPPOSITE_EFFECT | False | False | 0.5 | False | 1.622 | True |

| record_type | asset_a | asset_b | common_event_dates | event_date_jaccard | overlap_vs_smaller_set | direction_overlap | pnl_correlation | unique_event_count | related_equity_contracts | asset_level_event_count | effective_distinct_event_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pair_overlap | M2K | MES | 30 | 0.2586 | 0.7143 | 0.6667 | 0.7531 | 86 | True |  |  |
| pair_overlap | M2K | MNQ | 47 | 0.3431 | 0.5875 | 0.5957 | 0.04933 | 90 | True |  |  |
| pair_overlap | M2K | NQ | 41 | 0.1621 | 0.3942 | 0.6829 | -0.1871 | 212 | True |  |  |
| pair_overlap | M2K | MGC | 13 | 0.06373 | 0.125 | 0.3846 | -0.1835 | 191 | False |  |  |
| pair_overlap | M2K | MBT | 18 | 0.1011 | 0.1957 | 0.5 | -0.08632 | 160 | False |  |  |
| pair_overlap | MES | MNQ | 36 | 0.4186 | 0.8571 | 0.8056 | -0.2898 | 50 | True |  |  |
| pair_overlap | MES | NQ | 34 | 0.1717 | 0.8095 | 0.8235 | 0.2171 | 164 | True |  |  |
| pair_overlap | MES | MGC | 6 | 0.04027 | 0.1429 | 0.1667 |  | 143 | False |  |  |
| pair_overlap | MES | MBT | 3 | 0.0229 | 0.07143 | 0 |  | 128 | False |  |  |
| pair_overlap | MNQ | NQ | 66 | 0.3235 | 0.825 | 0.9545 | -0.9887 | 138 | True |  |  |
| pair_overlap | MNQ | MGC | 10 | 0.05464 | 0.125 | 0.4 | 0.1859 | 173 | False |  |  |
| pair_overlap | MNQ | MBT | 7 | 0.04242 | 0.0875 | 0.4286 |  | 158 | False |  |  |
| pair_overlap | NQ | MGC | 14 | 0.04844 | 0.1239 | 0.4286 | -0.08248 | 275 | False |  |  |
| pair_overlap | NQ | MBT | 10 | 0.03676 | 0.1087 | 0.6 | 0.3178 | 262 | False |  |  |
| pair_overlap | MGC | MBT | 18 | 0.09626 | 0.1957 | 0.5 | -0.01415 | 169 | False |  |  |
| distinct_event_summary | US_EQUITY_INDICES |  | 22 |  |  |  |  | 258 |  | 416 | 0.6202 |

MNQ and NQ are explicitly treated as related expressions, not independent confirmations. The distinct-event summary counts the union of M2K/MES/MNQ/NQ dates rather than summing contracts.

## 5. Asset-class interpretation

- US equity indices: M2K=WEAK_SUPPORT, MES=NO_EFFECT, MNQ=NO_EFFECT, NQ=NO_EFFECT.
- Gold: MGC=OPPOSITE_EFFECT.
- Bitcoin: MBT=OPPOSITE_EFFECT.

The classes are kept separate; contracts are not pooled into a synthetic cross-asset Sharpe.

## 6. Temporal stability

| asset | period | trades | net_pnl | sharpe | pf | expectancy | maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | IS | 375 | -1392 | -1.794 | 0.6753 | -3.712 | -1663 |
| M2K | OOS | 135 | 323.5 | 1.353 | 1.305 | 2.396 | -251.5 |
| M2K | pre_2020 | 70 | -145.5 | -3.915 | 0.6321 | -2.079 | -132 |
| M2K | 2020_2022 | 252 | -769 | -1.336 | 0.7575 | -3.052 | -1572 |
| M2K | 2023_plus | 188 | -154 | -0.4066 | 0.9135 | -0.8191 | -574.5 |
| MES | IS | 253 | 1415 | 1.359 | 1.401 | 5.593 | -973.8 |
| MES | OOS | 66 | 358.8 | 1.582 | 1.351 | 5.436 | -243.8 |
| MES | pre_2020 | 60 | -341.2 | -5.585 | 0.3666 | -5.688 | -326.2 |
| MES | 2020_2022 | 165 | 2016 | 2.475 | 1.792 | 12.22 | -973.8 |
| MES | 2023_plus | 94 | 98.75 | 0.34 | 1.068 | 1.051 | -393.8 |
| MNQ | IS | 357 | -4279 | -2.302 | 0.5805 | -11.99 | -4542 |
| MNQ | OOS | 102 | 265 | 0.4737 | 1.096 | 2.598 | -710.5 |
| MNQ | pre_2020 | 69 | -153 | -2.129 | 0.7238 | -2.217 | -199 |
| MNQ | 2020_2022 | 255 | -4298 | -2.803 | 0.5178 | -16.86 | -4361 |
| MNQ | 2023_plus | 135 | 437.5 | 0.6392 | 1.125 | 3.241 | -710.5 |
| NQ | IS | 965 | 7942 | 0.33 | 1.101 | 8.231 | -9738 |
| NQ | OOS | 253 | 1.16e+04 | 0.7959 | 1.179 | 45.84 | -1.191e+04 |
| NQ | pre_2020 | 826 | -9240 | -1.228 | 0.8098 | -11.19 | -9325 |
| NQ | 2020_2022 | 252 | 3.305e+04 | 2.1 | 1.606 | 131.2 | -9158 |
| NQ | 2023_plus | 140 | -4270 | -0.5867 | 0.8938 | -30.5 | -1.191e+04 |
| MGC | IS | 516 | -2745 | -3.696 | 0.5066 | -5.32 | -3047 |
| MGC | OOS | 132 | 3941 | 1.768 | 1.996 | 29.86 | -1105 |
| MGC | pre_2020 | 375 | -2076 | -5.865 | 0.3779 | -5.535 | -2096 |
| MGC | 2020_2022 | 132 | -587 | -1.941 | 0.7191 | -4.447 | -941 |
| MGC | 2023_plus | 141 | 3858 | 1.686 | 1.942 | 27.37 | -1105 |
| MBT | IS | 595 | -1552 | -2.533 | 0.6543 | -2.608 | -1660 |
| MBT | OOS | 118 | -116.3 | -0.5911 | 0.9134 | -0.9853 | -318.4 |
| MBT | pre_2020 | 0 |  |  |  |  |  |
| MBT | 2020_2022 | 298 | -658.9 | -2.006 | 0.7095 | -2.211 | -675.9 |
| MBT | 2023_plus | 415 | -1009 | -2.051 | 0.7167 | -2.432 | -1079 |

Yearly, rolling-12-month, best-year removal, best-5-trade removal, best-10-trade removal, and annual volume-signal stability are in `temporal_stability.csv`.

The positive OOS appearances are not persistent: M2K, MNQ, and MGC have negative IS histories; NQ is concentrated in 2020-2022 and is negative in 2023+; and every asset is negative after removing its ten best OOS trades.

## 7. Volume-signal stability

| record_type | asset | year | events | median_rvol | median_volume_percentile | median_expected_volume | signal_frequency_per_session |
| --- | --- | --- | --- | --- | --- | --- | --- |
| volume_signal_stability | M2K | 2024 | 65 | 0.6046 | 0.2 | 371 | 0.2995 |
| volume_signal_stability | M2K | 2025 | 91 | 0.6698 | 0.2 | 411 | 0.3597 |
| volume_signal_stability | M2K | 2026 | 13 | 0.1664 | 0 | 555 | 0.2364 |
| volume_signal_stability | MES | 2024 | 55 | 0.4135 | 0.1 | 6138 | 0.2535 |
| volume_signal_stability | MES | 2025 | 21 | 0.2718 | 0.05 | 1.084e+04 | 0.083 |
| volume_signal_stability | MES | 2026 | 13 | 0.3709 | 0.1 | 9634 | 0.2364 |
| volume_signal_stability | MNQ | 2024 | 70 | 0.673 | 0.15 | 8493 | 0.3226 |
| volume_signal_stability | MNQ | 2025 | 52 | 0.7088 | 0.2 | 1.443e+04 | 0.2055 |
| volume_signal_stability | MNQ | 2026 | 13 | 0.5195 | 0.1 | 1.166e+04 | 0.2364 |
| volume_signal_stability | NQ | 2021 | 46 | 0.6142 | 0.15 | 3212 | 0.3594 |
| volume_signal_stability | NQ | 2022 | 98 | 0.791 | 0.2 | 4630 | 0.3858 |
| volume_signal_stability | NQ | 2023 | 47 | 0.5395 | 0.15 | 4164 | 0.1858 |
| volume_signal_stability | NQ | 2024 | 75 | 0.646 | 0.1 | 4224 | 0.2941 |
| volume_signal_stability | NQ | 2025 | 57 | 0.6779 | 0.15 | 4970 | 0.2253 |
| volume_signal_stability | NQ | 2026 | 10 | 0.2947 | 0.1 | 3195 | 0.1818 |
| volume_signal_stability | MGC | 2023 | 41 | 0.7008 | 0.25 | 570 | 0.201 |
| volume_signal_stability | MGC | 2024 | 40 | 0.6566 | 0.15 | 874.2 | 0.1544 |
| volume_signal_stability | MGC | 2025 | 35 | 0.7751 | 0.2 | 1732 | 0.1357 |
| volume_signal_stability | MGC | 2026 | 44 | 0.6375 | 0.075 | 3326 | 0.2353 |
| volume_signal_stability | MBT | 2025 | 73 | 0.754 | 0.35 | 280 | 0.3244 |
| volume_signal_stability | MBT | 2026 | 63 | 0.6785 | 0.3 | 208.5 | 0.3214 |

The qualifying RVOL distribution and event frequency move materially through time, while expected-volume levels also shift with contract liquidity (especially MGC). The filter is causal, but its absolute liquidity context is not stationary.

## 8. Time of day

| record_type | asset | bucket | trades | net_pnl | sharpe | pf | expectancy | win_rate | maxdd | reversal_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_of_day | M2K | open_0_2h | 11 | 46.5 | 3.246 | 1.705 | 4.227 | 0.5455 | -34 | 0.6364 |
| time_of_day | M2K | mid_2_4h | 28 | -144 | -4.787 | 0.4745 | -5.143 | 0.3214 | -165.5 | 0.3929 |
| time_of_day | M2K | late_4h_plus | 96 | 421 | 2.201 | 1.585 | 4.385 | 0.5417 | -178 | 0.6146 |
| time_of_day | MES | open_0_2h | 8 | 82.5 | 4.21 | 2.5 | 10.31 | 0.375 | -26.25 | 0.375 |
| time_of_day | MES | mid_2_4h | 18 | 108.8 | 2.646 | 1.596 | 6.042 | 0.4444 | -157.5 | 0.6111 |
| time_of_day | MES | late_4h_plus | 40 | 167.5 | 0.9479 | 1.214 | 4.188 | 0.425 | -366.2 | 0.5 |
| time_of_day | MNQ | open_0_2h | 9 | -239 | -5.588 | 0.2971 | -26.56 | 0.3333 | -269.5 | 0.6667 |
| time_of_day | MNQ | mid_2_4h | 25 | 137.5 | 1.35 | 1.253 | 5.5 | 0.64 | -299.5 | 0.36 |
| time_of_day | MNQ | late_4h_plus | 68 | 366.5 | 0.8253 | 1.194 | 5.39 | 0.3824 | -610 | 0.5588 |
| time_of_day | NQ | open_0_2h | 16 | 1585 | 2.04 | 1.454 | 99.06 | 0.5 | -1922 | 0.5 |
| time_of_day | NQ | mid_2_4h | 55 | 4302 | 1.818 | 1.386 | 78.23 | 0.5455 | -4495 | 0.5455 |
| time_of_day | NQ | late_4h_plus | 182 | 5710 | 0.4617 | 1.114 | 31.37 | 0.5 | -1.308e+04 | 0.5055 |
| time_of_day | MGC | open_0_2h | 95 | 2739 | 1.747 | 1.786 | 28.84 | 0.5158 | -1081 | 0.4421 |
| time_of_day | MGC | mid_2_4h | 19 | 1155 | 5.575 | 6.951 | 60.76 | 0.4737 | -194 | 0.3684 |
| time_of_day | MGC | late_4h_plus | 18 | 47 | 0.6565 | 1.168 | 2.611 | 0.2778 | -223 | 0.6111 |
| time_of_day | MBT | open_0_2h | 17 | -30.94 | -0.7804 | 0.8668 | -1.82 | 0.4118 | -225.2 | 0.4706 |
| time_of_day | MBT | mid_2_4h | 33 | -145.6 | -2.425 | 0.7038 | -4.411 | 0.4848 | -251.8 | 0.4545 |
| time_of_day | MBT | late_4h_plus | 68 | 60.24 | 0.49 | 1.097 | 0.8859 | 0.4706 | -254.8 | 0.4412 |

The result is not uniformly a midday effect: M2K is strongest late, while MGC's inherited continuation result is concentrated near the open. Bucket signs vary sharply, which is further evidence against a common stable mechanism. These are diagnostics only; no time filter was introduced.

## 9. Directional symmetry

| asset | direction | trades | net_pnl | sharpe | pf | expectancy | win_rate | maxdd | reversal_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | fade_long_after_down | 80 | 216.5 | 1.354 | 1.315 | 2.706 | 0.4875 | -237.5 | 0.55 |
| M2K | fade_short_after_up | 72 | 101 | 0.6024 | 1.146 | 1.403 | 0.4583 | -198.5 | 0.5278 |
| MES | fade_long_after_down | 34 | -60 | -0.7352 | 0.9104 | -1.765 | 0.5294 | -263.8 | 0.5882 |
| MES | fade_short_after_up | 40 | 330 | 1.75 | 1.463 | 8.25 | 0.35 | -295 | 0.45 |
| MNQ | fade_long_after_down | 53 | -1029 | -3.119 | 0.5307 | -19.42 | 0.4717 | -1254 | 0.5094 |
| MNQ | fade_short_after_up | 55 | 116.5 | 0.3339 | 1.071 | 2.118 | 0.4909 | -703.5 | 0.5273 |
| NQ | fade_long_after_down | 148 | -4265 | -0.5352 | 0.9022 | -28.82 | 0.5068 | -1.152e+04 | 0.5068 |
| NQ | fade_short_after_up | 130 | 2970 | 0.3573 | 1.069 | 22.85 | 0.4846 | -9082 | 0.4923 |
| MGC | fade_long_after_down | 66 | -1885 | -2.781 | 0.5921 | -28.56 | 0.4242 | -1868 | 0.4242 |
| MGC | fade_short_after_up | 78 | -1313 | -2.479 | 0.6506 | -16.83 | 0.4615 | -1946 | 0.4872 |
| MBT | fade_long_after_down | 63 | -703.7 | -5.913 | 0.3569 | -11.17 | 0.3492 | -685 | 0.3968 |
| MBT | fade_short_after_up | 64 | -73.48 | -0.6277 | 0.9025 | -1.148 | 0.4062 | -234.7 | 0.4844 |

## 10. Cost and latency stress

| asset | scenario | trades | net_pnl | sharpe | pf | expectancy | maxdd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | nominal | 135 | 323.5 | 1.353 | 1.305 | 2.396 | -251.5 |
| M2K | plus_1_tick_side | 135 | 188.5 | 0.7855 | 1.167 | 1.396 | -271.5 |
| M2K | plus_2_ticks_side | 135 | 53.5 | 0.2221 | 1.045 | 0.3963 | -291.5 |
| M2K | double_commission | 135 | -14 | -0.05802 | 0.9887 | -0.1037 | -301.5 |
| M2K | delayed_1_bar | 100 | 268.5 | 1.75 | 1.308 | 2.685 | -193.5 |
| M2K | delayed_2_bar | 93 | 321.5 | 2.63 | 1.404 | 3.457 | -110.5 |
| MES | nominal | 66 | 358.8 | 1.582 | 1.351 | 5.436 | -243.8 |
| MES | plus_1_tick_side | 66 | 193.8 | 0.8572 | 1.173 | 2.936 | -273.8 |
| MES | plus_2_ticks_side | 66 | 28.75 | 0.1276 | 1.024 | 0.4356 | -333.8 |
| MES | double_commission | 66 | 193.8 | 0.8572 | 1.173 | 2.936 | -273.8 |
| MES | delayed_1_bar | 56 | 157.5 | 1.003 | 1.158 | 2.812 | -290 |
| MES | delayed_2_bar | 52 | 200 | 1.32 | 1.212 | 3.846 | -385 |
| MNQ | nominal | 102 | 265 | 0.4737 | 1.096 | 2.598 | -710.5 |
| MNQ | plus_1_tick_side | 102 | 163 | 0.2917 | 1.058 | 1.598 | -740.5 |
| MNQ | plus_2_ticks_side | 102 | 61 | 0.1093 | 1.021 | 0.598 | -770.5 |
| MNQ | double_commission | 102 | 10 | 0.01793 | 1.003 | 0.09804 | -785.5 |
| MNQ | delayed_1_bar | 72 | 214 | 0.5043 | 1.102 | 2.972 | -1005 |
| MNQ | delayed_2_bar | 69 | 161 | 0.4804 | 1.075 | 2.333 | -479.5 |
| NQ | nominal | 253 | 1.16e+04 | 0.7959 | 1.179 | 45.84 | -1.191e+04 |
| NQ | plus_1_tick_side | 253 | 9068 | 0.6226 | 1.138 | 35.84 | -1.237e+04 |
| NQ | plus_2_ticks_side | 253 | 6538 | 0.4491 | 1.097 | 25.84 | -1.283e+04 |
| NQ | double_commission | 253 | 1.096e+04 | 0.7526 | 1.169 | 43.34 | -1.202e+04 |
| NQ | delayed_1_bar | 190 | 1.194e+04 | 1.309 | 1.225 | 62.87 | -1.178e+04 |
| NQ | delayed_2_bar | 184 | 2.379e+04 | 2.477 | 1.487 | 129.3 | -9140 |
| MGC | nominal | 132 | 3941 | 1.768 | 1.996 | 29.86 | -1105 |
| MGC | plus_1_tick_side | 132 | 3677 | 1.652 | 1.897 | 27.86 | -1273 |
| MGC | plus_2_ticks_side | 132 | 3413 | 1.535 | 1.804 | 25.86 | -1441 |
| MGC | double_commission | 132 | 3611 | 1.623 | 1.873 | 27.36 | -1315 |
| MGC | delayed_1_bar | 124 | 3344 | 2.419 | 2.028 | 26.97 | -1081 |
| MGC | delayed_2_bar | 122 | 924 | 1.411 | 1.254 | 7.574 | -1135 |
| MBT | nominal | 118 | -116.3 | -0.5911 | 0.9134 | -0.9853 | -318.4 |
| MBT | plus_1_tick_side | 118 | -234.3 | -1.19 | 0.8336 | -1.985 | -406.7 |
| MBT | plus_2_ticks_side | 118 | -352.3 | -1.788 | 0.7614 | -2.985 | -505.7 |
| MBT | double_commission | 118 | -449 | -2.276 | 0.707 | -3.805 | -586.9 |
| MBT | delayed_1_bar | 114 | -94.98 | -0.512 | 0.9192 | -0.8332 | -264.8 |
| MBT | delayed_2_bar | 110 | -330.2 | -2.015 | 0.7128 | -3.002 | -346.6 |

## 11. Economic interpretation

| asset | economic_path | A_high_volume_30m_usd | B_low_normal_volume_30m_usd | B_minus_A_30m_usd |
| --- | --- | --- | --- | --- |
| M2K | IMMEDIATE_REVERSAL_THROUGH_30M | -0.6733 | -7.513 | -6.84 |
| MES | IMMEDIATE_REVERSAL_THROUGH_30M | 1.233 | -7.034 | -8.267 |
| MNQ | IMMEDIATE_REVERSAL_THROUGH_30M | 6.452 | -7.634 | -14.09 |
| NQ | DELAYED_REVERSAL | 63.32 | -91.63 | -154.9 |
| MGC | STRONGER_LOW_VOLUME_CONTINUATION_OPPOSITE_HYPOTHESIS | -2.457 | 72.88 | 75.33 |
| MBT | STRONGER_LOW_VOLUME_CONTINUATION_OPPOSITE_HYPOTHESIS | -0.4532 | 1.448 | 1.901 |

The per-asset 1/3/6/12-bar A/B paths are in `event_studies/`. M2K, MES, and MNQ reverse immediately through 30 minutes; NQ shows delayed reversal. MGC and MBT instead show stronger continuation on low/normal volume, the opposite of the proposed reversal mechanism. Most reversal paths dissipate or flip by 60 minutes. This sign disagreement is why profitable asset-specific representatives do not establish one common cross-asset effect.

## 12. Diversification

| status |
| --- |
| NOT_RUN_NO_STANDALONE_SURVIVOR |

Diversification was run only for strict standalone survivors. An empty gated result is recorded explicitly rather than backfilling weaker assets.

## 13. Topstep

| status |
| --- |
| NOT_RUN_NO_EXECUTABLE_MICRO_SURVIVOR |

NQ is excluded as an execution candidate. The paired/common-path simulation is run only for strict executable micro survivors and does not optimize weights or contract counts.

## Final decision

| asset | classification | beats_thinning | plus2_cost_positive | positive_oos_year_fraction | remove_best_10_positive | large_low_minus_high_forward_usd | meaningful_2x2_difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2K | WEAK_SUPPORT | True | True | 0.6667 | False | -4.03 | True |
| MES | NO_EFFECT | False | True | 0.3333 | False | -6.137 | True |
| MNQ | NO_EFFECT | False | True | 0.3333 | False | -6.296 | True |
| NQ | NO_EFFECT | False | True | 0.5 | False | -130.9 | True |
| MGC | OPPOSITE_EFFECT | True | True | 0.5 | False | 66.72 | True |
| MBT | OPPOSITE_EFFECT | False | False | 0.5 | False | 1.622 | True |

Classification counts: `{'NO_EFFECT': 3, 'OPPOSITE_EFFECT': 2, 'WEAK_SUPPORT': 1}`.

**Final global classification: `REJECTED`.**