# MNQ ORB 3-State Variant Audit

## Objective

Audit why `median_plateau_compact` beats `single_15_60` in the existing smoke run,
without relaunching a broad campaign, and decide whether the edge looks broad-based or outlier-driven.

- Source run: `export\mnq_orb_3state_vol_sizing_variant_smoke_20260531_173822`

## Summary

| variant_name | sharpe | net_pnl | max_drawdown | monthly_hit_rate | rolling_20d_max_drawdown | rolling_60d_max_drawdown | avg_multiplier | bucket_switches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| median_plateau_compact | 2.688 | 31710.000 | -2774.000 | 0.590 | -4440.000 | -6484.500 | 0.653 | 563 |
| median_fast15_slow_60_70_80 | 2.562 | 30938.000 | -2774.000 | 0.602 | -4702.500 | -6219.500 | 0.655 | 561 |
| single_15_60 | 2.414 | 27959.000 | -2420.500 | 0.566 | -4056.500 | -7010.500 | 0.652 | 567 |

## Broad-Based Or Outlier-Driven

- median_plateau_compact broad-based improvement: `no`
- Positive diff day share vs single_15_60: `0.531`
- Positive diff month share vs single_15_60: `0.241`
- Top 3 positive contribution share: `0.289`
- Top 10 positive contribution share: `0.619`
- Excess PnL after removing top 5 positive diff days: `-4229.5`

## Baseline Vs Ensemble

- single_15_60: Sharpe `2.414`, net PnL `27959.0`, maxDD `-2420.5`
- median_plateau_compact: Sharpe `2.688`, net PnL `31710.0`, maxDD `-2774.0`
- Verdict: Stay with single_15_60.
- Rationale: The gain looks outlier-driven and comes with a worse max drawdown profile.

## Top Positive Contribution Days

| session_date | daily_pnl_diff | baseline_daily_pnl_usd | variant_daily_pnl_usd | baseline_bucket_label | variant_bucket_label |
| --- | --- | --- | --- | --- | --- |
| 2024-08-07 00:00:00 | 1350.000 | 0.000 | 1350.000 | no_trade | mid |
| 2023-08-23 00:00:00 | 1281.000 | 0.000 | 1281.000 | no_trade | mid |
| 2024-08-13 00:00:00 | 1212.000 | 0.000 | 1212.000 | no_trade | mid |
| 2022-04-26 00:00:00 | 860.000 | 0.000 | 860.000 | no_trade | mid |
| 2020-07-20 00:00:00 | 730.000 | 730.000 | 1460.000 | low | mid |
| 2024-03-19 00:00:00 | 685.500 | 0.000 | 685.500 | no_trade | mid |
| 2023-11-02 00:00:00 | 578.000 | -722.500 | -144.500 | mid | high |
| 2021-03-30 00:00:00 | 571.500 | -571.500 | 0.000 | mid | no_trade |
| 2025-07-01 00:00:00 | 532.000 | -665.000 | -133.000 | mid | high |
| 2021-10-07 00:00:00 | 441.000 | -441.000 | 0.000 | mid | no_trade |

## Top Negative Contribution Days

| session_date | daily_pnl_diff | baseline_daily_pnl_usd | variant_daily_pnl_usd | baseline_bucket_label | variant_bucket_label |
| --- | --- | --- | --- | --- | --- |
| 2022-07-19 00:00:00 | -1083.000 | 1444.000 | 361.000 | mid | high |
| 2021-01-20 00:00:00 | -969.000 | 1292.000 | 323.000 | mid | high |
| 2023-08-17 00:00:00 | -784.500 | 784.500 | 0.000 | mid | no_trade |
| 2020-11-24 00:00:00 | -712.000 | 1424.000 | 712.000 | mid | low |
| 2021-07-19 00:00:00 | -708.000 | 0.000 | -708.000 | no_trade | mid |
| 2024-09-16 00:00:00 | -618.000 | 0.000 | -618.000 | no_trade | mid |
| 2023-05-24 00:00:00 | -600.000 | -120.000 | -720.000 | high | mid |
| 2025-07-29 00:00:00 | -564.000 | 564.000 | 0.000 | mid | no_trade |
| 2021-07-13 00:00:00 | -549.000 | -183.000 | -732.000 | high | mid |
| 2021-11-29 00:00:00 | -492.000 | 492.000 | 0.000 | mid | no_trade |

## Monthly Return Snapshot

| variant_name | month | monthly_pnl_usd | monthly_return | positive_month |
| --- | --- | --- | --- | --- |
| median_fast15_slow_60_70_80 | 2019-05 | -176.000 | -0.004 | no |
| median_fast15_slow_60_70_80 | 2019-06 | -60.500 | -0.001 | no |
| median_fast15_slow_60_70_80 | 2019-07 | 0.000 | 0.000 | no |
| median_fast15_slow_60_70_80 | 2019-08 | 998.000 | 0.020 | yes |
| median_fast15_slow_60_70_80 | 2019-09 | -2.500 | -0.000 | no |
| median_fast15_slow_60_70_80 | 2019-10 | 122.500 | 0.002 | yes |
| median_fast15_slow_60_70_80 | 2019-11 | 0.000 | 0.000 | no |
| median_fast15_slow_60_70_80 | 2019-12 | -51.000 | -0.001 | no |
| median_fast15_slow_60_70_80 | 2020-01 | 774.000 | 0.015 | yes |
| median_fast15_slow_60_70_80 | 2020-02 | -177.500 | -0.004 | no |
| median_fast15_slow_60_70_80 | 2020-03 | 855.500 | 0.017 | yes |
| median_fast15_slow_60_70_80 | 2020-04 | -137.500 | -0.003 | no |
| median_fast15_slow_60_70_80 | 2020-05 | -1217.000 | -0.024 | no |
| median_fast15_slow_60_70_80 | 2020-06 | 1991.500 | 0.040 | yes |
| median_fast15_slow_60_70_80 | 2020-07 | 108.500 | 0.002 | yes |
| median_fast15_slow_60_70_80 | 2020-08 | -744.500 | -0.015 | no |
| median_fast15_slow_60_70_80 | 2020-09 | 714.000 | 0.014 | yes |
| median_fast15_slow_60_70_80 | 2020-10 | -715.500 | -0.014 | no |
