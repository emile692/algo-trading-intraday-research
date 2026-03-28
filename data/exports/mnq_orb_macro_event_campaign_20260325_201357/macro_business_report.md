# Macro Business Report

## Scope
- Source export: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Source variant: `nominal`
- Primary scope: `oos`
- Calendar features: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\processed\economic_calendar\economic_calendar_daily_features.csv`
- Overlap window: `2026-01-07` to `2026-03-19`
- Overlap sample: `52` sessions | `12` traded days | `12` trades
- Rolling common starts across variants: `49`
- Challenge rules: target `3,000` | trailing DD `2,000` | daily loss `1,000`
- Economic objective reused from Topstep optimization: `pass_rate * 3,000 - fail_rate * 100`

## Sample Limits
- The production macro calendar currently overlaps only a recent 2026 slice of the audited strategy history, so this study is deliberately an overlap-window decision aid rather than a full-history claim.
- Rare priority cohorts: cpi_or_nfp_day (2 calendar / 0 traded), fomc_day (1 calendar / 0 traded), other_high_impact_macro_day (1 calendar / 0 traded)
- Traded event-day coverage inside the overlap: hard-filter variants `0` impacted traded days | deleverage variants `0` impacted traded days.

## Cohort Snapshot

```text
                cohort_name  calendar_day_count  traded_day_count  total_pnl_usd  avg_pnl_per_calendar_day  profit_factor  rolling_pass_rate  rolling_probability_breaching_max_loss_limit  rolling_expected_days_to_pass
             cpi_or_nfp_day                   2                 0            0.0                       0.0       0.000000                0.0                                           0.0                            NaN
                   fomc_day                   1                 0            0.0                       0.0       0.000000                0.0                                           0.0                            NaN
other_high_impact_macro_day                   1                 0            0.0                       0.0       0.000000                0.0                                           0.0                            NaN
                 normal_day                  48                12          432.0                       9.0       1.233198                0.0                                           0.0                            NaN
```

## Variant Ranking

```text
 rank                     variant_name variant_family  impacted_traded_days  rolling_pass_rate  rolling_global_max_loss_violation_rate  rolling_mean_days_to_pass  rolling_expected_net_profit_per_day  sample_total_pnl_usd
    1                         baseline       baseline                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    2 deleverage_all_high_impact_0.25x     deleverage                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    3  deleverage_all_high_impact_0.5x     deleverage                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    4         deleverage_cpi_nfp_0.25x     deleverage                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    5          deleverage_cpi_nfp_0.5x     deleverage                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    6            deleverage_fomc_0.25x     deleverage                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    7             deleverage_fomc_0.5x     deleverage                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    8             skip_all_high_impact    hard_filter                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
    9                     skip_cpi_nfp    hard_filter                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
   10                        skip_fomc    hard_filter                     0                0.0                                     0.0                        NaN                           -23.333333                 432.0
```

## Verdict
- Hard filtering improves survivability: **not measurable in sample**. Best hard filter: `skip_all_high_impact` | pass `0.0%` | max-loss breach `0.0%` | days to pass `nan` | expected net/day `-23.33`.
- Hard filtering hurts passing speed too much: **not measurable in sample**.
- Deleveraging is superior to hard filtering: **not measurable in sample**. Best deleverage: `deleverage_all_high_impact_0.25x` | pass `0.0%` | max-loss breach `0.0%` | days to pass `nan` | expected net/day `-23.33`.
- Recommended policy for Topstep 50K: **Keep `baseline`; the overlap sample contains zero traded high-impact macro days, so macro overlays are not identified yet.** Best overall row: `baseline` | pass `0.0%` | max-loss breach `0.0%` | days to pass `nan` | expected net/day `-23.33`.

## Baseline
- `baseline` | pass `0.0%` | max-loss breach `0.0%` | days to pass `nan` | expected net/day `-23.33`
