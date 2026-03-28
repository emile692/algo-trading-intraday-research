# Topstep Optimization Report

## Scope
- Source run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Primary scope: `oos`
- OOS rolling common start dates: `383`
- Rules: start `50,000` | target `3,000` | trailing DD `2,000` | daily loss `1,000` | max trading days `60`
- Economic objective: `pass_rate * payout_value - fail_rate * reset_cost` with payout `3,000` and reset `100`

## Best Variant
- Overall top row by expected profit per day: **historical_rolling / nominal_x1.8**
- Historical rolling winner: **nominal_x1.8** | expected profit/day `307.60` | pass `28.5%` | fail `71.5%`
- Block bootstrap winner: **nominal_x1.8** | expected profit/day `234.57` | pass `22.1%` | fail `77.8%`

## Trade-Off
- Higher leverage only helps if the extra pass-rate offsets the additional daily-loss and trailing-DD failures; this campaign ranks variants on that economic trade-off rather than raw pnl.
- The recommended row has pass<=20d probability `22.1%` and pass<=30d probability `22.1%`.
- Recommended live configuration: **nominal_x1.8** under **block_bootstrap** emphasis, because it delivers expected profit/day `234.57` with pass `22.1%` and fail `77.8%`.

## Top Ranking

```text
 rank    simulation_mode            variant  pass_rate  fail_rate  avg_time_to_pass  avg_time_to_fail  expected_profit_per_cycle  expected_days_per_cycle  expected_profit_per_day  probability_pass_within_20_days  probability_pass_within_30_days
    1 historical_rolling       nominal_x1.8   0.284595   0.715405          3.944954          1.985401                 782.245431                 2.543081               307.597536                         0.284595                         0.284595
    2 historical_rolling       nominal_x1.5   0.318538   0.681462          5.196721          3.065134                 887.467363                 3.744125               237.029289                         0.318538                         0.318538
    3    block_bootstrap       nominal_x1.8   0.221500   0.778500          3.483070          2.221580                 586.650000                 2.501000               234.566174                         0.221500                         0.221500
    4    block_bootstrap       nominal_x1.5   0.279500   0.720500          5.162791          3.684941                 766.450000                 4.098000               187.030259                         0.279500                         0.279500
    5 historical_rolling sizing_3state_x1.8   0.370757   0.629243          7.112676          4.933610                1049.347258                 5.741514               182.764893                         0.370757                         0.370757
    6 historical_rolling       nominal_x1.2   0.556136   0.443864          9.436620          8.447059                1624.020888                 8.997389               180.499129                         0.532637                         0.556136
    7    block_bootstrap       nominal_x1.2   0.533500   0.466500          9.367385          8.853162                1553.850000                 9.127500               170.238291                         0.512000                         0.532000
    8    block_bootstrap sizing_3state_x1.8   0.374500   0.625500          6.875834          5.883293                1060.950000                 6.255000               169.616307                         0.371500                         0.374500
```
