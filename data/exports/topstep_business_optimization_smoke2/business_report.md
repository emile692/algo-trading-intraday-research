# Topstep Business Optimization

## Scope
- Source export: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Scope used: `oos`
- Bootstrap paths per configuration: `10`
- Block size: `5`
- Seed: `42`
- Challenge rules: target `3,000` | trailing DD `2,000` | daily loss `1,000`
- Funded phase: `60` traded days max, payout every `1,000` cumulative funded profit

## Best Configuration By Plan
- **no_activation_fee**: `nominal_x1.2` -> `nominal` | expected net/day `83.87` | pass `100.0%` | avg resets `0.80` | avg payouts `6.00`.
- **standard**: `nominal` -> `nominal` | expected net/day `73.57` | pass `100.0%` | avg resets `0.80` | avg payouts `5.40`.

## Trade-Off
- Overall best row: **no_activation_fee / nominal_x1.2 -> nominal** with expected net/day `83.87` and expected net/cycle `5812.52`.
- Aggressive challenge leverage helps only if faster passes offset more resets and the extra subscription drag from longer retry loops.
- Funded-side value is captured through realized payout counts, while raw funded mark-to-market is still reported separately in the summary table.

## Reset-Cost Sensitivity
- Most fee-sensitive pairing: **nominal_x1.2 -> nominal** | delta profit/day `31.62` for `no_activation_fee - standard`.
- Standard has cheaper monthly/reset costs but pays activation; No Activation Fee removes the pass fee but taxes every challenge day and every reset more heavily.

## Top Ranking

```text
 overall_rank  plan_rank              plan challenge_strategy    funded_strategy  pass_rate  avg_days_to_pass  avg_resets  avg_payouts  total_cost  expected_net_profit_per_cycle  expected_net_profit_per_day
            1          1 no_activation_fee       nominal_x1.2            nominal        1.0              27.6         0.8          6.0  187.480000                    5812.520000                    83.874747
            2          1          standard            nominal            nominal        1.0              33.5         0.8          5.4  242.916667                    5157.083333                    73.567523
            3          2 no_activation_fee            nominal            nominal        1.0              26.0         0.6          4.2  159.866667                    4040.133333                    70.385598
            4          2          standard       nominal_x1.5      sizing_3state        1.0              20.8         2.0          4.6  280.973333                    4319.026667                    61.877173
            5          3          standard       nominal_x1.8 sizing_3state_x1.2        1.0              17.9         2.8          4.1  315.436667                    3784.563333                    58.675401
            6          3 no_activation_fee       nominal_x1.2      sizing_3state        1.0              34.3         1.2          5.2  255.423333                    4944.576667                    58.654527
            7          4 no_activation_fee       nominal_x1.2 sizing_3state_x1.2        1.0              30.8         0.9          4.2  210.006667                    3989.993333                    56.197089
            8          4          standard       nominal_x1.8            nominal        1.0              28.6         5.2          3.5  450.513333                    3049.486667                    54.068913
            9          5          standard       nominal_x1.5 sizing_3state_x1.2        1.0              22.1         2.3          3.7  297.796667                    3402.203333                    53.662513
           10          5 no_activation_fee       nominal_x1.5      sizing_3state        1.0              27.6         3.9          4.8  525.380000                    4274.620000                    52.838319
```

## Assumptions
- Challenge attempts are repeated with resets until first pass or a technical cap is hit (`max_challenge_attempts`, `challenge_safety_max_days`).
- Subscription cost is prorated on challenge trading days only because the available audited input is daily strategy PnL.
- Funded payouts are counted each time cumulative funded profit crosses another +1,000 USD threshold; payouts do not reduce simulated account equity in this stylized model.

