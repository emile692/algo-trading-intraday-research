# Topstep Business Optimization

## Scope
- Source export: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Scope used: `oos`
- Bootstrap paths per configuration: `2000`
- Block size: `5`
- Seed: `42`
- Challenge rules: target `3,000` | trailing DD `2,000` | daily loss `1,000`
- Funded phase: `60` traded days max, payout every `1,000` cumulative funded profit

## Best Configuration By Plan
- **no_activation_fee**: `nominal_x1.8` -> `nominal` | expected net/day `61.79` | pass `100.0%` | avg resets `3.24` | avg payouts `3.62`.
- **standard**: `nominal_x1.8` -> `nominal` | expected net/day `65.00` | pass `100.0%` | avg resets `3.27` | avg payouts `3.75`.

## Trade-Off
- Overall best row: **standard / nominal_x1.8 -> nominal** with expected net/day `65.00` and expected net/cycle `3415.35`.
- Aggressive challenge leverage helps only if faster passes offset more resets and the extra subscription drag from longer retry loops.
- Funded-side value is captured through realized payout counts, while raw funded mark-to-market is still reported separately in the summary table.

## Reset-Cost Sensitivity
- Most fee-sensitive pairing: **nominal_x1.8 -> nominal** | delta profit/day `-3.22` for `no_activation_fee - standard`.
- Standard has cheaper monthly/reset costs but pays activation; No Activation Fee removes the pass fee but taxes every challenge day and every reset more heavily.

## Top Ranking

```text
 overall_rank  plan_rank              plan challenge_strategy    funded_strategy  pass_rate  avg_days_to_pass  avg_resets  avg_payouts  total_cost  expected_net_profit_per_cycle  expected_net_profit_per_day
            1          1          standard       nominal_x1.8            nominal        1.0           18.7575      3.2655       3.7550  339.646750                    3415.353250                    65.004202
            2          1 no_activation_fee       nominal_x1.8            nominal        1.0           18.2925      3.2435       3.6190  420.004250                    3198.995750                    61.785898
            3          2 no_activation_fee       nominal_x1.5            nominal        1.0           23.2530      2.4850       3.8210  355.350900                    3465.649100                    60.068448
            4          2          standard       nominal_x1.5            nominal        1.0           23.6440      2.5225       3.7545  311.221033                    3443.278967                    59.886237
            5          3 no_activation_fee       nominal_x1.2            nominal        1.0           28.7505      0.9805       3.8755  211.334650                    3664.165350                    58.107398
            6          4 no_activation_fee       nominal_x1.8 sizing_3state_x1.2        1.0           17.5985      3.0840       3.7960  400.097217                    3395.902783                    57.400552
            7          3          standard       nominal_x1.2            nominal        1.0           29.2065      0.9790       3.8250  244.674950                    3580.325050                    56.907336
            8          4          standard       nominal_x1.8 sizing_3state_x1.2        1.0           18.3535      3.2285       3.7150  337.173883                    3377.826117                    56.711093
            9          5 no_activation_fee       nominal_x1.5 sizing_3state_x1.2        1.0           23.2525      2.4820       3.8950  355.022083                    3539.977917                    54.319967
           10          5          standard       nominal_x1.5 sizing_3state_x1.2        1.0           23.3190      2.4645       3.7900  307.848200                    3482.151800                    54.123627
```

## Assumptions
- Challenge attempts are repeated with resets until first pass or a technical cap is hit (`max_challenge_attempts`, `challenge_safety_max_days`).
- Subscription cost is prorated on challenge trading days only because the available audited input is daily strategy PnL.
- Funded payouts are counted each time cumulative funded profit crosses another +1,000 USD threshold; payouts do not reduce simulated account equity in this stylized model.

