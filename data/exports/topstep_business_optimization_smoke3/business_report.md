# Topstep Business Optimization

## Scope
- Source export: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Scope used: `oos`
- Bootstrap paths per configuration: `5`
- Block size: `5`
- Seed: `42`
- Challenge rules: target `3,000` | trailing DD `2,000` | daily loss `1,000`
- Funded phase: `60` traded days max, payout every `1,000` cumulative funded profit

## Best Configuration By Plan
- **no_activation_fee**: `nominal` -> `nominal` | expected net/day `78.28` | pass `100.0%` | avg resets `1.00` | avg payouts `4.80`.
- **standard**: `nominal_x1.2` -> `sizing_3state_x1.2` | expected net/day `69.43` | pass `100.0%` | avg resets `0.40` | avg payouts `4.40`.

## Trade-Off
- Overall best row: **no_activation_fee / nominal -> nominal** with expected net/day `78.28` and expected net/cycle `4587.09`.
- Aggressive challenge leverage helps only if faster passes offset more resets and the extra subscription drag from longer retry loops.
- Funded-side value is captured through realized payout counts, while raw funded mark-to-market is still reported separately in the summary table.

## Reset-Cost Sensitivity
- Most fee-sensitive pairing: **nominal_x1.8 -> nominal** | delta profit/day `-35.71` for `no_activation_fee - standard`.
- Standard has cheaper monthly/reset costs but pays activation; No Activation Fee removes the pass fee but taxes every challenge day and every reset more heavily.

## Top Ranking

```text
 overall_rank  plan_rank              plan challenge_strategy    funded_strategy  pass_rate  avg_days_to_pass  avg_resets  avg_payouts  total_cost  expected_net_profit_per_cycle  expected_net_profit_per_day
            1          1 no_activation_fee            nominal            nominal        1.0              28.6         1.0          4.8  212.913333                    4587.086667                    78.277929
            2          1          standard       nominal_x1.2 sizing_3state_x1.2        1.0              23.2         0.4          4.4  206.493333                    4193.506667                    69.428918
            3          2          standard       nominal_x1.2      sizing_3state        1.0              25.4         1.2          5.8  249.286667                    5550.713333                    65.456525
            4          3          standard       nominal_x1.5      sizing_3state        1.0              14.4         1.2          4.8  231.320000                    4568.680000                    64.166854
            5          4          standard       nominal_x1.8            nominal        1.0              26.2         4.2          4.0  397.593333                    3602.406667                    61.057740
            6          5          standard       nominal_x1.5            nominal        1.0              34.6         3.6          5.0  381.913333                    4618.086667                    56.733251
            7          2 no_activation_fee       nominal_x1.2            nominal        1.0              36.2         1.4          4.2  284.126667                    3915.873333                    56.424688
            8          3 no_activation_fee       nominal_x1.5      sizing_3state        1.0              17.8         1.6          4.4  239.073333                    4160.926667                    55.627362
            9          6          standard       nominal_x1.8 sizing_3state_x1.2        1.0              21.6         3.4          4.0  350.880000                    3649.120000                    54.302381
           10          7          standard       nominal_x1.2            nominal        1.0              56.2         2.0          5.6  338.793333                    5261.206667                    54.016496
```

## Assumptions
- Challenge attempts are repeated with resets until first pass or a technical cap is hit (`max_challenge_attempts`, `challenge_safety_max_days`).
- Subscription cost is prorated on challenge trading days only because the available audited input is daily strategy PnL.
- Funded payouts are counted each time cumulative funded profit crosses another +1,000 USD threshold; payouts do not reduce simulated account equity in this stylized model.

