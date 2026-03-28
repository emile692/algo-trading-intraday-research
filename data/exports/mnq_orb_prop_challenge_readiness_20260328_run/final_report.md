# MNQ ORB Prop Challenge Readiness

## Scope
- Source export: `data\exports\mnq_orb_vvix_sizing_modulation_20260328_run`
- Scope used: `oos`
- Common rolling starts across included variants: `282`
- Core comparison: `baseline_3state` vs `baseline_3state_vvix_modulator`
- Additional bounds included: `['baseline_nominal', 'baseline_vvix_modulator']`

## Central Rules
- Account size: `50,000 USD`
- Profit target: `3,000 USD`
- Daily loss limit: `1000.0`
- Static max loss: `None`
- Trailing drawdown: `2000.0`
- Max trading days: `35`
- Daily cut on DLL hit: `True`

## Primary Challenge Readout
- Best challenge row on primary settings: `baseline_3state` | pass `55.3%` | breach `35.5%` | median days `16.0` | expected attempt pnl `1581.91`.
- Runner-up core row: `baseline_3state_vvix_modulator` | pass `52.5%` | breach `32.3%` | median days `16.0`.
- Best funded lens row on primary settings: `baseline_3state` | complete `21.8%` | breach `78.2%` | follow-up profit/day `46.70`.

## Risk Profile Summary

```text
                  variant_name risk_profile  pass_rate  breach_rate  median_days_to_pass  expected_net_profit_per_attempt
               baseline_3state    assertive   0.627660     0.343972                 14.0                      1697.765957
               baseline_3state         base   0.553191     0.354610                 16.0                      1581.907801
               baseline_3state conservative   0.475177     0.212766                 22.0                      1737.376330
baseline_3state_vvix_modulator    assertive   0.595745     0.390071                 12.0                      1688.665957
baseline_3state_vvix_modulator         base   0.524823     0.322695                 16.0                      1713.962766
baseline_3state_vvix_modulator conservative   0.457447     0.262411                 21.0                      1605.199468
              baseline_nominal    assertive   0.276596     0.198582                 24.0                      1443.480851
              baseline_nominal         base   0.166667     0.195035                 28.0                      1187.361702
              baseline_nominal conservative   0.021277     0.092199                 33.5                       887.212766
       baseline_vvix_modulator    assertive   0.496454     0.503546                  8.5                      1325.929787
       baseline_vvix_modulator         base   0.475177     0.521277                 11.0                      1186.464539
       baseline_vvix_modulator conservative   0.439716     0.358156                 13.5                      1370.015957
```

## Stress Summary

```text
                  variant_name   stress_profile  pass_rate  breach_rate  median_days_to_pass  expected_net_profit_per_attempt
               baseline_3state slippage_nominal   0.553191     0.354610                 16.0                      1581.907801
               baseline_3state      slippage_x2   0.542553     0.358156                 16.0                      1549.680851
               baseline_3state      slippage_x3   0.528369     0.368794                 16.0                      1525.113475
baseline_3state_vvix_modulator slippage_nominal   0.524823     0.322695                 16.0                      1713.962766
baseline_3state_vvix_modulator      slippage_x2   0.521277     0.322695                 16.0                      1687.359929
baseline_3state_vvix_modulator      slippage_x3   0.510638     0.326241                 15.0                      1659.556738
              baseline_nominal slippage_nominal   0.166667     0.195035                 28.0                      1187.361702
              baseline_nominal      slippage_x2   0.166667     0.195035                 28.0                      1156.781915
              baseline_nominal      slippage_x3   0.163121     0.195035                 27.5                      1128.241135
       baseline_vvix_modulator slippage_nominal   0.475177     0.521277                 11.0                      1186.464539
       baseline_vvix_modulator      slippage_x2   0.471631     0.521277                 11.0                      1162.090426
       baseline_vvix_modulator      slippage_x3   0.460993     0.528369                 11.0                      1093.941489
```

## Direct Answers
- Best configuration to pass the challenge: `baseline_3state` with risk profile `assertive`.
- Best configuration for longer-term / funded offense: `baseline_3state`.
- Does the VVIX modulator improve challenge business value even if it does not beat raw 3-state performance? `Yes`.
- Single universal configuration or split challenge/funded setup? `A single configuration is sufficient on current evidence`.
- Most defendable live launch risk profile: `assertive` on `baseline_3state`.

## Funded Lens

```text
                  variant_name risk_profile   stress_profile  followup_complete_rate  followup_breach_rate  followup_expected_profit_per_trading_day  followup_big_negative_day_rate
               baseline_3state    assertive slippage_nominal                0.186441              0.813559                                 77.324923                        0.063889
               baseline_3state    assertive      slippage_x2                0.120690              0.879310                                 75.204349                        0.065235
               baseline_3state    assertive      slippage_x3                0.121387              0.878613                                 72.213868                        0.065256
               baseline_3state         base slippage_nominal                0.217949              0.782051                                 46.696571                        0.000000
               baseline_3state         base      slippage_x2                0.215686              0.784314                                 44.385684                        0.000000
               baseline_3state         base      slippage_x3                0.174497              0.825503                                 41.562591                        0.000000
               baseline_3state conservative slippage_nominal                0.432836              0.567164                                 46.266665                        0.000000
               baseline_3state conservative      slippage_x2                0.427481              0.572519                                 43.911892                        0.000000
               baseline_3state conservative      slippage_x3                0.417323              0.582677                                 40.920121                        0.000000
baseline_3state_vvix_modulator    assertive slippage_nominal                0.065476              0.934524                                -16.090561                        0.086120
baseline_3state_vvix_modulator    assertive      slippage_x2                0.065868              0.934132                                -20.495607                        0.086609
baseline_3state_vvix_modulator    assertive      slippage_x3                0.067073              0.932927                                -23.652115                        0.087582
baseline_3state_vvix_modulator         base slippage_nominal                0.324324              0.675676                                 35.273391                        0.000000
baseline_3state_vvix_modulator         base      slippage_x2                0.319728              0.680272                                 30.998010                        0.000000
baseline_3state_vvix_modulator         base      slippage_x3                0.326389              0.673611                                 30.806998                        0.000000
baseline_3state_vvix_modulator conservative slippage_nominal                0.503876              0.496124                                 44.818686                        0.000000
baseline_3state_vvix_modulator conservative      slippage_x2                0.500000              0.500000                                 41.388898                        0.000000
baseline_3state_vvix_modulator conservative      slippage_x3                0.500000              0.500000                                 39.559121                        0.000000
              baseline_nominal    assertive slippage_nominal                0.205128              0.794872                                -14.725682                        0.000000
              baseline_nominal    assertive      slippage_x2                0.186667              0.813333                                -16.037746                        0.000000
              baseline_nominal    assertive      slippage_x3                0.191781              0.808219                                -18.421243                        0.000000
              baseline_nominal         base slippage_nominal                0.276596              0.723404                                -12.775222                        0.000000
              baseline_nominal         base      slippage_x2                0.276596              0.723404                                -14.129801                        0.000000
              baseline_nominal         base      slippage_x3                0.282609              0.717391                                -15.801373                        0.000000
              baseline_nominal conservative slippage_nominal                1.000000              0.000000                                 24.026042                        0.000000
              baseline_nominal conservative      slippage_x2                1.000000              0.000000                                 23.276042                        0.000000
              baseline_nominal conservative      slippage_x3                1.000000              0.000000                                 21.220000                        0.000000
       baseline_vvix_modulator    assertive slippage_nominal                0.000000              1.000000                                 13.624110                        0.243851
       baseline_vvix_modulator    assertive      slippage_x2                0.000000              1.000000                                  8.328909                        0.244787
       baseline_vvix_modulator    assertive      slippage_x3                0.000000              1.000000                                  4.757097                        0.276512
       baseline_vvix_modulator         base slippage_nominal                0.059701              0.940299                                 -0.900126                        0.000000
       baseline_vvix_modulator         base      slippage_x2                0.000000              1.000000                                 -8.366696                        0.000000
       baseline_vvix_modulator         base      slippage_x3                0.000000              1.000000                                -12.373899                        0.000000
       baseline_vvix_modulator conservative slippage_nominal                0.346774              0.653226                                 53.372994                        0.000000
       baseline_vvix_modulator conservative      slippage_x2                0.349593              0.650407                                 52.046657                        0.000000
       baseline_vvix_modulator conservative      slippage_x3                0.352459              0.647541                                 49.428918                        0.000000
```

## Business Verdict
- Launch readiness: `not_defendable_yet`.
- Stress flip detected: `False`.
- This report is decision-oriented and does not claim discovery of a new alpha source.

