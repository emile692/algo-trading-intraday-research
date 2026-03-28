# Stress Test Report

- Verdict: `strategie tres fragile a la microstructure`

```text
             scenario                                                     notes  overall_net_pnl  overall_profit_factor  overall_sharpe_ratio  overall_max_drawdown  oos_net_pnl  oos_profit_factor  oos_sharpe_ratio  oos_max_drawdown  delta_oos_net_pnl_vs_nominal  delta_oos_profit_factor_vs_nominal  delta_oos_sharpe_vs_nominal
   combined_x2_plus25                            Slippage x2 + commission +25%.     -3612.460714               0.980251             -0.148398          -9090.728571 -1000.128571           0.985611         -0.108638      -9090.728571                     -2240.875                           -0.032678                    -0.243707
   combined_x3_plus50                            Slippage x3 + commission +50%.    -11173.585714               0.940786             -0.457739         -11569.767857 -3241.003571           0.954477         -0.351286     -10741.728571                     -4481.750                           -0.063813                    -0.486354
commission_plus_25pct                                          Commission +25%.      1040.539286               1.005801              0.042817          -8074.728571   378.871429           1.005533          0.041210      -8074.728571                      -861.875                           -0.012757                    -0.093859
commission_plus_50pct                                          Commission +50%.     -1867.585714               0.989715             -0.076768          -8709.728571  -483.003571           0.993012         -0.052492      -8709.728571                     -1723.750                           -0.025278                    -0.187561
  entry_penalty_1tick              Extra one-tick entry penalty on every trade.      1622.164286               1.009066              0.066765          -7947.728571   551.246429           1.008065          0.059969      -7947.728571                      -689.500                           -0.010225                    -0.075099
              nominal                                                 Base run.      3948.664286               1.022287              0.162656          -7548.971429  1240.746429           1.018290          0.135068      -7548.971429                         0.000                            0.000000                     0.000000
open_penalty_early15m Extra one-tick entry penalty during the first 15 minutes.      3890.164286               1.021952              0.160249          -7557.471429  1225.746429           1.018066          0.133437      -7557.471429                       -15.000                           -0.000224                    -0.001631
          slippage_x2                                         Slippage doubled.      -704.335714               0.996102             -0.028964          -8455.728571  -138.253571           0.997992         -0.015030      -8455.728571                     -1379.000                           -0.020297                    -0.150099
          slippage_x3                                         Slippage tripled.     -5357.335714               0.970923             -0.219936          -9471.728571 -1517.253571           0.978293         -0.164728      -9471.728571                     -2758.000                           -0.039997                    -0.299796
```

- Cost stresses are applied as path-preserving overlays on the corrected nominal trade log.
- This is exact for fixed cost changes and intentionally conservative for extra entry penalties.
