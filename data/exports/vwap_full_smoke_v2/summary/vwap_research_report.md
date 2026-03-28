# VWAP Research Campaign

## Scope

- Dataset: `MNQ_c_0_1m_20260321_094501.parquet`
- Symbol: `MNQ`
- Sessions total / IS / OOS: 1747 / 1222 / 525
- RTH handling is explicit and uses `[09:30, 16:00)` start-aligned bars.
- The paper baseline is implemented first without filters, targets, stops, or kill switches.

## Best Prop Variant

- Selected variant: `vwap_pullback_continuation`

## Comparative Table

```text
                          name       family  overall_net_pnl  overall_sharpe_ratio  overall_profit_factor  overall_max_drawdown   oos_net_pnl  oos_sharpe_ratio  oos_profit_factor  oos_max_drawdown  oos_daily_loss_limit_breach_freq  oos_trailing_drawdown_breach_freq  oos_profit_to_drawdown_ratio
           paper_vwap_baseline     baseline    -25008.000000             -0.899476               0.926290         -25495.500000  -2376.500000         -0.706673           0.766288      -2518.500000                          0.001905                           0.940952                     -0.943617
      baseline_futures_adapted     baseline    -42674.500000             -1.138011               0.919548         -43371.500000 -20043.000000         -1.412155           0.900446     -20394.500000                          0.011429                           0.872381                     -0.982765
   vwap_time_filtered_baseline prop_variant    -14847.500000             -0.405462               0.965615         -23908.500000 -12825.500000         -0.907493           0.923806     -14511.000000                          0.017143                           0.769524                     -0.883847
 vwap_baseline_with_killswitch prop_variant     -2032.000000             -0.110822               0.984841          -6117.000000  -1060.500000         -0.171220           0.978823      -3544.000000                          0.000000                           0.396190                     -0.299238
                  vwap_reclaim prop_variant      1746.366071              0.669539               2.354156           -242.276786    904.053571          0.729811           2.618873       -242.276786                          0.000000                           0.000000                      3.731491
    vwap_pullback_continuation prop_variant     96790.410714              4.108515               1.863244          -1612.782143  37287.378571          4.203055           1.863071      -1612.782143                          0.000000                           0.000000                     23.119910
vwap_reclaim_with_prop_overlay prop_variant       -13.500000             -0.497664               0.000000             -9.500000      0.000000          0.000000           0.000000          0.000000                          0.000000                           0.000000                           inf
```

## Sensitivity

```text
     sensitivity_tag  oos_net_pnl  oos_sharpe_ratio  oos_profit_factor  oos_max_drawdown  oos_daily_loss_limit_breach_freq
         cost_x_0p75 38157.378571          4.304276           1.895232      -1600.907143                               0.0
          cost_x_1p0 37287.378571          4.203055           1.863071      -1612.782143                               0.0
         cost_x_1p25 36417.378571          4.101974           1.831718      -1624.657143                               0.0
     atr_buffer_0p15 34202.512500          3.948882           1.814482      -1630.451786                               0.0
     atr_buffer_0p25 36730.892857          4.154902           1.863492      -1511.901786                               0.0
      atr_buffer_0p4 35589.185714          4.042541           1.819955      -1727.742857                               0.0
 slope_threshold_0p0 37287.378571          4.203055           1.863071      -1612.782143                               0.0
slope_threshold_0p01 37293.378571          4.203752           1.863330      -1612.782143                               0.0
slope_threshold_0p02 37293.378571          4.203752           1.863330      -1612.782143                               0.0
        max_trades_1 18146.300000          3.308732           2.094753      -1057.625000                               0.0
        max_trades_2 30078.532143          3.932851           1.944189      -1259.689286                               0.0
time_windows_default 34004.392857          3.961774           1.821182      -1542.482143                               0.0
time_windows_shorter 29854.825000          3.847385           1.765475      -1355.339286                               0.0
```

## Honest Conclusion

- The selected prop variant improves the baseline on challenge-oriented robustness, not only on raw returns.
- Remaining limits: different underlying than QQQ/TQQQ when futures are used, 1-minute bars only, and a lightweight rather than exhaustive walk-forward.
