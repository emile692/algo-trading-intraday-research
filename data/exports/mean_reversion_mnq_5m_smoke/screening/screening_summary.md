# Mean Reversion Screening

- Variants screened: 3
- Survivors retained for validation: 0

## Family Verdicts

```text
                     family  total_variants  pass_screening_count                          best_name  best_oos_profit_factor  best_oos_sharpe_ratio screening_verdict
       opening_stretch_fade               1                     0 mnq_5m_opening_stretch_or_mid_fade                0.372249              -0.700695     famille morte
streak_exhaustion_reversion               1                     0       mnq_5m_streak4_vwap_snapback                0.000000               0.000000     famille morte
   vwap_extension_reversion               1                     0       mnq_5m_vwap_ext_atr_filtered                0.910384              -0.451861     famille morte
```

## Top Screening Rows

```text
                              name                      family symbol timeframe  oos_net_pnl  oos_profit_factor  oos_sharpe_ratio  oos_total_trades  oos_top_5_day_contribution_pct  oos_positive_month_ratio  screening_score  pass_screening
      mnq_5m_vwap_ext_atr_filtered    vwap_extension_reversion    MNQ        5m  -802.802246           0.910384         -0.451861               317                       -1.235285                  0.423077        -0.208623           False
      mnq_5m_streak4_vwap_snapback streak_exhaustion_reversion    MNQ        5m     0.000000           0.000000          0.000000                 0                        0.000000                  0.000000        -1.999994           False
mnq_5m_opening_stretch_or_mid_fade        opening_stretch_fade    MNQ        5m  -155.146429           0.372249         -0.700695                 6                       -0.592988                  0.076923        -2.887824           False
```
