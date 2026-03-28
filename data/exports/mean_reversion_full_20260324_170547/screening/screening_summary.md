# Mean Reversion Screening

- Variants screened: 28
- Survivors retained for validation: 0

## Family Verdicts

```text
                     family  total_variants  pass_screening_count                        best_name  best_oos_profit_factor  best_oos_sharpe_ratio screening_verdict
 bollinger_zscore_reversion               5                     0  mes_5m_bollinger_20x2_immediate                     inf               1.198828     famille morte
      keltner_band_snapback               6                     0 m2k_5m_keltner_ema30_2p0_2closes                0.428510               0.000000     famille morte
       opening_stretch_fade               4                     0 m2k_5m_opening_stretch_open_fade                0.936402              -0.077045     famille morte
  rsi_stochastic_contrarian               4                     0        mes_15m_rsi3_exit_extreme                0.930551               0.000000     famille morte
streak_exhaustion_reversion               4                     0        m2k_5m_streak3_local_mean                     inf               0.981488     famille morte
   vwap_extension_reversion               5                     0    mnq_15m_vwap_ext_atr_filtered                3.024649               0.868457     famille morte
```

## Top Screening Rows

```text
                             name                      family symbol timeframe  oos_net_pnl  oos_profit_factor  oos_sharpe_ratio  oos_total_trades  oos_top_5_day_contribution_pct  oos_positive_month_ratio  screening_score  pass_screening
 m2k_5m_opening_stretch_open_fade        opening_stretch_fade    M2K        5m   -11.987500           0.936402         -0.077045                20                      -12.429614                  0.269231        14.287179           False
        mes_15m_rsi3_exit_extreme   rsi_stochastic_contrarian    MES       15m    -9.781559           0.930551         -0.071172                13                      -11.580879                  0.153846        13.064797           False
  mes_5m_bollinger_20x2_immediate  bollinger_zscore_reversion    MES        5m   -79.375000           0.972570         -0.093150               162                       -9.977165                  0.423077        12.066511           False
  mes_5m_bollinger_30x2p5_reentry  bollinger_zscore_reversion    MES        5m    76.458333                inf          1.198828                 3                        1.000000                  0.076923         2.841245           False
 mes_15m_bollinger_30x2p5_reentry  bollinger_zscore_reversion    MES       15m    65.750000                inf          0.796655                 2                        1.000000                  0.076923         2.662038           False
 mes_15m_bollinger_20x2_immediate  bollinger_zscore_reversion    MES       15m  -149.678571           0.870077         -0.273863                45                       -3.332558                  0.346154         2.465471           False
    mnq_15m_vwap_ext_atr_filtered    vwap_extension_reversion    MNQ       15m   294.528571           3.024649          0.868457                 9                        1.386962                  0.230769         2.421846           False
        m2k_5m_streak3_local_mean streak_exhaustion_reversion    M2K        5m    11.775000                inf          0.981488                 2                        1.000000                  0.076923         2.215542           False
     mnq_5m_vwap_ext_atr_filtered    vwap_extension_reversion    MNQ        5m  -802.802246           0.910384         -0.451861               317                       -1.235285                  0.423077        -0.208623           False
m2k_15m_opening_stretch_open_fade        opening_stretch_fade    M2K       15m  -152.112500           0.668681         -0.541840                23                       -1.413428                  0.307692        -0.958055           False
           mes_5m_vwap_ext_zscore    vwap_extension_reversion    MES        5m  -514.298495           0.792963         -0.585326                88                       -1.067114                  0.423077        -1.199072           False
       mgc_5m_vwap_ext_structural    vwap_extension_reversion    MGC        5m  -640.009230           0.744921         -0.673327               182                       -1.018205                  0.230769        -1.497932           False
     mnq_5m_streak4_vwap_snapback streak_exhaustion_reversion    MNQ        5m     0.000000           0.000000          0.000000                 0                        0.000000                  0.000000        -1.999994           False
    mnq_15m_streak4_vwap_snapback streak_exhaustion_reversion    MNQ       15m     0.000000           0.000000          0.000000                 0                        0.000000                  0.000000        -1.999994           False
 m2k_5m_keltner_ema30_2p0_2closes       keltner_band_snapback    M2K        5m     0.000000           0.000000          0.000000                 0                        0.000000                  0.000000        -1.999994           False
m2k_15m_keltner_ema30_2p0_2closes       keltner_band_snapback    M2K       15m     0.000000           0.000000          0.000000                 0                        0.000000                  0.000000        -1.999994           False
            mgc_15m_rsi5_reversal   rsi_stochastic_contrarian    MGC       15m     0.000000           0.000000          0.000000                 0                        0.000000                  0.000000        -1.999994           False
 mgc_15m_bollinger_30x2p5_reentry  bollinger_zscore_reversion    MGC       15m     0.000000           0.000000          0.000000                 0                        0.000000                  0.000000        -1.999994           False
       m2k_15m_streak3_local_mean streak_exhaustion_reversion    M2K       15m  -395.067857           0.639016         -1.023888                90                       -0.658305                  0.384615        -2.042253           False
 mgc_15m_stoch_8_3_3_exit_extreme   rsi_stochastic_contrarian    MGC       15m  -260.931060           0.392917         -0.670355                32                       -0.389863                  0.153846        -2.566997           False
```
