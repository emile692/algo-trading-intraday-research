# Volatility Compression -> Expansion Breakout Validation

## Methodology

- Independent VCEB strategy built from scratch on top of the repo execution assumptions, without changing ORB / VWAP research modules.
- Universe: `MNQ, MES, M2K, MGC`.
- Source data: 1-minute intraday futures data, resampled deterministically to 5-minute OHLCV bars.
- Session filter: RTH only `09:30:00` -> `16:00:00`.
- Allowed entry window: `09:45:00` -> `15:30:00`.
- V1 fixed parameters: ATR `48`, expansion threshold `1.20`, breakout buffer `0.10 * ATR`, time stop `12` bars.
- Compression percentile calibration: `20` prior RTH sessions (1560 bars) with minimum history `5` sessions (390 bars), strict ex ante only.
- IS/OOS split: chronological `70%` / `30%` per instrument.

## Data Coverage

- `MNQ`: source `MNQ_c_0_1m_20260321_094501.parquet`, resampled `MNQ_c_0_5m_20260321_094501.parquet`, analysed sessions `124`.
- `MES`: source `MES_c_0_1m_20260322_135702.parquet`, resampled `MES_c_0_5m_20260322_135702.parquet`, analysed sessions `124`.
- `M2K`: source `M2K_c_0_1m_20260322_134808.parquet`, resampled `M2K_c_0_5m_20260322_134808.parquet`, analysed sessions `124`.
- `MGC`: source `MGC_c_0_1m_20260322_155729.parquet`, resampled `MGC_c_0_5m_20260322_155729.parquet`, analysed sessions `114`.

## Cross-Asset Screening

```text
       variant_name  box_lookback  compression_threshold  target_r  asset_count  oos_positive_assets  oos_profit_factor_gt_1_assets  oos_total_trades  oos_total_net_pnl  oos_median_profit_factor  oos_median_sharpe  oos_mean_expectancy  oos_median_win_rate  oos_worst_max_drawdown  oos_median_holding_minutes  oos_median_time_in_market_pct  oos_median_top_5_day_contribution_pct  screening_score  pass_screening
vceb_n16_ct15_tr2p2            16                   15.0       2.2            4                    0                              0                42        -467.075000                  0.843521          -0.617695           -16.618021             0.320513             -302.750000                       40.00                       0.026484                              -5.639146         5.246770           False
 vceb_n8_ct15_tr2p2             8                   15.0       2.2            4                    1                              1                53         168.653125                  0.870047          -0.406939             2.988752             0.318182             -336.380208                       22.50                       0.025978                              -2.681435         2.683792           False
 vceb_n8_ct20_tr1p8             8                   20.0       1.8            4                    1                              1                72          52.676042                  0.815823          -0.970285             0.013676             0.376471             -476.260417                       20.00                       0.033738                              -2.475187         1.894527           False
vceb_n16_ct20_tr2p2            16                   20.0       2.2            4                    0                              0                49        -580.475000                  0.698421          -1.440849           -13.944097             0.291667             -429.062500                       42.50                       0.033820                              -2.825191         1.409384           False
vceb_n12_ct20_tr1p8            12                   20.0       1.8            4                    1                              1                47         -51.500000                  0.725298          -1.269545            -0.432109             0.440972             -260.625000                       27.50                       0.027034                              -1.913574         0.860708           False
 vceb_n8_ct15_tr1p8             8                   15.0       1.8            4                    2                              2                53         166.813542                  1.074071          -0.250151             4.505407             0.431818             -336.380208                       17.50                       0.022773                               0.078552         0.410175           False
 vceb_n8_ct20_tr2p2             8                   20.0       2.2            4                    1                              1                72         -35.259375                  0.792901          -1.060803            -2.450568             0.324619             -476.260417                       27.50                       0.037449                              -1.119500         0.289572           False
vceb_n16_ct15_tr1p8            16                   15.0       1.8            4                    1                              1                42        -500.337500                  0.797925          -0.833893           -17.887447             0.320513             -302.750000                       35.00                       0.025472                              -1.050605         0.125448           False
vceb_n12_ct20_tr2p2            12                   20.0       2.2            4                    1                              1                47        -151.812500                  0.715443          -1.384494            -3.116276             0.418750             -260.625000                       33.75                       0.027554                              -1.207029         0.002437           False
vceb_n12_ct15_tr2p2            12                   15.0       2.2            4                    0                              0                33        -496.962500                  0.564356          -2.165044           -16.104025             0.266667             -227.250000                       25.00                       0.017081                              -1.606692        -0.288691           False
vceb_n16_ct20_tr1p8            16                   20.0       1.8            4                    1                              1                49        -621.687500                  0.642206          -1.706716           -15.433876             0.291667             -429.062500                       37.50                       0.032808                              -0.922211        -0.644215           False
vceb_n12_ct15_tr1p8            12                   15.0       1.8            4                    1                              1                33        -373.625000                  0.532220          -2.325445           -10.868655             0.281818             -227.250000                       22.50                       0.016575                              -0.441963        -1.396791           False
```

## Survivor Validation

```text
       variant_name  box_lookback  compression_threshold  target_r  screening_score  asset_count  oos_positive_assets  oos_total_trades  oos_total_net_pnl  oos_median_profit_factor  oos_median_sharpe  oos_median_top_5_day_contribution_pct  stress_positive_rows  stress_total_rows  worst_stress_oos_net_pnl  worst_stress_oos_profit_factor  median_mfe_r  median_mae_r  pct_trades_reaching_1r_mfe
 vceb_n8_ct15_tr2p2             8                   15.0       2.2         2.683792            4                    1                53         168.653125                  0.870047          -0.406939                              -2.681435                     2                  8               -326.348958                        0.506039      0.655367     -0.928571                    0.415094
vceb_n16_ct15_tr2p2            16                   15.0       2.2         5.246770            4                    0                42        -467.075000                  0.843521          -0.617695                              -5.639146                     0                  8               -318.500000                        0.000000      0.572158     -0.887712                    0.380952
```

## Research Verdict

- Verdict: `non_defendable`.
- Best variant: `vceb_n8_ct15_tr2p2`.
- Cross-asset read: `mono_asset`.
- OOS positive assets: `1`.
- OOS total trades: `53`.
- OOS total net PnL: `168.65` USD.
- Conclusion: The best observed configuration stays too fragile cross-asset, too sparse, or too cost-sensitive for a defendable V2.

## Export Inventory

- `screening_summary.csv`
- `instrument_variant_summary.csv`
- `oos_yearly_summary.csv`
- `stress_test_summary.csv`
- `survivor_validation_summary.csv`
- `final_report.md`
- `final_verdict.json`