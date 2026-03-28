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

- `MNQ`: source `MNQ_c_0_1m_20260321_094501.parquet`, resampled `MNQ_c_0_5m_20260321_094501.parquet`, analysed sessions `1747`.
- `MES`: source `MES_c_0_1m_20260322_135702.parquet`, resampled `MES_c_0_5m_20260322_135702.parquet`, analysed sessions `1747`.
- `M2K`: source `M2K_c_0_1m_20260322_134808.parquet`, resampled `M2K_c_0_5m_20260322_134808.parquet`, analysed sessions `1747`.
- `MGC`: source `MGC_c_0_1m_20260322_155729.parquet`, resampled `MGC_c_0_5m_20260322_155729.parquet`, analysed sessions `3128`.

## Cross-Asset Screening

```text
       variant_name  box_lookback  compression_threshold  target_r  asset_count  oos_positive_assets  oos_profit_factor_gt_1_assets  oos_total_trades  oos_total_net_pnl  oos_median_profit_factor  oos_median_sharpe  oos_mean_expectancy  oos_median_win_rate  oos_worst_max_drawdown  oos_median_holding_minutes  oos_median_time_in_market_pct  oos_median_top_5_day_contribution_pct  screening_score  pass_screening
vceb_n12_ct20_tr1p8            12                   20.0       1.8            4                    1                              1               808       -2717.911458                  0.828002          -0.551246            -3.015628             0.396207            -2099.067708                       35.00                       0.031832                               1.303746        -2.419013           False
vceb_n16_ct15_tr1p8            16                   15.0       1.8            4                    0                              0               525       -3914.750000                  0.695050          -1.073316            -7.340502             0.368525            -1608.037500                       42.50                       0.023431                               0.665957        -2.630132           False
vceb_n16_ct15_tr2p2            16                   15.0       2.2            4                    0                              0               523       -4041.637500                  0.652166          -1.090788            -7.597706             0.362269            -1466.637500                       47.50                       0.024530                               0.716086        -2.752790           False
vceb_n16_ct20_tr1p8            16                   20.0       1.8            4                    0                              0               676       -4976.700000                  0.697886          -1.077374            -7.286208             0.378660            -2083.181250                       47.50                       0.031050                               0.638530        -2.768652           False
 vceb_n8_ct15_tr2p2             8                   15.0       2.2            4                    0                              0               749       -4322.391667                  0.831400          -0.714503            -5.443560             0.359847            -2559.078125                       30.00                       0.028987                               1.099353        -2.814035           False
 vceb_n8_ct20_tr1p8             8                   20.0       1.8            4                    0                              0              1026       -6513.766667                  0.804557          -1.005554            -6.055576             0.366962            -3019.015625                       27.50                       0.036496                               0.697730        -2.820593           False
 vceb_n8_ct20_tr2p2             8                   20.0       2.2            4                    0                              0              1023       -6954.645833                  0.783704          -1.105088            -6.507037             0.346573            -3293.828125                       32.50                       0.038962                               0.606605        -2.855515           False
vceb_n16_ct20_tr2p2            16                   20.0       2.2            4                    0                              0               674       -5323.137500                  0.705290          -0.995904            -7.814987             0.364563            -2562.881250                       50.00                       0.032381                               0.722957        -2.896477           False
vceb_n12_ct20_tr2p2            12                   20.0       2.2            4                    0                              0               806       -3822.136458                  0.806089          -0.632225            -4.378964             0.381144            -2450.630208                       40.00                       0.033773                               1.291697        -2.928089           False
 vceb_n8_ct15_tr1p8             8                   15.0       1.8            4                    0                              0               750       -4319.737500                  0.866006          -0.513160            -5.374566             0.381926            -2593.755208                       27.50                       0.026874                               1.524847        -3.028128           False
vceb_n12_ct15_tr2p2            12                   15.0       2.2            4                    1                              1               621       -2076.286458                  0.845291          -0.598656            -3.260852             0.366238            -1563.956250                       38.75                       0.024750                               2.063117        -3.071039           False
vceb_n12_ct15_tr1p8            12                   15.0       1.8            4                    1                              1               622        -946.073958                  0.902179          -0.355709            -1.461645             0.386273            -1339.056250                       33.75                       0.023284                               3.749032        -4.267944           False
```

## Survivor Validation

```text
       variant_name  box_lookback  compression_threshold  target_r  screening_score  asset_count  oos_positive_assets  oos_total_trades  oos_total_net_pnl  oos_median_profit_factor  oos_median_sharpe  oos_median_top_5_day_contribution_pct  stress_positive_rows  stress_total_rows  worst_stress_oos_net_pnl  worst_stress_oos_profit_factor  median_mfe_r  median_mae_r  pct_trades_reaching_1r_mfe
vceb_n12_ct20_tr1p8            12                   20.0       1.8        -2.419013            4                    1               808       -2717.911458                  0.828002          -0.551246                               1.303746                     2                  8              -2736.755208                        0.634756      0.637425     -0.842825                    0.346535
vceb_n16_ct15_tr1p8            16                   15.0       1.8        -2.630132            4                    0               525       -3914.750000                  0.695050          -1.073316                               0.665957                     0                  8              -1836.362500                        0.501505      0.587413     -0.774194                    0.302857
```

## Research Verdict

- Verdict: `non_defendable`.
- Best variant: `vceb_n12_ct20_tr1p8`.
- Cross-asset read: `mono_asset`.
- OOS positive assets: `1`.
- OOS total trades: `808`.
- OOS total net PnL: `-2717.91` USD.
- Conclusion: The best observed configuration stays too fragile cross-asset, too sparse, or too cost-sensitive for a defendable V2.

## Export Inventory

- `screening_summary.csv`
- `instrument_variant_summary.csv`
- `oos_yearly_summary.csv`
- `stress_test_summary.csv`
- `survivor_validation_summary.csv`
- `final_report.md`
- `final_verdict.json`