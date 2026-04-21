# MNQ ORB + Pullback Weighting Campaign

## Protocol

- Sleeves are fixed: ORB research configuration and volume climax pullback configuration are not optimized here.
- Execution assumptions are inherited from the audited sleeve implementations.
- Portfolio calibration uses IS rows only: risk scaling factors and inverse-risk weights use IS daily volatility.
- OOS metrics are computed strictly after the common-calendar IS/OOS split.
- Weighting methods are simple and static in OOS: no adaptive recalibration.

## Variants

- Static weights: direct blend of sleeve daily returns.
- Risk-scaled weights: each sleeve is scaled to the average IS daily volatility, then fixed weights are applied.
- Inverse-risk weights: nominal weights are adjusted by inverse IS volatility, then frozen.

Nominal weight grid:

orb20_pull80, orb25_pull75, orb33_pull67, orb40_pull60, orb50_pull50, orb60_pull40, orb67_pull33, orb75_pull25, orb80_pull20

## Calibration

```json
{
  "calibration_scope": "is",
  "risk_measure": "daily_return_volatility",
  "orb_is_vol": 0.0017188041318622947,
  "pullback_is_vol": 0.003088729274434446,
  "target_is_vol": 0.0024037667031483706,
  "risk_scaled_orb_scale": 1.398511126770408,
  "risk_scaled_pullback_scale": 0.7782380680121297,
  "raw_risk_scaled_orb_scale": 1.398511126770408,
  "raw_risk_scaled_pullback_scale": 0.7782380680121297
}
```

## Source Data

```json
{
  "source": "rebuilt_or_loaded_sleeves",
  "source_path": "D:\\Business\\Trading\\VSCODE\\algo-trading-intraday-research\\data\\exports\\mnq_orb_pullback_weighting_20260419_094221\\source_daily_returns.csv",
  "dataset_path": "D:\\Business\\Trading\\VSCODE\\algo-trading-intraday-research\\data\\processed\\parquet\\MNQ_c_0_1m_20260321_094501.parquet",
  "orb": {
    "source": "rebuilt_from_minute_data",
    "config_source": "loaded_config_json:D:\\Business\\Trading\\VSCODE\\algo-trading-intraday-research\\export\\orb_research_campaign\\top_configs_prop_score.csv",
    "experiment": "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate",
    "config": {
      "baseline_entry": {
        "or_minutes": 15,
        "opening_time": "09:30:00",
        "direction": "long",
        "one_trade_per_day": true,
        "entry_buffer_ticks": 2,
        "stop_buffer_ticks": 2,
        "target_multiple": 2.0,
        "vwap_confirmation": true,
        "vwap_column": "continuous_session_vwap",
        "time_exit": "16:00:00",
        "account_size_usd": 50000.0,
        "risk_per_trade_pct": 0.5,
        "tick_size": 0.25,
        "entry_on_next_open": true
      },
      "baseline_ensemble": {
        "atr_window": 14,
        "q_lows_pct": [
          20,
          25,
          30
        ],
        "q_highs_pct": [
          90,
          95
        ],
        "vote_threshold": 0.5
      },
      "compression": {
        "mode": "weak_close",
        "usage": "soft_vote_bonus",
        "soft_bonus_votes": 1.0
      },
      "dynamic_threshold": {
        "mode": "noise_area_gate",
        "noise_lookback": 30,
        "noise_vm": 1.0,
        "threshold_style": "max_or_high_noise",
        "noise_k": 0.0,
        "atr_k": 0.0,
        "confirm_bars": 1,
        "schedule": "continuous_on_bar_close"
      },
      "exit": {
        "mode": "baseline",
        "force_exit_time": null,
        "stagnation_bars": null,
        "stagnation_min_r_multiple": 0.15,
        "partial_fraction": 0.5
      }
    },
    "trades": 298,
    "sessions": 2143
  },
  "pullback": {
    "source": "loaded_daily_equity_export",
    "directory": "D:\\Business\\Trading\\VSCODE\\algo-trading-intraday-research\\data\\exports\\volume_climax_pullback_mnq_risk_sizing_refinement_20260406_231223",
    "campaign_variant_name": "risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true",
    "alpha_variant_name": "dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2",
    "sessions": 1747
  },
  "common_sessions": 1747,
  "is_sessions": 1222,
  "oos_sessions": 525
}
```

## IS Ranking

```text
                 variant_name  net_profit_usd  cagr_pct  sharpe  sortino  max_drawdown_usd  max_daily_drawdown_usd  calmar  return_over_drawdown  composite_score
    risk_scaled__orb20_pull80       17475.174     6.441   1.997    3.451           936.529                 136.805   4.490                18.660            2.558
    risk_scaled__orb25_pull75       16750.469     6.201   2.012    3.167           810.983                 128.164   4.943                20.655            2.556
   inverse_risk__orb25_pull75       18131.750     6.655   2.012    3.167           884.692                 137.388   4.952                20.495            2.546
         static__orb33_pull67       19198.380     7.001   2.003    3.369          1009.990                 147.294   4.628                19.008            2.544
   inverse_risk__orb20_pull80       19681.496     7.156   1.997    3.451          1067.691                 151.741   4.504                18.434            2.541
         static__orb40_pull60       17531.175     6.459   2.015    3.035           862.118                 131.757   4.895                20.335            2.538
         static__orb25_pull75       21147.601     7.622   1.976    3.672          1244.738                 165.092   4.194                16.990            2.519
    risk_scaled__orb33_pull67       15602.158     5.818   2.008    2.664           784.835                 134.773   4.727                19.880            2.503
   inverse_risk__orb33_pull67       15910.097     5.922   2.008    2.664           801.799                 137.759   4.729                19.843            2.500
         static__orb20_pull80       22390.011     8.010   1.956    3.791          1397.023                 176.238   3.990                16.027            2.500
         static__orb50_pull50       15210.281     5.686   1.997    2.500           776.040                 145.138   4.650                19.600            2.478
   inverse_risk__orb40_pull60       14187.815     5.339   1.967    2.257           838.194                 155.725   3.990                16.927            2.418
    risk_scaled__orb40_pull60       14608.687     5.483   1.967    2.257           865.487                 160.878   3.991                16.879            2.414
         static__orb60_pull40       12959.342     4.916   1.907    1.976           944.755                 168.135   3.207                13.717            2.323
baseline__pullback_standalone       27550.550     9.570   1.870    2.901          2104.850                 221.000   3.370                13.089            2.318
```

## OOS Ranking

```text
              variant_name  net_profit_usd  cagr_pct  sharpe  sortino  max_drawdown_usd  max_daily_drawdown_usd  calmar  return_over_drawdown  composite_score
      static__orb50_pull50        6235.906     5.843   2.326    2.965           485.678                 153.940   6.639                12.840            2.654
 risk_scaled__orb33_pull67        6213.439     5.822   2.285    3.001           492.732                 150.048   6.609                12.610            2.650
      static__orb60_pull40        5935.211     5.569   2.418    2.732           431.593                 157.255   6.575                13.752            2.649
inverse_risk__orb40_pull60        6100.629     5.720   2.377    2.866           461.273                 155.436   6.611                13.226            2.648
inverse_risk__orb33_pull67        6327.274     5.926   2.285    3.001           502.229                 152.925   6.612                12.598            2.648
 risk_scaled__orb40_pull60        6270.174     5.874   2.377    2.866           474.766                 159.960   6.615                13.207            2.645
inverse_risk__orb50_pull50        5807.078     5.452   2.429    2.600           418.310                 158.656   6.545                13.882            2.637
 risk_scaled__orb50_pull50        6348.893     5.945   2.429    2.600           455.444                 174.201   6.558                13.940            2.626
      static__orb67_pull33        5723.926     5.376   2.424    2.510           413.882                 159.561   6.524                13.830            2.625
      static__orb40_pull60        6535.231     6.115   2.181    3.082           540.105                 150.601   6.360                12.100            2.621
 risk_scaled__orb25_pull75        6146.959     5.762   2.141    3.097           513.164                 138.778   6.264                11.979            2.620
inverse_risk__orb25_pull75        6611.001     6.183   2.141    3.097           553.977                 149.749   6.276                11.934            2.611
      static__orb33_pull67        6743.924     6.304   2.071    3.105           578.404                 148.249   6.138                11.660            2.591
inverse_risk__orb60_pull40        5544.121     5.212   2.378    2.308           404.340                 167.559   6.478                13.712            2.591
 risk_scaled__orb20_pull80        6104.524     5.723   2.040    3.102           525.879                 131.765   6.063                11.608            2.588
```

## OOS Subperiods

```text
        variant_name           scope  net_profit_usd  sharpe  max_drawdown_usd  composite_score
static__orb20_pull80  oos_first_half        5035.489   2.351           542.142            2.735
static__orb20_pull80        oos_full        7129.634   1.876           649.962            2.510
static__orb20_pull80 oos_second_half        1902.540   1.281           590.493            1.883
static__orb20_pull80   oos_year_2024        3292.544   2.130           542.142            2.596
static__orb20_pull80   oos_year_2025        2731.063   1.655           609.805            2.257
static__orb20_pull80   oos_year_2026         823.956   1.863           202.744            2.509
static__orb50_pull50  oos_first_half        4112.903   2.761           441.301            2.797
static__orb50_pull50        oos_full        6235.906   2.326           485.678            2.654
static__orb50_pull50 oos_second_half        1961.643   1.808           448.764            2.286
static__orb50_pull50   oos_year_2024        3100.231   2.724           441.301            2.730
static__orb50_pull50   oos_year_2025        2410.932   2.079           457.322            2.492
static__orb50_pull50   oos_year_2026         516.751   1.816           134.035            2.470
static__orb75_pull25  oos_first_half        3342.955   2.621           401.036            2.653
static__orb75_pull25        oos_full        5481.670   2.351           401.036            2.575
static__orb75_pull25 oos_second_half        2004.684   2.051           375.706            2.384
static__orb75_pull25   oos_year_2024        2930.958   2.678           401.036            2.639
static__orb75_pull25   oos_year_2025        2138.225   2.208           378.630            2.490
static__orb75_pull25   oos_year_2026         260.122   1.512           170.684            1.915
```

## Pairwise Correlation

```text
          scope start_date   end_date  n_days  orb_pullback_daily_corr  orb_daily_vol  pullback_daily_vol  calibration_used
             is 2019-05-06 2024-02-22    1222                  -0.0360         0.0017              0.0031              True
            oos 2024-02-23 2026-03-19     525                  -0.0433         0.0016              0.0027             False
       oos_full 2024-02-23 2026-03-19     525                  -0.0433         0.0016              0.0027             False
 oos_first_half 2024-02-23 2025-03-06     262                  -0.0649         0.0018              0.0031             False
oos_second_half 2025-03-07 2026-03-19     263                  -0.0058         0.0014              0.0022             False
  oos_year_2024 2024-02-23 2024-12-31     217                  -0.0775         0.0019              0.0028             False
  oos_year_2025 2025-01-02 2025-12-31     253                  -0.0084         0.0014              0.0025             False
  oos_year_2026 2026-01-02 2026-03-19      55                  -0.0002         0.0008              0.0032             False
```

## Conclusion

- 50/50 acceptable: yes. Static 50/50 OOS Sharpe=2.326, maxDD=485.7, score=2.654.
- Pullback overweight improves composite vs 50/50: no.
- Main production recommendation: `static__orb50_pull50` (OOS Sharpe=2.326, maxDD=485.7, score=2.654).
- Conservative / prop-safe recommendation: `static__orb75_pull25` (maxDD=401.0, max daily loss=172.1).
- Aggressive but still defendable recommendation: `static__orb20_pull80` (net=7129.6, Sharpe=1.876, maxDD=650.0).
