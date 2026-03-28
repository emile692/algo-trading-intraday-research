# MNQ ORB TopstepX 50K Simulation

## Scope

- Source export: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Scope principal: `oos`
- Bootstrap: `2000` paths, block size `5`
- Dataset: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet`
- Source of truth rules: 50K start, +3K target, trailing MLL 2K, no daily loss limit, consistency 50%.
- Assumption retained: no breakeven lock is added to the trailing floor; it remains 2,000 below the running high-watermark.

## Comparison

```text
             ruleset_name  nominal_rolling_pass_rate  sizing_rolling_pass_rate  nominal_rolling_fail_rate  sizing_rolling_fail_rate  nominal_bootstrap_pass_rate  sizing_bootstrap_pass_rate  nominal_bootstrap_fail_rate  sizing_bootstrap_fail_rate  nominal_bootstrap_inconsistency_rate  sizing_bootstrap_inconsistency_rate  nominal_bootstrap_delayed_pass_rate  sizing_bootstrap_delayed_pass_rate  nominal_bootstrap_failed_after_target_rate  sizing_bootstrap_failed_after_target_rate  rolling_pass_rate_edge_sizing_minus_nominal  rolling_fail_rate_edge_sizing_minus_nominal  pass_rate_edge_sizing_minus_nominal  fail_rate_edge_sizing_minus_nominal  speed_edge_nominal_minus_sizing_days_to_pass  inconsistency_edge_sizing_minus_nominal  consistency_inactive_in_sample                                                                             verdict
topstepx_50k_extended_60d                   0.587467                  0.527415                   0.412533                  0.472585                       0.5515                      0.6350                       0.4480                      0.3595                                   0.0                                  0.0                                  0.0                                 0.0                                         0.0                                        0.0                                    -0.060052                                     0.060052                               0.0835                              -0.0885                                          -7.0                                      0.0                            True consistency neutre; nominal meilleur sur rolling reel, sizing meilleur en bootstrap
    topstepx_50k_main_35d                   0.600950                  0.522565                   0.399050                  0.399050                       0.5320                      0.5835                       0.4605                      0.3185                                   0.0                                  0.0                                  0.0                                 0.0                                         0.0                                        0.0                                    -0.078385                                     0.000000                               0.0515                              -0.1420                                          -7.0                                      0.0                            True consistency neutre; nominal meilleur sur rolling reel, sizing meilleur en bootstrap
```

## Variant Summary

```text
             ruleset_name                           variant_name  rolling_pass_rate  rolling_fail_rate  rolling_expire_rate  rolling_economic_target_without_immediate_validation_rate  rolling_delayed_pass_after_inconsistency_rate  bootstrap_pass_rate  bootstrap_fail_rate  bootstrap_expire_rate  bootstrap_economic_target_without_immediate_validation_rate  bootstrap_delayed_pass_after_inconsistency_rate
    topstepx_50k_main_35d                                nominal           0.600950           0.399050             0.000000                                                        0.0                                            0.0               0.5320               0.4605                 0.0075                                                          0.0                                              0.0
    topstepx_50k_main_35d sizing_3state_realized_vol_ratio_15_60           0.522565           0.399050             0.078385                                                        0.0                                            0.0               0.5835               0.3185                 0.0980                                                          0.0                                              0.0
topstepx_50k_extended_60d                                nominal           0.587467           0.412533             0.000000                                                        0.0                                            0.0               0.5515               0.4480                 0.0005                                                          0.0                                              0.0
topstepx_50k_extended_60d sizing_3state_realized_vol_ratio_15_60           0.527415           0.472585             0.000000                                                        0.0                                            0.0               0.6350               0.3595                 0.0055                                                          0.0                                              0.0
```

## Verdict

- `la consistency target ne change pas la hierarchie; nominal reste meilleur sur l'historique reel, sizing_3state domine le bootstrap et la survivabilite simulee`
