# MNQ ORB Prop Challenge Simulation

## Perimetre

- Source export: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Scope principal: `oos`
- Bootstrap: `2000` paths en block bootstrap `5` jours
- Seed: `42`
- Variantes comparees strictement: `nominal` vs `sizing_3state_realized_vol_ratio_15_60`

## Audit Data

- Source run timestamp: `2026-03-25T15:05:10.767721`
- Dataset: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet`
- Aggregation rule: `majority_50`
- Primary decision evidence uses the OOS slice of the existing regime/sizing export.

## Rulesets

```text
         ruleset_name                 family                              resembles                                                                            description  account_size_usd  profit_target_usd  max_traded_days  daily_loss_limit_usd  static_max_loss_usd  trailing_drawdown_usd                                                                                         notes
   classic_static_30d classic_fixed_drawdown    Topstep-like fixed target challenge     Fixed target with static max loss, realized daily limit, and 30 traded-day expiry.           50000.0             3000.0               30                1000.0               2000.0                    NaN        Stylized fixed-drawdown evaluation; daily rule is enforced on realized daily PnL only.
  trailing_strict_35d      trailing_drawdown          Trailing-DD evaluation family Fixed target with end-of-day trailing drawdown, daily limit, and 35 traded-day expiry.           50000.0             3000.0               35                1000.0                  NaN                 2000.0 Trailing floor = peak equity minus trailing allowance, updated on end-of-day realized equity.
static_permissive_45d        static_drawdown Static-DD permissive evaluation family     Fixed target with wider static max loss, no daily limit, and 45 traded-day expiry.           50000.0             3000.0               45                   NaN               2500.0                    NaN             Useful to test whether the hierarchy changes once daily stop pressure is relaxed.
```

## Comparative Summary

```text
         ruleset_name  nominal_bootstrap_pass_rate  sizing_bootstrap_pass_rate  nominal_bootstrap_fail_rate  sizing_bootstrap_fail_rate  nominal_bootstrap_median_days_to_pass  sizing_bootstrap_median_days_to_pass  nominal_bootstrap_global_violation_rate  sizing_bootstrap_global_violation_rate  pass_rate_edge_sizing_minus_nominal  rolling_pass_rate_edge_sizing_minus_nominal  fail_rate_edge_sizing_minus_nominal  survival_edge_nominal_minus_sizing_global_violation  speed_edge_nominal_minus_sizing_days_to_pass                                        verdict
   classic_static_30d                       0.6085                       0.551                        0.264                      0.1285                                   13.0                                  16.0                                    0.264                                  0.1285                              -0.0575                                    -0.091902                              -0.1355                                               0.1355                                          -3.0 nominal gagne en pass rate mais paye en risque
static_permissive_45d                       0.7190                       0.702                        0.222                      0.0995                                   15.0                                  20.0                                    0.222                                  0.0995                              -0.0170                                    -0.061111                              -0.1225                                               0.1225                                          -5.0    sizing_3state plus survivant mais plus lent
  trailing_strict_35d                       0.5315                       0.551                        0.455                      0.3405                                   11.0                                  17.0                                    0.455                                  0.3405                               0.0195                                    -0.066762                              -0.1145                                               0.1145                                          -6.0    sizing_3state plus survivant mais plus lent
```

## Variant Summary

```text
         ruleset_name                           variant_name  rolling_start_pass_rate  rolling_start_fail_rate  rolling_start_expire_rate  rolling_start_median_days_to_pass  bootstrap_pass_rate  bootstrap_fail_rate  bootstrap_expire_rate  bootstrap_median_days_to_pass  bootstrap_global_max_loss_violation_rate  bootstrap_daily_loss_violation_rate
   classic_static_30d                                nominal                 0.618834                 0.345291                   0.035874                               13.0               0.6085               0.2640                 0.1275                           13.0                                    0.2640                                  0.0
   classic_static_30d sizing_3state_realized_vol_ratio_15_60                 0.526932                 0.243560                   0.229508                               14.0               0.5510               0.1285                 0.3205                           16.0                                    0.1285                                  0.0
  trailing_strict_35d                                nominal                 0.589327                 0.410673                   0.000000                               11.5               0.5315               0.4550                 0.0135                           11.0                                    0.4550                                  0.0
  trailing_strict_35d sizing_3state_realized_vol_ratio_15_60                 0.522565                 0.399050                   0.078385                               14.0               0.5510               0.3405                 0.1085                           17.0                                    0.3405                                  0.0
static_permissive_45d                                nominal                 0.683333                 0.276190                   0.040476                               13.0               0.7190               0.2220                 0.0590                           15.0                                    0.2220                                  0.0
static_permissive_45d sizing_3state_realized_vol_ratio_15_60                 0.622222                 0.256790                   0.120988                               15.0               0.7020               0.0995                 0.1985                           20.0                                    0.0995                                  0.0
```

## Assumptions

- Daily loss and trailing drawdown are enforced on realized end-of-day PnL, not intraday mark-to-market.
- Expiry is measured in traded days only, which matches the stated objective better than wall-clock sessions for a one-trade-per-day strategy.
- The bootstrap keeps path dependence simple by resampling contiguous daily blocks instead of individual days.

## Verdict Final

- `la hierarchie depend du ruleset`
