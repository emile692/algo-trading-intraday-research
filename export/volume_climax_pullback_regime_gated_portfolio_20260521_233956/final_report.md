# Volume Climax Pullback Regime-Gated Portfolio

## 1. Executive Summary
- M2K-only baseline: `474.78 USD`, `PF 1.91`.
- Raw M2K+MGC baseline: `845.99 USD`, `PF 2.68`.
- Best strict regime-gated portfolio: `800.29 USD`, `PF 2.20`, verdict `weak_watchlist`.
- Average MGC retention rate: `52.34%`.

## 2. Selected Rules By Fold
| fold_id | rule_id | allocation_scheme | fitted_params_json |
| --- | --- | --- | --- |
| fold_1 | always_on__conditional_equal_weight | conditional_equal_weight | {} |
| fold_2 | always_on__conditional_equal_weight | conditional_equal_weight | {} |
| fold_3 | atr_pct_mid_q30_q70__conditional_inverse_vol | conditional_inverse_vol | {"high_quantile": 0.7, "high_threshold": 0.705, "low_quantile": 0.3, "low_threshold": 0.323} |
| fold_4 | atr_pct_mid_q20_q80__conditional_equal_weight | conditional_equal_weight | {"high_quantile": 0.8, "high_threshold": 0.738, "low_quantile": 0.2, "low_threshold": 0.28600000000000003} |
| fold_5 | atr_pct_mid_q20_q80__conditional_equal_weight | conditional_equal_weight | {"high_quantile": 0.8, "high_threshold": 0.75, "low_quantile": 0.2, "low_threshold": 0.292} |

## 3. Strict Fold Breakdown
| fold_id | selected_rule_id | regime_family | allocation_scheme | fitted_params_json | train_score | test_net_pnl | test_profit_factor | test_trades | test_win_rate | test_avg_trade | test_median_trade | test_max_drawdown | test_max_daily_drawdown | test_monthly_hit_rate | test_top1_contribution_pct | test_top3_contribution_pct | m2k_only_test_net_pnl | mgc_only_test_net_pnl | raw_m2k_mgc_test_net_pnl | mgc_test_retention_rate | strict_train_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_1 | always_on__conditional_equal_weight | always_on | conditional_equal_weight | {} | 0.8721 | -42.3750 | 0.0000 | 2 | 0.0000 | -21.1875 | -21.1875 | -30.0000 | -30.0000 | 0.0000 | 0.2920 | 1.0000 | -42.3750 | 0.0000 | -21.1875 | 0.0000 | True |
| fold_2 | always_on__conditional_equal_weight | always_on | conditional_equal_weight | {} | 0.1835 | 105.8750 | 1.4439 | 18 | 0.4444 | 5.8819 | -9.5625 | -167.6250 | -52.5000 | 0.4000 | 0.9079 | 2.1405 | 130.1250 | -48.5000 | 40.8125 | 1.0000 | True |
| fold_3 | atr_pct_mid_q30_q70__conditional_inverse_vol | atr_pct_between | conditional_inverse_vol | {"high_quantile": 0.7, "high_threshold": 0.705, "low_quantile": 0.3, "low_threshold": 0.323} | 1.7797 | 199.4147 | 2.1049 | 18 | 0.4444 | 11.0786 | -7.3125 | -93.1007 | -35.2500 | 0.5556 | 0.6143 | 1.4920 | 140.4000 | -5.3000 | 67.5500 | 0.5714 | True |
| fold_4 | atr_pct_mid_q20_q80__conditional_equal_weight | atr_pct_between | conditional_equal_weight | {"high_quantile": 0.8, "high_threshold": 0.738, "low_quantile": 0.2, "low_threshold": 0.28600000000000003} | 2.0944 | 226.6250 | 2.4446 | 11 | 0.3636 | 20.6023 | -13.7500 | -105.0000 | -34.2500 | 0.5000 | 0.6828 | 1.3933 | 69.6250 | 562.0000 | 315.8125 | 0.5455 | True |
| fold_5 | atr_pct_mid_q20_q80__conditional_equal_weight | atr_pct_between | conditional_equal_weight | {"high_quantile": 0.8, "high_threshold": 0.75, "low_quantile": 0.2, "low_threshold": 0.292} | 3.4996 | 310.7500 | 7.4740 | 4 | 0.5000 | 77.6875 | 50.2500 | -12.0000 | -36.0000 | 0.5000 | 0.7924 | 1.1158 | 177.0000 | 709.0000 | 443.0000 | 0.5000 | True |

## 4. Baseline Comparison
| entity_name | selection_basis | deployable | net_pnl | profit_factor | trades | positive_folds | max_drawdown | max_daily_drawdown | win_rate | avg_trade | median_trade | monthly_hit_rate | active_months | mgc_trade_retention_rate | mgc_contribution_pnl | top1_contribution_pct | top3_contribution_pct | top5_contribution_pct | worst1_contribution_pct | worst3_contribution_pct | worst5_contribution_pct | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m2k_only_baseline | strict_train_only | False | 474.7750 | 1.9135 | 39 | 4 | -133.1250 | -52.5000 | 0.3846 | 12.1737 | -11.6250 | 0.4400 | 25 | 0.0000 | 0.0000 | 0.4739 | 1.0579 | 1.5018 | -0.1106 | -0.2614 | -0.4068 | baseline |
| mgc_only_baseline | strict_train_only | False | 1217.2000 | 3.4892 | 23 | 2 | -265.0000 | -81.5000 | 0.4783 | 52.9217 | -11.5000 | 0.6250 | 16 | 1.0000 | 1217.2000 | 0.4046 | 0.8084 | 1.0730 | -0.0670 | -0.1836 | -0.2625 | baseline |
| raw_m2k_mgc_equal_weight | strict_train_only | False | 845.9875 | 2.6773 | 62 | 4 | -107.8125 | -40.7500 | 0.4194 | 13.6450 | -5.7812 | 0.5312 | 32 | 1.0000 | 608.6000 | 0.2911 | 0.5866 | 0.8130 | -0.0482 | -0.1321 | -0.1918 | baseline |

## 5. MGC Retention
| fold_id | selected_rule_id | mgc_train_trades_base | mgc_train_trades_retained | mgc_train_retention_rate | mgc_test_trades_base | mgc_test_trades_retained | mgc_test_retention_rate | mgc_test_pnl_base | mgc_test_pnl_retained | allocation_scheme |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fold_1 | always_on__conditional_equal_weight | 4 | 4 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | conditional_equal_weight |
| fold_2 | always_on__conditional_equal_weight | 4 | 4 | 1.0000 | 3 | 3 | 1.0000 | -48.5000 | -48.5000 | conditional_equal_weight |
| fold_3 | atr_pct_mid_q30_q70__conditional_inverse_vol | 30 | 12 | 0.4000 | 7 | 4 | 0.5714 | -5.3000 | 108.2000 | conditional_inverse_vol |
| fold_4 | atr_pct_mid_q20_q80__conditional_equal_weight | 35 | 22 | 0.6286 | 11 | 6 | 0.5455 | 562.0000 | 314.0000 | conditional_equal_weight |
| fold_5 | atr_pct_mid_q20_q80__conditional_equal_weight | 46 | 29 | 0.6304 | 2 | 1 | 0.5000 | 709.0000 | 492.5000 | conditional_equal_weight |

## 6. Verdict
- Final strict verdict: `weak_watchlist`
- Fold rule path: `[{"fold_id": "fold_1", "rule_id": "always_on__conditional_equal_weight", "allocation_scheme": "conditional_equal_weight"}, {"fold_id": "fold_2", "rule_id": "always_on__conditional_equal_weight", "allocation_scheme": "conditional_equal_weight"}, {"fold_id": "fold_3", "rule_id": "atr_pct_mid_q30_q70__conditional_inverse_vol", "allocation_scheme": "conditional_inverse_vol"}, {"fold_id": "fold_4", "rule_id": "atr_pct_mid_q20_q80__conditional_equal_weight", "allocation_scheme": "conditional_equal_weight"}, {"fold_id": "fold_5", "rule_id": "atr_pct_mid_q20_q80__conditional_equal_weight", "allocation_scheme": "conditional_equal_weight"}]`
- Posthoc diagnostics remain non-deployable by construction.
