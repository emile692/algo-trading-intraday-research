# Volume Climax Pullback Micro Portfolio - Final Report

## Scope
- Motors combined without changing their alpha or execution assumptions.
- Portfolio capital reference: `50,000 USD`.
- Capped overlay rule: open-risk cap `250 USD` on the sum of active initial trade risk; exits free capacity first; simultaneous entrants are prorated pro-rata.
- MES default chosen explicitly from its robust zone: `risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true`.
- MES alternate tested inside the same zone: `risk_pct_0p0015__max_contracts_5__skip_trade_if_too_small_true`.

## OOS Leaders
- Best single: `M2K_only__standalone__core_default` | net `5674.63` | Sharpe `1.733` | maxDD `684.00` | pass `True`.
- Best pair: `M2K_MES__equal_weight_notional__core_default` | net `5832.52` | Sharpe `2.090` | maxDD `457.71` | pass `True`.
- Best 3-way: `MNQ_M2K_MES__equal_weight_notional__core_default` | net `6173.59` | Sharpe `2.162` | maxDD `372.14` | pass `True`.

```text
                                 portfolio_variant_name portfolio_family        allocation_scheme    config_bundle  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_portfolio_score  oos_pass_target_3000_usd_without_breaching_2000_dd
       MNQ_M2K_MES__equal_weight_notional__core_default        three_way    equal_weight_notional     core_default      6173.585417    2.161911            372.141667                  372.141667            12.774452                                                True
    MNQ_M2K_MES__equal_weight_risk_budget__core_default        three_way equal_weight_risk_budget     core_default      6304.818750    2.043970            387.483333                  387.483333            12.349541                                                True
    MNQ_M2K_MES__equal_weight_risk_budget__mnq_alt_perf        three_way equal_weight_risk_budget     mnq_alt_perf      6498.157939    2.143083            588.506757                  588.506757            11.833701                                                True
           M2K_MES__equal_weight_notional__core_default             pair    equal_weight_notional     core_default      5832.515625    2.090064            457.712500                  457.712500            11.675353                                                True
MNQ_M2K_MES__equal_weight_risk_budget__conservative_mix        three_way equal_weight_risk_budget conservative_mix      5543.472500    1.970250            328.285000                  328.285000            11.618906                                                True
           MNQ_M2K__equal_weight_notional__core_default             pair    equal_weight_notional     core_default      6265.175000    2.060251            527.600000                  527.600000            11.604380                                                True
        MNQ_M2K__equal_weight_risk_budget__core_default             pair equal_weight_risk_budget     core_default      6462.025000    1.936542            531.500000                  531.500000            11.284943                                                True
        MNQ_MES__equal_weight_risk_budget__core_default             pair equal_weight_risk_budget     core_default      6484.874107    1.774896            437.357143                  437.357143            11.178992                                                True
        M2K_MES__equal_weight_risk_budget__core_default             pair equal_weight_risk_budget     core_default      5864.093750    1.972206            476.670000                  476.670000            11.131232                                                True
           MNQ_MES__equal_weight_notional__core_default             pair    equal_weight_notional     core_default      6423.065625    1.764867            451.750000                  451.750000            10.997907                                                True
```

## Decision Readout
1. Meilleur single OOS: `M2K_only__standalone__core_default`.
2. Meilleure pair OOS: `M2K_MES__equal_weight_notional__core_default`.
3. Meilleur 3-way OOS: `MNQ_M2K_MES__equal_weight_notional__core_default`.
4. Meilleur portefeuille contre MNQ seul: recommended `MNQ_M2K_MES__equal_weight_notional__core_default` vs `MNQ_only__standalone__core_default` = net `-682.14` | Sharpe `+0.567` | maxDD `-466.08`.
5. Correlations journalières OOS entre moteurs default: `M2K/MNQ` `0.277`, `M2K/MES` `0.067`, `MES/MNQ` `0.432`.
6. Valeur marginale de MES dans `MNQ + M2K` sous le schema risk-budget default: net `-157.21` | Sharpe `+0.107` | maxDD `-144.02` | score `+1.065`.
7. Portefeuille recherche: `MNQ_M2K_MES__equal_weight_notional__core_default` | prop-safe: `MNQ_M2K_MES__equal_weight_risk_budget__conservative_mix` | agressif mais defendable: `MNQ_M2K_MES__capped_overlay__core_default`.
8. Verdict net: `retenir_MNQ_plus_M2K_plus_MES`.

## Interpretation
- La meilleure pair n'est pas `MNQ + M2K` mais `M2K_MES__equal_weight_notional__core_default`.
- L'ajout de MES a une vraie valeur marginale.
- Le portefeuille recommande un 3-way.
