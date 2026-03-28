# VWAP Reranking Summary

- Global verdict: `Aucune variante n'est assez robuste pour meriter une validation approfondie supplementaire.`
- Survivors after primary filter: `none`
- Eliminated immediately: `vwap_reclaim`

```text
             strategy_id                         role  oos_net_pnl  oos_profit_factor  oos_sharpe_ratio  pnl_slip_x2  positive_oos_splits  total_splits   prop_verdict                concentration_verdict                   final_bucket
     paper_vwap_baseline     paper_baseline_reference -3239.500000           0.978870         -0.231765 -7120.500000                    0             4 non defendable forte dependance aux meilleurs jours           reference_officielle
baseline_futures_adapted realistic_baseline_reference -3239.500000           0.978870         -0.231765 -7120.500000                    0             4 non defendable forte dependance aux meilleurs jours baseline_realiste_de_reference
            vwap_reclaim                    candidate -1593.166667           0.810555         -0.567988 -1709.166667                    0             4 non defendable forte dependance aux meilleurs jours         eliminee immediatement
```
