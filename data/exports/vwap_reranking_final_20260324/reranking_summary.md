# VWAP Reranking Summary

- Global verdict: `Aucune variante n'est assez robuste pour meriter une validation approfondie supplementaire.`
- Survivors after primary filter: `none`
- Eliminated immediately: `vwap_baseline_regime_filtered, vwap_baseline_trade_capped, vwap_baseline_with_killswitch, vwap_reclaim_with_prop_overlay, vwap_time_filtered_baseline`

```text
                   strategy_id                         role   oos_net_pnl  oos_profit_factor  oos_sharpe_ratio   pnl_slip_x2  positive_oos_splits  total_splits                                          prop_verdict                concentration_verdict                   final_bucket
           paper_vwap_baseline     paper_baseline_reference  -2376.500000           0.766288         -0.706673  -2989.500000                    0             4                                        non defendable forte dependance aux meilleurs jours           reference_officielle
      baseline_futures_adapted realistic_baseline_reference -20043.000000           0.900446         -1.412155 -28667.000000                    0             4                                        non defendable forte dependance aux meilleurs jours baseline_realiste_de_reference
 vwap_baseline_regime_filtered                    candidate  -9733.000000           0.873185         -2.177875 -13669.000000                    0             4                                        non defendable forte dependance aux meilleurs jours         eliminee immediatement
    vwap_baseline_trade_capped                    candidate  -1821.500000           0.976590         -0.225655  -4819.500000                    0             4                                        non defendable forte dependance aux meilleurs jours         eliminee immediatement
 vwap_baseline_with_killswitch                    candidate  -1060.500000           0.978823         -0.171220  -3031.500000                    1             4                                        non defendable forte dependance aux meilleurs jours         eliminee immediatement
                  vwap_reclaim                    candidate    151.053571           1.140522          0.125315    132.053571                    1             4 potentiellement compatible sous contraintes prudentes forte dependance aux meilleurs jours interessante mais trop fragile
vwap_reclaim_with_prop_overlay                    candidate      0.000000           0.000000          0.000000      0.000000                    0             4                                        non defendable      dependance moderee aux outliers         eliminee immediatement
   vwap_time_filtered_baseline                    candidate -11177.500000           0.929453         -0.807055 -16452.500000                    0             4                                        non defendable forte dependance aux meilleurs jours         eliminee immediatement
```
