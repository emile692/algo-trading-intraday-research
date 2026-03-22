# ORB Research Campaign Report

This report enforces the required double reading:
1) incremental overlay on frozen baseline,
2) full re-optimization of baseline + new bricks.

## Compression Filter

### Effet marginal pur
- Baseline fixe OOS prop_score=2.7544, net_pnl=6343.00, maxDD=-1177.50.
- Meilleur overlay OOS: `compression__weak_close__soft` avec prop_score=2.4805, net_pnl=5703.00, maxDD=-1227.00.

### Effet apres re-optimisation
- Best actif: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate` prop_score=4.1939, net_pnl=5155.50, maxDD=-590.50.
- Best inactif: `full_reopt__seed__dynamic__noise_area_gate__L30__vm1p0__k0p0__atrk0p0__c1__continuous_on_bar_close` prop_score=4.0027, net_pnl=4701.50, maxDD=-590.50.

### Conclusion robuste ou non
- Conclusion: gain robuste detecte.

## VWAP Exit / Trailing

### Effet marginal pur
- Baseline fixe OOS prop_score=2.7544, net_pnl=6343.00, maxDD=-1177.50.
- Meilleur overlay OOS: `exit__trailing_vwap` avec prop_score=0.8766, net_pnl=2623.86, maxDD=-1432.63.

### Effet apres re-optimisation
- Best actif: `full_reopt__trial_0049` prop_score=3.9497, net_pnl=5120.21, maxDD=-499.50.
- Best inactif: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate` prop_score=4.1939, net_pnl=5155.50, maxDD=-590.50.

### Conclusion robuste ou non
- Conclusion: gain non robuste ou principalement lie au deplacement d'optimum.

## Dynamic Breakout Threshold

### Effet marginal pur
- Baseline fixe OOS prop_score=2.7544, net_pnl=6343.00, maxDD=-1177.50.
- Meilleur overlay OOS: `dynamic__noise_area_gate__L30__vm1p0__k0p0__atrk0p0__c1__continuous_on_bar_close` avec prop_score=4.0318, net_pnl=4701.50, maxDD=-590.50.

### Effet apres re-optimisation
- Best actif: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate` prop_score=4.1939, net_pnl=5155.50, maxDD=-590.50.
- Best inactif: `full_reopt__baseline_ref` prop_score=2.7331, net_pnl=6343.00, maxDD=-1177.50.

### Conclusion robuste ou non
- Conclusion: gain non robuste ou principalement lie au deplacement d'optimum.

## Final Answers

1. Compression en overlay pur: non.
2. Compression apres full re-opt: oui.
3. VWAP exits en overlay pur: non.
4. VWAP exits apres full re-opt: non.
5. Dynamic threshold en overlay pur: oui.
6. Dynamic threshold apres full re-opt: oui.
7. Les gains persistent-ils apres re-optimisation complete: voir sections ci-dessus.
8. Clusters robustes ou pics isoles: voir `parameter_cluster_summary.md`.
9. Config robuste recommandee: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate`.
10. Config prop-firm oriented recommandee: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate` (classement par prop_score).
