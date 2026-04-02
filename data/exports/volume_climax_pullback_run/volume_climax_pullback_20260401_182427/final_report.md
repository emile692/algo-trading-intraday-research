# Volume Climax Pullback - Verdict

- Variantes testées: **1152**.
- Survivants crédibles (PF>1 et PnL OOS>0): **107**.
- Variantes mortes: **1045**.

## Meilleure config globale
- climax_plus_bar_quality_1h_vq0p95_vl50_mb0p5_ra1p2_srNone_msNone_wkNone_sb0_tick_rr1p0_ts2_ovall_rth (MGC 1h)
- OOS pnl=575.00, PF=1.40, Sharpe=0.56

## Réponses de recherche
1. Edge volume-only: voir lignes family=pure_climax dans summary_variants.csv.
2. Contribution stretch: comparer climax_plus_stretch vs pure_climax.
3. Impact rejection: comparer climax_plus_rejection / combined_qsr vs sans rejection.
4. Actif le plus réactif: ranking_oos_by_asset.csv.
5. 1h vs 5m: ranking_oos_by_timeframe.csv (comparabilité des overlays documentée dans les noms de variants).
6. Pertinence portefeuille: survivants + stabilité IS/OOS.

## Note comparabilité 1h
- exclude_first_10m et exclude_lunch sont appliqués avec filtres horaires; sur 1h l'effet est discretisé par barres horaires et n'est pas strictement équivalent à 5m.