# MNQ ORB VVIX + 3-State Phase 2 Validation

## Baseline And Integration

- Baseline ORB conservee: OR30 / direction `both` / RR `2.00` / VWAP confirmation `True`.
- Filtre ATR structurel conserve via l'ensemble `majority_50`; aucune modification du signal ORB.
- Survivor VVIX principal fige: `filter_drop_low__vvix_pct_63_t1`.
- Overlay sizing 3-state fige: `sizing_3state_realized_vol_ratio_15_60`.
- Dataset: `MNQ_c_0_1m_20260321_094501.parquet` | sessions IS/OOS d'origine: `1222` / `525`.

## Coverage

- Sessions selectionnees par la baseline ATR: `1134`.
- Sessions avec contexte VVIX t-1 exploitable: `1120`.
- Sessions avec bucket 3-state exploitable: `1134`.
- Univers commun teste pour les 4 variantes coeur: `1120` sessions.
- IS/OOS dans l'univers commun: `787` / `333`.

## Four Core Configurations

- `baseline_nominal`: OOS pnl `9786.50` | Sharpe `1.952` | PF `1.338` | maxDD `-3814.50`.
- `baseline_3state`: OOS pnl `28198.50` | Sharpe `2.624` | PF `1.532` | maxDD `-5950.50`.
- `baseline_vvix_nominal`: OOS pnl `9652.00` | Sharpe `2.163` | PF `1.474` | maxDD `-2129.00`.
- `baseline_vvix_3state`: OOS pnl `20716.50` | Sharpe `2.245` | PF `1.555` | maxDD `-4262.50`.

## Attribution

- Candidat tradable le plus fort sur cette phase 2: `baseline_3state` (OOS Sharpe `2.624`).
- Variante defensive la plus convaincante: `baseline_vvix_nominal` (OOS maxDD `-2129.00`).
- Incremental VVIX au-dessus du 3-state seul: Sharpe delta `-0.379`, PF delta `+0.023`, maxDD improvement `+28.4%`, trade coverage `68.8%`.
- Incremental 3-state au-dessus du VVIX seul: Sharpe delta `+0.082`, PF delta `+0.082`, maxDD improvement `-100.2%`.
- Interaction combinee vs somme additive: Sharpe excess `-0.590`, PF excess `-0.112`, expectancy excess `-6.50`, maxDD excess `+0.1%`.
- Robustesse locale: `sensitivity_vvix_pct_126_nominal` | OOS Sharpe delta vs baseline nominal `+0.043` | maxDD improvement `34.7%`.

## Direct Answers

- Le filtre VVIX survivor reste-t-il robuste une fois combine au 3-state ? Oui.
- La combinaison VVIX + 3-state est-elle meilleure que baseline + 3-state seul ? Non: le 3-state seul garde le meilleur moteur de perf, la combinaison n apportant surtout qu un profil plus defensif.
- La combinaison VVIX + 3-state est-elle meilleure que baseline + VVIX seul ? Oui, surtout via le bloc d allocation.
- Les deux blocs semblent-ils complementaires ou redondants ? Plutot redondants / additifs sans vraie synergie.
- Le gain eventuel vient-il surtout de la selection des jours, du controle du risque, ou d'une vraie amelioration du moteur ? `drawdown_reduction`.
- Y a-t-il un candidat suffisamment propre pour devenir la nouvelle baseline tradable ? Pas encore: le 3-state seul reste la reference tradable la plus solide, la combinaison restant plutot une variante defensive comparee.

## Methodology Notes

- Le signal ORB, le filtre ATR structurel, les regles d'execution et les couts du repo restent inchanges dans cette campagne.
- Le survivor VVIX et le 3-state sont appliques comme overlays geles issus de campagnes precedentes; aucun redemarrage d'une grille massive VVIX ou sizing.
- Les inputs VVIX restent strictement lagges t-1; aucune information future n'est utilisee.
- La comparaison des 4 variantes coeur se fait sur le meme univers commun de sessions pour garder l'attribution propre.

## Exports

- `screening_summary.csv`
- `validation_summary.csv`
- `full_variant_results.csv`
- `component_comparison_summary.csv`
- `regime_impact_summary.csv`
- `final_report.md`
- `final_verdict.json`
- `variants/<variant>/...`
