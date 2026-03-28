# MNQ ORB VVIX Sizing Modulation Validation

## Baseline And Scope

- Baseline ORB conservee: OR30 / direction `both` / RR `2.00` / VWAP confirmation `True`.
- Filtre ATR structurel conserve via l'ensemble `majority_50`; aucun changement du signal ORB ni des hypotheses d'execution du repo.
- Dataset: `MNQ_c_0_1m_20260321_094501.parquet` | sessions IS/OOS d'origine: `1222` / `525`.
- Univers commun teste pour toutes les comparaisons coeur: `1120` sessions.

## Calibration

- Modulateur VVIX primaire choisi sur IS: `candidate_low_penalty_025__vvix_pct_63_t1` via `vvix_pct_63_t1`.
- Mode de combinaison retenu sur IS: `candidate_low_penalty_025__vvix_pct_63_t1` (mode `multiplicative`).
- Survivor hard-filter de reference: `filter_drop_low__vvix_pct_63_t1`.
- 3-state de reference: `sizing_3state_realized_vol_ratio_15_60`.
- Buckets VVIX calibres sur IS seulement puis figes sur OOS.

## Four Core Configurations

- `baseline_nominal`: OOS pnl `9786.50` | Sharpe `1.952` | PF `1.338` | maxDD `-3814.50`.
- `baseline_3state`: OOS pnl `28198.50` | Sharpe `2.624` | PF `1.532` | maxDD `-5950.50`.
- `baseline_vvix_modulator`: OOS pnl `29922.50` | Sharpe `2.345` | PF `1.499` | maxDD `-5810.50`.
- `baseline_3state_vvix_modulator`: OOS pnl `22249.00` | Sharpe `2.405` | PF `1.586` | maxDD `-4262.50`.
- `reference_vvix_hard_filter_nominal`: OOS pnl `9652.00` | Sharpe `2.163` | PF `1.474`.

## Attribution

- Modulator vs hard filter nominal: Sharpe delta `+0.181`, PF delta `+0.026`, net pnl ratio `3.100`.
- VVIX modulator seul vs 3-state: Sharpe delta `-0.279`, PF delta `-0.033`, maxDD improvement `+2.4%`.
- VVIX + 3-state vs 3-state seul: Sharpe delta `-0.219`, PF delta `+0.054`, maxDD improvement `+28.4%`, trade coverage `74.8%`.
- Interaction combinee vs somme additive: Sharpe excess `-0.611`, PF excess `-0.107`, expectancy excess `-76.17`, maxDD excess `+96.6%`.
- En OOS, bucket VVIX `low`: `41` sessions | avg multiplier VVIX `0.25` | avg final multiplier combine `0.12`.

## Direct Answers

- Le VVIX fonctionne-t-il mieux comme sizing modulator que comme hard filter ? Oui, dans le cadre de cette campagne compacte.
- Le VVIX modulator seul fait-il mieux que le nominal ? Oui.
- Le VVIX modulator seul fait-il mieux que le 3-state ? Non, le 3-state garde l avantage principal.
- La combinaison VVIX modulator + 3-state ameliore-t-elle le 3-state ? Non, ou seulement de facon trop marginale pour declasser le 3-state.
- Le gain eventuel vient-il surtout du drawdown, de la regularite, ou d'un vrai moteur de perf ? `drawdown_reduction`.
- Faut-il promouvoir le VVIX comme bloc de sizing ? `keep_as_defensive_modulator_only`.

## Methodology Notes

- La comparaison coeur se fait sur le meme univers commun de sessions VVIX + 3-state pour garder l'attribution propre.
- Le signal ORB, les couts, le slippage, les commissions et le filtre ATR baseline restent inchanges.
- Les features VVIX sont strictement laggees t-1.
- Aucune reouverture d'une large campagne VIX/VVIX: seul un set compact de modulateurs interpretable autour de `vvix_pct_63_t1` est teste, avec une sensibilite locale sur `vvix_pct_126_t1`.

## Exports

- `screening_summary.csv`
- `validation_summary.csv`
- `full_variant_results.csv`
- `sizing_component_comparison_summary.csv`
- `vvix_modulation_regime_summary.csv`
- `final_report.md`
- `final_verdict.json`
- `variants/<variant>/...`
