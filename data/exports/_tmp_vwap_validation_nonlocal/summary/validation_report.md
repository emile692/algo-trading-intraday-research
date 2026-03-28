# VWAP Pullback Continuation Validation

## Executive Summary

- Final category: `non defendable en l'etat`
- Validation is based on the corrected next-open discrete execution semantics, not on the legacy discovery run.

## Nominal Corrected Run

```text
  scope     net_pnl  profit_factor  sharpe_ratio  max_drawdown  total_trades  expectancy_per_trade
overall 3948.664286       1.022287      0.162656  -7548.971429          4653              0.848628
     is 2707.917857       1.024768      0.184330  -4683.475000          3274              0.827098
    oos 1240.746429       1.018290      0.135068  -7548.971429          1379              0.899744
```

## Final Verdict Blocks

- Robustesse statistique: Le run de decouverte historique n'est pas defendable tel quel a cause d'une fuite temporelle discrete entree-close/same-bar; le rerun corrige ne montre plus qu'un edge tres faible.
- Robustesse execution: Le profil se degrade vite des qu'on stresse l'execution.
- Robustesse parametrique: La topologie locale reste etroite ou fragile.
- Robustesse temporelle: Le resultat reste tres dependant du split choisi.
- Viabilite prop firm: La strategie reste trop fragile pour un challenge prop en l'etat.

## Artifact Pointers

- Stress: `data\exports\_tmp_vwap_validation_nonlocal\stress\stress_test_summary.csv`
- Local robustness: not generated
- Multi-split: `data\exports\_tmp_vwap_validation_nonlocal\multi_split\split_summary.csv`
- Concentration: `data\exports\_tmp_vwap_validation_nonlocal\concentration\concentration_summary.csv`
- Challenge mode: `data\exports\_tmp_vwap_validation_nonlocal\challenge_mode\challenge_mode_summary.csv`
- Cross instrument: not generated
