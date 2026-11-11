# Research

La chaîne active est désormais le portefeuille equal-contract à neuf sleeves; la chaîne à six sleeves reste la baseline historique reproductible.

| Dossier | Rôle |
|---|---|
| `baseline_backtests` | moteur et baselines frozen partagés |
| `money_management_lab` | overlays de sortie/sizing MNQ verrouillés |
| `locked_four_sleeve_portfolio` | bloc MNQ et allocation constrained ERC |
| `mgc_baseline_backtests` | préparation/session MGC |
| `mgc_wide_parameter_landscape` | application des paramètres MGC requise par le lock final |
| `mgc_final_two_sleeve_lock_20260926` | manifeste final des deux sleeves MGC |
| `mbt_three_sleeve_freeze_20261004` | régions et ensembles de votes MBT verrouillés |
| `locked_eight_sleeve_portfolio_20261103` | référence active equal-contract à huit sleeves (S04 en shadow, règle d'arrêt des sleeves) |
| `locked_nine_sleeve_portfolio_20261101` | référence du 1er novembre 2026 à neuf sleeves, remplacée, conservée intacte pour audit |
| `frozen_vote_forward_replay_20261101` | replay des votes gelés S02–S06 sur 2026-03-20 → 2026-10-02 |
| `book_exports` | book packages versionnés pour `algo-trading-prop-firm-simulator` (PnL quotidien, plus bas intraday, cible Topstep) |
| `locked_nine_sleeve_portfolio` | référence du 4 octobre 2026, remplacée, conservée intacte pour audit |

Les conclusions des branches supprimées sont conservées dans [`docs/RESEARCH_CLOSURE_2026.md`](../docs/RESEARCH_CLOSURE_2026.md). Ne pas recréer les campagnes fermées sur les mêmes fenêtres de données.
