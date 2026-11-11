# Carte du dépôt après clôture

## Référence active

- `research/locked_four_sleeve_portfolio/` — construction et allocation du bloc MNQ.
- `research/mgc_final_two_sleeve_lock_20260926/` — manifeste et spécifications des deux sleeves MGC.
- `research/money_management_lab/` — règles de sortie/sizing verrouillées nécessaires au bloc MNQ.
- `research/baseline_backtests/` — moteur frozen partagé.
- `research/mgc_baseline_backtests/` — adaptation et données de session MGC.
- `research/mgc_wide_parameter_landscape/` — utilitaires de paramètres requis par le script de lock MGC.
- `research/mbt_three_sleeve_freeze_20261004/` — régions MBT gelées, votes et trades reproductibles.
- `research/mbt_nine_sleeve_topstep_20261004/` — intégration et diagnostics Topstep des neuf sleeves.
- `research/locked_eight_sleeve_portfolio_20261103/` — référence active à huit sleeves, ledger, manifeste de hashes, comparaison Topstep et seuils de la règle d'arrêt.
- `research/locked_nine_sleeve_portfolio_20261101/` — référence du 1er novembre 2026, remplacée, conservée intacte pour audit.
- `research/frozen_vote_forward_replay_20261101/` — replay forward des votes gelés MNQ/MGC.
- `research/locked_nine_sleeve_portfolio/` — référence du 4 octobre 2026, remplacée, conservée intacte pour audit.
- `strategy_registry/` — registre des mécanismes et spécifications frozen.
- `export/mnq_orb_vwap_atr_retained/` — rapport/trades ORB canonique.

## Code général

- `src/` — moteur, données, stratégies et analytics réutilisables.
- `tests/` — tests du framework; certaines suites historiques restent présentes parce qu'elles protègent du code partagé.
- `notebooks/research_main.ipynb` — notebook générique.
- `notebooks/finals/mnq_orb_retained_final.ipynb` — notebook final ORB non exécuté.

## Données

- `data/downloaded/` — téléchargements source, notamment les jeux Databento payants.
- `data/processed/parquet/` — versions normalisées dont les hashes sont référencés par les locks.
- `data/raw/` — petites sources auxiliaires.
- `data/exports/` — répertoire régénérable, nettoyé lors de la clôture.

Les doubles apparents entre `downloaded` et `processed/parquet` ne sont pas byte-identiques. Les versions processed portent les hashes utilisés dans les rapports; elles sont donc conservées.

## Archives de résultats

Les sorties lourdes des campagnes fermées ont été déplacées dans une archive externe vérifiable. Son emplacement, ses volumes et le hash de son manifeste figurent dans [`RESULT_ARCHIVE_2026_10_06.md`](RESULT_ARCHIVE_2026_10_06.md). Les conclusions synthétiques restent dans [`RESEARCH_FAILURE_REGISTER_2026.md`](RESEARCH_FAILURE_REGISTER_2026.md) et [`RESEARCH_CLOSURE_2026.md`](RESEARCH_CLOSURE_2026.md).
