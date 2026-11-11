# Référence opérationnelle — neuf sleeves equal-contract

> **Statut depuis le 3 novembre 2026 : référence remplacée.** La référence active est désormais [`ACTIVE_EIGHT_SLEEVES.md`](ACTIVE_EIGHT_SLEEVES.md) : `mnq_atr_trend_flip` passe en shadow. Ce document et son répertoire de lock restent intacts pour audit.

Date de référence : **1er novembre 2026**. Statut : **LOCKED_ACTIVE_REFERENCE**. Identifiant : `nine_sleeve_equal_contract_reference_20261101`.

Le portefeuille opérationnel de référence reste le mapping entier equal-contract suivant :

```text
[1,1,1,1,1,1,1,1,1]
```

Chaque valeur représente une copie entière de la sleeve correspondante. Aucun poids fractionnaire, levier global ou sélection par PASS rate n'est appliqué.

## Sleeves verrouillées

| Ordre | Sleeve | Instrument | Copies | Définition exécutable |
|---:|---|---|---:|---|
| 1 | `mnq_orb_vwap_atr` | MNQ | 1 | sleeve historique gelée |
| 2 | `mnq_volatility_breakout` | MNQ | 1 | vote de 42 cellules |
| 3 | `mnq_compression_breakout` | MNQ | 1 | vote de 27 cellules |
| 4 | `mnq_atr_trend_flip` | MNQ | 1 | vote de 9 cellules |
| 5 | `mgc_volatility_breakout` | MGC | 1 | vote de 9 cellules |
| 6 | `mgc_gap_continuation` | MGC | 1 | vote de 9 cellules |
| 7 | `mbt_rolling_range_breakout` | MBT | 1 | vote de 6 cellules |
| 8 | `mbt_trend_pullback_ma_structure` | MBT | 1 | vote de 9 cellules |
| 9 | `mbt_volatility_breakout` | MBT | 1 | vote de 12 cellules |

Les sleeves 2 à 9 exécutent leurs ensembles de vote gelés : chaque cellule calcule une position virtuelle `−1/0/+1`, la moyenne est discrétisée vers une cible agrégée `−1/0/+1` au seuil `±0,50`, puis exécutée à l'open de la minute suivante. Pour S3 à S6, le cluster est le voisinage immédiat de l'ancien point, choisi sans critère de performance ; S7 perd une cellule en doublon exact ; S8 et S9 gardent leur région du 4 octobre. Manifestes : [`frozen_ensembles/`](../strategy_registry/psf_mnq/frozen_ensembles/).

## Référence Topstep 50K

Bootstrap mobile groupé, 8 000 chemins, blocs 3/5/10/20, seed `20261004`, fenêtre commune 2024-01-02 → 2026-03-19, coûts nominaux, règles 50K avec target 3 000 USD, MLL 2 000 USD intraday et consistance 55 %. Les deux références sont simulées **sur les mêmes trajectoires** :

| Horizon | PASS | MLL | EXPIRY | PASS de la référence du 4 octobre |
|---|---:|---:|---:|---:|
| 35 jours tradés | 30,8000 % | 50,5000 % | 18,7000 % | 35,0250 % |
| **60 jours tradés** | **38,9500 %** | **54,6125 %** | **6,4375 %** | 45,2375 % |

Le chiffre de gouvernance arrondi est **39 % de PASS à 60 jours**, soit **6,3 points de moins** que la référence remplacée, avec 6,5 points de MLL en plus. Sur la même fenêtre, le PnL du book passe de `43 225 USD` à `37 044 USD` et le Sharpe quotidien de `1,83` à `1,54`. Cette fenêtre est celle sur laquelle les anciens points S3–S6 avaient été retenus : leurs résultats y étaient optimistes, ce qui explique une partie de l'écart. Il s'agit d'estimations bootstrap rétrospectives, pas de probabilités garanties ni d'un OOS vierge.

## Sources canoniques

- Configuration verrouillée : [`locked_nine_sleeve_config.json`](../research/locked_nine_sleeve_portfolio_20261101/locked_nine_sleeve_config.json)
- Manifeste de hashes : [`lock_manifest.json`](../research/locked_nine_sleeve_portfolio_20261101/lock_manifest.json)
- PnL quotidien canonique : [`daily_pnl.csv`](../research/locked_nine_sleeve_portfolio_20261101/daily_pnl.csv)
- Script de reverrouillage et comparaison Topstep : [`relock_nine_sleeves.py`](../research/locked_nine_sleeve_portfolio_20261101/relock_nine_sleeves.py), [`topstep_summary.csv`](../research/locked_nine_sleeve_portfolio_20261101/topstep_summary.csv)
- Sélection des clusters : [`cluster_selection_cscv.json`](../research/signal_sheet_parameter_refinement/cluster_selection/cluster_selection_cscv.json)
- Référence remplacée, conservée intacte : [`locked_nine_sleeve_portfolio/`](../research/locked_nine_sleeve_portfolio/SUPERSEDED.md)

## Gouvernance

- Ne pas retuner les neuf sleeves ou les seuils de vote à partir des données déjà vues.
- Ne pas remplacer l'equal-contract par l'equal-risk entier : ce mapping n'est pas la référence active.
- Point de vigilance : le vote de `mbt_volatility_breakout` est nul sur son legacy OOS (Sharpe `−0,040`) alors que ses cellules y sont positives ; la sleeve est à revoir.
- Toute future promotion doit être versionnée et ne doit jamais écraser cette référence, la référence du 4 octobre 2026 ni la baseline historique à six sleeves.

## Frontières OOS et paper trading

- Le ledger commun historique des neuf sleeves s'arrête le **19 mars 2026**. Ce n'est pas un OOS vierge : les données avaient déjà été consultées, notamment pour choisir les clusters.
- Les données MNQ/MGC postérieures au 19 mars 2026 et MBT postérieures au 2 octobre 2026 n'ont pas servi à la sélection des clusters.
- Le book du 1er novembre 2026 démarre son suivi prospectif à la **première séance après le 1er novembre 2026**. Le paper trading ouvert le 5 octobre 2026 appartient à la référence remplacée et ne doit pas être présenté comme un historique de cette nouvelle référence.
