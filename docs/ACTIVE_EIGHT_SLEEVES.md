# Référence opérationnelle — huit sleeves equal-contract

Date de référence : **3 novembre 2026**. Statut : **LOCKED_ACTIVE_REFERENCE**. Identifiant : `eight_sleeve_equal_contract_reference_20261103`.

Le portefeuille opérationnel de référence est le mapping entier equal-contract suivant, sans `mnq_atr_trend_flip` :

```text
[1,1,1,1,1,1,1,1]
```

Chaque valeur représente une copie entière de la sleeve correspondante. Aucun poids fractionnaire, levier global ou sélection par PASS rate n'est appliqué.

## Sleeves verrouillées

| Ordre | Sleeve | Instrument | Copies | Définition exécutable |
|---:|---|---|---:|---|
| 1 | `mnq_orb_vwap_atr` | MNQ | 1 | sleeve historique gelée |
| 2 | `mnq_volatility_breakout` | MNQ | 1 | vote de 42 cellules |
| 3 | `mnq_compression_breakout` | MNQ | 1 | vote de 27 cellules |
| 5 | `mgc_volatility_breakout` | MGC | 1 | vote de 9 cellules |
| 6 | `mgc_gap_continuation` | MGC | 1 | vote de 9 cellules |
| 7 | `mbt_rolling_range_breakout` | MBT | 1 | vote de 6 cellules |
| 8 | `mbt_trend_pullback_ma_structure` | MBT | 1 | vote de 9 cellules |
| 9 | `mbt_volatility_breakout` | MBT | 1 | vote de 12 cellules |
| 4 | `mnq_atr_trend_flip` | MNQ | **0** | **SHADOW_FORWARD_ONLY**, vote de 9 cellules suivi sans allocation |

Les huit sleeves actives gardent octet pour octet les sources gelées du 1er novembre 2026 ; le test du lock le vérifie. Les numéros S01–S09 des fiches signal restent inchangés.

## Pourquoi S04 sort

Sur la fenêtre du replay forward, du 20 mars au 2 octobre 2026 (141 séances MNQ/MGC jamais utilisées pour la sélection), le vote gelé de S04 perd `6 363,50 USD`, avec un Sharpe de `−2,71`. Sa pire fenêtre de 141 séances entre 2019 et le 19 mars 2026 était de `−3 267 USD` : la perte forward est environ deux fois pire que tout son historique. S04 portait aussi 30 % du risque du book et une corrélation de `0,57` avec S02.

Sur la même fenêtre, le bloc MNQ/MGC passe de `−3 625 USD` (Sharpe `−0,81`, drawdown `−8 897 USD`) à `+2 739 USD` (Sharpe `0,83`, drawdown `−2 476 USD`) sans S04. **Cette fenêtre a motivé la décision : elle ne peut plus servir à la valider.** Seul le suivi prospectif après le 3 novembre 2026 jugera ce book.

## Référence Topstep 50K

Bootstrap mobile groupé, 8 000 chemins, blocs 3/5/10/20, seed `20261004`, fenêtre commune 2024-01-02 → 2026-03-19, coûts nominaux, règles 50K avec target 3 000 USD, MLL 2 000 USD intraday et consistance 55 %. Les trois références sont simulées **sur les mêmes trajectoires** ; le harnais reproduit exactement les PASS publiés des deux précédentes avant d'écrire quoi que ce soit.

| Horizon | PASS | MLL | EXPIRY | PASS 9 sleeves (1er nov.) | PASS 9 sleeves (4 oct.) |
|---|---:|---:|---:|---:|---:|
| 35 jours tradés | 30,6750 % | 37,9625 % | 31,3625 % | 30,8000 % | 35,0250 % |
| **60 jours tradés** | **45,5875 %** | **44,9500 %** | **9,4625 %** | 38,9500 % | 45,2375 % |

Sur la fenêtre commune, le book passe de `37 044 USD` à `31 661 USD`, mais le Sharpe quotidien remonte de `1,54` à `1,75` et le drawdown intraday maximal de `−7 256 USD` à `−6 060 USD`. Le PASS à 60 jours gagne 6,6 points, le MLL en perd 9,7 ; à 35 jours, le PASS est stable et les comptes expirent plus souvent, faute de rendement suffisant. Sur cette fenêtre, S04 était encore rentable : le gain Topstep vient de la baisse du risque, pas de la suppression de pertes. Ce sont des estimations bootstrap rétrospectives, pas des probabilités garanties.

## Règle d'arrêt des sleeves

Enregistrée le 3 novembre 2026, elle s'applique prospectivement à toutes les sleeves, actives ou en shadow :

- **Mesure** : somme du PnL quotidien net sur les 126 dernières séances du calendrier de la sleeve.
- **Seuil** : la pire valeur de cette même somme sur l'historique gelé de la sleeve ([`sleeve_stop_rule_thresholds.csv`](../research/locked_eight_sleeve_portfolio_20261103/sleeve_stop_rule_thresholds.csv)). Le seuil ne se déclenche donc jamais sur l'historique ; un 1er percentile se serait déclenché sur 1 % des fenêtres passées, S04 dès avril 2020.
- **Action** : la sleeve passe en `SHADOW_FORWARD_ONLY` dans un nouveau lock versionné, sans aucun retuning.
- **Réintégration** : uniquement par une nouvelle décision versionnée, après au moins 252 séances en shadow.

Rejouée à titre descriptif sur la fenêtre forward, la règle aurait coupé S04 le 17 juillet 2026, et elle est encore franchie au 2 octobre. Elle aurait aussi coupé **S05** `mgc_volatility_breakout` du 28 juillet au 25 août 2026, à `−1 789 USD` pour un seuil de `−1 021 USD`, surtout à cause de pertes de février–mars 2026. S05 est depuis remontée à `+700 USD` et n'est plus sous le seuil. Elle reste donc active, mais c'est le premier candidat à surveiller.

## Exécution : netting par instrument (règle active depuis le 15 novembre 2026)

Les sleeves ne sont plus envoyées séparément. Le mapping `[1,1,1,1,1,1,1,1]`, les signaux et les tailles restent ceux du lock ; seule l'exécution change.

| Instrument | Position nette envoyée |
|---|---|
| MNQ | S01 + S02 + S03 (S01 à son sizing natif de 1 à 4 contrats, S04 en shadow exclue) |
| MGC | S05 + S06 |
| MBT | S07 + S08 + S09 |

- **Ordres.** Seul l'écart entre la cible nette et la position du compte est envoyé, à l'open de la minute suivante. Le stop et la cible de S01 sont des ordres stop et limite indépendants. Leur quantité est l'effet qu'aurait la sortie de S01 sur la cible nette, recalculée chaque minute.
- **Limites Topstep.** Un micro compte 0,1 lot, un MBT 1 lot (à confirmer dans TopstepX). Combine 50K : 5 lots. XFA : 2 lots sous 1 500 USD de solde, 3 lots jusqu'à 2 000, puis 5. Une entrée qui dépasserait la limite est réduite. Une position ouverte n'est jamais coupée. Quand de la place se libère, l'événement suivant sur l'instrument rejoint la cible nette, même si cet événement est une sortie (rattrapage).
- **Aucun overlay.** Pas de stop journalier, de plafond MBT, de sizing par la vol ni de verrou de gain : tous ont été testés et rejetés.
- **PnL.** Le PnL brut est identique à la somme des sleeves. Les coûts sont légèrement plus bas (10 506 contre 10 548 USD sur 1 203 séances IS). Chaque sleeve garde son PnL virtuel et sa règle d'arrêt.

Règle gelée et empreintes : [`netting_rule_20261115.json`](../strategy_registry/execution/netting_rule_20261115.json), vérifié par [`test_active_netting_rule.py`](../tests/test_active_netting_rule.py). Étude, verdict `ADOPT_AGGREGATION_ONLY` : [clôture](../research/book_aggregation_risk_r1_20261112/HYPOTHESIS_CLOSURE.md). Exemples réels : [Book, section 6](signal_sheets/BOOK.md).

**Défaut connu du PnL de référence.** [`daily_pnl.csv`](../research/locked_eight_sleeve_portfolio_20261103/daily_pnl.csv) aligne MGC et MBT sur le calendrier MNQ, ce qui écarte 8 séances de 2024-01 → 2026-03 (−509 USD). Les chiffres Topstep de ce lock en sont légèrement optimistes. Une correction passe par un nouveau lock versionné.

## Sources canoniques

- Configuration verrouillée : [`locked_eight_sleeve_config.json`](../research/locked_eight_sleeve_portfolio_20261103/locked_eight_sleeve_config.json)
- Manifeste de hashes : [`lock_manifest.json`](../research/locked_eight_sleeve_portfolio_20261103/lock_manifest.json)
- PnL quotidien canonique : [`daily_pnl.csv`](../research/locked_eight_sleeve_portfolio_20261103/daily_pnl.csv)
- Script de verrouillage et comparaison Topstep : [`relock_eight_sleeves.py`](../research/locked_eight_sleeve_portfolio_20261103/relock_eight_sleeves.py), [`topstep_summary.csv`](../research/locked_eight_sleeve_portfolio_20261103/topstep_summary.csv)
- Contexte forward et rejeu de la règle d'arrêt : [`forward_context.json`](../research/locked_eight_sleeve_portfolio_20261103/forward_context.json), [replay des votes gelés](../research/frozen_vote_forward_replay_20261101/)
- Références remplacées, conservées intactes : [neuf sleeves du 1er novembre](../research/locked_nine_sleeve_portfolio_20261101/SUPERSEDED.md), [neuf sleeves du 4 octobre](../research/locked_nine_sleeve_portfolio/SUPERSEDED.md)

## Gouvernance

- Ne pas retuner les sleeves ou les seuils de vote à partir des données déjà vues ; la fenêtre du 20 mars au 2 octobre 2026 n'est plus out-of-sample pour ce book.
- Ne pas modifier la règle d'arrêt a posteriori pour sauver ou couper une sleeve.
- Point de vigilance : le vote de `mbt_volatility_breakout` est nul sur son legacy OOS (Sharpe `−0,040`) alors que ses cellules y sont positives ; la sleeve est à revoir.
- Toute future promotion doit être versionnée et ne doit écraser aucune référence antérieure.
- Le suivi prospectif de ce book démarre à la **première séance après le 3 novembre 2026**.
