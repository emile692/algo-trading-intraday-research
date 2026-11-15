# Référence opérationnelle — huit sleeves equal-contract

Date de référence : **15 novembre 2026**. Statut : **LOCKED_ACTIVE_REFERENCE**. Identifiant : `eight_sleeve_equal_contract_reference_20261115`.

> **Relock du 15 novembre 2026.** Il remplace `eight_sleeve_equal_contract_reference_20261103` et corrige deux défauts relevés par l'audit `audit_20261115`, sans rien retuner :
> - **S01 est désormais la sleeve de la fiche 01** : le vote de 18 modèles ATR {14, 16, 18} × q_low {20, 25, 30} × q_high {80, 85}, 701 trades. Le lock du 3 novembre contenait le ledger historique `baseline_fixed` (ATR 14 × q_high {90, 95}, 823 trades).
> - **Le PnL quotidien est calculé sur le calendrier union** des trois instruments : 572 séances au lieu de 564. Les 8 vendredis d'échéance trimestrielle tradés par MGC et MBT sont réintégrés (−509,26 USD).
>
> Les sept autres sleeves actives, S04 en shadow et la règle d'arrêt sont inchangés. Sur 2024-01-02 → 2026-03-19, le book fait `30 857,96 USD` (Sharpe `1,712`, drawdown clôture `−5 870 USD`, intraday `−6 505 USD`), contre `31 661 USD` pour le lock du 3 novembre.

Le portefeuille opérationnel de référence est le mapping entier equal-contract suivant, sans `mnq_atr_trend_flip` :

```text
[1,1,1,1,1,1,1,1]
```

Chaque valeur représente une copie entière de la sleeve correspondante. Aucun poids fractionnaire, levier global ou sélection par PASS rate n'est appliqué.

## Sleeves verrouillées

| Ordre | Sleeve | Instrument | Copies | Définition exécutable |
|---:|---|---|---:|---|
| 1 | `mnq_orb_vwap_atr` | MNQ | 1 (sizing natif 1–4 contrats) | vote ATR de 18 modèles, seuils IS figés (fiche 01) |
| 2 | `mnq_volatility_breakout` | MNQ | 1 | vote de 42 cellules |
| 3 | `mnq_compression_breakout` | MNQ | 1 | vote de 27 cellules |
| 5 | `mgc_volatility_breakout` | MGC | 1 | vote de 9 cellules — `ACTIVE_UNDER_REVIEW` depuis le 15/11 |
| 6 | `mgc_gap_continuation` | MGC | 1 | vote de 9 cellules |
| 7 | `mbt_rolling_range_breakout` | MBT | 1 | vote de 6 cellules — `ACTIVE_UNDER_REVIEW` depuis le 15/11 |
| 8 | `mbt_trend_pullback_ma_structure` | MBT | 1 | vote de 9 cellules — `ACTIVE_UNDER_REVIEW` depuis le 15/11 |
| 9 | `mbt_volatility_breakout` | MBT | 1 | vote de 12 cellules — `ACTIVE_UNDER_REVIEW` depuis le 15/11 |
| 4 | `mnq_atr_trend_flip` | MNQ | **0** | **SHADOW_FORWARD_ONLY**, vote de 9 cellules suivi sans allocation |

S02 à S09 gardent octet pour octet les sources gelées du 1er novembre 2026. S01 pointe vers `research/signal_sheet_01_atr18`, avec les seuils IS de `thresholds_is.csv` figés : ils ne sont jamais recalibrés. Les tests du lock vérifient les deux. Les numéros S01–S09 des fiches signal restent inchangés.

## Pourquoi S04 sort

Sur la fenêtre du replay forward, du 20 mars au 2 octobre 2026 (141 séances MNQ/MGC jamais utilisées pour la sélection), le vote gelé de S04 perd `6 363,50 USD`, avec un Sharpe de `−2,71`. Sa pire fenêtre de 141 séances entre 2019 et le 19 mars 2026 était de `−3 267 USD` : la perte forward est environ deux fois pire que tout son historique. S04 portait aussi 30 % du risque du book et une corrélation de `0,57` avec S02.

Sur la même fenêtre, le bloc MNQ/MGC passe de `−3 625 USD` (Sharpe `−0,81`, drawdown `−8 897 USD`) à `+2 739 USD` (Sharpe `0,83`, drawdown `−2 476 USD`) sans S04. **Cette fenêtre a motivé la décision : elle ne peut plus servir à la valider.** Seul le suivi prospectif après le 3 novembre 2026 jugera ce book.

Ces chiffres de bloc incluent S01 `baseline_fixed`. Le S01 de référence (ATR 18) n'a pas été calculé sur cette fenêtre : elle est réservée par l'option (O) du [pré-enregistrement de l'audit](../audit_20261115/preregistration.md), en attente de validation.

## Référence Topstep 50K

**Lock du 15 novembre** : même harnais (bootstrap mobile groupé, 8 000 chemins, blocs 3/5/10/20, seed `20261004`, mêmes règles 50K), sur 2024-01-02 → 2026-03-19. Le calendrier union compte 572 séances, et l'échantillonneur porte sur les séances où au moins une sleeve active trade. Les trajectoires diffèrent donc de celles des locks précédents.

| Horizon | PASS | MLL | EXPIRY |
|---|---:|---:|---:|
| 35 jours tradés | 29,0250 % | 38,6750 % | 32,3000 % |
| **60 jours tradés** | **43,7500 %** | **46,1375 %** | **10,1125 %** |

Avant d'écrire quoi que ce soit, le harnais reproduit exactement les PASS publiés des locks du 4 octobre, du 1er novembre et du 3 novembre, sur leurs trajectoires d'origine. Sur ces mêmes trajectoires et sur le calendrier MNQ, passer S01 à l'ensemble ATR 18 donne un PASS à 60 jours de 45,4375 %, contre 45,5875 %. L'essentiel de la baisse vers 43,75 % vient donc du calendrier corrigé. Détail : [`topstep_harness_mnq_calendar.csv`](../research/locked_eight_sleeve_portfolio_20261115/topstep_harness_mnq_calendar.csv).

*Référence du 3 novembre, conservée pour comparaison :* bootstrap mobile groupé, 8 000 chemins, blocs 3/5/10/20, seed `20261004`, fenêtre commune 2024-01-02 → 2026-03-19, coûts nominaux, règles 50K avec target 3 000 USD, MLL 2 000 USD intraday et consistance 55 %. Les trois références sont simulées **sur les mêmes trajectoires** ; le harnais reproduit exactement les PASS publiés des deux précédentes avant d'écrire quoi que ce soit.

| Horizon | PASS | MLL | EXPIRY | PASS 9 sleeves (1er nov.) | PASS 9 sleeves (4 oct.) |
|---|---:|---:|---:|---:|---:|
| 35 jours tradés | 30,6750 % | 37,9625 % | 31,3625 % | 30,8000 % | 35,0250 % |
| **60 jours tradés** | **45,5875 %** | **44,9500 %** | **9,4625 %** | 38,9500 % | 45,2375 % |

Sur la fenêtre commune, le book passe de `37 044 USD` à `31 661 USD`, mais le Sharpe quotidien remonte de `1,54` à `1,75` et le drawdown intraday maximal de `−7 256 USD` à `−6 060 USD`. Le PASS à 60 jours gagne 6,6 points, le MLL en perd 9,7 ; à 35 jours, le PASS est stable et les comptes expirent plus souvent, faute de rendement suffisant. Sur cette fenêtre, S04 était encore rentable : le gain Topstep vient de la baisse du risque, pas de la suppression de pertes. Ce sont des estimations bootstrap rétrospectives, pas des probabilités garanties.

## Règle d'arrêt des sleeves (version du 15 novembre 2026)

Décidée le 15 novembre 2026 après l'audit, elle remplace la règle du 3 novembre. Elle s'applique à toutes les sleeves, actives ou en shadow, ainsi qu'au book. Le suivi commence à la première séance après le 3 novembre 2026 ; h désigne le nombre de séances écoulées depuis.

| Niveau | Déclencheur | Action |
|---|---|---|
| **Shadow** | **R2** : le drawdown maximal depuis le début du suivi dépasse le p99 du drawdown bootstrap à h ([courbe des seuils](../audit_20261115/outputs/05_r2_threshold_curve.csv)) | La sleeve passe en `SHADOW_FORWARD_ONLY` dans un nouveau lock versionné, sans aucun retuning |
| **Revue** | **R0** : la somme des 126 dernières séances passe sous le pire de l'historique ([seuils](../research/locked_eight_sleeve_portfolio_20261115/sleeve_stop_rule_thresholds.csv)). **R3** : la somme des 120 dernières séances passe sous son p5 | Statut `ACTIVE_UNDER_REVIEW` : suivi hebdomadaire renforcé, sans action automatique ni changement d'allocation |
| **Book** | R3 en alerte, R2 en revue générale | Aucune action automatique |

- **Pourquoi R2 seule.** C'est la seule règle qui coupe rarement une sleeve saine : environ 7 % en 250 séances (médiane des 9 sleeves en simulation). Faire passer en shadow sur « R0 ou R2 » en couperait 21 %. En contrepartie, la détection est lente : seulement 14 % des sleeves à Sharpe nul sont détectées en 250 séances. Avec 250 séances, aucune règle ne sépare de façon fiable une sleeve saine d'une sleeve morte ([audit, phase 5](../audit_20261115/05_regles_arret.md)).
- **Réintégration.** Uniquement par une nouvelle décision versionnée, après au moins 252 séances en shadow, si le PnL cumulé en shadow n'est jamais passé sous son p5(h) et qu'il est positif ou nul à 252.
- **Validité.** La courbe R2 est estimée jusqu'à h = 500, soit vers novembre 2028. Elle doit être réestimée par une décision versionnée avant cette date.
- **Application rétroactive**, décidée le 15 novembre, sur la fenêtre du 20 mars au 2 octobre 2026 :
  - **S04** reste en shadow : R2 s'est déclenchée du 9 septembre au 2 octobre ;
  - **S05** `mgc_volatility_breakout` passe en revue : R0 du 28 juillet au 25 août, R3 du 20 juillet au 18 août ;
  - **S07** `mbt_rolling_range_breakout` passe en revue : R3 le 18 août ;
  - **S08** `mbt_trend_pullback_ma_structure` passe en revue : R3 du 1er juin au 12 août ;
  - **S09** `mbt_volatility_breakout` passe en revue : R3 du 16 juin au 1er octobre, et son vote est nul sur sa validation réutilisée ;
  - aucune sleeve active ne passe en shadow, donc le book ne change pas.

Règle versionnée et seuils : [`stop_rule_20261115.json`](../strategy_registry/forward_monitoring/stop_rule_20261115.json), vérifiée par [`test_sleeve_stop_rule_20261115.py`](../tests/test_sleeve_stop_rule_20261115.py). Gabarit de suivi hebdomadaire : [`05_suivi_hebdo_gabarit.csv`](../audit_20261115/05_suivi_hebdo_gabarit.csv).

## Exécution : netting par instrument (règle active depuis le 15 novembre 2026)

Les sleeves ne sont plus envoyées séparément. Le mapping `[1,1,1,1,1,1,1,1]`, les signaux et les tailles restent ceux du lock ; seule l'exécution change.

| Instrument | Position nette envoyée |
|---|---|
| MNQ | S01 + S02 + S03 (S01 à son sizing natif de 1 à 4 contrats, S04 en shadow exclue) |
| MGC | S05 + S06 |
| MBT | S07 + S08 + S09 |

- **Ordres.** Seul l'écart entre la cible nette et la position du compte est envoyé, à l'open de la minute suivante. Le stop et la cible de S01 sont des ordres stop et limite indépendants. Leur quantité est l'effet qu'aurait la sortie de S01 sur la cible nette, recalculée chaque minute.
- **Limites Topstep.** Un micro compte 0,1 lot, un MBT 1 lot (à confirmer dans TopstepX). Combine 50K : 5 lots. XFA : 2 lots sous 1 500 USD de solde, 3 lots jusqu'à 2 000, puis 5. Une entrée qui dépasserait la limite est réduite. Une position ouverte n'est jamais coupée. Quand de la place se libère, l'événement suivant sur l'instrument rejoint la cible nette, même si cet événement est une sortie (rattrapage).
- **Aucun overlay.** Pas de stop journalier, de plafond MBT, de sizing par la vol ni de verrou de gain : tous ont été testés et rejetés. Seule exception, le stop catastrophe ci-dessous : c'est un ordre de protection, qui ne s'est jamais déclenché sur l'historique.
- **PnL.** Le PnL brut est identique à la somme des sleeves. Les coûts sont légèrement plus bas : 10 506 contre 10 548 USD sur 1 203 séances IS dans l'étude, menée avec S01 `baseline_fixed`. Avec le S01 de référence, sur 2024-01-02 → 2026-03-19, ils valent 7 637 contre 7 665 USD, pour un PnL net de 30 886 USD en netting contre 30 858 en séparé ([`netting_check.json`](../research/locked_eight_sleeve_portfolio_20261115/netting_check.json)). Chaque sleeve garde son PnL virtuel et sa règle d'arrêt.

Règle gelée et empreintes : [`netting_rule_20261115_r2.json`](../strategy_registry/execution/netting_rule_20261115_r2.json). Elle reprend la logique de [`netting_rule_20261115.json`](../strategy_registry/execution/netting_rule_20261115.json) à l'identique et l'applique au lock du 15 novembre. Elle est vérifiée par [`test_locked_eight_sleeve_portfolio_20261115.py`](../tests/test_locked_eight_sleeve_portfolio_20261115.py). Étude, verdict `ADOPT_AGGREGATION_ONLY` : [clôture](../research/book_aggregation_risk_r1_20261112/HYPOTHESIS_CLOSURE.md). Exemples réels : [Book, section 6](signal_sheets/BOOK.md).

### Stop catastrophe (décidé le 15 novembre 2026)

Le stop catastrophe protège contre une position orpheline (déconnexion, bug de plateforme ou de code). Ce n'est pas un stop alpha.

- **Un ordre stop par instrument**, posé chez le broker sur la position nette, à **3 × ATR14(J−1)** du prix moyen d'entrée.
- **L'ATR** est calculé sur des barres quotidiennes construites à partir des barres 1 minute de 09:30 à 16:00 New York. C'est la moyenne simple du true range sur 14 séances, décalée d'une séance, exprimée en points.
- **En netting** :

  | Cas | Prix moyen | Stop |
  |---|---|---|
  | ouverture depuis flat | prix du fill | posé à 3 × ATR |
  | augmentation | pondéré | recalculé |
  | réduction | conservé | niveau conservé, quantité réduite |
  | retournement | prix du fill | recalculé de l'autre côté |
  | retour à plat | — | ordre annulé |

- **Après déclenchement**, l'instrument reste à plat jusqu'à la fin de la séance. Le stop et la cible de S01 restent posés indépendamment.
- **Sur l'historique**, le stop ne s'est jamais déclenché (2019–2026 pour MNQ et MGC, 2021–2026 pour MBT). Il ne change donc rien au comportement des sleeves.
- **La journée** est bornée par la limite de perte journalière (DLL) Topstep de 1 000 USD, appliquée au niveau du compte.

Ce stop ne modifie pas la règle r2 : la clause « aucun overlay » vise les couches de risque rejetées. Règle versionnée : [`catastrophe_stop_20261115.json`](../strategy_registry/execution/catastrophe_stop_20261115.json), vérifiée par [`test_catastrophe_stop_rule.py`](../tests/test_catastrophe_stop_rule.py). Étude : [audit, phase 10](../audit_20261115/10_stop_catastrophe.md).

**Défaut de calendrier corrigé le 15 novembre.** Le lock du 3 novembre alignait MGC et MBT sur le calendrier MNQ (−509 USD sur 8 séances). Le lock du 15 novembre est sur le calendrier union.

**Défaut de données connu, non corrigé.** Le fichier MNQ `c.0` n'a aucune barre sur les vendredis d'échéance trimestrielle (27 jours de 2019 à 2025). Les sleeves MNQ n'ont donc jamais été backtestées ces jours-là, alors que la production les trade.

## Sources canoniques

- Configuration verrouillée : [`locked_eight_sleeve_config.json`](../research/locked_eight_sleeve_portfolio_20261115/locked_eight_sleeve_config.json)
- Manifeste de hashes : [`lock_manifest.json`](../research/locked_eight_sleeve_portfolio_20261115/lock_manifest.json)
- PnL quotidien canonique : [`daily_pnl.csv`](../research/locked_eight_sleeve_portfolio_20261115/daily_pnl.csv)
- Script de verrouillage et comparaison Topstep : [`relock_eight_sleeves_20261115.py`](../research/locked_eight_sleeve_portfolio_20261115/relock_eight_sleeves_20261115.py), [`topstep_summary.csv`](../research/locked_eight_sleeve_portfolio_20261115/topstep_summary.csv)
- Contexte forward et rejeu de la règle d'arrêt, lock du 3 novembre, avec S01 `baseline_fixed` : [`forward_context.json`](../research/locked_eight_sleeve_portfolio_20261103/forward_context.json), [replay des votes gelés](../research/frozen_vote_forward_replay_20261101/)
- Références remplacées, conservées intactes : [huit sleeves du 3 novembre](../research/locked_eight_sleeve_portfolio_20261103/SUPERSEDED.md), [neuf sleeves du 1er novembre](../research/locked_nine_sleeve_portfolio_20261101/SUPERSEDED.md), [neuf sleeves du 4 octobre](../research/locked_nine_sleeve_portfolio/SUPERSEDED.md)

## Gouvernance

- Ne pas retuner les sleeves ou les seuils de vote à partir des données déjà vues ; la fenêtre du 20 mars au 2 octobre 2026 n'est plus out-of-sample pour ce book.
- Ne pas modifier la règle d'arrêt a posteriori pour sauver ou couper une sleeve. **Exception documentée du 15 novembre 2026** : la règle a été changée, de R0 vers R2 seule pour le passage en shadow, alors qu'on savait déjà que R0 s'était déclenchée pour S05 sur le replay. Le changement repose sur les taux de fausses alarmes simulés par l'audit, pas sur le résultat de S05. Il est appliqué de la même façon à toutes les sleeves, S04 comprise, que la nouvelle règle maintient en shadow. Il reste néanmoins une décision prise après avoir vu les données. La prochaine modification ne pourra se faire qu'avant toute lecture forward.
- Point de vigilance : le vote de `mbt_volatility_breakout` est nul sur son legacy OOS (Sharpe `−0,040`) alors que ses cellules y sont positives. La sleeve est `ACTIVE_UNDER_REVIEW` depuis le 15 novembre 2026.
- Toute future promotion doit être versionnée et ne doit écraser aucune référence antérieure.
- Le suivi prospectif de ce book démarre à la **première séance après le 3 novembre 2026**.
