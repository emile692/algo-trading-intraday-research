# Pré-enregistrement — validation hors échantillon de la référence du 01/11/2026

- **Rédigé le** : 2026-11-15T16:37:32+01:00
- **Auteur** : audit `audit_20261115`, sur instruction de l'utilisateur. Statut : **EN ATTENTE DE VALIDATION**.
- **Table des seuils** : [`outputs/04_criteria.csv`](outputs/04_criteria.csv), sha256 `5acd6272217653096c2fea1b4732dab491853bd437971773ba559e648d45e901`.
- **Empreinte de ce fichier** : `preregistration.sha256`. Toute modification après validation invalide le pré-enregistrement.

## 1. Données évaluées

La phase 3 a établi qu'**aucune donnée vierge n'existe sur disque**. Ce pré-enregistrement porte donc sur deux évaluations :

- **(P) Suivi prospectif, évaluation principale.** Les séances postérieures au 2026-11-03, pas encore téléchargées. Les critères sont lus aux horizons H = 20, 60, 120 et 250 séances. Chaque lecture se fait **une seule fois**, quand l'horizon est atteint, et ne se refait jamais.
  - Sleeve : H compte les séances de son instrument, à partir de la première séance après le 2026-11-03.
  - Book : H compte les séances du calendrier union.
- **(O) Option, à valider séparément : S01 one-shot sur 2026-03-20 → 2026-10-02.** Pour S01 seul. « Partiellement contaminé » : la version `baseline_fixed`, qui partage l'entrée, la sortie et le sizing, a déjà été lue sur cette fenêtre. H est le nombre de séances MNQ de la fenêtre. Le seuil est lu dans la table à cet H, avant tout calcul de PnL.

Ni (P) ni (O) n'est chargé avant votre validation écrite.

## 2. Configuration testée

Référence `nine_sleeve_equal_contract_reference_20261101`, avec un S01 remplacé : **S01 est l'ensemble ATR 18 de la fiche 01** (décision de l'utilisateur du 2026-11-15), et non `baseline_fixed` que contient le lock. Une copie entière par sleeve. Exécution **séparée** : chaque sleeve est rejouée seule, son PnL est reporté seul, et le PnL du book est la somme des sleeves. Le netting n'est pas testé ici. Coûts nominaux : 1,25 USD par côté (MNQ, MGC), 1,41 USD (MBT), 1 tick de slippage par côté.

| Code | Sleeve | Définition | Fichiers gelés (sha256) |
|---|---|---|---|
| S01 | `mnq_orb_vwap_atr` | ensemble ATR 18 de la fiche 01 | `research/signal_sheet_01_atr18/thresholds_is.csv` `5e2ff402faa5ada1…`<br>`research/signal_sheet_01_atr18/summary.json` `80aab9d844a368ac…`<br>`scripts/rebuild_signal_sheet_01_atr18.py` `b2ea84eec45520ab…`<br>`src/strategy/orb.py` `c2151d1347a7265d…` |
| S02 | `mnq_volatility_breakout` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/volatility_breakout_42.json` `e3e912add4f77be6…` |
| S03 | `mnq_compression_breakout` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/mnq_compression_breakout_vote_27.json` `4666855d602f315e…` |
| S04 | `mnq_atr_trend_flip` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/mnq_atr_trend_flip_vote_9.json` `25f751eb91b8ab39…` |
| S05 | `mgc_volatility_breakout` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/mgc_volatility_breakout_vote_9.json` `a299d91d77e93033…` |
| S06 | `mgc_gap_continuation` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/mgc_gap_continuation_vote_9.json` `db0d016fb314d416…` |
| S07 | `mbt_rolling_range_breakout` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/mbt_rolling_range_breakout_vote_6.json` `d498bb725d38abdf…` |
| S08 | `mbt_trend_pullback_ma_structure` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/mbt_trend_pullback_ma_structure_vote_9.json` `ebad020847e81698…` |
| S09 | `mbt_volatility_breakout` | vote gelé | `strategy_registry/psf_mnq/frozen_ensembles/mbt_volatility_breakout_vote_12.json` `03dc629dde4b389f…` |

Exigences d'exécution :

- S01 utilise les seuils figés de `thresholds_is.csv`. Ils ne sont **pas** recalibrés : le découpage 70/30 du script de reconstruction se déplacerait sur un fichier plus long.
- S02 à S09 utilisent les cellules et le vote de leurs manifestes, avec le moteur du repo, sans changement.
- Books évalués : **Book 8** (S01, S02, S03, S05, S06, S07, S08, S09) et **Book 9** (les mêmes, plus S04).

## 3. Métriques et conventions

- PnL quotidien net en USD, coûts nominaux, jours sans trade inclus.
- **PnL cumulé à H** : somme des H premières séances.
- **maxDD à H** : drawdown maximal clôture-à-clôture de l'equity cumulée depuis 0 sur les H premières séances (valeur ≤ 0).
- Sharpe (√252), nombre de trades et pire jour : reportés, sans critère.

## 4. Distribution de référence

- Bootstrap stationnaire de Politis-Romano sur l'historique de chaque série jusqu'au 2026-03-19 : MNQ/MGC depuis le 2019-05-06, MBT depuis le 2021-05-03, books depuis le 2021-05-03 sur le calendrier union.
- Bloc moyen **20** pour les critères. Le bloc 5 est calculé et reporté, mais pas utilisé.
- 10 000 chemins de 260 séances, seed 20261119.
- Les seuils de tous les H de 1 à 260 sont dans la table hachée ci-dessus. Les historiques incluent la validation réutilisée : la distribution reflète donc la performance déjà vue, optimiste pour les sleeves sélectionnées sur cette fenêtre.

## 5. Critères (lus tels quels, sans réinterprétation)

- **Sleeve — ÉCHEC** si son PnL cumulé à H est **strictement inférieur** au p5 de sa distribution bootstrap à H. **COHÉRENT** sinon.
- **Book — ÉCHEC** si son PnL cumulé à H est hors de [p5 ; p95], ou si son maxDD à H est hors de [p5 ; p95]. Bornes incluses dans l'intervalle. Le sens est reporté : « ÉCHEC-BAS » si PnL < p5 ou maxDD < p5, « ÉCHEC-HAUT » si PnL > p95 ou maxDD > p95 (drawdown plus faible que prévu). **COHÉRENT** sinon.
- Aucun critère de succès plus fort que la cohérence.

Seuils de PnL aux horizons de lecture (bloc 20) :

| Série | Historique (séances) | p5 PnL H=20 | p5 PnL H=60 | p5 PnL H=120 | p5 PnL H=250 |
|---|---:|---:|---:|---:|---:|
| S01 | 1747 | −589 | −718 | −679 | −148 |
| S02 | 1747 | −1 228 | −1 863 | −2 326 | −2 321 |
| S03 | 1747 | −264 | −427 | −602 | −760 |
| S04 | 1747 | −1 098 | −1 811 | −2 371 | −3 129 |
| S05 | 1775 | −283 | −680 | −972 | −1 250 |
| S06 | 1775 | −444 | −701 | −913 | −1 102 |
| S07 | 952 | −406 | −675 | −820 | −994 |
| S08 | 952 | −539 | −879 | −1 104 | −1 233 |
| S09 | 952 | −399 | −502 | −480 | −238 |
| Book 8 | 1261 | −1 737 | −1 941 | −1 081 | 1 987 |
| Book 9 | 1261 | −2 077 | −2 566 | −1 616 | 1 750 |

Seuils des books :

| Book | H | PnL p5 | PnL p95 | maxDD p5 | maxDD p95 |
|---|---:|---:|---:|---:|---:|
| Book 8 | 20 | −1 737 | 4 104 | −2 510 | −366 |
| Book 8 | 60 | −1 941 | 8 054 | −4 448 | −834 |
| Book 8 | 120 | −1 081 | 12 860 | −5 388 | −1 208 |
| Book 8 | 250 | 1 987 | 21 942 | −6 552 | −1 870 |
| Book 9 | 20 | −2 077 | 5 004 | −3 019 | −589 |
| Book 9 | 60 | −2 566 | 10 490 | −5 202 | −1 065 |
| Book 9 | 120 | −1 616 | 16 214 | −6 364 | −1 566 |
| Book 9 | 250 | 1 750 | 27 133 | −7 903 | −2 270 |

## 6. Taux de fausses alarmes attendu si le modèle est vrai

Ce taux est calculé sur les mêmes chemins bootstrap. Il sert à lire le verdict : avec 8 sleeves et 4 lectures, plusieurs ÉCHEC isolés sont attendus même sans dégradation.

| Série | P(échec) H=20 | H=60 | H=120 | H=250 | P(au moins un échec sur 4 lectures) |
|---|---:|---:|---:|---:|---:|
| S01 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 14,0 % |
| S02 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 13,2 % |
| S03 | 4,9 % | 5,0 % | 5,0 % | 5,0 % | 13,0 % |
| S04 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 14,0 % |
| S05 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 12,5 % |
| S06 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 12,9 % |
| S07 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 13,4 % |
| S08 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 13,0 % |
| S09 | 5,0 % | 5,0 % | 5,0 % | 5,0 % | 13,6 % |
| Book 8 | 16,1 % | 17,1 % | 17,2 % | 17,1 % | 44,1 % |
| Book 9 | 16,4 % | 17,2 % | 17,1 % | 17,3 % | 44,1 % |

## 7. Engagements

1. Ces résultats ne serviront **à aucun retuning** : ni paramètre de signal, ni cluster, ni seuil de vote, ni taille, ni horaire, ni coût.
2. Chaque lecture est faite une fois. Si un bug apparaît après la lecture, il est documenté et l'évaluation s'arrête. Aucun recalcul.
3. Un ÉCHEC n'entraîne aucune action automatique. Toute décision de statut (actif, shadow) relève de l'utilisateur et de la règle d'arrêt en vigueur (phase 5).
4. Ce fichier et `04_criteria.csv` ne sont plus modifiés après validation. Leur sha256 est vérifié avant chaque lecture.

## 8. Points à confirmer à la validation

- **Lecture bilatérale du critère du book.** La mission dit « hors de l'intervalle 5 %–95 % ». Un PnL trop *bon* ou un drawdown trop *faible* serait donc classé ÉCHEC-HAUT. Je l'ai écrit ainsi ; dites-moi si seul le côté bas doit compter.
- **Option (O)** : la lancer ou non.
- **S01 en production.** La règle de netting du 15/11 trade `baseline_fixed`. Pour que (P) teste le S01 de référence, il faut calculer en forward le PnL virtuel de l'ensemble ATR 18. Je reporterai les deux, mais seul l'ATR 18 sera jugé.
