# R2 — Market Intraday Momentum (Gao, Han, Li, Zhou, JFE 2018) : architecture de recherche

Date : 2026-11-11. Statut : **PROPOSITION — en attente de validation. Aucun backtest lancé.**

Après validation, ce document est transcrit en `FROZEN_SPECIFICATION.json` avec empreinte sha256 (`SPEC_LOCK.json`), comme les autres campagnes du repository, avant tout calcul de performance.

## 1. Hypothèse

Le rendement entre la clôture précédente et 10:00 ET prédit le signe du rendement entre 15:30 et 16:00 ET. Signe attendu : positif (continuation).

## 2. Ce qui a déjà été vu dans ce repository

- `research/close_of_day_intraday_momentum/` (fermée le 2026-09-29, `CLOSED_COST_CONSTRAINED`) a testé un signal **voisin mais différent** : signe du rendement entre la clôture précédente et **15:30**, puis position sur les 30 dernières minutes. Résultat : 8 contrats sur 8 négatifs après coûts, dont NQ, MNQ, MES, MGC et MCL ; effet brut présent seulement sur 6E et ZN. Le rapport détaillé a disparu, seule la fiche reste.
- R2 teste le signal original du papier (clôture précédente → **10:00**). C'est une hypothèse distincte, mais le résultat précédent abaisse l'a priori sur les indices.
- Les campagnes momentum du 2026-11-11 (open→open+h, momentum glissant) ont conclu : effet brut faible, inférieur aux coûts.
- Aucun rendement de la stratégie R2 n'a été calculé avant ce document.

## 3. Règle de la campagne A (une seule définition)

| Élément | Convention |
|---|---|
| Fuseau | `America/New_York`, zone IANA |
| Clôture précédente | close de la barre 15:59 de la dernière séance cash complète (ou 12:59 si clôture anticipée), 5 jours calendaires au plus |
| Prix à 10:00 | close de la barre 09:59 |
| Signal | `sign(P_10:00 / clôture précédente − 1)`, connu à 10:00, soit 5 h 30 avant l'entrée |
| Signal nul | pas de position |
| Entrée | open de la barre 15:30, dans le sens du signal |
| Sortie | close de la barre 15:59 |
| Taille | 1 contrat |
| Coûts | 1 tick par côté + commission par côté (MNQ 3,50 USD, MES 5,00, MGC 4,50, MCL 4,50 aller-retour) |
| Séances exclues | clôtures anticipées, fériés, trous de données, roll (changement d'`instrument_id` entre clôture précédente et sortie) |
| Tolérance | aucune sur MNQ, MES, NQ ; 5 minutes sur MGC, MCL, CL |

La clôture précédente est un prix négocié à 16:00, pas le prix de settlement. Le prix de 16:00 du jour n'entre dans aucune décision.

## 4. Instruments et horloges

Testés séparément, jamais agrégés avant les résultats individuels.

| Instrument | Rôle | Horloge principale | Périodes |
|---|---|---|---|
| MNQ | priorité 1 | 09:30 / 10:00 / 15:30 / 16:00 | IS, OOS, EXT |
| MES | priorité 2 | idem | IS, OOS |
| MGC | priorité 3 | idem | IS, OOS, EXT (pas de PRE : marché vide à 15:30 avant 2019) |
| MCL | priorité 4 | idem | IS (depuis 2021-07), OOS, EXT |
| NQ | benchmark académique | idem | PRE 2010-06 → 2019-05 |
| CL avec l'économie de MCL | proxy pétrole | idem | PRE 2015-01 → 2021-07 |

**Point à trancher (MGC, MCL).** Le brief demande 15:30–16:00 ET pour tous. Or l'or settle à 13:30 et le pétrole à 14:30 : à 16:00 il n'y a pas de « clôture » sur ces marchés, et l'amplitude y est 2 à 3 fois plus faible. Proposition : garder 15:30–16:00 comme test principal (réplication littérale), et ajouter dans la campagne B une seule horloge native par actif, fixée ici et non optimisée : MGC 08:20 / 08:50 / 13:00 / 13:30, MCL 09:00 / 09:30 / 14:00 / 14:30.

## 5. Périodes

Celles du repository, fixées sans regarder les performances de R2.

| Période | Dates | Usage |
|---|---|---|
| PRE | NQ 2010-06-07 → 2019-05-03 ; CL 2015-01-01 → 2021-07-09 | réplication sur une période et un contrat distincts |
| IS | 2019-05-06 → 2023-12-31 | test principal |
| OOS | 2024-01-01 → 2026-03-19 | verdict |
| EXT | 2026-03-20 → fin des fichiers (MNQ, MGC 2026-10-02 ; MCL 2026-11-10) | information |

Séances éligibles : MNQ 1 123 / 533 / 130 (IS / OOS / EXT), MES 1 123 / 533 / 0, MGC 1 108 / 529 / 131, MCL 567 / 506 / 153, NQ PRE 2 147, CL PRE 1 030.

**Lecture de l'OOS.** La campagne A n'a aucun paramètre : rien n'est sélectionné, donc lire IS, PRE et OOS en une fois ne crée pas de biais de sélection. Proposition : A est lue sur toutes les périodes. Les variantes B et C sont lues sur PRE et IS seulement ; leur OOS n'est lu que si la campagne A passe ses seuils, et jamais pour choisir une variante.

**Walk-forward.** A n'a rien à ajuster ; le walk-forward se réduit à la stabilité par année civile et par blocs glissants de 252 séances. Pour B et C, tout seuil (volatilité, terciles) est recalculé sur une fenêtre glissante passée de 252 séances.

## 6. Campagne A — sorties

Par instrument et par période : PnL brut et net (USD par contrat et bp), Sharpe quotidien annualisé (jours sans trade = 0), rendement annualisé, drawdown, taux de réussite, profit factor, nombre de séances, distribution (quantiles, asymétrie, pire jour), long et short séparés, années civiles, terciles de volatilité (VIX de la veille, terciles glissants).

Régression de réplication : `r_dernière = a + b · r_première`, erreurs Newey-West, R². C'est le chiffre comparable au papier ; il est rapporté à part de la stratégie.

Comparaisons :

1. toujours long 15:30 → 16:00 ; toujours short ;
2. exposition directionnelle comparable : même proportion de jours longs que le signal, dates tirées au hasard (5 000 tirages) ;
3. placebo par permutation : signaux permutés entre jours à l'intérieur de chaque année (5 000 tirages) ;
4. placebo décalé : signal de la veille appliqué au jour ;
5. placebo de fenêtre : même signal appliqué à 14:30 → 15:00 et 15:00 → 15:30.

Tests : t Newey-West de la moyenne nette (lag `floor(4·(N/100)^(2/9))`), intervalle de confiance par bootstrap par blocs stationnaires, p de permutation. Correction de Holm sur les quatre instruments principaux.

Sensibilités d'exécution (non sélectionnables) : coût ×2 ; entrée à l'open de 15:31 ; entrée au close de la barre 15:30 ; sortie à l'open de 16:00 ; horaires décalés de ±5 minutes.

## 7. Seuils fixés à l'avance (par instrument)

| Seuil | Condition |
|---|---|
| S1 | IS : moyenne nette > 0, t Newey-West ≥ 2,0 |
| S2 | IS : p de permutation ≤ 0,05 |
| S3 | IS : moyenne nette supérieure à celle de l'exposition directionnelle comparable (p ≤ 0,05) |
| S4 | IS : encore positif à coût ×2 et avec entrée retardée d'une minute |
| S5 | IS : plus de la moitié des années civiles positives |
| S6 | PRE (NQ pour les indices, CL pour MCL) : pente `b` > 0 avec t ≥ 2,0 |
| S7 | OOS : moyenne nette > 0, t ≥ 1,0 |

| Verdict | Règle |
|---|---|
| PASS | S1 à S7 sur MNQ ou MES, et valeur marginale positive dans le book |
| CONDITIONAL PASS | S1 échoue mais l'effet passe S1–S7 dans un régime défini à l'avance (campagne C), en IS et en OOS |
| TRADEOFF | S1 à S7 passent, valeur marginale nulle ou négative dans le book |
| FAIL | pente brute non significative, ou effet brut inférieur aux coûts |
| INCONCLUSIVE | moins de 250 séances éligibles ou données non exécutables |

## 8. Campagne B — robustesse (après A, PRE et IS)

Cinq variantes, toutes rapportées, aucune ne remplace la référence :

1. signal normalisé : `r_première / σ`, σ = écart-type des 20 `r_première` précédents ; position seulement si |z| ≥ 0,5 ;
2. signal sans gap : rendement 09:30 → 10:00 seul ;
3. gap seul : clôture précédente → 09:30 (décompose 1 et 2) ;
4. avant-dernière demi-heure : signe du rendement 15:00 → 15:30 (second prédicteur du papier) ;
5. horloge native MGC et MCL (section 4).

Sensibilités secondaires, hors comptage : première fenêtre de 15, 45, 60 minutes ; dernière fenêtre de 15, 45, 60 minutes. Tableau complet publié, aucune cellule retenue.

Biais de sélection : 5 variantes × 4 instruments = 20 tests ; p ajustées par Holm, seuil brut équivalent ≈ 0,0025.

## 9. Campagne C — régimes (après B, PRE et IS)

Catégories fixées ici, seuils par terciles glissants sur 252 séances passées, tous connus avant 15:30 :

- volatilité : VIX de clôture de la veille ; en complément, amplitude réalisée 09:30 → 10:00 ;
- volume : volume de la première demi-heure / médiane des 20 séances précédentes (semaines de roll exclues) ;
- annonces macro : **non testable avec les données locales** (10 événements). Option : jours de décision FOMC seulement, à partir du calendrier public de la Fed, si tu le souhaites.

Un seul filtre à la fois, pas d'interaction. Sortie : pente et PnL net par tercile, test de différence haut − bas (Newey-West).

## 10. Diversification contre le book gelé

Book : `eight_sleeve_equal_contract_reference_20261103`, lecture seule, PnL quotidien 2024-01-02 → 2026-03-19. L'analyse porte donc sur la fenêtre OOS uniquement, avec 1 contrat par instrument.

Sorties : corrélation des PnL quotidiens avec le book et avec chaque sleeve ; corrélation les jours négatifs du book ; contribution au PnL ; Sharpe et drawdown avec et sans la stratégie ; leave-one-out à 9 sleeves ; contribution au risque en USD (part de variance et de drawdown) ; PnL de la stratégie pendant les cinq pires drawdowns du book. Standalone et marginal sont jugés séparément.

Le book actif, ses paramètres et ses fichiers ne sont pas modifiés.

## 11. Limites par rapport au papier

| Papier | Ici | Conséquence |
|---|---|---|
| SPY, 1993–2013 | futures micro, 2019–2026 ; NQ depuis 2010-06 | seulement 3,5 ans de recouvrement ; pas de S&P avant 2019 |
| ETF cash, pas de cotation la nuit | futures cotés 23 h sur 24 | à 09:30 le future a déjà intégré la nuit ; le « gap » n'est pas un saut mais un rendement Globex négociable |
| clôture = enchère de 16:00 | dernier prix négocié avant 16:00 ; le future continue jusqu'à 17:00 | la sortie ne capture pas l'enchère de clôture |
| stratégie sans coûts dans le test principal | 1 tick + commission par côté | réplication académique (pente brute) et stratégie nette rapportées séparément |
| tous les jours de bourse | jours d'expiration trimestrielle et lendemain absents (roll `c.0`) | environ 8 séances par an perdues, dont les « quadruple witching » |
| effet plus fort en récession, forte volatilité, fort volume, jours d'annonces | pas de calendrier macro fiable ; une seule récession courte (2020) | régime macro non testable |
| actions | MGC, MCL | hors du champ principal du papier ; test d'étendue |

## 12. Livrables

`research_design.md`, `data_audit.md`, `audit_data.py`, puis après validation : `FROZEN_SPECIFICATION.json`, `SPEC_LOCK.json`, `run_campaign_a.py`, `run_campaign_b.py`, `run_campaign_c.py`, `run_diversification.py`, `results/`, `final_report.md`. En cas d'échec : `HYPOTHESIS_CLOSURE.md` et fiche dans `docs/RESEARCH_FAILURE_REGISTER_2026.md`.
