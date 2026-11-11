# R1 — Conditional VWAP Mean Reversion : plan de campagne

Date : 2026-11-11. Statut : **PROPOSITION — en attente de validation. Aucun backtest lancé.**

Après validation, ce plan est transcrit en `FROZEN_SPECIFICATION.json` avec empreinte sha256 (`SPEC_LOCK.json`). Un self-test synthétique doit passer avant tout calcul sur données réelles.

## 1. Principe

Six tests pré-enregistrés (deux définitions par hypothèse) sur trois instruments, MNQ, MGC et MCL, évalués séparément. MES n'est ajouté qu'en seconde étape, si MNQ ou MGC passe la porte IS.

**Aucune sélection de meilleure cellule.** Chaque test a une cellule centrale fixée ici, qui est la seule candidate. La grille autour ne sert qu'à mesurer la stabilité de la région (heatmaps). C'est la méthode retenue le 2026-11-01 après que la CSCV a montré que les règles d'optimisation surapprennent.

## 2. Conventions communes

| Élément | Règle |
|---|---|
| Sessions ET | MNQ/MES 09:30–16:00 ; MGC 08:20–13:30 ; MCL 09:00–14:30 |
| Signaux | évalués à la clôture de chaque barre de 5 minutes (minutes 4, 9, …) |
| Fenêtre d'entrée | de l'ouverture + 30 min à la fin de séance − 30 min |
| Sortie forcée | MNQ/MES 15:55 ; MGC 13:25 ; MCL 14:25, avant les fenêtres de settlement |
| Entrée | ordre au marché à l'open de la minute suivant la barre de signal, + 1 tick |
| Stop | stop au marché : pire du niveau et de l'open de la minute qui le touche, + 1 tick |
| Objectif | ordre limite, exécuté si la minute traite 1 tick au-delà ; si stop et objectif tombent dans la même minute, le stop passe en premier |
| Coûts | 1,25 USD par côté + slippage ci-dessus |
| Position | une seule à la fois par instrument et par test |
| Réarmement | après une sortie, nouvelle entrée du même côté seulement après que le score est repassé sous seuil − 0,5 |
| Séances exclues | 4 avant chaque roll (MNQ, MES), 2 avant (MGC, MCL) ; séances écourtées ou complètes à moins de 90 % |
| VWAP | Σ(prix typique · volume) / Σ volume depuis l'ouverture de séance, prix typique = (H + L + C) / 3, barres 1 min |
| σ_VWAP | √(Σ v · (prix typique − VWAP)² / Σ v), cumulé depuis l'ouverture |
| σ_jour | écart-type de log(clôture/ouverture) sur les 20 séances précédentes (au moins 15) |
| σ5 | écart-type des rendements log 5 min intra-séance sur les 20 séances précédentes |
| Unité | R = PnL net / risque initial (distance d'entrée au stop × valeur du point) ; USD pour 1 contrat |

Toutes les grandeurs ci-dessus ne dépendent que de barres closes avant le signal.

## 3. Hypothèses et définitions

### H1 — VWAP deviation reversion

Score à la barre t :
- N1 : `z_t = (C_t − VWAP_t) / σ_VWAP,t` ;
- N2 : `z_t = (C_t / VWAP_t − 1) / (σ_jour · √τ_t)`, τ_t = fraction de séance écoulée. Le prix diffuse autour du VWAP comme √temps ; N2 en tient compte, N1 utilise la dispersion observée du jour.

Condition « pas de dynamique directionnelle forte » : efficiency ratio de séance `ER_t = |C_t − O_séance| / Σ|ΔC|` sur les barres de 5 minutes depuis l'ouverture, avec `ER_t ≤ e`.

Entrée contrarienne si `|z_t| ≥ k` et `ER_t ≤ e` : short si z > 0, long si z < 0.

Sorties :
- objectif = VWAP courant (le niveau bouge) ;
- stop = entrée ∓ 1,0 × |entrée − VWAP au signal| ;
- temps maximum 60 minutes, puis sortie forcée de séance.

| Test | Grille | Cellule centrale |
|---|---|---|
| H1-N1 | k ∈ {2,0 ; 2,25 ; 2,5} × e ∈ {0,20 ; 0,30 ; sans} | k = 2,25, e = 0,30 |
| H1-N2 | k ∈ {0,75 ; 1,0 ; 1,5} × e ∈ {0,20 ; 0,30 ; sans} | k = 1,0, e = 0,30 |

Les seuils sont choisis par fréquence (audit section 6, sans performance) : environ 2,5, 1,5 et 0,5 croisements par séance sur MNQ.

### H2 — Volatility exhaustion

Fenêtre du mouvement : les 6 dernières barres de 5 minutes (30 minutes), toutes dans la séance.

- Amplitude et vitesse : `X_t = (C_t − C_{t−6}) / (σ5 · √6)`, avec `|X_t| ≥ x`.
- Décélération, deux définitions :
  - D1 contraction : `√(Σ r² sur les barres t−1, t) / √(Σ r² sur t−5 … t−2) · √2 ≤ 0,5`. Volatilité par barre des 10 dernières minutes au plus à la moitié de celle des 20 minutes précédentes.
  - D2 arrêt : rendement de la barre t de signe opposé au mouvement ou nul, et la barre t ne fait pas de nouvel extrême du mouvement.
- Volume (option) : volume moyen des barres t−1, t ≤ volume moyen des barres t−5 … t−2, soit un essoufflement de participation.

Entrée contrarienne. Sorties :
- objectif = retracement de 50 % de `C_t − C_{t−6}` ;
- stop = extrême du mouvement (plus haut ou plus bas des 6 barres) ± 1 tick, avec une distance minimale de 0,5 · σ5 · √6 · prix ;
- temps maximum 60 minutes.

| Test | Grille | Cellule centrale |
|---|---|---|
| H2-D1 | x ∈ {2,0 ; 2,5 ; 3,0} × volume {sans ; avec} | x = 2,5, sans volume |
| H2-D2 | idem | idem |

Différence avec `consecutive_exhaustion` (rejeté) : l'entrée exige ici une décélération observée, pas seulement une série de barres.

### H3 — Lunch-hour mean reversion

Fenêtre d'entrée fixe, issue du profil de volume de l'audit et non d'une performance : 11:30–13:30 ET pour MNQ, MES et MCL ; 11:30–13:00 pour MGC. Sortie forcée : 14:00 pour MNQ/MES, 13:25 pour MGC, 14:25 pour MCL.

Condition de faible activité (« sans confirmation volumique ») : volume des 30 dernières minutes ≤ médiane du volume de la même demi-heure sur les 20 séances précédentes.

| Test | Extension | Objectif | Grille | Cellule centrale |
|---|---|---|---|---|
| H3-VWAP | N1 \|z\| ≥ k | VWAP courant | k ∈ {1,75 ; 2,0 ; 2,5} × filtre volume {sans ; avec} | k = 2,0, avec filtre |
| H3-RANGE | position dans le range de séance `P = (C − L_s) / (H_s − L_s)` ≥ p (short) ou ≤ 1 − p (long) | milieu du range au signal | p ∈ {0,85 ; 0,90 ; 0,95} × filtre volume | p = 0,90, avec filtre |

Stop : entrée ∓ 1,0 × distance à l'objectif. Sortie au plus tard à la sortie forcée.

Total : 42 cellules par instrument, 126 pour MNQ, MGC et MCL. Pas d'autre paramètre.

## 4. Périodes

| Période | Dates | Instruments | Rôle |
|---|---|---|---|
| PRE | NQ 2010-06-07 → 2019-05-03 (économie MNQ) ; CL 2015-01-01 → 2021-07-09 (économie MCL) | proxys de MNQ et MCL ; MGC n'en a pas | réplication indépendante |
| IS | 2019-05-06 → 2023-12-31 (MCL depuis 2021-07-12) | tous | porte |
| OOS | 2024-01-01 → 2026-03-19 | tous | verdict, lu une seule fois, sur accord |
| EXT | 2026-03-20 → 2026-10-02 (MNQ, MGC) ; → 2026-11-10 (MCL) | | seule fenêtre vierge pour H1 sur MNQ/MGC |

Ce sont les splits gelés du repository. Rappel de l'audit : l'OOS de la famille VWAP extension a déjà été lu sur MNQ et MGC (campagne du 2026-10-06). Le verdict H1 sur ces deux actifs s'appuiera donc d'abord sur PRE et EXT.

**Walk-forward.** La cellule candidate étant fixée, rien n'est réajusté. Le walk-forward teste la règle de sélection elle-même : chaque année de 2021 à 2023, on choisit la cellule de meilleur Sharpe lissé sur le voisinage, sur les années précédentes, puis on l'applique à l'année suivante. On compare à la cellule centrale. C'est un diagnostic, pas une sélection. S'y ajoutent la CSCV/PBO sur la grille IS (12 blocs) et des blocs glissants de 252 séances.

## 5. Porte IS (par test et par instrument, cellule centrale)

| Porte | Condition |
|---|---|
| G1 | PnL quotidien net moyen > 0, t Newey-West ≥ 2,0, et p de Holm ≤ 0,05 sur les 18 tests (6 tests × 3 instruments) |
| G2 | région stable : au moins 2/3 des cellules du test nettes positives, et tous les voisins immédiats de la cellule centrale positifs |
| G3 | encore positif à coût ×2 et avec une entrée retardée d'une minute |
| G4 | plus de la moitié des années IS positives, et aucune année ne pèse plus de 50 % du PnL |
| G5 | long et short tous deux positifs ; une asymétrie n'est acceptée que si l'autre côté est ≥ 0 en brut |
| G6 | placebo : entrées à des instants tirés au hasard dans la même tranche horaire, même côté, même nombre (1 000 tirages), p ≤ 0,05 ; direction inverse négative |
| G7 | PRE (MNQ via NQ, MCL via CL) : net > 0, t ≥ 1,0. Pour MGC, porte remplacée par EXT net > 0 |

Si un test passe G1 à G7, on te demande avant de lire l'OOS. Verdict OOS : net > 0 avec t ≥ 1,0, et R moyen par trade OOS ≥ 50 % du R moyen IS (sinon « nettement dégradé »).

## 6. Sorties produites

- **Par test et par cellule** : trades, exposition (minutes en position / minutes de séance), PnL brut et net, R moyen, t NW, Sharpe, drawdown, taux de réussite, profit factor, distribution des R (quantiles, asymétrie), MAE/MFE en R sur barres 1 min.
- **Découpages** : long et short ; années civiles ; régimes de volatilité (terciles de VIX de la veille pour MNQ/MES, terciles de σ_jour pour MGC/MCL, seuils sur 252 séances passées) ; heure d'entrée par demi-heure.
- **Heatmaps** : k × e (H1), x × volume (H2), k ou p × volume (H3), pour chaque instrument et chaque période lue.
- **Stabilité** : part de cellules positives, PBO, walk-forward, blocs glissants.
- **Coûts** : ×2 et +1 tick par côté.
- **Placebos** : instants aléatoires, direction inverse, entrée retardée d'une minute.

## 7. Diversification (candidats ayant passé l'OOS uniquement)

- **Book principal** : les 6 sleeves que tu as listées (ATR Trend Flip comprise). **Book secondaire** : le lock actif à 8 sleeves (`_20261103`).
- **Fenêtres** : OOS 2024-01 → 2026-03, puis EXT jusqu'au 2026-10-02 (6 sleeves MNQ/MGC).
- **Mesures** :
  - corrélation des PnL quotidiens avec chaque sleeve et avec le book ;
  - corrélation pendant les journées négatives du book et pendant ses 5 pires drawdowns ;
  - contribution au PnL ;
  - Sharpe et max drawdown avec et sans le candidat ;
  - leave-one-out ;
  - part de variance et de drawdown en USD.
- **Sizing** :
  1. 1 contrat (le book de référence est à contrats égaux) ;
  2. risque fixe en USD : `contrats = floor(100 USD / (distance au stop × valeur du point))`, entre 1 et 5, via `resolve_position_size`.

  Aucun autre sizing.

## 8. Verdicts

| Verdict | Règle |
|---|---|
| PASS | un test passe G1–G7 et l'OOS, et apporte une valeur marginale : le Sharpe du book monte, le drawdown ne se dégrade pas de plus de 10 %, et la corrélation pendant les jours négatifs du book est ≤ 0,2 |
| CONDITIONAL PASS | passe G1–G7 mais OOS non lisible (H1 sur MNQ/MGC) ou EXT trop court ; validation forward requise |
| TRADEOFF | passe G1–G7 et l'OOS, sans valeur marginale |
| FAIL | aucune cellule centrale ne passe la porte IS |
| INCONCLUSIVE | moins de 100 trades IS sur la cellule centrale |

Les résultats négatifs sont conservés ; en cas d'échec : `HYPOTHESIS_CLOSURE.md` et fiche au registre.

## 9. Implémentation

- **Runner** : un seul script `run_r1.py`, avec les sous-commandes `selftest`, `is`, `oos`, `div` et `report`.
  - Données en barres 1 min chargées une fois ; signaux sur barres 5 min agrégées.
  - Exécution simulée minute par minute.
  - Specs et coûts repris de `src/config/settings.py`, sizing de `src/risk/position_sizing.py`, VWAP contrôlé contre `src/features/intraday.py`.
- **Self-test** sur données synthétiques :
  - injection d'une réversion connue → R positif ;
  - marche aléatoire → R ≈ −coût ;
  - perturber les minutes postérieures au signal ne change pas le signal ;
  - le stop passe avant l'objectif dans une minute ambiguë.
- **Charge de calcul** : 126 cellules sur environ 1 100 séances, quelques minutes.
