# Agrégation par instrument et couche de risque du book à 8 sleeves — plan d'étude

Date : 2026-11-12. Statut : **CLOS le 2026-11-13 — `ADOPT_AGGREGATION_ONLY`** ([clôture](HYPOTHESIS_CLOSURE.md)). Gelé le 2026-11-12 avant tout calcul de performance. Paramètres Topstep fournis par l'utilisateur le 2026-11-12 (section 10).

Ce plan est transcrit en `FROZEN_SPECIFICATION.json`, dont l'empreinte sha256 est dans `SPEC_LOCK.json`. En cas d'écart, c'est le JSON qui fait foi. Les self-tests de la section 8 doivent passer avant tout calcul sur données réelles.

## 1. Objet

Le book actif `eight_sleeve_equal_contract_reference_20261103` additionne huit sleeves indépendantes sur trois instruments : MNQ (S01, S02, S03), MGC (S05, S06) et MBT (S07, S08, S09). Un compte ne peut pas être long et short sur le même contrat. Il faut donc :

1. **Agréger** : une seule position par instrument, calculée à partir des sleeves, sans nouveau modèle de signal.
2. **Risque** : dimensionner les sleeves et poser des limites par instrument et sur le book, dans les contraintes d'un compte Topstep 50K.

Objectif de décision (2026-11-11) : **toucher les payouts le plus vite possible.** On juge le cycle complet Combine → XFA → payout, pas le PASS à 60 jours.

Hors périmètre : les sleeves elles-mêmes (signaux, votes, seuils, sorties) restent gelées octet pour octet. Aucune sleeve n'est ajoutée ni retirée ici. La règle d'arrêt du 2026-11-03 continue de s'appliquer sur le PnL virtuel de chaque sleeve.

## 2. Constats de départ (descriptifs, établis le 2026-11-12)

Mesurés sur les trades exportés, sur la fenêtre commune de chaque instrument :

| Instrument | Fenêtre | Positions opposées entre sleeves | \|net\| max | Contrats économisés par le netting |
|---|---|---|---:|---:|
| MGC | 2019-07 → 2026-03 | jamais | 2 | 0,5 % |
| MNQ | 2024-01 → 2026-03 | 7 / 15 / 23 jours selon la paire | 3 | 1,4 % |
| MBT | 2021-06 → 2026-10 | S08-S09 : 111 jours ; S07-S09 : 54 jours | 3 (16 % des minutes en position) | 0,1 % |

S07 et S08 sont dans le même sens pendant 83 % des minutes où S07 est en position.

**Mise en garde.** La répartition du risque du book (S02 porte un ES 95 de 585 USD, pour 799 sur le book) a été lue sur la fenêtre 2024-01 → 2026-03, qui est l'OOS du repository. Elle a motivé une partie de cette étude. Cette fenêtre ne peut donc plus servir de verdict indépendant pour la couche de risque (section 6).

## 3. Couche A — agrégation (aucun paramètre)

Chaque sleeve garde un **état virtuel** : direction d_s ∈ {−1, 0, +1}, PnL virtuel et statistiques de la règle d'arrêt. La couche B convertit cet état en contrats n_s.

**Cible nette de l'instrument i** : `N_i = Σ_s n_s · d_s`, puis plafonnement et overlays de la couche B. L'agent d'exécution n'envoie que le delta entre la cible et la position courante.

**Exécution des événements de sleeve, selon le modèle de fill natif de chaque sleeve :**

| Événement | Fill du delta net |
|---|---|
| Changement de cible d'une sleeve à vote (S02, S03, S05–S09) | open de la minute suivante, + 1 tick par côté (identique aux backtests) |
| Entrée de S01 | fill natif de S01 |
| Stop ou cible de S01 | ordre stop ou limite posé sur le compte, au prix natif de S01 (stop : + 1 tick). Sa quantité = `N_i(après sortie S01) − N_i(courant)`, recalculée à chaque minute et éventuellement nulle si le plafond est saturé |
| Sortie forcée | à l'heure de chaque sleeve ; tout le book est plat avant 16:08 ET (15:08 CT) |

S01 conserve ainsi les sorties avec lesquelles elle a été validée : son SL/TP fait partie de son alpha. Son **sizing natif (1 à 4 contrats au risque de 0,5 %) est abandonné**, car la couche B dimensionne toutes les sleeves de la même façon. Le changement est mesuré à part (section 7).

**Coûts** : 1,25 USD de commission par côté plus le slippage, appliqués aux seuls contrats réellement échangés au niveau de l'instrument.

**Propriété attendue** : sans plafond ni overlay, le PnL brut netté est égal à la somme des PnL bruts des sleeves. C'est le test T1.

## 4. Couche B — risque

Les étapes s'ajoutent l'une après l'autre. Chacune a une configuration **par défaut, la plus simple**, conservée sauf amélioration nette (section 5).

### B0 — référence
Le lock actuel : 1 unité par sleeve, S01 à son sizing natif, sans plafond. C'est le témoin de toutes les comparaisons.

### B1 — budget de risque statique
- `ES_s` : ES 95 % du PnL quotidien de la sleeve **pour 1 contrat**, sur toute la période IS de la sleeve (jours inactifs inclus). Estimé une fois et gelé.
- Poids inverses du risque : `w_s ∝ 1 / ES_s`.
- Contrats : `n_s = max(1, ⌊λ · w_s + 0,5⌋)`. On prend le plus grand λ tel que :
  - l'ES 95 du book sur la fenêtre commune IS est ≤ k · MLL ;
  - le pic de lots ne dépasse pas la limite de la section 4-B3.
- Candidats : **k ∈ {0,30 ; 0,40 ; 0,50}**. Pour situer, le book actuel est autour de 0,40 sur l'OOS.
- Aucune sleeve ne descend à 0 : retirer une sleeve relève de la règle d'arrêt, pas du sizing.

### B2 — modulation par la volatilité de l'instrument
- `σ_i,t` : écart-type des rendements log quotidiens de l'instrument sur les 20 séances précédentes (au moins 15), connu avant l'ouverture.
- `σ_ref,i` : médiane de `σ_i,t` sur l'IS, gelée.
- `m_i,t = clamp(σ_ref,i / σ_i,t ; 0,5 ; 1,5)`, puis `n_s,t = clamp(⌊n_s · m_i,t + 0,5⌋ ; 1 ; ⌈1,5 · n_s⌉)`.
- Une seule variante : activée ou non. Pas de grille sur le lookback ni sur les bornes.
- Diagnostic préalable obligatoire, sur l'IS : PnL par contrat de chaque sleeve par tercile de `σ_i,t`. Les sleeves sont des breakouts, donc probablement longues de volatilité : réduire en haute volatilité peut couper l'edge.
- On n'utilise pas la volatilité du PnL récent des sleeves. Ce PnL est trop clairsemé (S05 n'est active que 11 % des jours), et ce serait un deleveraging sur la performance passée.

### B3 — limites dures et overlays

| Élément | Règle | Candidats |
|---|---|---|
| Plafond de lots du book | lots = (micros MNQ + micros MGC) / 10 + nombre de MBT (1 MBT = 1 lot, à vérifier dans TopstepX Risk Settings). Combine 50K : 5 lots. XFA : 2 lots sous 1 500 USD de solde, 3 lots de 1 500 à 2 000, 5 lots au-dessus ; palier lu sur le solde de fin de séance précédente | gelé, aucun choix |
| Plafond \|N\| sur MBT | cible nette tronquée au plafond | **{1 ; 2 ; aucun}** |
| Dépassement de plafond | l'événement qui ferait dépasser est réduit au maximum autorisé ; les positions déjà ouvertes ne sont pas réduites | gelé |
| Stop journalier par instrument | si PnL intraday de l'instrument (réalisé + latent depuis l'ouverture) ≤ −q · ES95_i, l'instrument est mis à plat jusqu'à la fin de séance ; les sleeves continuent en virtuel | **q ∈ {aucun ; 1,0 ; 1,5}**, ES95_i à la taille B1 retenue, sur l'IS |
| Stop journalier du book | si PnL intraday du book ≤ −Y, tout est mis à plat jusqu'à la fin de séance | **Y ∈ {aucun ; 0,5 · MLL}** |
| Verrou de gain journalier | Combine uniquement : si PnL intraday du jour ≥ 0,55 × objectif courant − 50 USD, tout est mis à plat jusqu'à la fin de séance. Les 50 USD couvrent le slippage de la mise à plat | **{aucun ; actif}** |

Pas de take-profit par instrument : les breakouts ont une asymétrie positive. Le seul plafonnement de gain admis est le verrou de consistance du book.

Hors simulation, côté opérationnel : un stop catastrophe posé chez le broker sur chaque instrument, loin du marché, pour couvrir une déconnexion.

## 5. Évaluation et règle de sélection (IS uniquement)

### Périodes

| Période | Dates | Rôle |
|---|---|---|
| IS par sleeve | MNQ, MGC : 2019-05-06 → 2023-12-31 ; MBT : 2021-05 → 2023-12-31 | estimation des ES_s et de σ_ref |
| IS commune | 2021-06-22 → 2023-12-29 (les 8 sleeves actives) | simulation du cycle Topstep, sélection |
| OOS | 2024-01-02 → 2026-03-19 | verdict lu une fois, **contaminé** (section 2) |
| EXT | 2026-03-20 → 2026-10-02 | contaminée (a servi à retirer S04) ; descriptif seulement |
| Forward | après le 2026-11-03 | seul verdict propre |

### Évaluateur du cycle Topstep (nouveau module)
- **Règles** : section 10. Combine et XFA ont des limites distinctes ; le Scaling Plan ne s'applique qu'au XFA.
- **MLL sur l'équité intraday minute par minute** issue de la simulation nettée, pas sur le PnL de clôture.
- **Bootstrap par blocs mobiles** de séances (blocs de 3, 5, 10 et 20 ; 8 000 chemins ; seed `20261112`). Chaque séance tirée garde son chemin intraday. Tous les candidats sont simulés sur les mêmes chemins.
- **Métrique primaire** : cash net espéré (payouts reçus − frais payés) après 126 séances tradées, en relançant un Combine après chaque échec.
- **Métriques secondaires** : délai médian jusqu'au premier payout ; P(premier payout ≤ 60 séances) ; P(MLL) en Combine et en XFA ; pic de lots ; Sharpe et drawdown intraday du book.

### Sélection séquentielle
Ordre : B1 → B2 → plafond MBT → stop instrument → stop book → verrou de consistance.

À chaque étape, le candidat le plus simple (« aucun », ou k = 0,40 pour B1) est conservé, **sauf** si un autre candidat :
1. améliore la métrique primaire d'au moins 10 % ;
2. la bat sur au moins 55 % des chemins appariés ;
3. ne dégrade pas P(MLL en XFA) de plus de 2 points.

En cas d'égalité, on garde le plus simple. Une quinzaine de configurations sont évaluées au total.

### Validation de la règle
CSCV/PBO sur les configurations évaluées, avec 10 blocs de la fenêtre IS commune. Si PBO > 0,5, on retient la configuration A + B1 (k = 0,40) sans overlay.

## 6. Verdicts

| Verdict | Règle |
|---|---|
| ADOPT | la configuration retenue bat B0 sur l'OOS (métrique primaire, mêmes chemins, lecture unique), sans hausse de P(MLL en XFA) de plus de 2 points. Promotion en lock versionné, avec suivi forward |
| ADOPT_AGGREGATION_ONLY | la couche A passe T1–T5, mais aucune option B ne survit à la sélection ou au verdict. Seul le netting est adopté, avec le sizing B0 |
| REJECT | échec des self-tests, ou couche B dégradée sur l'OOS ; on reste sur B0 |

Dans tous les cas, le verdict forward est lu sur les séances postérieures au 2026-11-03. En cas d'échec, on écrit `HYPOTHESIS_CLOSURE.md` et une fiche au registre des échecs.

## 7. Sorties

- PnL quotidien et intraday du book et de chaque instrument, par configuration.
- Trades nettés et contrats échangés.
- Contributions au risque (ES et variance) par sleeve et par instrument.
- Coût du passage de S01 de son sizing natif à 1 unité, sur l'IS.
- Diagnostic B2 par tercile de volatilité.
- Tables du cycle Topstep pour chaque configuration et chaque étape de sélection.
- PBO.
- Sensibilité, sans sélection : le même pipeline sur un compte 100K (MLL et objectif mis à l'échelle, plafond de lots du 100K).

## 8. Self-tests (avant toute donnée réelle)

- **T1 netting** : en B0, sans plafond, le PnL brut quotidien netté égale la somme des PnL bruts des sleeves à 0,01 USD près ; les coûts nettés sont ≤ la somme des coûts.
- **T2 S01** : à 1 unité, chaque trade natif de S01 est reproduit (mêmes prix et mêmes horaires) quand S01 est seule.
- **T3 plafonds** : le plafond de lots et les plafonds par instrument ne sont jamais dépassés, y compris sur un scénario synthétique où les huit sleeves entrent ensemble.
- **T4 overlays** : après déclenchement d'un stop instrument ou book, la position reste nulle jusqu'à la fin de séance, et les états virtuels continuent d'évoluer.
- **T5 pas de lookahead** : `σ_i,t`, les ES et les plafonds n'utilisent que des données antérieures à la décision ; perturber les minutes postérieures ne change pas les décisions.

## 9. Implémentation

- `src/portfolio/aggregator.py` : états virtuels → contrats → cible nette → ordres en delta. Le même code sert au backtest et à l'exécution live.
- `src/analytics/topstep_lifecycle/` : évaluateur du cycle complet, aux règles de 2026-11.
- `run_study.py` dans ce dossier, avec les sous-commandes `selftest`, `estimate`, `is`, `oos` et `report`.

Entrées gelées, contrôlées par sha256 :
- trades de S02–S09 : `research/signal_sheet_0N_cluster/trades.csv` ;
- trades de S01 : `export/mnq_orb_vwap_atr_retained/trades.csv` ;
- barres 1 min de MNQ, MGC et MBT pour les prix intraday.

## 10. Règles Topstep 50K gelées

Paramètres fournis par l'utilisateur le 2026-11-12. Sources : articles help.topstep.com sur le Scaling Plan, la tarification, la consistance et les restrictions MBT.

| Paramètre | Combine 50K | XFA 50K |
|---|---|---|
| Solde de départ | 50 000 USD | 0 USD |
| Taille maximale | 5 lots (50 micros) | Scaling Plan : < 1 500 USD → 2 lots ; 1 500–2 000 → 3 ; > 2 000 → 5. Le palier change à la séance suivante |
| MBT | 1 MBT = 1 lot (pas le ratio micro 10:1) | idem |
| Objectif | 3 000 USD, porté à meilleur jour / 0,55 si le meilleur jour dépasse 1 650 USD ; au moins 2 jours tradés | aucun |
| Consistance | meilleur jour / profit total ≤ 55 %, strict ; le meilleur jour historique ne baisse jamais | aucune (parcours standard) |
| MLL | 2 000 USD, niveau suivi sur le plus haut solde de fin de séance, plafonné au solde de départ ; violation testée sur l'équité intraday | même mécanique, niveau de départ −2 000 USD, plafonné à 0 ; ramené à 0 après un payout |
| Payout | — | parcours standard (règles relevées le 2026-11-11, non reconfirmées le 12) : 5 jours gagnants ≥ 150 USD depuis le dernier payout, puis payout demandé dès l'éligibilité = min(50 % du solde ; 2 000 USD), le trader touche 90 % |

**Frais.** La sélection se fait sur le parcours Standard ; le parcours No Activation Fee est publié à titre d'information.

| Frais | Standard | No Activation Fee |
|---|---:|---:|
| Abonnement Combine, toutes les 21 séances tant que l'on est en Combine | 49 USD | 95 USD |
| Reset après MLL en Combine | 49 USD | 95 USD |
| Activation XFA | 149 USD | 0 USD |

Après une perte du XFA par MLL, on rachète un Combine. Les taxes ne sont pas modélisées.

**Conventions et réserves :**
- Aucune Daily Loss Limit Topstep n'est modélisée, car elle ne figure pas dans les paramètres fournis.
- L'équivalence 1 MBT = 1 lot est une lecture prudente de la documentation, qui doit être confirmée dans TopstepX Risk Settings. Une sensibilité à 0,1 lot sera publiée, sans servir à la sélection.
- Au premier palier XFA (2 lots), trois MBT plus des micros ne tiennent pas. La règle de troncature de B3 s'applique alors d'office.
