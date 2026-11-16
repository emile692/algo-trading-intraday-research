# S01 — bandes d'ATR rapportées au prix — clôture

- **Statut** : `CLOSED_IS`. Aucun des deux bras ne passe tous les gates IS pré-enregistrés ; aucun holdout n'a été lu.
- **Date** : 2026-11-15.
- **Spécification** : [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `e7246355…a0f586d` ([SPEC_LOCK.json](SPEC_LOCK.json)), figée à 21:38:50 avant tout calcul sur les variantes.
- **Commande** : `python -m research.s01_atr_price_normalized_20261115.run_campaign run`. Les contrôles techniques se relancent avec `… selftest`.
- **Origine** : décision 1 de l'audit [audit_20261115](../../audit_20261115/RAPPORT.md), option (b), lancée par l'utilisateur le 2026-11-15.
- **Holdouts non lus**, encore vierges pour ces règles :
  - PRE NQ 2010-06 → 2019-05 ;
  - VAL 2024-02-28 → 2026-03-19 ;
  - EXT 2026-03-20 → 2026-10-02.

## Hypothèse

S01 s'assèche parce que deux de ses règles sont exprimées en unités MNQ absolues, alors que le prix a doublé :
1. les bornes du filtre d'ATR, en points ;
2. le saut d'un trade quand un seul contrat risque plus de 250 USD.

L'hypothèse : exprimer l'ATR des bougies d'une minute en fraction du prix, avec les mêmes quantiles IS, rend au filtre sa fréquence sans changer son sens, et conserve l'edge.

| Bras | Règle |
|---|---|
| REF | S01 ATR 18 de la fiche 01, bornes en points (`thresholds_is.csv`). Comparateur, jamais sélectionné. |
| A | Vote à 18 modèles sur `atr_p / close`, quantiles calibrés sur l'IS 2019-05-05 → 2024-02-27. Dimensionnement inchangé. |
| B | A, plus un plancher : 1 contrat quand le risque d'un contrat est compris entre 250 et 500 USD (le trade est sauté au-delà). |

Rien d'autre ne change : entrée, sortie, vote 9/18, coûts nominaux, fenêtre de calibration.

## Contrôles techniques (avant le gel, sur la référence seule)

| Contrôle | Résultat |
|---|---|
| T1 — REF rejoué sur le fichier MNQ long | 701 trades sur 701 identiques jusqu'au 2026-03-19, écart de PnL nul |
| T2 — quantiles en points recalibrés | identiques à `thresholds_is.csv` (écart max 4·10⁻¹⁵) |
| T3 — construction du bras B | les trades de REF sont inclus ; chaque trade ajouté a une quantité de 1 et un risque compris entre 250 et 500 USD |

## Résultats

### Étape 0 — fréquence, sans aucun PnL

G0 compare la médiane des trades exécutés par trimestre sur 2025 T1 → 2026 T3 à la même médiane sur les trimestres IS 2019 T3 → 2023 T4. Le seuil est de 0,60.

| Bras | Médiane IS | Médiane 2025 T1 → 2026 T3 | Ratio | G0 | 2026 T1 / T2 / T3 |
|---|---:|---:|---:|---|---|
| REF | 32,5 | 10 | 0,31 | échec | 2 / 9 / 2 |
| A | 29 | 15 | **0,52** | **échec** | 9 / 15 / 8 |
| B | 31,5 | 30 | 0,95 | réussi | 31 / 37 / 30 |

### Étape 1 — IS 2019-05-05 → 2024-02-27

Les chiffres portent sur 1 252 séances de semaine. Le PnL est quotidien, net, en USD, avec les coûts nominaux. Le Sharpe est annualisé par √252, jours sans trade inclus ; le t est le t HAC de Lo.

| Bras | Trades | Net USD | Sharpe | t HAC | maxDD | PF | PSR(0) | DSR (2 essais) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| REF | 514 | 13 852 | 1,251 | 3,09 | −2 149 | 1,32 | | |
| A | 508 | 10 598 | 0,976 | 2,46 | −2 291 | 1,25 | 0,987 | 0,983 |
| B | 552 | 9 664 | 0,844 | 2,06 | −2 272 | 1,20 | 0,973 | 0,965 |

| Gate | Seuil | A | B |
|---|---|---|---|
| G0 fréquence | ratio ≥ 0,60 | **0,52, échec** | 0,95, réussi |
| G1 trades IS | ≥ 0,75 × 514 = 386 | 508, réussi | 552, réussi |
| G2 net > 0 et t ≥ 2 | | t = 2,46, réussi | t = 2,06, réussi |
| G3 Sharpe | ≥ 0,75 × 1,251 = 0,938 | 0,976, réussi | **0,844, échec** |
| G4 années positives (2019–2023) | ≥ 3 sur 5 | 5 sur 5, réussi | 5 sur 5, réussi |

Le PnL net par année est le suivant (2024 ne compte que janvier et février, hors G4) :

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 (jan.–fév.) |
|---|---:|---:|---:|---:|---:|---:|
| REF | 1 376 | 1 313 | 5 948 | 2 043 | 2 702 | 470 |
| A | 696 | 2 269 | 4 494 | 1 409 | 1 913 | −184 |
| B | 696 | 1 714 | 4 606 | 1 186 | 1 703 | −242 |

Règle de sélection : A s'il passe G0 à G4, sinon B s'il les passe, sinon `CLOSED_IS`. Aucun bras ne passe, **la campagne est donc fermée** et aucun holdout n'est lu.

## Lecture

- **Le filtre en ATR / prix ne rend qu'une partie de la fréquence.** A trade 9, 15 et 8 fois sur les trois premiers trimestres de 2026, contre 2, 9 et 2 pour REF, mais le ratio n'atteint que 0,52.
- **Le reste vient du dimensionnement.** Il suffit de l'assouplir (bras B) pour revenir à la fréquence de l'IS (0,95). Le saut des trades à plus de 250 USD de risque est donc la deuxième cause de l'assèchement, et en 2026 c'est la plus forte.
- **Les trades que le plancher ajoute ont perdu de l'argent sur l'IS.** B − A = −935 USD sur 44 trades de plus, soit environ −21 USD par trade.
  - Sur 2019–2024, un risque supérieur à 250 USD pour un contrat signifie un range d'ouverture réellement large : de 0,7 à 1,6 % du prix. Le saut fonctionnait donc comme un filtre implicite des ouvertures très larges, et pas seulement comme un artefact.
  - En 2026, au prix d'environ 29 500, ce même seuil coupe dès 0,4 % du prix, c'est-à-dire des journées plus ordinaires. L'IS ne permet pas de dire si ces journées-là sont mauvaises.
- **Sur l'IS, A reste significatif** (t 2,46, DSR 0,98), mais il fait moins bien que REF : Sharpe 0,98 contre 1,25.
  - Une partie de l'écart est attendue : les bornes en points de REF viennent d'un cluster choisi sur ce même IS, celles de A non.
  - L'IS ne prouve pas pour autant que A est meilleur.
- **Rien n'est sauvé après coup.** Abaisser G0 à 0,50 ferait passer A, mais la règle de non-sauvetage de la spécification l'interdit.

## Conséquences

- **S01 ATR 18 reste la sleeve de référence, inchangée.** Il faut revenir à la décision 1 de l'audit : (a) ne rien changer et laisser le forward juger, ou (c) mettre S01 en revue.
- **Le holdout NQ 2010-06 → 2019-05 reste vierge pour l'ORB**, ce qui préserve ce juge pour une éventuelle réouverture.
  - Il permet de tester un filtre rapporté au prix.
  - Il ne permet presque pas de tester le dimensionnement : aux prix de 2010–2019, un contrat dépasse rarement 250 USD de risque.

## Réouverture

- **Elle exige un nouvel identifiant de campagne et une spécification qui déclare vus l'IS de A et de B**, ainsi que ce document.
- **Piste non testée** : un seuil de saut exprimé en pourcentage du prix plutôt qu'en USD.
  - Son IS sera fortement corrélé à celui de A et de B, déjà vu.
  - Le seul juge propre serait le forward, puisque PRE ne mesure presque pas le dimensionnement.
- **Ce qu'il ne faut pas faire** : relancer A avec un G0 abaissé, ou B avec un plafond autre que 500 USD.

## Fichiers

| Fichier | Contenu |
|---|---|
| [results/selftest.json](results/selftest.json) | T1, T2, T3 |
| [results/stage0.json](results/stage0.json), [results/stage0_trades_per_quarter.csv](results/stage0_trades_per_quarter.csv) | G0 et nombre de trades par trimestre de REF, A et B (aucun PnL) |
| [results/stage1.json](results/stage1.json), [results/stage1_is_daily.csv](results/stage1_is_daily.csv) | métriques IS, gates, PSR et DSR, PnL quotidien IS |
| [results/thresholds_is_atr_over_price.csv](results/thresholds_is_atr_over_price.csv) | les 18 bornes ATR / prix calibrées sur l'IS |
| [results/stage2.json](results/stage2.json) | `CLOSED_IS`, `read: false` |
| [results/run_log.txt](results/run_log.txt) | sortie console |
