# USD Inflation, second signal « Inflation Trend » (croisement de moyennes mobiles)

Date : 2026-11-17. Protocole figé avant le premier calcul.

## Contexte

La stratégie existante (`C:\Data_local\excel_repo\FX_USD_Inflation.xlsx`) prend une position continue dans [−1, +1] sur DX1 :
signal A « Inflation Acceleration » = moyenne de sign(YoY(t−lag) − YoY(t−lag−W)), W = 15…60 jours calendaires,
lag = 1…3 jours ouvrés (138 sous-signaux), sur le nowcast CPI US QuantCube (YoY, quotidien).
P&L(t) = pos(t−1) × r_DX1(t) − |pos(t−1) − pos(t−2)| × 2 bp. IS = 2016-02 → 2021-12, OOS = 2022-01 → 2026-09.

Une slide Goldman Sachs (FICC & Equities, oct. 2026) décrit la même stratégie avec un second signal :
« Inflation Trend : underlying trend in CPI, based on two moving averages (short and long-term) ».

## Hypothèse

H1 : un signal de tendance B = sign(MA_court(YoY) − MA_long(YoY)) apporte une information que A n'a pas,
de sorte que la combinaison C = 0,5·A + 0,5·B améliore le Sharpe net de A.

Réserve a priori : A et B sont deux mesures du même momentum du YoY ; une forte corrélation est attendue.

## Spécification figée

- Données : feuilles `QuantCube Data` (iso2 = US) et `Market Data` (DX1 Curncy) du classeur, sans modification.
- Moteur identique à l'Excel (même lag d'exécution, même coût 2 bp one-way, même convention de moyenne sur les sous-signaux valides).
- B_ens (principal) : moyenne de sign(MA_s − MA_l) sur s ∈ {10, 20, 30} j, l ∈ {60, 90, 120, 180} j (calendaires, moyennes strictes),
  lag ∈ {1, 2, 3} jours ouvrés → 36 sous-signaux. Pas de sélection de paire.
- C = 0,5·A + 0,5·B_ens (poids figé, pas optimisé).
- Fenêtre de comparaison IS : 2016-09-01 → 2021-12-31 (toutes les moyennes de B_ens valides). A est aussi rapporté sur l'IS Excel complet.
- Grille plateau / CSCV : s ∈ {1, 5, 10, 20, 30, 45, 60} × l ∈ {60, 90, 120, 180, 250, 365}, s < l, fenêtre commune 2017-03-01 → 2021-12-31.

## Portes IS (toutes requises)

| Porte | Critère |
|---|---|
| G1 | Sharpe net IS de B_ens seul > 0 |
| G2 | ΔSharpe net IS (C − A) > 0 |
| G3 | ≥ 60 % des paires (s, l) de la grille ont ΔSharpe IS (0,5·A + 0,5·B_paire − A) > 0 |
| G4 | PBO (CSCV, 10 blocs, candidats = A seul + 0,5·A + 0,5·B_paire) < 0,5 |

Informatif seulement : corrélation A/B, alpha de B sur A (t Newey-West), bootstrap par blocs de ΔSharpe, Δ par année, poids de mélange.

## Verdict OOS (lu une fois, seulement si G1–G4 passent)

Complément validé si ΔSharpe OOS (C − A) ≥ 0 et Sharpe OOS de B_ens > 0. Sinon CLOSED_OOS.
Si une porte IS échoue : CLOSED_IS ; l'OOS est calculé mais reste non lu pour la décision.
