# Clôture — USD Inflation, second signal « Inflation Trend »

**Statut : `CLOSED_IS`** (2026-11-17). OOS de B_ens et C non lu (`results/oos_sealed.json`), mais **OOS 2022-01 → 2026-09 de B seul lu sur toute la grille fast × slow** à la demande de l'utilisateur après la clôture (voir « Lecture post-clôture »).

- Spec gelée : `research_design.md` (sha256 `b4bb5c5f…ae4bd`), script `run.py` (sha256 `9be7bb20…6ff622`).
- Entrée : `C:\Data_local\excel_repo\FX_USD_Inflation.xlsx` (sha256 `b1039f76…c9f8f8`), feuilles `QuantCube Data` (US) et `Market Data` (DX1).
- Reproduction : `uv run --no-project --with pandas --with openpyxl python -I run.py`.

## Hypothèse

Une slide Goldman Sachs (oct. 2026) décrit la stratégie USD Inflation avec deux signaux : l'accélération du YoY (signal A, déjà en place dans le classeur) et une tendance par deux moyennes mobiles (signal B). H1 : B = sign(MA_court(YoY) − MA_long(YoY)) apporte une information que A n'a pas, et C = 0,5·A + 0,5·B améliore le Sharpe net de A.

## Contrôle technique

Le moteur Python reproduit l'Excel exactement : écart maximal 0 sur le signal et sur le P&L journalier (2 726 jours). Sharpe IS de A 0,782, global 1,034, identiques au classeur.

## Résultats IS (2016-09-01 → 2021-12-31, net de 2 bp one-way)

| | A (actuel) | B_ens (36 sous-signaux) | C = 0,5 A + 0,5 B |
|---|---|---|---|
| Rendement annualisé | 4,04 % | 4,12 % | 4,08 % |
| Volatilité | 5,35 % | 5,12 % | 4,98 % |
| Sharpe | 0,755 | 0,803 | 0,819 |
| Max drawdown | −8,6 % | −10,7 % | −8,6 % |
| Turnover annuel | 894 % | 701 % | 766 % |

- Corrélation des signaux A/B 0,78 (même signe 83 % du temps), des P&L 0,81.
- Alpha de B sur A : +0,98 %/an, t Newey-West 0,73.
- Bootstrap par blocs de 20 jours : P(ΔSharpe C − A > 0) = 0,66, intervalle 5–95 % [−0,15 ; +0,27].
- Par année, Sharpe A / B / C : 2016 1,80 / 3,88 / 3,09 ; 2017 0,68 / 0,76 / 0,75 ; 2018 1,61 / 0,95 / 1,39 ; 2019 −0,75 / −1,49 / −1,23 ; 2020 0,54 / 0,83 / 0,71 ; 2021 1,11 / 1,23 / 1,18.
- Sur 2017-03 → 2021-12 (sans les quatre derniers mois de 2016) : A 0,753, B_ens 0,638, C 0,734. **Le gain de C vient de sept.–déc. 2016.**

## Portes

| Porte | Valeur | Résultat |
|---|---|---|
| G1 Sharpe IS B_ens > 0 | 0,80 | passe |
| G2 ΔSharpe IS (C − A) > 0 | +0,065 | passe |
| G3 ≥ 60 % des paires (s, l) avec ΔSharpe > 0 | 34 % (14/41) ; 3/12 dans la zone centrale s 10–30, l 60–180 | **échoue** |
| G4 PBO < 0,5 | 0,85 | **échoue** |

## Lecture

B n'est pas un signal différent de A : c'est le même momentum du YoY, lissé. Seul, il gagne de l'argent sur presque toute la grille (Sharpe IS médian 0,43 ; 41 paires sur 41 positives). Ajouté à A, il n'apporte rien de stable. Là où il est peu corrélé à A (l = 250 ou 365 jours), l'effet change de signe d'une colonne à l'autre (l = 250 : 7/7 négatives ; l = 365 : 6/7 positives), ce qui ressemble à du bruit. Le PBO de 0,85 montre que choisir la meilleure paire sur l'IS mènerait presque toujours à une paire moyenne ou mauvaise hors échantillon.

La baisse de turnover (−128 points, soit environ 0,03 %/an de coût) est négligeable.

Les chiffres de la slide GS (Sharpe 0,82 sur 2007–2026, vol 9,3 %, DD −20,7 %) portent sur une autre période, une autre réplication du DXY et un historique pré-2016 reconstruit sur les taux monétaires. Ils ne se comparent pas au Sharpe global de 1,03 du classeur et ne disent rien de l'apport propre de B.

Réserve commune à A et B, non testée ici : l'export QuantCube est un millésime unique. Si le nowcast est révisé, le backtest a pu voir des valeurs qui n'étaient pas encore publiées.

## Lecture post-clôture (demande de l'utilisateur, 2026-11-17)

`heatmap.py` : Sharpe net de B seul, une paire (MA rapide, MA lente) par cellule, 99 paires (rapide 1–120 j, lente 20–365 j), lag d'un seul jour. IS commun 2017-03 → 2021-12, OOS 2022-01 → 2026-09. Figures et CSV dans `results/heatmap_*`.

| Lag | A au même lag (IS / OOS) | B médiane IS / OOS | Paires > 0 IS / OOS | Paires battant A IS / OOS | Corr. de rang IS↔OOS | Meilleure IS → OOS |
|---|---|---|---|---|---|---|
| 1 (nowcast t−2 pour le rendement de t) | 0,73 / 1,26 | 0,54 / 0,84 | 95 / 98 sur 99 | 6 / 9 | 0,69 | (10, 45) 0,91 → 1,17 |
| 0 (nowcast t−1, suppose publication avant la clôture DX) | 0,68 / 1,25 | 0,54 / 0,84 | 96 / 99 sur 99 | 14 / 9 | 0,64 | (45, 60) 0,96 → 1,23 |

B est un signal robuste seul : positif presque partout, et un paysage stable entre IS et OOS. Il reste en dessous de A sur les deux périodes, et le lag ne change presque rien. Le verdict ne change pas. Comme l'OOS de la famille est désormais lu, aucune paire ni aucun poids ne peut être choisi à partir de ces cartes.

## Réouverture

Seulement avec une information nouvelle, sous un nouvel identifiant et une spec qui déclare l'IS de B vu :
- un historique point-in-time du nowcast (millésimes) ;
- ou un historique plus long (avant 2016) sur lequel B serait jugé sans retuning de la grille.

Pas de choix d'une paire (s, l) ni d'un poids de mélange après coup : le poids 0,25 sur A (Sharpe IS 0,822) n'est pas une piste.
