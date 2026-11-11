# Topstep 50K — portefeuille gelé 6 vs 9 sleeves

Date d'exécution : 4 octobre 2026.

## Verdict

`NEUTRAL`

L'ajout des trois ensembles MBT augmente le PASS dans toutes les familles et tous les stress testés. Il améliore aussi la MLL à 60 jours pour l'equal-contract et le cluster-cap. Mais l'equal-risk entier augmente légèrement la MLL en nominal et sous stress de coûts, tandis que les drawdowns historiques absolus s'élargissent. Le critère préfixé exigeait que **toutes** les familles nominales à 60 jours gagnent plus de 2 points de PASS et réduisent la MLL; cette condition n'est pas satisfaite.

## Règle exécutable des trois sleeves MBT

Chaque cellule de la région gelée calcule ses entrées, stops et sorties indépendamment. À chaque clôture minute, les positions `-1/0/+1` des cellules sont moyennées. Le contrat agrégé est ajusté à l'open de la minute suivante :

- moyenne ≥ +0,5 : `+1 MBT`;
- moyenne ≤ -0,5 : `-1 MBT`;
- sinon : `0`;
- une égalité exacte à ±0,5 est arrondie en s'éloignant de zéro;
- hard flatten à 16:08 ET.

Cette causalité d'une minute évite d'utiliser un stop intrabar avant qu'il soit observable. Les coûts ne sont payés que lorsque la position finale change. Les trois ensembles utilisent respectivement 8, 9 et 12 cellules. Aucun paramètre source n'a été modifié.

## Règles Topstep appliquées

Simulation Trading Combine 50K : target 3 000 USD, MLL 2 000 USD, consistance 55 %, horizons diagnostiques 35/60 jours tradés. La MLL est contrôlée minute par minute avec un marquage OHLC adverse conservateur, puis remontée en fin de journée et bloquée au capital initial.

Topstep indique actuellement que la MLL inclut le PnL latent et est surveillée en temps réel, que la consistance du Combine est de 55 %, et que la limite 50K est de 5 minis ou 50 micros :

- https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit
- https://help.topstep.com/en/articles/8284197-trading-combine-parameters
- https://help.topstep.com/en/articles/8284208-consistency-at-topstep

Les coûts gelés des six sleeves sont conservés. MBT utilise 2,82 USD round-turn et un tick de slippage par côté, conformément au barème TopstepX du 28 juillet 2026 : https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees.

## Mappings entiers

Ordre des six sleeves : MNQ ORB, MNQ volatility, MNQ compression, MNQ ATR trend, MGC volatility, MGC gap. Ordre MBT : rolling, trend pullback, volatility.

| Famille | Mapping 6 | Ajout MBT | Exposition micro maximale observée |
|---|---|---|---:|
| Equal-contract | `[1,1,1,1,1,1]` | `[1,1,1]` | 7 |
| Equal-risk entier | `[2,1,5,1,2,2]` | `[4,3,3]` | 19 |
| MBT cluster-cap | `[1,1,1,1,1,1]` | rolling/trend fusionnés dans 1 slot + volatility 1 | 6 |

Le mapping equal-risk est obtenu mécaniquement par `round(max(vol)/vol_i)`, minimum 1, plafond 10, sans lecture du PASS. Le cluster-cap discrétise la moyenne des deux positions déjà causales rolling/trend et limite donc ce couple à un seul contrat.

Aucun mapping n'approche la limite de 50 micros. Maxima par actif : equal-contract 4 MNQ / 2 MGC / 3 MBT; equal-risk 9 / 4 / 10; cluster-cap 4 / 2 / 2.

## Bootstrap mobile — nominal

8 000 chemins par ligne, blocs de 3/5/10/20 séances, seed fixe `20261004`.

### Horizon 35 jours

| Famille | Livre | PASS | MLL | EXPIRY | Médiane jours vers PASS |
|---|---|---:|---:|---:|---:|
| Equal-contract | 6 | 25,69 % | 38,86 % | 35,45 % | 21 |
| Equal-contract | 9 | **35,03 %** | 43,05 % | 21,93 % | 19 |
| Equal-risk entier | 6 | 28,68 % | 59,71 % | 11,61 % | 16 |
| Equal-risk entier | 9 | **33,94 %** | 62,93 % | 3,14 % | 11 |
| Cluster-cap | 6 | 25,69 % | 38,86 % | 35,45 % | 21 |
| Cluster-cap | 9 | **32,71 %** | 41,49 % | 25,80 % | 19 |

À 35 jours, l'ajout MBT transforme beaucoup d'expirations en décisions plus rapides : davantage de PASS, mais également davantage de MLL. C'est un effet de risques concurrents, pas une amélioration pure du drawdown.

### Horizon 60 jours

| Famille | Livre | PASS | MLL | EXPIRY | Médiane jours vers PASS |
|---|---|---:|---:|---:|---:|
| Equal-contract | 6 | 38,24 % | 49,23 % | 12,54 % | 27 |
| Equal-contract | 9 | **45,24 %** | **48,11 %** | 6,65 % | 23 |
| Equal-risk entier | 6 | 33,25 % | 62,18 % | 4,58 % | 18 |
| Equal-risk entier | 9 | **35,85 %** | 62,91 % | 1,24 % | 12 |
| Cluster-cap | 6 | 38,24 % | 49,23 % | 12,54 % | 27 |
| Cluster-cap | 9 | **44,01 %** | **48,04 %** | 7,95 % | 24 |

L'equal-contract donne le meilleur gain équilibré : PASS +7,00 points, MLL -1,11 point. Le cluster-cap est proche : PASS +5,78 points, MLL -1,19 point. L'equal-risk est trop agressif : PASS +2,60 points, MLL +0,74 point.

Les rolling starts historiques vont dans la même direction à 60 jours : equal-contract 48,70 % → 53,09 % de PASS; cluster-cap 48,70 % → 52,10 %; equal-risk 40,52 % → 41,72 %.

## Stress à 60 jours — delta 9 moins 6

| Famille | Stress | Δ PASS | Δ MLL |
|---|---|---:|---:|
| Equal-contract | 2 ticks/côté | +5,95 pts | -0,89 pt |
| Equal-contract | pertes ×1,25 | +5,81 pts | -3,11 pts |
| Equal-contract | coûts + pertes ×1,25 | +5,06 pts | -2,74 pts |
| Equal-risk entier | 2 ticks/côté | +1,65 pt | **+1,88 pt** |
| Equal-risk entier | pertes ×1,25 | +3,86 pts | -2,05 pts |
| Equal-risk entier | coûts + pertes ×1,25 | +3,16 pts | -1,46 pt |
| Cluster-cap | 2 ticks/côté | +4,99 pts | -1,13 pt |
| Cluster-cap | pertes ×1,25 | +3,78 pts | -1,65 pt |
| Cluster-cap | coûts + pertes ×1,25 | +3,29 pts | -1,38 pt |

L'ajout MBT reste favorable au PASS dans chaque stress. L'exception de robustesse est l'equal-risk sous coûts seuls, où la MLL se dégrade de 1,88 point.

## Drawdown et clustering observés

| Famille nominale | Livre | Sharpe quotidien | Pire jour | Pires 5 jours | Plus longue série perdante | MaxDD intraday historique |
|---|---|---:|---:|---:|---:|---:|
| Equal-contract | 6 | 1,61 | -1 234 USD | -5 283 USD | 8 | -6 314 USD |
| Equal-contract | 9 | **1,83** | -1 378 USD | -6 147 USD | **7** | -7 223 USD |
| Equal-risk entier | 6 | 1,63 | -2 835 USD | -8 253 USD | 9 | -7 379 USD |
| Equal-risk entier | 9 | **1,93** | -2 943 USD | -12 705 USD | **7** | -11 935 USD |
| Cluster-cap | 6 | 1,61 | -1 234 USD | -5 283 USD | 8 | -6 314 USD |
| Cluster-cap | 9 | **1,79** | -1 378 USD | -5 908 USD | 9 | -7 240 USD |

Ces MaxDD sont des trajectoires historiques sans liquidation Topstep; ils montrent le risque absolu si l'on laisse courir le livre. Le pire PnL intraday depuis l'ouverture d'une séance s'améliore néanmoins en equal-contract : -1 738 USD pour les six contre -1 538 USD pour les neuf. La conclusion drawdown est donc mixte : meilleure diversification quotidienne et meilleur PASS, mais amplitude cumulée plus grande lorsque plusieurs pertes se regroupent.

Le nombre maximal de sleeves perdantes le même jour passe de 4 à 6 en equal-contract; leur nombre moyen pendant une journée négative passe de 1,75 à 2,87. Le cluster-cap ramène ces valeurs à 5 et 2,62. En revanche, la pire contribution négative brute simultanée ne se dégrade que de -1 384 à -1 418 USD en equal-contract. Les pertes sont donc plus souvent distribuées entre plusieurs sleeves, sans explosion comparable du pire choc journalier brut.

## Lecture opérationnelle

- Le mapping **equal-contract** est le meilleur candidat shadow : amélioration forte du PASS, baisse légère de la MLL à 60 jours, exposition maximale très basse.
- Le **cluster-cap** sacrifie environ 1,2 point de PASS par rapport à l'equal-contract neuf, tout en réduisant l'exposition MBT maximale de 3 à 2 contrats. C'est le choix de gouvernance le plus prudent.
- L'**equal-risk entier** ne doit pas être promu : ses multiplicateurs `[4,3,3]` sur MBT accélèrent les résultats, mais rendent la MLL et les drawdowns trop sensibles.
- Les données MBT ayant servi au freeze, cette étude reste rétrospective. Elle ne remplace pas une observation prospective postérieure au 2 octobre 2026.

