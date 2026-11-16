# Phase 8 — Concentration et régimes

*Book à 8 sleeves (S01 = ATR 18), calendrier union, 2021-05-03 → 2026-03-19, sauf mention. Les sleeves sont mesurées sur tout leur historique, IS et validation réutilisée confondus : c'est une analyse **descriptive**. Barres quotidiennes RTH 09:30–16:00 ET, MNQ/MGC coupés au 2026-03-19, MBT au 2026-10-02. Les bornes des terciles sont fixées sur l'IS de chaque instrument. Script : [phase8_regimes.py](phase8_regimes.py). Tables : [08_regimes.csv](outputs/08_regimes.csv), [08_clustering.csv](outputs/08_clustering.csv), [08_worst_weeks.csv](outputs/08_worst_weeks.csv), [08_worst_months.csv](outputs/08_worst_months.csv), [08_net_position_levels.csv](outputs/08_net_position_levels.csv).*

## Contribution au risque — book à 8 sleeves, calendrier corrigé

**2024-01-02 → 2026-03-19**

| Sleeve | PnL | Vol. quotidienne | Pire jour | maxDD | Contribution au risque |
|---|---:|---:|---:|---:|---:|
| S01 | 6 035,00 | 130,74 | −249,00 | −875,50 | 7,3 % |
| S02 | 9 279,00 | 288,70 | −1 138,00 | −3 714,00 | 41,0 % |
| S03 | 592,50 | 58,53 | −268,50 | −940,50 | 3,7 % |
| S05 | 1 717,50 | 122,88 | −874,50 | −1 622,50 | 10,7 % |
| S06 | 3 932,50 | 149,08 | −909,50 | −1 564,50 | 13,0 % |
| S07 | 2 209,08 | 73,36 | −355,32 | −1 285,80 | 5,2 % |
| S08 | 4 135,08 | 105,16 | −533,64 | −1 542,56 | 9,1 % |
| S09 | 2 957,30 | 99,36 | −229,32 | −1 916,40 | 9,9 % |

**2021-05-03 → 2026-03-19**

| Sleeve | PnL | Vol. quotidienne | Pire jour | maxDD | Contribution au risque |
|---|---:|---:|---:|---:|---:|
| S01 | 15 155,00 | 136,91 | −249,00 | −2 148,50 | 13,7 % |
| S02 | 22 160,50 | 240,09 | −1 138,00 | −3 714,00 | 44,0 % |
| S03 | 2 119,50 | 50,84 | −268,50 | −1 202,50 | 3,4 % |
| S05 | 3 009,00 | 86,06 | −874,50 | −1 622,50 | 8,1 % |
| S06 | 4 510,00 | 105,00 | −909,50 | −1 564,50 | 9,8 % |
| S07 | 2 242,22 | 57,71 | −451,32 | −1 285,80 | 5,0 % |
| S08 | 4 569,90 | 79,87 | −533,64 | −1 542,56 | 8,2 % |
| S09 | 5 629,48 | 73,64 | −229,32 | −1 916,40 | 7,8 % |

## Position nette par instrument (moteur de netting, sans limite de lots)

Sur 1261 séances, la position nette maximale est de 4 MNQ, 2 MGC et 3 MBT, soit au plus 8 USD par point MNQ, 20 USD par point MGC et 0,3 USD par point MBT. Le pic simultané est de 3,3 lots Topstep (1 MBT = 1 lot). Les trois instruments sont en position le même jour sur 182 séances.

| Instrument | Position nette | Part du temps (09:30–16:08) |
|---|---:|---:|
| MNQ | -2 | 0,1 % |
| MNQ | -1 | 19,8 % |
| MNQ | +0 | 47,3 % |
| MNQ | +1 | 21,0 % |
| MNQ | +2 | 10,4 % |
| MNQ | +3 | 1,3 % |
| MGC | -2 | 0,4 % |
| MGC | -1 | 6,6 % |
| MGC | +0 | 86,1 % |
| MGC | +1 | 6,9 % |
| MGC | +2 | 0,1 % |
| MBT | -3 | 3,4 % |
| MBT | -2 | 8,3 % |
| MBT | -1 | 13,4 % |
| MBT | +0 | 51,1 % |
| MBT | +1 | 12,7 % |
| MBT | +2 | 7,0 % |
| MBT | +3 | 4,2 % |

## Régimes — book à 8 sleeves

| Variable | Instrument | Régime | Séances | PnL | PnL moyen | Sharpe | Jours gagnants |
|---|---|---|---:|---:|---:|---:|---:|
| ATR% J−1 (ex ante) | MNQ | bas | 519 | 33 794 | 65,11 | 2,51 | 49 % |
| ATR% J−1 (ex ante) | MNQ | moyen | 411 | 10 715 | 26,07 | 1,10 | 49 % |
| ATR% J−1 (ex ante) | MNQ | haut | 311 | 15 082 | 48,49 | 1,82 | 55 % |
| Jour MNQ abs(C−O)/(H−L) (ex post) | MNQ | bas | 426 | −52 187 | −122,50 | −6,30 | 26 % |
| Jour MNQ abs(C−O)/(H−L) (ex post) | MNQ | moyen | 397 | −6 410 | −16,15 | −0,72 | 44 % |
| Jour MNQ abs(C−O)/(H−L) (ex post) | MNQ | haut | 418 | 118 188 | 282,75 | 10,75 | 81 % |
| ER20 fin J−1 (ex ante) | MNQ | bas | 483 | 26 375 | 54,61 | 2,36 | 51 % |
| ER20 fin J−1 (ex ante) | MNQ | moyen | 366 | 9 472 | 25,88 | 0,95 | 49 % |
| ER20 fin J−1 (ex ante) | MNQ | haut | 392 | 23 743 | 60,57 | 2,31 | 52 % |
| ER20 centré (ex post) | MNQ | bas | 482 | 35 685 | 74,04 | 2,72 | 52 % |
| ER20 centré (ex post) | MNQ | moyen | 359 | 20 056 | 55,87 | 2,01 | 53 % |
| ER20 centré (ex post) | MNQ | haut | 390 | 4 153 | 10,65 | 0,54 | 47 % |
| ATR% J−1 (ex ante) | MGC | bas | 159 | 4 262 | 26,81 | 1,29 | 52 % |
| ATR% J−1 (ex ante) | MGC | moyen | 510 | 22 031 | 43,20 | 2,14 | 52 % |
| ATR% J−1 (ex ante) | MGC | haut | 592 | 33 103 | 55,92 | 1,88 | 49 % |
| ATR% J−1 (ex ante) | MBT | bas | 470 | 27 740 | 59,02 | 2,45 | 54 % |
| ATR% J−1 (ex ante) | MBT | moyen | 406 | 25 027 | 61,64 | 2,18 | 49 % |
| ATR% J−1 (ex ante) | MBT | haut | 371 | 6 194 | 16,70 | 0,73 | 48 % |

## Régimes — Sharpe par sleeve et par tercile

| Sleeve | ATR% J−1 (ex ante) bas / moyen / haut (Sharpe) | Jour MNQ abs(C−O)/(H−L) (ex post) bas / moyen / haut (Sharpe) | ER20 fin J−1 (ex ante) bas / moyen / haut (Sharpe) | ER20 centré (ex post) bas / moyen / haut (Sharpe) |
|---|---|---|---|---|
| S01 | 1,73 / 1,06 / 0,94 | −6,69 / 0,95 / 7,22 | 1,39 / 0,87 / 1,60 | 1,25 / 1,09 / 1,53 |
| S02 | 1,37 / 0,54 / 0,70 | −8,18 / −1,72 / 8,81 | 1,63 / −0,26 / 0,93 | 1,59 / 0,53 / −0,11 |
| S03 | −0,28 / 0,04 / 1,29 | 0,45 / −0,80 / 1,19 | 0,79 / −0,12 / 0,64 | 0,50 / 0,56 / 0,25 |
| S04 | 0,30 / 0,05 / 0,87 | −7,06 / 0,23 / 4,94 | 0,81 / 0,59 / −0,08 | 0,44 / 0,89 / −0,47 |
| S05 | 1,22 / −0,20 / 0,49 | 1,08 / 0,13 / 0,21 | 0,40 / −0,22 / 0,65 | −0,39 / 1,00 / 0,20 |
| S06 | 0,42 / −0,27 / 1,06 | 1,25 / 0,57 / 0,22 | 0,33 / 0,15 / 1,22 | 1,42 / 0,91 / −1,09 |
| S07 | 1,32 / 0,09 / −0,12 | 0,60 / −0,98 / 2,26 | 0,87 / −0,03 / 0,72 | −0,78 / 0,08 / 1,92 |
| S08 | 1,09 / 0,34 / 0,76 | 0,23 / −0,44 / 2,28 | 2,08 / −0,70 / 0,42 | −0,15 / −0,15 / 2,22 |
| S09 | 0,70 / 1,15 / 0,76 | −0,86 / 0,15 / 3,80 | 1,38 / 1,19 / 0,15 | 1,10 / 1,35 / 0,04 |

Détail par sleeve (PnL, séances, jours gagnants) dans [08_regimes.csv](outputs/08_regimes.csv).

## Regroupement des pertes

Jours où au moins 4 des 8 sleeves perdent simultanément : **5,2 %** des séances.

**Les 10 pires semaines**

| période | book | pire sleeve | PnL pire sleeve | sleeves perdantes | jour MNQ dominant | ER20 MNQ dominant (ex post) | ATR% MNQ dominant |
|---|---|---|---|---|---|---|---|
| 2025-03-07 | −2 386,02 | S02 | −1 625,50 | 3 | bas | haut | moyen |
| 2026-03-06 | −2 111,52 | S05 | −1 212,00 | 4 | bas | bas | moyen |
| 2025-01-24 | −2 011,34 | S08 | −833,24 | 6 | bas | bas | moyen |
| 2026-01-23 | −1 799,20 | S02 | −666,00 | 5 | moyen | moyen | bas |
| 2025-01-31 | −1 709,92 | S02 | −1 360,00 | 4 | moyen | moyen | moyen |
| 2022-05-13 | −1 341,20 | S02 | −1 340,50 | 3 | moyen | moyen | haut |
| 2025-12-12 | −1 256,34 | S08 | −477,42 | 6 | moyen | moyen | moyen |
| 2025-03-21 | −1 232,70 | S02 | −709,50 | 5 | moyen | bas | haut |
| 2021-06-11 | −977,60 | S01 | −480,00 | 4 | moyen | haut | bas |
| 2025-06-13 | −970,12 | S09 | −401,10 | 5 | bas | haut | bas |

**Les 5 pires mois**

| période | book | pire sleeve | PnL pire sleeve | sleeves perdantes | jour MNQ dominant | ER20 MNQ dominant (ex post) | ATR% MNQ dominant |
|---|---|---|---|---|---|---|---|
| 2025-03-31 | −3 787,44 | S02 | −2 235,00 | 6 | moyen | bas | haut |
| 2025-01-31 | −2 816,68 | S02 | −1 818,00 | 6 | moyen | bas | moyen |
| 2026-03-31 | −2 535,14 | S05 | −1 212,00 | 6 | bas | bas | moyen |
| 2023-09-30 | −1 696,02 | S01 | −1 204,50 | 7 | moyen | bas | bas |
| 2025-07-31 | −1 354,10 | S02 | −975,50 | 4 | bas | haut | bas |

**Pertes simultanées par régime**

| Variable | Régime | Séances | P(≥ 4 sleeves perdantes) | PnL book | PnL moyen | Sharpe |
|---|---|---:|---:|---:|---:|---:|
| Jour MNQ (ex post) | bas | 426 | 6,3 % | −52 187 | −122,50 | −6,30 |
| Jour MNQ (ex post) | moyen | 397 | 7,6 % | −6 410 | −16,15 | −0,72 |
| Jour MNQ (ex post) | haut | 418 | 2,2 % | 118 188 | 282,75 | 10,75 |
| ER20 centré (ex post) | bas | 482 | 5,0 % | 35 685 | 74,04 | 2,72 |
| ER20 centré (ex post) | moyen | 359 | 6,1 % | 20 056 | 55,87 | 2,01 |
| ER20 centré (ex post) | haut | 390 | 4,9 % | 4 153 | 10,65 | 0,54 |
| ER20 fin J−1 (ex ante) | bas | 483 | 5,0 % | 26 375 | 54,61 | 2,36 |
| ER20 fin J−1 (ex ante) | moyen | 366 | 5,2 % | 9 472 | 25,88 | 0,95 |
| ER20 fin J−1 (ex ante) | haut | 392 | 5,9 % | 23 743 | 60,57 | 2,31 |

## Le book perd-il de façon groupée en range ou en retour à la moyenne ?

**À l'échelle de la journée, oui, et c'est le principal risque du book.** Ce risque est concentré sur les sleeves MNQ directionnelles.

- Sur les journées de range MNQ (tercile bas de abs(C−O)/(H−L), ex post), le book à 8 sleeves perd 52 187 USD en 426 séances (Sharpe −6,3, 26 % de jours gagnants). Sur les journées de tendance (tercile haut), il gagne 118 188 USD (81 % de jours gagnants).
- Les pertes de range viennent de S01 (−25 713 USD), de S02 (−51 612 USD) et de S04 en shadow (−40 840 USD). Ce sont des stratégies de breakout, qui paient leurs faux départs les jours sans suite.
- Les sleeves MGC (S05, S06) et MBT (S07, S08) ne dépendent presque pas du type de journée MNQ. C'est leur apport de diversification.
- Les journées de range ne font pas perdre 4 sleeves ou plus en même temps beaucoup plus souvent : 6,3 % contre 5,2 % en moyenne et 2,2 % en tendance. La perte est groupée par instrument, pas sur tout le book.

**À l'échelle du mois, non.** Le book fait mieux pendant les mois de faible efficiency ratio MNQ, donc plus hachés (ER20 centré bas : Sharpe 2,72, contre 0,54 en ER20 haut). Les journées de tendance intraday surviennent aussi pendant les mois sans tendance de fond. L'ER20 ex ante, connu la veille, ne trie pas le résultat (Sharpe 2,36 / 0,95 / 2,31) : il n'y a là aucun filtre exploitable sans nouvelle hypothèse.

**Concentration.**

- S02 porte 41 % du risque du book sur 2024-01 → 2026-03, et 44 % depuis 2021.
- Elle est la pire sleeve de 5 des 10 pires semaines et de 3 des 5 pires mois.
- Les pires périodes se regroupent sur janvier et mars 2025 (−6 604 USD à eux deux) et sur mars 2026.

Ces résultats sont **descriptifs**. Le type de journée n'est connu qu'à la clôture et ne peut pas servir de filtre. Ils disent où le book est exposé, pas comment l'éviter.
