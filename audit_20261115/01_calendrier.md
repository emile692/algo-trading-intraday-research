# Phase 1 — Correction du calendrier

*Fenêtre : 2024-01-02 → 2026-03-19. PnL quotidien net en USD, coûts nominaux du lock. Sharpe : √252 × moyenne / écart-type, jours sans trade inclus. S01 : ensemble ATR 18 de la fiche 01 (décision du 2026-11-15). Script : [phase1_calendar.py](phase1_calendar.py). Sorties : [daily_pnl_union.csv](outputs/daily_pnl_union.csv), [01_missing_sessions.csv](outputs/01_missing_sessions.csv), [01_section4_risk.csv](outputs/01_section4_risk.csv), [01_section5_loo.csv](outputs/01_section5_loo.csv).*

## Ce qui change

Le lock aligne les neuf sleeves sur les 564 séances du calendrier MNQ. Le calendrier union des trois instruments compte **572 séances** sur la même fenêtre. Les 8 séances ajoutées sont des jours où MGC ou MBT tradent sans séance MNQ dans les fichiers. Elles pèsent −509,26 USD pour le book à 8 sleeves et −509,26 USD pour le book à 9. Aucune séance du lock n'est absente de l'union (0).

Deux corrections se cumulent dans la colonne C : le S01 de référence (B − A) et le calendrier (C − B). Contrôle : sur l'union, la somme quotidienne de chaque sleeve égale la somme de ses trades (écart max < 10⁻⁶ USD).

## Book avant / après

| Variante | Book | Séances | PnL | Vol. quotidienne | Sharpe | Pire jour | maxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — lock publié (S01 `baseline_fixed`, calendrier MNQ) | 9 | 564 | 37 044,22 | 677,34 | 1,539 | −1 664,46 | −6 305,18 |
| A — lock publié (S01 `baseline_fixed`, calendrier MNQ) | 8 | 564 | 31 661,22 | 509,00 | 1,751 | −1 655,96 | −5 425,58 |
| B — S01 de référence, calendrier MNQ | 9 | 564 | 36 750,22 | 671,07 | 1,541 | −1 664,46 | −6 350,78 |
| B — S01 de référence, calendrier MNQ | 8 | 564 | 31 367,22 | 503,43 | 1,754 | −1 655,96 | −5 626,08 |
| C — **S01 de référence, calendrier union (corrigé)** | 9 | 572 | 36 240,96 | 666,67 | 1,509 | −1 664,46 | −6 594,74 |
| C — **S01 de référence, calendrier union (corrigé)** | 8 | 572 | 30 857,96 | 500,28 | 1,712 | −1 655,96 | −5 870,04 |

Book à 8 sleeves corrigé : **30 857,96 USD, Sharpe 1,712, maxDD −5 870,04, pire jour −1 655,96** (lock publié : 31 661,22, 1,751, −5 425,58).

## Séances manquantes du lock

Les sleeves MNQ ne tradent pas ces jours-là : leurs colonnes, toutes nulles, sont omises.

| session_date | instruments | S05 | S06 | S07 | S08 | S09 | book8 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024-03-15 | MGC+MBT | 0,00 | 0,00 | −23,32 | −143,82 | −41,32 | −208,46 |
| 2024-06-21 | MGC+MBT | 165,50 | 0,00 | −38,82 | −5,82 | 0,00 | 120,86 |
| 2024-09-20 | MGC+MBT | 0,00 | 0,00 | 0,00 | −13,32 | 0,00 | −13,32 |
| 2024-12-20 | MGC+MBT | −85,50 | 0,00 | 0,00 | 21,18 | −28,32 | −92,64 |
| 2025-03-21 | MGC+MBT | −150,50 | 0,00 | 0,00 | −93,46 | 0,00 | −243,96 |
| 2025-06-20 | MGC+MBT | 0,00 | −165,50 | −83,82 | 65,18 | 152,18 | −31,96 |
| 2025-09-19 | MGC+MBT | 0,00 | 0,00 | 0,00 | 62,68 | 0,18 | 62,86 |
| 2025-12-19 | MGC+MBT | 0,00 | 0,00 | 0,00 | −102,64 | 0,00 | −102,64 |
| **Total** | | −70,50 | −165,50 | −145,96 | −210,02 | 82,72 | −509,26 |

## Section 4 recalculée — risque réalisé, book à 9 sleeves

« Avant » = lock publié (A). « Après » = S01 de référence et calendrier union (C). Contribution au risque = cov(sleeve, book) / var(book).

| Sleeve | Vol. avant | Vol. après | Pire jour avant | Pire jour après | maxDD avant | maxDD après | Contrib. risque avant | Contrib. risque après |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S01 | 145,57 | 130,74 | −249,00 | −249,00 | −1 081,50 | −875,50 | 5,9 % | 4,6 % |
| S02 | 290,74 | 288,70 | −1 138,00 | −1 138,00 | −3 714,00 | −3 714,00 | 33,1 % | 33,4 % |
| S03 | 58,95 | 58,53 | −268,50 | −268,50 | −940,50 | −940,50 | 3,2 % | 3,3 % |
| S04 | 279,22 | 277,26 | −823,50 | −823,50 | −4 212,50 | −4 212,50 | 30,3 % | 30,5 % |
| S05 | 123,33 | 122,88 | −874,50 | −874,50 | −1 622,50 | −1 622,50 | 5,3 % | 5,4 % |
| S06 | 149,96 | 149,08 | −909,50 | −909,50 | −1 399,00 | −1 564,50 | 6,8 % | 7,0 % |
| S07 | 73,76 | 73,36 | −355,32 | −355,32 | −1 285,80 | −1 285,80 | 3,3 % | 3,4 % |
| S08 | 105,46 | 105,16 | −533,64 | −533,64 | −1 449,10 | −1 542,56 | 5,4 % | 5,6 % |
| S09 | 99,84 | 99,36 | −229,32 | −229,32 | −2 068,58 | −1 916,40 | 6,8 % | 6,9 % |

## Section 4 recalculée — book actif à 8 sleeves

| Sleeve | Vol. avant | Vol. après | Pire jour avant | Pire jour après | maxDD avant | maxDD après | Contrib. risque avant | Contrib. risque après |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S01 | 145,57 | 130,74 | −249,00 | −249,00 | −1 081,50 | −875,50 | 8,9 % | 7,3 % |
| S02 | 290,74 | 288,70 | −1 138,00 | −1 138,00 | −3 714,00 | −3 714,00 | 40,6 % | 41,0 % |
| S03 | 58,95 | 58,53 | −268,50 | −268,50 | −940,50 | −940,50 | 3,6 % | 3,7 % |
| S05 | 123,33 | 122,88 | −874,50 | −874,50 | −1 622,50 | −1 622,50 | 10,5 % | 10,7 % |
| S06 | 149,96 | 149,08 | −909,50 | −909,50 | −1 399,00 | −1 564,50 | 12,7 % | 13,0 % |
| S07 | 73,76 | 73,36 | −355,32 | −355,32 | −1 285,80 | −1 285,80 | 5,2 % | 5,2 % |
| S08 | 105,46 | 105,16 | −533,64 | −533,64 | −1 449,10 | −1 542,56 | 8,7 % | 9,1 % |
| S09 | 99,84 | 99,36 | −229,32 | −229,32 | −2 068,58 | −1 916,40 | 9,8 % | 9,9 % |

## Section 5 recalculée — leave-one-out, book à 9 sleeves

| Sleeve retirée | PnL avant | PnL après | Sharpe avant | Sharpe après | maxDD avant | maxDD après | ΔmaxDD après |
|---|---:|---:|---:|---:|---:|---:|---:|
| S01 | 30 715,22 | 30 205,96 | 1,324 | 1,292 | −6 670,64 | −7 007,24 | −412,50 |
| S02 | 27 765,22 | 26 961,96 | 1,595 | 1,557 | −5 704,60 | −5 386,86 | 1 207,88 |
| S03 | 36 451,72 | 35 648,46 | 1,559 | 1,528 | −6 543,68 | −6 833,24 | −238,50 |
| S04 | 31 661,22 | 30 857,96 | 1,751 | 1,712 | −5 425,58 | −5 870,04 | 724,70 |
| S05 | 35 256,22 | 34 523,46 | 1,522 | 1,494 | −6 123,18 | −6 262,24 | 332,50 |
| S06 | 32 946,22 | 32 308,46 | 1,433 | 1,410 | −6 416,78 | −6 728,74 | −134,00 |
| S07 | 34 689,18 | 34 031,88 | 1,482 | 1,457 | −6 171,16 | −6 097,86 | 496,88 |
| S08 | 32 699,12 | 32 105,88 | 1,419 | 1,399 | −6 368,38 | −6 263,18 | 331,56 |
| S09 | 34 169,64 | 33 283,66 | 1,509 | 1,473 | −6 279,24 | −6 373,20 | 221,54 |

## Leave-one-out, book actif à 8 sleeves

| Sleeve retirée | PnL avant | PnL après | Sharpe avant | Sharpe après | maxDD avant | maxDD après | ΔmaxDD après |
|---|---:|---:|---:|---:|---:|---:|---:|
| S01 | 25 332,22 | 24 822,96 | 1,474 | 1,433 | −5 995,08 | −6 239,04 | −369,00 |
| S02 | 22 382,22 | 21 578,96 | 1,727 | 1,672 | −3 667,28 | −3 259,04 | 2 611,00 |
| S03 | 31 068,72 | 30 265,46 | 1,771 | 1,732 | −5 660,58 | −6 105,04 | −235,00 |
| S05 | 29 873,22 | 29 140,46 | 1,793 | 1,758 | −5 124,28 | −5 285,74 | 584,30 |
| S06 | 27 563,22 | 26 925,46 | 1,670 | 1,641 | −5 440,28 | −5 752,24 | 117,80 |
| S07 | 29 306,18 | 28 648,88 | 1,692 | 1,660 | −4 860,52 | −5 236,98 | 633,06 |
| S08 | 27 316,12 | 26 722,88 | 1,620 | 1,596 | −4 407,94 | −4 815,54 | 1 054,50 |
| S09 | 28 786,64 | 27 900,66 | 1,735 | 1,687 | −5 270,22 | −5 536,38 | 333,66 |

Le panel corrigé `outputs/daily_pnl_union.csv` couvre 2019-05-06 → 2026-10-02 sur le calendrier union. Une sleeve y vaut NaN hors de la plage de ses données (MNQ/MGC après le 2026-03-19, MBT avant le 2021-05-03). Toutes les phases suivantes l'utilisent.
