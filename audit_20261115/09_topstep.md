# Phase 9 — Book sous contrainte Topstep

*Exécution agrégée par instrument, book à 8 sleeves, S01 = ATR 18, calendrier union. IS 2021-05-03 → 2023-12-29 (choix), validation 2024-01-02 → 2026-03-19 (reportée). 1 MBT = 1 lot, un micro = 0,1 lot. Une entrée qui dépasse la limite est réduite ; une position ouverte n'est jamais coupée. Script : [phase9_topstep.py](phase9_topstep.py), simulateur [sim_variants.py](sim_variants.py). Tables : [09_topstep.csv](outputs/09_topstep.csv), [09_lost_by_sleeve.csv](outputs/09_lost_by_sleeve.csv), [09_lifecycle.csv](outputs/09_lifecycle.csv).*

**Gate** : avec la variante (i), le simulateur reproduit `src/portfolio/aggregator.py` sur les 4 configurations de limite (2, 3, 5 lots et sans limite), séance par séance, sur l'IS. Écart max : 0.0e+00 USD.

## Chiffres de tête — règle actuelle (i), palier XFA dynamique

Le palier dynamique suit le solde d'un XFA qui part de 0 : 2 lots sous 1 500 USD, 3 lots jusqu'à 2 000, puis 5. La ligne « 5 lots » correspond au Combine 50K. Elle ne contraint presque jamais le book, dont le pic est de 3,3 lots.

| Fenêtre | Palier | PnL | Sharpe | maxDD | Pire jour | Séances réduites | PnL perdu vs sans limite |
|---|---|---:|---:|---:|---:|---:|---:|
| IS | XFA dynamique | 29 142 | 2,316 | −2 567 | −967 | 21 / 689 | −574 |
| validation | XFA dynamique | 29 270 | 1,647 | −5 863 | −1 656 | 41 / 572 | 1 616 |
| IS | 5 lots | 28 568 | 2,253 | −2 891 | −967 | 0 / 689 | 0 |
| validation | 5 lots | 30 886 | 1,713 | −5 863 | −1 656 | 0 / 572 | 0 |

## PnL perdu à cause de la limite, par sleeve (variante i, palier dynamique)

Pour chaque instrument, la perte est exacte : PnL sans limite moins PnL avec limite. Sa répartition entre les sleeves de l'instrument se fait au prorata des contrats refusés à l'entrée.

| Fenêtre | S01 | S02 | S03 | S05 | S06 | S07 | S08 | S09 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS — PnL perdu | 6 | −24 | 6 | 0 | 0 | 129 | −631 | −61 |
| IS — contrats refusés | 0 | 3 | 0 | 0 | 0 | 5 | 14 | 5 |
| validation — PnL perdu | −314 | 949 | −410 | 116 | 0 | 556 | 160 | 560 |
| validation — contrats refusés | 5 | 5 | 3 | 1 | 0 | 12 | 33 | 11 |

## Variantes d'exécution (signaux inchangés)

- **(i)** Règle actuelle : premier arrivé, premier servi, avec rattrapage.
- **(ii)** Pas de rattrapage après la dernière heure d'entrée de la ou des sleeves sous-servies de l'instrument.
- **(iii)** Priorité fixée sur l'IS : une entrée ne peut pas consommer la place réservée aux sleeves à plat de priorité plus haute. Le score est le PnL IS par unité de contribution au risque et par dixième de lot.
- **(iv)** Aucune sleeve MBT tant que le palier est inférieur à 3 lots.

| Sleeve | Dernière entrée observée (IS) | Contribution au risque IS | PnL IS | Priorité (iii), 0 = haute |
|---|---|---:|---:|---:|
| S01 | 15:22 | 29,6 % | 9 120 | 2 |
| S02 | 16:02 | 51,2 % | 12 882 | 4 |
| S03 | 15:51 | 2,6 % | 1 527 | 1 |
| S05 | 13:08 | 1,8 % | 1 292 | 0 |
| S06 | 08:36 | 2,0 % | 578 | 3 |
| S07 | 15:31 | 4,2 % | 33 | 7 |
| S08 | 15:47 | 5,8 % | 435 | 6 |
| S09 | 15:55 | 2,8 % | 2 672 | 5 |

| Variante | Fenêtre | Palier | PnL | Sharpe | maxDD | Pire jour | Séances réduites | PnL perdu |
|---|---|---|---:|---:|---:|---:|---:|---:|
| i | IS | 2 lots | 26 920 | 2,283 | −2 331 | −967 | 196 | 1 648 |
| i | IS | 3 lots | 27 947 | 2,243 | −2 891 | −967 | 74 | 620 |
| i | IS | 5 lots | 28 568 | 2,253 | −2 891 | −967 | 0 | 0 |
| i | IS | XFA dynamique | 29 142 | 2,316 | −2 567 | −967 | 21 | −574 |
| i | validation | 2 lots | 25 857 | 1,582 | −5 359 | −1 656 | 279 | 5 029 |
| i | validation | 3 lots | 29 410 | 1,674 | −5 481 | −1 656 | 113 | 1 476 |
| i | validation | 5 lots | 30 886 | 1,713 | −5 863 | −1 656 | 0 | 0 |
| i | validation | XFA dynamique | 29 270 | 1,647 | −5 863 | −1 656 | 41 | 1 616 |
| ii | IS | 2 lots | 27 140 | 2,299 | −2 331 | −967 | 196 | 1 427 |
| ii | IS | 3 lots | 27 947 | 2,243 | −2 891 | −967 | 74 | 620 |
| ii | IS | 5 lots | 28 568 | 2,253 | −2 891 | −967 | 0 | 0 |
| ii | IS | XFA dynamique | 29 258 | 2,324 | −2 567 | −967 | 21 | −690 |
| ii | validation | 2 lots | 25 909 | 1,584 | −5 345 | −1 656 | 279 | 4 977 |
| ii | validation | 3 lots | 29 410 | 1,674 | −5 481 | −1 656 | 113 | 1 476 |
| ii | validation | 5 lots | 30 886 | 1,713 | −5 863 | −1 656 | 0 | 0 |
| ii | validation | XFA dynamique | 29 291 | 1,648 | −5 863 | −1 656 | 41 | 1 595 |
| iii | IS | 2 lots | 26 768 | 2,249 | −2 694 | −973 | 377 | 1 799 |
| iii | IS | 3 lots | 27 468 | 2,231 | −2 612 | −967 | 168 | 1 100 |
| iii | IS | 5 lots | 28 568 | 2,253 | −2 891 | −967 | 0 | 0 |
| iii | IS | XFA dynamique | 29 389 | 2,344 | −2 567 | −967 | 41 | −821 |
| iii | validation | 2 lots | 25 886 | 1,599 | −4 706 | −1 296 | 447 | 5 000 |
| iii | validation | 3 lots | 28 898 | 1,673 | −5 542 | −1 656 | 222 | 1 988 |
| iii | validation | 5 lots | 30 886 | 1,713 | −5 863 | −1 656 | 0 | 0 |
| iii | validation | XFA dynamique | 30 566 | 1,720 | −5 863 | −1 656 | 63 | 320 |
| iv | IS | 2 lots | 25 416 | 2,172 | −2 536 | −979 | 413 | 3 152 |
| iv | IS | 3 lots | 27 947 | 2,243 | −2 891 | −967 | 74 | 620 |
| iv | IS | 5 lots | 28 568 | 2,253 | −2 891 | −967 | 0 | 0 |
| iv | IS | XFA dynamique | 29 334 | 2,341 | −2 567 | −967 | 41 | −767 |
| iv | validation | 2 lots | 21 584 | 1,435 | −3 948 | −1 138 | 495 | 9 301 |
| iv | validation | 3 lots | 29 410 | 1,674 | −5 481 | −1 656 | 113 | 1 476 |
| iv | validation | 5 lots | 30 886 | 1,713 | −5 863 | −1 656 | 0 | 0 |
| iv | validation | XFA dynamique | 29 472 | 1,665 | −5 863 | −1 656 | 76 | 1 414 |

## Cycle Combine → XFA → payout (complément)

Les réglages sont ceux de l'étude d'agrégation : 8 000 chemins de 126 séances tradées, blocs de 3, 5, 10 et 20, seed 20261112, frais standard, palier XFA relu chaque séance. Les chemins sont tirés dans chaque fenêtre séparément.

| Variante | Fenêtre | Cash net espéré | Cash net médian | P(payout) | Séances médianes au 1er payout | P(XFA perdu) | P(PASS) |
|---|---|---:|---:|---:|---:|---:|---:|
| i | IS | 607 | −50 | 78,8 % | 80 | 64,8 % | 89,8 % |
| i | validation | 972 | −81 | 71,6 % | 82 | 73,7 % | 89,5 % |
| ii | IS | 614 | −45 | 78,9 % | 80 | 64,8 % | 89,8 % |
| ii | validation | 975 | −77 | 71,7 % | 82 | 73,6 % | 89,5 % |
| iii | IS | 596 | −52 | 78,4 % | 81 | 64,6 % | 89,8 % |
| iii | validation | 1 030 | −84 | 70,1 % | 85 | 71,3 % | 89,5 % |
| iv | IS | 527 | −84 | 78,6 % | 81 | 66,0 % | 89,8 % |
| iv | validation | 865 | −172 | 69,3 % | 87 | 71,3 % | 89,5 % |

## Lecture

1. **Chiffre de tête sous contrainte** (règle actuelle, palier XFA dynamique) : 29 142 USD sur l'IS 2021-05 → 2023-12 (Sharpe 2,32, maxDD −2 567 USD) et 29 270 USD sur la validation 2024-01 → 2026-03 (Sharpe 1,65, maxDD −5 863 USD, pire jour −1 656 USD). Sans limite, la validation fait 30 886 USD : la limite coûte 1 616 USD sur la validation, et en fait gagner 574 sur l'IS, où les entrées refusées étaient perdantes en moyenne.
2. **La limite ne mord vraiment qu'au palier de 2 lots.** Un XFA qui démarre à 0 passe vite au-dessus de 2 000 USD dans l'historique, d'où seulement 21 (IS) et 41 (validation) séances réduites en dynamique. À 2 lots fixes, 196 et 279 séances sont réduites. MBT, à 1 lot par contrat, est la cause : S08 a le plus de contrats refusés.
3. **Choix sur l'IS, avec pour critère le cash net espéré du cycle complet (objectif payout)** :
   - (ii) fait +7 USD par rapport à (i), ce qui est indiscernable du bruit ;
   - (iii) fait −11 USD ;
   - (iv) fait −80 USD, et perd 3 152 USD de PnL historique à 2 lots.

   **Aucune variante ne bat nettement la règle actuelle : (i) reste la règle.** (iv) est rejetée. Sur la validation, simplement reportée, (iii) fait un peu mieux (1 030 contre 972 USD) ; ce n'est pas un motif de changement, puisque l'IS ne la favorisait pas.
4. **Constat d'exécution.** S02 a des entrées observées jusqu'à 16:02 sur l'IS, soit 6 minutes avant le flatten de 16:08. Son vote peut se retourner après la dernière entrée des cellules, quand une cellule sort.
5. **Le cycle reste défavorable à la médiane.** Le cash net médian est négatif (−50 USD sur l'IS) et l'XFA est perdu dans 65 à 74 % des chemins, quelle que soit la variante. Le cash espéré positif vient d'une minorité de chemins à plusieurs payouts.

La répartition du PnL perdu entre sleeves d'un même instrument est une approximation, au prorata des contrats refusés. Ses signes peuvent être contre-intuitifs quand les entrées refusées étaient perdantes.
