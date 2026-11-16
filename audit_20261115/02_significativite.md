# Phase 2 — Significativité statistique

*Données : panel corrigé de la phase 1. S01 = ensemble ATR 18. PnL quotidien net en USD, coûts nominaux. Sharpe = √252 × moyenne / écart-type, jours sans trade inclus. Calendrier : séances de l'instrument pour une sleeve, calendrier union pour un book. Script : [phase2_significance.py](phase2_significance.py), [stats.py](stats.py). Tables : [02_significance.csv](outputs/02_significance.csv), [02_deflated_sharpe.csv](outputs/02_deflated_sharpe.csv).*

## Méthodes

- **Erreur type de Lo (2002)** : estimateur GMM du Sharpe, matrice de covariance de long terme de Newey-West (noyau de Bartlett, retard ⌊4 (T/100)^(2/9)⌋, soit 7 ou 8 jours). Elle tient compte de l'autocorrélation et des queues épaisses. t = Sharpe / SE.
- **Bootstrap stationnaire de Politis-Romano** : blocs de longueur géométrique, moyennes 5 et 20 séances, 10 000 tirages, seed 20261115. IC à 95 % par percentiles.
- **PSR(0)** : probabilité que le vrai Sharpe soit positif, avec correction de skew et de kurtosis (Bailey et López de Prado 2012).
- **DSR** : PSR contre SR₀, le Sharpe maximal attendu parmi N essais sans talent, compte tenu de la dispersion des Sharpe IS entre essais (Bailey et López de Prado 2014). Il est calculé sur l'IS, la fenêtre de sélection, et sur le total.
- **Book** : IS 2021-05-03 → 2023-12-31, validation 2024-01-02 → 2026-03-19, total 2021-05-03 → 2026-03-19, soit les fenêtres où les trois instruments ont des données. Pour les sleeves MBT, 2024-01 → 2025-05 fait partie de leur IS : la « validation » du book est donc mixte.

**Règle de classement, fixée avant le calcul** : « négative » si le Sharpe est < 0 ; « distinguable de zéro » si t ≥ 1,96 et que les bornes basses des deux IC bootstrap sont > 0 ; « indistinguable » sinon.

## Classement

| Sleeve | IS | Validation réutilisée | Total |
|---|---|---|---|
| S01 | distinguable de zéro | distinguable de zéro | **distinguable de zéro** |
| S02 | indistinguable | indistinguable | **distinguable de zéro** |
| S03 | indistinguable | indistinguable | **indistinguable** |
| S04 | indistinguable | indistinguable | **indistinguable** |
| S05 | indistinguable | indistinguable | **indistinguable** |
| S06 | indistinguable | indistinguable | **indistinguable** |
| S07 | indistinguable | indistinguable | **indistinguable** |
| S08 | indistinguable | indistinguable | **indistinguable** |
| S09 | distinguable de zéro | négative | **distinguable de zéro** |
| Book 8 | distinguable de zéro | distinguable de zéro | **distinguable de zéro** |
| Book 9 | distinguable de zéro | distinguable de zéro | **distinguable de zéro** |

## Détail

| Série | Fenêtre | Dates | Séances | PnL | Sharpe | SE Lo (HAC) | t | IC 95 % bloc 5 | IC 95 % bloc 20 | PSR(0) | Classe |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| S01 | IS | 2019-05-06 → 2024-02-27 | 1225 | 13 852,00 | 1,265 | 0,410 | 3,09 | [0,46 ; 2,06] | [0,52 ; 2,00] | 0,998 | distinguable de zéro |
| S01 | validation réutilisée | 2024-02-28 → 2026-03-19 | 522 | 5 564,50 | 1,325 | 0,582 | 2,28 | [0,18 ; 2,40] | [0,37 ; 2,23] | 0,977 | distinguable de zéro |
| S01 | total | 2019-05-06 → 2026-03-19 | 1747 | 19 416,50 | 1,281 | 0,331 | 3,86 | [0,63 ; 1,94] | [0,69 ; 1,87] | 1,000 | distinguable de zéro |
| S02 | IS | 2019-05-06 → 2023-12-29 | 1184 | 10 980,00 | 0,859 | 0,474 | 1,81 | [−0,07 ; 1,79] | [−0,02 ; 1,70] | 0,968 | indistinguable |
| S02 | validation réutilisée | 2024-01-02 → 2026-03-19 | 563 | 9 279,00 | 0,899 | 0,620 | 1,45 | [−0,40 ; 2,05] | [−0,25 ; 2,01] | 0,928 | indistinguable |
| S02 | total | 2019-05-06 → 2026-03-19 | 1747 | 20 259,00 | 0,848 | 0,367 | 2,31 | [0,11 ; 1,53] | [0,18 ; 1,50] | 0,991 | distinguable de zéro |
| S03 | IS | 2019-05-06 → 2023-12-29 | 1184 | 1 663,00 | 0,546 | 0,415 | 1,32 | [−0,30 ; 1,33] | [−0,33 ; 1,31] | 0,890 | indistinguable |
| S03 | validation réutilisée | 2024-01-02 → 2026-03-19 | 563 | 592,50 | 0,283 | 0,566 | 0,50 | [−1,08 ; 1,24] | [−1,06 ; 1,24] | 0,674 | indistinguable |
| S03 | total | 2019-05-06 → 2026-03-19 | 1747 | 2 255,50 | 0,432 | 0,323 | 1,34 | [−0,28 ; 1,04] | [−0,29 ; 1,03] | 0,889 | indistinguable |
| S04 | IS | 2019-05-06 → 2023-12-29 | 1184 | 4 692,00 | 0,354 | 0,422 | 0,84 | [−0,47 ; 1,18] | [−0,39 ; 1,04] | 0,781 | indistinguable |
| S04 | validation réutilisée | 2024-01-02 → 2026-03-19 | 563 | 5 383,00 | 0,543 | 0,609 | 0,89 | [−0,83 ; 1,60] | [−0,86 ; 1,55] | 0,809 | indistinguable |
| S04 | total | 2019-05-06 → 2026-03-19 | 1747 | 10 075,00 | 0,424 | 0,345 | 1,23 | [−0,31 ; 1,06] | [−0,25 ; 1,02] | 0,878 | indistinguable |
| S05 | IS | 2019-05-06 → 2023-12-29 | 1203 | 1 418,00 | 0,519 | 0,491 | 1,06 | [−0,45 ; 1,43] | [−0,55 ; 1,46] | 0,876 | indistinguable |
| S05 | validation réutilisée | 2024-01-02 → 2026-03-19 | 572 | 1 717,50 | 0,388 | 0,602 | 0,64 | [−1,25 ; 1,37] | [−1,17 ; 1,42] | 0,747 | indistinguable |
| S05 | total | 2019-05-06 → 2026-03-19 | 1775 | 3 135,50 | 0,370 | 0,337 | 1,10 | [−0,46 ; 0,96] | [−0,44 ; 1,00] | 0,877 | indistinguable |
| S06 | IS | 2019-05-06 → 2023-12-29 | 1203 | 3 098,00 | 0,878 | 0,398 | 2,20 | [0,03 ; 1,63] | [−0,03 ; 1,66] | 0,983 | indistinguable |
| S06 | validation réutilisée | 2024-01-02 → 2026-03-19 | 572 | 3 932,50 | 0,732 | 0,637 | 1,15 | [−0,78 ; 1,83] | [−0,73 ; 1,79] | 0,899 | indistinguable |
| S06 | total | 2019-05-06 → 2026-03-19 | 1775 | 7 030,50 | 0,677 | 0,343 | 1,97 | [−0,07 ; 1,28] | [−0,05 ; 1,28] | 0,984 | indistinguable |
| S07 | IS | 2021-05-03 → 2025-05-28 | 759 | 1 135,24 | 0,390 | 0,534 | 0,73 | [−0,63 ; 1,46] | [−0,54 ; 1,33] | 0,748 | indistinguable |
| S07 | validation réutilisée | 2025-05-29 → 2026-10-02 | 326 | 947,24 | 0,638 | 0,892 | 0,71 | [−1,17 ; 2,28] | [−0,92 ; 2,05] | 0,770 | indistinguable |
| S07 | total | 2021-05-03 → 2026-10-02 | 1085 | 2 082,48 | 0,473 | 0,465 | 1,02 | [−0,43 ; 1,38] | [−0,35 ; 1,29] | 0,836 | indistinguable |
| S08 | IS | 2021-05-03 → 2025-05-28 | 759 | 2 535,40 | 0,621 | 0,586 | 1,06 | [−0,53 ; 1,78] | [−0,50 ; 1,77] | 0,858 | indistinguable |
| S08 | validation réutilisée | 2025-05-29 → 2026-10-02 | 326 | 1 364,66 | 0,687 | 0,919 | 0,75 | [−1,13 ; 2,45] | [−1,20 ; 2,37] | 0,788 | indistinguable |
| S08 | total | 2021-05-03 → 2026-10-02 | 1085 | 3 900,06 | 0,642 | 0,495 | 1,30 | [−0,33 ; 1,61] | [−0,34 ; 1,62] | 0,909 | indistinguable |
| S09 | IS | 2021-05-03 → 2025-05-28 | 759 | 4 989,46 | 1,314 | 0,497 | 2,65 | [0,31 ; 2,27] | [0,56 ; 2,15] | 0,992 | distinguable de zéro |
| S09 | validation réutilisée | 2025-05-29 → 2026-10-02 | 326 | −73,22 | −0,040 | 0,789 | −0,05 | [−1,68 ; 1,42] | [−1,83 ; 1,40] | 0,482 | négative |
| S09 | total | 2021-05-03 → 2026-10-02 | 1085 | 4 916,24 | 0,874 | 0,416 | 2,10 | [0,03 ; 1,67] | [0,10 ; 1,64] | 0,970 | distinguable de zéro |
| Book 8 | IS | 2021-05-03 → 2023-12-29 | 689 | 28 537,64 | 2,251 | 0,549 | 4,10 | [1,16 ; 3,34] | [1,28 ; 3,18] | 1,000 | distinguable de zéro |
| Book 8 | validation réutilisée | 2024-01-02 → 2026-03-19 | 572 | 30 857,96 | 1,712 | 0,586 | 2,92 | [0,45 ; 2,78] | [0,48 ; 2,83] | 0,999 | distinguable de zéro |
| Book 8 | total | 2021-05-03 → 2026-03-19 | 1261 | 59 395,60 | 1,869 | 0,387 | 4,83 | [1,08 ; 2,62] | [1,11 ; 2,61] | 1,000 | distinguable de zéro |
| Book 9 | IS | 2021-05-03 → 2023-12-29 | 689 | 34 161,64 | 1,949 | 0,509 | 3,83 | [0,91 ; 2,94] | [1,05 ; 2,80] | 1,000 | distinguable de zéro |
| Book 9 | validation réutilisée | 2024-01-02 → 2026-03-19 | 572 | 36 240,96 | 1,509 | 0,536 | 2,82 | [0,32 ; 2,49] | [0,36 ; 2,48] | 0,997 | distinguable de zéro |
| Book 9 | total | 2021-05-03 → 2026-03-19 | 1261 | 70 402,60 | 1,644 | 0,354 | 4,64 | [0,91 ; 2,33] | [0,96 ; 2,30] | 1,000 | distinguable de zéro |

## Deflated Sharpe Ratio

Un DSR > 0,95 signifie que le Sharpe observé reste significatif après prise en compte du nombre d'essais. La colonne N = 1000 est un stress arbitraire, pour les essais non documentés.

| Sleeve | Fenêtre | Sharpe | σ des Sharpe d'essais | N bas | SR₀ bas | DSR bas | N haut | SR₀ haut | DSR haut | DSR N = 1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S01 | IS | 1,265 | 0,224 | 160 | 0,603 | 0,935 | 400 | 0,669 | 0,913 | 0,889 |
| S01 | total | 1,281 | 0,224 | 160 | 0,603 | 0,968 | 400 | 0,669 | 0,953 | 0,934 |
| S02 | IS | 0,859 | 0,298 | 770 | 0,948 | 0,424 | 790 | 0,950 | 0,422 | 0,405 |
| S02 | total | 0,848 | 0,298 | 770 | 0,948 | 0,390 | 790 | 0,950 | 0,388 | 0,367 |
| S03 | IS | 0,546 | 0,182 | 45 | 0,406 | 0,623 | 45 | 0,406 | 0,623 | 0,460 |
| S03 | total | 0,432 | 0,182 | 45 | 0,406 | 0,529 | 45 | 0,406 | 0,529 | 0,326 |
| S04 | IS | 0,354 | 0,182 | 25 | 0,363 | 0,492 | 45 | 0,406 | 0,455 | 0,302 |
| S04 | total | 0,424 | 0,182 | 25 | 0,363 | 0,567 | 45 | 0,406 | 0,520 | 0,323 |
| S05 | IS | 0,519 | 0,257 | 25 | 0,514 | 0,505 | 187 | 0,706 | 0,339 | 0,239 |
| S05 | total | 0,370 | 0,257 | 25 | 0,514 | 0,326 | 187 | 0,706 | 0,146 | 0,071 |
| S06 | IS | 0,878 | 0,129 | 25 | 0,257 | 0,933 | 25 | 0,257 | 0,933 | 0,866 |
| S06 | total | 0,677 | 0,129 | 25 | 0,257 | 0,908 | 25 | 0,257 | 0,908 | 0,793 |
| S07 | IS | 0,390 | 0,229 | 30 | 0,475 | 0,442 | 38 | 0,497 | 0,427 | 0,271 |
| S07 | total | 0,473 | 0,229 | 30 | 0,475 | 0,498 | 38 | 0,497 | 0,480 | 0,286 |
| S08 | IS | 0,621 | 0,180 | 25 | 0,360 | 0,674 | 34 | 0,383 | 0,659 | 0,523 |
| S08 | total | 0,642 | 0,180 | 25 | 0,360 | 0,721 | 34 | 0,383 | 0,705 | 0,545 |
| S09 | IS | 1,314 | 0,374 | 36 | 0,803 | 0,827 | 48 | 0,846 | 0,806 | 0,571 |
| S09 | total | 0,874 | 0,374 | 36 | 0,803 | 0,561 | 48 | 0,846 | 0,525 | 0,229 |

Sources des essais :

- **S01** : cube ATR × q_low × q_high, 160 cellules (git 72ded70^, data/exports/atr_local_robustness_20260320_194355) ; N haut : + 224 variantes des campagnes ORB archivées (metrics_by_scope.csv) + 16 variantes des deux campagnes atr_ensemble du 2026-03-20. Sharpe du cube calculé dans l'ancien moteur ORB (avant la correction du cutoff du 18/09) ; sa dispersion est utilisée telle quelle.
- **S02** : cube S02, 770 cellules (research/signal_sheet_02_cube) ; N haut : + 20 overlays de sortie du money_management_lab. Sharpe du cube en rendements d'un compte de 50K (même convention que la fiche 02).
- **S03** : grille de raffinement s03, 45 cellules (research/signal_sheet_parameter_refinement/s03) ; N haut : aucune campagne antérieure retrouvée : N haut = N bas. Sharpe en USD (metric_block).
- **S04** : grille de raffinement s04, 25 cellules (research/signal_sheet_parameter_refinement/s04) ; N haut : + 20 overlays de sortie du money_management_lab. Sharpe en USD (metric_block).
- **S05** : grille de raffinement s05, 25 cellules (research/signal_sheet_parameter_refinement/s05) ; N haut : + 162 cellules de mgc_volatility_breakout_refinement_v0_20260925. Sharpe en USD (metric_block).
- **S06** : grille de raffinement s06, 25 cellules (research/signal_sheet_parameter_refinement/s06) ; N haut : aucune campagne antérieure retrouvée : N haut = N bas. Sharpe en USD (metric_block).
- **S07** : grille de raffinement s07, 30 cellules (research/signal_sheet_parameter_refinement/s07) ; N haut : + cellules de mbt_three_sleeve_freeze_20261004. Sharpe en USD (metric_block).
- **S08** : grille de raffinement s08, 25 cellules (research/signal_sheet_parameter_refinement/s08) ; N haut : + cellules de mbt_three_sleeve_freeze_20261004. Sharpe en USD (metric_block).
- **S09** : grille de raffinement s09, 36 cellules (research/signal_sheet_parameter_refinement/s09) ; N haut : + cellules de mbt_three_sleeve_freeze_20261004. Sharpe en USD (metric_block).

## Lecture

1. **Une seule sleeve est distinguable de zéro sur chaque fenêtre prise isolément : S01.** Son t de Lo vaut 3,09 sur l'IS et 2,28 sur la validation. Son DSR vaut 0,91 à 0,94 sur l'IS selon N, et passe 0,95 sur le total. C'est la seule sleeve dont le DSR approche le seuil.
2. **S02 et S09 ne sont « distinguables » que sur le total.**
   - S02 n'atteint t ≥ 1,96 ni sur l'IS ni sur la validation. Après déflation par les 770 cellules du cube, son DSR tombe à 0,39–0,42 : son Sharpe IS de 0,86 est inférieur au maximum attendu par hasard parmi 770 essais (SR₀ ≈ 0,95).
   - S09 doit tout à son IS (Sharpe 1,31). Il est nul sur la validation (−0,04), et son DSR sur le total vaut 0,53.
3. **S03 à S08 sont indistinguables de zéro sur toutes les fenêtres.** S06 est la plus proche (t = 1,97 sur le total, DSR 0,91–0,93), mais un des IC bootstrap contient 0. S05 a le DSR le plus faible (0,07 à 0,33 sur le total).
4. **Le book est distinguable de zéro sur toutes les fenêtres** (t de 2,8 à 4,8, borne basse de l'IC ≥ 0,32). C'est un effet de diversification : des sleeves individuellement faibles, peu corrélées, s'additionnent. Le DSR n'est pas calculé pour le book : aucun nombre d'essais au niveau du book n'est documenté. Or le choix des sleeves et de S04 a été fait en voyant la fenêtre 2024-01 → 2026-03.
5. **La validation n'est pas propre.** Toutes les fenêtres « validation » ont été consultées avant le gel des clusters. Les t de validation sont donc un plafond, pas une preuve indépendante.

Conséquence pratique : à l'exception de S01, aucune sleeve n'a à elle seule un historique qui permettrait de détecter rapidement une dégradation. C'est ce qui rend la règle d'arrêt de la phase 5 nécessairement lente ou bruitée.
