# Diagnostic macro et unités de risque — IS uniquement

## Résumé

- Bornes verrouillées : A/NQ du 2010-06-07 au 2024-02-22 ; B/MGC du 2015-01-02 au 2023-03-16. Aucune statistique post-IS.
- `R = PnL $ / (ATR14 Wilder de la séance précédente × valeur du point)` ; 1 946/1 954 trades NQ ont un ATR disponible après warm-up, contre 1 333/1 333 pour MGC.
- A/NQ : brut -21.49 R, coûts 19.51 R, net -41.00 R ; secondaire : brut $-58,914.64, coûts $24,325.00, net $-83,239.64.
- B/MGC : brut -3.39 R, coûts 44.13 R, net -47.52 R ; secondaire : brut $-764.00, coûts $5,998.50, net $-6,762.50.
- 36 hypothèses planifiées (34 anciennes + A1 + B1), 35 estimables ; Holm utilise la multiplicité conservatrice 36. Aucun rejet à 5 %.
- A1 ne sépare pas les pertes : différence macro moins hors macro = +0,0028 R/trade, p=0,8926, Holm=1.
- B1 est non estimable : zéro climax dans la fenêtre d’annonce parmi 1 333 trades ; la séance MGC exclut par définition la fenêtre FOMC de 14:00.
- La queue A reste concentrée, mais moins qu’en dollars : les dix pires R expliquent 39,7 % de la perte nette R du bucket NQ [0,50–0,75).
- B0d exclut un artefact de roll dominant : 76 trades roll ±2 séances, net -1,82 R/-$266, contre -45,70 R/-$6 496,50 hors roll.
- B0b utilise les dix contrats GC les plus liquides pour 18,7575 $ : 5 137 924 lignes, un seul appel payant, sans clé persistée.

## Calendriers officiels parsés

Les lignes CPI et Employment Situation sont extraites des pages annuelles BLS. Les dates FOMC sont extraites des liens HTML officiels `monetaryYYYYMMDDa.htm` des policy statements de la Fed ; aucune date n’est codée en dur. Les 9 dates de 2010, 9 de 2019 et 12 de 2020 incluent donc les statements non programmés effectivement publiés. A1 est strictement le jour de publication.

### Couverture dans A

| year | CPI | FOMC | NFP |
| --- | --- | --- | --- |
| 2010 | 7 | 5 | 6 |
| 2011 | 12 | 8 | 12 |
| 2012 | 12 | 8 | 12 |
| 2013 | 12 | 8 | 12 |
| 2014 | 12 | 8 | 12 |
| 2015 | 12 | 8 | 12 |
| 2016 | 12 | 8 | 12 |
| 2017 | 12 | 8 | 12 |
| 2018 | 12 | 8 | 12 |
| 2019 | 12 | 9 | 12 |
| 2020 | 12 | 12 | 12 |
| 2021 | 12 | 8 | 12 |
| 2022 | 12 | 8 | 12 |
| 2023 | 12 | 8 | 12 |
| 2024 | 2 | 1 | 2 |

### Couverture dans B

| year | CPI | FOMC | NFP |
| --- | --- | --- | --- |
| 2015 | 12 | 8 | 12 |
| 2016 | 12 | 8 | 12 |
| 2017 | 12 | 8 | 12 |
| 2018 | 12 | 8 | 12 |
| 2019 | 12 | 9 | 12 |
| 2020 | 12 | 12 | 12 |
| 2021 | 12 | 8 | 12 |
| 2022 | 12 | 8 | 12 |
| 2023 | 3 | 1 | 3 |

Le fichier de couverture complet conserve aussi la première et la dernière date officielle par année et les comptes des pages sources.

## Contrastes en R et A1/B1

Les dollars sont secondaires et descriptifs : ils ne créent pas une seconde famille de tests.

### Huit plus petites p-values brutes

| signal | variable | bucket_a | bucket_b | mean_net_R_difference_a_minus_b | p_value_R | holm_p_R |
| --- | --- | --- | --- | --- | --- | --- |
| A | A4_recovery_tercile | high | low | 0.0388 | 0.0184 | 0.6611 |
| B | B5_volume_tercile | low | mid | 0.0235 | 0.0325 | 1.0000 |
| A | A5_gap_bucket | >=1.00 | [0.50,0.75) | 0.0521 | 0.0712 | 1.0000 |
| B | B5_move_tercile | low | mid | 0.0198 | 0.0926 | 1.0000 |
| B | B4_gold_rv20_tercile | high | low | 0.0179 | 0.1389 | 1.0000 |
| B | session_bucket | 08:20-09:30 | 09:30-11:00 | -0.0339 | 0.1491 | 1.0000 |
| A | A4_recovery_tercile | high | mid | 0.0242 | 0.1537 | 1.0000 |
| B | session_bucket | 08:20-09:30 | 11:00-13:25 | -0.0324 | 0.1578 | 1.0000 |

La plus petite p-value brute est A4, reprise haute contre basse, p=0,0184 ; elle devient 0,6611 après Holm. Aucun résultat n’est compatible avec un état stable identifié au seuil familial de 5 %.

### A1 — jour CPI, NFP ou FOMC

| bucket | trades | gross_R | cost_R | net_R | mean_net_R | gross_usd | cost_usd | net_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro | 247 | -2.193 | 2.399 | -4.592 | -0.019 | -14,058.93 | 3,087.50 | -17,146.43 |
| no_macro | 1699 | -19.298 | 17.108 | -36.406 | -0.021 | -44,855.71 | 21,237.50 | -66,093.21 |

### B1 — barre d’annonce

| bucket | trades | gross_R | cost_R | net_R | mean_net_R | gross_usd | cost_usd | net_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outside | 1333 | -3.392 | 44.130 | -47.523 | -0.036 | -764.00 | 5,998.50 | -6,762.50 |

B1 ne possède que le bucket `outside`. Ce n’est pas un zéro de performance filtrée mais une absence d’observation dans la cellule d’annonce ; aucun contraste ni performance contrefactuelle n’est calculé.

## Queue de A/NQ

Les dix pires trades sont choisis séparément dans chaque bucket sur `net_R`. La MAE est reconstruite barre par barre ; son maximum concorde avec le ledger à moins de 1e-12 dollar. Une fraction de reprise positive signifie reprise du gap, négative extension.

### Concentration

| bucket | trades | bucket_net_R | worst10_net_R | worst10_share_of_bucket_net_loss_R | worst10_share_of_all_negative_trade_R |
| --- | --- | --- | --- | --- | --- |
| >=1.00 | 281 | 2.419 | -9.440 | — | 0.219 |
| [0.25,0.50) | 908 | -17.929 | -7.136 | 0.398 | 0.066 |
| [0.50,0.75) | 497 | -21.622 | -8.582 | 0.397 | 0.110 |
| [0.75,1.00) | 260 | -3.867 | -8.070 | 2.087 | 0.210 |

Pour [0,75–1,00), la part de 2,087 signifie que les dix pires cumulent plus de deux fois la perte nette R du bucket, les autres trades compensant une partie de cette queue. Pour ≥1,00, le bucket est globalement positif en R, donc aucune « part de perte nette » n’est définie.

### Description des 40 trades

| bucket | session_date | exit_reason | net_R | net_$ | MAE_5m_R | MAE_10m_R | MAE_20m_R | MAE_pic_R | heure_pic_MAE | VIX_t-1 | VIX/VIX3M_t-1 | trajet_gap_20m | fraction_reprise_20m | macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >=1.00 | 2017-10-27 | time | -1.323 | -1,162.50 | 0.381 | 0.398 | 0.614 | 1.394 | 11:50 | 11.300 | 0.874 | extension | -0.065 | none |
| >=1.00 | 2017-05-17 | stop | -1.285 | -852.50 | 0.339 | 0.618 | 0.949 | 1.522 | 10:50 | 10.650 | 0.842 | reprise | 0.056 | none |
| >=1.00 | 2016-04-22 | stop | -0.972 | -1,037.50 | 0.131 | 0.141 | 0.187 | 0.988 | 10:45 | 13.950 | 0.814 | reprise | 0.116 | none |
| >=1.00 | 2016-09-09 | time | -0.961 | -657.50 | 0.139 | 0.227 | 0.278 | 0.958 | 11:50 | 12.510 | 0.791 | reprise | 0.164 | none |
| >=1.00 | 2021-09-20 | time | -0.918 | -2,867.50 | 0.183 | 0.219 | 0.378 | 0.970 | 11:50 | 20.810 | 0.896 | reprise | 0.087 | none |
| >=1.00 | 2014-07-31 | stop | -0.838 | -587.50 | 0.243 | 0.407 | 0.599 | 0.863 | 10:50 | 13.330 | 0.935 | reprise | 0.116 | none |
| >=1.00 | 2020-03-02 | time | -0.834 | -3,207.50 | 0.082 | 0.082 | 0.082 | 0.868 | 11:50 | 40.110 | 1.344 | reprise | 0.056 | none |
| >=1.00 | 2018-10-26 | stop | -0.819 | -2,742.50 | 0.200 | 0.200 | 0.668 | 0.857 | 10:55 | 24.220 | 1.079 | reprise | 0.427 | none |
| >=1.00 | 2018-06-19 | stop | -0.779 | -1,157.50 | 0.259 | 0.387 | 0.572 | 0.797 | 10:30 | 12.310 | 0.850 | reprise | 0.399 | none |
| >=1.00 | 2016-11-07 | time | -0.710 | -692.50 | 0.041 | 0.072 | 0.149 | 0.749 | 11:45 | 22.510 | 1.065 | extension | -0.037 | none |
| [0.25,0.50) | 2019-12-02 | stop | -0.990 | -1,287.50 | 0.115 | 0.273 | 0.915 | 1.019 | 10:30 | 12.620 | 0.814 | extension | -4.426 | none |
| [0.25,0.50) | 2022-05-05 | stop | -0.776 | -6,262.50 | 0.077 | 0.163 | 0.387 | 0.854 | 11:05 | 25.420 | 0.905 | extension | -0.636 | none |
| [0.25,0.50) | 2014-07-08 | stop | -0.776 | -447.50 | 0.173 | 0.217 | 0.399 | 0.867 | 10:30 | 11.330 | 0.880 | extension | -2.783 | none |
| [0.25,0.50) | 2019-06-07 | stop | -0.694 | -1,602.50 | 0.041 | 0.115 | 0.223 | 0.736 | 10:55 | 15.930 | 0.941 | extension | -1.163 | NFP |
| [0.25,0.50) | 2010-09-28 | stop | -0.664 | -382.50 | 0.156 | 0.156 | 0.156 | 0.668 | 11:25 | 22.540 | 0.894 | reprise | 5.167 | none |
| [0.25,0.50) | 2021-01-07 | time | -0.659 | -2,332.50 | 0.147 | 0.315 | 0.535 | 0.773 | 11:45 | 25.070 | 0.936 | extension | -0.516 | none |
| [0.25,0.50) | 2013-04-10 | stop | -0.652 | -402.50 | 0.032 | 0.130 | 0.526 | 0.656 | 10:30 | 12.840 | 0.888 | extension | -1.081 | none |
| [0.25,0.50) | 2019-06-10 | time | -0.645 | -1,542.50 | 0.029 | 0.098 | 0.236 | 0.644 | 11:50 | 16.300 | 0.950 | extension | -0.621 | none |
| [0.25,0.50) | 2018-02-02 | stop | -0.642 | -897.50 | 0.347 | 0.397 | 0.680 | 0.680 | 10:00 | 13.470 | 0.915 | extension | -0.667 | NFP |
| [0.25,0.50) | 2017-01-03 | stop | -0.638 | -617.50 | 0.083 | 0.170 | 0.646 | 0.646 | 10:05 | 14.040 | 0.849 | reprise | 0.047 | none |
| [0.50,0.75) | 2020-09-03 | stop | -1.218 | -4,642.50 | 0.191 | 0.193 | 0.193 | 1.512 | 10:45 | 26.570 | 0.813 | extension | -0.243 | none |
| [0.50,0.75) | 2021-05-04 | time | -0.980 | -3,867.50 | 0.142 | 0.279 | 0.407 | 1.001 | 11:50 | 18.310 | 0.842 | extension | -0.865 | none |
| [0.50,0.75) | 2011-08-04 | stop | -0.898 | -772.50 | 0.151 | 0.151 | 0.151 | 0.964 | 11:45 | 23.380 | 1.011 | extension | -0.177 | none |
| [0.50,0.75) | 2012-04-04 | stop | -0.853 | -492.50 | 0.052 | 0.052 | 0.217 | 0.858 | 10:55 | 15.660 | 0.833 | extension | -0.157 | none |
| [0.50,0.75) | 2022-01-24 | stop | -0.848 | -5,587.50 | 0.260 | 0.260 | 0.316 | 0.962 | 11:45 | 28.850 | 0.963 | extension | -0.234 | none |
| [0.50,0.75) | 2015-08-21 | stop | -0.809 | -1,012.50 | 0.108 | 0.108 | 0.108 | 0.831 | 11:05 | 19.140 | 1.016 | reprise | 0.190 | none |
| [0.50,0.75) | 2015-08-20 | stop | -0.779 | -897.50 | 0.078 | 0.160 | 0.360 | 0.776 | 10:20 | 15.250 | 0.924 | extension | -0.047 | none |
| [0.50,0.75) | 2018-10-04 | time | -0.743 | -1,192.50 | 0.041 | 0.041 | 0.159 | 0.860 | 11:50 | 11.610 | 0.791 | extension | -1.292 | none |
| [0.50,0.75) | 2011-07-27 | stop | -0.728 | -492.50 | 0.148 | 0.163 | 0.362 | 0.799 | 10:25 | 20.230 | 0.968 | extension | -0.610 | none |
| [0.50,0.75) | 2010-11-23 | stop | -0.726 | -442.50 | 0.279 | 0.427 | 0.435 | 0.796 | 11:30 | 18.370 | 0.861 | extension | -0.036 | none |
| [0.75,1.00) | 2014-03-24 | time | -0.903 | -797.50 | 0.159 | 0.159 | 0.243 | 1.184 | 11:10 | 15.000 | 0.927 | extension | -0.650 | none |
| [0.75,1.00) | 2012-09-06 | time | -0.890 | -512.50 | 0.104 | 0.113 | 0.426 | 0.912 | 11:45 | 17.740 | 0.887 | extension | -0.831 | none |
| [0.75,1.00) | 2018-06-25 | time | -0.850 | -1,342.50 | 0.051 | 0.051 | 0.199 | 0.918 | 11:35 | 13.770 | 0.888 | extension | -0.735 | none |
| [0.75,1.00) | 2013-04-17 | stop | -0.825 | -557.50 | 0.126 | 0.237 | 0.326 | 0.814 | 10:40 | 13.960 | 0.925 | extension | -0.028 | none |
| [0.75,1.00) | 2014-09-22 | stop | -0.818 | -597.50 | 0.048 | 0.048 | 0.164 | 0.883 | 11:15 | 12.110 | 0.842 | extension | -0.427 | none |
| [0.75,1.00) | 2018-08-15 | stop | -0.804 | -1,312.50 | 0.107 | 0.218 | 0.625 | 0.849 | 10:25 | 13.310 | 0.867 | extension | -0.111 | none |
| [0.75,1.00) | 2012-11-07 | stop | -0.794 | -552.50 | 0.086 | 0.158 | 0.302 | 0.840 | 10:50 | 17.580 | 0.964 | reprise | 0.045 | none |
| [0.75,1.00) | 2010-09-01 | stop | -0.770 | -527.50 | 0.124 | 0.387 | 0.686 | 0.759 | 10:30 | 26.050 | 0.897 | extension | -0.052 | none |
| [0.75,1.00) | 2019-05-13 | time | -0.745 | -1,617.50 | 0.338 | 0.403 | 0.615 | 0.790 | 11:50 | 16.040 | 0.949 | reprise | 0.069 | none |
| [0.75,1.00) | 2012-06-25 | stop | -0.670 | -567.50 | 0.142 | 0.154 | 0.207 | 0.673 | 11:35 | 18.110 | 0.842 | extension | -0.261 | none |

Lecture synthétique : 27 des 40 queues prolongent encore le gap durant les 20 premières minutes ; 27 sortent au stop et 13 au temps. Deux seulement sont des jours macro, tous deux dans [0,25–0,50). La queue [0,50–0,75) n’est donc pas une queue d’annonces macro.

## B0a — liquidité MGC et PnL annuel

| year | median_bar_volume | pct_volume_le_1 | median_daily_volume | trades | gross_usd | cost_usd | net_usd | gross_R | cost_R | net_R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 | 7.0 | 10.87% | 652.5 | 193 | -95.00 | 868.50 | -963.50 | -0.903 | 6.573 | -7.476 |
| 2016 | 16.0 | 3.74% | 1,442.5 | 197 | -60.00 | 886.50 | -946.50 | 0.057 | 6.007 | -5.949 |
| 2017 | 16.0 | 2.37% | 1,459.0 | 217 | 211.00 | 976.50 | -765.50 | 1.874 | 8.898 | -7.024 |
| 2018 | 26.0 | 1.05% | 2,274.0 | 175 | 229.00 | 787.50 | -558.50 | 2.242 | 8.096 | -5.854 |
| 2019 | 96.0 | 0.43% | 8,079.5 | 178 | -609.00 | 801.00 | -1,410.00 | -5.306 | 6.736 | -12.042 |
| 2020 | 387.0 | 0.30% | 29,225.0 | 121 | -1,100.00 | 544.50 | -1,644.50 | -3.943 | 2.486 | -6.429 |
| 2021 | 270.0 | 0.38% | 22,777.0 | 114 | 338.00 | 513.00 | -175.00 | 1.313 | 2.422 | -1.108 |
| 2022 | 232.0 | 0.12% | 18,759.5 | 108 | 53.00 | 486.00 | -433.00 | -0.101 | 2.264 | -2.365 |
| 2023 | 322.5 | 0.22% | 25,894.0 | 30 | 269.00 | 135.00 | 134.00 | 1.373 | 0.650 | 0.724 |

La médiane par barre passe de 7 contrats en 2015 à 387 en 2020, et le volume journalier médian de 652,5 à 29 225. Pourtant les pires années nettes en R incluent 2019 (-12,04 R) et 2020 (-6,43 R) : la seule illiquidité des premières années n’explique pas monotoniquement la perte. Le constat robuste est plutôt un brut quasi nul (-3,39 R) absorbé par 44,13 R de coûts.

## B0d — roll ±2 séances contre hors roll

| roll_window | trades | gross_usd | cost_usd | net_usd | gross_R | cost_R | net_R | mean_net_R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outside_roll | 1257 | -840.00 | 5,656.50 | -6,496.50 | -4.174 | 41.525 | -45.700 | -0.036 |
| roll_pm2 | 76 | 76.00 | 342.00 | -266.00 | 0.782 | 2.605 | -1.823 | -0.024 |

Les fenêtres de roll ne représentent que 5,7 % des trades et 3,8 % de la perte nette R. Elles ont même un brut positif (+0,78 R). Le roll n’explique donc pas la perte canonique.

## B0b — volume GC agrégé, exécution MGC inchangée

Le volume est la somme, minute par minute, des rangs continus `GC.v.0` à `GC.v.9`, puis agrégé en barres de cinq minutes. Le RVOL GC est causal : volume courant divisé par la médiane de la même tranche sur les vingt séances antérieures. Les prix, confirmations, fills, coûts et PnL restent ceux de MGC.

| version | trades | gross_usd | cost_usd | net_usd | gross_R | cost_R | net_R | mean_net_R | raw_signal_events | event_overlap_count | event_overlap_vs_version | executable_overlap_count | executable_overlap_vs_version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGC_volume_canonical | 1333 | -764.00 | 5,998.50 | -6,762.50 | -3.3925 | 44.1302 | -47.5227 | -0.0357 | 1534 | 1234 | 0.8044 | 1063 | 0.7974 |
| GC_v0_v9_volume | 1149 | -700.00 | 5,170.50 | -5,870.50 | -3.8024 | 37.9062 | -41.7086 | -0.0363 | 1327 | 1234 | 0.9299 | 1063 | 0.9252 |

Le volume GC conserve 1 234 des 1 534 événements bruts MGC (80,44 %) et 1 063 des 1 333 trades exécutables (79,74 %). Inversement, 92,99 % des événements GC et 92,52 % de ses trades sont communs avec MGC. Le point décisif est l’espérance : −0,03630 R/trade avec GC contre −0,03565 avec MGC. Le remplacement de la mesure de volume ne restaure donc aucun edge.

### B0b et roll GC

| roll_pm2 | trades | gross_usd | cost_usd | net_usd | gross_R | cost_R | net_R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outside_roll | 1077 | -461.00 | 4,846.50 | -5,307.50 | -2.464 | 35.528 | -37.991 |
| roll_pm2 | 72 | -239.00 | 324.00 | -563.00 | -1.339 | 2.379 | -3.718 |

### Qualité Databento

| version | all_trades | trades_on_degraded_dates | all_net_usd | degraded_dates_net_usd | excluding_degraded_net_usd | all_net_R | degraded_dates_net_R | excluding_degraded_net_R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MGC_volume_canonical | 1333 | 6.000 | -6,762.500 | -11.000 | -6,751.500 | -47.523 | -0.245 | -47.278 |
| GC_v0_v9_volume | 1149 | 6.000 | -5,870.500 | -11.000 | -5,859.500 | -41.709 | -0.245 | -41.464 |

Treize dates GLBX.MDP3 sont marquées `degraded`, mais six trades seulement sont touchés. Leur retrait laisse B0b à −41,46 R et −5 859,50 $, sans changement de conclusion.

### B6 — random thinning de B0b

| candidate_trades | price_only_trades | candidate_sharpe | candidate_expectancy | sharpe_percentile | expectancy_percentile |
| --- | --- | --- | --- | --- | --- |
| 1149 | 1896 | -2.8468 | -5.1092 | 0.3860 | 0.2000 |

Le Sharpe de B0b est au percentile 38,6 % des échantillons de même taille et son expectancy au percentile 20,0 %. La sélection GC n’est pas supérieure à un amincissement aléatoire du même univers de prix.

## Acquisition Databento GC exécutée

- Dataset : `GLBX.MDP3`.
- Symboles : `GC.v.0` à `GC.v.9`, `stype_in=continuous`, afin de rester sous le crédit disponible tout en agrégeant les dix contrats les plus liquides.
- Schéma : `ohlcv-1m`, puis agrégation 5 minutes identique à MGC.
- Fenêtre API : début inclusif `2015-01-02T00:00:00Z`, fin exclusive `2023-03-17T00:00:00Z`.
- Coût estimé juste avant acquisition : **18,757453262806 $**, sous le garde dur de 20 $.
- Résultat : 5 137 924 lignes, 110 identifiants d’instrument, aucun doublon contrat-minute ; un seul appel `get_range`.
- SHA-256 brut : `262ebb63cd9b9c519d9e9c1112ab304d085c5c1370d969f7be176dce12a3bc3b`.
- SHA-256 agrégat 5 minutes : `bc94abc8950d926a0099d79eb7ee80e0c3936e96b1832e0a97d9de777f042a6e`.

## Conclusions reformulées

- **Signal A — (c2) edge nul avec queue gauche concentrée.** Le brut normalisé est négatif (-21,49 R), aucun des 35 contrastes estimables ne survit à Holm, A1 est nul, mais les dix pires R absorbent environ 40 % de la perte nette dans les deux premiers buckets et plus de 200 % dans [0,75–1,00). Il existe une queue, pas un état exploitable démontré.
- **Signal B — (c1) edge nul et pertes diffuses.** B0b invalide l’explication (b) : malgré environ 20 % d’événements différents, GC donne pratiquement la même espérance par trade que MGC. Le brut reste proche de zéro/négatif, les coûts dominent, les pertes subsistent hors roll et après exclusion des dates dégradées, et le random thinning n’indique aucune sélection particulière.

Aucune règle de trading ni performance filtrée n’est produite.
