# Diagnostic IS — gap reversion NQ et MGC volume climax

## Résumé (10 lignes maximum)

1. Périmètre strict: A/NQ 2010-06-07–2024-02-22; micros seulement en robustesse; B/MGC 2015-01-02–2023-03-16.
2. Les seuils A sont exactement emboîtés; toute l'analyse emploie quatre buckets exclusifs.
3. NQ perd surtout dans `[0.25,0.50)` (-54,7 k$) et `[0.50,0.75)` (-48,4 k$); les buckets supérieurs compensent partiellement.
4. Dans `[0.50,0.75)`, les 10 pires trades expliquent 74,1 % de la perte nette, avec une forte concentration en 2020-2022.
5. Les coûts relatifs NQ diminuent fortement avec le niveau de prix; la dégradation vient du brut normalisé, pas des coûts.
6. A4 et A5 sont descriptivement forts, mais changent de signe selon la sous-période; aucun contraste ne survit à Holm.
7. B perd -6,76 k$ net: -0,76 k$ brut et 6,00 k$ de coûts; le brut se détériore surtout en 2019-2020.
8. La mesure B0c est déjà la mesure canonique; le volume MGC augmente très fortement au cours de l'IS et B0b/GC manque.
9. Sur 34 tests disponibles, aucun rejet à 5 % après Holm; plusieurs buckets sont des proxies temporels.
10. Conclusion A: (c) pertes diffuses/régime; conclusion B: (b) problème de mesure probable, non clos sans volume GC.

## Garde-fous et tests multiples

Nombre total de contrastes: **34**. Rejets Holm 5 %: **0**. Bootstrap: 2000 tirages, blocs de 5 dates. Aucun résultat post-IS n'est calculé.

Variables bloquées faute de données: A1 macro, B0b volume GC et B1 annonces. A2 utilise désormais les fichiers officiels Cboe VIX/VIX3M. Aucune règle de trading ni performance filtrée n'est produite.

## PnL brut, coûts et net

| signal | asset | configuration | trades | gross_pnl | costs | net_pnl | net_per_trade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | NQ | >=1.00 | 282 | 4663.21 | 3525.00 | 1138.21 | 4.04 |
| A | NQ | [0.25,0.50) | 914 | -43241.79 | 11425.00 | -54666.79 | -59.81 |
| A | NQ | [0.50,0.75) | 498 | -42128.21 | 6225.00 | -48353.21 | -97.09 |
| A | NQ | [0.75,1.00) | 260 | 21728.57 | 3250.00 | 18478.57 | 71.07 |
| B | MGC | confirmed_fade_m1.5_rvol2.0 | 1333 | -764.00 | 5998.50 | -6762.50 | -5.07 |

Détails annuels et trimestriels: `step1_tables/pnl_by_year.csv` et `step1_tables/pnl_by_quarter.csv`. Les normalisations annuelles sont dans `normalized_pnl_yearly_signal_a.csv` et `normalized_pnl_yearly_signal_b.csv`.

## Concentration prioritaire NQ `[0.50,0.75)`

La nouvelle fenêtre contient 498 trades et -48353.21 $ net. Les dix pires trades perdent 35835.00 $, soit 74.1% de la perte nette. L'écart avec l'estimation ~400/-35,9 k$ provient des 98 trades ajoutés entre l'ancienne coupure NQ et février 2024.

| signal | asset | configuration | rank | start_date | trough_date | end_date | duration_calendar_days | depth | trades | recovered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | NQ | [0.50,0.75) | 1 | 2010-06-11 | 2023-02-02 | 2024-02-07 | 4989 | -56095.71 | 498 | False |
| B | MGC | confirmed_fade_m1.5_rvol2.0 | 1 | 2015-02-04 | 2023-01-18 | 2023-03-16 | 2962 | -7024.50 | 1331 | False |
| B | MGC | confirmed_fade_m1.5_rvol2.0 | 2 | 2015-02-03 | 2015-02-03 | 2015-02-04 | 1 | -18.00 | 4 | True |

## Classement descriptif des sources de pertes

| signal | variable | worst_bucket | bucket_net_pnl | share_of_signal_net_loss |
| --- | --- | --- | --- | --- |
| A | A2_term_structure | contango | -92542.500 | 1.110 |
| A | A2_vix_tercile | high | -67062.857 | 0.804 |
| A | A3_ret20 | against_trend | -65253.214 | 0.782 |
| A | A4_recovery_tercile | low | -62587.500 | 0.750 |
| A | A3_sma50 | against_trend | -59221.786 | 0.710 |
| A | A5_gap_bucket | [0.25,0.50) | -54666.786 | 0.655 |
| A | A6_direction | gap_up | -49852.500 | 0.598 |
| A | A5_gap_atr_tercile | low | -34986.786 | 0.419 |
| B | roll_window | outside_roll | -6496.500 | 0.961 |
| B | B3_sma100 | against_trend | -4208.000 | 0.622 |
| B | B3_ret50 | against_trend | -3758.500 | 0.556 |
| B | session_bucket | 09:30-11:00 | -3467.500 | 0.513 |
| B | B5_volume_tercile | mid | -3441.500 | 0.509 |
| B | B5_move_tercile | mid | -2853.500 | 0.422 |
| B | B4_gold_rv20_tercile | high | -2737.500 | 0.405 |

Une part supérieure à 100 % signifie que d'autres buckets compensent une partie de la perte. Ce classement est descriptif: il ne transforme aucun bucket en filtre.

## Vérification de mesure B

### B0a — liquidité MGC

| year | bars | median_bar_volume | pct_zero_volume | pct_volume_le_1 | median_daily_volume |
| --- | --- | --- | --- | --- | --- |
| 2015 | 14091 | 7.0000 | 0.0000 | 0.1087 | 652.5000 |
| 2016 | 15374 | 16.0000 | 0.0000 | 0.0374 | 1442.5000 |
| 2017 | 15406 | 16.0000 | 0.0000 | 0.0237 | 1459.0000 |
| 2018 | 15502 | 26.0000 | 0.0000 | 0.0105 | 2274.0000 |
| 2019 | 15649 | 96.0000 | 0.0000 | 0.0043 | 8079.5000 |
| 2020 | 15687 | 387.0000 | 0.0000 | 0.0030 | 29225.0000 |
| 2021 | 15771 | 270.0000 | 0.0000 | 0.0038 | 22777.0000 |
| 2022 | 15875 | 232.0000 | 0.0000 | 0.0012 | 18759.5000 |
| 2023 | 3232 | 322.5000 | 0.0000 | 0.0022 | 25894.0000 |

Le volume médian journalier passe de 652,5 contrats en 2015 à 18 759,5 en 2022. Les barres nulles sont absentes, tandis que la part de barres à volume ≤1 chute de 10,9 % à 0,12 %. La population de mesure change donc fortement.

### B0c et B0d

B0c est identique à l'original: RVOL = volume de la barre / médiane de la même tranche sur les 20 séances précédentes, sans lookahead.

| roll_window | trades | gross_pnl | costs | net_pnl |
| --- | --- | --- | --- | --- |
| outside_roll | 1257 | -840.000 | 5656.500 | -6496.500 |
| roll_pm2 | 76 | 76.000 | 342.000 | -266.000 |

B0b n'est pas exécuté: aucun volume GC n'est présent. Une requête Databento par contrat ou `GC.v.0` est nécessaire.

### B6 — random thinning IS

| version | candidate_trades | price_only_trades | candidate_sharpe | candidate_expectancy | sharpe_percentile | expectancy_percentile | seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_equals_B0c | 1333 | 1896 | -2.962 | -5.073 | 0.234 | 0.154 | 20261008 |

## Classement des variables disponibles

| signal | asset | variable | mean_pnl_spread | worst_bucket_net_pnl |
| --- | --- | --- | --- | --- |
| A | NQ | A5_gap_bucket | 168.166 | -54666.786 |
| A | NQ | A2_term_structure | 111.118 | -92542.500 |
| A | NQ | A4_recovery_tercile | 104.203 | -62587.500 |
| A | NQ | A2_vix_tercile | 93.641 | -67062.857 |
| A | NQ | A3_ret20 | 52.942 | -65253.214 |
| A | NQ | A3_sma50 | 40.852 | -59221.786 |
| A | NQ | A5_gap_atr_tercile | 27.035 | -34986.786 |
| A | NQ | A6_direction | 4.634 | -49852.500 |
| B | MGC | B5_volume_tercile | 4.384 | -3441.500 |
| B | MGC | B5_move_tercile | 2.284 | -2853.500 |
| B | MGC | B3_sma100 | 1.740 | -4208.000 |
| B | MGC | B4_gold_rv20_tercile | 1.670 | -2737.500 |
| B | MGC | roll_window | 1.668 | -6496.500 |
| B | MGC | session_bucket | 1.535 | -3467.500 |
| B | MGC | B3_ret50 | 0.418 | -3758.500 |

## Tests les plus proches du seuil nominal

| signal | variable | bucket_a | bucket_b | cluster_t | p_value | holm_p | bootstrap95_low | bootstrap95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | B5_volume_tercile | low | mid | 2.2632 | 0.0236 | 0.8032 | 0.5851 | 8.2215 |
| A | A5_gap_bucket | [0.50,0.75) | [0.75,1.00) | -2.1858 | 0.0288 | 0.9514 | -337.3866 | 4.4135 |
| A | A4_recovery_tercile | high | low | 2.0941 | 0.0363 | 1.0000 | 10.2095 | 193.4140 |
| A | A5_gap_bucket | [0.25,0.50) | [0.75,1.00) | -1.9387 | 0.0525 | 1.0000 | -287.0823 | 8.3615 |
| A | A2_vix_tercile | high | low | -1.6380 | 0.1014 | 1.0000 | -207.7380 | 16.6556 |
| B | B5_volume_tercile | high | mid | 1.5770 | 0.1148 | 1.0000 | -1.9121 | 7.8817 |
| A | A2_vix_tercile | high | mid | -1.4471 | 0.1479 | 1.0000 | -218.8987 | 35.6611 |
| A | A3_ret20 | against_trend | with_trend | -1.2323 | 0.2178 | 1.0000 | -131.5833 | 25.7666 |

Aucun de ces contrastes ne reste significatif après Holm.

## Variables suspectes d'être des proxies temporels

| signal | asset | variable | bucket | trades | top_3_year_share | proxy_flag_gt_60pct |
| --- | --- | --- | --- | --- | --- | --- |
| A | NQ | A2_term_structure | backwardation | 153 | 0.647 | True |
| B | MGC | B4_gold_rv20_tercile | high | 445 | 0.629 | True |
| B | MGC | B4_gold_rv20_tercile | low | 444 | 0.822 | True |

## Interactions autorisées — deux variables les plus explicatives

| bucket_a | bucket_b | trades | gross_pnl | costs | net_pnl | avg_net | signal | asset | variable_a | variable_b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >=1.00 | backwardation | 20 | -3850.000 | 250.000 | -4100.000 | -205.000 | A | NQ | A5_gap_bucket | A2_term_structure |
| >=1.00 | contango | 262 | 8513.214 | 3275.000 | 5238.214 | 19.993 | A | NQ | A5_gap_bucket | A2_term_structure |
| [0.25,0.50) | backwardation | 69 | -3958.214 | 862.500 | -4820.714 | -69.865 | A | NQ | A5_gap_bucket | A2_term_structure |
| [0.25,0.50) | contango | 845 | -39283.571 | 10562.500 | -49846.071 | -58.989 | A | NQ | A5_gap_bucket | A2_term_structure |
| [0.50,0.75) | backwardation | 43 | 4275.000 | 537.500 | 3737.500 | 86.919 | A | NQ | A5_gap_bucket | A2_term_structure |
| [0.50,0.75) | contango | 455 | -46403.214 | 5687.500 | -52090.714 | -114.485 | A | NQ | A5_gap_bucket | A2_term_structure |
| [0.75,1.00) | backwardation | 21 | 14585.000 | 262.500 | 14322.500 | 682.024 | A | NQ | A5_gap_bucket | A2_term_structure |
| [0.75,1.00) | contango | 239 | 7143.571 | 2987.500 | 4156.071 | 17.389 | A | NQ | A5_gap_bucket | A2_term_structure |
| high | high | 242 | -320.000 | 1089.000 | -1409.000 | -5.822 | B | MGC | B5_volume_tercile | B5_move_tercile |
| high | low | 92 | 142.000 | 414.000 | -272.000 | -2.957 | B | MGC | B5_volume_tercile | B5_move_tercile |
| high | mid | 109 | 388.000 | 490.500 | -102.500 | -0.940 | B | MGC | B5_volume_tercile | B5_move_tercile |
| low | high | 86 | 427.000 | 387.000 | 40.000 | 0.465 | B | MGC | B5_volume_tercile | B5_move_tercile |
| low | low | 190 | 197.000 | 855.000 | -658.000 | -3.463 | B | MGC | B5_volume_tercile | B5_move_tercile |
| low | mid | 167 | -130.000 | 751.500 | -881.500 | -5.278 | B | MGC | B5_volume_tercile | B5_move_tercile |
| mid | high | 115 | -143.000 | 517.500 | -660.500 | -5.743 | B | MGC | B5_volume_tercile | B5_move_tercile |
| mid | low | 161 | -187.000 | 724.500 | -911.500 | -5.661 | B | MGC | B5_volume_tercile | B5_move_tercile |
| mid | mid | 167 | -1118.000 | 751.500 | -1869.500 | -11.195 | B | MGC | B5_volume_tercile | B5_move_tercile |

## Tableaux par variable

### A — A2_term_structure

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backwardation | 153 | 11051.7857 | 1912.5000 | 9139.2857 | 59.7339 | 11.9468 | 0.5229 | 1.1299 | 0.5050 | -0.1096 |
| contango | 1801 | -70030.0000 | 22512.5000 | -92542.5000 | -51.3840 | -10.2768 | 0.4781 | 0.8276 | -2.4626 | 1.1096 |

### A — A2_vix_tercile

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 649 | -58950.3571 | 8112.5000 | -67062.8571 | -103.3326 | -20.6665 | 0.4515 | 0.7964 | -1.9308 | 0.8041 |
| low | 652 | 1831.0714 | 8150.0000 | -6318.9286 | -9.6916 | -1.9383 | 0.5061 | 0.9445 | -0.4791 | 0.0758 |
| mid | 653 | -1858.9286 | 8162.5000 | -10021.4286 | -15.3468 | -3.0694 | 0.4870 | 0.9389 | -0.5301 | 0.1202 |

### A — A3_ret20

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| against_trend | 927 | -53665.7143 | 11587.5000 | -65253.2143 | -70.3918 | -14.0784 | 0.4854 | 0.7918 | -2.2443 | 0.7824 |
| with_trend | 1017 | -5033.9286 | 12712.5000 | -17746.4286 | -17.4498 | -3.4900 | 0.4798 | 0.9394 | -0.5937 | 0.2128 |

### A — A3_sma50

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| against_trend | 913 | -47809.2857 | 11412.5000 | -59221.7857 | -64.8650 | -12.9730 | 0.4896 | 0.8096 | -2.0242 | 0.7101 |
| with_trend | 1013 | -11663.2143 | 12662.5000 | -24325.7143 | -24.0135 | -4.8027 | 0.4748 | 0.9173 | -0.8188 | 0.2917 |

### A — A4_recovery_tercile

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 652 | 13406.7857 | 8150.0000 | 5256.7857 | 8.0626 | 1.6125 | 0.5506 | 1.0398 | 0.2990 | -0.0630 |
| low | 651 | -54450.0000 | 8137.5000 | -62587.5000 | -96.1406 | -19.2281 | 0.4439 | 0.7455 | -2.2964 | 0.7504 |
| mid | 651 | -17935.0000 | 8137.5000 | -26072.5000 | -40.0499 | -8.0100 | 0.4501 | 0.8863 | -0.9956 | 0.3126 |

### A — A5_gap_atr_tercile

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 649 | -22726.0714 | 8112.5000 | -30838.5714 | -47.5171 | -9.5034 | 0.4607 | 0.8848 | -0.9883 | 0.3698 |
| low | 649 | -26874.2857 | 8112.5000 | -34986.7857 | -53.9088 | -10.7818 | 0.5069 | 0.7787 | -2.0186 | 0.4195 |
| mid | 648 | -9314.2857 | 8100.0000 | -17414.2857 | -26.8739 | -5.3748 | 0.4784 | 0.9036 | -0.8061 | 0.2088 |

### A — A5_gap_bucket

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >=1.00 | 282 | 4663.2143 | 3525.0000 | 1138.2143 | 4.0362 | 0.8072 | 0.5071 | 1.0111 | 0.0574 | -0.0136 |
| [0.25,0.50) | 914 | -43241.7857 | 11425.0000 | -54666.7857 | -59.8105 | -11.9621 | 0.4945 | 0.7858 | -2.2853 | 0.6555 |
| [0.50,0.75) | 498 | -42128.2143 | 6225.0000 | -48353.2143 | -97.0948 | -19.4190 | 0.4518 | 0.7334 | -2.1446 | 0.5798 |
| [0.75,1.00) | 260 | 21728.5714 | 3250.0000 | 18478.5714 | 71.0714 | 14.2143 | 0.4654 | 1.2709 | 1.1397 | -0.2216 |

### A — A6_direction

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap_down | 838 | -23075.7143 | 10475.0000 | -33550.7143 | -40.0367 | -8.0073 | 0.5167 | 0.8785 | -1.1335 | 0.4023 |
| gap_up | 1116 | -35902.5000 | 13950.0000 | -49852.5000 | -44.6707 | -8.9341 | 0.4552 | 0.8495 | -1.6957 | 0.5977 |

### B — B3_ret50

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| against_trend | 715 | -541.0000 | 3217.5000 | -3758.5000 | -5.2566 | -5.2566 | 0.4490 | 0.5874 | -4.6859 | 0.5558 |
| with_trend | 593 | -201.0000 | 2668.5000 | -2869.5000 | -4.8390 | -4.8390 | 0.4047 | 0.6034 | -3.5058 | 0.4243 |

### B — B3_sma100

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| against_trend | 706 | -1031.0000 | 3177.0000 | -4208.0000 | -5.9603 | -5.9603 | 0.4306 | 0.5502 | -5.2461 | 0.6223 |
| with_trend | 561 | 157.0000 | 2524.5000 | -2367.5000 | -4.2201 | -4.2201 | 0.4189 | 0.6491 | -2.9149 | 0.3501 |

### B — B4_gold_rv20_tercile

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 445 | -735.0000 | 2002.5000 | -2737.5000 | -6.1517 | -6.1517 | 0.4539 | 0.6238 | -3.0186 | 0.4048 |
| low | 444 | -37.0000 | 1998.0000 | -2035.0000 | -4.5833 | -4.5833 | 0.3964 | 0.5315 | -4.6475 | 0.3009 |
| mid | 444 | 8.0000 | 1998.0000 | -1990.0000 | -4.4820 | -4.4820 | 0.4302 | 0.6034 | -3.5648 | 0.2943 |

### B — B5_move_tercile

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 443 | -36.0000 | 1993.5000 | -2029.5000 | -4.5813 | -4.5813 | 0.4628 | 0.6273 | -3.0621 | 0.3001 |
| low | 443 | 152.0000 | 1993.5000 | -1841.5000 | -4.1569 | -4.1569 | 0.4379 | 0.6319 | -3.3647 | 0.2723 |
| mid | 443 | -860.0000 | 1993.5000 | -2853.5000 | -6.4413 | -6.4413 | 0.3837 | 0.5361 | -3.7265 | 0.4220 |

### B — B5_volume_tercile

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 443 | 210.0000 | 1993.5000 | -1783.5000 | -4.0260 | -4.0260 | 0.4379 | 0.6671 | -2.3973 | 0.2637 |
| low | 443 | 494.0000 | 1993.5000 | -1499.5000 | -3.3849 | -3.3849 | 0.4244 | 0.6872 | -2.7630 | 0.2217 |
| mid | 443 | -1448.0000 | 1993.5000 | -3441.5000 | -7.7686 | -7.7686 | 0.4221 | 0.4664 | -5.0014 | 0.5089 |

### B — roll_window

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outside_roll | 1257 | -840.0000 | 5656.5000 | -6496.5000 | -5.1683 | -5.1683 | 0.4256 | 0.5878 | -5.7595 | 0.9607 |
| roll_pm2 | 76 | 76.0000 | 342.0000 | -266.0000 | -3.5000 | -3.5000 | 0.4474 | 0.6970 | -1.1578 | 0.0393 |

### B — session_bucket

| bucket | trades | gross_pnl | costs | net_pnl | avg_net_usd | avg_net_ticks | hit_rate | profit_factor | t_stat | share_of_total_net_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 08:20-09:30 | 96 | -152.0000 | 432.0000 | -584.0000 | -6.0833 | -6.0833 | 0.4167 | 0.6266 | -1.4777 | 0.0864 |
| 09:30-11:00 | 641 | -583.0000 | 2884.5000 | -3467.5000 | -5.4095 | -5.4095 | 0.4415 | 0.6260 | -3.9914 | 0.5128 |
| 11:00-13:25 | 596 | -29.0000 | 2682.0000 | -2711.0000 | -4.5487 | -4.5487 | 0.4128 | 0.5328 | -4.2039 | 0.4009 |

## Matrices année × bucket

### A — A2_term_structure

| year | n_backwardation | n_contango | pnl_backwardation | pnl_contango |
| --- | --- | --- | --- | --- |
| 2010 | 1 | 82 | 117.50 | -1922.86 |
| 2011 | 34 | 109 | 587.50 | -3538.57 |
| 2012 | 0 | 135 | 0.00 | -5888.57 |
| 2013 | 3 | 133 | -1047.50 | 444.64 |
| 2014 | 10 | 112 | -1356.07 | -6019.29 |
| 2015 | 14 | 123 | 2315.00 | -6170.36 |
| 2016 | 11 | 128 | -1047.50 | -868.57 |
| 2017 | 1 | 146 | -357.50 | 1904.64 |
| 2018 | 28 | 119 | 1805.00 | -8829.29 |
| 2019 | 9 | 160 | 297.86 | 1177.86 |
| 2020 | 37 | 114 | -2277.50 | -8440.71 |
| 2021 | 0 | 147 | 0.00 | -18602.86 |
| 2022 | 5 | 131 | 10102.50 | -38701.79 |
| 2023 | 0 | 139 | 0.00 | 4730.71 |
| 2024 | 0 | 23 | 0.00 | -1817.50 |

### A — A2_vix_tercile

| year | n_high | n_low | n_mid | pnl_high | pnl_low | pnl_mid |
| --- | --- | --- | --- | --- | --- | --- |
| 2010 | 64 | 0 | 19 | -1782.50 | 0.00 | -22.86 |
| 2011 | 81 | 0 | 62 | -1553.21 | 0.00 | -1397.86 |
| 2012 | 33 | 7 | 95 | -2410.00 | -91.79 | -3386.79 |
| 2013 | 2 | 93 | 41 | -160.00 | 154.64 | -597.50 |
| 2014 | 8 | 79 | 35 | -296.79 | -2558.93 | -4519.64 |
| 2015 | 28 | 52 | 57 | 4615.00 | -2336.79 | -6133.57 |
| 2016 | 30 | 65 | 44 | -2430.00 | -277.14 | 791.07 |
| 2017 | 0 | 142 | 5 | 0.00 | 2424.64 | -877.50 |
| 2018 | 39 | 63 | 45 | 1067.50 | -3794.29 | -4297.50 |
| 2019 | 14 | 72 | 83 | 3680.36 | -2536.79 | 332.14 |
| 2020 | 122 | 16 | 13 | -10636.07 | 1305.00 | -1387.14 |
| 2021 | 70 | 0 | 77 | -22202.86 | 0.00 | 3600.00 |
| 2022 | 129 | 0 | 7 | -26456.79 | 0.00 | -2142.50 |
| 2023 | 29 | 45 | 65 | -8497.50 | 2307.50 | 10920.71 |
| 2024 | 0 | 18 | 5 | 0.00 | -915.00 | -902.50 |

### A — A3_ret20

| year | n_against_trend | n_with_trend | pnl_against_trend | pnl_with_trend |
| --- | --- | --- | --- | --- |
| 2010 | 34 | 39 | -487.86 | -913.93 |
| 2011 | 70 | 73 | -1363.57 | -1587.50 |
| 2012 | 61 | 74 | -749.29 | -5139.29 |
| 2013 | 63 | 73 | -233.57 | -369.29 |
| 2014 | 63 | 59 | -7934.64 | 559.29 |
| 2015 | 72 | 65 | -3557.86 | -297.50 |
| 2016 | 74 | 65 | -665.71 | -1250.36 |
| 2017 | 57 | 90 | 748.21 | 798.93 |
| 2018 | 72 | 75 | -5480.00 | -1544.29 |
| 2019 | 71 | 98 | -4582.50 | 6058.21 |
| 2020 | 55 | 96 | -11995.00 | 1276.79 |
| 2021 | 70 | 77 | -21578.21 | 2975.36 |
| 2022 | 74 | 62 | -3209.29 | -25390.00 |
| 2023 | 78 | 61 | -11381.43 | 16112.14 |
| 2024 | 13 | 10 | 7217.50 | -9035.00 |

### A — A3_sma50

| year | n_against_trend | n_with_trend | pnl_against_trend | pnl_with_trend |
| --- | --- | --- | --- | --- |
| 2010 | 21 | 34 | -773.21 | -1176.43 |
| 2011 | 62 | 81 | -299.29 | -2651.79 |
| 2012 | 67 | 68 | -2088.57 | -3800.00 |
| 2013 | 55 | 81 | 576.43 | -1179.29 |
| 2014 | 59 | 63 | -6346.43 | -1028.93 |
| 2015 | 80 | 57 | -3093.21 | -762.14 |
| 2016 | 74 | 65 | -995.71 | -920.36 |
| 2017 | 58 | 89 | 2770.71 | -1223.57 |
| 2018 | 68 | 79 | -621.79 | -6402.50 |
| 2019 | 79 | 90 | -5017.50 | 6493.21 |
| 2020 | 59 | 92 | -20070.00 | 9351.79 |
| 2021 | 66 | 81 | -16609.29 | -1993.57 |
| 2022 | 74 | 62 | -11865.00 | -16734.29 |
| 2023 | 80 | 59 | -3291.43 | 8022.14 |
| 2024 | 11 | 12 | 8502.50 | -10320.00 |

### A — A4_recovery_tercile

| year | n_high | n_low | n_mid | pnl_high | pnl_low | pnl_mid |
| --- | --- | --- | --- | --- | --- | --- |
| 2010 | 38 | 25 | 20 | -587.86 | 32.50 | -1250.00 |
| 2011 | 50 | 40 | 53 | -743.57 | 165.00 | -2372.50 |
| 2012 | 43 | 37 | 55 | 686.43 | -2627.50 | -3947.50 |
| 2013 | 52 | 40 | 44 | -2127.86 | -1190.00 | 2715.00 |
| 2014 | 50 | 32 | 40 | -1590.36 | -4445.00 | -1340.00 |
| 2015 | 44 | 55 | 38 | -1682.86 | -4182.50 | 2010.00 |
| 2016 | 45 | 55 | 39 | 1613.93 | 2007.50 | -5537.50 |
| 2017 | 40 | 55 | 52 | 1884.64 | -1127.50 | 790.00 |
| 2018 | 45 | 50 | 52 | 255.71 | -13890.00 | 6610.00 |
| 2019 | 50 | 70 | 49 | 5578.21 | -535.00 | -3567.50 |
| 2020 | 43 | 45 | 63 | 10356.79 | -4352.50 | -16722.50 |
| 2021 | 57 | 47 | 43 | -4142.86 | -6317.50 | -8142.50 |
| 2022 | 43 | 52 | 41 | -851.79 | -25385.00 | -2362.50 |
| 2023 | 50 | 40 | 49 | -651.79 | 675.00 | 4707.50 |
| 2024 | 2 | 8 | 13 | -2740.00 | -1415.00 | 2337.50 |

### A — A5_gap_atr_tercile

| year | n_high | n_low | n_mid | pnl_high | pnl_low | pnl_mid |
| --- | --- | --- | --- | --- | --- | --- |
| 2010 | 25 | 28 | 22 | -1932.50 | -489.64 | 780.36 |
| 2011 | 46 | 45 | 52 | -3540.00 | 695.71 | -106.79 |
| 2012 | 40 | 41 | 54 | -265.00 | -2493.93 | -3129.64 |
| 2013 | 46 | 42 | 48 | 140.00 | -727.86 | -15.00 |
| 2014 | 37 | 43 | 42 | -3245.36 | -1968.21 | -2161.79 |
| 2015 | 47 | 46 | 44 | 470.36 | -310.36 | -4015.36 |
| 2016 | 41 | 57 | 41 | -4247.50 | 598.93 | 1732.50 |
| 2017 | 43 | 51 | 53 | 822.50 | 1711.07 | -986.43 |
| 2018 | 51 | 55 | 41 | 4967.86 | -3814.64 | -8177.50 |
| 2019 | 53 | 61 | 55 | 3462.50 | -1744.29 | -242.50 |
| 2020 | 65 | 40 | 46 | -4837.14 | -4681.07 | -1200.00 |
| 2021 | 51 | 49 | 47 | -15192.50 | -8908.57 | 5498.21 |
| 2022 | 44 | 43 | 49 | -21126.79 | -3922.50 | -3550.00 |
| 2023 | 49 | 44 | 46 | 12887.50 | -5891.43 | -2265.36 |
| 2024 | 11 | 4 | 8 | 797.50 | -3040.00 | 425.00 |

### A — A5_gap_bucket

| year | n_>=1.00 | n_[0.25,0.50) | n_[0.50,0.75) | n_[0.75,1.00) | pnl_>=1.00 | pnl_[0.25,0.50) | pnl_[0.50,0.75) | pnl_[0.75,1.00) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2010 | 13 | 41 | 22 | 7 | 727.50 | -921.07 | -989.29 | -622.50 |
| 2011 | 19 | 73 | 33 | 18 | -1862.50 | 308.93 | -1018.93 | -378.57 |
| 2012 | 20 | 63 | 36 | 16 | 510.00 | -2373.57 | -1310.00 | -2715.00 |
| 2013 | 20 | 59 | 38 | 19 | 1545.00 | -1215.36 | -345.00 | -587.50 |
| 2014 | 12 | 58 | 27 | 25 | 85.00 | -3038.57 | -3468.57 | -953.21 |
| 2015 | 16 | 66 | 29 | 26 | 4325.00 | -4203.93 | -2694.29 | -1282.14 |
| 2016 | 19 | 71 | 35 | 14 | -2737.50 | 2633.93 | -437.50 | -1375.00 |
| 2017 | 16 | 64 | 44 | 23 | -1245.00 | 417.14 | 1397.50 | 977.50 |
| 2018 | 21 | 79 | 28 | 19 | 1842.50 | -509.29 | -5940.00 | -2417.50 |
| 2019 | 30 | 74 | 44 | 21 | 6760.00 | -2996.79 | 815.00 | -3102.50 |
| 2020 | 31 | 57 | 42 | 21 | -21382.50 | -273.57 | -5309.64 | 16247.50 |
| 2021 | 19 | 66 | 43 | 19 | 2332.50 | -9500.36 | -12607.50 | 1172.50 |
| 2022 | 22 | 61 | 38 | 15 | 14833.21 | -20902.50 | -19862.50 | -2667.50 |
| 2023 | 18 | 71 | 35 | 15 | -4060.00 | -10994.29 | 5832.50 | 13952.50 |
| 2024 | 6 | 11 | 4 | 2 | -535.00 | -1097.50 | -2415.00 | 2230.00 |

### A — A6_direction

| year | n_gap_down | n_gap_up | pnl_gap_down | pnl_gap_up |
| --- | --- | --- | --- | --- |
| 2010 | 32 | 51 | 575.36 | -2380.71 |
| 2011 | 70 | 73 | -786.79 | -2164.29 |
| 2012 | 58 | 77 | -2851.07 | -3037.50 |
| 2013 | 50 | 86 | 633.93 | -1236.79 |
| 2014 | 52 | 70 | -2498.93 | -4876.43 |
| 2015 | 65 | 72 | -2415.36 | -1440.00 |
| 2016 | 68 | 71 | -2114.64 | 198.57 |
| 2017 | 52 | 95 | 3330.71 | -1783.57 |
| 2018 | 61 | 86 | -7519.29 | 495.00 |
| 2019 | 71 | 98 | 4162.14 | -2686.43 |
| 2020 | 52 | 99 | -1902.50 | -8815.71 |
| 2021 | 57 | 90 | -4789.29 | -13813.57 |
| 2022 | 71 | 65 | -21687.50 | -6911.79 |
| 2023 | 68 | 71 | -4190.00 | 8920.71 |
| 2024 | 11 | 12 | 8502.50 | -10320.00 |

### B — B3_ret50

| year | n_against_trend | n_with_trend | pnl_against_trend | pnl_with_trend |
| --- | --- | --- | --- | --- |
| 2015 | 86 | 82 | -455.00 | -374.00 |
| 2016 | 103 | 94 | -268.50 | -678.00 |
| 2017 | 123 | 94 | -336.50 | -429.00 |
| 2018 | 100 | 75 | -143.00 | -415.50 |
| 2019 | 94 | 84 | -1157.00 | -253.00 |
| 2020 | 71 | 50 | -999.50 | -645.00 |
| 2021 | 66 | 48 | -556.00 | 381.00 |
| 2022 | 54 | 54 | -60.00 | -373.00 |
| 2023 | 18 | 12 | 217.00 | -83.00 |

### B — B3_sma100

| year | n_against_trend | n_with_trend | pnl_against_trend | pnl_with_trend |
| --- | --- | --- | --- | --- |
| 2015 | 74 | 53 | -577.00 | -199.50 |
| 2016 | 109 | 88 | -379.50 | -567.00 |
| 2017 | 126 | 91 | -448.00 | -317.50 |
| 2018 | 95 | 80 | -202.50 | -356.00 |
| 2019 | 95 | 83 | -1136.50 | -273.50 |
| 2020 | 77 | 44 | -1058.50 | -586.00 |
| 2021 | 56 | 58 | -530.00 | 355.00 |
| 2022 | 56 | 52 | -93.00 | -340.00 |
| 2023 | 18 | 12 | 217.00 | -83.00 |

### B — B4_gold_rv20_tercile

| year | n_high | n_low | n_mid | pnl_high | pnl_low | pnl_mid |
| --- | --- | --- | --- | --- | --- | --- |
| 2015 | 75 | 7 | 111 | -189.50 | -37.50 | -736.50 |
| 2016 | 128 | 30 | 39 | -498.00 | -140.00 | -308.50 |
| 2017 | 9 | 159 | 49 | -198.50 | -492.50 | -74.50 |
| 2018 | 7 | 118 | 50 | 7.50 | -446.00 | -120.00 |
| 2019 | 21 | 88 | 69 | -131.50 | -652.00 | -626.50 |
| 2020 | 77 | 31 | 13 | -1404.50 | -448.50 | 208.50 |
| 2021 | 69 | 2 | 43 | 44.50 | -4.00 | -215.50 |
| 2022 | 52 | 4 | 52 | -273.00 | 10.00 | -170.00 |
| 2023 | 7 | 5 | 18 | -94.50 | 175.50 | 53.00 |

### B — B5_move_tercile

| year | n_high | n_low | n_mid | pnl_high | pnl_low | pnl_mid |
| --- | --- | --- | --- | --- | --- | --- |
| 2015 | 65 | 61 | 63 | -275.50 | -170.50 | -479.50 |
| 2016 | 70 | 62 | 65 | -168.00 | -301.00 | -477.50 |
| 2017 | 83 | 72 | 62 | -263.50 | -327.00 | -175.00 |
| 2018 | 65 | 54 | 56 | -0.50 | -364.00 | -194.00 |
| 2019 | 54 | 53 | 71 | -435.00 | -268.50 | -706.50 |
| 2020 | 35 | 44 | 42 | -403.50 | -619.00 | -622.00 |
| 2021 | 37 | 39 | 38 | -114.50 | 87.50 | -148.00 |
| 2022 | 24 | 45 | 39 | -324.00 | 131.50 | -240.50 |
| 2023 | 10 | 13 | 7 | -45.00 | -10.50 | 189.50 |

### B — B5_volume_tercile

| year | n_high | n_low | n_mid | pnl_high | pnl_low | pnl_mid |
| --- | --- | --- | --- | --- | --- | --- |
| 2015 | 95 | 39 | 55 | -684.50 | 26.50 | -267.50 |
| 2016 | 68 | 55 | 74 | -229.00 | -102.50 | -615.00 |
| 2017 | 92 | 66 | 59 | -178.00 | -189.00 | -398.50 |
| 2018 | 64 | 55 | 56 | -71.00 | -263.50 | -224.00 |
| 2019 | 50 | 67 | 61 | -406.00 | -529.50 | -474.50 |
| 2020 | 26 | 51 | 44 | -330.00 | -292.50 | -1022.00 |
| 2021 | 21 | 58 | 35 | -176.50 | 178.00 | -176.50 |
| 2022 | 21 | 42 | 45 | 268.50 | -263.00 | -438.50 |
| 2023 | 6 | 10 | 14 | 23.00 | -64.00 | 175.00 |

### B — roll_window

| year | n_outside_roll | n_roll_pm2 | pnl_outside_roll | pnl_roll_pm2 |
| --- | --- | --- | --- | --- |
| 2015 | 180 | 13 | -881.00 | -82.50 |
| 2016 | 188 | 9 | -871.00 | -75.50 |
| 2017 | 201 | 16 | -824.50 | 59.00 |
| 2018 | 168 | 7 | -538.00 | -20.50 |
| 2019 | 167 | 11 | -1410.50 | 0.50 |
| 2020 | 109 | 12 | -1459.50 | -185.00 |
| 2021 | 111 | 3 | -126.50 | -48.50 |
| 2022 | 103 | 5 | -519.50 | 86.50 |
| 2023 | 30 | 0 | 134.00 | 0.00 |

### B — session_bucket

| year | n_08:20-09:30 | n_09:30-11:00 | n_11:00-13:25 | pnl_08:20-09:30 | pnl_09:30-11:00 | pnl_11:00-13:25 |
| --- | --- | --- | --- | --- | --- | --- |
| 2015 | 16 | 96 | 81 | -256.00 | -547.00 | -160.50 |
| 2016 | 14 | 95 | 88 | -346.00 | -71.50 | -529.00 |
| 2017 | 15 | 101 | 101 | -146.50 | -308.50 | -310.50 |
| 2018 | 13 | 65 | 97 | -64.50 | -111.50 | -382.50 |
| 2019 | 7 | 88 | 83 | -132.50 | -487.00 | -790.50 |
| 2020 | 10 | 69 | 42 | 303.00 | -1146.50 | -801.00 |
| 2021 | 9 | 62 | 43 | -82.50 | 21.00 | -113.50 |
| 2022 | 12 | 52 | 44 | 141.00 | -968.00 | 394.00 |
| 2023 | 0 | 13 | 17 | 0.00 | 151.50 | -17.50 |

## Ce qui reste inexpliqué

Pour A, aucun état disponible n'explique de façon stable la perte à travers 2010-2019, 2020-2022 et 2023-2024. A4/A5 localisent descriptivement des pertes, mais leur signe change et les intervalles corrigés ne permettent pas une attribution stable. A1 reste absent.

Pour B, les coûts absorbent un brut presque plat, mais le changement massif de liquidité MGC et l'absence de GC empêchent de séparer un vrai changement de mécanisme d'un problème de mesure du climax. B1 reste également absent.

## Conclusions imposées

- **Signal A: (c) pertes diffuses sans état explicatif stable — alpha de régime.**
- **Signal B: (b) problème de mesure probable.** Cette conclusion reste conditionnelle à B0b, qui est précisément le test de mesure manquant.

Aucune règle de trading finale n'est proposée.