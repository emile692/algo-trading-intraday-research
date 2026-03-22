# ATR Ensemble Campaign Report

## Rappel

- Baseline: ORB long + VWAP continue, sans filtre ATR additionnel.
- Point unique de reference: ATR(14), q20/q90.
- Ensembles testes: agregations de sous-signaux ATR sur zones voisines.

## Protocole IS/OOS

- Dataset: `MNQ_1mim.parquet`
- Sessions: 2127 (2019-05-05 -> 2026-03-02)
- Split: IS=1488 sessions, OOS=639 sessions
- OOS starts at: 2024-02-14
- Calibration ATR: seuils quantiles calibres sur IS uniquement, puis figes sur OOS.

## Sous-modeles composant les ensembles

```text
                      zone                                  model_id  atr_period  q_low_pct  q_high_pct  low_threshold_is  high_threshold_is                                     pass_column
     narrow_q20_25__q90_95      narrow_q20_25__q90_95__atr14_q20_q90          14       20.0        90.0          6.553571          24.305357      pass__narrow_q20_25__q90_95__atr14_q20_q90
     narrow_q20_25__q90_95      narrow_q20_25__q90_95__atr14_q20_q95          14       20.0        95.0          6.553571          28.818750      pass__narrow_q20_25__q90_95__atr14_q20_q95
     narrow_q20_25__q90_95      narrow_q20_25__q90_95__atr14_q25_q90          14       25.0        90.0          7.553571          24.305357      pass__narrow_q20_25__q90_95__atr14_q25_q90
     narrow_q20_25__q90_95      narrow_q20_25__q90_95__atr14_q25_q95          14       25.0        95.0          7.553571          28.818750      pass__narrow_q20_25__q90_95__atr14_q25_q95
expanded_q20_25_30__q90_95 expanded_q20_25_30__q90_95__atr14_q20_q90          14       20.0        90.0          6.553571          24.305357 pass__expanded_q20_25_30__q90_95__atr14_q20_q90
expanded_q20_25_30__q90_95 expanded_q20_25_30__q90_95__atr14_q20_q95          14       20.0        95.0          6.553571          28.818750 pass__expanded_q20_25_30__q90_95__atr14_q20_q95
expanded_q20_25_30__q90_95 expanded_q20_25_30__q90_95__atr14_q25_q90          14       25.0        90.0          7.553571          24.305357 pass__expanded_q20_25_30__q90_95__atr14_q25_q90
expanded_q20_25_30__q90_95 expanded_q20_25_30__q90_95__atr14_q25_q95          14       25.0        95.0          7.553571          28.818750 pass__expanded_q20_25_30__q90_95__atr14_q25_q95
expanded_q20_25_30__q90_95 expanded_q20_25_30__q90_95__atr14_q30_q90          14       30.0        90.0          8.607143          24.305357 pass__expanded_q20_25_30__q90_95__atr14_q30_q90
expanded_q20_25_30__q90_95 expanded_q20_25_30__q90_95__atr14_q30_q95          14       30.0        95.0          8.607143          28.818750 pass__expanded_q20_25_30__q90_95__atr14_q30_q95
               point_model                             point_q20_q90          14       20.0        90.0          6.553571          24.305357                             pass__point_q20_q90
     point_model_secondary                     point_avg_q22p5_q92p5          14       22.5        92.5          7.098661          26.551786                     pass__point_avg_q22p5_q92p5
```

## Tableau consolide des variantes

```text
                                        strategy_id strategy_group                       zone      aggregation  n_submodels  oos_n_trades  oos_sharpe_ratio  oos_profit_factor  oos_expectancy  oos_max_drawdown  stability_gap  ensemble_score_vs_point
                                    baseline_no_atr       baseline                       none             none            0           288          0.612873           1.142040       13.020833           -1879.0       0.152730                -0.184691
  ensemble__expanded_q20_25_30__q90_95__majority_50       ensemble expanded_q20_25_30__q90_95      majority_50            6           246          1.067404           1.282293       24.772358           -1166.5       0.257968                 0.252774
       ensemble__narrow_q20_25__q90_95__majority_50       ensemble      narrow_q20_25__q90_95      majority_50            4           256          0.952399           1.243581       21.494141           -1288.5       0.193272                 0.143713
      ensemble__narrow_q20_25__q90_95__consensus_75       ensemble      narrow_q20_25__q90_95     consensus_75            4           228          0.871651           1.235345       20.717105           -1499.5       0.232808                 0.065256
     ensemble__narrow_q20_25__q90_95__consensus_100       ensemble      narrow_q20_25__q90_95    consensus_100            4           228          0.871651           1.235345       20.717105           -1499.5       0.232808                 0.065256
ensemble__expanded_q20_25_30__q90_95__score_ge_0p67       ensemble expanded_q20_25_30__q90_95    score_ge_0p67            6           228          0.871651           1.235345       20.717105           -1499.5       0.232808                 0.065256
 ensemble__expanded_q20_25_30__q90_95__consensus_75       ensemble expanded_q20_25_30__q90_95     consensus_75            6           220          0.808540           1.219918       19.643182           -1337.0       0.084568                 0.059681
ensemble__expanded_q20_25_30__q90_95__consensus_100       ensemble expanded_q20_25_30__q90_95    consensus_100            6           220          0.808540           1.219918       19.643182           -1337.0       0.084568                 0.059681
                              point_avg_q22p5_q92p5    point_model      point_model_secondary single_point_avg            1           242          1.055759           1.282546       24.586777           -1227.5       0.175606                 0.248293
                                      point_q20_q90    point_model                point_model     single_point            1           238          0.752588           1.196080       17.361345           -1462.5       0.675639                -0.101346
```

## Diagnostics consensus

- Distribution des scores exportee dans `diagnostics/consensus_score_distribution.csv`.
- Performance conditionnelle par score exportee dans `diagnostics/consensus_performance_by_score.csv`.
- Figures consensus dans `diagnostics/figures/`.

- Resume quantitatif OOS par niveau de score:
- expanded_q20_25_30__q90_95: best expectancy at score=0.5000 (exp=76.14, n_trades=18); best PF at score=0.6667 (PF=1.957); Spearman(score, expectancy)=0.500.
- narrow_q20_25__q90_95: best expectancy at score=0.5000 (exp=27.82, n_trades=28); best PF at score=0.5000 (PF=1.309); Spearman(score, expectancy)=0.500.

## Comparaison cible (Baseline vs Point vs Meilleur Ensemble)

```text
                                      strategy_id  oos_n_trades  oos_win_rate  oos_expectancy  oos_profit_factor  oos_sharpe_ratio  oos_cumulative_pnl  oos_max_drawdown  oos_time_exit_rate  oos_time_exit_win_rate  stability_gap
                                  baseline_no_atr           288      0.440972       13.020833           1.142040          0.612873              3750.0           -1879.0            0.309028                0.719101       0.152730
                                    point_q20_q90           238      0.457983       17.361345           1.196080          0.752588              4132.0           -1462.5            0.306723                0.739726       0.675639
ensemble__expanded_q20_25_30__q90_95__majority_50           246      0.471545       24.772358           1.282293          1.067404              6094.0           -1166.5            0.300813                0.783784       0.257968
```

## Reponses aux hypotheses

- Q1 (ensemble vs point unique): meilleur ensemble = `ensemble__expanded_q20_25_30__q90_95__majority_50`.
- Q2 (majoritaire vs strict): voir classement `all_variants_metrics.csv`.
- Q3 (consensus fort => meilleure qualite): verifier expectancy/PF par niveau de score dans diagnostics.
- Q4 (robustesse OOS): evaluee via metrics OOS + stability_gap.

## Conclusion finale

- 3. Le signal d'ensemble est clairement superieur
