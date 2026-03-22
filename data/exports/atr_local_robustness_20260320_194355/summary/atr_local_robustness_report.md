# ATR Local Robustness Report

## Candidat de depart

- Strategie candidate: ORB baseline + filtre ATR borne.
- Point de reference: ATR(14), q_low=q20, q_high=q90.
- Objectif: verifier si la performance vient d'un plateau local stable ou d'un point fragile.

## Protocole IS/OOS (anti-lookahead)

- Dataset: `MNQ_1mim.parquet`
- Sessions totales: 2127 (2019-05-05 -> 2026-03-02)
- Split chronologique: IS=1488 sessions, OOS=639 sessions
- Date de coupure: OOS commence le 2024-02-14
- Calibration ATR: quantiles estimes uniquement sur IS, puis figes pour OOS.
- Aucune recalibration sur OOS.

## Grille testee

- ATR period: [10, 12, 14, 16, 18, 21, 30, 60]
- q_low: [10, 15, 20, 25, 30]
- q_high: [80, 85, 90, 95]
- Combinaisons valides testees: 160

## Performance absolue

- Baseline IS: Sharpe=0.721, PF=1.150, Expectancy=13.52, Trades=837
- Baseline OOS: Sharpe=0.613, PF=1.142, Expectancy=13.02, Trades=288

Top OOS (apres filtre trade floor):

```text
 atr_period  q_low_pct  q_high_pct  oos_n_trades  oos_sharpe_ratio  oos_profit_factor  oos_expectancy  local_robustness_score  robust_oos_flag
         12         30          80           177          2.165506           1.392544       33.166667                1.105302             True
         12         25          80           184          2.164698           1.395935       32.820652                1.106187             True
         18         30          85           200          2.008372           1.356402       30.680000                1.086558             True
         21         30          85           196          2.007336           1.354568       30.882653                1.036829             True
         18         25          85           205          1.946012           1.345201       29.678049                1.055335             True
         16         25          80           184          1.884895           1.336428       28.657609                1.005653             True
         18         30          80           169          1.821997           1.318140       27.713018                0.939530             True
         12         20          80           195          1.818677           1.323665       27.238462                0.972594             True
```

Top IS (apres filtre trade floor):

```text
 atr_period  q_low_pct  q_high_pct  is_n_trades  is_sharpe_ratio  is_profit_factor  is_expectancy
         60         30          80          451         2.342268          1.411904      33.804878
         60         30          85          489         2.166608          1.378407      31.472393
         18         30          80          443         2.116055          1.366591      31.546275
         10         30          85          475         2.077989          1.364964      31.177895
         21         20          80          538         2.062072          1.357289      30.065056
         18         25          80          492         2.048456          1.353372      30.138211
         18         30          85          478         1.998818          1.345817      30.039749
         10         20          85          571         1.995639          1.347737      29.476357
```

## Robustesse locale autour de (14, q20, q90)

- Candidat: OOS Sharpe=1.145, PF=1.196, Expectancy=17.36, Trades=238
- Candidat robuste OOS (regle 2/3 metriques + trade floor): True
- Neighbors robustes immediats: 6/6 (100.0%)
- Taille du cluster robuste contenant le candidat: 138
- Zone cluster candidat: ATR 10-60, q_low q10-q30, q_high q80-q95
- Moyennes cluster (OOS): Sharpe=1.350, PF=1.235, Expectancy=20.69

## Stabilite OOS

- Analyse stabilite via deltas OOS-IS et score local_robustness_score.
- Heatmaps de stabilite disponibles dans `heatmaps/stability/` et `heatmaps/aggregated/`.

## Verdict

- 1. Le candidat ATR est localement robuste
- Recommandation: Conserver le candidat actuel (14/q20/q90) ou un voisin proche dans le meme plateau.

## Artefacts

- Table complete: `D:/Business/Trading/VSCODE/algo-trading-intraday-research/data/exports/atr_local_robustness_20260320_194355/data/atr_local_grid_all_combinations.csv`
- Metrics IS: `D:/Business/Trading/VSCODE/algo-trading-intraday-research/data/exports/atr_local_robustness_20260320_194355/tables/metrics_is.csv`
- Metrics OOS: `D:/Business/Trading/VSCODE/algo-trading-intraday-research/data/exports/atr_local_robustness_20260320_194355/tables/metrics_oos.csv`
- Clusters robustes: `D:/Business/Trading/VSCODE/algo-trading-intraday-research/data/exports/atr_local_robustness_20260320_194355/tables/robust_clusters.csv`
- Voisinage candidat: `D:/Business/Trading/VSCODE/algo-trading-intraday-research/data/exports/atr_local_robustness_20260320_194355/tables/candidate_neighbors.csv`
- Heatmaps: `D:/Business/Trading/VSCODE/algo-trading-intraday-research/data/exports/atr_local_robustness_20260320_194355/heatmaps`
