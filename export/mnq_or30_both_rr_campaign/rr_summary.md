# MNQ OR30 Both RR Campaign

- Symbol: `MNQ`
- Opening range: `30 minutes`
- Direction mode: `both`
- Data source: processed parquet from `data/processed/parquet`
- RR values tested: 2, 3, 5, 10

## Comparison Table

  rr folder     best_rule   robust_rule  best_overall_score  best_overall_sharpe  best_overall_pf  best_overall_net_pnl  best_overall_maxdd_abs  best_oos_score  best_oos_sharpe  best_oos_pf  best_oos_net_pnl  best_oos_maxdd_abs  best_cell_atr  best_cell_q_low  best_cell_q_high  robust_clusters
 2.0   rr_2   majority_50   majority_50            0.980703             0.776227         1.153465               47018.0                 11763.0        1.970671         1.905281     1.428075           35575.0              9765.0             30               27                90                2
 3.0   rr_3   majority_50   majority_50           -0.161733             0.463585         1.092962               28911.0                 14567.0        0.743092         1.280779     1.287620           24552.0             13939.0             28               27                94                1
 5.0   rr_5   majority_50   majority_50           -0.025518             0.477741         1.099737               31086.5                 14567.0        0.784510         1.225859     1.292406           25161.5             13939.0             28               27                94                2
10.0  rr_10 unanimity_100 unanimity_100           -1.016178             0.285385         1.060553               17056.5                 16314.0        0.912463         1.303447     1.323918           25149.5             13390.0             28               27                94                1

## Direct Readout

- Best RR by overall score: `RR 2` with `majority_50` | overall score `0.981` | overall Sharpe `0.776` | overall PF `1.153`.
- Best RR by OOS score: `RR 2` with `majority_50` | OOS score `1.971` | OOS Sharpe `1.905` | OOS PF `1.428`.

## Per-RR Folders

### RR 2

- Folder: `rr_2`
- Best ensemble: `majority_50`
- Best cell: `ATR 30 / q27/q90`
- Overall score / Sharpe / PF: `0.981` / `0.776` / `1.153`
- OOS score / Sharpe / PF: `1.971` / `1.905` / `1.428`

### RR 3

- Folder: `rr_3`
- Best ensemble: `majority_50`
- Best cell: `ATR 28 / q27/q94`
- Overall score / Sharpe / PF: `-0.162` / `0.464` / `1.093`
- OOS score / Sharpe / PF: `0.743` / `1.281` / `1.288`

### RR 5

- Folder: `rr_5`
- Best ensemble: `majority_50`
- Best cell: `ATR 28 / q27/q94`
- Overall score / Sharpe / PF: `-0.026` / `0.478` / `1.100`
- OOS score / Sharpe / PF: `0.785` / `1.226` / `1.292`

### RR 10

- Folder: `rr_10`
- Best ensemble: `unanimity_100`
- Best cell: `ATR 28 / q27/q94`
- Overall score / Sharpe / PF: `-1.016` / `0.285` / `1.061`
- OOS score / Sharpe / PF: `0.912` / `1.303` / `1.324`
