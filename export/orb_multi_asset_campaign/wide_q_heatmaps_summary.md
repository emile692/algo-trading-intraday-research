# Wide Heatmaps Summary

- Objective: fixed ATR windows, Q_LOW on Y, Q_HIGH on X.
- ATR windows tested: 10, 14, 20, 26, 30, 40, 60.
- Logic for ATR windows:
  10/14 = short-term sensitivity, 20/26 = medium-term anchor, 30/40 = smoother swing-adapted range, 60 = slow regime filter.
- Q_LOW grid: 0 to 30.
- Q_HIGH grid: 70 to 100.

## MES

- Best cell per ATR (by OOS composite score):
  - ATR 10: q22/q72, score=-1.054, PF=1.064, Sharpe=0.285.
  - ATR 14: q23/q70, score=-0.265, PF=1.137, Sharpe=0.560.
  - ATR 20: q29/q75, score=0.101, PF=1.161, Sharpe=0.672.
  - ATR 26: q27/q73, score=0.731, PF=1.220, Sharpe=0.870.
  - ATR 30: q24/q70, score=0.741, PF=1.196, Sharpe=0.761.
  - ATR 40: q30/q71, score=1.132, PF=1.225, Sharpe=0.850.
  - ATR 60: q30/q70, score=-0.461, PF=1.104, Sharpe=0.422.
- Aggregated robust pair (median across ATR windows):
  - q29/q71, median score=-0.047, median PF=1.146, median Sharpe=0.566, best ATR by score=40.

## M2K

- Best cell per ATR (by OOS composite score):
  - ATR 10: q7/q88, score=-2.247, PF=1.009, Sharpe=0.044.
  - ATR 14: q8/q91, score=-2.143, PF=1.016, Sharpe=0.079.
  - ATR 20: q14/q94, score=-1.895, PF=1.030, Sharpe=0.147.
  - ATR 26: q8/q87, score=-1.193, PF=1.059, Sharpe=0.283.
  - ATR 30: q8/q88, score=-1.329, PF=1.062, Sharpe=0.297.
  - ATR 40: q28/q78, score=-0.847, PF=1.083, Sharpe=0.347.
  - ATR 60: q21/q70, score=0.308, PF=1.163, Sharpe=0.633.
- Aggregated robust pair (median across ATR windows):
  - q8/q84, median score=-2.293, median PF=1.007, median Sharpe=0.033, best ATR by score=30.
