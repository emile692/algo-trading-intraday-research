# Wide Heatmaps Summary

- OR window: 30 minutes.
- Direction: `both`.
- Target multiple: `2.0`.
- Objective: fixed ATR windows, Q_LOW on Y, Q_HIGH on X, scored on full period (IS+OOS).
- ATR windows tested: 10, 14, 20, 26, 30, 40, 60.
- Logic for ATR windows:
  10/14 = short-term sensitivity, 20/26 = medium-term anchor, 30/40 = smoother swing-adapted range, 60 = slow regime filter.
- Q_LOW grid: 0 to 30.
- Q_HIGH grid: 70 to 100.

## MNQ

- Best cell per ATR (by OVERALL composite score):
  - ATR 10: q24/q71, score=1.664, PF=1.235, Sharpe=0.926.
  - ATR 14: q25/q75, score=1.782, PF=1.250, Sharpe=1.027.
  - ATR 20: q23/q73, score=1.705, PF=1.216, Sharpe=0.890.
  - ATR 26: q21/q73, score=1.844, PF=1.240, Sharpe=0.997.
  - ATR 30: q24/q71, score=1.747, PF=1.230, Sharpe=0.917.
  - ATR 40: q24/q70, score=1.705, PF=1.247, Sharpe=0.954.
  - ATR 60: q17/q72, score=1.768, PF=1.265, Sharpe=1.111.
- Aggregated robust pair (median across ATR windows):
  - q24/q72, median score=1.637, median PF=1.217, median Sharpe=0.879, best ATR by score=26.
