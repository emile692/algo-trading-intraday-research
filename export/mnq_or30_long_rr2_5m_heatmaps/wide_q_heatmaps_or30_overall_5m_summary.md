# Wide Heatmaps Summary

- OR window: 30 minutes.
- Direction: `long`.
- Target multiple: `2.0`.
- Data timeframe: `5m`.
- Objective: fixed ATR windows, Q_LOW on Y, Q_HIGH on X, scored on full period (IS+OOS).
- ATR windows tested: 10, 14, 20, 26, 30, 40, 60.
- Logic for ATR windows:
  10/14 = short-term sensitivity, 20/26 = medium-term anchor, 30/40 = smoother swing-adapted range, 60 = slow regime filter.
- Q_LOW grid: 0 to 30.
- Q_HIGH grid: 70 to 100.

## MNQ

- Best cell per ATR (by OVERALL composite score):
  - ATR 10: q28/q73, score=2.417, PF=1.371, Sharpe=1.128.
  - ATR 14: q26/q70, score=2.118, PF=1.283, Sharpe=0.879.
  - ATR 20: q29/q71, score=2.012, PF=1.256, Sharpe=0.797.
  - ATR 26: q21/q71, score=1.917, PF=1.248, Sharpe=0.828.
  - ATR 30: q23/q70, score=2.012, PF=1.234, Sharpe=0.765.
  - ATR 40: q21/q76, score=1.689, PF=1.182, Sharpe=0.662.
  - ATR 60: q30/q79, score=1.567, PF=1.187, Sharpe=0.662.
- Aggregated robust pair (median across ATR windows):
  - q23/q70, median score=1.904, median PF=1.234, median Sharpe=0.765, best ATR by score=14.
