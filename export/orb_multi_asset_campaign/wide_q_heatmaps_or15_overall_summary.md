# Wide Heatmaps Summary

- OR window: 15 minutes.
- Objective: fixed ATR windows, Q_LOW on Y, Q_HIGH on X, scored on full period (IS+OOS).
- ATR windows tested: 10, 14, 20, 26, 30, 40, 60.
- Logic for ATR windows:
  10/14 = short-term sensitivity, 20/26 = medium-term anchor, 30/40 = smoother swing-adapted range, 60 = slow regime filter.
- Q_LOW grid: 0 to 30.
- Q_HIGH grid: 70 to 100.

## MNQ

- Best cell per ATR (by OVERALL composite score):
  - ATR 10: q28/q72, score=2.326, PF=1.373, Sharpe=1.240.
  - ATR 14: q28/q70, score=2.294, PF=1.380, Sharpe=1.218.
  - ATR 20: q22/q70, score=2.558, PF=1.399, Sharpe=1.336.
  - ATR 26: q21/q70, score=2.249, PF=1.362, Sharpe=1.248.
  - ATR 30: q23/q70, score=2.316, PF=1.385, Sharpe=1.283.
  - ATR 40: q26/q71, score=2.278, PF=1.364, Sharpe=1.222.
  - ATR 60: q28/q71, score=2.411, PF=1.444, Sharpe=1.431.
- Aggregated robust pair (median across ATR windows):
  - q26/q70, median score=2.257, median PF=1.362, median Sharpe=1.200, best ATR by score=20.
