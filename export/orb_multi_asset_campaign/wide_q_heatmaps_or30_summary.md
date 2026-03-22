# Wide Heatmaps Summary

- OR window: 30 minutes.
- Objective: fixed ATR windows, Q_LOW on Y, Q_HIGH on X.
- ATR windows tested: 10, 14, 20, 26, 30, 40, 60.
- Logic for ATR windows:
  10/14 = short-term sensitivity, 20/26 = medium-term anchor, 30/40 = smoother swing-adapted range, 60 = slow regime filter.
- Q_LOW grid: 0 to 30.
- Q_HIGH grid: 70 to 100.

## MNQ

- Best cell per ATR (by OOS composite score):
  - ATR 10: q30/q79, score=2.024, PF=1.320, Sharpe=1.009.
  - ATR 14: q30/q75, score=2.907, PF=1.486, Sharpe=1.304.
  - ATR 20: q30/q74, score=2.696, PF=1.526, Sharpe=1.375.
  - ATR 26: q30/q75, score=2.238, PF=1.396, Sharpe=1.071.
  - ATR 30: q24/q85, score=1.538, PF=1.286, Sharpe=1.008.
  - ATR 40: q30/q82, score=1.498, PF=1.241, Sharpe=0.784.
  - ATR 60: q30/q70, score=2.151, PF=1.553, Sharpe=1.324.
- Aggregated robust pair (median across ATR windows):
  - q30/q73, median score=1.949, median PF=1.390, median Sharpe=1.041, best ATR by score=20.

## MGC

- Best cell per ATR (by OOS composite score):
  - ATR 10: q29/q86, score=2.012, PF=1.335, Sharpe=1.079.
  - ATR 14: q26/q85, score=1.781, PF=1.323, Sharpe=1.034.
  - ATR 20: q23/q77, score=1.931, PF=1.351, Sharpe=0.910.
  - ATR 26: q30/q78, score=1.239, PF=1.280, Sharpe=0.756.
  - ATR 30: q30/q99, score=1.324, PF=1.249, Sharpe=1.045.
  - ATR 40: q27/q81, score=1.531, PF=1.306, Sharpe=0.835.
  - ATR 60: q28/q100, score=1.512, PF=1.247, Sharpe=1.053.
- Aggregated robust pair (median across ATR windows):
  - q27/q99, median score=1.168, median PF=1.223, median Sharpe=0.954, best ATR by score=14.
