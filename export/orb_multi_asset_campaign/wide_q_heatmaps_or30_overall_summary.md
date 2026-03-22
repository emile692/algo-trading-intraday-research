# Wide Heatmaps Summary

- OR window: 30 minutes.
- Objective: fixed ATR windows, Q_LOW on Y, Q_HIGH on X, scored on full period (IS+OOS).
- ATR windows tested: 10, 14, 20, 26, 30, 40, 60.
- Logic for ATR windows:
  10/14 = short-term sensitivity, 20/26 = medium-term anchor, 30/40 = smoother swing-adapted range, 60 = slow regime filter.
- Q_LOW grid: 0 to 30.
- Q_HIGH grid: 70 to 100.

## MNQ

- Best cell per ATR (by OVERALL composite score):
  - ATR 10: q24/q73, score=2.371, PF=1.287, Sharpe=0.955.
  - ATR 14: q27/q75, score=2.432, PF=1.345, Sharpe=1.116.
  - ATR 20: q27/q70, score=2.677, PF=1.375, Sharpe=1.112.
  - ATR 26: q30/q70, score=2.532, PF=1.394, Sharpe=1.130.
  - ATR 30: q25/q70, score=2.425, PF=1.331, Sharpe=1.017.
  - ATR 40: q30/q73, score=2.420, PF=1.317, Sharpe=0.979.
  - ATR 60: q21/q70, score=2.389, PF=1.361, Sharpe=1.144.
- Aggregated robust pair (median across ATR windows):
  - q29/q70, median score=2.388, median PF=1.336, median Sharpe=0.992, best ATR by score=20.

## MES

- Best cell per ATR (by OVERALL composite score):
  - ATR 10: q27/q89, score=-1.276, PF=1.049, Sharpe=0.227.
  - ATR 14: q26/q74, score=-1.169, PF=1.053, Sharpe=0.213.
  - ATR 20: q27/q75, score=-1.002, PF=1.060, Sharpe=0.242.
  - ATR 26: q27/q75, score=-1.035, PF=1.065, Sharpe=0.260.
  - ATR 30: q29/q74, score=-0.632, PF=1.075, Sharpe=0.291.
  - ATR 40: q29/q73, score=-0.388, PF=1.092, Sharpe=0.346.
  - ATR 60: q30/q73, score=-1.097, PF=1.055, Sharpe=0.207.
- Aggregated robust pair (median across ATR windows):
  - q27/q75, median score=-1.174, median PF=1.052, median Sharpe=0.206, best ATR by score=20.

## MGC

- Best cell per ATR (by OVERALL composite score):
  - ATR 10: q28/q99, score=-2.842, PF=0.983, Sharpe=-0.079.
  - ATR 14: q30/q99, score=-3.165, PF=0.962, Sharpe=-0.171.
  - ATR 20: q30/q99, score=-3.115, PF=0.964, Sharpe=-0.164.
  - ATR 26: q30/q99, score=-2.911, PF=0.976, Sharpe=-0.109.
  - ATR 30: q30/q99, score=-2.804, PF=0.980, Sharpe=-0.089.
  - ATR 40: q30/q100, score=-2.491, PF=0.995, Sharpe=-0.023.
  - ATR 60: q30/q100, score=-2.534, PF=0.995, Sharpe=-0.022.
- Aggregated robust pair (median across ATR windows):
  - q30/q99, median score=-2.855, median PF=0.980, median Sharpe=-0.092, best ATR by score=40.

## M2K

- Best cell per ATR (by OVERALL composite score):
  - ATR 10: q13/q78, score=-0.940, PF=1.074, Sharpe=0.298.
  - ATR 14: q19/q89, score=-0.713, PF=1.074, Sharpe=0.313.
  - ATR 20: q22/q86, score=-0.187, PF=1.099, Sharpe=0.395.
  - ATR 26: q20/q88, score=-0.226, PF=1.098, Sharpe=0.402.
  - ATR 30: q19/q88, score=0.015, PF=1.102, Sharpe=0.423.
  - ATR 40: q18/q85, score=0.265, PF=1.121, Sharpe=0.489.
  - ATR 60: q19/q73, score=0.496, PF=1.157, Sharpe=0.558.
- Aggregated robust pair (median across ATR windows):
  - q20/q88, median score=-0.226, median PF=1.098, median Sharpe=0.402, best ATR by score=60.
