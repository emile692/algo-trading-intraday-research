# MNQ Ensemble ORB Summary

- Dataset: `MNQ_c_0_1m_20260321_094501.parquet`
- Tradable sessions analysed: 1747
- IS/OOS split: 1222 / 525 sessions
- Instrument specs loaded from config: tick_size=0.25, tick_value=0.5, point_value=2.0, commission_per_side=1.25, slippage_ticks=1

## Baseline Transfer

- Baseline transfer rule: `majority_50`
- OOS score=1.971, PF=1.428, Sharpe=1.905, Return/DD=3.643

## Final Recommendations

- Best OOS ensemble: `majority_50` (score=1.971, PF=1.428, Sharpe=1.905, MaxDD=-9765.00).
- Most robust ensemble: `majority_50` (robustness=0.499).
- Baseline-like ensemble retained if solid: `majority_50`.
- Best point cell: `ATR 30 / q27/q90` (score=2.174).
- Most robust point cell: `ATR 26 / q25/q94` (local robustness=1.192).

## Robust Clusters

- Cluster 1: 45 cells, ATR 25-30, Qlow 25-30, Qhigh 93-95, avg score=1.987.
- Cluster 2: 2 cells, ATR 25-25, Qlow 29-30, Qhigh 94-94, avg score=2.118.

## Campaign Readout

- Transferability verdict: transfer acceptable.
- Baseline MNQ-style majority rule gap vs best ensemble: 0.000 score points.
- Sessions with incomplete opening window: 401.
- Sessions missing part of RTH bars: 477.

## Exports

- Full results: `mnq_results_full.csv`
- Top configs: `mnq_top_configs.csv`
- Aggregation summary: `mnq_aggregation_summary.csv`
- Charts folder: `charts/`
