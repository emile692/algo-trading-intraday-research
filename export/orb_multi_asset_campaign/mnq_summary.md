# MNQ Ensemble ORB Summary

- Dataset: `MNQ_c_0_1m_20260321_094501.parquet`
- Tradable sessions analysed: 1747
- IS/OOS split: 1222 / 525 sessions
- Instrument specs loaded from config: tick_size=0.25, tick_value=0.5, point_value=2.0, commission_per_side=1.25, slippage_ticks=1

## Baseline Transfer

- Baseline transfer rule: `majority_50`
- OOS score=1.244, PF=1.227, Sharpe=1.018, Return/DD=2.859

## Final Recommendations

- Best OOS ensemble: `unanimity_100` (score=1.402, PF=1.277, Sharpe=1.165, MaxDD=-7263.50).
- Most robust ensemble: `unanimity_100` (robustness=1.364).
- Baseline-like ensemble retained if solid: `unanimity_100`.
- Best point cell: `ATR 25 / q30/q94` (score=1.503).
- Most robust point cell: `ATR 25 / q30/q94` (local robustness=1.223).

## Robust Clusters

- Cluster 1: 11 cells, ATR 25-27, Qlow 26-30, Qhigh 94-95, avg score=1.325.
- Cluster 3: 5 cells, ATR 29-30, Qlow 28-30, Qhigh 95-95, avg score=1.223.
- Cluster 2: 4 cells, ATR 27-28, Qlow 27-29, Qhigh 93-93, avg score=1.208.

## Campaign Readout

- Transferability verdict: transfer acceptable.
- Baseline MNQ-style majority rule gap vs best ensemble: 0.158 score points.
- Sessions with incomplete opening window: 399.
- Sessions missing part of RTH bars: 477.

## Exports

- Full results: `mnq_results_full.csv`
- Top configs: `mnq_top_configs.csv`
- Aggregation summary: `mnq_aggregation_summary.csv`
- Charts folder: `charts/`
