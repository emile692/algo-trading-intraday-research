# MES Ensemble ORB Summary

- Dataset: `MES_c_0_1m_20260322_135702.parquet`
- Tradable sessions analysed: 1747
- IS/OOS split: 1222 / 525 sessions
- Instrument specs loaded from config: tick_size=0.25, tick_value=1.25, point_value=5.0, commission_per_side=1.25, slippage_ticks=1

## Baseline Transfer

- Baseline transfer rule: `majority_50`
- OOS score=-2.778, PF=0.985, Sharpe=-0.081, Return/DD=-0.125

## Final Recommendations

- Best OOS ensemble: `majority_50` (score=-2.778, PF=0.985, Sharpe=-0.081, MaxDD=-13851.25).
- Most robust ensemble: `majority_50` (robustness=-3.061).
- Baseline-like ensemble retained if solid: `majority_50`.
- Best point cell: `ATR 26 / q27/q91` (score=-2.357).
- Most robust point cell: `ATR 26 / q27/q91` (local robustness=-2.400).

## Robust Clusters

- No robust cluster detected.

## Campaign Readout

- Transferability verdict: transfer weak / requires caution.
- Baseline MNQ-style majority rule gap vs best ensemble: 0.000 score points.
- Sessions with incomplete opening window: 399.
- Sessions missing part of RTH bars: 477.

## Exports

- Full results: `mes_results_full.csv`
- Top configs: `mes_top_configs.csv`
- Aggregation summary: `mes_aggregation_summary.csv`
- Charts folder: `charts/`
