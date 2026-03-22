# Baseline Integrity Report

## Baseline Mapping

- Data loading: `src/data/loader.py` -> `load_ohlcv_file`.
- Feature engineering: `src/features/intraday.py`, `src/features/opening_range.py`, `src/features/volatility.py`.
- ORB generation: `src/strategy/orb.py` (long-only + VWAP confirmation).
- ATR ensemble filter: `src/analytics/atr_ensemble_campaign.py` (cross-product quantiles + vote threshold).
- Backtest engine: `src/engine/backtester.py` (costs/slippage/risk sizing).
- Metrics: `src/analytics/metrics.py`.
- Exports: campaign outputs under `export/orb_research_campaign/`.

## Reference

- Dataset: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\dowloaded\MNQ_c_0_1m_20260321_094501.parquet`
- Legacy baseline strategy id: `ensemble__expanded_q20_25_30__q90_95__majority_50`
- Legacy selected days: 962

## Numerical Non-Regression

| metric | reference | new_pipeline | diff | rel_diff | pass |
|---|---:|---:|---:|---:|:---:|
| n_trades | 828.00000000 | 828.00000000 | 0.00000000 | 0.000000% | yes |
| cumulative_pnl | 19719.50000000 | 19719.50000000 | 0.00000000 | 0.000000% | yes |
| sharpe_ratio | 1.66692800 | 1.66692800 | 0.00000000 | 0.000000% | yes |
| profit_factor | 1.26795529 | 1.26795529 | 0.00000000 | 0.000000% | yes |
| expectancy | 23.81582126 | 23.81582126 | 0.00000000 | 0.000000% | yes |
| max_drawdown | -2132.50000000 | -2132.50000000 | 0.00000000 | 0.000000% | yes |

## Equity Checksum

- reference checksum: `b7f1f5ad7ab6ff29be81380e33ea30956c00fdd01a97a06742ea15b779f97cb7`
- new checksum: `b7f1f5ad7ab6ff29be81380e33ea30956c00fdd01a97a06742ea15b779f97cb7`
- checksum pass: `yes`

## Integrity Verdict

- Baseline preserved within tolerance.
