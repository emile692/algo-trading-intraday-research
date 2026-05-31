# MNQ ORB 3-state high-bucket stress

Source export:
- `C:\Data\Perso\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`

Tested variants:
- `low=0.50x`, `mid=1.00x`, `high in [0.75, 0.5, 0.25, 0.0]`

Reference bucket map:
- `high`: [1.140780, 1.822058] -> default `0.75x`
- `low`: [0.336552, 0.942945] -> default `0.50x`
- `mid`: [0.942945, 1.140780] -> default `1.00x`

## Recommendation

Recommended live-oriented variant: **`high_0p25`** with `high=0.25x`.

Why:
- OOS Sharpe: `3.023` vs current `0.75x` at `2.638`
- OOS net PnL: `27,959.0 USD` vs current `0.75x` at `28,826.0 USD`
- OOS max drawdown: `-2,420.5 USD` vs current `0.75x` at `-5,934.5 USD`
- OOS worst daily PnL: `-744.0 USD` vs current `0.75x` at `-744.0 USD`
- OOS trades: `257` vs current `0.75x` at `326`

Live interpretation:
- The objective here is not to maximize raw PnL.
- The preferred variant is the one that preserves a useful share of OOS PnL while reducing path risk, especially drawdown and worst day.
- `high=0.00x` should not be selected blindly if it removes too much PnL or too many trades.

## Ranking

 rank variant_name  high_bucket_multiplier  oos_annualized_sharpe  oos_max_drawdown_usd  oos_net_pnl_usd  oos_worst_daily_pnl_usd  oos_n_trades
    1    high_0p25                    0.25               3.023167               -2420.5          27959.0                   -744.0           257
    2     high_0p0                    0.00               2.978803               -2446.5          27321.5                   -744.0           226
    3     high_0p5                    0.50               2.816178               -3795.5          27812.5                   -744.0           323
    4    high_0p75                    0.75               2.637890               -5934.5          28826.0                   -744.0           326

## Conservative takeaway

- If the `high` bucket contributes weakly or negatively while worsening drawdown, it should be cut aggressively.
- If `high=0.00x` improves risk but leaves too little PnL or diversification, prefer an intermediate setting.
- The final choice should bias toward smoother live behavior rather than the highest backtest gross.
