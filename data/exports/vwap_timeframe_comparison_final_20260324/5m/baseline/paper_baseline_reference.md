# Paper Baseline Reference

- Official reference baseline is rerun under the current leak-free semantics.
- Signal: previous close vs previous session VWAP.
- Execution: next open, always-in-market during RTH, flat overnight.
- Session: `09:30:00` -> `16:00:00`.
- Costs/profile: `paper_reference`.

```text
        strategy_id                     signal_rule      execution_rule              session  flat_overnight       quantity_mode  initial_capital_usd execution_profile  overall_total_trades  overall_net_pnl  overall_profit_factor  overall_sharpe_ratio  overall_max_drawdown  overall_worst_daily_loss_usd  overall_daily_loss_limit_breach_freq  overall_trailing_drawdown_breach_freq  oos_total_trades  oos_net_pnl  oos_profit_factor  oos_sharpe_ratio  oos_max_drawdown  oos_worst_daily_loss_usd  oos_daily_loss_limit_breach_freq  oos_trailing_drawdown_breach_freq  oos_net_pnl_slippage_x2  oos_profit_factor_slippage_x2  oos_sharpe_ratio_slippage_x2
paper_vwap_baseline close[t-1] vs session VWAP[t-1] next open leak-free 09:30:00 -> 16:00:00            True paper_full_notional              25000.0   paper_reference                 13051          -8172.5               0.980261             -0.227347              -14632.0                       -2065.5                              0.003434                               0.765312              3881      -3239.5            0.97887         -0.231765           -7327.0                   -2065.5                          0.011429                           0.638095                  -7120.5                       0.954443                     -0.505742
```
