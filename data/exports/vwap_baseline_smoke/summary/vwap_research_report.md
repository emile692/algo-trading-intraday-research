# VWAP Research Campaign

## Scope

- Dataset: `MNQ_c_0_1m_20260321_094501.parquet`
- Symbol: `MNQ`
- Sessions total / IS / OOS: 1747 / 1222 / 525
- RTH handling is explicit and uses `[09:30, 16:00)` start-aligned bars.
- The paper baseline is implemented first without filters, targets, stops, or kill switches.

## Best Prop Variant

- Selected variant: `paper_vwap_baseline`

## Comparative Table

```text
               name   family  overall_net_pnl  overall_sharpe_ratio  overall_profit_factor  overall_max_drawdown  oos_net_pnl  oos_sharpe_ratio  oos_profit_factor  oos_max_drawdown  oos_daily_loss_limit_breach_freq  oos_trailing_drawdown_breach_freq  oos_profit_to_drawdown_ratio
paper_vwap_baseline baseline          -4838.0             -0.613966               0.885277               -9177.5          0.0               0.0                0.0               0.0                               0.0                                0.0                           inf
```
