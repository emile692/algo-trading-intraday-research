# Replication Sanity Check

Paper reference stylized facts on QQQ:
- Low hit rate around 17%
- Large gain/loss asymmetry around 5.7x
- High trade count around 21,967 trades
- PnL concentrated in the opening phase and final hour
- Trend-following intraday profile

Observed on `MNQ` with this repo:
- Hit rate: 15.07%
- Average gain/loss ratio: 4.97
- Number of trades: 3703
- Early+late session PnL share: 153.48%
- Profit factor: 0.89

Coherent points:
- The hit rate remains low, consistent with a trend follower.
- Winners remain materially larger than losers.
- A large share of PnL still comes from the open and the final hour.

Plausible divergences:
- The tested underlying can differ from QQQ/TQQQ, especially on futures.
- The repo uses explicit futures-like slippage and contract commissions when applicable.
- Execution is forced at the next bar open with start-aligned timestamps.
- RTH handling is explicit and excludes the synthetic `16:00` start bar.
