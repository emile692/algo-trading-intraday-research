# MNQ prod opposite-breakout live recommendation

## Scope
This readout is MNQ-only and is anchored on the prod-like mini-run `export/orb_opposite_breakout_invalidation_20260529_094539`, because the earlier multi-asset export `20260528_231811` was built on an OR15 / 0.5% risk campaign and does not exactly match the validated live MNQ sleeve.

Sources read:
- Existing quick readout: `export/orb_opposite_breakout_invalidation_20260528_231811/quick_readout/*`
- Prod-like mini-run: `export/orb_opposite_breakout_invalidation_20260529_094539/*`
- Cached MNQ prod bars/features: `.cache/research/orb_opposite_breakout_invalidation/MNQ_mnq_prod_*`

## Detected prod config
See `mnq_prod_config_detected.md`. Confidence remains `medium`: the repo contains a retained OR15 notebook and a later audited OR30 validation/export. For this question, the mini-run used the later audited OR30 long-only sleeve with selected-session controls because that is the strongest prod/live candidate found in the repo.

## Direct answers
- A. Are post-downside-first-breakout MNQ trades losing? No. On the prod-like set they made `1915.42` across `207` trades, but Sharpe was only `0.156`, profit factor `1.041`, and max drawdown `-6404.32`. They are positive, but materially weaker than the rest of the book.
- B. Does removing them help? Yes. `close_1m buffer 1` improves Sharpe from `0.716` to `0.802` and max drawdown from `-8645.12` to `-7775.34`. `n_closes 1m confirm 2` is stronger still: net PnL `30191.76`, Sharpe `0.872`, max drawdown `-6843.70`.
- C. Is `touch buffer 0` too aggressive operationally? Probably yes. It removes trades earlier, but it is the most wick-sensitive rule and gives weaker MNQ metrics than both `close_1m buffer 1` and `n_closes confirm 2`.
- D. Best live rule? `DEPLOY_INVALIDATE_N_CLOSES` is the best fit from the prod-like run. It is still simple and interpretable, avoids wick-only invalidations, materially improves Sharpe and drawdown, and beats the close-1m and close-5m variants on this MNQ-only test.

## Core comparison

| Rule | Net PnL | Sharpe | Sortino | Max DD | Max Daily DD | PF | Trades | Trades removed | Invalidated days | PnL removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_no_opposite_invalidation` | 27409.64 | 0.716 | 1.706 | -8645.12 | -8645.12 | 1.137 | 785 | 0 | 0 | 0.00 |
| `invalidate_on_opposite_touch__buffer_0` | 24766.06 | 0.747 | 1.746 | -7263.72 | -7263.72 | 1.167 | 562 | 223 | 768 | 2643.58 |
| `invalidate_on_opposite_close_1m__buffer_0` | 26839.86 | 0.795 | 1.887 | -7775.34 | -7775.34 | 1.176 | 580 | 205 | 745 | 569.78 |
| `invalidate_on_opposite_close_1m__buffer_1` | 27103.66 | 0.802 | 1.905 | -7775.34 | -7775.34 | 1.177 | 581 | 204 | 744 | 305.98 |
| `invalidate_on_opposite_close_1m__buffer_2` | 27103.66 | 0.802 | 1.905 | -7775.34 | -7775.34 | 1.177 | 581 | 204 | 743 | 305.98 |
| `invalidate_on_opposite_n_closes_1m__buffer_0__confirm_2` | 30191.76 | 0.872 | 2.101 | -6843.70 | -6843.70 | 1.191 | 602 | 183 | 725 | -2782.12 |
| `invalidate_on_opposite_n_closes_1m__buffer_2__confirm_3` | 30356.62 | 0.867 | 2.114 | -6832.06 | -6832.06 | 1.187 | 616 | 169 | 706 | -2946.98 |
| `invalidate_on_opposite_close_5m__buffer_0` | 26644.96 | 0.772 | 1.837 | -8047.58 | -8047.58 | 1.167 | 606 | 179 | 716 | 764.68 |

Interpretation:
- `touch buffer 0` improves Sharpe versus baseline, but cuts net PnL to `24766.06` and is operationally the most fragile to sweeps.
- `close_1m buffer 1` is the cleanest simple live candidate: it keeps net PnL near baseline, lifts Sharpe to `0.802`, and reduces max drawdown by `869.78`.
- `n_closes 1m confirm 2` is the strongest overall MNQ result and improves both return and drawdown, which means the filtered trades were net harmful in aggregate on this prod-like sample.
- `close_5m` works, but it is consistently later than `close_1m` and leaves more bad trades alive. It looks more robust than `touch`, but weaker than `close_1m` and `n_closes` for this MNQ sleeve.

## Downside first breakout decomposition
- Baseline downside-first-breakout trades: `207` trades, net PnL `1915.42`, Sharpe `0.156`, max drawdown `-6404.32`, profit factor `1.041`.
- That subset is not outright losing, but it is far weaker than the no-downside subset, which contributed about `25494.22` of PnL on the same sample.
- `invalidate_on_opposite_touch__buffer_0` invalidates `207` downside-trade days, captures `102` losing downside days (100.0%), but also removes `105` winning downside days (100.0%).
- `invalidate_on_opposite_close_1m__buffer_1` invalidates `195` downside-trade days, captures `101` losing downside days (99.0%), but also removes `94` winning downside days (89.5%).
- `invalidate_on_opposite_n_closes_1m__buffer_0__confirm_2` invalidates `184` downside-trade days, captures `97` losing downside days (95.1%), but also removes `87` winning downside days (82.9%).
- `invalidate_on_opposite_close_5m__buffer_0` invalidates `180` downside-trade days, captures `96` losing downside days (94.1%), but also removes `84` winning downside days (80.0%).

## Annual stability
- `baseline_no_opposite_invalidation`: positive years `5`, negative years `3`, median year PnL `813.56`, worst year `-2180.20`.
- `invalidate_on_opposite_close_1m__buffer_1`: positive years `4`, negative years `4`, median year PnL `729.80`, worst year `-2688.10`.
- `invalidate_on_opposite_n_closes_1m__buffer_0__confirm_2`: positive years `4`, negative years `4`, median year PnL `1288.99`, worst year `-2666.04`.
- `invalidate_on_opposite_close_5m__buffer_0`: positive years `4`, negative years `4`, median year PnL `526.99`, worst year `-3702.74`.
- `invalidate_on_opposite_touch__buffer_0`: positive years `5`, negative years `3`, median year PnL `550.37`, worst year `-2275.24`.

## 5-minute close confirmation

| rule | buffer_ticks | net_pnl | Sharpe | max_drawdown | max_daily_drawdown | trade_count | trades_removed | pnl_removed | invalidated_days_count | invalidated_days_pct | daily_loss_limit_breaches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 27409.64 | 0.716 | -8645.12 | -8645.12 | 785 | 0 | 0.00 | 0 | 0.000 | 0 |
| close_1m | 0 | 26839.86 | 0.795 | -7775.34 | -7775.34 | 580 | 205 | 569.78 | 745 | 0.665 | 0 |
| close_1m | 1 | 27103.66 | 0.802 | -7775.34 | -7775.34 | 581 | 204 | 305.98 | 744 | 0.664 | 0 |
| close_1m | 2 | 27103.66 | 0.802 | -7775.34 | -7775.34 | 581 | 204 | 305.98 | 743 | 0.663 | 0 |
| close_5m | 0 | 26644.96 | 0.772 | -8047.58 | -8047.58 | 606 | 179 | 764.68 | 716 | 0.639 | 0 |
| close_5m | 1 | 26203.58 | 0.758 | -8047.58 | -8047.58 | 608 | 177 | 1206.06 | 713 | 0.637 | 0 |
| close_5m | 2 | 26535.86 | 0.768 | -8047.58 | -8047.58 | 609 | 176 | 873.78 | 711 | 0.635 | 0 |
| close_5m | 4 | 26535.86 | 0.768 | -8047.58 | -8047.58 | 609 | 176 | 873.78 | 709 | 0.633 | 0 |
| touch | 0 | 24766.06 | 0.747 | -7263.72 | -7263.72 | 562 | 223 | 2643.58 | 768 | 0.686 | 0 |

Among baseline downside-trade sessions, `180` had at least one causal 5-minute close below OR low. `98.9%` later reclaimed OR low and `99.4%` later reclaimed VWAP, which explains why `touch` is too eager and why `close_5m` remains cleaner than touch but still slower than `close_1m`.
`close_5m buffer 0` captures fewer losing downside days than `close_1m buffer 1` and `n_closes confirm 2`, while still removing a meaningful number of winning days. On this MNQ run it improves baseline, but it is not the best compromise.
Verdict for 5-minute confirmation: do not prefer it here. It is more robust than touch, but it arrives too late and leaves too many weak trades alive versus the 1-minute close family. If execution simplicity requires a single-bar confirmation, prefer `close_1m buffer 1`; if the stack can support two-bar confirmation cleanly, prefer `n_closes_1m confirm 2`.

## Reclaim configs
- `reclaim require_vwap false` lifts net PnL to `27975.86` but only modestly improves Sharpe to `0.731`. It is a different trade thesis, not just an invalidation overlay.
- `reclaim require_vwap true` is weaker still. Recommendation: keep reclaim separate from the main ORB long-only production rule set for now.

## Final live verdict
`DEPLOY_INVALIDATE_N_CLOSES`

Reasoning:
- The MNQ prod-like sample does not support keeping baseline unchanged. A downside-first-breakout day is still tradable sometimes, but that sleeve is much lower quality than the rest of the book.
- `n_closes 1m confirm 2` is the best overall invalidation rule on MNQ-only: higher net PnL than baseline (`30191.76` vs `27409.64`), higher Sharpe (`0.872` vs `0.716`), and smaller max drawdown (`-6843.70` vs `-8645.12`).
- `close_1m buffer 1` remains the best fallback if the execution repo should stay on a simpler one-bar semantics. It is materially more robust live than touch and still clearly better than baseline.

Recommended YAML:
```yaml
opposite_breakout_policy:
  enabled: true
  mode: invalidate_for_day
  confirmation: n_closes_1m
  buffer_ticks: 0
  confirm_bars: 2
  allow_reclaim: false
```

Fallback YAML if only one-bar confirmation is desired:
```yaml
opposite_breakout_policy:
  enabled: true
  mode: invalidate_for_day
  confirmation: close_1m
  buffer_ticks: 1
  confirm_bars: 1
  allow_reclaim: false
```

## Mini-run provenance
- Command used: `uv run python -m src.analytics.orb_opposite_breakout_invalidation_campaign --prod-mnq-only --output-root export --use-cache --profile`
- Runtime: `313.123s` wall clock, `309.102s` recorded in `run_metadata.json`
- Export: `export/orb_opposite_breakout_invalidation_20260529_094539`