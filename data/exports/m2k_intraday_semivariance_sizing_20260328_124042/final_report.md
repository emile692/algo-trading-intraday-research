# M2K Intraday Semivariance Sizing Follow-Up

## Methodology

- Asset only: `M2K`.
- Baseline reused unchanged from `D:\Business\Trading\VSCODE\algo-trading-intraday-research\notebooks\orb_M2K_final_ensemble_validation.ipynb` with OR `15m`, direction `long`, rule `majority_50`.
- Overlay scope restricted to post-signal sizing modulation; no alpha, execution, slippage, cost, or portfolio-accounting changes.
- Semivariance horizons tested: `30m, 60m, 90m, session`.
- Downside-focused features ranked with strict prior-history rolling percentiles only; no same-bar or future leakage.
- Variant thresholds were screened on IS only, then OOS was used only to decide whether any IS-promoted candidate is defensible.

## Variants Tested

- `downside_downsize`: feature grid `['rs_minus_pct', 'rs_minus_share_pct', 'rs_ratio_pct']` x horizons `['30m', '60m', '90m', 'session']` x thresholds `[0.85, 0.9, 0.95]` x down-multipliers `[0.75, 0.5]`.
- `downside_three_state`: same downside features/horizons/thresholds with low-threshold `0.25` and multiplier pairs `[(1.1, 0.75), (1.25, 0.5)]`.
- `conditional_downsize_with_context`: downside features `['rs_minus_share_pct', 'rs_ratio_pct']` confirmed by contexts `['wide_or', 'opening_gap', 'high_atr']`.
- `reference_downside_hard_skip`: small comparison grid on features `['rs_minus_pct']` and horizons `['60m', 'session']`.

## Baseline OOS

- Sharpe `-0.462` | net PnL `-8157.50` | maxDD `-14044.50` | PF `0.909` | trades `257`.

## IS Promotion Candidates

- `downside_downsize__rs_minus_pct__session__t90__d50`: IS Sharpe delta `+0.027`, IS maxDD improvement `+7.5%`, OOS Sharpe delta `-0.118`, OOS maxDD improvement `-7.5%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_minus_share_pct__90m__t95__d50`: IS Sharpe delta `+0.045`, IS maxDD improvement `+2.9%`, OOS Sharpe delta `-0.039`, OOS maxDD improvement `+2.6%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_ratio_pct__90m__t95__d50`: IS Sharpe delta `+0.045`, IS maxDD improvement `+2.9%`, OOS Sharpe delta `-0.039`, OOS maxDD improvement `+2.6%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_minus_share_pct__session__t90__d50`: IS Sharpe delta `+0.023`, IS maxDD improvement `+5.9%`, OOS Sharpe delta `+0.034`, OOS maxDD improvement `+11.4%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_ratio_pct__session__t90__d50`: IS Sharpe delta `+0.023`, IS maxDD improvement `+5.9%`, OOS Sharpe delta `+0.034`, OOS maxDD improvement `+11.4%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_minus_pct__30m__t85__d50`: IS Sharpe delta `+0.019`, IS maxDD improvement `+6.5%`, OOS Sharpe delta `+0.122`, OOS maxDD improvement `+25.8%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_minus_pct__session__t95__d50`: IS Sharpe delta `+0.023`, IS maxDD improvement `+4.6%`, OOS Sharpe delta `-0.110`, OOS maxDD improvement `-6.9%`, OOS trade retention `100.0%`.
- `downside_three_state__rs_minus_pct__30m__t85__lo25__u125__d50`: IS Sharpe delta `+0.027`, IS maxDD improvement `+3.7%`, OOS Sharpe delta `+0.146`, OOS maxDD improvement `+18.9%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_minus_pct__session__t90__d75`: IS Sharpe delta `+0.020`, IS maxDD improvement `+4.5%`, OOS Sharpe delta `-0.052`, OOS maxDD improvement `-2.4%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_minus_share_pct__90m__t95__d75`: IS Sharpe delta `+0.025`, IS maxDD improvement `+2.0%`, OOS Sharpe delta `-0.026`, OOS maxDD improvement `+0.6%`, OOS trade retention `100.0%`.

## OOS Ranking

- `downside_downsize__rs_minus_share_pct__90m__t90__d50`: OOS Sharpe `-0.270` (delta `+0.192`), net PnL `-4574.00`, maxDD `-10233.50` (improvement `+27.1%`), PF `0.945`, trades `257`.
- `downside_downsize__rs_ratio_pct__90m__t90__d50`: OOS Sharpe `-0.270` (delta `+0.192`), net PnL `-4574.00`, maxDD `-10233.50` (improvement `+27.1%`), PF `0.945`, trades `257`.
- `downside_downsize__rs_minus_share_pct__60m__t90__d50`: OOS Sharpe `-0.272` (delta `+0.190`), net PnL `-4589.00`, maxDD `-11347.50` (improvement `+19.2%`), PF `0.944`, trades `257`.
- `downside_downsize__rs_ratio_pct__60m__t90__d50`: OOS Sharpe `-0.272` (delta `+0.190`), net PnL `-4589.00`, maxDD `-11347.50` (improvement `+19.2%`), PF `0.944`, trades `257`.
- `downside_three_state__rs_minus_share_pct__90m__t90__lo25__u125__d50`: OOS Sharpe `-0.274` (delta `+0.189`), net PnL `-4958.50`, maxDD `-11742.50` (improvement `+16.4%`), PF `0.944`, trades `257`.
- `downside_three_state__rs_ratio_pct__90m__t90__lo25__u125__d50`: OOS Sharpe `-0.274` (delta `+0.189`), net PnL `-4958.50`, maxDD `-11742.50` (improvement `+16.4%`), PF `0.944`, trades `257`.
- `downside_three_state__rs_minus_share_pct__60m__t90__lo25__u125__d50`: OOS Sharpe `-0.297` (delta `+0.165`), net PnL `-5327.50`, maxDD `-12323.50` (improvement `+12.3%`), PF `0.939`, trades `257`.
- `downside_three_state__rs_ratio_pct__60m__t90__lo25__u125__d50`: OOS Sharpe `-0.297` (delta `+0.165`), net PnL `-5327.50`, maxDD `-12323.50` (improvement `+12.3%`), PF `0.939`, trades `257`.
- `reference_downside_hard_skip__rs_minus_pct__60m__t95`: OOS Sharpe `-0.300` (delta `+0.162`), net PnL `-5254.50`, maxDD `-12497.50` (improvement `+11.0%`), PF `0.939`, trades `249`.
- `downside_downsize__rs_minus_share_pct__60m__t85__d50`: OOS Sharpe `-0.301` (delta `+0.162`), net PnL `-4979.00`, maxDD `-11106.00` (improvement `+20.9%`), PF `0.938`, trades `257`.

## Verdict

- Best IS-screened candidate: `downside_downsize__rs_minus_pct__session__t90__d50` with IS score `0.165`.
- Best ex-post OOS row: `downside_downsize__rs_minus_share_pct__90m__t90__d50` with Sharpe delta `+0.192` and maxDD improvement `+27.1%`.
- Promotion decision: `YES`. Promoted variant `downside_three_state__rs_minus_pct__30m__t85__lo25__u125__d50` improved both OOS Sharpe and OOS max drawdown while retaining `100.0%` of baseline trades, with clustered support count `2`.
