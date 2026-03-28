# M2K Intraday Semivariance Sizing Follow-Up

## Methodology

- Asset only: `M2K`.
- Baseline reused unchanged from `D:\Business\Trading\VSCODE\algo-trading-intraday-research\notebooks\orb_M2K_final_ensemble_validation.ipynb` with OR `15m`, direction `long`, rule `majority_50`.
- Overlay scope restricted to post-signal sizing modulation; no alpha, execution, slippage, cost, or portfolio-accounting changes.
- Semivariance horizons tested: `30m, session`.
- Downside-focused features ranked with strict prior-history rolling percentiles only; no same-bar or future leakage.
- Variant thresholds were screened on IS only, then OOS was used only to decide whether any IS-promoted candidate is defensible.

## Variants Tested

- `downside_downsize`: feature grid `['rs_minus_share_pct', 'rs_ratio_pct']` x horizons `['30m', 'session']` x thresholds `[0.85, 0.9]` x down-multipliers `[0.75]`.
- `downside_three_state`: same downside features/horizons/thresholds with low-threshold `0.25` and multiplier pairs `[(1.1, 0.75)]`.
- `conditional_downsize_with_context`: downside features `['rs_minus_share_pct']` confirmed by contexts `['wide_or', 'opening_gap']`.
- `reference_downside_hard_skip`: small comparison grid on features `['rs_minus_pct']` and horizons `['session']`.

## Baseline OOS

- Sharpe `-0.462` | net PnL `-8157.50` | maxDD `-14044.50` | PF `0.909` | trades `257`.

## IS Promotion Candidates

- `downside_downsize__rs_minus_share_pct__30m__t85__d75`: IS Sharpe delta `+0.001`, IS maxDD improvement `+4.9%`, OOS Sharpe delta `+0.014`, OOS maxDD improvement `+5.8%`, OOS trade retention `100.0%`.
- `downside_downsize__rs_ratio_pct__30m__t85__d75`: IS Sharpe delta `+0.001`, IS maxDD improvement `+4.9%`, OOS Sharpe delta `+0.014`, OOS maxDD improvement `+5.8%`, OOS trade retention `100.0%`.
- `conditional_downsize_with_context__rs_minus_share_pct__session__t85__d75__opening_gap`: IS Sharpe delta `+0.016`, IS maxDD improvement `+0.9%`, OOS Sharpe delta `-0.005`, OOS maxDD improvement `-0.2%`, OOS trade retention `100.0%`.
- `downside_three_state__rs_minus_share_pct__30m__t85__lo25__u110__d75`: IS Sharpe delta `+0.004`, IS maxDD improvement `+1.8%`, OOS Sharpe delta `+0.021`, OOS maxDD improvement `+4.4%`, OOS trade retention `100.0%`.
- `downside_three_state__rs_ratio_pct__30m__t85__lo25__u110__d75`: IS Sharpe delta `+0.004`, IS maxDD improvement `+1.8%`, OOS Sharpe delta `+0.021`, OOS maxDD improvement `+4.4%`, OOS trade retention `100.0%`.

## OOS Ranking

- `conditional_downsize_with_context__rs_minus_share_pct__30m__t85__d75__opening_gap`: OOS Sharpe `-0.437` (delta `+0.026`), net PnL `-7632.00`, maxDD `-13519.00` (improvement `+3.7%`), PF `0.914`, trades `257`.
- `downside_downsize__rs_minus_share_pct__session__t85__d75`: OOS Sharpe `-0.437` (delta `+0.026`), net PnL `-7397.00`, maxDD `-12992.50` (improvement `+7.5%`), PF `0.913`, trades `257`.
- `downside_downsize__rs_ratio_pct__session__t85__d75`: OOS Sharpe `-0.437` (delta `+0.026`), net PnL `-7397.00`, maxDD `-12992.50` (improvement `+7.5%`), PF `0.913`, trades `257`.
- `downside_three_state__rs_minus_share_pct__30m__t85__lo25__u110__d75`: OOS Sharpe `-0.442` (delta `+0.021`), net PnL `-7682.00`, maxDD `-13428.00` (improvement `+4.4%`), PF `0.913`, trades `257`.
- `downside_three_state__rs_ratio_pct__30m__t85__lo25__u110__d75`: OOS Sharpe `-0.442` (delta `+0.021`), net PnL `-7682.00`, maxDD `-13428.00` (improvement `+4.4%`), PF `0.913`, trades `257`.
- `downside_three_state__rs_minus_share_pct__session__t85__lo25__u110__d75`: OOS Sharpe `-0.443` (delta `+0.019`), net PnL `-7725.50`, maxDD `-13261.00` (improvement `+5.6%`), PF `0.912`, trades `257`.
- `downside_three_state__rs_ratio_pct__session__t85__lo25__u110__d75`: OOS Sharpe `-0.443` (delta `+0.019`), net PnL `-7725.50`, maxDD `-13261.00` (improvement `+5.6%`), PF `0.912`, trades `257`.
- `downside_three_state__rs_minus_share_pct__30m__t90__lo25__u110__d75`: OOS Sharpe `-0.445` (delta `+0.017`), net PnL `-7790.00`, maxDD `-13536.00` (improvement `+3.6%`), PF `0.912`, trades `257`.
- `downside_three_state__rs_ratio_pct__30m__t90__lo25__u110__d75`: OOS Sharpe `-0.445` (delta `+0.017`), net PnL `-7790.00`, maxDD `-13536.00` (improvement `+3.6%`), PF `0.912`, trades `257`.
- `downside_downsize__rs_minus_share_pct__30m__t85__d75`: OOS Sharpe `-0.449` (delta `+0.014`), net PnL `-7572.00`, maxDD `-13231.00` (improvement `+5.8%`), PF `0.912`, trades `257`.

## Verdict

- Best IS-screened candidate: `downside_downsize__rs_minus_share_pct__30m__t85__d75` with IS score `0.126`.
- Best ex-post OOS row: `conditional_downsize_with_context__rs_minus_share_pct__30m__t85__d75__opening_gap` with Sharpe delta `+0.026` and maxDD improvement `+3.7%`.
- Promotion decision: `NO`. No IS-screened variant achieved the required OOS Sharpe improvement, OOS max-drawdown improvement, trade retention, and non-isolated support simultaneously.
