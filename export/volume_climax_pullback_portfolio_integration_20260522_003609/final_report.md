# Volume Climax Pullback Portfolio Integration

## Executive Summary
- Baseline only OOS net PnL: `24714.00 USD`, Sharpe `2.09`, PF `1.40`.
- Pullback sleeve only OOS net PnL: `213.33 USD`, Sharpe `0.89`, PF `2.05`.
- Main strict integrated portfolio `baseline_plus_pullback_equal_notional`: `24927.33 USD`, Sharpe `2.11`, PF `1.41`.
- Pullback sleeve vs baseline daily correlation on OOS overlap: `nan`.
- Integrated combo vs baseline daily correlation on OOS overlap: `1.000`.
- Prop constraint impact for the main combo: `0` daily loss breaches, historical status `fail`.
- Bootstrap for the main combo: median `24616.23 USD`, p05 `10071.52 USD`, probability positive `99.8%`.
- Final verdict: `diversifier_watchlist`.

## Inputs
- Baseline daily pnl path: `data\exports\mnq_orb_vvix_sizing_modulation_20260328_run\variants\baseline_3state\daily_results.csv`
- Pullback survivor export: `export\volume_climax_pullback_survivor_audit_20260521_091448`

## Notes
- The pullback sleeve is fixed from the survivor audit strict train-only output.
- No pullback re-optimization, no posthoc asset filtering, and no new gating were used here.
- Risk scaling, when present, is calibrated on the integration train window only (`<= 2023-12-31`).
