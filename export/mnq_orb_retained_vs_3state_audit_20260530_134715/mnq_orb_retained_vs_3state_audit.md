# MNQ ORB retained final vs sizing_3state audit

Sources:
- Regime export: `C:\Data\Perso\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Dataset used to rebuild retained final: `C:\Data\Perso\algo-trading-intraday-research\data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet`

## Verdict

The comparison is **not size-normalized and not trade-set identical**.

- `sizing_3state` is a true sizing overlay on **its own nominal ORB baseline**.
- That nominal baseline is **not** the same as `retained final`.
- So `retained final` vs `sizing_3state` mixes:
  1. a different ORB signal/trade set,
  2. a different base risk budget,
  3. then the 3-state multiplier overlay.

## 1. Same trade set?

- Same entry timestamps: `no`
- Same directions: `no`
- Same exits: `no`
- Same stop/target logic: `no`
- Same costs/slippage assumptions: `mostly yes at execution-assumption level`, but not enough to make the comparison normalized.

Evidence:
- Retained trades: `298`
- sizing_3state trades: `1107`
- Overlap on `(entry_time, direction)`: `36`
- Retained-only trades: `262`
- sizing_3state-only trades: `1071`
- Comparable overlapping trades with same exit: `29` / `36`
- Comparable overlapping trades with same stop/target levels: `25` / `36`

Config mismatch behind that result:
- `retained final`: OR `15` / direction `long` / risk `0.50%`
- `sizing_3state` baseline export: OR `30` / direction `both` / risk `1.50%`

## 2. Same base sizing before multiplier?

- Same initial capital: `yes` (`50,000 USD` vs `50,000 USD`)
- Same risk per trade: `no` (`0.50%` retained vs `1.50%` sizing baseline)
- Same contract cap: `effectively yes, no explicit cap in either path`
- Same compounding / equity update logic: `yes in practice, both are static-risk trade logs accumulated into an equity curve`
- Same fixed vs dynamic sizing assumption: `no`
  - `retained final`: fixed baseline risk `%` per trade
  - `sizing_3state`: same static baseline risk `%`, then multiplied by regime bucket

Additional fee sanity:
- Retained median fee per contract: `2.5`
- sizing_3state median fee per contract: `2.5`

## 3. realized_vol_ratio_15_60 bucketing

Actual exported mapping for `sizing_3state_realized_vol_ratio_15_60`:

- `low`: [0.336552, 0.942945] -> `0.50x` (IS composite `-3.313`)
- `mid`: [0.942945, 1.140780] -> `1.00x` (IS composite `1.622`)
- `high`: [1.140780, 1.822058] -> `0.75x` (IS composite `-3.254`)

Resolution of the apparent inconsistency:
- Code tuple `(0.50, 0.75, 1.00)` is **rank-ordered**: worst IS bucket -> middle bucket -> best IS bucket.
- Notebook summary `0.50x / 1.00x / 0.75x` is **label-ordered** here for this feature: `low / mid / high`.
- For this specific feature, the IS ranking is:
  - `low` = worst -> `0.50x`
  - `high` = second -> `0.75x`
  - `mid` = best -> `1.00x`

## 4. Trade-level comparison

- CSV written to `trade_level_comparison.csv`
- This table is matched on `(entry_time, direction)`.
- Because the trade sets differ, many rows are one-sided by construction.

## 5. Sanity check

- All exported 3-state multipliers <= 1.0: `true`
- Flag count where `pnl_ratio > 1` while multiplier <= 1: `31`

Interpretation:
- Those flags do **not** falsify the overlay logic.
- They mainly show that the public comparison is not normalized, because the 3-state branch starts from a larger base risk budget (`1.5%` vs `0.5%`) and also from a different trade set.

## 6. Diagnostic curves

Generated curves:
- `retained final original`
- `retained trade set rescaled with 3-state base sizing, multiplier forced to 1.0`
- `retained trade set rescaled with actual 3-state multipliers`

Interpretation:
- `forced 1.0` isolates the effect of the **larger 3-state base sizing** on the same retained trades.
- `actual multiplier` adds the regime overlay on top of that same retained trade set.
- The gap between retained original and forced `1.0` is base-sizing.
- The gap between forced `1.0` and actual multiplier is the overlay itself.

Artifacts:
- Markdown report: `mnq_orb_retained_vs_3state_audit.md`
- Trade CSV: `trade_level_comparison.csv`
- Plot HTML: `diagnostic_equity_curves.html`
- Plot PNG: `diagnostic_equity_curves.png`
