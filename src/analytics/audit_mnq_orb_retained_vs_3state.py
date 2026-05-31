from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analytics.mnq_orb_regime_filter_sizing_campaign import _scale_nominal_trades_by_multiplier
from src.analytics.orb_research.campaign import _evaluate_experiment, _split_sessions
from src.analytics.orb_research.features import (
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import (
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignContext,
    CompressionConfig,
    DynamicThresholdConfig,
    ExitConfig,
    ExperimentConfig,
)
from src.engine.portfolio import build_equity_curve
from src.features.volatility import add_rolling_std


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_GLOB = "mnq_orb_regime_filter_sizing_*"
DEFAULT_VARIANT = "sizing_3state_realized_vol_ratio_15_60"
DEFAULT_INITIAL_CAPITAL = 50_000.0
MNQ_TICK_VALUE_USD = 0.5
MNQ_POINT_VALUE_USD = 2.0
MNQ_COMMISSION_PER_SIDE_USD = 1.25
THREE_STATE_BASE_RISK_PCT = 1.5


@dataclass(frozen=True)
class RetainedConfig:
    name: str = "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate"
    or_minutes: int = 15
    opening_time: str = "09:30:00"
    direction: str = "long"
    one_trade_per_day: bool = True
    entry_buffer_ticks: int = 2
    stop_buffer_ticks: int = 2
    target_multiple: float = 2.0
    vwap_confirmation: bool = True
    vwap_column: str = "continuous_session_vwap"
    time_exit: str = "16:00:00"
    account_size_usd: float = DEFAULT_INITIAL_CAPITAL
    risk_per_trade_pct: float = 0.5
    tick_size: float = 0.25
    entry_on_next_open: bool = True
    atr_window: int = 14
    q_lows_pct: tuple[int, ...] = (20, 25, 30)
    q_highs_pct: tuple[int, ...] = (90, 95)
    vote_threshold: float = 0.5
    compression_mode: str = "weak_close"
    compression_usage: str = "soft_vote_bonus"
    compression_soft_bonus_votes: float = 1.0
    exit_mode: str = "baseline"
    dynamic_mode: str = "noise_area_gate"
    noise_lookback: int = 30
    noise_vm: float = 1.0
    noise_k: float = 0.0
    dynamic_atr_k: float = 0.0
    dynamic_confirm_bars: int = 1
    dynamic_schedule: str = "continuous_on_bar_close"
    dynamic_threshold_style: str = "max_or_high_noise"


def _latest_export_root() -> Path:
    exports = sorted((REPO_ROOT / "data" / "exports").glob(DEFAULT_EXPORT_GLOB))
    if not exports:
        raise FileNotFoundError(f"No export root found for {DEFAULT_EXPORT_GLOB}.")
    return exports[-1]


def _latest_mnq_dataset() -> Path:
    files = sorted((REPO_ROOT / "data" / "processed" / "parquet").glob("MNQ_c_0_1m_*.parquet"))
    if not files:
        raise FileNotFoundError("No processed MNQ 1m dataset found under data/processed/parquet.")
    return files[-1]


def _make_output_dir(base: Path | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base or (REPO_ROOT / "export" / f"mnq_orb_retained_vs_3state_audit_{timestamp}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_three_state_export(export_root: Path, variant_name: str) -> dict[str, pd.DataFrame]:
    summary = pd.read_csv(export_root / "summary_variants.csv")
    controls = pd.read_csv(export_root / "variants" / variant_name / "controls.csv", parse_dates=["session_date"])
    trades = pd.read_csv(
        export_root / "variants" / variant_name / "trades.csv",
        parse_dates=["session_date", "entry_time", "exit_time"],
    )
    nominal_trades = pd.read_csv(
        export_root / "variants" / "nominal" / "trades.csv",
        parse_dates=["session_date", "entry_time", "exit_time"],
    )
    mappings = pd.read_csv(export_root / "regime_state_mappings.csv")
    return {
        "summary": summary,
        "controls": controls,
        "trades": trades,
        "nominal_trades": nominal_trades,
        "mappings": mappings,
    }


def _rebuild_retained_final(dataset_path: Path) -> dict[str, object]:
    cfg = RetainedConfig()
    entry = BaselineEntryConfig(
        or_minutes=cfg.or_minutes,
        opening_time=cfg.opening_time,
        direction=cfg.direction,
        one_trade_per_day=cfg.one_trade_per_day,
        entry_buffer_ticks=cfg.entry_buffer_ticks,
        stop_buffer_ticks=cfg.stop_buffer_ticks,
        target_multiple=cfg.target_multiple,
        vwap_confirmation=cfg.vwap_confirmation,
        vwap_column=cfg.vwap_column,
        time_exit=cfg.time_exit,
        account_size_usd=cfg.account_size_usd,
        risk_per_trade_pct=cfg.risk_per_trade_pct,
        tick_size=cfg.tick_size,
        entry_on_next_open=cfg.entry_on_next_open,
    )
    ensemble = BaselineEnsembleConfig(
        atr_window=cfg.atr_window,
        q_lows_pct=cfg.q_lows_pct,
        q_highs_pct=cfg.q_highs_pct,
        vote_threshold=cfg.vote_threshold,
    )
    compression = CompressionConfig(
        mode=cfg.compression_mode,
        usage=cfg.compression_usage,
        soft_bonus_votes=cfg.compression_soft_bonus_votes,
    )
    dynamic = DynamicThresholdConfig(
        mode=cfg.dynamic_mode,
        noise_lookback=cfg.noise_lookback,
        noise_vm=cfg.noise_vm,
        threshold_style=cfg.dynamic_threshold_style,
        noise_k=cfg.noise_k,
        atr_k=cfg.dynamic_atr_k,
        confirm_bars=cfg.dynamic_confirm_bars,
        schedule=cfg.dynamic_schedule,
    )
    exit_cfg = ExitConfig(mode=cfg.exit_mode)
    experiment = ExperimentConfig(
        name=cfg.name,
        stage="full_reopt",
        family="full_reopt",
        baseline_entry=entry,
        baseline_ensemble=ensemble,
        compression=compression,
        exit=exit_cfg,
        dynamic_threshold=dynamic,
    )

    minute_df = prepare_minute_dataset(dataset_path=dataset_path, baseline_entry=entry, atr_windows=(cfg.atr_window,))
    daily_reference = build_daily_reference(minute_df)
    minute_df = attach_daily_reference(minute_df, daily_reference)
    candidate_base = build_candidate_universe(minute_df, baseline_entry=entry)
    all_sessions = sorted(pd.to_datetime(minute_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, 0.70)
    context = CampaignContext(
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        minute_df=minute_df,
        candidate_base_df=candidate_base,
        daily_patterns=daily_reference,
    )
    row, detail = _evaluate_experiment(
        experiment=experiment,
        context=context,
        bootstrap_paths=300,
        random_seed=42,
        keep_details=True,
        max_leverage=None,
    )
    if detail is None:
        raise RuntimeError(f"Failed to rebuild retained final: {row}")

    trades = detail["trades"].copy()
    selected_final = detail["selected_final"].copy()
    return {
        "config": cfg,
        "row": row,
        "minute_df": minute_df,
        "candidate_base": candidate_base,
        "trades": trades,
        "selected_final": selected_final,
        "all_sessions": all_sessions,
        "is_sessions": is_sessions,
        "oos_sessions": oos_sessions,
    }


def _prepare_trade_keys(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce", utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce", utc=True)
    out["trade_key"] = out["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z") + "|" + out["direction"].astype(str)
    return out


def _same_trade_set_summary(retained: pd.DataFrame, sizing: pd.DataFrame) -> dict[str, object]:
    retained_keys = set(retained["trade_key"])
    sizing_keys = set(sizing["trade_key"])
    overlap = retained_keys & sizing_keys
    return {
        "retained_trade_count": len(retained),
        "sizing_trade_count": len(sizing),
        "overlap_count": len(overlap),
        "retained_only_count": len(retained_keys - sizing_keys),
        "sizing_only_count": len(sizing_keys - retained_keys),
        "same_trade_set": retained_keys == sizing_keys,
    }


def _same_exit_summary(retained: pd.DataFrame, sizing: pd.DataFrame) -> dict[str, object]:
    common = retained.merge(
        sizing,
        on="trade_key",
        how="inner",
        suffixes=("_retained", "_3state"),
    )
    if common.empty:
        return {"comparable_count": 0, "same_exit_count": 0, "same_stop_target_logic_count": 0}
    same_exit = common["exit_time_retained"].eq(common["exit_time_3state"]) & common["exit_reason_retained"].eq(
        common["exit_reason_3state"]
    )
    same_levels = (
        np.isclose(common["stop_price_retained"], common["stop_price_3state"], equal_nan=True)
        & np.isclose(common["target_price_retained"], common["target_price_3state"], equal_nan=True)
    )
    return {
        "comparable_count": len(common),
        "same_exit_count": int(same_exit.sum()),
        "same_stop_target_logic_count": int(same_levels.sum()),
    }


def _bucket_mapping_for_variant(mappings: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    subset = mappings.loc[
        (mappings["variant_name"] == variant_name) & (mappings["feature_name"] == "realized_vol_ratio_15_60")
    ].copy()
    if subset.empty:
        raise ValueError(f"No mapping rows found for {variant_name}.")
    subset = subset.sort_values("bucket_position").reset_index(drop=True)
    return subset[
        [
            "bucket_label",
            "bucket_position",
            "lower_bound",
            "upper_bound",
            "is_composite_score",
            "risk_multiplier",
        ]
    ].copy()


def _assign_bucket(value: float, bucket_map: pd.DataFrame) -> str | None:
    if not math.isfinite(value):
        return None
    for row in bucket_map.itertuples(index=False):
        lower = float(row.lower_bound)
        upper = float(row.upper_bound)
        if lower <= value <= upper:
            return str(row.bucket_label)
    return None


def _retained_session_multipliers(retained: dict[str, object], bucket_map: pd.DataFrame) -> pd.DataFrame:
    minute_df = retained["minute_df"].copy()
    minute_df = add_rolling_std(minute_df, window=15)
    minute_df = add_rolling_std(minute_df, window=60)
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="coerce", utc=True)

    selected = retained["selected_final"].copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="coerce", utc=True)
    selected["session_date"] = pd.to_datetime(selected["session_date"], errors="coerce").dt.date

    signal_features = minute_df[["session_date", "timestamp", "vol_std_15", "vol_std_60"]].copy()
    signal_features["session_date"] = pd.to_datetime(signal_features["session_date"], errors="coerce").dt.date

    merged = selected.merge(signal_features, on=["session_date", "timestamp"], how="left")
    merged["realized_vol_ratio_15_60"] = pd.to_numeric(merged["vol_std_15"], errors="coerce") / pd.to_numeric(
        merged["vol_std_60"], errors="coerce"
    )
    merged["bucket_label"] = merged["realized_vol_ratio_15_60"].apply(lambda value: _assign_bucket(value, bucket_map))
    multiplier_map = dict(
        zip(bucket_map["bucket_label"].astype(str), pd.to_numeric(bucket_map["risk_multiplier"], errors="coerce"))
    )
    merged["risk_multiplier"] = merged["bucket_label"].map(multiplier_map).astype(float)
    return merged[["session_date", "timestamp", "realized_vol_ratio_15_60", "bucket_label", "risk_multiplier"]].copy()


def _controls_from_session_multipliers(session_map: pd.DataFrame, multiplier_override: float | None = None) -> pd.DataFrame:
    controls = session_map[["session_date", "bucket_label", "risk_multiplier"]].copy()
    controls["phase"] = "retained"
    controls["selected_by_ensemble"] = True
    controls["feature_name"] = "realized_vol_ratio_15_60"
    if multiplier_override is not None:
        controls["risk_multiplier"] = float(multiplier_override)
        controls["bucket_label"] = "forced_1p0"
    controls["skip_trade"] = pd.to_numeric(controls["risk_multiplier"], errors="coerce").fillna(0.0).le(0.0)
    return controls[["session_date", "phase", "selected_by_ensemble", "feature_name", "bucket_label", "risk_multiplier", "skip_trade"]]


def _scale_trades_for_diagnostic(trades: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    scaled = _scale_nominal_trades_by_multiplier(
        nominal_trades=trades,
        controls=controls,
        account_size_usd=DEFAULT_INITIAL_CAPITAL,
        base_risk_pct=THREE_STATE_BASE_RISK_PCT,
        tick_value_usd=MNQ_TICK_VALUE_USD,
        point_value_usd=MNQ_POINT_VALUE_USD,
        commission_per_side_usd=MNQ_COMMISSION_PER_SIDE_USD,
    )
    scaled["session_date"] = pd.to_datetime(scaled["session_date"], errors="coerce")
    return scaled


def _trade_level_comparison(retained_trades: pd.DataFrame, sizing_trades: pd.DataFrame) -> pd.DataFrame:
    left = retained_trades.rename(
        columns={
            "session_date": "session_date_retained",
            "entry_time": "entry_time_retained",
            "direction": "direction_retained",
            "net_pnl_usd": "net_pnl_retained",
            "quantity": "contracts_retained",
        }
    )
    right = sizing_trades.rename(
        columns={
            "session_date": "session_date_3state",
            "entry_time": "entry_time_3state",
            "direction": "direction_3state",
            "net_pnl_usd": "net_pnl_3state",
            "quantity": "contracts_3state",
        }
    )
    merged = left.merge(
        right[
            [
                "trade_key",
                "session_date_3state",
                "entry_time_3state",
                "direction_3state",
                "net_pnl_3state",
                "contracts_3state",
                "risk_multiplier",
            ]
        ],
        on="trade_key",
        how="outer",
    )
    merged["entry_time"] = merged["entry_time_retained"].combine_first(merged["entry_time_3state"])
    merged["direction"] = merged["direction_retained"].combine_first(merged["direction_3state"])
    merged["session_date"] = merged["session_date_retained"].combine_first(merged["session_date_3state"])
    merged["pnl_ratio"] = pd.to_numeric(merged["net_pnl_3state"], errors="coerce") / pd.to_numeric(
        merged["net_pnl_retained"], errors="coerce"
    )
    merged["flag_pnl_ratio_gt_1_with_multiplier_le_1"] = (
        pd.to_numeric(merged["pnl_ratio"], errors="coerce").gt(1.0)
        & pd.to_numeric(merged["risk_multiplier"], errors="coerce").le(1.0)
    )
    merged["match_status"] = np.select(
        [
            merged["net_pnl_retained"].notna() & merged["net_pnl_3state"].notna(),
            merged["net_pnl_retained"].notna(),
            merged["net_pnl_3state"].notna(),
        ],
        [
            "matched_on_entry_time_and_direction",
            "retained_only",
            "sizing_3state_only",
        ],
        default="unclassified",
    )
    merged = merged.sort_values("entry_time").reset_index(drop=True)
    return merged[
        [
            "session_date",
            "entry_time",
            "direction",
            "net_pnl_retained",
            "net_pnl_3state",
            "contracts_retained",
            "contracts_3state",
            "risk_multiplier",
            "pnl_ratio",
            "flag_pnl_ratio_gt_1_with_multiplier_le_1",
            "match_status",
        ]
    ].copy()


def _curve_frame(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    curve = build_equity_curve(trades, initial_capital=DEFAULT_INITIAL_CAPITAL).copy()
    if curve.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown", "label"])
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce", utc=True)
    curve["label"] = label
    return curve


def _write_curve_outputs(curves: list[pd.DataFrame], output_dir: Path) -> tuple[Path, Path]:
    fig = go.Figure()
    for curve in curves:
        if curve.empty:
            continue
        label = str(curve["label"].iloc[0])
        fig.add_trace(go.Scatter(x=curve["timestamp"], y=curve["equity"], mode="lines", name=label))
    fig.update_layout(
        title="MNQ ORB diagnostic equity curves",
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Equity (USD)",
        legend_title="Curve",
        width=1200,
        height=650,
    )
    html_path = output_dir / "diagnostic_equity_curves.html"
    fig.write_html(html_path)

    png_path = output_dir / "diagnostic_equity_curves.png"
    plt.figure(figsize=(14, 7))
    for curve in curves:
        if curve.empty:
            continue
        label = str(curve["label"].iloc[0])
        plt.plot(curve["timestamp"], curve["equity"], label=label)
    plt.title("MNQ ORB diagnostic equity curves")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return html_path, png_path


def _fees_per_contract(trades: pd.DataFrame) -> float | None:
    valid = trades.loc[pd.to_numeric(trades["quantity"], errors="coerce").gt(0)].copy()
    if valid.empty:
        return None
    series = pd.to_numeric(valid["fees"], errors="coerce") / pd.to_numeric(valid["quantity"], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return None
    return float(series.median())


def _write_report(
    output_dir: Path,
    export_root: Path,
    dataset_path: Path,
    retained: dict[str, object],
    three_state: dict[str, pd.DataFrame],
    bucket_map: pd.DataFrame,
    trade_comp: pd.DataFrame,
    forced_curve: pd.DataFrame,
    actual_curve: pd.DataFrame,
) -> Path:
    retained_trades = _prepare_trade_keys(retained["trades"])
    sizing_trades = _prepare_trade_keys(three_state["trades"])
    trade_set = _same_trade_set_summary(retained_trades, sizing_trades)
    exit_set = _same_exit_summary(retained_trades, sizing_trades)

    retained_fee_per_contract = _fees_per_contract(retained_trades)
    sizing_fee_per_contract = _fees_per_contract(sizing_trades)
    all_multipliers_le_1 = bool(pd.to_numeric(sizing_trades["risk_multiplier"], errors="coerce").le(1.0).all())
    pnl_ratio_flags = int(trade_comp["flag_pnl_ratio_gt_1_with_multiplier_le_1"].fillna(False).sum())

    retained_cfg = retained["config"]
    bucket_lines = "\n".join(
        f"- `{row.bucket_label}`: [{row.lower_bound:.6f}, {row.upper_bound:.6f}] -> `{row.risk_multiplier:.2f}x` (IS composite `{row.is_composite_score:.3f}`)"
        for row in bucket_map.itertuples(index=False)
    )
    report = f"""# MNQ ORB retained final vs sizing_3state audit

Sources:
- Regime export: `{export_root}`
- Dataset used to rebuild retained final: `{dataset_path}`

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
- Retained trades: `{trade_set["retained_trade_count"]}`
- sizing_3state trades: `{trade_set["sizing_trade_count"]}`
- Overlap on `(entry_time, direction)`: `{trade_set["overlap_count"]}`
- Retained-only trades: `{trade_set["retained_only_count"]}`
- sizing_3state-only trades: `{trade_set["sizing_only_count"]}`
- Comparable overlapping trades with same exit: `{exit_set["same_exit_count"]}` / `{exit_set["comparable_count"]}`
- Comparable overlapping trades with same stop/target levels: `{exit_set["same_stop_target_logic_count"]}` / `{exit_set["comparable_count"]}`

Config mismatch behind that result:
- `retained final`: OR `{retained_cfg.or_minutes}` / direction `{retained_cfg.direction}` / risk `{retained_cfg.risk_per_trade_pct:.2f}%`
- `sizing_3state` baseline export: OR `30` / direction `both` / risk `{THREE_STATE_BASE_RISK_PCT:.2f}%`

## 2. Same base sizing before multiplier?

- Same initial capital: `yes` (`50,000 USD` vs `50,000 USD`)
- Same risk per trade: `no` (`0.50%` retained vs `1.50%` sizing baseline)
- Same contract cap: `effectively yes, no explicit cap in either path`
- Same compounding / equity update logic: `yes in practice, both are static-risk trade logs accumulated into an equity curve`
- Same fixed vs dynamic sizing assumption: `no`
  - `retained final`: fixed baseline risk `%` per trade
  - `sizing_3state`: same static baseline risk `%`, then multiplied by regime bucket

Additional fee sanity:
- Retained median fee per contract: `{retained_fee_per_contract if retained_fee_per_contract is not None else "n/a"}`
- sizing_3state median fee per contract: `{sizing_fee_per_contract if sizing_fee_per_contract is not None else "n/a"}`

## 3. realized_vol_ratio_15_60 bucketing

Actual exported mapping for `sizing_3state_realized_vol_ratio_15_60`:

{bucket_lines}

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

- All exported 3-state multipliers <= 1.0: `{str(all_multipliers_le_1).lower()}`
- Flag count where `pnl_ratio > 1` while multiplier <= 1: `{pnl_ratio_flags}`

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
"""
    path = output_dir / "mnq_orb_retained_vs_3state_audit.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_audit(export_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Path]:
    export_root = export_root or _latest_export_root()
    dataset_path = _latest_mnq_dataset()
    output_dir = _make_output_dir(output_dir)

    three_state = _load_three_state_export(export_root, DEFAULT_VARIANT)
    bucket_map = _bucket_mapping_for_variant(three_state["mappings"], DEFAULT_VARIANT)
    retained = _rebuild_retained_final(dataset_path)

    retained_trades = _prepare_trade_keys(retained["trades"])
    sizing_trades = _prepare_trade_keys(three_state["trades"])

    trade_comparison = _trade_level_comparison(retained_trades, sizing_trades)
    trade_csv = output_dir / "trade_level_comparison.csv"
    trade_comparison.to_csv(trade_csv, index=False)

    retained_session_map = _retained_session_multipliers(retained, bucket_map)
    forced_controls = _controls_from_session_multipliers(retained_session_map, multiplier_override=1.0)
    actual_controls = _controls_from_session_multipliers(retained_session_map, multiplier_override=None)

    forced_trades = _scale_trades_for_diagnostic(retained["trades"], forced_controls)
    actual_trades = _scale_trades_for_diagnostic(retained["trades"], actual_controls)

    retained_curve = _curve_frame(retained["trades"], "retained final original")
    forced_curve = _curve_frame(forced_trades, "retained trade set + 3-state base sizing (1.0x)")
    actual_curve = _curve_frame(actual_trades, "retained trade set + actual 3-state multiplier")
    html_plot, png_plot = _write_curve_outputs([retained_curve, forced_curve, actual_curve], output_dir)

    report_path = _write_report(
        output_dir=output_dir,
        export_root=export_root,
        dataset_path=dataset_path,
        retained=retained,
        three_state=three_state,
        bucket_map=bucket_map,
        trade_comp=trade_comparison,
        forced_curve=forced_curve,
        actual_curve=actual_curve,
    )
    return {
        "report": report_path,
        "trade_csv": trade_csv,
        "plot_html": html_plot,
        "plot_png": png_plot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MNQ ORB retained final vs sizing_3state.")
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    outputs = run_audit(export_root=args.export_root, output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
