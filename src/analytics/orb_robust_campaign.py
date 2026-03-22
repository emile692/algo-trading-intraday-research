"""Disciplined ORB robustness campaign runner focused on simple, explainable filters."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.config.orb_campaign import DEFAULT_CAMPAIGN_DATASET
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.config.settings import DEFAULT_INITIAL_CAPITAL_USD
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.backtester import run_backtest
from src.engine.execution_model import ExecutionModel
from src.engine.portfolio import build_equity_curve
from src.features.intraday import add_continuous_session_vwap, add_intraday_features, add_session_vwap
from src.features.opening_range import compute_opening_range
from src.features.volatility import add_atr
from src.strategy.orb import ORBStrategy


@dataclass(frozen=True)
class BaselineSpec:
    """Baseline strategy specification to reproduce exactly."""

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
    account_size_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    risk_per_trade_pct: float = 0.5
    tick_size: float = 0.25
    entry_on_next_open: bool = True


@dataclass(frozen=True)
class VariantSpec:
    """Single-filter variant definition."""

    name: str
    block: str
    hypothesis: str
    column: str
    lower: float | None = None
    upper: float | None = None
    strictly_positive: bool = False


FEATURE_COLS_FOR_ANALYSIS = [
    "atr_14",
    "or_width",
    "or_width_atr_ratio",
    "price_vwap_distance_atr",
    "continuous_vwap_slope_5",
    "continuous_vwap_slope_5_atr",
]


def _safe_ratio(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    numerator_series = pd.Series(numerator)
    denominator_series = pd.Series(denominator)
    out = numerator_series / denominator_series.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_relative_delta(candidate: float, baseline: float) -> float:
    if baseline == 0 or not math.isfinite(baseline) or not math.isfinite(candidate):
        return 0.0
    return (candidate - baseline) / abs(baseline)


def _profit_factor_from_pnl(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    if losses <= 0:
        return float(np.inf) if wins > 0 else 0.0
    return float(wins / losses)


def _serialize_for_json(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _save_json(path: Path, payload: dict[str, object]) -> None:
    cleaned = {key: _serialize_for_json(value) for key, value in payload.items()}
    path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")


def _make_output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "baseline": root / "baseline",
        "diagnostics": root / "diagnostics",
        "campaigns": root / "campaigns",
        "finalists": root / "finalists",
        "summary": root / "summary",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def prepare_feature_dataset(
    dataset_path: Path,
    baseline: BaselineSpec,
) -> pd.DataFrame:
    """Prepare full-session feature set consistent with the notebook baseline flow."""
    raw = load_ohlcv_file(dataset_path)
    raw = clean_ohlcv(raw)
    feat = add_intraday_features(raw)
    feat = add_atr(feat, window=14)
    feat = add_session_vwap(feat)
    feat = add_continuous_session_vwap(feat, session_start_hour=18)
    feat = compute_opening_range(feat, or_minutes=baseline.or_minutes, opening_time=baseline.opening_time)

    feat["or_width_atr_ratio"] = _safe_ratio(feat["or_width"], feat["atr_14"])
    feat["price_vwap_distance"] = feat["close"] - feat["continuous_session_vwap"]
    feat["price_vwap_distance_atr"] = _safe_ratio(feat["price_vwap_distance"], feat["atr_14"])
    feat["continuous_vwap_slope_5"] = feat.groupby("continuous_session_date")["continuous_session_vwap"].diff(5)
    feat["continuous_vwap_slope_5_atr"] = _safe_ratio(feat["continuous_vwap_slope_5"], feat["atr_14"])
    return feat


def build_strategy_from_baseline(baseline: BaselineSpec, one_trade_per_day: bool) -> ORBStrategy:
    """Instantiate ORB strategy using baseline configuration."""
    return ORBStrategy(
        or_minutes=baseline.or_minutes,
        direction=baseline.direction,
        one_trade_per_day=one_trade_per_day,
        entry_buffer_ticks=baseline.entry_buffer_ticks,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        opening_time=baseline.opening_time,
        time_exit=baseline.time_exit,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=baseline.risk_per_trade_pct,
        tick_size=baseline.tick_size,
        vwap_confirmation=baseline.vwap_confirmation,
        vwap_column=baseline.vwap_column,
    )


def run_backtest_for_signals(
    signal_df: pd.DataFrame,
    baseline: BaselineSpec,
    execution_model: ExecutionModel,
) -> pd.DataFrame:
    """Run backtest with baseline execution assumptions."""
    return run_backtest(
        signal_df,
        execution_model=execution_model,
        time_exit=baseline.time_exit,
        stop_buffer_ticks=baseline.stop_buffer_ticks,
        target_multiple=baseline.target_multiple,
        account_size_usd=baseline.account_size_usd,
        risk_per_trade_pct=baseline.risk_per_trade_pct,
        entry_on_next_open=baseline.entry_on_next_open,
    )


def baseline_vwap_column(df: pd.DataFrame) -> str:
    return "continuous_session_vwap" if "continuous_session_vwap" in df.columns else "session_vwap"


def _selected_signals(signal_df: pd.DataFrame) -> pd.DataFrame:
    selected = signal_df.loc[signal_df["signal"].ne(0)].copy()
    selected = selected.sort_values("timestamp")
    selected = selected.drop_duplicates(subset=["session_date"], keep="first")
    selected = selected.rename(columns={"timestamp": "signal_time"})
    keep_cols = ["session_date", "signal_time", "close", baseline_vwap_column(signal_df)] + FEATURE_COLS_FOR_ANALYSIS
    keep_cols = [col for col in keep_cols if col in selected.columns]
    return selected[keep_cols]


def enrich_trades_with_signal_features(
    trades: pd.DataFrame,
    signal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach signal-time features to each trade (one trade max per session)."""
    selected = _selected_signals(signal_df)
    if trades.empty:
        empty = trades.copy()
        for column in selected.columns:
            if column != "session_date":
                empty[column] = pd.Series(dtype=float)
        return empty
    merged = trades.merge(selected, on="session_date", how="left")
    return merged


def _save_equity_artifacts(
    trades: pd.DataFrame,
    initial_capital: float,
    output_dir: Path,
    stem: str,
) -> pd.DataFrame:
    equity = build_equity_curve(trades, initial_capital=initial_capital)
    equity.to_csv(output_dir / f"{stem}_equity_curve.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if equity.empty:
        axes[0].text(0.5, 0.5, "No trades", ha="center", va="center")
        axes[1].text(0.5, 0.5, "No trades", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
    else:
        axes[0].plot(equity["timestamp"], equity["equity"], color="#111827", linewidth=1.5)
        axes[0].set_title("Equity Curve")
        axes[0].set_ylabel("Equity (USD)")

        axes[1].plot(equity["timestamp"], equity["drawdown"], color="#b91c1c", linewidth=1.2)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("USD")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}_equity_drawdown.png", dpi=150)
    plt.close(fig)
    return equity

def _compute_bucket_table(
    trades_enriched: pd.DataFrame,
    feature_col: str,
    n_buckets: int = 5,
) -> pd.DataFrame:
    if trades_enriched.empty or feature_col not in trades_enriched.columns:
        return pd.DataFrame()

    working = trades_enriched.copy()
    working = working.dropna(subset=[feature_col])
    if working.empty:
        return pd.DataFrame()

    unique_count = int(working[feature_col].nunique())
    if unique_count < 2:
        return pd.DataFrame()

    q = min(n_buckets, unique_count)
    try:
        working["bucket"] = pd.qcut(working[feature_col], q=q, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for interval, group in working.groupby("bucket", sort=True):
        pnl = group["net_pnl_usd"].astype(float)
        time_exit = group.loc[group["exit_reason"] == "time_exit"]
        row = {
            "feature": feature_col,
            "bucket": str(interval),
            "bucket_lower": float(interval.left),
            "bucket_upper": float(interval.right),
            "n_trades": int(len(group)),
            "win_rate": float((pnl > 0).mean()),
            "avg_pnl": float(pnl.mean()),
            "expectancy": float(pnl.mean()),
            "profit_factor": _profit_factor_from_pnl(pnl),
            "stop_hit_rate": float((group["exit_reason"] == "stop").mean()),
            "target_hit_rate": float((group["exit_reason"] == "target").mean()),
            "time_exit_rate": float((group["exit_reason"] == "time_exit").mean()),
            "time_exit_win_rate": float((time_exit["net_pnl_usd"] > 0).mean()) if len(time_exit) > 0 else 0.0,
            "cumulative_pnl": float(pnl.sum()),
            "time_exit_pnl": float(time_exit["net_pnl_usd"].sum()) if len(time_exit) > 0 else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["bucket_lower", "bucket_upper"]).reset_index(drop=True)


def _variant_filter_mask(candidate_rows: pd.DataFrame, variant: VariantSpec) -> pd.Series:
    values = candidate_rows[variant.column]
    mask = values.notna()
    if variant.lower is not None:
        mask &= values >= variant.lower
    if variant.upper is not None:
        mask &= values <= variant.upper
    if variant.strictly_positive:
        mask &= values > 0.0
    return mask


def _select_first_per_session_after_filter(
    candidate_signal_df: pd.DataFrame,
    pass_mask: pd.Series,
) -> pd.DataFrame:
    out = candidate_signal_df.copy()
    original_signal = out["signal"].copy()
    candidate_rows = out.loc[original_signal.ne(0)].copy()

    out["variant_filter_pass"] = False
    if not candidate_rows.empty:
        aligned_mask = pass_mask.reindex(candidate_rows.index).fillna(False).astype(bool)
        out.loc[candidate_rows.index, "variant_filter_pass"] = aligned_mask.values
        passed = candidate_rows.loc[aligned_mask]
        selected_idx = (
            passed.sort_values("timestamp").groupby("session_date", sort=True).head(1).index
            if not passed.empty
            else pd.Index([])
        )
    else:
        selected_idx = pd.Index([])

    out["signal"] = 0
    if len(selected_idx) > 0:
        out.loc[selected_idx, "signal"] = original_signal.loc[selected_idx].astype(int)
    return out


def _quantile(series: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(q))


def _build_variant_specs_from_diagnostics(trades_enriched: pd.DataFrame) -> list[VariantSpec]:
    specs: list[VariantSpec] = []

    def maybe_add(spec: VariantSpec | None) -> None:
        if spec is not None:
            specs.append(spec)

    def make_range_spec(
        name: str,
        block: str,
        hypothesis: str,
        column: str,
        lower: float | None,
        upper: float | None,
        strictly_positive: bool = False,
    ) -> VariantSpec | None:
        if lower is None or upper is None:
            return None
        if not math.isfinite(lower) or not math.isfinite(upper):
            return None
        if lower >= upper:
            return None
        return VariantSpec(
            name=name,
            block=block,
            hypothesis=hypothesis,
            column=column,
            lower=lower,
            upper=upper,
            strictly_positive=strictly_positive,
        )

    ratio = trades_enriched["or_width_atr_ratio"]
    atr = trades_enriched["atr_14"]
    distance = trades_enriched["price_vwap_distance_atr"]
    slope_atr = trades_enriched["continuous_vwap_slope_5_atr"]

    ratio_q10, ratio_q20, ratio_q30, ratio_q80, ratio_q85, ratio_q90 = (
        _quantile(ratio, 0.10),
        _quantile(ratio, 0.20),
        _quantile(ratio, 0.30),
        _quantile(ratio, 0.80),
        _quantile(ratio, 0.85),
        _quantile(ratio, 0.90),
    )
    atr_q20, atr_q30, atr_q35, atr_q85, atr_q90 = (
        _quantile(atr, 0.20),
        _quantile(atr, 0.30),
        _quantile(atr, 0.35),
        _quantile(atr, 0.85),
        _quantile(atr, 0.90),
    )
    dist_q20, dist_q30, dist_q85, dist_q95 = (
        _quantile(distance, 0.20),
        _quantile(distance, 0.30),
        _quantile(distance, 0.85),
        _quantile(distance, 0.95),
    )
    slope_atr_q40, slope_atr_q50, slope_atr_q95 = (
        _quantile(slope_atr, 0.40),
        _quantile(slope_atr, 0.50),
        _quantile(slope_atr, 0.95),
    )

    maybe_add(
        make_range_spec(
            name="a_ratio_q10_q90",
            block="A_or_width_over_atr",
            hypothesis="OR width/ATR too extreme degrades follow-through quality.",
            column="or_width_atr_ratio",
            lower=ratio_q10,
            upper=ratio_q90,
        )
    )
    maybe_add(
        make_range_spec(
            name="a_ratio_q20_q80",
            block="A_or_width_over_atr",
            hypothesis="Mid-range OR width/ATR may avoid both compression and blow-off opens.",
            column="or_width_atr_ratio",
            lower=ratio_q20,
            upper=ratio_q80,
        )
    )
    maybe_add(
        make_range_spec(
            name="a_ratio_q30_q85",
            block="A_or_width_over_atr",
            hypothesis="Tighter OR width/ATR window can improve risk-adjusted entries.",
            column="or_width_atr_ratio",
            lower=ratio_q30,
            upper=ratio_q85,
        )
    )

    maybe_add(
        make_range_spec(
            name="b_atr_q20_q90",
            block="B_atr_regime",
            hypothesis="Avoid low-vol and extreme-vol ATR tails at signal time.",
            column="atr_14",
            lower=atr_q20,
            upper=atr_q90,
        )
    )
    if atr_q30 is not None and math.isfinite(atr_q30):
        specs.append(
            VariantSpec(
                name="b_atr_ge_q30",
                block="B_atr_regime",
                hypothesis="Require at least moderate ATR regime for expansion potential.",
                column="atr_14",
                lower=atr_q30,
                upper=None,
            )
        )
    maybe_add(
        make_range_spec(
            name="b_atr_q35_q85",
            block="B_atr_regime",
            hypothesis="Conservative bounded ATR regime around the center of distribution.",
            column="atr_14",
            lower=atr_q35,
            upper=atr_q85,
        )
    )

    specs.append(
        VariantSpec(
            name="c_vwap_slope_pos",
            block="C_breakout_structure",
            hypothesis="Long breakouts are stronger when continuous VWAP slope is positive.",
            column="continuous_vwap_slope_5",
            strictly_positive=True,
        )
    )
    if slope_atr_q50 is not None and math.isfinite(slope_atr_q50):
        specs.append(
            VariantSpec(
                name="c_vwap_slope_atr_ge_median",
                block="C_breakout_structure",
                hypothesis="Require at least median VWAP slope normalized by ATR.",
                column="continuous_vwap_slope_5_atr",
                lower=slope_atr_q50,
            )
        )
    maybe_add(
        make_range_spec(
            name="c_vwap_slope_atr_q40_q95",
            block="C_breakout_structure",
            hypothesis="Bounded VWAP slope/ATR to avoid flat and overextended slope states.",
            column="continuous_vwap_slope_5_atr",
            lower=slope_atr_q40,
            upper=slope_atr_q95,
        )
    )

    maybe_add(
        make_range_spec(
            name="d_price_vwap_dist_atr_q20_q95",
            block="D_price_vwap_distance",
            hypothesis="Breakout quality deteriorates when price-VWAP distance is too small or too stretched.",
            column="price_vwap_distance_atr",
            lower=dist_q20,
            upper=dist_q95,
            strictly_positive=True,
        )
    )
    maybe_add(
        make_range_spec(
            name="d_price_vwap_dist_atr_q30_q85",
            block="D_price_vwap_distance",
            hypothesis="Centered positive distance from VWAP can stabilize post-breakout path.",
            column="price_vwap_distance_atr",
            lower=dist_q30,
            upper=dist_q85,
            strictly_positive=True,
        )
    )
    if dist_q20 is not None and math.isfinite(dist_q20):
        specs.append(
            VariantSpec(
                name="d_price_vwap_dist_atr_ge_q20",
                block="D_price_vwap_distance",
                hypothesis="Exclude near-VWAP breakouts with low displacement.",
                column="price_vwap_distance_atr",
                lower=dist_q20,
                strictly_positive=True,
            )
        )

    unique_specs: dict[str, VariantSpec] = {}
    for spec in specs:
        unique_specs[spec.name] = spec
    return list(unique_specs.values())


def _candidate_robustness_row(
    row: pd.Series,
    baseline_row: pd.Series,
) -> pd.Series:
    sharpe_rel = _safe_relative_delta(float(row["sharpe_ratio"]), float(baseline_row["sharpe_ratio"]))
    pf_rel = _safe_relative_delta(float(row["profit_factor"]), float(baseline_row["profit_factor"]))
    expectancy_rel = _safe_relative_delta(float(row["expectancy"]), float(baseline_row["expectancy"]))
    dd_rel = _safe_relative_delta(
        abs(float(baseline_row["max_drawdown"])),
        abs(float(row["max_drawdown"])),
    )
    win_rel = _safe_relative_delta(float(row["win_rate"]), float(baseline_row["win_rate"]))

    trade_floor = float(row["n_trades"]) >= 0.55 * float(baseline_row["n_trades"])
    improvements = sum(
        [
            sharpe_rel > 0.03,
            pf_rel > 0.02,
            expectancy_rel > 0.02,
            dd_rel > 0.02,
        ]
    )
    major_degrade = any(
        [
            sharpe_rel < -0.08,
            pf_rel < -0.06,
            expectancy_rel < -0.06,
            dd_rel < -0.10,
        ]
    )
    credible = bool(trade_floor and improvements >= 2 and not major_degrade)

    score = (
        0.32 * sharpe_rel
        + 0.26 * pf_rel
        + 0.22 * expectancy_rel
        + 0.15 * dd_rel
        + 0.05 * win_rel
    )

    return pd.Series(
        {
            "delta_sharpe_pct": 100.0 * sharpe_rel,
            "delta_profit_factor_pct": 100.0 * pf_rel,
            "delta_expectancy_pct": 100.0 * expectancy_rel,
            "drawdown_improvement_pct": 100.0 * dd_rel,
            "delta_win_rate_pct": 100.0 * win_rel,
            "trade_floor_ok": trade_floor,
            "improvement_count": int(improvements),
            "major_degrade": major_degrade,
            "signal_credible_phase3": credible,
            "phase3_robustness_score": float(score),
        }
    )

def _compute_subperiods(all_sessions: list[pd.Timestamp]) -> list[dict[str, object]]:
    if not all_sessions:
        return []
    chunks = np.array_split(np.array(all_sessions, dtype=object), 3)
    periods: list[dict[str, object]] = []
    for i, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        start = pd.Timestamp(chunk[0]).date()
        end = pd.Timestamp(chunk[-1]).date()
        periods.append(
            {
                "label": f"period_{i}",
                "start": start,
                "end": end,
                "sessions": {pd.Timestamp(value).date() for value in chunk},
            }
        )
    return periods


def _metrics_for_session_subset(
    trades: pd.DataFrame,
    session_subset: set,
    initial_capital: float,
) -> dict[str, object]:
    subset_trades = trades.loc[trades["session_date"].isin(session_subset)].copy()
    subset_sessions = sorted(session_subset)
    return compute_metrics(
        subset_trades,
        session_dates=subset_sessions,
        initial_capital=initial_capital,
    )


def _metrics_by_year(
    trades: pd.DataFrame,
    all_sessions: list[pd.Timestamp],
    initial_capital: float,
    candidate_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sessions_by_year: dict[int, set] = {}
    for session in all_sessions:
        sessions_by_year.setdefault(session.year, set()).add(session.date())

    for year in sorted(sessions_by_year):
        metrics = _metrics_for_session_subset(trades, sessions_by_year[year], initial_capital=initial_capital)
        rows.append(
            {
                "candidate": candidate_name,
                "year": year,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _metrics_by_subperiod(
    trades: pd.DataFrame,
    periods: list[dict[str, object]],
    initial_capital: float,
    candidate_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period in periods:
        metrics = _metrics_for_session_subset(
            trades,
            period["sessions"],
            initial_capital=initial_capital,
        )
        rows.append(
            {
                "candidate": candidate_name,
                "period": period["label"],
                "start_date": period["start"],
                "end_date": period["end"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _metrics_is_oos(
    trades: pd.DataFrame,
    all_sessions: list[pd.Timestamp],
    initial_capital: float,
    candidate_name: str,
) -> pd.DataFrame:
    if not all_sessions:
        return pd.DataFrame()
    split_idx = max(1, int(len(all_sessions) * 0.70))
    is_sessions = {session.date() for session in all_sessions[:split_idx]}
    oos_sessions = {session.date() for session in all_sessions[split_idx:]}
    rows = []
    for split_name, sessions in (("in_sample_70pct", is_sessions), ("out_of_sample_30pct", oos_sessions)):
        metrics = _metrics_for_session_subset(trades, sessions, initial_capital=initial_capital)
        rows.append({"candidate": candidate_name, "split": split_name, **metrics})
    return pd.DataFrame(rows)


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_orb_robust_campaign(
    dataset_path: Path,
    output_dir: Path,
    baseline: BaselineSpec | None = None,
) -> dict[str, Path]:
    """Execute the 4-phase disciplined ORB campaign and export all artifacts."""
    ensure_directories()
    baseline = baseline or BaselineSpec()
    dirs = _make_output_dirs(output_dir)

    feature_df = prepare_feature_dataset(dataset_path=dataset_path, baseline=baseline)
    all_sessions = sorted(pd.to_datetime(feature_df["session_date"]).dt.date.unique())

    execution_model = ExecutionModel()

    exact_baseline_strategy = build_strategy_from_baseline(baseline, one_trade_per_day=True)
    baseline_signal_df = exact_baseline_strategy.generate_signals(feature_df)
    baseline_trades = run_backtest_for_signals(
        signal_df=baseline_signal_df,
        baseline=baseline,
        execution_model=execution_model,
    )
    baseline_metrics = compute_metrics(
        baseline_trades,
        signal_df=baseline_signal_df,
        session_dates=all_sessions,
        initial_capital=baseline.account_size_usd,
    )
    baseline_trades_enriched = enrich_trades_with_signal_features(baseline_trades, baseline_signal_df)

    baseline_metrics_row = pd.DataFrame([baseline_metrics])
    baseline_metrics_row.to_csv(dirs["baseline"] / "baseline_metrics.csv", index=False)
    _save_json(dirs["baseline"] / "baseline_metrics.json", baseline_metrics)
    baseline_trades_enriched.to_csv(dirs["baseline"] / "baseline_trades.csv", index=False)
    _selected_signals(baseline_signal_df).to_csv(dirs["baseline"] / "baseline_selected_signals.csv", index=False)
    _save_equity_artifacts(
        trades=baseline_trades,
        initial_capital=baseline.account_size_usd,
        output_dir=dirs["baseline"],
        stem="baseline",
    )

    baseline_summary = "\n".join(
        [
            "# Baseline Summary",
            "",
            f"- Dataset: `{dataset_path.name}`",
            f"- Session coverage: {all_sessions[0]} to {all_sessions[-1]} ({len(all_sessions)} sessions)",
            f"- Config: `{asdict(baseline)}`",
            f"- Trades: {baseline_metrics['n_trades']}",
            f"- Sharpe: {baseline_metrics['sharpe_ratio']:.4f}",
            f"- Profit factor: {baseline_metrics['profit_factor']:.4f}",
            f"- Expectancy: {baseline_metrics['expectancy']:.4f}",
            f"- Max drawdown: {baseline_metrics['max_drawdown']:.2f}",
            f"- Cumulative PnL: {baseline_metrics['cumulative_pnl']:.2f}",
            f"- Time exit rate: {baseline_metrics.get('time_exit_rate', 0.0):.4f}",
            f"- Time exit win rate: {baseline_metrics.get('time_exit_win_rate', 0.0):.4f}",
        ]
    )
    _write_markdown(dirs["baseline"] / "baseline_summary.md", baseline_summary)

    diagnostics_tables: dict[str, pd.DataFrame] = {}
    for feature in [
        "atr_14",
        "or_width",
        "or_width_atr_ratio",
        "price_vwap_distance_atr",
        "continuous_vwap_slope_5",
    ]:
        table = _compute_bucket_table(baseline_trades_enriched, feature_col=feature, n_buckets=5)
        diagnostics_tables[feature] = table
        table.to_csv(dirs["diagnostics"] / f"diagnostic_buckets_{feature}.csv", index=False)

    diagnostics_consolidated = pd.concat(
        [table for table in diagnostics_tables.values() if not table.empty],
        ignore_index=True,
    ) if any(not table.empty for table in diagnostics_tables.values()) else pd.DataFrame()
    diagnostics_consolidated.to_csv(dirs["diagnostics"] / "diagnostic_buckets_consolidated.csv", index=False)

    time_exit_regime = baseline_trades_enriched.copy()
    if not time_exit_regime.empty:
        for feature in ["atr_14", "or_width_atr_ratio", "price_vwap_distance_atr"]:
            table = _compute_bucket_table(time_exit_regime, feature_col=feature, n_buckets=4)
            if table.empty:
                continue
            table["time_exit_edge_proxy"] = table["time_exit_win_rate"] - table["win_rate"]
            table.to_csv(dirs["diagnostics"] / f"time_exit_regime_{feature}.csv", index=False)

    # Variant filters are applied only to exact baseline day-entry signals.
    # This keeps the strategy definition fixed and avoids hidden extra degrees of freedom.
    baseline_signal_rows = baseline_signal_df.loc[baseline_signal_df["signal"].ne(0)].copy()

    variant_specs = _build_variant_specs_from_diagnostics(baseline_trades_enriched)
    variants_rows: list[dict[str, object]] = []
    variants_trades: dict[str, pd.DataFrame] = {}
    variants_signals: dict[str, pd.DataFrame] = {}

    for spec in variant_specs:
        pass_mask = _variant_filter_mask(baseline_signal_rows, spec)
        variant_signal_df = baseline_signal_df.copy()
        variant_signal_df["variant_filter_pass"] = False
        if not baseline_signal_rows.empty:
            aligned_mask = pass_mask.reindex(baseline_signal_rows.index).fillna(False).astype(bool)
            variant_signal_df.loc[baseline_signal_rows.index, "variant_filter_pass"] = aligned_mask.values
            filtered_out_idx = baseline_signal_rows.index[~aligned_mask]
            variant_signal_df.loc[filtered_out_idx, "signal"] = 0
        variant_trades = run_backtest_for_signals(
            signal_df=variant_signal_df,
            baseline=baseline,
            execution_model=execution_model,
        )
        variant_metrics = compute_metrics(
            variant_trades,
            signal_df=variant_signal_df,
            session_dates=all_sessions,
            initial_capital=baseline.account_size_usd,
        )
        variant_trades_enriched = enrich_trades_with_signal_features(variant_trades, variant_signal_df)

        variants_signals[spec.name] = variant_signal_df
        variants_trades[spec.name] = variant_trades_enriched
        variant_trades_enriched.to_csv(dirs["campaigns"] / f"trades_{spec.name}.csv", index=False)

        row = {
            **asdict(spec),
            "candidate_signals_before_filter": int(len(baseline_signal_rows)),
            "candidate_signals_after_filter": int(pass_mask.sum()),
            "selected_signal_days": int(variant_signal_df["signal"].ne(0).sum()),
            **variant_metrics,
        }
        variants_rows.append(row)

    variants_df = pd.DataFrame(variants_rows)
    if variants_df.empty:
        raise RuntimeError("No variant was generated. Diagnostics quantiles collapsed unexpectedly.")

    baseline_series = pd.Series(baseline_metrics)
    robustness = variants_df.apply(lambda row: _candidate_robustness_row(row, baseline_series), axis=1)
    variants_df = pd.concat([variants_df, robustness], axis=1)
    variants_df = variants_df.sort_values(
        by=["signal_credible_phase3", "phase3_robustness_score", "profit_factor", "expectancy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    variants_df.to_csv(dirs["campaigns"] / "variants_consolidated.csv", index=False)

    block_rows: list[dict[str, object]] = []
    for block, block_df in variants_df.groupby("block", sort=True):
        best = block_df.iloc[0]
        conclusion = "signal credible" if bool(best["signal_credible_phase3"]) else "pas convaincant"
        block_rows.append(
            {
                "block": block,
                "best_variant": best["name"],
                "best_phase3_score": float(best["phase3_robustness_score"]),
                "best_profit_factor": float(best["profit_factor"]),
                "best_sharpe": float(best["sharpe_ratio"]),
                "best_expectancy": float(best["expectancy"]),
                "best_max_drawdown": float(best["max_drawdown"]),
                "best_n_trades": int(best["n_trades"]),
                "conclusion": conclusion,
            }
        )
    block_summary_df = pd.DataFrame(block_rows).sort_values("block")
    block_summary_df.to_csv(dirs["campaigns"] / "block_summaries.csv", index=False)

    promising = variants_df.loc[variants_df["signal_credible_phase3"]].copy()
    if promising.empty:
        promising = variants_df.head(2).copy()
    else:
        promising = promising.head(3).copy()
    promising.to_csv(dirs["campaigns"] / "promising_candidates.csv", index=False)

    periods = _compute_subperiods(all_sessions)
    finalists_overall_rows: list[dict[str, object]] = []
    finalists_year_rows: list[pd.DataFrame] = []
    finalists_period_rows: list[pd.DataFrame] = []
    finalists_is_oos_rows: list[pd.DataFrame] = []

    finalists = ["baseline_exact"] + promising["name"].tolist()
    finalists_trades = {"baseline_exact": baseline_trades_enriched, **variants_trades}
    finalists_signals = {"baseline_exact": baseline_signal_df, **variants_signals}

    for name in finalists:
        candidate_trades = finalists_trades[name]
        candidate_signal_df_view = finalists_signals[name]
        overall_metrics = compute_metrics(
            candidate_trades,
            signal_df=candidate_signal_df_view,
            session_dates=all_sessions,
            initial_capital=baseline.account_size_usd,
        )
        finalists_overall_rows.append({"candidate": name, **overall_metrics})

        by_year = _metrics_by_year(
            trades=candidate_trades,
            all_sessions=[pd.Timestamp(session) for session in all_sessions],
            initial_capital=baseline.account_size_usd,
            candidate_name=name,
        )
        by_period = _metrics_by_subperiod(
            trades=candidate_trades,
            periods=periods,
            initial_capital=baseline.account_size_usd,
            candidate_name=name,
        )
        is_oos = _metrics_is_oos(
            trades=candidate_trades,
            all_sessions=[pd.Timestamp(session) for session in all_sessions],
            initial_capital=baseline.account_size_usd,
            candidate_name=name,
        )

        finalists_year_rows.append(by_year)
        finalists_period_rows.append(by_period)
        finalists_is_oos_rows.append(is_oos)

        candidate_dir = dirs["finalists"] / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_trades.to_csv(candidate_dir / "trades.csv", index=False)
        pd.DataFrame([overall_metrics]).to_csv(candidate_dir / "overall_metrics.csv", index=False)
        _save_json(candidate_dir / "overall_metrics.json", overall_metrics)
        by_year.to_csv(candidate_dir / "metrics_by_year.csv", index=False)
        by_period.to_csv(candidate_dir / "metrics_by_subperiod.csv", index=False)
        is_oos.to_csv(candidate_dir / "metrics_is_oos.csv", index=False)
        _save_equity_artifacts(
            trades=candidate_trades,
            initial_capital=baseline.account_size_usd,
            output_dir=candidate_dir,
            stem=name,
        )

    finalists_overall_df = pd.DataFrame(finalists_overall_rows).sort_values("candidate")
    finalists_year_df = pd.concat(finalists_year_rows, ignore_index=True) if finalists_year_rows else pd.DataFrame()
    finalists_period_df = (
        pd.concat(finalists_period_rows, ignore_index=True) if finalists_period_rows else pd.DataFrame()
    )
    finalists_is_oos_df = pd.concat(finalists_is_oos_rows, ignore_index=True) if finalists_is_oos_rows else pd.DataFrame()

    finalists_overall_df.to_csv(dirs["finalists"] / "finalists_overall_metrics.csv", index=False)
    finalists_year_df.to_csv(dirs["finalists"] / "finalists_metrics_by_year.csv", index=False)
    finalists_period_df.to_csv(dirs["finalists"] / "finalists_metrics_by_subperiod.csv", index=False)
    finalists_is_oos_df.to_csv(dirs["finalists"] / "finalists_metrics_is_oos.csv", index=False)

    baseline_overall = finalists_overall_df.loc[finalists_overall_df["candidate"] == "baseline_exact"].iloc[0]
    baseline_oos = finalists_is_oos_df.loc[
        (finalists_is_oos_df["candidate"] == "baseline_exact")
        & (finalists_is_oos_df["split"] == "out_of_sample_30pct")
    ]
    baseline_oos_row = baseline_oos.iloc[0] if not baseline_oos.empty else baseline_overall

    recommendation_rows: list[dict[str, object]] = []
    for _, row in finalists_overall_df.iterrows():
        if row["candidate"] == "baseline_exact":
            continue
        candidate_name = str(row["candidate"])
        candidate_oos_df = finalists_is_oos_df.loc[
            (finalists_is_oos_df["candidate"] == candidate_name)
            & (finalists_is_oos_df["split"] == "out_of_sample_30pct")
        ]
        candidate_oos = candidate_oos_df.iloc[0] if not candidate_oos_df.empty else row

        improves_overall = sum(
            [
                float(row["sharpe_ratio"]) > float(baseline_overall["sharpe_ratio"]),
                float(row["profit_factor"]) > float(baseline_overall["profit_factor"]),
                float(row["expectancy"]) > float(baseline_overall["expectancy"]),
                abs(float(row["max_drawdown"])) < abs(float(baseline_overall["max_drawdown"])),
            ]
        )
        oos_ok = sum(
            [
                float(candidate_oos["sharpe_ratio"]) >= 1.02 * float(baseline_oos_row["sharpe_ratio"]),
                float(candidate_oos["profit_factor"]) >= 1.02 * float(baseline_oos_row["profit_factor"]),
                abs(float(candidate_oos["max_drawdown"])) <= 1.05 * abs(float(baseline_oos_row["max_drawdown"])),
            ]
        ) >= 2
        stable_years = finalists_year_df.loc[finalists_year_df["candidate"] == candidate_name].copy()
        baseline_years = finalists_year_df.loc[finalists_year_df["candidate"] == "baseline_exact"].copy()
        joined_years = stable_years.merge(
            baseline_years[["year", "expectancy", "profit_factor"]],
            on="year",
            how="left",
            suffixes=("_cand", "_base"),
        )
        if joined_years.empty:
            year_stability = False
        else:
            year_stability = bool(
                (
                    (joined_years["expectancy_cand"] >= joined_years["expectancy_base"])
                    | (joined_years["profit_factor_cand"] >= joined_years["profit_factor_base"])
                ).mean()
                >= 0.50
            )

        beats_baseline = bool(improves_overall >= 2 and oos_ok and year_stability)
        recommendation_rows.append(
            {
                "candidate": candidate_name,
                "improves_overall_metric_count": int(improves_overall),
                "oos_check_passed": bool(oos_ok),
                "year_stability_passed": bool(year_stability),
                "beats_baseline": beats_baseline,
                "overall_sharpe": float(row["sharpe_ratio"]),
                "overall_profit_factor": float(row["profit_factor"]),
                "overall_expectancy": float(row["expectancy"]),
                "overall_abs_drawdown": abs(float(row["max_drawdown"])),
            }
        )

    recommendation_df = pd.DataFrame(recommendation_rows).sort_values(
        by=[
            "beats_baseline",
            "improves_overall_metric_count",
            "oos_check_passed",
            "year_stability_passed",
            "overall_sharpe",
            "overall_profit_factor",
            "overall_expectancy",
            "overall_abs_drawdown",
        ],
        ascending=[False, False, False, False, False, False, False, True],
    )
    recommendation_df.to_csv(dirs["summary"] / "final_recommendation_table.csv", index=False)

    best_candidate_line = "No candidate produced enough robust evidence to beat the baseline."
    if not recommendation_df.empty:
        top = recommendation_df.iloc[0]
        if bool(top["beats_baseline"]):
            best_candidate_line = f"{top['candidate']} shows robust evidence of beating the baseline."
        else:
            best_candidate_line = (
                f"Top candidate by validation checks: {top['candidate']}, but evidence is insufficient to declare it superior."
            )

    phase3_view = variants_df[
        [
            "name",
            "block",
            "n_trades",
            "profit_factor",
            "sharpe_ratio",
            "expectancy",
            "max_drawdown",
            "phase3_robustness_score",
            "signal_credible_phase3",
        ]
    ].copy()
    phase3_text = phase3_view.head(10).to_string(index=False)

    report_lines = [
        "# ORB Robust Campaign Report",
        "",
        "## Baseline",
        "",
        f"- Dataset: `{dataset_path.name}`",
        f"- Baseline config: `{asdict(baseline)}`",
        f"- Sessions: {all_sessions[0]} to {all_sessions[-1]} ({len(all_sessions)} sessions)",
        f"- Baseline trades: {int(baseline_metrics['n_trades'])}",
        f"- Baseline Sharpe: {baseline_metrics['sharpe_ratio']:.4f}",
        f"- Baseline profit factor: {baseline_metrics['profit_factor']:.4f}",
        f"- Baseline expectancy: {baseline_metrics['expectancy']:.4f}",
        f"- Baseline max drawdown: {baseline_metrics['max_drawdown']:.2f}",
        f"- Baseline cumulative PnL: {baseline_metrics['cumulative_pnl']:.2f}",
        "",
        "## Tested Hypotheses",
        "",
        "- A: OR width normalized by ATR can filter weak or overextended opening regimes.",
        "- B: ATR regime bounds at signal time can reduce poor volatility conditions.",
        "- C: Breakout structure quality improves when continuous VWAP slope confirms long direction.",
        "- D: Price-to-VWAP distance normalized by ATR helps avoid weak/overstretched entries.",
        "",
        "## Phase 3 Snapshot",
        "",
        "```text",
        phase3_text,
        "```",
        "",
        "## Block Conclusions",
        "",
    ]
    for _, row in block_summary_df.iterrows():
        report_lines.append(
            f"- {row['block']}: best={row['best_variant']} -> {row['conclusion']} "
            f"(score={row['best_phase3_score']:.4f}, PF={row['best_profit_factor']:.3f}, "
            f"Sharpe={row['best_sharpe']:.3f}, DD={row['best_max_drawdown']:.2f})"
        )

    report_lines.extend(
        [
            "",
            "## Final Validation",
            "",
            best_candidate_line,
            "",
            "## Recommendation",
            "",
            "- Promote a candidate only if multi-metric improvements are preserved out-of-sample and not concentrated in one year.",
            "- If no candidate passes those checks, keep the baseline and continue with similarly parsimonious regime filters.",
        ]
    )
    _write_markdown(dirs["summary"] / "final_campaign_report.md", "\n".join(report_lines))

    return {
        "output_root": dirs["root"],
        "baseline_summary": dirs["baseline"] / "baseline_summary.md",
        "diagnostics_consolidated": dirs["diagnostics"] / "diagnostic_buckets_consolidated.csv",
        "variants_table": dirs["campaigns"] / "variants_consolidated.csv",
        "finalists_overall": dirs["finalists"] / "finalists_overall_metrics.csv",
        "final_report": dirs["summary"] / "final_campaign_report.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the disciplined ORB robust campaign.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_CAMPAIGN_DATASET,
        help="Path to source OHLCV dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to data/exports/orb_robust_campaign_<timestamp>.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"orb_robust_campaign_{timestamp}")
    artifacts = run_orb_robust_campaign(dataset_path=args.dataset, output_dir=output_dir)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
