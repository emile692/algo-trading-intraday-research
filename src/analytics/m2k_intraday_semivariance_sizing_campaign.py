"""M2K-only realized semivariance sizing follow-up campaign."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.metrics import compute_metrics
from src.analytics.orb_multi_asset_campaign import (
    BaselineSpec,
    SearchGrid,
    SymbolAnalysis,
    analyze_symbol,
    analyze_symbol_cache_pass_matrix,
    resolve_aggregation_threshold,
    resolve_processed_dataset,
)
from src.config.paths import EXPORTS_DIR, ensure_directories
from src.features.semivariance import DEFAULT_EPS, add_realized_semivariance_features, add_rolling_percentile_ranks
from src.features.volatility import add_atr


SYMBOL = "M2K"
HORIZONS = ("30m", "60m", "90m", "session")
WEEKDAY_LABELS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
}
CONTEXT_COLUMNS = {
    "wide_or": "or_width_pct",
    "opening_gap": "opening_gap_pct",
    "high_atr": "atr_pct",
}


@dataclass(frozen=True)
class M2KBaselineConfig:
    symbol: str
    source_reference: str
    source_note: str
    baseline: BaselineSpec
    grid: SearchGrid
    aggregation_rule: str
    timeframe: str = "1m"
    dataset_path: Path | None = None


@dataclass(frozen=True)
class M2KSemivarianceSizingSpec:
    is_fraction: float = 0.70
    semivariance_horizons: tuple[str, ...] = HORIZONS
    downside_feature_keys: tuple[str, ...] = ("rs_minus_pct", "rs_minus_share_pct", "rs_ratio_pct")
    downside_thresholds: tuple[float, ...] = (0.85, 0.90, 0.95)
    downside_multipliers: tuple[float, ...] = (0.75, 0.50)
    three_state_pairs: tuple[tuple[float, float], ...] = ((1.10, 0.75), (1.25, 0.50))
    three_state_low_threshold: float = 0.25
    context_keys: tuple[str, ...] = ("wide_or", "opening_gap", "high_atr")
    conditional_feature_keys: tuple[str, ...] = ("rs_minus_share_pct", "rs_ratio_pct")
    reference_skip_feature_keys: tuple[str, ...] = ("rs_minus_pct",)
    reference_skip_horizons: tuple[str, ...] = ("60m", "session")
    reference_skip_thresholds: tuple[float, ...] = (0.90, 0.95)
    percentile_history: int = 252
    min_percentile_history: int = 60
    min_trade_retention: float = 0.70
    min_is_trade_count: int = 60
    output_root: Path | None = None
    baseline_config: M2KBaselineConfig | None = None


@dataclass
class M2KVariantRun:
    name: str
    family: str
    description: str
    feature_key: str
    horizon: str
    context_key: str | None
    parameters: dict[str, Any]
    controls: pd.DataFrame
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    summary_by_scope: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class M2KCampaignArtifacts:
    output_dir: Path
    spec: M2KSemivarianceSizingSpec
    baseline_config: M2KBaselineConfig
    analysis: SymbolAnalysis
    selected_sessions: set
    trade_features: pd.DataFrame
    baseline_run: M2KVariantRun
    variant_runs: dict[str, M2KVariantRun]
    variant_results: pd.DataFrame
    promotion_candidates: pd.DataFrame
    final_verdict: dict[str, Any]


def _default_baseline_config() -> M2KBaselineConfig:
    return M2KBaselineConfig(
        symbol=SYMBOL,
        source_reference=str(ROOT / "notebooks" / "orb_M2K_final_ensemble_validation.ipynb"),
        source_note="Final audited leak-free M2K ensemble validation notebook baseline.",
        baseline=BaselineSpec(
            or_minutes=15,
            opening_time="09:30:00",
            direction="long",
            one_trade_per_day=True,
            entry_buffer_ticks=2,
            stop_buffer_ticks=2,
            target_multiple=2.0,
            vwap_confirmation=True,
            vwap_column="continuous_session_vwap",
            time_exit="16:00:00",
            account_size_usd=50_000.0,
            risk_per_trade_pct=1.5,
            entry_on_next_open=True,
        ),
        grid=SearchGrid(
            atr_periods=(25, 26, 27, 28, 29, 30),
            q_lows_pct=(25, 26, 27, 28, 29, 30),
            q_highs_pct=(90, 91, 92, 93, 94, 95),
            aggregation_rules=("majority_50", "consensus_75", "unanimity_100"),
        ),
        aggregation_rule="majority_50",
    )


def default_campaign_spec(*, output_root: Path | None = None) -> M2KSemivarianceSizingSpec:
    return M2KSemivarianceSizingSpec(output_root=output_root, baseline_config=_default_baseline_config())


def smoke_campaign_spec(*, output_root: Path | None = None) -> M2KSemivarianceSizingSpec:
    spec = default_campaign_spec(output_root=output_root)
    return M2KSemivarianceSizingSpec(
        is_fraction=spec.is_fraction,
        semivariance_horizons=("30m", "session"),
        downside_feature_keys=("rs_minus_share_pct", "rs_ratio_pct"),
        downside_thresholds=(0.85, 0.90),
        downside_multipliers=(0.75,),
        three_state_pairs=((1.10, 0.75),),
        three_state_low_threshold=0.25,
        context_keys=("wide_or", "opening_gap"),
        conditional_feature_keys=("rs_minus_share_pct",),
        reference_skip_feature_keys=("rs_minus_pct",),
        reference_skip_horizons=("session",),
        reference_skip_thresholds=(0.90,),
        percentile_history=40,
        min_percentile_history=10,
        min_trade_retention=0.60,
        min_is_trade_count=10,
        output_root=spec.output_root,
        baseline_config=spec.baseline_config,
    )


def _serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_serialize_value(payload), indent=2, sort_keys=True), encoding="utf-8")


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(float(denominator)) or abs(float(denominator)) <= DEFAULT_EPS:
        return float(default)
    return float(numerator) / float(denominator)


def _safe_series_div(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    *,
    eps: float = DEFAULT_EPS,
) -> pd.Series:
    num = pd.to_numeric(pd.Series(numerator), errors="coerce")
    den = pd.to_numeric(pd.Series(denominator), errors="coerce").fillna(0.0).clip(lower=float(eps))
    return num / den


def _subset_by_sessions(frame: pd.DataFrame, sessions: list | set) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out["session_date"].isin(session_set)].copy()


def _selected_ensemble_sessions(analysis: SymbolAnalysis, aggregation_rule: str) -> set:
    point_pass = analyze_symbol_cache_pass_matrix(analysis)
    pass_cols = [column for column in point_pass.columns if column.startswith("pass__")]
    if not pass_cols:
        return set()
    scored = point_pass.copy()
    scored["consensus_score"] = scored[pass_cols].sum(axis=1) / float(len(pass_cols))
    threshold = resolve_aggregation_threshold(aggregation_rule)
    return set(pd.to_datetime(scored.loc[scored["consensus_score"] >= threshold, "session_date"]).dt.date)


def _daily_results_from_trades(trades: pd.DataFrame, sessions: list, initial_capital: float) -> pd.DataFrame:
    session_index = pd.Index(pd.to_datetime(pd.Index(sessions)).date)
    daily = pd.DataFrame({"session_date": session_index})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
        daily["daily_trade_count"] = 0
        daily["daily_loss_count"] = 0
    else:
        view = trades.copy()
        view["session_date"] = pd.to_datetime(view["session_date"]).dt.date
        view["loss_trade"] = pd.to_numeric(view["net_pnl_usd"], errors="coerce").lt(0)
        grouped = (
            view.groupby("session_date", as_index=False)
            .agg(
                daily_pnl_usd=("net_pnl_usd", "sum"),
                daily_trade_count=("trade_id", "count"),
                daily_loss_count=("loss_trade", "sum"),
            )
        )
        daily = daily.merge(grouped, on="session_date", how="left").fillna(
            {"daily_pnl_usd": 0.0, "daily_trade_count": 0, "daily_loss_count": 0}
        )

    daily = daily.sort_values("session_date").reset_index(drop=True)
    daily["equity"] = float(initial_capital) + pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    return daily


def _negative_streak_lengths(values: pd.Series) -> list[int]:
    streaks: list[int] = []
    current = 0
    for value in pd.Series(values, dtype=float).fillna(0.0):
        if value < 0:
            current += 1
        elif current > 0:
            streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def _sortino_ratio(daily_pnl: pd.Series, capital: float) -> float:
    if capital <= 0:
        return 0.0
    returns = pd.Series(daily_pnl, dtype=float) / capital
    downside = returns[returns < 0]
    if len(returns) < 2 or downside.empty:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std <= 0:
        return 0.0
    return float((returns.mean() / downside_std) * math.sqrt(252.0))


def _trades_per_month(n_trades: int, n_sessions: int, trading_days_per_month: float = 21.0) -> float:
    months = float(n_sessions) / float(trading_days_per_month) if n_sessions > 0 else 0.0
    if months <= 0:
        return 0.0
    return float(n_trades) / months


def _summarize_scope(trades: pd.DataFrame, sessions: list, *, initial_capital: float) -> dict[str, Any]:
    metrics = compute_metrics(trades, session_dates=sessions, initial_capital=initial_capital)
    daily = _daily_results_from_trades(trades, sessions, initial_capital=initial_capital)
    daily_pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    n_days_traded = int((pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0.0) > 0).sum())
    trade_count = int(metrics.get("n_trades", 0))
    return {
        "net_pnl": float(metrics.get("cumulative_pnl", 0.0)),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "sortino": _sortino_ratio(daily_pnl, initial_capital),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "win_rate": float(metrics.get("win_rate", 0.0)),
        "average_trade": float(metrics.get("expectancy", 0.0)),
        "trade_count": trade_count,
        "n_days_traded": n_days_traded,
        "participation_rate": float(metrics.get("percent_of_days_traded", 0.0)),
        "trades_per_month": _trades_per_month(trade_count, len(sessions)),
        "longest_losing_streak_trade": int(metrics.get("longest_loss_streak", 0)),
        "longest_losing_streak_daily": int(max(_negative_streak_lengths(daily_pnl), default=0)),
    }


def _build_summary_by_scope(
    trades: pd.DataFrame,
    *,
    all_sessions: list,
    is_sessions: list,
    oos_sessions: list,
    initial_capital: float,
) -> pd.DataFrame:
    rows = []
    for scope, sessions in (("overall", all_sessions), ("is", is_sessions), ("oos", oos_sessions)):
        rows.append(
            {
                "scope": scope,
                **_summarize_scope(_subset_by_sessions(trades, sessions), sessions, initial_capital=initial_capital),
            }
        )
    return pd.DataFrame(rows)


def _scope_value(summary_by_scope: pd.DataFrame, scope: str, column: str) -> Any:
    row = summary_by_scope.loc[summary_by_scope["scope"] == scope]
    if row.empty:
        return np.nan
    return row.iloc[0].get(column, np.nan)


def _build_session_reference_features(frame: pd.DataFrame, opening_time: str, time_exit: str) -> pd.DataFrame:
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"]).copy()
    working["session_date"] = pd.to_datetime(working["session_date"]).dt.date
    if "continuous_session_date" in working.columns:
        working["continuous_session_key"] = pd.to_datetime(working["continuous_session_date"]).dt.date
    else:
        working["continuous_session_key"] = working["session_date"]

    open_ts = pd.Timestamp(opening_time)
    close_ts = pd.Timestamp(time_exit)
    open_minutes = int(open_ts.hour * 60 + open_ts.minute)
    close_minutes = int(close_ts.hour * 60 + close_ts.minute)
    working["minute_of_day"] = working["timestamp"].dt.hour * 60 + working["timestamp"].dt.minute
    working["bar_date"] = working["timestamp"].dt.date

    rth = working.loc[working["minute_of_day"].between(open_minutes, close_minutes, inclusive="both")].copy()
    if rth.empty:
        return pd.DataFrame(
            columns=["session_date", "rth_open", "rth_close", "prev_rth_close", "atr_20_open", "overnight_range_pts"]
        )

    rth = rth.sort_values("timestamp")
    rth_open = (
        rth.groupby("session_date", sort=True)
        .first()[["open", "atr_20"]]
        .rename(columns={"open": "rth_open", "atr_20": "atr_20_open"})
    )
    rth_close = rth.groupby("session_date", sort=True).last()[["close"]].rename(columns={"close": "rth_close"})
    references = rth_open.join(rth_close, how="outer").reset_index()
    references["prev_rth_close"] = pd.to_numeric(references["rth_close"], errors="coerce").shift(1)

    overnight_mask = (working["bar_date"] < working["continuous_session_key"]) | (working["minute_of_day"] < open_minutes)
    overnight = (
        working.loc[overnight_mask]
        .groupby("continuous_session_key", sort=True)
        .agg(overnight_high=("high", "max"), overnight_low=("low", "min"))
        .reset_index()
        .rename(columns={"continuous_session_key": "session_date"})
    )
    overnight["overnight_range_pts"] = pd.to_numeric(overnight["overnight_high"], errors="coerce") - pd.to_numeric(
        overnight["overnight_low"], errors="coerce"
    )
    return references.merge(overnight[["session_date", "overnight_range_pts"]], on="session_date", how="left")


def _build_trade_feature_frame(
    analysis: SymbolAnalysis,
    *,
    selected_sessions: set,
    percentile_history: int,
    min_percentile_history: int,
    horizons: tuple[str, ...],
) -> pd.DataFrame:
    signal_enriched = add_atr(analysis.signal_df.copy(), window=20)
    semivar_signal = add_realized_semivariance_features(
        signal_enriched,
        session_open_time=analysis.baseline.opening_time,
        rth_end_time=analysis.baseline.time_exit,
        window_minutes=(30, 60, 90),
    )
    references = _build_session_reference_features(
        semivar_signal,
        opening_time=analysis.baseline.opening_time,
        time_exit=analysis.baseline.time_exit,
    )

    selected_candidates = analysis.candidate_df.copy()
    selected_candidates["session_date"] = pd.to_datetime(selected_candidates["session_date"]).dt.date
    selected_candidates = selected_candidates.loc[selected_candidates["session_date"].isin(selected_sessions)].copy()

    signal_rows = semivar_signal.loc[selected_candidates["signal_index"].tolist()].copy()
    signal_rows = signal_rows.reset_index().rename(columns={"index": "signal_index"})
    signal_rows["session_date"] = pd.to_datetime(signal_rows["session_date"]).dt.date

    selected_trades = _subset_by_sessions(analysis.baseline_trades, selected_sessions)
    trade_features = selected_candidates.merge(
        signal_rows,
        on=["session_date", "signal_index"],
        how="inner",
        suffixes=("", "_signal"),
    )
    trade_features = trade_features.merge(references, on="session_date", how="left")
    trade_features = trade_features.merge(
        selected_trades[
            [
                "session_date",
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "quantity",
                "net_pnl_usd",
                "pnl_usd",
                "fees",
                "exit_reason",
                "risk_per_contract_usd",
                "pnl_ticks",
                "entry_price",
                "trade_risk_usd",
            ]
        ],
        on="session_date",
        how="inner",
    )
    trade_features = trade_features.sort_values(["session_date", "timestamp"]).reset_index(drop=True)
    is_set = set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)
    trade_features["phase"] = np.where(trade_features["session_date"].isin(is_set), "is", "oos")
    trade_features["breakout_side"] = pd.Series(trade_features["direction"], dtype="string").str.lower()
    trade_features["weekday_name"] = pd.to_numeric(trade_features["weekday"], errors="coerce").map(WEEKDAY_LABELS)
    trade_features["or_width_pts"] = pd.to_numeric(trade_features["or_width"], errors="coerce")
    trade_features["atr_20_signal"] = pd.to_numeric(trade_features["atr_20"], errors="coerce")
    trade_features["opening_gap_abs_atr20"] = _safe_series_div(
        (pd.to_numeric(trade_features["rth_open"], errors="coerce") - pd.to_numeric(trade_features["prev_rth_close"], errors="coerce")).abs(),
        pd.to_numeric(trade_features["atr_20_open"], errors="coerce"),
    )

    pct_columns: list[str] = ["opening_gap_abs_atr20", "or_width_pts", "atr_20_signal"]
    for horizon in horizons:
        trade_features[f"rs_ratio_{horizon}"] = _safe_series_div(
            pd.to_numeric(trade_features[f"rs_minus_{horizon}"], errors="coerce"),
            pd.to_numeric(trade_features[f"rs_plus_{horizon}"], errors="coerce"),
        )
        pct_columns.extend(
            [
                f"rs_minus_{horizon}",
                f"rs_plus_{horizon}",
                f"rv_{horizon}",
                f"rs_minus_share_{horizon}",
                f"rs_ratio_{horizon}",
            ]
        )

    trade_features = add_rolling_percentile_ranks(
        trade_features,
        columns=tuple(pct_columns),
        lookback=percentile_history,
        min_history=min_percentile_history,
    )
    rename_map = {
        "opening_gap_abs_atr20_pct": "opening_gap_pct",
        "or_width_pts_pct": "or_width_pct",
        "atr_20_signal_pct": "atr_pct",
    }
    for horizon in horizons:
        rename_map[f"rs_minus_{horizon}_pct"] = f"rs_minus_pct_{horizon}"
        rename_map[f"rs_plus_{horizon}_pct"] = f"rs_plus_pct_{horizon}"
        rename_map[f"rv_{horizon}_pct"] = f"rv_pct_{horizon}"
        rename_map[f"rs_minus_share_{horizon}_pct"] = f"rs_minus_share_pct_{horizon}"
        rename_map[f"rs_ratio_{horizon}_pct"] = f"rs_ratio_pct_{horizon}"
    trade_features = trade_features.rename(columns=rename_map)
    return trade_features


def _feature_column(feature_key: str, horizon: str) -> str:
    return f"{feature_key}_{horizon}"


def _context_column(context_key: str) -> str:
    if context_key not in CONTEXT_COLUMNS:
        raise ValueError(f"Unsupported context key '{context_key}'.")
    return CONTEXT_COLUMNS[context_key]


def _variant_name(
    family: str,
    feature_key: str,
    horizon: str,
    threshold: float,
    *,
    down_multiplier: float | None = None,
    up_multiplier: float | None = None,
    low_threshold: float | None = None,
    context_key: str | None = None,
) -> str:
    parts = [family, feature_key, horizon, f"t{int(round(threshold * 100.0)):02d}"]
    if low_threshold is not None:
        parts.append(f"lo{int(round(low_threshold * 100.0)):02d}")
    if up_multiplier is not None:
        parts.append(f"u{int(round(up_multiplier * 100.0)):03d}")
    if down_multiplier is not None:
        parts.append(f"d{int(round(down_multiplier * 100.0)):02d}")
    if context_key is not None:
        parts.append(context_key)
    return "__".join(parts)


def _apply_trade_multipliers(
    nominal_trades: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    account_size_usd: float,
    base_risk_pct: float,
    tick_value_usd: float,
    point_value_usd: float,
    commission_per_side_usd: float,
) -> pd.DataFrame:
    if nominal_trades.empty or controls.empty:
        return pd.DataFrame(columns=nominal_trades.columns)

    trades = nominal_trades.copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"]).dt.date
    control_map = dict(zip(pd.to_datetime(controls["session_date"]).dt.date, controls["risk_multiplier"]))
    trades["risk_multiplier"] = trades["session_date"].map(control_map).fillna(0.0).astype(float)
    trades = trades.loc[trades["risk_multiplier"] > 0.0].copy()
    if trades.empty:
        return trades

    risk_budget_base = float(account_size_usd) * float(base_risk_pct) / 100.0
    per_contract_risk = pd.to_numeric(trades["risk_per_contract_usd"], errors="coerce")
    scaled_budget = risk_budget_base * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["quantity"] = np.floor(scaled_budget / per_contract_risk).astype(int)
    trades = trades.loc[trades["quantity"] >= 1].copy()
    if trades.empty:
        return trades

    fees_per_contract = 2.0 * float(commission_per_side_usd)
    quantity = pd.to_numeric(trades["quantity"], errors="coerce")
    trades["risk_per_trade_pct"] = float(base_risk_pct) * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["risk_budget_usd"] = risk_budget_base * pd.to_numeric(trades["risk_multiplier"], errors="coerce")
    trades["actual_risk_usd"] = quantity * per_contract_risk.loc[trades.index]
    trades["trade_risk_usd"] = trades["actual_risk_usd"]
    trades["pnl_usd"] = pd.to_numeric(trades["pnl_ticks"], errors="coerce") * float(tick_value_usd) * quantity
    trades["fees"] = fees_per_contract * quantity
    trades["net_pnl_usd"] = pd.to_numeric(trades["pnl_usd"], errors="coerce") - pd.to_numeric(trades["fees"], errors="coerce")
    trades["notional_usd"] = pd.to_numeric(trades["entry_price"], errors="coerce") * float(point_value_usd) * quantity
    trades["leverage_used"] = trades["notional_usd"] / float(account_size_usd)
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    trades["trade_id"] = np.arange(1, len(trades) + 1)
    return trades


def _build_variant_controls(
    trade_features: pd.DataFrame,
    *,
    family: str,
    feature_key: str,
    horizon: str,
    threshold: float,
    down_multiplier: float | None = None,
    up_multiplier: float | None = None,
    low_threshold: float | None = None,
    context_key: str | None = None,
) -> pd.DataFrame:
    feature_col = _feature_column(feature_key, horizon)
    if feature_col not in trade_features.columns:
        raise ValueError(f"Missing feature column '{feature_col}'.")

    columns = [
        "session_date",
        "phase",
        "trade_id",
        "weekday_name",
        feature_col,
        "opening_gap_pct",
        "or_width_pct",
        "atr_pct",
    ]
    controls = trade_features[columns].copy()
    controls = controls.rename(columns={feature_col: "feature_pct"})
    controls["family"] = family
    controls["feature_key"] = feature_key
    controls["horizon"] = horizon
    controls["high_threshold"] = float(threshold)
    controls["low_threshold"] = np.nan if low_threshold is None else float(low_threshold)
    controls["down_multiplier"] = np.nan if down_multiplier is None else float(down_multiplier)
    controls["up_multiplier"] = np.nan if up_multiplier is None else float(up_multiplier)
    controls["context_key"] = context_key

    downside_high = pd.to_numeric(controls["feature_pct"], errors="coerce") >= float(threshold)
    if low_threshold is not None:
        downside_low = pd.to_numeric(controls["feature_pct"], errors="coerce") <= float(low_threshold)
    else:
        downside_low = pd.Series(False, index=controls.index)
    if context_key is None:
        context_pct = pd.Series(np.nan, index=controls.index, dtype=float)
        context_confirm = pd.Series(True, index=controls.index, dtype=bool)
    else:
        context_pct = pd.to_numeric(controls[_context_column(context_key)], errors="coerce")
        context_confirm = context_pct >= float(threshold)

    missing_context = pd.to_numeric(controls["feature_pct"], errors="coerce").isna()
    if context_key is not None:
        missing_context = missing_context | context_pct.isna()

    controls["downside_high"] = pd.Series(downside_high, index=controls.index).fillna(False)
    controls["downside_low"] = pd.Series(downside_low, index=controls.index).fillna(False)
    controls["context_confirm"] = pd.Series(context_confirm, index=controls.index).fillna(False)
    controls["missing_context"] = pd.Series(missing_context, index=controls.index).fillna(True)

    if family == "downside_downsize":
        if down_multiplier is None:
            raise ValueError("downside_downsize requires down_multiplier.")
        risk_multiplier = np.where(controls["downside_high"], float(down_multiplier), 1.0)
        state = np.where(controls["downside_high"], "hostile", "nominal")
    elif family == "downside_three_state":
        if down_multiplier is None or up_multiplier is None or low_threshold is None:
            raise ValueError("downside_three_state requires low_threshold, up_multiplier, and down_multiplier.")
        risk_multiplier = np.where(
            controls["downside_high"],
            float(down_multiplier),
            np.where(controls["downside_low"], float(up_multiplier), 1.0),
        )
        state = np.where(controls["downside_high"], "hostile", np.where(controls["downside_low"], "favorable", "neutral"))
    elif family == "conditional_downsize_with_context":
        if down_multiplier is None or context_key is None:
            raise ValueError("conditional_downsize_with_context requires down_multiplier and context_key.")
        active = controls["downside_high"] & controls["context_confirm"]
        risk_multiplier = np.where(active, float(down_multiplier), 1.0)
        state = np.where(active, f"hostile_{context_key}", np.where(controls["downside_high"], "unconfirmed_hostile", "nominal"))
    elif family == "reference_downside_hard_skip":
        risk_multiplier = np.where(controls["downside_high"], 0.0, 1.0)
        state = np.where(controls["downside_high"], "skip_ref", "nominal")
    else:
        raise ValueError(f"Unsupported family '{family}'.")

    controls["risk_multiplier"] = np.where(controls["missing_context"], 1.0, risk_multiplier).astype(float)
    controls["state"] = np.where(controls["missing_context"], "insufficient_history", state)
    controls["skip_trade"] = controls["risk_multiplier"].eq(0.0)
    controls["downscaled_trade"] = controls["risk_multiplier"].between(0.0, 1.0, inclusive="neither")
    controls["upscaled_trade"] = controls["risk_multiplier"] > 1.0
    return controls.sort_values("session_date").reset_index(drop=True)


def _variant_diagnostics(nominal_trades: pd.DataFrame, variant_trades: pd.DataFrame, controls: pd.DataFrame) -> dict[str, Any]:
    nominal = nominal_trades.copy()
    nominal["session_date"] = pd.to_datetime(nominal["session_date"]).dt.date
    control_view = controls.copy()
    control_view["session_date"] = pd.to_datetime(control_view["session_date"]).dt.date
    merged = nominal.merge(
        control_view[["session_date", "risk_multiplier", "skip_trade", "downscaled_trade", "upscaled_trade", "state"]],
        on="session_date",
        how="left",
    )

    skipped = merged.loc[merged["skip_trade"].fillna(False)].copy()
    downscaled = merged.loc[merged["downscaled_trade"].fillna(False)].copy()
    upscaled = merged.loc[merged["upscaled_trade"].fillna(False)].copy()
    realized = variant_trades.copy()
    realized["session_date"] = pd.to_datetime(realized["session_date"]).dt.date

    baseline_by_session = nominal.set_index("session_date")["net_pnl_usd"]
    realized_by_session = realized.groupby("session_date")["net_pnl_usd"].sum() if not realized.empty else pd.Series(dtype=float)
    pnl_delta = (baseline_by_session - realized_by_session.reindex(baseline_by_session.index, fill_value=0.0)).astype(float)
    downscaled_delta = pnl_delta.loc[pnl_delta.index.isin(pd.Index(downscaled["session_date"]).unique())]
    upscaled_delta = (-pnl_delta).loc[pnl_delta.index.isin(pd.Index(upscaled["session_date"]).unique())]

    skipped_pnl = pd.to_numeric(skipped["net_pnl_usd"], errors="coerce")
    downscaled_pnl = pd.to_numeric(downscaled["net_pnl_usd"], errors="coerce")
    upscaled_pnl = pd.to_numeric(upscaled["net_pnl_usd"], errors="coerce")
    return {
        "skipped_trades": int(len(skipped)),
        "downscaled_trades": int(len(downscaled)),
        "upscaled_trades": int(len(upscaled)),
        "losers_avoided": int((skipped_pnl < 0).sum()),
        "winners_sacrificed": int((skipped_pnl > 0).sum()),
        "losers_avoided_usd": float(-skipped_pnl[skipped_pnl < 0].sum()),
        "winners_sacrificed_usd": float(skipped_pnl[skipped_pnl > 0].sum()),
        "downscaled_losers": int((downscaled_pnl < 0).sum()),
        "downscaled_winners": int((downscaled_pnl > 0).sum()),
        "downsizing_loss_saved_usd": float(downscaled_delta[downscaled_delta < 0].abs().sum()),
        "downsizing_profit_given_up_usd": float(downscaled_delta[downscaled_delta > 0].sum()),
        "upscaled_losers": int((upscaled_pnl < 0).sum()),
        "upscaled_winners": int((upscaled_pnl > 0).sum()),
        "upsizing_loss_added_usd": float((-upscaled_delta[upscaled_delta < 0]).sum()),
        "upsizing_profit_added_usd": float(upscaled_delta[upscaled_delta > 0].sum()),
    }


def _build_variant_run(
    *,
    name: str,
    family: str,
    description: str,
    feature_key: str,
    horizon: str,
    context_key: str | None,
    parameters: dict[str, Any],
    nominal_trades: pd.DataFrame,
    controls: pd.DataFrame,
    analysis: SymbolAnalysis,
) -> M2KVariantRun:
    trades = _apply_trade_multipliers(
        nominal_trades=nominal_trades,
        controls=controls,
        account_size_usd=analysis.baseline.account_size_usd,
        base_risk_pct=analysis.baseline.risk_per_trade_pct,
        tick_value_usd=float(analysis.instrument_spec["tick_value_usd"]),
        point_value_usd=float(analysis.instrument_spec["point_value_usd"]),
        commission_per_side_usd=float(analysis.instrument_spec["commission_per_side_usd"]),
    )
    daily_results = _daily_results_from_trades(trades, analysis.all_sessions, initial_capital=analysis.baseline.account_size_usd)
    summary_by_scope = _build_summary_by_scope(
        trades,
        all_sessions=analysis.all_sessions,
        is_sessions=analysis.is_sessions,
        oos_sessions=analysis.oos_sessions,
        initial_capital=analysis.baseline.account_size_usd,
    )
    diagnostics = _variant_diagnostics(nominal_trades=nominal_trades, variant_trades=trades, controls=controls)
    return M2KVariantRun(
        name=name,
        family=family,
        description=description,
        feature_key=feature_key,
        horizon=horizon,
        context_key=context_key,
        parameters=parameters,
        controls=controls,
        trades=trades,
        daily_results=daily_results,
        summary_by_scope=summary_by_scope,
        diagnostics=diagnostics,
    )


def _build_baseline_run(analysis: SymbolAnalysis, *, selected_sessions: set) -> M2KVariantRun:
    selected_trades = _subset_by_sessions(analysis.baseline_trades, selected_sessions)
    controls = pd.DataFrame(
        {
            "session_date": sorted(selected_sessions),
            "phase": np.where(
                pd.Index(sorted(selected_sessions)).isin(set(pd.to_datetime(pd.Index(analysis.is_sessions)).date)),
                "is",
                "oos",
            ),
            "risk_multiplier": 1.0,
            "skip_trade": False,
            "downscaled_trade": False,
            "upscaled_trade": False,
            "state": "baseline",
        }
    )
    daily_results = _daily_results_from_trades(selected_trades, analysis.all_sessions, initial_capital=analysis.baseline.account_size_usd)
    summary_by_scope = _build_summary_by_scope(
        selected_trades,
        all_sessions=analysis.all_sessions,
        is_sessions=analysis.is_sessions,
        oos_sessions=analysis.oos_sessions,
        initial_capital=analysis.baseline.account_size_usd,
    )
    return M2KVariantRun(
        name="baseline",
        family="baseline",
        description="Unchanged audited M2K baseline after the existing ensemble selection.",
        feature_key="baseline",
        horizon="baseline",
        context_key=None,
        parameters={},
        controls=controls,
        trades=selected_trades,
        daily_results=daily_results,
        summary_by_scope=summary_by_scope,
        diagnostics={
            "skipped_trades": 0,
            "downscaled_trades": 0,
            "upscaled_trades": 0,
            "losers_avoided": 0,
            "winners_sacrificed": 0,
            "losers_avoided_usd": 0.0,
            "winners_sacrificed_usd": 0.0,
            "downscaled_losers": 0,
            "downscaled_winners": 0,
            "downsizing_loss_saved_usd": 0.0,
            "downsizing_profit_given_up_usd": 0.0,
            "upscaled_losers": 0,
            "upscaled_winners": 0,
            "upsizing_loss_added_usd": 0.0,
            "upsizing_profit_added_usd": 0.0,
        },
    )


def _variant_row(run: M2KVariantRun, baseline_run: M2KVariantRun) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant_name": run.name,
        "family": run.family,
        "description": run.description,
        "feature_key": run.feature_key,
        "horizon": run.horizon,
        "context_key": run.context_key or "",
        "parameters_json": json.dumps({key: _serialize_value(value) for key, value in run.parameters.items()}, sort_keys=True),
        **run.diagnostics,
    }
    for scope in ("overall", "is", "oos"):
        for column in (
            "net_pnl",
            "sharpe",
            "sortino",
            "max_drawdown",
            "profit_factor",
            "win_rate",
            "average_trade",
            "trade_count",
            "n_days_traded",
            "participation_rate",
            "trades_per_month",
            "longest_losing_streak_trade",
            "longest_losing_streak_daily",
        ):
            row[f"{scope}_{column}"] = _scope_value(run.summary_by_scope, scope, column)

    for key, value in run.parameters.items():
        row[key] = _serialize_value(value)

    baseline_is_dd = abs(float(_scope_value(baseline_run.summary_by_scope, "is", "max_drawdown")))
    baseline_oos_dd = abs(float(_scope_value(baseline_run.summary_by_scope, "oos", "max_drawdown")))
    baseline_is_pf = float(_scope_value(baseline_run.summary_by_scope, "is", "profit_factor"))
    baseline_oos_pf = float(_scope_value(baseline_run.summary_by_scope, "oos", "profit_factor"))
    baseline_is_trades = float(_scope_value(baseline_run.summary_by_scope, "is", "trade_count"))
    baseline_oos_trades = float(_scope_value(baseline_run.summary_by_scope, "oos", "trade_count"))

    row["is_sharpe_delta_vs_baseline"] = float(_scope_value(run.summary_by_scope, "is", "sharpe")) - float(
        _scope_value(baseline_run.summary_by_scope, "is", "sharpe")
    )
    row["oos_sharpe_delta_vs_baseline"] = float(_scope_value(run.summary_by_scope, "oos", "sharpe")) - float(
        _scope_value(baseline_run.summary_by_scope, "oos", "sharpe")
    )
    row["is_profit_factor_delta_vs_baseline"] = float(_scope_value(run.summary_by_scope, "is", "profit_factor")) - baseline_is_pf
    row["oos_profit_factor_delta_vs_baseline"] = float(_scope_value(run.summary_by_scope, "oos", "profit_factor")) - baseline_oos_pf
    row["is_max_drawdown_improvement_vs_baseline"] = _safe_div(
        baseline_is_dd - abs(float(_scope_value(run.summary_by_scope, "is", "max_drawdown"))),
        max(baseline_is_dd, 1.0),
        default=0.0,
    )
    row["oos_max_drawdown_improvement_vs_baseline"] = _safe_div(
        baseline_oos_dd - abs(float(_scope_value(run.summary_by_scope, "oos", "max_drawdown"))),
        max(baseline_oos_dd, 1.0),
        default=0.0,
    )
    row["is_trade_retention_vs_baseline"] = _safe_div(
        float(_scope_value(run.summary_by_scope, "is", "trade_count")),
        max(baseline_is_trades, 1.0),
        default=0.0,
    )
    row["oos_trade_retention_vs_baseline"] = _safe_div(
        float(_scope_value(run.summary_by_scope, "oos", "trade_count")),
        max(baseline_oos_trades, 1.0),
        default=0.0,
    )
    row["is_reference_variant"] = run.family == "reference_downside_hard_skip"
    return row


def _select_promotion_candidates(results_df: pd.DataFrame, spec: M2KSemivarianceSizingSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = results_df.copy()
    scored["promotion_candidate"] = False
    scored["promotion_candidate_rank"] = np.nan
    scored["oos_promotion_ready"] = False
    scored["support_cluster_count"] = 0
    scored["is_screen_score"] = np.nan

    if "baseline" not in set(scored["variant_name"].astype(str)):
        return scored, pd.DataFrame(columns=scored.columns)

    non_baseline = scored.loc[scored["variant_name"].astype(str) != "baseline"].copy()
    non_baseline["is_screen_score"] = (
        pd.to_numeric(non_baseline["is_sharpe_delta_vs_baseline"], errors="coerce").fillna(0.0)
        + 0.50 * pd.to_numeric(non_baseline["is_max_drawdown_improvement_vs_baseline"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(non_baseline["is_profit_factor_delta_vs_baseline"], errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(non_baseline["is_trade_retention_vs_baseline"], errors="coerce").fillna(0.0)
    )
    eligible = (
        ~non_baseline["is_reference_variant"].fillna(False)
        & (pd.to_numeric(non_baseline["is_sharpe_delta_vs_baseline"], errors="coerce") > 0.0)
        & (pd.to_numeric(non_baseline["is_max_drawdown_improvement_vs_baseline"], errors="coerce") > 0.0)
        & (pd.to_numeric(non_baseline["is_trade_retention_vs_baseline"], errors="coerce") >= float(spec.min_trade_retention))
        & (pd.to_numeric(non_baseline["is_trade_count"], errors="coerce") >= int(spec.min_is_trade_count))
    )
    non_baseline["promotion_candidate"] = eligible.fillna(False)
    promotable = (
        non_baseline["promotion_candidate"]
        & (pd.to_numeric(non_baseline["oos_sharpe_delta_vs_baseline"], errors="coerce") > 0.0)
        & (pd.to_numeric(non_baseline["oos_max_drawdown_improvement_vs_baseline"], errors="coerce") > 0.0)
        & (pd.to_numeric(non_baseline["oos_trade_retention_vs_baseline"], errors="coerce") >= float(spec.min_trade_retention))
    )
    non_baseline["oos_promotion_ready"] = promotable.fillna(False)
    support = (
        non_baseline.loc[non_baseline["oos_promotion_ready"]]
        .groupby(["family", "feature_key", "context_key"], dropna=False)
        .size()
        .rename("support_cluster_count")
        .reset_index()
    )
    if not support.empty:
        non_baseline = non_baseline.drop(columns=["support_cluster_count"], errors="ignore")
        non_baseline = non_baseline.merge(support, on=["family", "feature_key", "context_key"], how="left")
        non_baseline["support_cluster_count"] = pd.to_numeric(non_baseline["support_cluster_count"], errors="coerce").fillna(0).astype(int)

    candidate_rows = non_baseline.loc[non_baseline["promotion_candidate"]].copy()
    if not candidate_rows.empty:
        candidate_rows = candidate_rows.sort_values(
            ["is_screen_score", "is_sharpe_delta_vs_baseline", "is_max_drawdown_improvement_vs_baseline", "is_trade_retention_vs_baseline"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        candidate_rows["promotion_candidate_rank"] = np.arange(1, len(candidate_rows) + 1)
        rank_map = dict(zip(candidate_rows["variant_name"], candidate_rows["promotion_candidate_rank"]))
        non_baseline["promotion_candidate_rank"] = non_baseline["variant_name"].map(rank_map)
    else:
        candidate_rows = pd.DataFrame(columns=non_baseline.columns)

    merged = scored.merge(
        non_baseline[["variant_name", "promotion_candidate", "promotion_candidate_rank", "oos_promotion_ready", "support_cluster_count", "is_screen_score"]],
        on="variant_name",
        how="left",
        suffixes=("", "_new"),
    )
    for column in ("promotion_candidate", "oos_promotion_ready"):
        base_values = pd.Series(merged[column], index=merged.index, dtype="boolean")
        new_values = pd.Series(merged[f"{column}_new"], index=merged.index, dtype="boolean")
        merged[column] = new_values.combine_first(base_values).fillna(False).astype(bool)
    for column in ("promotion_candidate_rank", "support_cluster_count", "is_screen_score"):
        merged[column] = merged[f"{column}_new"].combine_first(merged[column])
    merged = merged.drop(columns=[column for column in merged.columns if column.endswith("_new")])
    return merged, candidate_rows


def _build_reference_bucket_summary(
    trade_features: pd.DataFrame,
    *,
    feature_key: str,
    horizon: str,
) -> pd.DataFrame:
    column = _feature_column(feature_key, horizon)
    if column not in trade_features.columns:
        return pd.DataFrame(columns=["phase", "weekday_name", "bucket", "trade_count", "mean_net_pnl", "win_rate"])

    view = trade_features[["phase", "weekday_name", "net_pnl_usd", column]].copy()
    values = pd.to_numeric(view[column], errors="coerce")
    view = view.loc[values.notna()].copy()
    if view.empty:
        return pd.DataFrame(columns=["phase", "weekday_name", "bucket", "trade_count", "mean_net_pnl", "win_rate"])
    view["bucket"] = pd.cut(
        pd.to_numeric(view[column], errors="coerce"),
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["low_25", "mid_25_50", "mid_50_75", "high_75_plus"],
    )
    grouped = (
        view.groupby(["phase", "weekday_name", "bucket"], dropna=False, observed=False)
        .agg(
            trade_count=("net_pnl_usd", "size"),
            mean_net_pnl=("net_pnl_usd", "mean"),
            win_rate=("net_pnl_usd", lambda series: float((pd.to_numeric(series, errors="coerce") > 0).mean()) if len(series) else 0.0),
        )
        .reset_index()
    )
    grouped["feature_key"] = feature_key
    grouped["horizon"] = horizon
    return grouped


def _build_final_verdict(results_df: pd.DataFrame, candidate_rows: pd.DataFrame) -> dict[str, Any]:
    baseline = results_df.loc[results_df["variant_name"] == "baseline"].iloc[0].to_dict()
    ex_post_best = (
        results_df.loc[results_df["variant_name"] != "baseline"]
        .sort_values(
            ["oos_sharpe_delta_vs_baseline", "oos_max_drawdown_improvement_vs_baseline", "oos_trade_retention_vs_baseline"],
            ascending=[False, False, False],
        )
        .head(1)
    )
    best_candidate = candidate_rows.head(1)
    promotable = candidate_rows.loc[
        candidate_rows["oos_promotion_ready"].fillna(False) & (pd.to_numeric(candidate_rows["support_cluster_count"], errors="coerce") >= 2)
    ].copy()
    if not promotable.empty:
        promotable = promotable.sort_values(
            ["oos_sharpe_delta_vs_baseline", "oos_max_drawdown_improvement_vs_baseline", "oos_trade_retention_vs_baseline"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        promoted = promotable.iloc[0].to_dict()
    else:
        promoted = None

    return {
        "credible_overlay": promoted is not None,
        "baseline_variant": "baseline",
        "baseline_oos_sharpe": float(baseline.get("oos_sharpe", 0.0)),
        "baseline_oos_net_pnl": float(baseline.get("oos_net_pnl", 0.0)),
        "baseline_oos_max_drawdown": float(baseline.get("oos_max_drawdown", 0.0)),
        "baseline_oos_profit_factor": float(baseline.get("oos_profit_factor", 0.0)),
        "baseline_oos_trade_count": int(baseline.get("oos_trade_count", 0)),
        "promotion_candidate_count": int(len(candidate_rows)),
        "promotion_ready_clustered_count": int(len(promotable)),
        "best_is_candidate": None if best_candidate.empty else best_candidate.iloc[0].to_dict(),
        "best_ex_post_oos_variant": None if ex_post_best.empty else ex_post_best.iloc[0].to_dict(),
        "promoted_variant": promoted,
    }


def _write_report(
    output_path: Path,
    spec: M2KSemivarianceSizingSpec,
    baseline_config: M2KBaselineConfig,
    results_df: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    final_verdict: dict[str, Any],
) -> None:
    baseline = results_df.loc[results_df["variant_name"] == "baseline"].iloc[0]
    top_oos = (
        results_df.loc[results_df["variant_name"] != "baseline"]
        .sort_values(
            ["oos_sharpe_delta_vs_baseline", "oos_max_drawdown_improvement_vs_baseline", "oos_trade_retention_vs_baseline"],
            ascending=[False, False, False],
        )
        .head(10)
    )
    promoted_variant = final_verdict.get("promoted_variant")
    best_is_candidate = final_verdict.get("best_is_candidate")
    best_ex_post = final_verdict.get("best_ex_post_oos_variant")

    lines = [
        "# M2K Intraday Semivariance Sizing Follow-Up",
        "",
        "## Methodology",
        "",
        f"- Asset only: `{SYMBOL}`.",
        f"- Baseline reused unchanged from `{baseline_config.source_reference}` with OR `{baseline_config.baseline.or_minutes}m`, direction `{baseline_config.baseline.direction}`, rule `{baseline_config.aggregation_rule}`.",
        "- Overlay scope restricted to post-signal sizing modulation; no alpha, execution, slippage, cost, or portfolio-accounting changes.",
        f"- Semivariance horizons tested: `{', '.join(spec.semivariance_horizons)}`.",
        "- Downside-focused features ranked with strict prior-history rolling percentiles only; no same-bar or future leakage.",
        "- Variant thresholds were screened on IS only, then OOS was used only to decide whether any IS-promoted candidate is defensible.",
        "",
        "## Variants Tested",
        "",
        f"- `downside_downsize`: feature grid `{list(spec.downside_feature_keys)}` x horizons `{list(spec.semivariance_horizons)}` x thresholds `{list(spec.downside_thresholds)}` x down-multipliers `{list(spec.downside_multipliers)}`.",
        f"- `downside_three_state`: same downside features/horizons/thresholds with low-threshold `{spec.three_state_low_threshold:.2f}` and multiplier pairs `{list(spec.three_state_pairs)}`.",
        f"- `conditional_downsize_with_context`: downside features `{list(spec.conditional_feature_keys)}` confirmed by contexts `{list(spec.context_keys)}`.",
        f"- `reference_downside_hard_skip`: small comparison grid on features `{list(spec.reference_skip_feature_keys)}` and horizons `{list(spec.reference_skip_horizons)}`.",
        "",
        "## Baseline OOS",
        "",
        f"- Sharpe `{float(baseline['oos_sharpe']):.3f}` | net PnL `{float(baseline['oos_net_pnl']):.2f}` | maxDD `{float(baseline['oos_max_drawdown']):.2f}` | PF `{float(baseline['oos_profit_factor']):.3f}` | trades `{int(baseline['oos_trade_count'])}`.",
        "",
        "## IS Promotion Candidates",
        "",
    ]
    if candidate_rows.empty:
        lines.append("- No overlay variant cleared the IS screen of higher Sharpe, lower drawdown, and adequate retained trade count.")
    else:
        for _, row in candidate_rows.head(10).iterrows():
            lines.append(
                f"- `{row['variant_name']}`: IS Sharpe delta `{float(row['is_sharpe_delta_vs_baseline']):+.3f}`, "
                f"IS maxDD improvement `{100.0 * float(row['is_max_drawdown_improvement_vs_baseline']):+.1f}%`, "
                f"OOS Sharpe delta `{float(row['oos_sharpe_delta_vs_baseline']):+.3f}`, "
                f"OOS maxDD improvement `{100.0 * float(row['oos_max_drawdown_improvement_vs_baseline']):+.1f}%`, "
                f"OOS trade retention `{100.0 * float(row['oos_trade_retention_vs_baseline']):.1f}%`."
            )

    lines.extend(["", "## OOS Ranking", ""])
    for _, row in top_oos.iterrows():
        lines.append(
            f"- `{row['variant_name']}`: OOS Sharpe `{float(row['oos_sharpe']):.3f}` "
            f"(delta `{float(row['oos_sharpe_delta_vs_baseline']):+.3f}`), "
            f"net PnL `{float(row['oos_net_pnl']):.2f}`, "
            f"maxDD `{float(row['oos_max_drawdown']):.2f}` "
            f"(improvement `{100.0 * float(row['oos_max_drawdown_improvement_vs_baseline']):+.1f}%`), "
            f"PF `{float(row['oos_profit_factor']):.3f}`, trades `{int(row['oos_trade_count'])}`."
        )

    lines.extend(["", "## Verdict", ""])
    if best_is_candidate:
        lines.append(
            f"- Best IS-screened candidate: `{best_is_candidate['variant_name']}` with IS score `{float(best_is_candidate['is_screen_score']):.3f}`."
        )
    if best_ex_post:
        lines.append(
            f"- Best ex-post OOS row: `{best_ex_post['variant_name']}` with Sharpe delta `{float(best_ex_post['oos_sharpe_delta_vs_baseline']):+.3f}` and maxDD improvement `{100.0 * float(best_ex_post['oos_max_drawdown_improvement_vs_baseline']):+.1f}%`."
        )
    if promoted_variant:
        lines.append(
            f"- Promotion decision: `YES`. Promoted variant `{promoted_variant['variant_name']}` improved both OOS Sharpe and OOS max drawdown while retaining `{100.0 * float(promoted_variant['oos_trade_retention_vs_baseline']):.1f}%` of baseline trades, with clustered support count `{int(promoted_variant['support_cluster_count'])}`."
        )
    else:
        lines.append(
            "- Promotion decision: `NO`. No IS-screened variant achieved the required OOS Sharpe improvement, OOS max-drawdown improvement, trade retention, and non-isolated support simultaneously."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(spec: M2KSemivarianceSizingSpec | None = None) -> M2KCampaignArtifacts:
    ensure_directories()
    active_spec = spec or default_campaign_spec()
    baseline_config = active_spec.baseline_config or _default_baseline_config()
    output_dir = active_spec.output_root
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = EXPORTS_DIR / f"m2k_intraday_semivariance_sizing_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = baseline_config.dataset_path or resolve_processed_dataset(baseline_config.symbol, timeframe=baseline_config.timeframe)
    analysis = analyze_symbol(
        baseline_config.symbol,
        baseline=baseline_config.baseline,
        grid=baseline_config.grid,
        is_fraction=active_spec.is_fraction,
        dataset_path=dataset_path,
        data_timeframe=baseline_config.timeframe,
    )
    selected_sessions = _selected_ensemble_sessions(analysis, baseline_config.aggregation_rule)
    trade_features = _build_trade_feature_frame(
        analysis,
        selected_sessions=selected_sessions,
        percentile_history=active_spec.percentile_history,
        min_percentile_history=active_spec.min_percentile_history,
        horizons=active_spec.semivariance_horizons,
    )
    baseline_run = _build_baseline_run(analysis, selected_sessions=selected_sessions)
    nominal_trades = baseline_run.trades.copy()

    variant_runs: dict[str, M2KVariantRun] = {"baseline": baseline_run}
    for feature_key in active_spec.downside_feature_keys:
        for horizon in active_spec.semivariance_horizons:
            for threshold in active_spec.downside_thresholds:
                for down_multiplier in active_spec.downside_multipliers:
                    controls = _build_variant_controls(
                        trade_features,
                        family="downside_downsize",
                        feature_key=feature_key,
                        horizon=horizon,
                        threshold=threshold,
                        down_multiplier=down_multiplier,
                    )
                    name = _variant_name("downside_downsize", feature_key, horizon, threshold, down_multiplier=down_multiplier)
                    variant_runs[name] = _build_variant_run(
                        name=name,
                        family="downside_downsize",
                        description="Reduce size when downside semivariance percentile is elevated.",
                        feature_key=feature_key,
                        horizon=horizon,
                        context_key=None,
                        parameters={"feature_key": feature_key, "high_threshold": threshold, "down_multiplier": down_multiplier},
                        nominal_trades=nominal_trades,
                        controls=controls,
                        analysis=analysis,
                    )

                for up_multiplier, down_multiplier in active_spec.three_state_pairs:
                    controls = _build_variant_controls(
                        trade_features,
                        family="downside_three_state",
                        feature_key=feature_key,
                        horizon=horizon,
                        threshold=threshold,
                        low_threshold=active_spec.three_state_low_threshold,
                        up_multiplier=up_multiplier,
                        down_multiplier=down_multiplier,
                    )
                    name = _variant_name(
                        "downside_three_state",
                        feature_key,
                        horizon,
                        threshold,
                        low_threshold=active_spec.three_state_low_threshold,
                        up_multiplier=up_multiplier,
                        down_multiplier=down_multiplier,
                    )
                    variant_runs[name] = _build_variant_run(
                        name=name,
                        family="downside_three_state",
                        description="Conservative three-state sizing using low / neutral / high downside semivariance.",
                        feature_key=feature_key,
                        horizon=horizon,
                        context_key=None,
                        parameters={
                            "feature_key": feature_key,
                            "low_threshold": active_spec.three_state_low_threshold,
                            "high_threshold": threshold,
                            "up_multiplier": up_multiplier,
                            "down_multiplier": down_multiplier,
                        },
                        nominal_trades=nominal_trades,
                        controls=controls,
                        analysis=analysis,
                    )

    for feature_key in active_spec.conditional_feature_keys:
        for horizon in active_spec.semivariance_horizons:
            for threshold in active_spec.downside_thresholds:
                for context_key in active_spec.context_keys:
                    for down_multiplier in active_spec.downside_multipliers:
                        controls = _build_variant_controls(
                            trade_features,
                            family="conditional_downsize_with_context",
                            feature_key=feature_key,
                            horizon=horizon,
                            threshold=threshold,
                            down_multiplier=down_multiplier,
                            context_key=context_key,
                        )
                        name = _variant_name(
                            "conditional_downsize_with_context",
                            feature_key,
                            horizon,
                            threshold,
                            down_multiplier=down_multiplier,
                            context_key=context_key,
                        )
                        variant_runs[name] = _build_variant_run(
                            name=name,
                            family="conditional_downsize_with_context",
                            description="Downsize only when downside semivariance is elevated and a context regime confirms it.",
                            feature_key=feature_key,
                            horizon=horizon,
                            context_key=context_key,
                            parameters={
                                "feature_key": feature_key,
                                "high_threshold": threshold,
                                "context_key": context_key,
                                "down_multiplier": down_multiplier,
                            },
                            nominal_trades=nominal_trades,
                            controls=controls,
                            analysis=analysis,
                        )

    for feature_key in active_spec.reference_skip_feature_keys:
        for horizon in active_spec.reference_skip_horizons:
            for threshold in active_spec.reference_skip_thresholds:
                controls = _build_variant_controls(
                    trade_features,
                    family="reference_downside_hard_skip",
                    feature_key=feature_key,
                    horizon=horizon,
                    threshold=threshold,
                )
                name = _variant_name("reference_downside_hard_skip", feature_key, horizon, threshold)
                variant_runs[name] = _build_variant_run(
                    name=name,
                    family="reference_downside_hard_skip",
                    description="Small reference hard-skip grid for comparison only.",
                    feature_key=feature_key,
                    horizon=horizon,
                    context_key=None,
                    parameters={"feature_key": feature_key, "high_threshold": threshold},
                    nominal_trades=nominal_trades,
                    controls=controls,
                    analysis=analysis,
                )

    variant_results = pd.DataFrame([_variant_row(run, baseline_run) for run in variant_runs.values()])
    variant_results = variant_results.sort_values(["family", "feature_key", "horizon", "variant_name"]).reset_index(drop=True)
    variant_results, promotion_candidates = _select_promotion_candidates(variant_results, active_spec)
    final_verdict = _build_final_verdict(variant_results, promotion_candidates)

    heatmap_ready = variant_results.loc[variant_results["variant_name"] != "baseline"].copy()
    heatmap_ready["context_key"] = heatmap_ready["context_key"].fillna("")
    heatmap_ready = heatmap_ready[
        [
            "variant_name",
            "family",
            "feature_key",
            "horizon",
            "context_key",
            "high_threshold",
            "low_threshold",
            "up_multiplier",
            "down_multiplier",
            "promotion_candidate",
            "oos_promotion_ready",
            "support_cluster_count",
            "is_screen_score",
            "is_sharpe",
            "oos_sharpe",
            "oos_sharpe_delta_vs_baseline",
            "oos_net_pnl",
            "oos_max_drawdown",
            "oos_max_drawdown_improvement_vs_baseline",
            "oos_profit_factor",
            "oos_trade_count",
        ]
    ].copy()

    best_bucket_feature = "rs_minus_share_pct"
    best_bucket_horizon = "60m"
    if final_verdict.get("best_is_candidate"):
        best_bucket_feature = str(final_verdict["best_is_candidate"].get("feature_key", best_bucket_feature))
        best_bucket_horizon = str(final_verdict["best_is_candidate"].get("horizon", best_bucket_horizon))
    weekday_bucket_summary = _build_reference_bucket_summary(
        trade_features,
        feature_key=best_bucket_feature,
        horizon=best_bucket_horizon,
    )

    variant_results.to_csv(output_dir / "variant_results.csv", index=False)
    promotion_candidates.to_csv(output_dir / "promotion_candidates.csv", index=False)
    heatmap_ready.to_csv(output_dir / "heatmap_ready.csv", index=False)
    trade_features.to_csv(output_dir / "m2k_trade_features.csv", index=False)
    weekday_bucket_summary.to_csv(output_dir / "weekday_downside_bucket_summary.csv", index=False)

    _write_report(output_dir / "final_report.md", active_spec, baseline_config, variant_results, promotion_candidates, final_verdict)
    _json_dump(output_dir / "final_verdict.json", final_verdict)
    _json_dump(
        output_dir / "run_metadata.json",
        {
            "spec": asdict(active_spec),
            "baseline_config": asdict(baseline_config),
            "dataset_path": dataset_path,
            "analysis_symbol": analysis.symbol,
            "selected_session_count": len(selected_sessions),
            "variant_count": len(variant_results),
        },
    )

    return M2KCampaignArtifacts(
        output_dir=output_dir,
        spec=active_spec,
        baseline_config=baseline_config,
        analysis=analysis,
        selected_sessions=selected_sessions,
        trade_features=trade_features,
        baseline_run=baseline_run,
        variant_runs=variant_runs,
        variant_results=variant_results,
        promotion_candidates=promotion_candidates,
        final_verdict=final_verdict,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a reduced smoke grid.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional export folder override.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = smoke_campaign_spec(output_root=args.output_root) if args.smoke else default_campaign_spec(output_root=args.output_root)
    artifacts = run_campaign(spec)
    print(artifacts.output_dir)


if __name__ == "__main__":
    main()
