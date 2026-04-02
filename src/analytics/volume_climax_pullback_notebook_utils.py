"""Helpers for client-facing Volume Climax Pullback V3 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.orb_notebook_utils import normalize_curve
from src.analytics.volume_climax_pullback_common import (
    filter_trades_by_sessions,
    load_symbol_data,
    resample_rth_1h,
    safe_float,
    split_sessions,
    summarize_scope,
)
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)


def _naive_normalized_index(values: Any) -> pd.Index:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(timestamps, pd.Series):
        timestamps = pd.Index(timestamps)
    return pd.Index(timestamps.tz_convert(None).normalize())


def variant_from_summary_row(row: pd.Series | dict[str, Any]) -> VolumeClimaxPullbackV2Variant:
    """Rebuild a V2/V3 variant object from a summary row."""
    series = row if isinstance(row, pd.Series) else pd.Series(row)

    def _optional_int(name: str) -> int | None:
        value = series.get(name)
        if pd.isna(value):
            return None
        return int(value)

    def _optional_float(name: str) -> float | None:
        value = series.get(name)
        if pd.isna(value):
            return None
        return float(value)

    return VolumeClimaxPullbackV2Variant(
        name=str(series.get("variant_name") or series.get("name")),
        family=str(series["family"]),
        timeframe=str(series.get("timeframe", "1h")),
        volume_quantile=float(series["volume_quantile"]),
        volume_lookback=int(series.get("volume_lookback", 50)),
        min_body_fraction=float(series["min_body_fraction"]),
        min_range_atr=float(series["min_range_atr"]),
        trend_ema_window=_optional_int("trend_ema_window"),
        ema_slope_threshold=_optional_float("ema_slope_threshold"),
        atr_percentile_low=_optional_float("atr_percentile_low"),
        atr_percentile_high=_optional_float("atr_percentile_high"),
        compression_ratio_max=_optional_float("compression_ratio_max"),
        entry_mode=str(series.get("entry_mode", "next_open")),
        pullback_fraction=_optional_float("pullback_fraction"),
        confirmation_window=_optional_int("confirmation_window"),
        exit_mode=str(series["exit_mode"]),
        rr_target=float(series.get("rr_target", 1.0)),
        atr_target_multiple=_optional_float("atr_target_multiple"),
        time_stop_bars=int(series["time_stop_bars"]),
        trailing_atr_multiple=float(series.get("trailing_atr_multiple", 0.5)),
        session_overlay=str(series.get("session_overlay", "all_rth")),
    )


def core_label(row: pd.Series | dict[str, Any]) -> str:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    return (
        f"vq{safe_float(series.get('volume_quantile')):.3f}"
        f" | bf{safe_float(series.get('min_body_fraction')):.1f}"
        f" | ra{safe_float(series.get('min_range_atr')):.1f}"
    )


def exit_profile_label(row: pd.Series | dict[str, Any]) -> str:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    return f"{series['exit_mode']} | ts{int(series['time_stop_bars'])}"


def regime_signature(row: pd.Series | dict[str, Any]) -> str:
    series = row if isinstance(row, pd.Series) else pd.Series(row)
    return (
        f"{series.get('ema_slope_filter', 'off')}"
        f" / {series.get('atr_percentile_band', 'off')}"
        f" / {series.get('compression_filter', 'off')}"
    )


def find_variant_row(
    summary: pd.DataFrame,
    *,
    symbol: str,
    family: str,
    exit_mode: str,
    time_stop_bars: int,
    volume_quantile: float,
    min_body_fraction: float,
    min_range_atr: float,
    ema_slope_filter: str = "off",
    atr_percentile_band: str = "off",
    compression_filter: str = "off",
    variant_name: str | None = None,
) -> pd.Series:
    """Resolve a variant row from explicit notebook parameters."""
    frame = summary.loc[summary["symbol"] == symbol].copy()
    if variant_name:
        match = frame.loc[frame["variant_name"] == variant_name].copy()
        if match.empty:
            raise ValueError(f"Variant {variant_name!r} not found for symbol {symbol!r}.")
        return match.iloc[0]

    mask = (
        frame["family"].astype(str).eq(str(family))
        & frame["exit_mode"].astype(str).eq(str(exit_mode))
        & pd.to_numeric(frame["time_stop_bars"], errors="coerce").astype("Int64").eq(int(time_stop_bars))
        & np.isclose(pd.to_numeric(frame["volume_quantile"], errors="coerce"), float(volume_quantile))
        & np.isclose(pd.to_numeric(frame["min_body_fraction"], errors="coerce"), float(min_body_fraction))
        & np.isclose(pd.to_numeric(frame["min_range_atr"], errors="coerce"), float(min_range_atr))
        & frame.get("ema_slope_filter", pd.Series("off", index=frame.index)).astype(str).eq(str(ema_slope_filter))
        & frame.get("atr_percentile_band", pd.Series("off", index=frame.index)).astype(str).eq(str(atr_percentile_band))
        & frame.get("compression_filter", pd.Series("off", index=frame.index)).astype(str).eq(str(compression_filter))
    )
    match = frame.loc[mask].copy()
    if match.empty:
        raise ValueError(
            "No variant matches the requested notebook parameters: "
            f"symbol={symbol}, family={family}, exit_mode={exit_mode}, ts={time_stop_bars}, "
            f"vq={volume_quantile}, bf={min_body_fraction}, ra={min_range_atr}, "
            f"ema={ema_slope_filter}, atr_band={atr_percentile_band}, compression={compression_filter}."
        )
    ordered = match.sort_values(
        ["selection_score", "oos_sharpe", "oos_net_pnl"],
        ascending=[False, False, False],
    )
    return ordered.iloc[0]


def build_daily_results_from_trades(trades: pd.DataFrame, sessions: list) -> pd.DataFrame:
    """Aggregate trade PnL to daily results, preserving flat sessions."""
    session_index = _naive_normalized_index(pd.Index(sessions))
    out = pd.DataFrame({"session_date": session_index})
    if trades.empty:
        out["daily_pnl_usd"] = 0.0
        return out

    grouped = trades.copy()
    grouped["session_date"] = _naive_normalized_index(grouped["session_date"])
    daily = grouped.groupby("session_date", as_index=True)["net_pnl_usd"].sum()
    out = out.merge(daily.rename("daily_pnl_usd"), left_on="session_date", right_index=True, how="left")
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    return out


def build_daily_curve(daily_results: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Convert daily results into an equity curve with drawdown fields."""
    if daily_results.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown", "drawdown_pct"])

    out = daily_results.copy()
    out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce")
    out = out.dropna(subset=["session_date"]).sort_values("session_date").reset_index(drop=True)
    out["daily_pnl_usd"] = pd.to_numeric(out["daily_pnl_usd"], errors="coerce").fillna(0.0)
    out["equity"] = float(initial_capital) + out["daily_pnl_usd"].cumsum()
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = np.where(
        out["peak_equity"].abs() > 1e-9,
        (out["equity"] / out["peak_equity"] - 1.0) * 100.0,
        0.0,
    )
    return normalize_curve(
        out.rename(columns={"session_date": "timestamp"})[["timestamp", "equity", "drawdown", "drawdown_pct"]]
    )


def build_buy_hold_benchmark_curve(bars: pd.DataFrame, sessions: list, initial_capital: float) -> pd.DataFrame:
    """Build a simple daily close-to-close buy-and-hold benchmark curve."""
    if bars.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown", "drawdown_pct"])

    scoped = bars.copy()
    scoped["timestamp"] = pd.to_datetime(scoped["timestamp"], errors="coerce")
    scoped["session_date"] = _naive_normalized_index(scoped["session_date"])
    closes = (
        scoped.sort_values(["session_date", "timestamp"])
        .groupby("session_date", as_index=True)["close"]
        .last()
    )
    session_index = _naive_normalized_index(pd.Index(sessions))
    closes = closes.reindex(session_index).ffill().dropna()
    if closes.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown", "drawdown_pct"])

    initial_price = float(closes.iloc[0])
    benchmark = pd.DataFrame(
        {
            "timestamp": closes.index,
            "equity": float(initial_capital) * (pd.to_numeric(closes, errors="coerce") / initial_price),
        }
    )
    benchmark["peak_equity"] = benchmark["equity"].cummax()
    benchmark["drawdown"] = benchmark["equity"] - benchmark["peak_equity"]
    benchmark["drawdown_pct"] = np.where(
        benchmark["peak_equity"].abs() > 1e-9,
        (benchmark["equity"] / benchmark["peak_equity"] - 1.0) * 100.0,
        0.0,
    )
    return normalize_curve(benchmark[["timestamp", "equity", "drawdown", "drawdown_pct"]])


def evaluate_variant(
    *,
    symbol: str,
    variant: VolumeClimaxPullbackV2Variant,
    initial_capital: float = 50_000.0,
    input_paths: dict[str, Path] | None = None,
    split_ratio: float = 0.7,
) -> dict[str, Any]:
    """Replay a single V2/V3 spec for notebook visualization."""
    raw = load_symbol_data(symbol, input_paths=input_paths)
    bars = resample_rth_1h(raw)
    if bars.empty:
        raise ValueError(f"No RTH 1h bars available for {symbol}.")

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars["session_date"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    sessions = list(pd.Index(bars["session_date"]).dropna().unique())
    if len(sessions) < 2:
        raise ValueError(f"Need at least two sessions to evaluate {symbol}.")

    split_frame = pd.DataFrame({"session_date": sessions})
    is_sessions, oos_sessions = split_sessions(split_frame, ratio=split_ratio)

    features = prepare_volume_climax_pullback_v2_features(bars)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, variant)
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
    trades = run_volume_climax_pullback_v2_backtest(signal_df, variant, execution_model, instrument).trades

    is_session_index = _naive_normalized_index(pd.Index(is_sessions))
    oos_session_index = _naive_normalized_index(pd.Index(oos_sessions))
    signal_session_index = _naive_normalized_index(signal_df["session_date"])

    is_signal = signal_df.loc[signal_session_index.isin(is_session_index)].copy()
    oos_signal = signal_df.loc[signal_session_index.isin(oos_session_index)].copy()
    is_trades = filter_trades_by_sessions(trades, is_sessions)
    oos_trades = filter_trades_by_sessions(trades, oos_sessions)

    overall_daily = build_daily_results_from_trades(trades, sessions)
    is_daily = build_daily_results_from_trades(is_trades, is_sessions)
    oos_daily = build_daily_results_from_trades(oos_trades, oos_sessions)

    metrics_by_scope = pd.DataFrame(
        [
            {"scope": "overall", **summarize_scope(trades, signal_df, sessions)},
            {"scope": "is", **summarize_scope(is_trades, is_signal, is_sessions)},
            {"scope": "oos", **summarize_scope(oos_trades, oos_signal, oos_sessions)},
        ]
    )

    return {
        "symbol": symbol,
        "variant": variant,
        "bars": bars,
        "signal_df": signal_df,
        "trades": trades,
        "is_trades": is_trades,
        "oos_trades": oos_trades,
        "sessions": sessions,
        "is_sessions": is_sessions,
        "oos_sessions": oos_sessions,
        "daily_results": overall_daily,
        "daily_results_is": is_daily,
        "daily_results_oos": oos_daily,
        "curve_full": build_daily_curve(overall_daily, initial_capital=initial_capital),
        "curve_oos": build_daily_curve(oos_daily, initial_capital=initial_capital),
        "benchmark_curve_full": build_buy_hold_benchmark_curve(bars, sessions, initial_capital=initial_capital),
        "benchmark_curve_oos": build_buy_hold_benchmark_curve(bars, oos_sessions, initial_capital=initial_capital),
        "metrics_by_scope": metrics_by_scope,
        "instrument": instrument,
    }
