"""Feature engineering and signal selection helpers for ORB research."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.features.intraday import add_continuous_session_vwap, add_intraday_features, add_session_vwap
from src.features.opening_range import compute_opening_range
from src.features.volatility import add_atr
from src.strategy.orb import ORBStrategy
from src.utils.time_utils import build_session_time

from .types import BaselineEntryConfig, CompressionConfig, DynamicThresholdConfig


def prepare_minute_dataset(
    dataset_path,
    baseline_entry: BaselineEntryConfig,
    atr_windows: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Load and enrich the minute dataset once for all experiments."""
    raw = load_ohlcv_file(dataset_path)
    raw = clean_ohlcv(raw)
    feat = add_intraday_features(raw)
    feat = add_session_vwap(feat)
    feat = add_continuous_session_vwap(feat, session_start_hour=18)
    feat = compute_opening_range(feat, or_minutes=baseline_entry.or_minutes, opening_time=baseline_entry.opening_time)
    feat = add_atr(feat, window=sorted({int(x) for x in atr_windows if int(x) > 0}))
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    return feat


def _between_times(series: pd.Series, start: str, end: str) -> pd.Series:
    start_t = dt.time.fromisoformat(start)
    end_t = dt.time.fromisoformat(end)
    times = series.dt.time
    if start_t <= end_t:
        return (times >= start_t) & (times <= end_t)
    return (times >= start_t) | (times <= end_t)


def build_daily_reference(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Build previous-day-aware daily features without look-ahead leakage."""
    rth_mask = _between_times(minute_df["timestamp"], "09:30:00", "16:00:00")
    rth = minute_df.loc[rth_mask, ["timestamp", "session_date", "open", "high", "low", "close"]].copy()

    def _first_open(group: pd.DataFrame) -> float:
        return float(group.iloc[0]["open"]) if not group.empty else np.nan

    daily = (
        rth.groupby("session_date", sort=True)
        .apply(
            lambda g: pd.Series(
                {
                    "open_rth": _first_open(g),
                    "high_rth": float(g["high"].max()),
                    "low_rth": float(g["low"].min()),
                    "close_rth": float(g.iloc[-1]["close"]),
                }
            )
        )
        .reset_index()
    )
    daily["range_rth"] = daily["high_rth"] - daily["low_rth"]
    daily["prev_close_rth"] = daily["close_rth"].shift(1)
    daily["prev_high_rth"] = daily["high_rth"].shift(1)
    daily["prev_low_rth"] = daily["low_rth"].shift(1)

    width = (daily["high_rth"] - daily["low_rth"]).replace(0.0, np.nan)
    close_pos = (daily["close_rth"] - daily["low_rth"]) / width
    daily["close_position"] = close_pos.fillna(0.5)

    nr4_raw = daily["range_rth"] <= daily["range_rth"].rolling(4, min_periods=4).min()
    nr7_raw = daily["range_rth"] <= daily["range_rth"].rolling(7, min_periods=7).min()
    inside_raw = (daily["high_rth"] < daily["prev_high_rth"]) & (daily["low_rth"] > daily["prev_low_rth"])
    outside_raw = (daily["high_rth"] > daily["prev_high_rth"]) & (daily["low_rth"] < daily["prev_low_rth"])

    tri_high = (daily["high_rth"] < daily["high_rth"].shift(1)) & (
        daily["high_rth"].shift(1) < daily["high_rth"].shift(2)
    )
    tri_low = (daily["low_rth"] > daily["low_rth"].shift(1)) & (daily["low_rth"].shift(1) > daily["low_rth"].shift(2))
    triangle_raw = tri_high & tri_low

    strong_close_raw = daily["close_position"] >= 0.75
    weak_close_raw = daily["close_position"] <= 0.25

    # Shift all pattern labels by one day to avoid leakage at day t open.
    daily["pattern_nr4"] = nr4_raw.shift(1).fillna(False)
    daily["pattern_nr7"] = nr7_raw.shift(1).fillna(False)
    daily["pattern_inside_day"] = inside_raw.shift(1).fillna(False)
    daily["pattern_outside_day"] = outside_raw.shift(1).fillna(False)
    daily["pattern_triangle"] = triangle_raw.shift(1).fillna(False)
    daily["pattern_strong_close"] = strong_close_raw.shift(1).fillna(False)
    daily["pattern_weak_close"] = weak_close_raw.shift(1).fillna(False)

    daily["pattern_nr4_or_nr7"] = daily["pattern_nr4"] | daily["pattern_nr7"]
    daily["pattern_nr4_or_triangle"] = daily["pattern_nr4"] | daily["pattern_triangle"]
    daily["pattern_nr4_or_nr7_or_triangle"] = (
        daily["pattern_nr4"] | daily["pattern_nr7"] | daily["pattern_triangle"]
    )

    return daily


def attach_daily_reference(minute_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    merged = minute_df.merge(daily_df, on="session_date", how="left")
    merged["open_prevclose_max"] = np.maximum(
        pd.to_numeric(merged["open_rth"], errors="coerce"),
        pd.to_numeric(merged["prev_close_rth"], errors="coerce"),
    )
    merged["open_prevclose_min"] = np.minimum(
        pd.to_numeric(merged["open_rth"], errors="coerce"),
        pd.to_numeric(merged["prev_close_rth"], errors="coerce"),
    )
    return merged


def build_candidate_universe(
    minute_df: pd.DataFrame,
    baseline_entry: BaselineEntryConfig,
) -> pd.DataFrame:
    """Generate all candidate breakout rows before one-trade/day reduction."""
    strategy = ORBStrategy(
        or_minutes=baseline_entry.or_minutes,
        direction=baseline_entry.direction,
        one_trade_per_day=False,
        entry_buffer_ticks=baseline_entry.entry_buffer_ticks,
        stop_buffer_ticks=baseline_entry.stop_buffer_ticks,
        target_multiple=baseline_entry.target_multiple,
        opening_time=baseline_entry.opening_time,
        time_exit=baseline_entry.time_exit,
        account_size_usd=baseline_entry.account_size_usd,
        risk_per_trade_pct=baseline_entry.risk_per_trade_pct,
        tick_size=baseline_entry.tick_size,
        vwap_confirmation=baseline_entry.vwap_confirmation,
        vwap_column=baseline_entry.vwap_column,
    )
    signals = strategy.generate_signals(minute_df)
    signals = signals.sort_values("timestamp").reset_index(drop=True)
    signals["candidate_base_pass"] = (
        signals["raw_signal"].eq(1)
        & signals["atr_filter_pass"].fillna(False)
        & signals["direction_filter_pass"].fillna(False)
    )
    return signals


def _schedule_mask(minute_of_day: pd.Series, schedule: str) -> pd.Series:
    if schedule == "continuous_on_bar_close":
        return pd.Series(True, index=minute_of_day.index)
    if schedule == "every_5m":
        return minute_of_day.mod(5).eq(0)
    if schedule == "every_15m":
        return minute_of_day.mod(15).eq(0)
    raise ValueError(f"Unsupported schedule: {schedule}")


def _consecutive_true_within_session(mask: pd.Series, session_dates: pd.Series) -> pd.Series:
    out = pd.Series(0, index=mask.index, dtype=int)
    for _, idx in session_dates.groupby(session_dates).groups.items():
        session_mask = mask.loc[idx].fillna(False)
        counter = 0
        values = []
        for flag in session_mask.tolist():
            counter = counter + 1 if flag else 0
            values.append(counter)
        out.loc[idx] = values
    return out


def compute_noise_sigma(minute_df: pd.DataFrame, lookback: int) -> pd.Series:
    """Compute noise sigma(t,m) as rolling mean abs move-from-open over previous sessions."""
    rth_mask = _between_times(minute_df["timestamp"], "09:30:00", "16:00:00")
    base = minute_df.loc[rth_mask, ["session_date", "minute_of_day", "close", "open_rth"]].copy()
    base = base.dropna(subset=["open_rth"])  # only sessions with known RTH open
    base["abs_move_from_open"] = (base["close"] / base["open_rth"] - 1.0).abs()

    pivot = base.pivot_table(index="session_date", columns="minute_of_day", values="abs_move_from_open", aggfunc="last")
    min_periods = max(2, int(round(lookback * 0.6)))
    min_periods = min(int(lookback), int(min_periods))
    sigma = pivot.shift(1).rolling(lookback, min_periods=min_periods).mean()

    sigma_long = (
        sigma.stack(dropna=False)
        .rename("noise_sigma")
        .reset_index()
        .rename(columns={"level_1": "minute_of_day"})
    )

    merged = minute_df[["session_date", "minute_of_day"]].merge(
        sigma_long,
        on=["session_date", "minute_of_day"],
        how="left",
    )
    return pd.to_numeric(merged["noise_sigma"], errors="coerce")


def dynamic_gate_mask(
    candidate_df: pd.DataFrame,
    config: DynamicThresholdConfig,
    noise_sigma: pd.Series | None,
    atr_col: str,
) -> pd.Series:
    """Return per-row dynamic-threshold pass mask (True when disabled)."""
    if config.mode == "disabled":
        return pd.Series(True, index=candidate_df.index)

    close = pd.to_numeric(candidate_df["close"], errors="coerce")
    or_high = pd.to_numeric(candidate_df["or_high"], errors="coerce")
    threshold = pd.Series(np.nan, index=candidate_df.index, dtype=float)

    if config.mode in {"noise_area_gate", "noise_area_gate_plus_close_confirmation", "noise_area_gate_plus_discrete_schedule"}:
        if noise_sigma is None:
            return pd.Series(False, index=candidate_df.index)

        sigma = pd.to_numeric(noise_sigma, errors="coerce").reindex(candidate_df.index)
        base_ref = pd.to_numeric(candidate_df["open_prevclose_max"], errors="coerce")

        upper_noise = base_ref * (1.0 + float(config.noise_vm) * sigma)
        noise_abs = (base_ref * sigma).abs()
        if config.threshold_style == "max_or_high_noise":
            threshold = np.maximum(or_high, upper_noise)
        elif config.threshold_style == "or_high_plus_k_noise_abs":
            threshold = or_high + float(config.noise_k) * noise_abs
        else:
            raise ValueError(f"Unsupported threshold_style: {config.threshold_style}")

    elif config.mode == "atr_threshold_gate":
        atr_value = pd.to_numeric(candidate_df.get(atr_col, np.nan), errors="coerce")
        threshold = or_high + float(config.atr_k) * atr_value

    elif config.mode == "close_confirmation_gate":
        threshold = or_high

    else:
        raise ValueError(f"Unsupported dynamic mode: {config.mode}")

    above = close > threshold
    confirm_bars = max(1, int(config.confirm_bars))
    streak = _consecutive_true_within_session(above.fillna(False), candidate_df["session_date"])
    confirm_ok = streak >= confirm_bars

    schedule = config.schedule
    if config.mode == "close_confirmation_gate":
        schedule = "continuous_on_bar_close"
    if config.mode == "noise_area_gate":
        schedule = "continuous_on_bar_close"

    schedule_ok = _schedule_mask(candidate_df["minute_of_day"], schedule)
    return above.fillna(False) & confirm_ok & schedule_ok


def compression_mask(candidate_df: pd.DataFrame, config: CompressionConfig) -> pd.Series:
    """Return session-level compression pattern mask mapped to candidate rows."""
    mode = config.mode
    if mode == "none":
        return pd.Series(True, index=candidate_df.index)

    column_map = {
        "nr4": "pattern_nr4",
        "nr7": "pattern_nr7",
        "triangle": "pattern_triangle",
        "nr4_or_nr7": "pattern_nr4_or_nr7",
        "nr4_or_triangle": "pattern_nr4_or_triangle",
        "nr4_or_nr7_or_triangle": "pattern_nr4_or_nr7_or_triangle",
        "inside_day": "pattern_inside_day",
        "outside_day": "pattern_outside_day",
        "strong_close": "pattern_strong_close",
        "weak_close": "pattern_weak_close",
    }
    col = column_map.get(mode)
    if col is None or col not in candidate_df.columns:
        raise ValueError(f"Unsupported compression mode: {mode}")
    return candidate_df[col].fillna(False).astype(bool)


def first_pass_signal_rows(
    candidate_df: pd.DataFrame,
    pass_mask: pd.Series,
) -> pd.DataFrame:
    """Select first passing candidate row per session (one trade max/day)."""
    working = candidate_df.loc[pass_mask.fillna(False)].copy()
    if working.empty:
        return working
    working = working.sort_values("timestamp")
    return working.groupby("session_date", sort=True).head(1).copy()


def calibrate_ensemble_thresholds(
    selected_signal_rows: pd.DataFrame,
    is_sessions: list,
    atr_col: str,
    q_lows_pct: tuple[int, ...],
    q_highs_pct: tuple[int, ...],
) -> list[tuple[float, float]]:
    """Calibrate ATR quantile bands on IS only."""
    is_values = pd.to_numeric(
        selected_signal_rows.loc[selected_signal_rows["session_date"].isin(set(is_sessions)), atr_col],
        errors="coerce",
    ).dropna()
    thresholds: list[tuple[float, float]] = []
    for q_low in q_lows_pct:
        for q_high in q_highs_pct:
            if q_low >= q_high:
                continue
            low = float(is_values.quantile(float(q_low) / 100.0)) if not is_values.empty else np.nan
            high = float(is_values.quantile(float(q_high) / 100.0)) if not is_values.empty else np.nan
            if math.isfinite(low) and math.isfinite(high) and low < high:
                thresholds.append((low, high))
    return thresholds


def apply_ensemble_selection(
    selected_signal_rows: pd.DataFrame,
    atr_col: str,
    thresholds: list[tuple[float, float]],
    vote_threshold: float,
    compression_config: CompressionConfig,
) -> pd.DataFrame:
    """Apply ATR-ensemble vote and optional compression soft-vote bonus."""
    out = selected_signal_rows.copy()
    if out.empty:
        out["ensemble_score"] = pd.Series(dtype=float)
        out["ensemble_selected"] = pd.Series(dtype=bool)
        return out

    atr = pd.to_numeric(out[atr_col], errors="coerce")
    if not thresholds:
        out["ensemble_score"] = 0.0
        out["ensemble_selected"] = False
        return out

    pass_columns: list[pd.Series] = []
    for low, high in thresholds:
        pass_columns.append(atr.between(low, high, inclusive="both"))

    pass_frame = pd.concat(pass_columns, axis=1)
    pass_count = pass_frame.sum(axis=1).astype(float)
    n_votes = float(pass_frame.shape[1])

    if compression_config.usage == "soft_vote_bonus" and compression_config.mode != "none":
        comp = compression_mask(out, compression_config).astype(float)
        bonus = float(max(0.0, compression_config.soft_bonus_votes))
        score = (pass_count + bonus * comp) / (n_votes + bonus if n_votes + bonus > 0 else 1.0)
    else:
        score = pass_count / n_votes

    out["ensemble_score"] = score
    out["ensemble_selected"] = out["ensemble_score"] >= float(vote_threshold)
    return out


def build_signal_frame_for_backtest(
    minute_df: pd.DataFrame,
    selected_signal_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return a backtest-ready dataframe with sparse signal column."""
    if selected_signal_rows.empty:
        out = minute_df.copy()
        out["signal"] = 0
        return out.iloc[:0].copy()

    selected_indices = pd.Index(selected_signal_rows.index)
    selected_sessions = set(pd.to_datetime(selected_signal_rows["session_date"]).dt.date)
    out = minute_df.loc[pd.to_datetime(minute_df["session_date"]).dt.date.isin(selected_sessions)].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["signal"] = 0

    # Preserve selection by timestamp + session_date after reindexing.
    key = selected_signal_rows[["session_date", "timestamp"]].copy()
    key["_selected"] = 1
    out = out.merge(key, on=["session_date", "timestamp"], how="left")
    out.loc[out["_selected"].eq(1), "signal"] = 1
    out = out.drop(columns=["_selected"])
    return out


def describe_baseline_entry_config(config: BaselineEntryConfig) -> dict[str, object]:
    return asdict(config)
