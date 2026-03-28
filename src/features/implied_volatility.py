"""Daily VIX/VVIX features built with strict t-1 alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import DATA_DIR


RAW_DATA_DIR = DATA_DIR / "raw"
DEFAULT_VIX_DAILY_PATH = RAW_DATA_DIR / "vix-daily.csv"
DEFAULT_VVIX_DAILY_PATH = RAW_DATA_DIR / "vvix-daily.csv"


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.where(den.ne(0)).divide(den.where(den.ne(0)))


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)

    def _percentile_last(window_values: np.ndarray) -> float:
        if np.isnan(window_values).any():
            return float("nan")
        current = float(window_values[-1])
        return float(np.mean(window_values <= current))

    return values.rolling(window, min_periods=window).apply(_percentile_last, raw=True)


def _finalize_daily_ohlc(raw: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "source_date": pd.to_datetime(raw[date_col], errors="coerce"),
            "close": pd.to_numeric(raw[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["source_date", "close"]).copy()
    out["source_date"] = out["source_date"].dt.normalize()
    return out.sort_values("source_date").drop_duplicates("source_date", keep="last").reset_index(drop=True)


def _load_standard_ohlc_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    columns = {str(column).strip().lower(): str(column) for column in raw.columns}
    if "date" not in columns or "close" not in columns:
        raise ValueError(f"Expected DATE/CLOSE columns in {path}.")
    return _finalize_daily_ohlc(raw, date_col=columns["date"], close_col=columns["close"])


def _load_yfinance_style_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, skiprows=[1, 2])
    if raw.empty:
        raise ValueError(f"No rows found in {path}.")
    date_col = str(raw.columns[0])
    close_col = next((str(column) for column in raw.columns if str(column).strip().lower() == "close"), None)
    if close_col is None:
        raise ValueError(f"Expected a Close column in {path}.")
    return _finalize_daily_ohlc(raw, date_col=date_col, close_col=close_col)


def load_daily_index_series(path: Path) -> pd.DataFrame:
    """Load a daily index series from the raw VIX/VVIX CSV formats used in the repo."""

    if not path.exists():
        raise FileNotFoundError(f"Daily implied-vol source file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().lower()

    if first_line.startswith("price,"):
        return _load_yfinance_style_csv(path)
    return _load_standard_ohlc_csv(path)


def load_vix_daily(path: Path = DEFAULT_VIX_DAILY_PATH) -> pd.DataFrame:
    return load_daily_index_series(path)


def load_vvix_daily(path: Path = DEFAULT_VVIX_DAILY_PATH) -> pd.DataFrame:
    return load_daily_index_series(path)


def _single_series_t1_features(
    daily: pd.DataFrame,
    prefix: str,
    percentile_windows: tuple[int, ...] = (63, 126, 252),
) -> pd.DataFrame:
    ordered = daily.sort_values("source_date").reset_index(drop=True).copy()
    ordered["session_date"] = ordered["source_date"].dt.date
    ordered[f"{prefix}_reference_date_t1"] = ordered["source_date"].shift(1).dt.date
    ordered[f"{prefix}_level_t1"] = pd.to_numeric(ordered["close"], errors="coerce").shift(1)

    lagged_level = pd.to_numeric(ordered[f"{prefix}_level_t1"], errors="coerce")
    for window in percentile_windows:
        ordered[f"{prefix}_pct_{int(window)}_t1"] = _rolling_percentile(lagged_level, int(window))

    ordered[f"{prefix}_change_1d"] = lagged_level.pct_change(1)
    ordered[f"{prefix}_change_5d"] = lagged_level.pct_change(5)

    keep_columns = ["session_date", f"{prefix}_reference_date_t1", f"{prefix}_level_t1"]
    keep_columns.extend(f"{prefix}_pct_{int(window)}_t1" for window in percentile_windows)
    keep_columns.extend([f"{prefix}_change_1d", f"{prefix}_change_5d"])
    return ordered[keep_columns].copy()


def build_vix_vvix_daily_features(
    vix_daily: pd.DataFrame,
    vvix_daily: pd.DataFrame,
    percentile_windows: tuple[int, ...] = (63, 126, 252),
) -> pd.DataFrame:
    """Return leak-free daily VIX/VVIX features aligned to the trade session date.

    For session date D, all `*_t1` fields use the previous available daily close
    known before the session open of D.
    """

    vix_features = _single_series_t1_features(vix_daily, prefix="vix", percentile_windows=percentile_windows)
    vvix_features = _single_series_t1_features(vvix_daily, prefix="vvix", percentile_windows=percentile_windows)

    out = vix_features.merge(vvix_features, on="session_date", how="inner", validate="one_to_one")
    out["vvix_over_vix_t1"] = _safe_divide(out["vvix_level_t1"], out["vix_level_t1"])
    return out.sort_values("session_date").reset_index(drop=True)


def load_vix_vvix_daily_features(
    vix_path: Path = DEFAULT_VIX_DAILY_PATH,
    vvix_path: Path = DEFAULT_VVIX_DAILY_PATH,
    percentile_windows: tuple[int, ...] = (63, 126, 252),
) -> pd.DataFrame:
    vix_daily = load_vix_daily(vix_path)
    vvix_daily = load_vvix_daily(vvix_path)
    return build_vix_vvix_daily_features(
        vix_daily=vix_daily,
        vvix_daily=vvix_daily,
        percentile_windows=percentile_windows,
    )
