"""Shared helpers for volume climax pullback research campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.data.session import extract_rth

REPO_ROOT = Path(__file__).resolve().parents[2]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(numeric):
        return float(default)
    return float(numeric)


def pf_for_ranking(value: Any) -> float:
    numeric = safe_float(value, default=0.0)
    if np.isinf(numeric):
        return 3.0
    return float(np.clip(numeric, 0.0, 3.0))


def latest_path_for_symbol(symbol: str, input_paths: dict[str, Path] | None = None) -> Path:
    if input_paths is not None and symbol in input_paths:
        return Path(input_paths[symbol])
    files = sorted((REPO_ROOT / "data" / "processed" / "parquet").glob(f"{symbol}_c_0_1m_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No input dataset found for {symbol} in data/processed/parquet.")
    return files[-1]


def load_latest_reference_run(root: Path, prefix: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Reference root does not exist: {root}")
    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No reference run found in {root} for prefix {prefix!r}.")
    return candidates[-1]


def load_symbol_data(symbol: str, input_paths: dict[str, Path] | None = None) -> pd.DataFrame:
    source_path = latest_path_for_symbol(symbol, input_paths=input_paths)
    return clean_ohlcv(load_ohlcv_file(source_path))


def resample_rth_1h(raw: pd.DataFrame) -> pd.DataFrame:
    scoped = extract_rth(raw.copy())
    if scoped.empty:
        return scoped
    scoped["timestamp"] = pd.to_datetime(scoped["timestamp"], errors="coerce")
    scoped = scoped.set_index("timestamp").sort_index()
    bars = scoped.resample("1h", label="left", closed="left", origin="start_day", offset="30min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return bars.dropna(subset=["open", "high", "low", "close"]).reset_index()


def split_sessions(frame: pd.DataFrame, ratio: float = 0.7) -> tuple[list, list]:
    sessions = sorted(pd.to_datetime(frame["session_date"]).dt.date.unique())
    if len(sessions) < 2:
        raise ValueError("Need at least two sessions to build an IS/OOS split.")
    cut = max(1, int(len(sessions) * ratio))
    cut = min(cut, len(sessions) - 1)
    return sessions[:cut], sessions[cut:]


def filter_trades_by_sessions(trades: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    allowed = set(pd.to_datetime(pd.Index(sessions)).date)
    session_dates = pd.to_datetime(trades["session_date"]).dt.date
    return trades.loc[session_dates.isin(allowed)].copy()


def summarize_scope(trades: pd.DataFrame, signal_df: pd.DataFrame, sessions: list) -> dict[str, float | int]:
    metrics = compute_metrics(trades, signal_df=signal_df, session_dates=sessions)
    return {
        "net_pnl": float(metrics.get("cumulative_pnl", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "sharpe": float(metrics.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "nb_trades": int(metrics.get("n_trades", 0)),
        "expectancy": float(metrics.get("expectancy", 0.0)),
        "hit_rate": float(metrics.get("win_rate", 0.0)),
        "avg_trade": float(trades["net_pnl_usd"].mean()) if not trades.empty else 0.0,
        "raw_signal_count": int(metrics.get("raw_signal_count", 0)),
    }
