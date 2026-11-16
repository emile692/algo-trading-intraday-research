"""Shared constants, loaders and metrics for the 2026-11-15 audit.

Read-only with respect to the repository: every loader reads frozen files, and
every writer targets ``audit_20261115/``.  Sharpe convention used throughout
(unless a reproduction explicitly needs the source convention):
sqrt(252) * mean / std(ddof=1) of the daily net PnL in USD, days without trade
included, on the calendar stated next to each figure.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent
OUT = AUDIT / "outputs"
CACHE = AUDIT / "cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEED = 20261115
ET = "America/New_York"

CODES = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09"]
NAMES = {
    "S01": "mnq_orb_vwap_atr", "S02": "mnq_volatility_breakout", "S03": "mnq_compression_breakout",
    "S04": "mnq_atr_trend_flip", "S05": "mgc_volatility_breakout", "S06": "mgc_gap_continuation",
    "S07": "mbt_rolling_range_breakout", "S08": "mbt_trend_pullback_ma_structure", "S09": "mbt_volatility_breakout",
}
CODE_OF = {v: k for k, v in NAMES.items()}
INSTRUMENT = {"S01": "MNQ", "S02": "MNQ", "S03": "MNQ", "S04": "MNQ", "S05": "MGC", "S06": "MGC",
              "S07": "MBT", "S08": "MBT", "S09": "MBT"}
ACTIVE8 = [c for c in CODES if c != "S04"]
POINT_VALUE = {"MNQ": 2.0, "MGC": 10.0, "MBT": 0.1}
TICK = {"MNQ": 0.25, "MGC": 0.1, "MBT": 5.0}

# Windows as frozen in the sleeves' campaign definitions.
WINDOWS = {
    "MNQ_MGC": {"full": ("2019-05-06", "2026-03-19"), "is": ("2019-05-06", "2023-12-31"),
                "val": ("2024-01-01", "2026-03-19")},
    "MBT": {"full": ("2021-05-03", "2026-10-02"), "is": ("2021-05-03", "2025-05-28"),
            "val": ("2025-05-29", "2026-10-02")},
    # S01 has its own historical split (70 % of sessions, Sunday rows included).
    "S01": {"full": ("2019-05-05", "2026-03-20"), "is": ("2019-05-05", "2024-02-27"),
            "val": ("2024-02-28", "2026-03-20")},
}
BOOK_WINDOW = ("2024-01-02", "2026-03-19")
DATA_CUTOFF_DECISION = "2026-11-01"


def window_of(code: str) -> dict[str, tuple[str, str]]:
    if code == "S01":
        return WINDOWS["S01"]
    return WINDOWS["MBT"] if INSTRUMENT[code] == "MBT" else WINDOWS["MNQ_MGC"]


def sleeve_dir(code: str) -> Path:
    if code == "S01":
        return ROOT / "research" / "signal_sheet_01_atr18"
    return ROOT / "research" / f"signal_sheet_{code[1:]}_cluster"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_daily(code: str) -> pd.Series:
    """Canonical frozen daily net PnL of one sleeve on its own source calendar."""
    d = pd.read_csv(sleeve_dir(code) / "daily_results.csv", parse_dates=["session_date"])
    return d.set_index("session_date")["daily_pnl_usd"].astype(float).rename(code)


def load_trades(code: str) -> pd.DataFrame:
    t = pd.read_csv(sleeve_dir(code) / "trades.csv")
    t["session_date"] = pd.to_datetime(t["session_date"])
    return t


def between(s: pd.Series | pd.DataFrame, a: str, b: str):
    idx = s.index if not isinstance(s, pd.DataFrame) or "session_date" not in s else s["session_date"]
    mask = (pd.DatetimeIndex(idx) >= pd.Timestamp(a)) & (pd.DatetimeIndex(idx) <= pd.Timestamp(b))
    return s[mask]


def sharpe(x) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(math.sqrt(252.0) * x.mean() / sd) if sd > 0 else float("nan")


def max_dd(x) -> float:
    eq = np.r_[0.0, np.cumsum(np.asarray(x, float))]
    return float((eq - np.maximum.accumulate(eq)).min())


def profit_factor(trade_pnl) -> float:
    p = np.asarray(trade_pnl, float)
    g, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(g / l) if l > 0 else float("nan")


def basic_metrics(daily: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, Any]:
    out = {"start": str(daily.index.min().date()) if len(daily) else None,
           "end": str(daily.index.max().date()) if len(daily) else None,
           "sessions": int(len(daily)), "net_pnl_usd": float(daily.sum()), "sharpe": sharpe(daily),
           "max_dd_usd": max_dd(daily), "worst_day_usd": float(daily.min()), "daily_vol_usd": float(daily.std(ddof=1))}
    if trades is not None:
        out["trades"] = int(len(trades))
        out["profit_factor"] = profit_factor(trades["net_pnl_usd"])
    return out


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def fr(x: float, nd: int = 2) -> str:
    """French number formatting: thin-space thousands, comma decimal, true minus sign."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/d"
    s = f"{abs(x):,.{nd}f}".replace(",", " ").replace(".", ",")
    return ("−" if x < 0 else "") + s


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def lock_daily_nine() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "research" / "locked_nine_sleeve_portfolio_20261101" / "daily_pnl.csv",
                    parse_dates=["session_date"]).set_index("session_date")
    return d.rename(columns=CODE_OF)
