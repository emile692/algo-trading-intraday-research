"""Data layer: 1-minute bars -> RTH daily bars, Wilder ATR (known at J-1), RTH signal bars of `tf` minutes.

Conventions
- Index of the 1-minute frame: bar START time, America/New_York.  A bar of `tf` minutes starting at s is only known
  (closed) at s + tf; that end time is the earliest moment a signal may use it.
- RTH = bars starting in [09:30, 16:00) ET.  The "day" of the setups is the RTH session.
- Continuous v.0 series are unadjusted: a day is only comparable with its neighbours when they are on the same
  contract (`instrument_id`), which the signal layer checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ET = "America/New_York"
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    tick_size: float
    point_value: float

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


SPECS = {
    "NQ": ContractSpec("NQ", 0.25, 20.0),
    "ES": ContractSpec("ES", 0.25, 50.0),
    "MNQ": ContractSpec("MNQ", 0.25, 2.0),
    "MES": ContractSpec("MES", 0.25, 5.0),
}


def load_minutes(path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load a Databento 1m parquet: columns open/high/low/close/volume/instrument_id, ET index, + date/minute."""
    df = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume", "instrument_id"])
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC").tz_convert(ET) if idx.tz is None else idx.tz_convert(ET)
    df.index = idx
    df.index.name = "timestamp"
    df = df.sort_index()
    if start is not None:
        df = df.loc[pd.Timestamp(start, tz=ET):]
    if end is not None:
        df = df.loc[: pd.Timestamp(end, tz=ET) - pd.Timedelta(minutes=1)]
    df = df[~df.index.duplicated(keep="first")].copy()
    df["instrument_id"] = df["instrument_id"].astype("int64")
    return add_clock(df)


def add_clock(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    out["date"] = idx.normalize().tz_localize(None)
    out["minute"] = idx.hour * 60 + idx.minute
    return out


def rth(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["minute"] >= RTH_OPEN_MIN) & (df["minute"] < RTH_CLOSE_MIN)]


def rth_daily(df: pd.DataFrame, min_bars: int = 180) -> pd.DataFrame:
    """One row per RTH session: OHLC, bar count, contract, validity (enough bars, a single contract)."""
    r = rth(df)
    g = r.groupby("date", sort=True)
    d = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
        "n_bars": g.size(), "iid": g["instrument_id"].last(), "n_iid": g["instrument_id"].nunique(),
    })
    d["valid"] = (d["n_bars"] >= min_bars) & (d["n_iid"] == 1)
    return d


def wilder_atr_prev(daily: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR of the RTH sessions, value for day J computed on sessions up to J-1 only.

    The True Range of a session that changes contract against the previous one uses its own high-low only
    (no artificial roll gap)."""
    prev_close = daily["close"].shift(1)
    same = daily["iid"].eq(daily["iid"].shift(1))
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev_close).abs(),
                    (daily["low"] - prev_close).abs()], axis=1).max(axis=1)
    tr = tr.where(same, daily["high"] - daily["low"])
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr.shift(1).rename(f"atr{period}_prev")


def signal_bars(df: pd.DataFrame, tf: int, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """RTH bars of `tf` minutes aligned on 09:30, with their END time.  With `asof`, bars ending after it are dropped
    (an incomplete bar is never visible)."""
    r = rth(df)
    bucket = (r["minute"] - RTH_OPEN_MIN) // tf
    g = r.groupby([r["date"], bucket], sort=True)
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "iid": g["instrument_id"].last()})
    bars.index.names = ["date", "bucket"]
    bars = bars.reset_index()
    start_min = RTH_OPEN_MIN + bars["bucket"] * tf
    end_min = np.minimum(start_min + tf, RTH_CLOSE_MIN)
    bars["end"] = (bars["date"].dt.tz_localize(ET) + pd.to_timedelta(end_min, unit="m"))
    bars["end_min"] = end_min
    if asof is not None:
        bars = bars[bars["end"] <= asof]
    return bars.reset_index(drop=True)
