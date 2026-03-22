"""Opening Range Breakout strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.settings import DEFAULT_TICK_SIZE
from src.utils.time_utils import build_session_time

_SIDE_MODE_ALIASES = {
    "both": "both",
    "long": "long",
    "short": "short",
    "long_only": "long",
    "short_only": "short",
}
_DIRECTION_FILTER_MODES = {"none", "vwap_only", "ema_only", "vwap_and_ema"}


def _normalize_side_mode(side_mode: str) -> str:
    """Normalize supported side-mode aliases."""
    normalized = _SIDE_MODE_ALIASES.get(side_mode)
    if normalized is None:
        raise ValueError("direction must be one of both, long, short, long_only, short_only.")
    return normalized


@dataclass
class ORBStrategy:
    """Simple ORB signal generator with optional regime and trend filters."""

    or_minutes: int = 30
    direction: str = "both"
    one_trade_per_day: bool = True
    entry_buffer_ticks: int = 0
    stop_buffer_ticks: int = 0
    target_multiple: float = 1.5
    opening_time: str = "09:00:00"
    time_exit: str = "15:55"
    account_size_usd: float | None = None
    risk_per_trade_pct: float | None = None
    tick_size: float = DEFAULT_TICK_SIZE
    atr_period: int | None = None
    atr_min: float | None = None
    atr_max: float | None = None
    atr_regime: str = "none"
    direction_filter_mode: str = "none"
    vwap_confirmation: bool = False
    vwap_min_distance_ticks: int = 0
    vwap_column: str = "session_vwap"
    ema_length: int | None = None
    filter_price_col: str = "close"

    def _atr_column(self) -> str | None:
        if self.atr_period is None:
            return None
        return f"atr_{self.atr_period}"

    def _passes_atr_filter(self, row: pd.Series) -> bool:
        atr_col = self._atr_column()
        if self.atr_regime == "none" or atr_col is None:
            return True
        if atr_col not in row.index:
            raise ValueError(f"Missing ATR column '{atr_col}' required by the strategy.")

        atr_value = row[atr_col]
        if pd.isna(atr_value):
            return False
        if self.atr_min is not None and atr_value < self.atr_min:
            return False
        if self.atr_max is not None and atr_value > self.atr_max:
            return False
        return True

    def _ensure_ema_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.ema_length is None:
            return df

        ema_col = f"ema_{self.ema_length}"
        if ema_col not in df.columns:
            if self.filter_price_col not in df.columns:
                raise ValueError(
                    f"Missing filter price column '{self.filter_price_col}' required for EMA calculation."
                )
            df = df.copy()
            df[ema_col] = df[self.filter_price_col].ewm(span=self.ema_length, adjust=False).mean()
        return df

    def _effective_direction_filter_mode(self) -> str:
        mode = self.direction_filter_mode
        if mode not in _DIRECTION_FILTER_MODES:
            raise ValueError(
                "direction_filter_mode must be one of none, vwap_only, ema_only, vwap_and_ema."
            )

        if not self.vwap_confirmation:
            return mode

        if mode == "none":
            return "vwap_only"
        if mode == "ema_only":
            return "vwap_and_ema"
        return mode

    def _vwap_confirmation_buffer(self) -> float:
        if self.vwap_min_distance_ticks < 0:
            raise ValueError("vwap_min_distance_ticks must be >= 0.")
        return self.vwap_min_distance_ticks * self.tick_size

    def _passes_direction_filter(self, row: pd.Series, signal: int) -> bool:
        mode = self._effective_direction_filter_mode()
        if mode == "none":
            return True

        price = row.get(self.filter_price_col)
        if pd.isna(price):
            return False

        comparisons: list[tuple[str, float, float]] = []
        if mode in ("vwap_only", "vwap_and_ema"):
            if self.vwap_column not in row.index:
                raise ValueError(f"Missing '{self.vwap_column}' column required by the strategy.")
            comparisons.append((self.vwap_column, row[self.vwap_column], self._vwap_confirmation_buffer()))

        if mode in ("ema_only", "vwap_and_ema"):
            if self.ema_length is None:
                raise ValueError("ema_length must be provided when an EMA filter is enabled.")
            ema_col = f"ema_{self.ema_length}"
            if ema_col not in row.index:
                raise ValueError(f"Missing EMA column '{ema_col}' required by the strategy.")
            comparisons.append((ema_col, row[ema_col], 0.0))

        for _, reference_price, min_gap in comparisons:
            if pd.isna(reference_price):
                return False
            if signal == 1 and price <= reference_price + min_gap:
                return False
            if signal == -1 and price >= reference_price - min_gap:
                return False
        return True

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ORB entry signals based on OR high/low breaks."""
        out = self._ensure_ema_column(df.copy())
        out["signal"] = 0
        out["raw_signal"] = 0
        out["atr_filter_pass"] = True
        out["direction_filter_pass"] = True
        out["filter_pass"] = True
        out["filtered_out"] = False

        buffer = self.entry_buffer_ticks * self.tick_size
        side_mode = _normalize_side_mode(self.direction)

        for _, session_df in out.groupby("session_date", sort=True):
            if session_df.empty:
                continue

            session_df = session_df.sort_values("timestamp")
            session_start = build_session_time(session_df["timestamp"].iloc[0], self.opening_time)
            or_expiry = session_start + pd.Timedelta(minutes=self.or_minutes)
            eligible = session_df["timestamp"] >= or_expiry
            valid_or = session_df["or_high"].notna() & session_df["or_low"].notna()
            long_break = eligible & valid_or & (session_df["close"] >= session_df["or_high"] + buffer)
            short_break = eligible & valid_or & (session_df["close"] <= session_df["or_low"] - buffer)

            raw_signal = pd.Series(0, index=session_df.index, dtype=int)
            if side_mode in ("both", "long"):
                raw_signal.loc[long_break] = 1
            if side_mode in ("both", "short"):
                short_mask = short_break & raw_signal.eq(0)
                raw_signal.loc[short_mask] = -1

            candidate_mask = raw_signal.ne(0)
            if not candidate_mask.any():
                continue

            atr_pass = pd.Series(True, index=session_df.index, dtype=bool)
            atr_col = self._atr_column()
            if self.atr_regime != "none" and atr_col is not None:
                if atr_col not in session_df.columns:
                    raise ValueError(f"Missing ATR column '{atr_col}' required by the strategy.")
                atr_values = session_df[atr_col]
                atr_pass = atr_values.notna()
                if self.atr_min is not None:
                    atr_pass &= atr_values >= self.atr_min
                if self.atr_max is not None:
                    atr_pass &= atr_values <= self.atr_max

            direction_pass = pd.Series(True, index=session_df.index, dtype=bool)
            direction_filter_mode = self._effective_direction_filter_mode()
            if direction_filter_mode != "none":
                if self.filter_price_col not in session_df.columns:
                    raise ValueError(
                        f"Missing filter price column '{self.filter_price_col}' required by the strategy."
                    )

                price = session_df[self.filter_price_col]
                direction_pass = price.notna()
                reference_columns: list[tuple[str, float]] = []

                if direction_filter_mode in ("vwap_only", "vwap_and_ema"):
                    if self.vwap_column not in session_df.columns:
                        raise ValueError(f"Missing '{self.vwap_column}' column required by the strategy.")
                    reference_columns.append((self.vwap_column, self._vwap_confirmation_buffer()))

                if direction_filter_mode in ("ema_only", "vwap_and_ema"):
                    if self.ema_length is None:
                        raise ValueError("ema_length must be provided when an EMA filter is enabled.")
                    ema_col = f"ema_{self.ema_length}"
                    if ema_col not in session_df.columns:
                        raise ValueError(f"Missing EMA column '{ema_col}' required by the strategy.")
                    reference_columns.append((ema_col, 0.0))

                for ref_col, min_gap in reference_columns:
                    reference_price = session_df[ref_col]
                    valid_reference = reference_price.notna()
                    long_ok = raw_signal.ne(1) | (valid_reference & (price > reference_price + min_gap))
                    short_ok = raw_signal.ne(-1) | (valid_reference & (price < reference_price - min_gap))
                    direction_pass &= long_ok & short_ok

            filter_pass = candidate_mask & atr_pass & direction_pass
            considered_index = session_df.index
            if self.one_trade_per_day and filter_pass.any():
                first_signal_idx = filter_pass[filter_pass].index[0]
                cutoff_position = session_df.index.get_loc(first_signal_idx)
                considered_index = session_df.index[: cutoff_position + 1]

            considered_candidates = raw_signal.loc[considered_index].ne(0)
            candidate_index = raw_signal.loc[considered_index][considered_candidates].index
            if len(candidate_index) == 0:
                continue

            out.loc[candidate_index, "raw_signal"] = raw_signal.loc[candidate_index]
            out.loc[candidate_index, "atr_filter_pass"] = atr_pass.loc[candidate_index]
            out.loc[candidate_index, "direction_filter_pass"] = direction_pass.loc[candidate_index]
            out.loc[candidate_index, "filter_pass"] = filter_pass.loc[candidate_index]
            out.loc[candidate_index, "filtered_out"] = ~filter_pass.loc[candidate_index]

            if self.one_trade_per_day:
                passing_candidates = filter_pass.loc[candidate_index]
                if passing_candidates.any():
                    signal_index = passing_candidates[passing_candidates].index[0]
                    out.at[signal_index, "signal"] = int(raw_signal.at[signal_index])
            else:
                passing_index = filter_pass.loc[candidate_index][filter_pass.loc[candidate_index]].index
                out.loc[passing_index, "signal"] = raw_signal.loc[passing_index]

        return out
