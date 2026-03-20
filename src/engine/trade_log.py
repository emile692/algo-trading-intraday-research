"""Trade log helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

TRADE_LOG_COLUMNS = [
    "trade_id",
    "session_date",
    "direction",
    "quantity",
    "entry_time",
    "entry_price",
    "stop_price",
    "target_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "account_size_usd",
    "risk_per_trade_pct",
    "risk_budget_usd",
    "risk_per_contract_usd",
    "actual_risk_usd",
    "trade_risk_usd",
    "notional_usd",
    "leverage_used",
    "pnl_points",
    "pnl_ticks",
    "pnl_usd",
    "fees",
    "net_pnl_usd",
]


def empty_trade_log() -> pd.DataFrame:
    """Return an empty trade log dataframe with standard columns."""
    return pd.DataFrame(columns=TRADE_LOG_COLUMNS)


def trade_to_record(trade_id: int, data: dict[str, Any]) -> dict[str, Any]:
    """Build standardized trade record dict."""
    row = {col: None for col in TRADE_LOG_COLUMNS}
    row["trade_id"] = trade_id
    row.update(data)
    return row
