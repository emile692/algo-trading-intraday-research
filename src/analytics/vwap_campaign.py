"""Full VWAP research campaign orchestration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analytics.vwap_metrics import (
    build_export_tables,
    build_long_short_stats,
    build_pnl_by_hour_table,
    build_rolling_metric_table,
    build_trade_hour_table,
    build_weekday_pnl_table,
    compute_extended_vwap_metrics,
)
from src.config.paths import EXPORTS_DIR, NOTEBOOKS_DIR, ensure_directories
from src.config.vwap_campaign import (
    VWAPCampaignSpec,
    VWAPVariantConfig,
    build_default_prop_constraints,
    build_default_vwap_variants,
    infer_symbol_from_dataset_path,
    resolve_default_vwap_dataset,
)
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.vwap_backtester import VWAPBacktestResult, build_execution_model_for_profile, run_vwap_backtest
from src.strategy.vwap import build_vwap_signal_frame, prepare_vwap_feature_frame


KEY_METRIC_COLUMNS = [
    "n_trades",
    "net_pnl",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "max_daily_drawdown",
    "worst_day",
    "worst_losing_days_streak",
    "worst_losing_trades_streak",
    "avg_trades_per_day",
    "max_trades_per_day",
    "expectancy_per_trade",
    "expectancy_per_day",
    "pct_green_days",
    "avg_gain_loss_ratio",
    "hit_rate",
    "avg_time_in_position_min",
    "days_to_target_pct",
    "daily_loss_limit_breach_freq",
    "trailing_drawdown_breach_freq",
    "empirical_prob_red_streak_ge_threshold",
    "profit_to_drawdown_ratio",
]


def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions to perform an IS/OOS split.")
    split_idx = int(len(all_sessions) * float(is_fraction))
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _subset_frame_by_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out_dates = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out_dates.isin(session_set)].copy().reset_index(drop=True)


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": metrics.get(key, np.nan) for key in KEY_METRIC_COLUMNS}


def _rank_prop_variants(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = results_df.copy()
    if ranked.empty:
        return ranked

    ranked["selection_score"] = (
        3.0 * ranked["oos_profit_to_drawdown_ratio"].replace([np.inf, -np.inf], 0.0).clip(-2.0, 5.0)
        + 2.0 * ranked["oos_sharpe_ratio"].clip(-2.0, 4.0)
        + 1.5 * ranked["oos_profit_factor"].replace([np.inf, -np.inf], 3.0).clip(0.0, 3.0)
        + 1.0 * ranked["oos_pct_green_days"].clip(0.0, 1.0)
        - 3.0 * (ranked["oos_daily_loss_limit_breach_freq"] * 10.0).clip(0.0, 5.0)
        - 3.0 * (ranked["oos_trailing_drawdown_breach_freq"] * 10.0).clip(0.0, 5.0)
        - 0.5 * ranked["oos_worst_losing_days_streak"].clip(0.0, 10.0)
        - 0.5 * ranked["oos_worst_losing_trades_streak"].clip(0.0, 10.0)
    )
    return ranked.sort_values(
        by=["selection_score", "oos_sharpe_ratio", "oos_profit_factor", "oos_expectancy_per_trade"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _paper_validation_markdown(symbol: str, metrics: dict[str, Any], hourly_table: pd.DataFrame) -> str:
    abs_hourly = pd.to_numeric(hourly_table["net_bar_pnl_usd"], errors="coerce").abs() if not hourly_table.empty else pd.Series(dtype=float)
    total_hourly_pnl = float(abs_hourly.sum()) if not hourly_table.empty else 0.0
    early_late = hourly_table.loc[
        hourly_table["hour"].isin(["09:00", "10:00", "11:00", "15:00"]),
        "net_bar_pnl_usd",
    ].abs().sum() if not hourly_table.empty else 0.0
    concentration_ratio = float(early_late / total_hourly_pnl) if total_hourly_pnl != 0 else 0.0

    lines = [
        "# Replication Sanity Check",
        "",
        "Paper reference stylized facts on QQQ:",
        "- Low hit rate around 17%",
        "- Large gain/loss asymmetry around 5.7x",
        "- High trade count around 21,967 trades",
        "- PnL concentrated in the opening phase and final hour",
        "- Trend-following intraday profile",
        "",
        f"Observed on `{symbol}` with this repo:",
        f"- Hit rate: {float(metrics.get('hit_rate', 0.0)):.2%}",
        f"- Average gain/loss ratio: {float(metrics.get('avg_gain_loss_ratio', 0.0)):.2f}",
        f"- Number of trades: {int(metrics.get('n_trades', 0))}",
        f"- Early+late session PnL share: {concentration_ratio:.2%}",
        f"- Profit factor: {float(metrics.get('profit_factor', 0.0)):.2f}",
        "",
        "Coherent points:",
    ]
    if float(metrics.get("hit_rate", 0.0)) <= 0.35:
        lines.append("- The hit rate remains low, consistent with a trend follower.")
    if float(metrics.get("avg_gain_loss_ratio", 0.0)) >= 2.0:
        lines.append("- Winners remain materially larger than losers.")
    if concentration_ratio >= 0.45:
        lines.append("- A large share of PnL still comes from the open and the final hour.")
    if float(metrics.get("profit_factor", 0.0)) > 1.0 and float(metrics.get("hit_rate", 0.0)) < 0.50:
        lines.append("- The profile is still low-hit-rate and trend-driven rather than mean-reverting.")

    lines.extend(
        [
            "",
            "Plausible divergences:",
            "- The tested underlying can differ from QQQ/TQQQ, especially on futures.",
            "- The repo uses explicit futures-like slippage and contract commissions when applicable.",
            "- Execution is forced at the next bar open with start-aligned timestamps.",
            "- RTH handling is explicit and excludes the synthetic `16:00` start bar.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _representative_days(best_daily: pd.DataFrame) -> pd.DataFrame:
    if best_daily.empty:
        return pd.DataFrame(columns=["label", "session_date"])

    trend_day = best_daily.sort_values("daily_pnl_usd", ascending=False).iloc[0]["session_date"]
    losing_day = best_daily.sort_values("daily_pnl_usd", ascending=True).iloc[0]["session_date"]

    choppy = best_daily.copy()
    median_abs = max(float(pd.to_numeric(choppy["daily_pnl_usd"], errors="coerce").abs().median()), 1.0)
    choppy["chop_score"] = (
        pd.to_numeric(choppy["daily_trade_count"], errors="coerce").fillna(0.0)
        - pd.to_numeric(choppy["daily_pnl_usd"], errors="coerce").abs() / median_abs
    )
    choppy_day = choppy.sort_values("chop_score", ascending=False).iloc[0]["session_date"]

    return pd.DataFrame(
        [
            {"label": "trend_day", "session_date": trend_day},
            {"label": "choppy_day", "session_date": choppy_day},
            {"label": "losing_day", "session_date": losing_day},
        ]
    )


def _plot_equity_comparison(
    baseline_curve: pd.DataFrame,
    best_curve: pd.DataFrame,
    baseline_name: str,
    best_name: str,
    output_path: Path,
) -> None:
    def _ts(frame: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if not baseline_curve.empty:
        axes[0].plot(_ts(baseline_curve), baseline_curve["equity"], label=baseline_name, linewidth=1.4)
        axes[1].plot(_ts(baseline_curve), baseline_curve["drawdown"], label=baseline_name, linewidth=1.4)
    if not best_curve.empty:
        axes[0].plot(_ts(best_curve), best_curve["equity"], label=best_name, linewidth=1.4)
        axes[1].plot(_ts(best_curve), best_curve["drawdown"], label=best_name, linewidth=1.4)
    axes[0].set_title("Equity Curve Comparison")
    axes[0].set_ylabel("Equity (USD)")
    axes[1].set_title("Drawdown Comparison")
    axes[1].set_ylabel("Drawdown (USD)")
    axes[1].set_xlabel("Timestamp")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    if not source.endswith("\n"):
        source = source + "\n"
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _path_expr_from_repo_root(path: Path) -> str:
    try:
        resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        rel = resolved.relative_to(ROOT)
        return "ROOT" + "".join(f' / "{part}"' for part in rel.parts)
    except Exception:
        return f'Path(r"{str(path)}")'


def _optional_literal(value: Any) -> str:
    return "None" if value is None else repr(value)


def _time_windows_literal(windows: tuple[Any, ...]) -> str:
    if not windows:
        return "()"
    lines = ["("]
    for window in windows:
        lines.append(f'    ("{window.start}", "{window.end}"),')
    lines.append(")")
    return "\n".join(lines)


def generate_vwap_validation_notebook(
    notebook_path: Path,
    output_dir: Path,
    spec: VWAPCampaignSpec,
    symbol: str,
    selected_variant: VWAPVariantConfig,
) -> Path:
    """Create the final executable notebook requested by the user."""
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    execution_model, instrument = build_execution_model_for_profile(
        symbol=symbol,
        profile_name=selected_variant.execution_profile,
    )
    dataset_expr = _path_expr_from_repo_root(spec.dataset_path)
    output_expr = _path_expr_from_repo_root(output_dir)
    time_windows_literal = _time_windows_literal(selected_variant.time_windows)

    intro_md = """# VWAP Strategy vs Buy and Hold (Client Friendly, Parametric)

Cette strategie reconstruit un trading intraday autour de la VWAP de session. Le notebook filtre d'abord les barres RTH entre `SESSION_START` et `SESSION_END`, recalcule la `session_vwap`, l'ATR et les structures intraday, puis applique la variante selectionnee. Pour une variante discrete comme `vwap_pullback_continuation`, un long n'est pris que si le marche reste dans un regime haussier au-dessus de la VWAP, que la pente de VWAP confirme ce biais, qu'un pullback reste propre par rapport a la VWAP, puis qu'une reprise de momentum valide l'entree; le short est le miroir exact. La sortie depend ensuite du recross de VWAP si `EXIT_ON_VWAP_RECROSS=True`, du stop structurel derive de l'ATR, ou de la cloture de session.

Les parametres interviennent sur trois couches. Les parametres de structure du signal comme `ATR_PERIOD`, `ATR_BUFFER`, `COMPRESSION_LENGTH`, `PULLBACK_LOOKBACK`, `SLOPE_LOOKBACK`, `SLOPE_THRESHOLD` et `TIME_WINDOWS` rendent l'entree plus ou moins selective: plus ils sont stricts, plus on reduit le nombre de trades pour privilegier les contextes juges propres. Les parametres de sizing et d'execution comme `QUANTITY_MODE`, `FIXED_QUANTITY`, `RISK_PER_TRADE_PCT`, `TICK_SIZE`, `POINT_VALUE_USD`, `COMMISSION_PER_SIDE_USD` et `SLIPPAGE_TICKS` ne changent pas le signal, mais transforment directement ce signal en exposition, en frais et donc en PnL reel. Enfin, les garde-fous prop comme `MAX_TRADES_PER_DAY`, `MAX_LOSSES_PER_DAY`, `DAILY_STOP_THRESHOLD_USD`, `CONSECUTIVE_LOSSES_THRESHOLD` et `DELEVERAGE_AFTER_LOSING_STREAK` coupent ou reduisent l'activite quand la journee ou la sequence de pertes se degrade, ce qui change surtout le profil de risque et la regularite du run.
"""

    setup_code = """from pathlib import Path
import sys

root = Path.cwd().resolve()
while root != root.parent and not (root / "pyproject.toml").exists():
    root = root.parent

if not (root / "pyproject.toml").exists():
    raise RuntimeError("Could not locate repository root from the current working directory.")

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

print(f"Project root: {root}")
"""

    imports_code = """import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Markdown, display

from src.analytics.orb_notebook_utils import (
    build_scope_readout_markdown,
    curve_annualized_return,
    curve_daily_sharpe,
    curve_daily_vol,
    curve_max_drawdown_pct,
    curve_total_return_pct,
    format_curve_stats_line,
    normalize_curve,
)
from src.analytics.vwap_metrics import compute_extended_vwap_metrics
from src.config.vwap_campaign import PropFirmConstraintConfig, TimeWindow, VWAPVariantConfig
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.engine.execution_model import ExecutionModel
from src.engine.portfolio import build_equity_curve
from src.engine.vwap_backtester import InstrumentDetails, run_vwap_backtest
from src.strategy.vwap import build_vwap_signal_frame, prepare_vwap_feature_frame

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 200)
"""

    config_code = f"""ROOT = root

# ----------------------
# Data / split settings
# ----------------------
DATASET_PATH = {dataset_expr}
OUTPUT_DIR = {output_expr}
SYMBOL = "{symbol}"
IS_FRACTION = {float(spec.is_fraction)}
SESSION_START = "{spec.session_start}"
SESSION_END = "{spec.session_end}"
VWAP_PRICE_MODE = "typical"
ROLLING_WINDOW_DAYS = {int(spec.rolling_window_days)}

# ----------------------
# Selected strategy settings
# ----------------------
SELECTED_VARIANT_NAME = "{selected_variant.name}"
VARIANT_FAMILY = "{selected_variant.family}"
MODE = "{selected_variant.mode}"
EXECUTION_PROFILE = "{selected_variant.execution_profile}"
INITIAL_CAPITAL_USD = {float(selected_variant.initial_capital_usd)}
QUANTITY_MODE = "{selected_variant.quantity_mode}"
FIXED_QUANTITY = {int(selected_variant.fixed_quantity)}
TIME_WINDOWS = {time_windows_literal}
SLOPE_LOOKBACK = {int(selected_variant.slope_lookback)}
SLOPE_THRESHOLD = {float(selected_variant.slope_threshold)}
ATR_PERIOD = {int(selected_variant.atr_period)}
ATR_BUFFER = {float(selected_variant.atr_buffer)}
COMPRESSION_LENGTH = {int(selected_variant.compression_length)}
PULLBACK_LOOKBACK = {int(selected_variant.pullback_lookback)}
MAX_TRADES_PER_DAY = {_optional_literal(selected_variant.max_trades_per_day)}
MAX_LOSSES_PER_DAY = {_optional_literal(selected_variant.max_losses_per_day)}
DAILY_STOP_THRESHOLD_USD = {_optional_literal(selected_variant.daily_stop_threshold_usd)}
CONSECUTIVE_LOSSES_THRESHOLD = {_optional_literal(selected_variant.consecutive_losses_threshold)}
DELEVERAGE_AFTER_LOSING_STREAK = {float(selected_variant.deleverage_after_losing_streak)}
RISK_PER_TRADE_PCT = {_optional_literal(selected_variant.risk_per_trade_pct)}
EXIT_ON_VWAP_RECROSS = {bool(selected_variant.exit_on_vwap_recross)}
USE_PARTIAL_EXIT = {bool(selected_variant.use_partial_exit)}
PARTIAL_EXIT_R_MULTIPLE = {float(selected_variant.partial_exit_r_multiple)}
KEEP_RUNNER_UNTIL_CLOSE = {bool(selected_variant.keep_runner_until_close)}
VARIANT_NOTES = {selected_variant.notes!r}

# ----------------------
# Instrument / execution settings
# ----------------------
ASSET_CLASS = "{instrument.asset_class}"
TICK_SIZE = {float(instrument.tick_size)}
TICK_VALUE_USD = {float(instrument.tick_value_usd)}
POINT_VALUE_USD = {float(instrument.point_value_usd)}
COMMISSION_PER_SIDE_USD = {float(execution_model.commission_per_side_usd)}
SLIPPAGE_TICKS = {int(execution_model.slippage_ticks)}

# ----------------------
# Prop-style evaluation settings
# ----------------------
PROP_ACCOUNT_SIZE_USD = {float(spec.prop_constraints.account_size_usd)}
PROFIT_TARGET_PCT = {float(spec.prop_constraints.profit_target_pct)}
DAILY_LOSS_LIMIT_USD = {float(spec.prop_constraints.daily_loss_limit_usd)}
TRAILING_DRAWDOWN_LIMIT_USD = {float(spec.prop_constraints.trailing_drawdown_limit_usd)}
CONSECUTIVE_RED_DAYS_THRESHOLD = {int(spec.prop_constraints.consecutive_red_days_threshold)}
TRADING_DAYS_PER_MONTH = {float(spec.prop_constraints.trading_days_per_month)}

# ----------------------
# Benchmark / notebook display settings
# ----------------------
BENCHMARK_LABEL = "Buy&Hold"
BENCHMARK_PRICE_COLUMN = "close"
BENCHMARK_INITIAL_CAPITAL_USD = INITIAL_CAPITAL_USD
RANKING_ROWS = 10
PLOT_TEMPLATE = "plotly_dark"
PLOT_WIDTH = 1800

print("DATASET_PATH =", DATASET_PATH)
print("OUTPUT_DIR =", OUTPUT_DIR)
print("SELECTED_VARIANT_NAME =", SELECTED_VARIANT_NAME)
print("MODE =", MODE)
print("INITIAL_CAPITAL_USD =", INITIAL_CAPITAL_USD)
print("RISK_PER_TRADE_PCT =", RISK_PER_TRADE_PCT)
print("COMMISSION_PER_SIDE_USD =", COMMISSION_PER_SIDE_USD)
print("SLIPPAGE_TICKS =", SLIPPAGE_TICKS)
"""

    snapshot_code = """def _format_windows(windows: tuple[tuple[str, str], ...]) -> str:
    if not windows:
        return "[]"
    return " | ".join([f"{start}->{end}" for start, end in windows])


parameter_snapshot = pd.DataFrame(
    [
        {"group": "data", "parameter": "DATASET_PATH", "value": str(DATASET_PATH)},
        {"group": "data", "parameter": "SYMBOL", "value": SYMBOL},
        {"group": "data", "parameter": "SESSION_START", "value": SESSION_START},
        {"group": "data", "parameter": "SESSION_END", "value": SESSION_END},
        {"group": "data", "parameter": "VWAP_PRICE_MODE", "value": VWAP_PRICE_MODE},
        {"group": "data", "parameter": "IS_FRACTION", "value": IS_FRACTION},
        {"group": "data", "parameter": "ROLLING_WINDOW_DAYS", "value": ROLLING_WINDOW_DAYS},
        {"group": "variant", "parameter": "SELECTED_VARIANT_NAME", "value": SELECTED_VARIANT_NAME},
        {"group": "variant", "parameter": "VARIANT_FAMILY", "value": VARIANT_FAMILY},
        {"group": "variant", "parameter": "MODE", "value": MODE},
        {"group": "variant", "parameter": "EXECUTION_PROFILE", "value": EXECUTION_PROFILE},
        {"group": "variant", "parameter": "INITIAL_CAPITAL_USD", "value": INITIAL_CAPITAL_USD},
        {"group": "variant", "parameter": "QUANTITY_MODE", "value": QUANTITY_MODE},
        {"group": "variant", "parameter": "FIXED_QUANTITY", "value": FIXED_QUANTITY},
        {"group": "variant", "parameter": "TIME_WINDOWS", "value": _format_windows(TIME_WINDOWS)},
        {"group": "variant", "parameter": "SLOPE_LOOKBACK", "value": SLOPE_LOOKBACK},
        {"group": "variant", "parameter": "SLOPE_THRESHOLD", "value": SLOPE_THRESHOLD},
        {"group": "variant", "parameter": "ATR_PERIOD", "value": ATR_PERIOD},
        {"group": "variant", "parameter": "ATR_BUFFER", "value": ATR_BUFFER},
        {"group": "variant", "parameter": "COMPRESSION_LENGTH", "value": COMPRESSION_LENGTH},
        {"group": "variant", "parameter": "PULLBACK_LOOKBACK", "value": PULLBACK_LOOKBACK},
        {"group": "variant", "parameter": "MAX_TRADES_PER_DAY", "value": MAX_TRADES_PER_DAY},
        {"group": "variant", "parameter": "MAX_LOSSES_PER_DAY", "value": MAX_LOSSES_PER_DAY},
        {"group": "variant", "parameter": "DAILY_STOP_THRESHOLD_USD", "value": DAILY_STOP_THRESHOLD_USD},
        {"group": "variant", "parameter": "CONSECUTIVE_LOSSES_THRESHOLD", "value": CONSECUTIVE_LOSSES_THRESHOLD},
        {"group": "variant", "parameter": "DELEVERAGE_AFTER_LOSING_STREAK", "value": DELEVERAGE_AFTER_LOSING_STREAK},
        {"group": "variant", "parameter": "RISK_PER_TRADE_PCT", "value": RISK_PER_TRADE_PCT},
        {"group": "variant", "parameter": "EXIT_ON_VWAP_RECROSS", "value": EXIT_ON_VWAP_RECROSS},
        {"group": "variant", "parameter": "USE_PARTIAL_EXIT", "value": USE_PARTIAL_EXIT},
        {"group": "variant", "parameter": "PARTIAL_EXIT_R_MULTIPLE", "value": PARTIAL_EXIT_R_MULTIPLE},
        {"group": "variant", "parameter": "KEEP_RUNNER_UNTIL_CLOSE", "value": KEEP_RUNNER_UNTIL_CLOSE},
        {"group": "variant", "parameter": "VARIANT_NOTES", "value": VARIANT_NOTES},
        {"group": "execution", "parameter": "ASSET_CLASS", "value": ASSET_CLASS},
        {"group": "execution", "parameter": "TICK_SIZE", "value": TICK_SIZE},
        {"group": "execution", "parameter": "TICK_VALUE_USD", "value": TICK_VALUE_USD},
        {"group": "execution", "parameter": "POINT_VALUE_USD", "value": POINT_VALUE_USD},
        {"group": "execution", "parameter": "COMMISSION_PER_SIDE_USD", "value": COMMISSION_PER_SIDE_USD},
        {"group": "execution", "parameter": "SLIPPAGE_TICKS", "value": SLIPPAGE_TICKS},
        {"group": "prop", "parameter": "PROP_ACCOUNT_SIZE_USD", "value": PROP_ACCOUNT_SIZE_USD},
        {"group": "prop", "parameter": "PROFIT_TARGET_PCT", "value": PROFIT_TARGET_PCT},
        {"group": "prop", "parameter": "DAILY_LOSS_LIMIT_USD", "value": DAILY_LOSS_LIMIT_USD},
        {"group": "prop", "parameter": "TRAILING_DRAWDOWN_LIMIT_USD", "value": TRAILING_DRAWDOWN_LIMIT_USD},
        {"group": "prop", "parameter": "CONSECUTIVE_RED_DAYS_THRESHOLD", "value": CONSECUTIVE_RED_DAYS_THRESHOLD},
        {"group": "prop", "parameter": "TRADING_DAYS_PER_MONTH", "value": TRADING_DAYS_PER_MONTH},
        {"group": "benchmark", "parameter": "BENCHMARK_LABEL", "value": BENCHMARK_LABEL},
        {"group": "benchmark", "parameter": "BENCHMARK_PRICE_COLUMN", "value": BENCHMARK_PRICE_COLUMN},
        {"group": "benchmark", "parameter": "BENCHMARK_INITIAL_CAPITAL_USD", "value": BENCHMARK_INITIAL_CAPITAL_USD},
    ]
)

display(parameter_snapshot)
display(Markdown(
    "### Notes\\n"
    "- `USE_PARTIAL_EXIT`, `PARTIAL_EXIT_R_MULTIPLE` and `KEEP_RUNNER_UNTIL_CLOSE` stay explicit here for completeness.\\n"
    "- In the current engine, the live logic mainly consumes the parameters that drive entry, stop, sizing, execution costs, prop limits and metrics."
))
"""

    load_code = """def _split_sessions(all_sessions: list, is_fraction: float) -> tuple[list, list]:
    if len(all_sessions) < 2:
        raise ValueError("Not enough sessions to perform an IS/OOS split.")
    split_idx = int(len(all_sessions) * float(is_fraction))
    split_idx = max(1, min(len(all_sessions) - 1, split_idx))
    return all_sessions[:split_idx], all_sessions[split_idx:]


def _subset_frame_by_sessions(frame: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = frame.copy()
    out_dates = pd.to_datetime(out["session_date"]).dt.date
    return out.loc[out_dates.isin(session_set)].copy().reset_index(drop=True)


def _subset_curve_to_sessions(curve: pd.DataFrame, sessions: list) -> pd.DataFrame:
    if curve.empty:
        return curve.copy()
    session_set = set(pd.to_datetime(pd.Index(sessions)).date)
    out = curve.copy()
    curve_dates = pd.to_datetime(out["timestamp"], errors="coerce").dt.date
    return out.loc[curve_dates.isin(session_set)].copy().reset_index(drop=True)


def _build_buy_hold_curve(frame: pd.DataFrame, initial_capital: float, price_column: str = "close") -> pd.DataFrame:
    bench_src = frame[["timestamp", price_column, "session_date"]].copy()
    bench_src["timestamp"] = pd.to_datetime(bench_src["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    bench_src[price_column] = pd.to_numeric(bench_src[price_column], errors="coerce")
    bench_src = bench_src.dropna(subset=["timestamp", price_column])
    daily_close = bench_src.groupby("session_date")[price_column].last().dropna()
    if daily_close.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown", "drawdown_pct"])

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(daily_close.index),
            "equity": float(initial_capital) * (daily_close / float(daily_close.iloc[0])),
        }
    ).sort_values("timestamp")
    out["peak"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] - out["peak"]
    return normalize_curve(out.drop(columns=["peak"]).reset_index(drop=True))


VARIANT = VWAPVariantConfig(
    name=SELECTED_VARIANT_NAME,
    family=VARIANT_FAMILY,
    mode=MODE,
    execution_profile=EXECUTION_PROFILE,
    initial_capital_usd=INITIAL_CAPITAL_USD,
    quantity_mode=QUANTITY_MODE,
    fixed_quantity=FIXED_QUANTITY,
    time_windows=tuple(TimeWindow(start, end) for start, end in TIME_WINDOWS),
    slope_lookback=SLOPE_LOOKBACK,
    slope_threshold=SLOPE_THRESHOLD,
    atr_period=ATR_PERIOD,
    atr_buffer=ATR_BUFFER,
    compression_length=COMPRESSION_LENGTH,
    pullback_lookback=PULLBACK_LOOKBACK,
    max_trades_per_day=MAX_TRADES_PER_DAY,
    max_losses_per_day=MAX_LOSSES_PER_DAY,
    daily_stop_threshold_usd=DAILY_STOP_THRESHOLD_USD,
    consecutive_losses_threshold=CONSECUTIVE_LOSSES_THRESHOLD,
    deleverage_after_losing_streak=DELEVERAGE_AFTER_LOSING_STREAK,
    risk_per_trade_pct=RISK_PER_TRADE_PCT,
    exit_on_vwap_recross=EXIT_ON_VWAP_RECROSS,
    use_partial_exit=USE_PARTIAL_EXIT,
    partial_exit_r_multiple=PARTIAL_EXIT_R_MULTIPLE,
    keep_runner_until_close=KEEP_RUNNER_UNTIL_CLOSE,
    notes=VARIANT_NOTES,
)

INSTRUMENT = InstrumentDetails(
    symbol=SYMBOL,
    asset_class=ASSET_CLASS,
    tick_size=TICK_SIZE,
    tick_value_usd=TICK_VALUE_USD,
    point_value_usd=POINT_VALUE_USD,
    commission_per_side_usd=COMMISSION_PER_SIDE_USD,
    slippage_ticks=SLIPPAGE_TICKS,
)

EXECUTION_MODEL = ExecutionModel(
    commission_per_side_usd=COMMISSION_PER_SIDE_USD,
    slippage_ticks=SLIPPAGE_TICKS,
    tick_size=TICK_SIZE,
)

PROP_CONSTRAINTS = PropFirmConstraintConfig(
    name="notebook_prop_reference",
    account_size_usd=PROP_ACCOUNT_SIZE_USD,
    profit_target_pct=PROFIT_TARGET_PCT,
    daily_loss_limit_usd=DAILY_LOSS_LIMIT_USD,
    trailing_drawdown_limit_usd=TRAILING_DRAWDOWN_LIMIT_USD,
    consecutive_red_days_threshold=CONSECUTIVE_RED_DAYS_THRESHOLD,
    trading_days_per_month=TRADING_DAYS_PER_MONTH,
)

raw = load_ohlcv_file(DATASET_PATH)
clean = clean_ohlcv(raw)
feature_frame = prepare_vwap_feature_frame(
    clean,
    session_start=SESSION_START,
    session_end=SESSION_END,
    atr_windows=(ATR_PERIOD,),
    vwap_price_mode=VWAP_PRICE_MODE,
)
signal_df = build_vwap_signal_frame(feature_frame, VARIANT)
result = run_vwap_backtest(signal_df, VARIANT, EXECUTION_MODEL, INSTRUMENT)

all_sessions = sorted(pd.to_datetime(feature_frame["session_date"]).dt.date.unique())
is_sessions, oos_sessions = _split_sessions(all_sessions, IS_FRACTION)

metrics_overall, rolling_table = compute_extended_vwap_metrics(
    trades=result.trades,
    daily_results=result.daily_results,
    bar_results=result.bar_results,
    signal_df=signal_df,
    initial_capital=INITIAL_CAPITAL_USD,
    prop_constraints=PROP_CONSTRAINTS,
    rolling_window_days=ROLLING_WINDOW_DAYS,
)
metrics_is, _ = compute_extended_vwap_metrics(
    trades=_subset_frame_by_sessions(result.trades, is_sessions),
    daily_results=_subset_frame_by_sessions(result.daily_results, is_sessions),
    bar_results=_subset_frame_by_sessions(result.bar_results, is_sessions),
    signal_df=_subset_frame_by_sessions(signal_df, is_sessions),
    initial_capital=INITIAL_CAPITAL_USD,
    prop_constraints=PROP_CONSTRAINTS,
    rolling_window_days=ROLLING_WINDOW_DAYS,
)
metrics_oos, _ = compute_extended_vwap_metrics(
    trades=_subset_frame_by_sessions(result.trades, oos_sessions),
    daily_results=_subset_frame_by_sessions(result.daily_results, oos_sessions),
    bar_results=_subset_frame_by_sessions(result.bar_results, oos_sessions),
    signal_df=_subset_frame_by_sessions(signal_df, oos_sessions),
    initial_capital=INITIAL_CAPITAL_USD,
    prop_constraints=PROP_CONSTRAINTS,
    rolling_window_days=ROLLING_WINDOW_DAYS,
)

selected_curve = normalize_curve(build_equity_curve(result.trades, initial_capital=INITIAL_CAPITAL_USD))
selected_curve_oos = normalize_curve(
    build_equity_curve(
        _subset_frame_by_sessions(result.trades, oos_sessions),
        initial_capital=INITIAL_CAPITAL_USD,
    )
)

feature_oos = feature_frame.loc[pd.to_datetime(feature_frame["session_date"]).dt.date.isin(set(oos_sessions))].copy()
benchmark_curve = _build_buy_hold_curve(
    feature_frame,
    BENCHMARK_INITIAL_CAPITAL_USD,
    price_column=BENCHMARK_PRICE_COLUMN,
)
benchmark_curve_is = _subset_curve_to_sessions(benchmark_curve, is_sessions)
benchmark_curve_oos = _build_buy_hold_curve(
    feature_oos,
    BENCHMARK_INITIAL_CAPITAL_USD,
    price_column=BENCHMARK_PRICE_COLUMN,
)

ranking_path = OUTPUT_DIR / "summary" / "prop_variant_ranking.csv"
comparative_path = OUTPUT_DIR / "summary" / "comparative_metrics.csv"
ranking_df = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()
comparative_df = pd.read_csv(comparative_path) if comparative_path.exists() else pd.DataFrame()

print("Sessions total / IS / OOS =", len(all_sessions), "/", len(is_sessions), "/", len(oos_sessions))
print("Trades overall / OOS =", len(result.trades), "/", len(_subset_frame_by_sessions(result.trades, oos_sessions)))
print("Net PnL overall =", float(metrics_overall.get("net_pnl", 0.0)))
print("Sharpe overall / OOS =", float(metrics_overall.get("sharpe_ratio", 0.0)), "/", float(metrics_oos.get("sharpe_ratio", 0.0)))
"""

    equity_code = """selected_ret = curve_total_return_pct(selected_curve, INITIAL_CAPITAL_USD)
selected_dd_pct = curve_max_drawdown_pct(selected_curve)
selected_cagr_pct = curve_annualized_return(selected_curve, INITIAL_CAPITAL_USD) * 100.0
selected_vol_pct = curve_daily_vol(selected_curve) * 100.0
selected_overall_sh = float(metrics_overall.get("sharpe_ratio", 0.0))
selected_overall_pf = float(metrics_overall.get("profit_factor", 0.0))
selected_overall_exp = float(metrics_overall.get("expectancy_per_trade", 0.0))

selected_oos_ret = curve_total_return_pct(selected_curve_oos, INITIAL_CAPITAL_USD)
selected_oos_dd_pct = curve_max_drawdown_pct(selected_curve_oos)
selected_oos_sh = float(metrics_oos.get("sharpe_ratio", 0.0))
selected_oos_pf = float(metrics_oos.get("profit_factor", 0.0))
selected_oos_exp = float(metrics_oos.get("expectancy_per_trade", 0.0))

bench_ret = curve_total_return_pct(benchmark_curve, BENCHMARK_INITIAL_CAPITAL_USD)
bench_dd_pct = curve_max_drawdown_pct(benchmark_curve)
bench_sh = curve_daily_sharpe(benchmark_curve)
bench_cagr_pct = curve_annualized_return(benchmark_curve, BENCHMARK_INITIAL_CAPITAL_USD) * 100.0
bench_vol_pct = curve_daily_vol(benchmark_curve) * 100.0
bench_is_sh = curve_daily_sharpe(benchmark_curve_is)
bench_oos_sh = curve_daily_sharpe(benchmark_curve_oos)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.70, 0.30],
    subplot_titles=("Equity Curve (USD)", "Drawdown (%)"),
)

fig.add_trace(
    go.Scatter(
        x=selected_curve["timestamp"],
        y=selected_curve["equity"],
        mode="lines",
        name=f"{SELECTED_VARIANT_NAME} Full Sample | Ret {selected_ret:.1f}% | Sharpe {selected_overall_sh:.2f} | PF {selected_overall_pf:.2f} | Exp/trade {selected_overall_exp:.1f}",
        line=dict(width=3.0, color="#22c55e"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=selected_curve["timestamp"],
        y=selected_curve["drawdown_pct"],
        mode="lines",
        name="DD Selected Variant",
        showlegend=False,
        line=dict(width=1.7, color="#22c55e", dash="dot"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=selected_curve_oos["timestamp"],
        y=selected_curve_oos["equity"],
        mode="lines",
        name=f"{SELECTED_VARIANT_NAME} OOS Only | Ret {selected_oos_ret:.1f}% | Sharpe {selected_oos_sh:.2f} | PF {selected_oos_pf:.2f} | Exp/trade {selected_oos_exp:.1f}",
        line=dict(width=2.4, color="#f59e0b"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=selected_curve_oos["timestamp"],
        y=selected_curve_oos["drawdown_pct"],
        mode="lines",
        name="DD Selected Variant OOS",
        showlegend=False,
        line=dict(width=1.5, color="#f59e0b", dash="dot"),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=benchmark_curve["timestamp"],
        y=benchmark_curve["equity"],
        mode="lines",
        name=f"{BENCHMARK_LABEL} | Ret {bench_ret:.1f}% | MaxDD {bench_dd_pct:.1f}%",
        line=dict(width=2.6, color="#38bdf8"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=benchmark_curve["timestamp"],
        y=benchmark_curve["drawdown_pct"],
        mode="lines",
        name="DD Buy&Hold",
        showlegend=False,
        line=dict(width=1.5, color="#38bdf8", dash="dot"),
    ),
    row=2,
    col=1,
)

fig.update_layout(
    template=PLOT_TEMPLATE,
    width=int(PLOT_WIDTH),
    height=900,
    title=f"VWAP Selected Variant vs {BENCHMARK_LABEL} | {SYMBOL}",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
)
fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
fig.show()

display(Markdown(build_scope_readout_markdown(
    full_curve=selected_curve,
    oos_curve=selected_curve_oos,
    initial_capital=INITIAL_CAPITAL_USD,
    full_label=f"{SELECTED_VARIANT_NAME} full-sample curve",
    oos_label=f"{SELECTED_VARIANT_NAME} OOS-only curve",
)))

display(Markdown(
    "### Strategy vs benchmark\\n"
    f"- {format_curve_stats_line(SELECTED_VARIANT_NAME, selected_overall_sh, selected_ret, selected_cagr_pct, selected_vol_pct, selected_dd_pct, pf=selected_overall_pf, exp=selected_overall_exp)}\\n"
    f"- {format_curve_stats_line(BENCHMARK_LABEL, bench_sh, bench_ret, bench_cagr_pct, bench_vol_pct, bench_dd_pct)}"
))
"""

    export_code = """if ranking_df.empty:
    print("No export ranking snapshot found in", OUTPUT_DIR / "summary")
else:
    display(Markdown(
        "### Export ranking snapshot\\n"
        "- This table comes from the saved campaign export in `OUTPUT_DIR`.\\n"
        "- If you manually edit the parameters above, this snapshot does not auto-refresh."
    ))
    display(
        ranking_df[
            [
                "name",
                "selection_score",
                "oos_net_pnl",
                "oos_sharpe_ratio",
                "oos_profit_factor",
                "oos_max_drawdown",
                "oos_daily_loss_limit_breach_freq",
                "oos_profit_to_drawdown_ratio",
            ]
        ].head(int(RANKING_ROWS))
    )
"""

    kpi_code = """kpi = pd.DataFrame([
    {
        "model": SELECTED_VARIANT_NAME,
        "overall_sharpe": selected_overall_sh,
        "is_sharpe": float(metrics_is.get("sharpe_ratio", 0.0)),
        "oos_sharpe": selected_oos_sh,
        "overall_profit_factor": selected_overall_pf,
        "oos_profit_factor": selected_oos_pf,
        "overall_expectancy_per_trade": selected_overall_exp,
        "oos_expectancy_per_trade": selected_oos_exp,
        "total_return_pct": selected_ret,
        "cagr_pct": selected_cagr_pct,
        "vol_pct": selected_vol_pct,
        "max_drawdown_pct": selected_dd_pct,
        "overall_net_pnl": float(metrics_overall.get("net_pnl", 0.0)),
        "oos_net_pnl": float(metrics_oos.get("net_pnl", 0.0)),
        "overall_n_trades": int(metrics_overall.get("n_trades", 0)),
        "oos_n_trades": int(metrics_oos.get("n_trades", 0)),
        "days_to_target_pct": metrics_overall.get("days_to_target_pct"),
        "daily_loss_limit_breach_freq": float(metrics_overall.get("daily_loss_limit_breach_freq", 0.0)),
        "trailing_drawdown_breach_freq": float(metrics_overall.get("trailing_drawdown_breach_freq", 0.0)),
        "profit_to_drawdown_ratio": float(metrics_overall.get("profit_to_drawdown_ratio", 0.0)),
    },
    {
        "model": BENCHMARK_LABEL,
        "overall_sharpe": bench_sh,
        "is_sharpe": bench_is_sh,
        "oos_sharpe": bench_oos_sh,
        "overall_profit_factor": np.nan,
        "oos_profit_factor": np.nan,
        "overall_expectancy_per_trade": np.nan,
        "oos_expectancy_per_trade": np.nan,
        "total_return_pct": bench_ret,
        "cagr_pct": bench_cagr_pct,
        "vol_pct": bench_vol_pct,
        "max_drawdown_pct": bench_dd_pct,
        "overall_net_pnl": float(benchmark_curve["equity"].iloc[-1] - BENCHMARK_INITIAL_CAPITAL_USD) if not benchmark_curve.empty else np.nan,
        "oos_net_pnl": float(benchmark_curve_oos["equity"].iloc[-1] - BENCHMARK_INITIAL_CAPITAL_USD) if not benchmark_curve_oos.empty else np.nan,
        "overall_n_trades": np.nan,
        "oos_n_trades": np.nan,
        "days_to_target_pct": np.nan,
        "daily_loss_limit_breach_freq": np.nan,
        "trailing_drawdown_breach_freq": np.nan,
        "profit_to_drawdown_ratio": np.nan,
    },
])
display(kpi)
"""

    notebook = {
        "cells": [
            _build_notebook_cell("markdown", intro_md),
            _build_notebook_cell("code", setup_code),
            _build_notebook_cell("code", imports_code),
            _build_notebook_cell("markdown", "## 1) Parameters (edit here)"),
            _build_notebook_cell("code", config_code),
            _build_notebook_cell("markdown", "## 2) Full parameter snapshot"),
            _build_notebook_cell("code", snapshot_code),
            _build_notebook_cell("markdown", "## 3) Build data, signals, backtest, and benchmark"),
            _build_notebook_cell("code", load_code),
            _build_notebook_cell("markdown", "## 4) Selected variant vs Buy and Hold"),
            _build_notebook_cell("code", equity_code),
            _build_notebook_cell("markdown", "## 5) KPI table"),
            _build_notebook_cell("code", kpi_code),
            _build_notebook_cell("markdown", "## 6) Export snapshot"),
            _build_notebook_cell("code", export_code),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def _export_variant_artifacts(
    output_root: Path,
    variant: VWAPVariantConfig,
    signal_df: pd.DataFrame,
    result: VWAPBacktestResult,
    metrics_overall: dict[str, Any],
    metrics_is: dict[str, Any],
    metrics_oos: dict[str, Any],
    rolling_table: pd.DataFrame,
) -> dict[str, Path]:
    variant_dir = output_root / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    export_tables = build_export_tables(
        trades=result.trades,
        daily_results=result.daily_results,
        bar_results=result.bar_results,
    )
    metrics_is_oos = pd.DataFrame(
        [
            {"split": "overall", **metrics_overall},
            {"split": "is", **metrics_is},
            {"split": "oos", **metrics_oos},
        ]
    )

    signal_df.to_parquet(variant_dir / "signal_frame.parquet", index=False)
    result.trades.to_csv(variant_dir / "trades.csv", index=False)
    result.trades.to_parquet(variant_dir / "trades.parquet", index=False)
    result.bar_results.to_parquet(variant_dir / "bar_results.parquet", index=False)
    result.daily_results.to_csv(variant_dir / "daily_results.csv", index=False)
    export_tables["equity_curve"].to_csv(variant_dir / "equity_curve.csv", index=False)
    export_tables["hourly_pnl"].to_csv(variant_dir / "hourly_pnl.csv", index=False)
    export_tables["trade_hourly"].to_csv(variant_dir / "trade_hourly.csv", index=False)
    export_tables["long_short"].to_csv(variant_dir / "long_short.csv", index=False)
    export_tables["weekday_pnl"].to_csv(variant_dir / "weekday_pnl.csv", index=False)
    pd.DataFrame([metrics_overall]).to_csv(variant_dir / "metrics_overall.csv", index=False)
    metrics_is_oos.to_csv(variant_dir / "metrics_is_oos.csv", index=False)
    rolling_table.to_csv(variant_dir / "rolling_20d_metrics.csv", index=False)

    return {
        "variant_dir": variant_dir,
        "trades_csv": variant_dir / "trades.csv",
        "signal_parquet": variant_dir / "signal_frame.parquet",
        "equity_curve_csv": variant_dir / "equity_curve.csv",
        "metrics_overall_csv": variant_dir / "metrics_overall.csv",
    }


def _run_sensitivity_suite(
    feature_df: pd.DataFrame,
    anchor_variant: VWAPVariantConfig,
    symbol: str,
    spec: VWAPCampaignSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    _, oos_sessions = _split_sessions(sorted(pd.to_datetime(feature_df["session_date"]).dt.date.unique()), spec.is_fraction)

    def _evaluate(candidate: VWAPVariantConfig, tag: str, cost_multiplier: float = 1.0) -> None:
        signal_df = build_vwap_signal_frame(feature_df, candidate)
        execution_model, instrument = build_execution_model_for_profile(
            symbol=symbol,
            profile_name=candidate.execution_profile,
            cost_multiplier=cost_multiplier,
        )
        result = run_vwap_backtest(signal_df, candidate, execution_model, instrument)
        oos_metrics, _ = compute_extended_vwap_metrics(
            trades=_subset_frame_by_sessions(result.trades, oos_sessions),
            daily_results=_subset_frame_by_sessions(result.daily_results, oos_sessions),
            bar_results=_subset_frame_by_sessions(result.bar_results, oos_sessions),
            signal_df=_subset_frame_by_sessions(signal_df, oos_sessions),
            initial_capital=candidate.initial_capital_usd,
            prop_constraints=spec.prop_constraints,
            rolling_window_days=spec.rolling_window_days,
        )
        rows.append(
            {
                "sensitivity_tag": tag,
                "variant_name": candidate.name,
                "cost_multiplier": cost_multiplier,
                "atr_buffer": candidate.atr_buffer,
                "slope_threshold": candidate.slope_threshold,
                "max_trades_per_day": candidate.max_trades_per_day,
                "time_windows": "|".join([f"{window.start}-{window.end}" for window in candidate.time_windows]),
                **_prefix_metrics(oos_metrics, "oos"),
            }
        )

    for multiplier in spec.sensitivity_cost_multipliers:
        _evaluate(anchor_variant, tag=f"cost_x_{str(multiplier).replace('.', 'p')}", cost_multiplier=multiplier)
    for atr_buffer in spec.sensitivity_atr_buffers:
        _evaluate(replace(anchor_variant, atr_buffer=atr_buffer), tag=f"atr_buffer_{str(atr_buffer).replace('.', 'p')}")
    for slope in spec.sensitivity_slope_thresholds:
        _evaluate(replace(anchor_variant, slope_threshold=slope), tag=f"slope_threshold_{str(slope).replace('.', 'p')}")
    for max_trades in spec.sensitivity_max_trades_per_day:
        _evaluate(replace(anchor_variant, max_trades_per_day=max_trades), tag=f"max_trades_{max_trades}")

    window_type = type(anchor_variant.time_windows[0]) if anchor_variant.time_windows else type(build_default_vwap_variants()[2].time_windows[0])
    default_windows = anchor_variant.time_windows or (
        window_type("09:35:00", "11:30:00"),
        window_type("15:00:00", "15:50:00"),
    )
    shorter_windows = (
        window_type("09:35:00", "11:00:00"),
        window_type("15:00:00", "15:45:00"),
    )
    _evaluate(replace(anchor_variant, time_windows=default_windows), tag="time_windows_default")
    _evaluate(replace(anchor_variant, time_windows=shorter_windows), tag="time_windows_shorter")

    return pd.DataFrame(rows)


def run_vwap_campaign(
    spec: VWAPCampaignSpec,
    output_dir: Path,
    notebook_path: Path | None = None,
    baseline_only: bool = False,
) -> dict[str, Path]:
    """Execute the full VWAP research campaign."""
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    symbol = infer_symbol_from_dataset_path(spec.dataset_path)
    variants = build_default_vwap_variants()
    if baseline_only:
        variants = [variant for variant in variants if variant.name == "paper_vwap_baseline"]

    raw = load_ohlcv_file(spec.dataset_path)
    clean = clean_ohlcv(raw)
    feature_df = prepare_vwap_feature_frame(
        clean,
        session_start=spec.session_start,
        session_end=spec.session_end,
        atr_windows=sorted({variant.atr_period for variant in variants}),
    )
    feature_df.to_parquet(summary_dir / "prepared_feature_frame.parquet", index=False)

    all_sessions = sorted(pd.to_datetime(feature_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, spec.is_fraction)

    rows: list[dict[str, Any]] = []
    artifacts_by_variant: dict[str, dict[str, Path]] = {}
    results_by_variant: dict[str, dict[str, Any]] = {}

    for variant in variants:
        signal_df = build_vwap_signal_frame(feature_df, variant)
        execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name=variant.execution_profile)
        result = run_vwap_backtest(signal_df, variant, execution_model, instrument)

        metrics_overall, rolling_table = compute_extended_vwap_metrics(
            trades=result.trades,
            daily_results=result.daily_results,
            bar_results=result.bar_results,
            signal_df=signal_df,
            initial_capital=variant.initial_capital_usd,
            prop_constraints=spec.prop_constraints,
            rolling_window_days=spec.rolling_window_days,
        )
        metrics_is, _ = compute_extended_vwap_metrics(
            trades=_subset_frame_by_sessions(result.trades, is_sessions),
            daily_results=_subset_frame_by_sessions(result.daily_results, is_sessions),
            bar_results=_subset_frame_by_sessions(result.bar_results, is_sessions),
            signal_df=_subset_frame_by_sessions(signal_df, is_sessions),
            initial_capital=variant.initial_capital_usd,
            prop_constraints=spec.prop_constraints,
            rolling_window_days=spec.rolling_window_days,
        )
        metrics_oos, _ = compute_extended_vwap_metrics(
            trades=_subset_frame_by_sessions(result.trades, oos_sessions),
            daily_results=_subset_frame_by_sessions(result.daily_results, oos_sessions),
            bar_results=_subset_frame_by_sessions(result.bar_results, oos_sessions),
            signal_df=_subset_frame_by_sessions(signal_df, oos_sessions),
            initial_capital=variant.initial_capital_usd,
            prop_constraints=spec.prop_constraints,
            rolling_window_days=spec.rolling_window_days,
        )

        artifacts_by_variant[variant.name] = _export_variant_artifacts(
            output_root=output_dir,
            variant=variant,
            signal_df=signal_df,
            result=result,
            metrics_overall=metrics_overall,
            metrics_is=metrics_is,
            metrics_oos=metrics_oos,
            rolling_table=rolling_table,
        )
        results_by_variant[variant.name] = {
            "variant": variant,
            "result": result,
            "metrics_overall": metrics_overall,
            "metrics_is": metrics_is,
            "metrics_oos": metrics_oos,
            "hourly_table": build_pnl_by_hour_table(result.bar_results),
            "trade_hour_table": build_trade_hour_table(result.trades),
            "long_short_table": build_long_short_stats(result.trades),
            "weekday_table": build_weekday_pnl_table(result.daily_results),
            "rolling_table": build_rolling_metric_table(result.daily_results, variant.initial_capital_usd, spec.rolling_window_days),
            "instrument": instrument,
        }

        rows.append(
            {
                "name": variant.name,
                "family": variant.family,
                "mode": variant.mode,
                "execution_profile": variant.execution_profile,
                "symbol": symbol,
                "initial_capital_usd": variant.initial_capital_usd,
                "notes": variant.notes,
                **_prefix_metrics(metrics_overall, "overall"),
                **_prefix_metrics(metrics_is, "is"),
                **_prefix_metrics(metrics_oos, "oos"),
            }
        )

    comparative_df = pd.DataFrame(rows)
    comparative_df.to_csv(summary_dir / "comparative_metrics.csv", index=False)
    comparative_df.to_parquet(summary_dir / "comparative_metrics.parquet", index=False)

    paper_variant = results_by_variant["paper_vwap_baseline"]
    (summary_dir / "replication_sanity_check.md").write_text(
        _paper_validation_markdown(
            symbol=symbol,
            metrics=paper_variant["metrics_overall"],
            hourly_table=paper_variant["hourly_table"],
        ),
        encoding="utf-8",
    )

    prop_ranked = _rank_prop_variants(comparative_df.loc[comparative_df["family"] == "prop_variant"].copy())
    prop_ranked.to_csv(summary_dir / "prop_variant_ranking.csv", index=False)

    best_variant_name = str(prop_ranked.iloc[0]["name"]) if not prop_ranked.empty else "paper_vwap_baseline"
    _representative_days(results_by_variant[best_variant_name]["result"].daily_results).to_csv(
        summary_dir / "representative_days.csv",
        index=False,
    )

    baseline_curve = pd.read_csv(artifacts_by_variant["paper_vwap_baseline"]["equity_curve_csv"])
    best_curve = pd.read_csv(artifacts_by_variant[best_variant_name]["equity_curve_csv"])
    _plot_equity_comparison(
        baseline_curve=baseline_curve,
        best_curve=best_curve,
        baseline_name="paper_vwap_baseline",
        best_name=best_variant_name,
        output_path=summary_dir / "baseline_vs_best_equity.png",
    )

    sensitivity_df = pd.DataFrame()
    if not baseline_only and best_variant_name in results_by_variant:
        sensitivity_df = _run_sensitivity_suite(
            feature_df=feature_df,
            anchor_variant=results_by_variant[best_variant_name]["variant"],
            symbol=symbol,
            spec=spec,
        )
        sensitivity_df.to_csv(summary_dir / "sensitivity_results.csv", index=False)

    report_lines = [
        "# VWAP Research Campaign",
        "",
        "## Scope",
        "",
        f"- Dataset: `{spec.dataset_path.name}`",
        f"- Symbol: `{symbol}`",
        f"- Sessions total / IS / OOS: {len(all_sessions)} / {len(is_sessions)} / {len(oos_sessions)}",
        "- RTH handling is explicit and uses `[09:30, 16:00)` start-aligned bars.",
        "- The paper baseline is implemented first without filters, targets, stops, or kill switches.",
        "",
        "## Best Prop Variant",
        "",
        f"- Selected variant: `{best_variant_name}`",
        "",
        "## Comparative Table",
        "",
        "```text",
        comparative_df[
            [
                "name",
                "family",
                "overall_net_pnl",
                "overall_sharpe_ratio",
                "overall_profit_factor",
                "overall_max_drawdown",
                "oos_net_pnl",
                "oos_sharpe_ratio",
                "oos_profit_factor",
                "oos_max_drawdown",
                "oos_daily_loss_limit_breach_freq",
                "oos_trailing_drawdown_breach_freq",
                "oos_profit_to_drawdown_ratio",
            ]
        ].to_string(index=False),
        "```",
        "",
    ]
    if not sensitivity_df.empty:
        report_lines.extend(
            [
                "## Sensitivity",
                "",
                "```text",
                sensitivity_df[
                    [
                        "sensitivity_tag",
                        "oos_net_pnl",
                        "oos_sharpe_ratio",
                        "oos_profit_factor",
                        "oos_max_drawdown",
                        "oos_daily_loss_limit_breach_freq",
                    ]
                ].to_string(index=False),
                "```",
                "",
            ]
        )

    baseline_row = comparative_df.loc[comparative_df["name"] == "paper_vwap_baseline"].iloc[0]
    if not prop_ranked.empty:
        best_row = prop_ranked.iloc[0]
        if (
            float(best_row["oos_profit_to_drawdown_ratio"]) > float(baseline_row["oos_profit_to_drawdown_ratio"])
            and float(best_row["oos_daily_loss_limit_breach_freq"]) <= float(baseline_row["oos_daily_loss_limit_breach_freq"])
        ):
            report_lines.extend(
                [
                    "## Honest Conclusion",
                    "",
                    "- The selected prop variant improves the baseline on challenge-oriented robustness, not only on raw returns.",
                    "- Remaining limits: different underlying than QQQ/TQQQ when futures are used, 1-minute bars only, and a lightweight rather than exhaustive walk-forward.",
                ]
            )
        else:
            report_lines.extend(
                [
                    "## Honest Conclusion",
                    "",
                    "- The selected prop variant is operationally cleaner, but the gain over the paper baseline remains mixed.",
                    "- Remaining limits: different underlying than QQQ/TQQQ when futures are used, 1-minute bars only, and a lightweight rather than exhaustive walk-forward.",
                ]
            )

    (summary_dir / "vwap_research_report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    generated_notebook = None
    if notebook_path is not None and not baseline_only:
        generated_notebook = generate_vwap_validation_notebook(
            notebook_path=notebook_path,
            output_dir=output_dir,
            spec=spec,
            symbol=symbol,
            selected_variant=results_by_variant[best_variant_name]["variant"],
        )

    (summary_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_timestamp": datetime.now().isoformat(),
                "dataset_path": str(spec.dataset_path),
                "symbol": symbol,
                "session_count_total": len(all_sessions),
                "session_count_is": len(is_sessions),
                "session_count_oos": len(oos_sessions),
                "is_fraction": float(spec.is_fraction),
                "best_variant_name": best_variant_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    artifacts: dict[str, Path] = {
        "output_dir": output_dir,
        "comparative_csv": summary_dir / "comparative_metrics.csv",
        "replication_report_md": summary_dir / "replication_sanity_check.md",
        "campaign_report_md": summary_dir / "vwap_research_report.md",
        "prop_ranking_csv": summary_dir / "prop_variant_ranking.csv",
        "representative_days_csv": summary_dir / "representative_days.csv",
        "prepared_feature_frame_parquet": summary_dir / "prepared_feature_frame.parquet",
    }
    if not sensitivity_df.empty:
        artifacts["sensitivity_csv"] = summary_dir / "sensitivity_results.csv"
    if generated_notebook is not None:
        artifacts["validation_notebook"] = generated_notebook
    return artifacts


def build_default_campaign_spec(dataset_path: Path | None = None) -> VWAPCampaignSpec:
    resolved_dataset = dataset_path or resolve_default_vwap_dataset()
    return VWAPCampaignSpec(
        dataset_path=resolved_dataset,
        prop_constraints=build_default_prop_constraints(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VWAP research campaign.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to the 1-minute dataset.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--notebook-path", type=Path, default=None, help="Optional final notebook path.")
    parser.add_argument("--is-fraction", type=float, default=0.70, help="Chronological IS share.")
    parser.add_argument("--baseline-only", action="store_true", help="Run only the exact paper baseline.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (EXPORTS_DIR / f"vwap_campaign_{timestamp}")
    notebook_path = args.notebook_path or (NOTEBOOKS_DIR / "vwap_final_validation.ipynb")

    spec = replace(build_default_campaign_spec(args.dataset), is_fraction=args.is_fraction)
    artifacts = run_vwap_campaign(
        spec=spec,
        output_dir=output_dir,
        notebook_path=None if args.baseline_only else notebook_path,
        baseline_only=args.baseline_only,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
