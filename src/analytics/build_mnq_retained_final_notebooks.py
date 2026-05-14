"""Build standalone final notebooks for the retained MNQ ORB and pullback sleeves.

The generated notebooks are fully self-contained from the client's point of view:
- no `src.*` imports inside the notebook
- full signal / sizing / backtest logic visible in code cells
- explicit IS/OOS parameter maps for each major parameter family
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "finals"

ORB_NOTEBOOK_PATH = NOTEBOOKS_DIR / "mnq_orb_retained_final.ipynb"
ORB_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_DIR / "mnq_orb_retained_final.executed.ipynb"
PULLBACK_NOTEBOOK_PATH = NOTEBOOKS_DIR / "mnq_pullback_retained_final.ipynb"
PULLBACK_EXECUTED_NOTEBOOK_PATH = NOTEBOOKS_DIR / "mnq_pullback_retained_final.executed.ipynb"


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8-sig")


def _extract_blocks(path: str, names: list[str]) -> str:
    source = _read(path)
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    found: dict[str, str] = {}

    for node in tree.body:
        node_names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_names = [node.name]
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    node_names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                node_names.append(node.target.id)

        keep = [name for name in node_names if name in names]
        if not keep:
            continue

        decorator_lines = [getattr(dec, "lineno", node.lineno) for dec in getattr(node, "decorator_list", [])]
        start = min(decorator_lines + [node.lineno]) - 1
        end = node.end_lineno
        block = "".join(lines[start:end]).rstrip()
        for name in keep:
            found[name] = block

    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(f"Could not extract {missing} from {path}")
    return "\n\n".join(found[name] for name in names)


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def _imports_cell() -> nbf.NotebookNode:
    return _code(
        """
        import json
        import math
        import datetime as dt
        from dataclasses import asdict, dataclass, field, replace
        from pathlib import Path
        from typing import Any, Iterable

        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from IPython.display import Markdown, display
        from plotly.subplots import make_subplots

        ROOT = Path.cwd().resolve()
        while ROOT != ROOT.parent and not (ROOT / "pyproject.toml").exists():
            ROOT = ROOT.parent
        if not (ROOT / "pyproject.toml").exists():
            raise RuntimeError("Impossible de retrouver la racine du repo depuis le notebook.")

        pd.set_option("display.max_columns", 300)
        pd.set_option("display.width", 240)
        """
    )


def _common_runtime_source() -> str:
    constants = """
    REPO_ROOT = ROOT
    DEFAULT_TIMEZONE = "America/New_York"
    RTH_START = "09:30"
    RTH_END = "16:00"
    ETH_START = "18:00"
    ETH_END = "17:00"

    INSTRUMENT_SPECS = {
        "NQ": {"tick_size": 0.25, "tick_value_usd": 5.0, "point_value_usd": 20.0, "commission_per_side_usd": 1.25, "slippage_ticks": 1},
        "MNQ": {"tick_size": 0.25, "tick_value_usd": 0.5, "point_value_usd": 2.0, "commission_per_side_usd": 1.25, "slippage_ticks": 1},
        "ES": {"tick_size": 0.25, "tick_value_usd": 12.5, "point_value_usd": 50.0, "commission_per_side_usd": 1.25, "slippage_ticks": 1},
        "MES": {"tick_size": 0.25, "tick_value_usd": 1.25, "point_value_usd": 5.0, "commission_per_side_usd": 1.25, "slippage_ticks": 1},
        "RTY": {"tick_size": 0.10, "tick_value_usd": 5.0, "point_value_usd": 50.0, "commission_per_side_usd": 1.25, "slippage_ticks": 1},
        "M2K": {"tick_size": 0.10, "tick_value_usd": 0.5, "point_value_usd": 5.0, "commission_per_side_usd": 1.25, "slippage_ticks": 1},
        "MGC": {"tick_size": 0.10, "tick_value_usd": 1.0, "point_value_usd": 10.0, "commission_per_side_usd": 1.25, "slippage_ticks": 1},
    }

    DEFAULT_SYMBOL = "MNQ"
    DEFAULT_TICK_SIZE = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["tick_size"]
    DEFAULT_TICK_VALUE_USD = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["tick_value_usd"]
    DEFAULT_POINT_VALUE_USD = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["point_value_usd"]
    DEFAULT_COMMISSION_PER_SIDE_USD = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["commission_per_side_usd"]
    DEFAULT_SLIPPAGE_TICKS = INSTRUMENT_SPECS[DEFAULT_SYMBOL]["slippage_ticks"]
    DEFAULT_INITIAL_CAPITAL_USD = 50_000.0
    DEFAULT_MONTHLY_SUBSCRIPTION_COST_USD = 150.0

    def get_instrument_spec(symbol: str) -> dict:
        key = symbol.upper()
        if key not in INSTRUMENT_SPECS:
            raise ValueError(f"Unknown instrument {symbol!r}")
        return INSTRUMENT_SPECS[key].copy()

    def resolve_dataset_path(explicit_path, symbol: str, timeframe: str = "1m") -> Path:
        if explicit_path is not None:
            return Path(explicit_path)
        files = sorted((REPO_ROOT / "data" / "processed" / "parquet").glob(f"{symbol}_c_0_{timeframe}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No processed dataset found for {symbol} {timeframe}.")
        return files[-1]

    def fmt_money(value):
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):,.1f} USD"

    def fmt_pct(value, digits=2):
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}%"

    def fmt_float(value, digits=3):
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"

    def parameter_table(rows):
        frame = pd.DataFrame(rows)
        display(frame)
        return frame

    def normalize_sessions(sessions):
        return pd.to_datetime(pd.Index(sessions), errors="coerce").normalize().tolist()

    def session_set(sessions):
        return set(pd.to_datetime(pd.Index(sessions), errors="coerce").date)

    def subset_trades(trades, sessions):
        if trades.empty:
            return trades.copy()
        allowed = session_set(sessions)
        out = trades.copy()
        out["_session_key"] = pd.to_datetime(out["session_date"], errors="coerce").dt.date
        return out.loc[out["_session_key"].isin(allowed)].drop(columns=["_session_key"]).copy().reset_index(drop=True)

    def daily_results_from_trades(trades, sessions, initial_capital):
        calendar = pd.DataFrame({"session_date": normalize_sessions(sessions)})
        calendar = calendar.dropna().drop_duplicates().sort_values("session_date").reset_index(drop=True)
        if calendar.empty:
            return pd.DataFrame(columns=["session_date", "daily_pnl_usd", "daily_trade_count", "equity", "peak_equity", "drawdown_usd", "drawdown_pct", "daily_return"])

        if trades.empty:
            daily = calendar.copy()
            daily["daily_pnl_usd"] = 0.0
            daily["daily_trade_count"] = 0
        else:
            view = trades.copy()
            view["session_date"] = pd.to_datetime(view["session_date"], errors="coerce").dt.normalize()
            grouped = (
                view.groupby("session_date", as_index=False)
                .agg(daily_pnl_usd=("net_pnl_usd", "sum"), daily_trade_count=("trade_id", "count"))
            )
            daily = calendar.merge(grouped, on="session_date", how="left")
            daily["daily_pnl_usd"] = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
            daily["daily_trade_count"] = pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0).astype(int)

        prev_equity = float(initial_capital)
        rets = []
        equities = []
        for pnl in pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0):
            rets.append(float(pnl) / prev_equity if prev_equity else 0.0)
            prev_equity += float(pnl)
            equities.append(prev_equity)

        daily["daily_return"] = rets
        daily["equity"] = equities
        daily["peak_equity"] = daily["equity"].cummax()
        daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
        daily["drawdown_pct"] = np.where(daily["peak_equity"] > 0, (daily["equity"] / daily["peak_equity"] - 1.0) * 100.0, 0.0)
        return daily

    def curve_from_returns(session_dates, daily_return, initial_capital, label="curve"):
        out = pd.DataFrame({"session_date": pd.to_datetime(pd.Series(session_dates), errors="coerce").dt.normalize()})
        out["daily_return"] = pd.to_numeric(pd.Series(daily_return), errors="coerce").fillna(0.0)
        if (out["daily_return"] <= -1.0).any():
            worst = float(out["daily_return"].min())
            raise ValueError(f"{label}: daily return <= -100% ({worst:.2%}).")
        out["equity"] = float(initial_capital) * (1.0 + out["daily_return"]).cumprod()
        out["daily_pnl_usd"] = out["equity"].diff().fillna(out["equity"].iloc[0] - float(initial_capital))
        out["peak_equity"] = out["equity"].cummax()
        out["drawdown_usd"] = out["equity"] - out["peak_equity"]
        out["drawdown_pct"] = np.where(out["peak_equity"] > 0, (out["equity"] / out["peak_equity"] - 1.0) * 100.0, 0.0)
        return out

    def curve_metrics(curve, trades=None, initial_capital=50_000.0):
        if curve.empty:
            return {
                "net_pnl_usd": 0.0,
                "return_pct": 0.0,
                "cagr_pct": np.nan,
                "sharpe": 0.0,
                "sortino": 0.0,
                "annualized_vol_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
                "worst_day_usd": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
            }
        ordered = curve.sort_values("session_date").reset_index(drop=True)
        rets = pd.to_numeric(ordered["daily_return"], errors="coerce").fillna(0.0)
        equity = pd.to_numeric(ordered["equity"], errors="coerce").fillna(float(initial_capital))
        pnl = pd.to_numeric(ordered["daily_pnl_usd"], errors="coerce").fillna(0.0)
        start = pd.Timestamp(ordered["session_date"].iloc[0])
        end = pd.Timestamp(ordered["session_date"].iloc[-1])
        years = max(((end - start).days + 1) / 365.25, 1 / 365.25)
        final_equity = float(equity.iloc[-1])
        sharpe = float(rets.mean() / rets.std(ddof=0) * math.sqrt(252.0)) if len(rets) > 1 and rets.std(ddof=0) > 0 else 0.0
        downside = rets[rets < 0]
        downside_std = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
        sortino = float(rets.mean() / downside_std * math.sqrt(252.0)) if downside_std > 0 else 0.0
        cagr = float(((final_equity / float(initial_capital)) ** (1.0 / years) - 1.0) * 100.0) if final_equity > 0 else np.nan
        if trades is not None and not trades.empty:
            trade_pnl = pd.to_numeric(trades["net_pnl_usd"], errors="coerce").fillna(0.0)
            wins = trade_pnl[trade_pnl > 0]
            losses = trade_pnl[trade_pnl < 0]
            profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
            n_trades = int(len(trade_pnl))
            win_rate = float((trade_pnl > 0).mean())
        else:
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
            n_trades = 0
            win_rate = np.nan
        return {
            "net_pnl_usd": float(final_equity - float(initial_capital)),
            "return_pct": float((final_equity / float(initial_capital) - 1.0) * 100.0),
            "cagr_pct": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "annualized_vol_pct": float(rets.std(ddof=0) * math.sqrt(252.0) * 100.0) if len(rets) > 1 else 0.0,
            "profit_factor": profit_factor,
            "max_drawdown_usd": float(abs(pd.to_numeric(ordered["drawdown_usd"], errors="coerce").min())),
            "max_drawdown_pct": float(abs(pd.to_numeric(ordered["drawdown_pct"], errors="coerce").min())),
            "worst_day_usd": float(pnl.min()),
            "n_trades": n_trades,
            "win_rate": win_rate,
        }

    def summarize_strategy_scopes(label, full_curve, full_trades, is_sessions, oos_sessions, initial_capital):
        is_curve = full_curve.loc[full_curve["session_date"].isin(set(normalize_sessions(is_sessions)))].copy()
        oos_trades = subset_trades(full_trades, oos_sessions)
        oos_curve = daily_results_from_trades(oos_trades, oos_sessions, initial_capital)
        rows = [
            {"strategy": label, "scope": "full", **curve_metrics(full_curve, full_trades, initial_capital)},
            {"strategy": label, "scope": "is", **curve_metrics(is_curve, subset_trades(full_trades, is_sessions), initial_capital)},
            {"strategy": label, "scope": "oos", **curve_metrics(oos_curve, oos_trades, initial_capital)},
        ]
        return pd.DataFrame(rows)

    def _metric_columns_for_scope(frame, scope, metric):
        mapping = {
            "sharpe": f"{scope}_sharpe_ratio",
            "net_pnl_usd": f"{scope}_net_pnl",
            "profit_factor": f"{scope}_profit_factor",
            "max_drawdown_usd": f"{scope}_max_drawdown",
            "prop_score": f"{scope}_prop_score",
            "n_trades": f"{scope}_nb_trades",
        }
        return mapping[metric]

    def plot_scope_heatmap(frame, x, metric, title, text_auto=".2f", color_continuous_scale="RdYlGn"):
        data = frame.copy()
        data[x] = data[x].astype(str)
        pivot = data.pivot_table(index="scope", columns=x, values=metric, aggfunc="mean")
        fig = px.imshow(pivot, aspect="auto", text_auto=text_auto, color_continuous_scale=color_continuous_scale, title=title)
        fig.update_layout(template=PLOT_TEMPLATE, width=1100, height=380)
        fig.update_xaxes(title=x)
        fig.update_yaxes(title="scope")
        fig.show()

    def plot_is_oos_heatmaps(frame, x, y, metric, title, text_auto=".2f", color_continuous_scale="RdYlGn"):
        scopes = [("is", "IS"), ("oos", "OOS")]
        fig = make_subplots(rows=1, cols=2, subplot_titles=[label for _, label in scopes], horizontal_spacing=0.10)
        zmin = None
        zmax = None
        if metric not in {"max_drawdown_usd"}:
            finite = pd.to_numeric(frame[metric], errors="coerce")
            if finite.notna().any():
                zmin = float(finite.min())
                zmax = float(finite.max())
        for idx, (scope, label) in enumerate(scopes, start=1):
            scoped = frame.loc[frame["scope"].eq(scope)].copy()
            scoped[x] = scoped[x].astype(str)
            scoped[y] = scoped[y].astype(str)
            pivot = scoped.pivot_table(index=y, columns=x, values=metric, aggfunc="mean").sort_index()
            heatmap = go.Heatmap(
                z=pivot.values,
                x=[str(v) for v in pivot.columns],
                y=[str(v) for v in pivot.index],
                colorscale=color_continuous_scale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=metric) if idx == 2 else None,
                showscale=idx == 2,
                text=np.round(pivot.values, 2 if metric not in {"net_pnl_usd", "max_drawdown_usd"} else 0),
                texttemplate="%{text}",
            )
            fig.add_trace(heatmap, row=1, col=idx)
        fig.update_layout(template=PLOT_TEMPLATE, width=1350, height=500, title=title)
        fig.update_xaxes(title=x, row=1, col=1)
        fig.update_xaxes(title=x, row=1, col=2)
        fig.update_yaxes(title=y, row=1, col=1)
        fig.update_yaxes(title=y, row=1, col=2)
        fig.show()

    def row_to_scope_records(row, extra=None):
        extra = extra or {}
        records = []
        for scope in ("is", "oos"):
            records.append(
                {
                    **extra,
                    "scope": scope,
                    "net_pnl_usd": float(row.get(f"{scope}_net_pnl", 0.0)),
                    "sharpe": float(row.get(f"{scope}_sharpe_ratio", 0.0)),
                    "profit_factor": float(row.get(f"{scope}_profit_factor", 0.0)),
                    "max_drawdown_usd": abs(float(row.get(f"{scope}_max_drawdown", 0.0))),
                    "prop_score": float(row.get(f"{scope}_prop_score", 0.0)),
                    "n_trades": int(row.get(f"{scope}_nb_trades", 0) or 0),
                }
            )
        return records
    """

    blocks = [
        _extract_blocks("src/data/cleaning.py", ["clean_ohlcv"]),
        _extract_blocks("src/data/loader.py", ["_normalize_ohlcv_frame", "load_ohlcv_csv", "load_ohlcv_file"]),
        _extract_blocks("src/data/session.py", ["_session_mask", "filter_session", "extract_rth", "add_session_date"]),
        _extract_blocks("src/utils/time_utils.py", ["build_session_time"]),
        _extract_blocks("src/features/intraday.py", ["add_intraday_features", "add_session_vwap", "add_continuous_session_vwap"]),
        _extract_blocks("src/features/opening_range.py", ["compute_opening_range"]),
        _extract_blocks("src/features/volatility.py", ["_normalize_windows", "add_atr"]),
        _extract_blocks("src/engine/trade_log.py", ["TRADE_LOG_COLUMNS", "empty_trade_log", "trade_to_record"]),
        _extract_blocks("src/engine/execution_model.py", ["ExecutionModel"]),
        _extract_blocks(
            "src/engine/vwap_backtester.py",
            [
                "EQUITY_TICK_SIZE",
                "EQUITY_POINT_VALUE_USD",
                "EQUITY_PAPER_COMMISSION_PER_SHARE",
                "InstrumentDetails",
                "resolve_instrument_details",
                "build_execution_model_for_profile",
            ],
        ),
        _extract_blocks("src/analytics/volume_climax_pullback_common.py", ["latest_path_for_symbol", "load_symbol_data", "resample_rth_1h", "split_sessions"]),
    ]
    return textwrap.dedent(constants).strip() + "\n\n" + "\n\n".join(blocks)


def _orb_runtime_source() -> str:
    blocks = [
        _extract_blocks("src/config/orb_campaign.py", ["PropConstraintConfig"]),
        _extract_blocks(
            "src/analytics/metrics.py",
            [
                "_loss_streak_lengths",
                "_resolve_session_dates",
                "_daily_pnl",
                "_prop_empty_metrics",
                "_compute_prop_metrics",
                "compute_metrics",
            ],
        ),
        _extract_blocks(
            "src/analytics/orb_research/types.py",
            [
                "BaselineEntryConfig",
                "BaselineEnsembleConfig",
                "CompressionConfig",
                "ExitConfig",
                "DynamicThresholdConfig",
                "ExperimentConfig",
                "CampaignContext",
            ],
        ),
        _extract_blocks("src/strategy/orb.py", ["_SIDE_MODE_ALIASES", "_DIRECTION_FILTER_MODES", "_normalize_side_mode", "ORBStrategy"]),
        _extract_blocks(
            "src/engine/backtester.py",
            [
                "_validate_risk_inputs",
                "_validate_backtest_inputs",
                "_compute_risk_per_contract_usd",
                "_apply_leverage_cap",
                "run_backtest",
            ],
        ),
        _extract_blocks(
            "src/analytics/orb_research/features.py",
            [
                "_between_times",
                "prepare_minute_dataset",
                "build_daily_reference",
                "attach_daily_reference",
                "build_candidate_universe",
                "_schedule_mask",
                "_consecutive_true_within_session",
                "compute_noise_sigma",
                "dynamic_gate_mask",
                "compression_mask",
                "first_pass_signal_rows",
                "calibrate_ensemble_thresholds",
                "apply_ensemble_selection",
                "build_signal_frame_for_backtest",
            ],
        ),
        _extract_blocks(
            "src/analytics/orb_research/exits.py",
            [
                "_compute_risk_per_contract_usd",
                "_apply_leverage_cap",
                "_resolve_force_exit_time",
                "_stagnation_hit",
                "run_exit_variant_backtest",
            ],
        ),
        _extract_blocks(
            "src/analytics/orb_research/evaluation.py",
            [
                "_daily_pnl_from_trades",
                "_run_lengths",
                "_challenge_outcome",
                "simulate_prop_challenge",
                "compute_extended_metrics",
                "compute_scores",
            ],
        ),
        _extract_blocks(
            "src/analytics/orb_research/campaign.py",
            ["_split_sessions", "_serialize_config", "_experiment_from_json", "_evaluate_experiment"],
        ),
    ]
    return "\n\n".join(blocks)


def _pullback_runtime_source() -> str:
    blocks = [
        _extract_blocks(
            "src/strategy/volume_climax_pullback_v2.py",
            [
                "VolumeClimaxPullbackV2Variant",
                "_rolling_percent_rank",
                "prepare_volume_climax_pullback_v2_features",
                "build_volume_climax_pullback_v2_signal_frame",
            ],
        ),
        _extract_blocks(
            "src/risk/position_sizing.py",
            [
                "FixedContractPositionSizing",
                "RiskPercentPositionSizing",
                "PositionSizingConfig",
                "PositionSizingDecision",
                "validate_position_sizing",
                "initial_capital_from_sizing",
                "compounds_realized_pnl",
                "resolve_position_size",
            ],
        ),
        _extract_blocks(
            "src/engine/volume_climax_pullback_v2_backtester.py",
            [
                "VolumeClimaxPullbackV2BacktestResult",
                "SIZING_DECISION_COLUMNS",
                "_float_or_nan",
                "_bool_or_false",
                "_empty_sizing_decision_log",
                "_entry_market_fill",
                "_entry_limit_fill",
                "_resolve_target_price",
                "_append_sizing_decision",
                "_trade_record",
                "_build_open_trade",
                "_confirmation_triggered",
                "_maybe_fill_pullback_limit",
                "_trail_stop_from_last_close",
                "_entry_row_from_pending_setup",
                "_finalize_trade",
                "run_volume_climax_pullback_v2_backtest",
            ],
        ),
    ]
    return "\n\n".join(blocks)


def _orb_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    nb.cells = [
        _md(
            """
            # MNQ ORB retenue - Notebook final client

            Ce notebook reconstruit de bout en bout la sleeve ORB retenue dans le portefeuille MNQ ORB + Pullback.

            Ce que contient ce livrable :
            - le chargement et le nettoyage des donnees MNQ 1 minute,
            - la logique ORB complete visible dans le notebook,
            - la reconstruction du signal, du filtrage et du backtest,
            - les heatmaps IS/OOS sur chaque famille de parametres importante,
            - une cellule de parametrage unique pour challenger facilement la strategie.
            """
        ),
        _imports_cell(),
        _code(_common_runtime_source()),
        _code(_orb_runtime_source()),
        _code(
            """
            SYMBOL = "MNQ"
            INITIAL_CAPITAL_USD = 50_000.0
            IS_FRACTION = 0.70
            PLOT_TEMPLATE = "plotly_dark"

            MNQ_1M_DATASET_PATH = None

            ORB_RESEARCH_CONFIG_NAME = "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate"
            ORB_OR_MINUTES = 15
            ORB_OPENING_TIME = "09:30:00"
            ORB_DIRECTION = "long"
            ORB_ONE_TRADE_PER_DAY = True
            ORB_ENTRY_BUFFER_TICKS = 2
            ORB_STOP_BUFFER_TICKS = 2
            ORB_TARGET_MULTIPLE = 2.0
            ORB_VWAP_CONFIRMATION = True
            ORB_VWAP_COLUMN = "continuous_session_vwap"
            ORB_TIME_EXIT = "16:00:00"
            ORB_RISK_PER_TRADE_PCT = 0.50
            ORB_MAX_LEVERAGE = None

            ORB_ATR_WINDOW = 14
            ORB_Q_LOW_PCTS = (20, 25, 30)
            ORB_Q_HIGH_PCTS = (90, 95)
            ORB_ENSEMBLE_VOTE_THRESHOLD = 0.50
            ORB_COMPRESSION_MODE = "weak_close"
            ORB_COMPRESSION_USAGE = "soft_vote_bonus"
            ORB_COMPRESSION_SOFT_BONUS_VOTES = 1.0
            ORB_EXIT_MODE = "baseline"
            ORB_DYNAMIC_MODE = "noise_area_gate"
            ORB_NOISE_LOOKBACK = 30
            ORB_NOISE_VM = 1.0
            ORB_NOISE_K = 0.0
            ORB_DYNAMIC_ATR_K = 0.0
            ORB_DYNAMIC_CONFIRM_BARS = 1
            ORB_DYNAMIC_SCHEDULE = "continuous_on_bar_close"
            ORB_DYNAMIC_THRESHOLD_STYLE = "max_or_high_noise"

            ORB_GRID_BOOTSTRAP_PATHS = 0
            ORB_BASELINE_BOOTSTRAP_PATHS = 300

            ORB_OR_MINUTES_GRID = (10, 15, 20, 30)
            ORB_TARGET_MULTIPLE_GRID = (1.5, 2.0, 2.5, 3.0)
            ORB_ENTRY_BUFFER_TICKS_GRID = (0, 1, 2, 3)
            ORB_STOP_BUFFER_TICKS_GRID = (0, 1, 2, 3)
            ORB_VWAP_CONFIRMATION_GRID = (False, True)
            ORB_TIME_EXIT_GRID = ("15:30:00", "16:00:00")
            ORB_RISK_PER_TRADE_PCT_GRID = (0.25, 0.50, 0.75, 1.00)
            ORB_COMPRESSION_MODE_GRID = ("none", "weak_close", "strong_close", "nr4", "nr7", "inside_day")
            ORB_COMPRESSION_USAGE_GRID = ("hard_filter", "soft_vote_bonus")
            ORB_NOISE_LOOKBACK_GRID = (10, 14, 20, 30)
            ORB_NOISE_VM_GRID = (0.75, 1.0, 1.25, 1.5)
            ORB_ATR_WINDOW_GRID = (10, 14, 20, 30)
            ORB_VOTE_THRESHOLD_GRID = (0.50, 0.67, 0.75)
            ORB_QUANTILE_SET_GRID = (
                ((20,), (90,)),
                ((20,), (95,)),
                ((25,), (95,)),
                ((20, 25), (90, 95)),
                ((20, 25, 30), (90, 95)),
            )
            ORB_DYNAMIC_MODE_GRID = ("disabled", "noise_area_gate", "atr_threshold_gate", "close_confirmation_gate")
            ORB_ATR_K_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
            ORB_CONFIRM_BARS_GRID = (1, 2, 3)

            dataset_path = resolve_dataset_path(MNQ_1M_DATASET_PATH, SYMBOL, timeframe="1m")
            instrument_spec = get_instrument_spec(SYMBOL)

            parameter_rows = [
                {"section": "global", "parameter": "SYMBOL", "value": SYMBOL, "meaning": "Contrat analyse."},
                {"section": "global", "parameter": "INITIAL_CAPITAL_USD", "value": INITIAL_CAPITAL_USD, "meaning": "Capital de reference pour la courbe d'equity et le sizing."},
                {"section": "global", "parameter": "IS_FRACTION", "value": IS_FRACTION, "meaning": "Part historique reservee a l'in-sample."},
                {"section": "data", "parameter": "MNQ_1M_DATASET_PATH", "value": str(dataset_path), "meaning": "Dataset minute recharge a chaque execution."},
                {"section": "entry", "parameter": "ORB_OR_MINUTES", "value": ORB_OR_MINUTES, "meaning": "Longueur de l'opening range."},
                {"section": "entry", "parameter": "ORB_DIRECTION", "value": ORB_DIRECTION, "meaning": "Direction retenue: long only."},
                {"section": "entry", "parameter": "ORB_ENTRY_BUFFER_TICKS", "value": ORB_ENTRY_BUFFER_TICKS, "meaning": "Buffer a casser au-dessus du range."},
                {"section": "entry", "parameter": "ORB_STOP_BUFFER_TICKS", "value": ORB_STOP_BUFFER_TICKS, "meaning": "Buffer ajoute sous le plus bas OR pour le stop."},
                {"section": "entry", "parameter": "ORB_TARGET_MULTIPLE", "value": ORB_TARGET_MULTIPLE, "meaning": "Target en multiple du risque initial."},
                {"section": "entry", "parameter": "ORB_VWAP_CONFIRMATION", "value": ORB_VWAP_CONFIRMATION, "meaning": "Confirmation par VWAP continu."},
                {"section": "entry", "parameter": "ORB_TIME_EXIT", "value": ORB_TIME_EXIT, "meaning": "Heure de sortie forcee."},
                {"section": "risk", "parameter": "ORB_RISK_PER_TRADE_PCT", "value": ORB_RISK_PER_TRADE_PCT, "meaning": "Risque par trade en % du capital."},
                {"section": "risk", "parameter": "ORB_MAX_LEVERAGE", "value": ORB_MAX_LEVERAGE, "meaning": "Cap notionnel optionnel."},
                {"section": "ensemble", "parameter": "ORB_ATR_WINDOW", "value": ORB_ATR_WINDOW, "meaning": "Fenetre ATR du filtre d'ensemble."},
                {"section": "ensemble", "parameter": "ORB_Q_LOW_PCTS / ORB_Q_HIGH_PCTS", "value": f"{ORB_Q_LOW_PCTS} / {ORB_Q_HIGH_PCTS}", "meaning": "Bandes de quantiles ATR qui votent pour retenir une session."},
                {"section": "ensemble", "parameter": "ORB_ENSEMBLE_VOTE_THRESHOLD", "value": ORB_ENSEMBLE_VOTE_THRESHOLD, "meaning": "Seuil de vote minimum."},
                {"section": "overlay", "parameter": "ORB_COMPRESSION_MODE / USAGE", "value": f"{ORB_COMPRESSION_MODE} / {ORB_COMPRESSION_USAGE}", "meaning": "Overlay daily retenu."},
                {"section": "overlay", "parameter": "ORB_DYNAMIC_MODE", "value": ORB_DYNAMIC_MODE, "meaning": "Famille de gate dynamique retenue."},
                {"section": "overlay", "parameter": "ORB_NOISE_LOOKBACK / ORB_NOISE_VM", "value": f"{ORB_NOISE_LOOKBACK} / {ORB_NOISE_VM}", "meaning": "Parametres du noise gate."},
            ]
            display(Markdown("## 0. Parametrage client"))
            params = parameter_table(parameter_rows)
            """
        ),
        _code(
            """
            raw_1m = load_symbol_data(SYMBOL, input_paths={SYMBOL: dataset_path})
            raw_1m["timestamp"] = pd.to_datetime(raw_1m["timestamp"], errors="coerce")

            display(Markdown("## 1. Data et sanity checks"))
            data_sanity = pd.DataFrame(
                [
                    {"item": "rows_1m", "value": f"{len(raw_1m):,}"},
                    {"item": "first_timestamp", "value": str(raw_1m["timestamp"].min())},
                    {"item": "last_timestamp", "value": str(raw_1m["timestamp"].max())},
                    {"item": "duplicate_timestamps", "value": int(raw_1m["timestamp"].duplicated().sum())},
                    {"item": "tick_size", "value": instrument_spec["tick_size"]},
                    {"item": "tick_value_usd", "value": instrument_spec["tick_value_usd"]},
                    {"item": "point_value_usd", "value": instrument_spec["point_value_usd"]},
                    {"item": "commission_per_side_usd", "value": instrument_spec["commission_per_side_usd"]},
                    {"item": "slippage_ticks", "value": instrument_spec["slippage_ticks"]},
                ]
            )
            display(data_sanity)

            orb_context_cache = {}

            def make_orb_entry(**overrides):
                payload = {
                    "or_minutes": ORB_OR_MINUTES,
                    "opening_time": ORB_OPENING_TIME,
                    "direction": ORB_DIRECTION,
                    "one_trade_per_day": ORB_ONE_TRADE_PER_DAY,
                    "entry_buffer_ticks": ORB_ENTRY_BUFFER_TICKS,
                    "stop_buffer_ticks": ORB_STOP_BUFFER_TICKS,
                    "target_multiple": ORB_TARGET_MULTIPLE,
                    "vwap_confirmation": ORB_VWAP_CONFIRMATION,
                    "vwap_column": ORB_VWAP_COLUMN,
                    "time_exit": ORB_TIME_EXIT,
                    "account_size_usd": INITIAL_CAPITAL_USD,
                    "risk_per_trade_pct": ORB_RISK_PER_TRADE_PCT,
                    "tick_size": float(instrument_spec["tick_size"]),
                    "entry_on_next_open": True,
                }
                payload.update(overrides)
                return BaselineEntryConfig(**payload)

            def make_orb_ensemble(**overrides):
                payload = {
                    "atr_window": ORB_ATR_WINDOW,
                    "q_lows_pct": tuple(ORB_Q_LOW_PCTS),
                    "q_highs_pct": tuple(ORB_Q_HIGH_PCTS),
                    "vote_threshold": ORB_ENSEMBLE_VOTE_THRESHOLD,
                }
                payload.update(overrides)
                return BaselineEnsembleConfig(**payload)

            def make_orb_compression(**overrides):
                payload = {
                    "mode": ORB_COMPRESSION_MODE,
                    "usage": ORB_COMPRESSION_USAGE,
                    "soft_bonus_votes": ORB_COMPRESSION_SOFT_BONUS_VOTES,
                }
                payload.update(overrides)
                return CompressionConfig(**payload)

            def make_orb_dynamic(**overrides):
                payload = {
                    "mode": ORB_DYNAMIC_MODE,
                    "noise_lookback": ORB_NOISE_LOOKBACK,
                    "noise_vm": ORB_NOISE_VM,
                    "threshold_style": ORB_DYNAMIC_THRESHOLD_STYLE,
                    "noise_k": ORB_NOISE_K,
                    "atr_k": ORB_DYNAMIC_ATR_K,
                    "confirm_bars": ORB_DYNAMIC_CONFIRM_BARS,
                    "schedule": ORB_DYNAMIC_SCHEDULE,
                }
                payload.update(overrides)
                return DynamicThresholdConfig(**payload)

            def make_orb_exit(**overrides):
                payload = {"mode": ORB_EXIT_MODE}
                payload.update(overrides)
                return ExitConfig(**payload)

            def make_orb_experiment(entry=None, ensemble=None, compression=None, dynamic=None, exit_cfg=None, name=None):
                return ExperimentConfig(
                    name=name or ORB_RESEARCH_CONFIG_NAME,
                    stage="full_reopt",
                    family="full_reopt",
                    baseline_entry=entry or make_orb_entry(),
                    baseline_ensemble=ensemble or make_orb_ensemble(),
                    compression=compression or make_orb_compression(),
                    exit=exit_cfg or make_orb_exit(),
                    dynamic_threshold=dynamic or make_orb_dynamic(),
                )

            def orb_context_key(entry_cfg, atr_windows):
                return (
                    int(entry_cfg.or_minutes),
                    str(entry_cfg.opening_time),
                    str(entry_cfg.direction),
                    int(entry_cfg.entry_buffer_ticks),
                    bool(entry_cfg.vwap_confirmation),
                    str(entry_cfg.vwap_column),
                    tuple(sorted(int(x) for x in atr_windows)),
                )

            def get_orb_context(entry_cfg, atr_windows):
                key = orb_context_key(entry_cfg, atr_windows)
                if key not in orb_context_cache:
                    minute_df = prepare_minute_dataset(dataset_path=dataset_path, baseline_entry=entry_cfg, atr_windows=atr_windows)
                    daily_reference = build_daily_reference(minute_df)
                    minute_df = attach_daily_reference(minute_df, daily_reference)
                    candidate_base = build_candidate_universe(minute_df, baseline_entry=entry_cfg)
                    all_sessions = sorted(pd.to_datetime(minute_df["session_date"]).dt.date.unique())
                    is_sessions, oos_sessions = _split_sessions(all_sessions, IS_FRACTION)
                    orb_context_cache[key] = CampaignContext(
                        all_sessions=all_sessions,
                        is_sessions=is_sessions,
                        oos_sessions=oos_sessions,
                        minute_df=minute_df,
                        candidate_base_df=candidate_base,
                        daily_patterns=daily_reference,
                    )
                return orb_context_cache[key]

            def evaluate_orb_experiment_local(experiment, keep_details=False, bootstrap_paths=0, random_seed=42, max_leverage=None):
                required_windows = sorted({int(experiment.baseline_ensemble.atr_window), *[int(x) for x in ORB_ATR_WINDOW_GRID]})
                context = get_orb_context(experiment.baseline_entry, required_windows)
                return _evaluate_experiment(
                    experiment=experiment,
                    context=context,
                    bootstrap_paths=bootstrap_paths,
                    random_seed=random_seed,
                    keep_details=keep_details,
                    max_leverage=max_leverage,
                ), context

            retained_experiment = make_orb_experiment()
            (retained_row, retained_detail), retained_context = evaluate_orb_experiment_local(
                retained_experiment,
                keep_details=True,
                bootstrap_paths=ORB_BASELINE_BOOTSTRAP_PATHS,
                random_seed=42,
                max_leverage=ORB_MAX_LEVERAGE,
            )

            if retained_detail is None:
                raise RuntimeError(f"Retained ORB evaluation failed: {retained_row}")

            orb_trades = retained_detail["trades"].copy()
            orb_signal_df = retained_detail["signal_df"].copy()
            orb_selected_final = retained_detail["selected_final"].copy()
            orb_all_sessions = normalize_sessions(retained_context.all_sessions)
            orb_is_sessions = normalize_sessions(retained_context.is_sessions)
            orb_oos_sessions = normalize_sessions(retained_context.oos_sessions)
            orb_daily_full = daily_results_from_trades(orb_trades, orb_all_sessions, INITIAL_CAPITAL_USD)
            orb_summary = summarize_strategy_scopes("ORB retained", orb_daily_full, orb_trades, orb_is_sessions, orb_oos_sessions, INITIAL_CAPITAL_USD)

            display(Markdown("## 2. Reconstruction de la strategie retenue"))
            display(Markdown(
                "Config retenue: `full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate` "
                "avec OR15 long-only, vote ATR, bonus `weak_close` et `noise_area_gate`."
            ))
            display(pd.DataFrame(
                [
                    {"item": "candidate_rows_raw", "value": int(retained_row.get("candidate_rows_raw", 0))},
                    {"item": "candidate_rows_after_overlays", "value": int(retained_row.get("candidate_rows_after_overlays", 0))},
                    {"item": "selected_days_after_ensemble", "value": int(retained_row.get("selected_days", 0))},
                    {"item": "full_trades", "value": int(len(orb_trades))},
                    {"item": "oos_trades", "value": int(len(subset_trades(orb_trades, orb_oos_sessions)))},
                ]
            ))

            config_rows = []
            for block, payload in [
                ("entry", asdict(retained_experiment.baseline_entry)),
                ("ensemble", asdict(retained_experiment.baseline_ensemble)),
                ("compression", asdict(retained_experiment.compression)),
                ("dynamic_threshold", asdict(retained_experiment.dynamic_threshold)),
                ("exit", asdict(retained_experiment.exit)),
            ]:
                for parameter, value in payload.items():
                    config_rows.append({"block": block, "parameter": parameter, "value": value})
            display(pd.DataFrame(config_rows))
            display(orb_summary.round(3))

            orb_is_curve = orb_daily_full.loc[orb_daily_full["session_date"].isin(set(orb_is_sessions))].copy()
            orb_oos_curve = daily_results_from_trades(subset_trades(orb_trades, orb_oos_sessions), orb_oos_sessions, INITIAL_CAPITAL_USD)

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Full sample", "OOS rebased"))
            fig.add_trace(go.Scatter(x=orb_daily_full["session_date"], y=orb_daily_full["equity"], mode="lines", name="ORB full", line=dict(color="#2563eb", width=2.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=orb_oos_curve["session_date"], y=orb_oos_curve["equity"], mode="lines", name="ORB OOS", line=dict(color="#16a34a", width=2.5)), row=1, col=2)
            fig.update_layout(template=PLOT_TEMPLATE, width=1350, height=500, legend=dict(orientation="h", y=-0.15))
            fig.update_yaxes(title_text="Equity USD", row=1, col=1)
            fig.update_yaxes(title_text="Equity USD", row=1, col=2)
            fig.show()

            display(Markdown("### Extrait des trades"))
            display(orb_trades.head(20))
            """
        ),
        _code(
            """
            display(Markdown("## 3. Heatmaps IS/OOS - Parametres d'entree, execution et risque"))

            entry_target_rows = []
            counter = 0
            for or_minutes in ORB_OR_MINUTES_GRID:
                for target_multiple in ORB_TARGET_MULTIPLE_GRID:
                    counter += 1
                    exp = make_orb_experiment(
                        entry=make_orb_entry(or_minutes=int(or_minutes), target_multiple=float(target_multiple)),
                        name=f"or{or_minutes}_tm{target_multiple}",
                    )
                    (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                    entry_target_rows.extend(row_to_scope_records(row, {"or_minutes": int(or_minutes), "target_multiple": float(target_multiple)}))
            orb_or_target_grid = pd.DataFrame(entry_target_rows)
            plot_is_oos_heatmaps(orb_or_target_grid, "target_multiple", "or_minutes", "sharpe", "ORB | OR minutes x target multiple | Sharpe")
            plot_is_oos_heatmaps(orb_or_target_grid, "target_multiple", "or_minutes", "net_pnl_usd", "ORB | OR minutes x target multiple | Net PnL", text_auto=".0f")

            buffer_rows = []
            counter = 1_000
            for entry_buffer in ORB_ENTRY_BUFFER_TICKS_GRID:
                for stop_buffer in ORB_STOP_BUFFER_TICKS_GRID:
                    counter += 1
                    exp = make_orb_experiment(
                        entry=make_orb_entry(entry_buffer_ticks=int(entry_buffer), stop_buffer_ticks=int(stop_buffer)),
                        name=f"eb{entry_buffer}_sb{stop_buffer}",
                    )
                    (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                    buffer_rows.extend(row_to_scope_records(row, {"entry_buffer_ticks": int(entry_buffer), "stop_buffer_ticks": int(stop_buffer)}))
            orb_buffer_grid = pd.DataFrame(buffer_rows)
            plot_is_oos_heatmaps(orb_buffer_grid, "entry_buffer_ticks", "stop_buffer_ticks", "sharpe", "ORB | entry buffer x stop buffer | Sharpe")
            plot_is_oos_heatmaps(orb_buffer_grid, "entry_buffer_ticks", "stop_buffer_ticks", "max_drawdown_usd", "ORB | entry buffer x stop buffer | Max DD", text_auto=".0f", color_continuous_scale="RdYlGn_r")

            vwap_time_rows = []
            counter = 2_000
            for vwap_confirmation in ORB_VWAP_CONFIRMATION_GRID:
                for time_exit in ORB_TIME_EXIT_GRID:
                    counter += 1
                    exp = make_orb_experiment(
                        entry=make_orb_entry(vwap_confirmation=bool(vwap_confirmation), time_exit=str(time_exit)),
                        name=f"vwap{int(bool(vwap_confirmation))}_tx{time_exit.replace(':','')}",
                    )
                    (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                    vwap_time_rows.extend(row_to_scope_records(row, {"vwap_confirmation": str(bool(vwap_confirmation)), "time_exit": str(time_exit)}))
            orb_vwap_time_grid = pd.DataFrame(vwap_time_rows)
            plot_is_oos_heatmaps(orb_vwap_time_grid, "time_exit", "vwap_confirmation", "sharpe", "ORB | VWAP confirmation x time exit | Sharpe")

            risk_rows = []
            counter = 3_000
            for risk_pct in ORB_RISK_PER_TRADE_PCT_GRID:
                counter += 1
                exp = make_orb_experiment(entry=make_orb_entry(risk_per_trade_pct=float(risk_pct)), name=f"risk_{risk_pct}")
                (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                risk_rows.extend(row_to_scope_records(row, {"risk_pct": float(risk_pct)}))
            orb_risk_grid = pd.DataFrame(risk_rows)
            plot_scope_heatmap(orb_risk_grid, "risk_pct", "sharpe", "ORB | risk per trade % | Sharpe")
            plot_scope_heatmap(orb_risk_grid, "risk_pct", "max_drawdown_usd", "ORB | risk per trade % | Max DD", text_auto=".0f", color_continuous_scale="RdYlGn_r")
            """
        ),
        _code(
            """
            display(Markdown("## 4. Heatmaps IS/OOS - Overlays, dynamique et ensemble"))

            compression_rows = []
            counter = 4_000
            for compression_mode in ORB_COMPRESSION_MODE_GRID:
                for compression_usage in ORB_COMPRESSION_USAGE_GRID:
                    counter += 1
                    exp = make_orb_experiment(
                        compression=make_orb_compression(mode=str(compression_mode), usage=str(compression_usage)),
                        name=f"comp_{compression_mode}_{compression_usage}",
                    )
                    (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                    compression_rows.extend(row_to_scope_records(row, {"compression_mode": str(compression_mode), "compression_usage": str(compression_usage)}))
            orb_compression_grid = pd.DataFrame(compression_rows)
            plot_is_oos_heatmaps(orb_compression_grid, "compression_usage", "compression_mode", "sharpe", "ORB | compression mode x usage | Sharpe")

            noise_rows = []
            counter = 5_000
            for lookback in ORB_NOISE_LOOKBACK_GRID:
                for vm in ORB_NOISE_VM_GRID:
                    counter += 1
                    exp = make_orb_experiment(
                        dynamic=make_orb_dynamic(mode="noise_area_gate", noise_lookback=int(lookback), noise_vm=float(vm), threshold_style="max_or_high_noise"),
                        name=f"noise_l{lookback}_vm{vm}",
                    )
                    (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                    noise_rows.extend(row_to_scope_records(row, {"noise_lookback": int(lookback), "noise_vm": float(vm)}))
            orb_noise_grid = pd.DataFrame(noise_rows)
            plot_is_oos_heatmaps(orb_noise_grid, "noise_vm", "noise_lookback", "sharpe", "ORB | noise lookback x noise VM | Sharpe")
            plot_is_oos_heatmaps(orb_noise_grid, "noise_vm", "noise_lookback", "max_drawdown_usd", "ORB | noise lookback x noise VM | Max DD", text_auto=".0f", color_continuous_scale="RdYlGn_r")

            ensemble_rows = []
            counter = 6_000
            for atr_window in ORB_ATR_WINDOW_GRID:
                for vote_threshold in ORB_VOTE_THRESHOLD_GRID:
                    counter += 1
                    exp = make_orb_experiment(
                        ensemble=make_orb_ensemble(atr_window=int(atr_window), vote_threshold=float(vote_threshold)),
                        name=f"atr{atr_window}_vote{vote_threshold}",
                    )
                    (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                    ensemble_rows.extend(row_to_scope_records(row, {"atr_window": int(atr_window), "vote_threshold": float(vote_threshold)}))
            orb_ensemble_grid = pd.DataFrame(ensemble_rows)
            plot_is_oos_heatmaps(orb_ensemble_grid, "vote_threshold", "atr_window", "sharpe", "ORB | ATR window x vote threshold | Sharpe")

            quantile_rows = []
            counter = 7_000
            for q_lows, q_highs in ORB_QUANTILE_SET_GRID:
                counter += 1
                label = f"low{list(q_lows)}_high{list(q_highs)}"
                exp = make_orb_experiment(
                    ensemble=make_orb_ensemble(q_lows_pct=tuple(q_lows), q_highs_pct=tuple(q_highs)),
                    name=f"quantiles_{counter}",
                )
                (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                quantile_rows.extend(row_to_scope_records(row, {"quantile_set": label}))
            orb_quantile_grid = pd.DataFrame(quantile_rows)
            plot_scope_heatmap(orb_quantile_grid, "quantile_set", "sharpe", "ORB | quantile-set alternatives | Sharpe")

            dynamic_family_rows = []
            counter = 8_000
            for mode in ORB_DYNAMIC_MODE_GRID:
                counter += 1
                if mode == "noise_area_gate":
                    dynamic_cfg = make_orb_dynamic(mode=mode, noise_lookback=ORB_NOISE_LOOKBACK, noise_vm=ORB_NOISE_VM, threshold_style="max_or_high_noise")
                elif mode == "atr_threshold_gate":
                    dynamic_cfg = make_orb_dynamic(mode=mode, atr_k=0.5)
                elif mode == "close_confirmation_gate":
                    dynamic_cfg = make_orb_dynamic(mode=mode, confirm_bars=1)
                else:
                    dynamic_cfg = make_orb_dynamic(mode=mode)
                exp = make_orb_experiment(dynamic=dynamic_cfg, name=f"dynamic_{mode}")
                (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                dynamic_family_rows.extend(row_to_scope_records(row, {"dynamic_mode": str(mode)}))
            orb_dynamic_family_grid = pd.DataFrame(dynamic_family_rows)
            plot_scope_heatmap(orb_dynamic_family_grid, "dynamic_mode", "sharpe", "ORB | dynamic mode family | Sharpe")

            atrk_rows = []
            counter = 9_000
            for atr_k in ORB_ATR_K_GRID:
                counter += 1
                exp = make_orb_experiment(
                    dynamic=make_orb_dynamic(mode="atr_threshold_gate", atr_k=float(atr_k)),
                    name=f"atrk_{atr_k}",
                )
                (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                atrk_rows.extend(row_to_scope_records(row, {"atr_k": float(atr_k)}))
            orb_atrk_grid = pd.DataFrame(atrk_rows)
            plot_scope_heatmap(orb_atrk_grid, "atr_k", "sharpe", "ORB | ATR threshold gate | Sharpe")

            confirm_rows = []
            counter = 10_000
            for confirm_bars in ORB_CONFIRM_BARS_GRID:
                counter += 1
                exp = make_orb_experiment(
                    dynamic=make_orb_dynamic(mode="close_confirmation_gate", confirm_bars=int(confirm_bars)),
                    name=f"confirm_{confirm_bars}",
                )
                (row, _), _ = evaluate_orb_experiment_local(exp, keep_details=False, bootstrap_paths=ORB_GRID_BOOTSTRAP_PATHS, random_seed=100 + counter, max_leverage=ORB_MAX_LEVERAGE)
                confirm_rows.extend(row_to_scope_records(row, {"confirm_bars": int(confirm_bars)}))
            orb_confirm_grid = pd.DataFrame(confirm_rows)
            plot_scope_heatmap(orb_confirm_grid, "confirm_bars", "sharpe", "ORB | close confirmation bars | Sharpe")
            """
        ),
        _code(
            """
            display(Markdown("## 5. Lecture finale"))
            oos_row = orb_summary.loc[orb_summary["scope"].eq("oos")].iloc[0]
            notes = [
                f"- ORB retenue OOS: net `{fmt_money(oos_row['net_pnl_usd'])}`, Sharpe `{fmt_float(oos_row['sharpe'])}`, maxDD `{fmt_money(oos_row['max_drawdown_usd'])}`.",
                "- Les heatmaps se lisent toujours en IS/OOS : l'objectif n'est pas seulement le meilleur point, mais la zone stable.",
                "- Les premieres zones a challenger si le client veut adapter la sleeve sont : `noise_lookback / noise_vm`, `atr_window / vote_threshold`, puis `or_minutes / target_multiple`.",
                "- Le notebook est autonome : toute la logique de signal, d'overlay et de backtest est visible dans les cellules de code ci-dessus.",
            ]
            display(Markdown("\\n".join(notes)))
            """
        ),
    ]
    return nb


def _pullback_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    nb.cells = [
        _md(
            """
            # MNQ Pullback retenu - Notebook final client

            Ce notebook reconstruit la sleeve `volume climax pullback` retenue dans l'etude MNQ ORB + Pullback.

            Important :
            - le code de la strategie est visible dans le notebook,
            - l'alpha et le sizing sont dissocies,
            - les heatmaps IS/OOS couvrent le signal, les modes d'entree, les sorties et le sizing,
            - le parametrage par defaut reproduit la sleeve utilisee dans le notebook equal-weight client.
            """
        ),
        _imports_cell(),
        _code(_common_runtime_source()),
        _code(_pullback_runtime_source()),
        _code(
            """
            SYMBOL = "MNQ"
            INITIAL_CAPITAL_USD = 50_000.0
            IS_FRACTION = 0.70
            PLOT_TEMPLATE = "plotly_dark"

            MNQ_1M_DATASET_PATH = None

            PULLBACK_VOLUME_QUANTILE = 0.95
            PULLBACK_VOLUME_LOOKBACK = 50
            PULLBACK_MIN_BODY_FRACTION = 0.50
            PULLBACK_MIN_RANGE_ATR = 1.20
            PULLBACK_ENTRY_MODE = "next_open"
            PULLBACK_PULLBACK_FRACTION = None
            PULLBACK_CONFIRMATION_WINDOW = None
            PULLBACK_EXIT_MODE = "atr_fraction"
            PULLBACK_RR_TARGET = 1.0
            PULLBACK_ATR_TARGET_MULTIPLE = 1.0
            PULLBACK_TIME_STOP_BARS = 2
            PULLBACK_TRAILING_ATR_MULTIPLE = 0.50

            PULLBACK_RISK_PCT = 0.0025
            PULLBACK_MAX_CONTRACTS = 6
            PULLBACK_SKIP_TRADE_IF_TOO_SMALL = True
            PULLBACK_COMPOUND_REALIZED_PNL = False

            PULLBACK_VOLUME_QUANTILE_GRID = (0.90, 0.95, 0.975, 0.99)
            PULLBACK_VOLUME_LOOKBACK_GRID = (30, 50, 70)
            PULLBACK_MIN_BODY_FRACTION_GRID = (0.40, 0.50, 0.60, 0.70)
            PULLBACK_MIN_RANGE_ATR_GRID = (1.0, 1.2, 1.5, 1.8)
            PULLBACK_ENTRY_LABEL_GRID = (
                ("next_open", None, None),
                ("pullback_limit", 0.25, None),
                ("pullback_limit", 0.50, None),
                ("confirmation", 0.25, 1),
                ("confirmation", 0.25, 2),
                ("confirmation", 0.50, 2),
            )
            PULLBACK_ATR_TARGET_GRID = (0.75, 1.0, 1.25, 1.5)
            PULLBACK_RR_TARGET_GRID = (0.75, 1.0, 1.25, 1.5)
            PULLBACK_TIME_STOP_BARS_GRID = (1, 2, 3, 4)
            PULLBACK_TRAILING_ATR_MULTIPLE_GRID = (0.25, 0.50, 0.75, 1.0)
            PULLBACK_RISK_PCT_GRID = (0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040)
            PULLBACK_MAX_CONTRACTS_GRID = (2, 3, 4, 5, 6)
            PULLBACK_SKIP_TOO_SMALL_GRID = (False, True)
            PULLBACK_COMPOUND_REALIZED_PNL_GRID = (False, True)

            dataset_path = resolve_dataset_path(MNQ_1M_DATASET_PATH, SYMBOL, timeframe="1m")
            instrument_spec = get_instrument_spec(SYMBOL)

            parameter_rows = [
                {"section": "global", "parameter": "SYMBOL", "value": SYMBOL, "meaning": "Contrat analyse."},
                {"section": "global", "parameter": "INITIAL_CAPITAL_USD", "value": INITIAL_CAPITAL_USD, "meaning": "Capital de reference."},
                {"section": "global", "parameter": "IS_FRACTION", "value": IS_FRACTION, "meaning": "Split chronologique IS/OOS."},
                {"section": "data", "parameter": "MNQ_1M_DATASET_PATH", "value": str(dataset_path), "meaning": "Source minute du notebook."},
                {"section": "alpha", "parameter": "PULLBACK_VOLUME_QUANTILE", "value": PULLBACK_VOLUME_QUANTILE, "meaning": "Seuil de volume historique du setup climax."},
                {"section": "alpha", "parameter": "PULLBACK_VOLUME_LOOKBACK", "value": PULLBACK_VOLUME_LOOKBACK, "meaning": "Fenetre historique du quantile de volume."},
                {"section": "alpha", "parameter": "PULLBACK_MIN_BODY_FRACTION", "value": PULLBACK_MIN_BODY_FRACTION, "meaning": "Qualite minimale du corps de la bougie de setup."},
                {"section": "alpha", "parameter": "PULLBACK_MIN_RANGE_ATR", "value": PULLBACK_MIN_RANGE_ATR, "meaning": "Range minimum du climax en ATR."},
                {"section": "entry", "parameter": "PULLBACK_ENTRY_MODE", "value": PULLBACK_ENTRY_MODE, "meaning": "Mode d'entree retenu dans la sleeve portefeuille."},
                {"section": "exit", "parameter": "PULLBACK_EXIT_MODE", "value": PULLBACK_EXIT_MODE, "meaning": "Mode de sortie retenu: atr_fraction."},
                {"section": "exit", "parameter": "PULLBACK_ATR_TARGET_MULTIPLE", "value": PULLBACK_ATR_TARGET_MULTIPLE, "meaning": "Distance du target en ATR."},
                {"section": "exit", "parameter": "PULLBACK_TIME_STOP_BARS", "value": PULLBACK_TIME_STOP_BARS, "meaning": "Nombre de barres max en position."},
                {"section": "exit", "parameter": "PULLBACK_TRAILING_ATR_MULTIPLE", "value": PULLBACK_TRAILING_ATR_MULTIPLE, "meaning": "Parametre du mode mixed."},
                {"section": "risk", "parameter": "PULLBACK_RISK_PCT", "value": PULLBACK_RISK_PCT, "meaning": "Risque par trade utilise par defaut dans le blend client."},
                {"section": "risk", "parameter": "PULLBACK_MAX_CONTRACTS", "value": PULLBACK_MAX_CONTRACTS, "meaning": "Cap de contrats par trade."},
                {"section": "risk", "parameter": "PULLBACK_SKIP_TRADE_IF_TOO_SMALL", "value": PULLBACK_SKIP_TRADE_IF_TOO_SMALL, "meaning": "Skip si le sizing tombe sous 1 contrat."},
                {"section": "risk", "parameter": "PULLBACK_COMPOUND_REALIZED_PNL", "value": PULLBACK_COMPOUND_REALIZED_PNL, "meaning": "False = sizing sur capital initial constant."},
            ]
            display(Markdown("## 0. Parametrage client"))
            params = parameter_table(parameter_rows)
            """
        ),
        _code(
            """
            raw_1m = load_symbol_data(SYMBOL, input_paths={SYMBOL: dataset_path})
            raw_1m["timestamp"] = pd.to_datetime(raw_1m["timestamp"], errors="coerce")
            bars_1h = resample_rth_1h(raw_1m)
            bars_1h["timestamp"] = pd.to_datetime(bars_1h["timestamp"], errors="coerce")
            bars_1h["session_date"] = pd.to_datetime(bars_1h["timestamp"]).dt.date
            all_sessions = normalize_sessions(sorted(pd.to_datetime(bars_1h["session_date"]).dt.date.unique()))
            is_sessions_raw, oos_sessions_raw = split_sessions(bars_1h[["session_date"]].copy(), ratio=IS_FRACTION)
            is_sessions = normalize_sessions(is_sessions_raw)
            oos_sessions = normalize_sessions(oos_sessions_raw)
            features_1h = prepare_volume_climax_pullback_v2_features(bars_1h)

            display(Markdown("## 1. Data et preparation 1h"))
            display(pd.DataFrame(
                [
                    {"item": "rows_1m", "value": f"{len(raw_1m):,}"},
                    {"item": "rows_1h_rth", "value": f"{len(bars_1h):,}"},
                    {"item": "sessions_all", "value": len(all_sessions)},
                    {"item": "sessions_is", "value": len(is_sessions)},
                    {"item": "sessions_oos", "value": len(oos_sessions)},
                    {"item": "first_timestamp_1h", "value": str(bars_1h["timestamp"].min())},
                    {"item": "last_timestamp_1h", "value": str(bars_1h["timestamp"].max())},
                ]
            ))

            def make_pullback_variant(**overrides):
                payload = {
                    "name": "client_pullback_retained",
                    "family": "retained",
                    "timeframe": "1h",
                    "volume_quantile": PULLBACK_VOLUME_QUANTILE,
                    "volume_lookback": PULLBACK_VOLUME_LOOKBACK,
                    "min_body_fraction": PULLBACK_MIN_BODY_FRACTION,
                    "min_range_atr": PULLBACK_MIN_RANGE_ATR,
                    "trend_ema_window": None,
                    "ema_slope_threshold": None,
                    "atr_percentile_low": None,
                    "atr_percentile_high": None,
                    "compression_ratio_max": None,
                    "entry_mode": PULLBACK_ENTRY_MODE,
                    "pullback_fraction": PULLBACK_PULLBACK_FRACTION,
                    "confirmation_window": PULLBACK_CONFIRMATION_WINDOW,
                    "exit_mode": PULLBACK_EXIT_MODE,
                    "rr_target": PULLBACK_RR_TARGET,
                    "atr_target_multiple": PULLBACK_ATR_TARGET_MULTIPLE,
                    "time_stop_bars": PULLBACK_TIME_STOP_BARS,
                    "trailing_atr_multiple": PULLBACK_TRAILING_ATR_MULTIPLE,
                    "session_overlay": "all_rth",
                }
                payload.update(overrides)
                return VolumeClimaxPullbackV2Variant(**payload)

            def make_pullback_sizing(**overrides):
                payload = {
                    "initial_capital_usd": INITIAL_CAPITAL_USD,
                    "risk_pct": PULLBACK_RISK_PCT,
                    "max_contracts": PULLBACK_MAX_CONTRACTS,
                    "skip_trade_if_too_small": PULLBACK_SKIP_TRADE_IF_TOO_SMALL,
                    "compound_realized_pnl": PULLBACK_COMPOUND_REALIZED_PNL,
                }
                payload.update(overrides)
                return RiskPercentPositionSizing(**payload)

            def build_pullback_signal(variant):
                return build_volume_climax_pullback_v2_signal_frame(features_1h, variant)

            def run_pullback_variant(variant, sizing, signal_df=None):
                signal_df = build_pullback_signal(variant) if signal_df is None else signal_df
                execution_model, instrument = build_execution_model_for_profile(symbol=SYMBOL, profile_name="repo_realistic")
                result = run_volume_climax_pullback_v2_backtest(
                    signal_df=signal_df,
                    variant=variant,
                    execution_model=execution_model,
                    instrument=instrument,
                    position_sizing=sizing,
                )
                full_curve = daily_results_from_trades(result.trades, all_sessions, INITIAL_CAPITAL_USD)
                summary = summarize_strategy_scopes("Pullback retained", full_curve, result.trades, is_sessions, oos_sessions, INITIAL_CAPITAL_USD)
                return signal_df, result, full_curve, summary

            retained_variant = make_pullback_variant()
            retained_sizing = make_pullback_sizing()
            pullback_signal_df, pullback_result, pullback_daily_full, pullback_summary = run_pullback_variant(retained_variant, retained_sizing)
            pullback_trades = pullback_result.trades.copy()

            display(Markdown("## 2. Reconstruction de la sleeve retenue"))
            display(Markdown(
                "Par defaut, ce notebook reproduit l'alpha `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2` "
                "avec le sizing `risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true`, c'est-a-dire la sleeve "
                "utilisee dans le notebook equal-weight client."
            ))
            display(pd.DataFrame([asdict(retained_variant)]).T.rename(columns={0: "value"}))
            display(pd.DataFrame([{
                "risk_pct": retained_sizing.risk_pct,
                "max_contracts": retained_sizing.max_contracts,
                "skip_trade_if_too_small": retained_sizing.skip_trade_if_too_small,
                "compound_realized_pnl": retained_sizing.compound_realized_pnl,
            }]))
            display(pullback_summary.round(3))

            pullback_oos_curve = daily_results_from_trades(subset_trades(pullback_trades, oos_sessions), oos_sessions, INITIAL_CAPITAL_USD)
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Full sample", "OOS rebased"))
            fig.add_trace(go.Scatter(x=pullback_daily_full["session_date"], y=pullback_daily_full["equity"], mode="lines", name="Pullback full", line=dict(color="#16a34a", width=2.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=pullback_oos_curve["session_date"], y=pullback_oos_curve["equity"], mode="lines", name="Pullback OOS", line=dict(color="#f59e0b", width=2.5)), row=1, col=2)
            fig.update_layout(template=PLOT_TEMPLATE, width=1350, height=500, legend=dict(orientation="h", y=-0.15))
            fig.update_yaxes(title_text="Equity USD", row=1, col=1)
            fig.update_yaxes(title_text="Equity USD", row=1, col=2)
            fig.show()

            display(Markdown("### Extrait des trades"))
            display(pullback_trades.head(20))
            """
        ),
        _code(
            """
            display(Markdown("## 3. Heatmaps IS/OOS - Alpha pur"))

            alpha_rows = []
            for volume_quantile in PULLBACK_VOLUME_QUANTILE_GRID:
                for body_fraction in PULLBACK_MIN_BODY_FRACTION_GRID:
                    for range_atr in PULLBACK_MIN_RANGE_ATR_GRID:
                        variant = make_pullback_variant(
                            volume_quantile=float(volume_quantile),
                            min_body_fraction=float(body_fraction),
                            min_range_atr=float(range_atr),
                        )
                        signal_df, result, full_curve, summary = run_pullback_variant(variant, retained_sizing)
                        for _, row in summary.iterrows():
                            alpha_rows.append(
                                {
                                    "scope": row["scope"],
                                    "volume_quantile": float(volume_quantile),
                                    "min_body_fraction": float(body_fraction),
                                    "min_range_atr": float(range_atr),
                                    "net_pnl_usd": float(row["net_pnl_usd"]),
                                    "sharpe": float(row["sharpe"]),
                                    "max_drawdown_usd": float(row["max_drawdown_usd"]),
                                }
                            )
            pullback_alpha_grid = pd.DataFrame(alpha_rows)
            for range_atr in sorted(pullback_alpha_grid["min_range_atr"].unique()):
                alpha_slice = pullback_alpha_grid.loc[pullback_alpha_grid["min_range_atr"].eq(range_atr)].copy()
                plot_is_oos_heatmaps(alpha_slice, "volume_quantile", "min_body_fraction", "sharpe", f"Pullback | volume quantile x body fraction | Sharpe | range_atr={range_atr}")
                plot_is_oos_heatmaps(alpha_slice, "volume_quantile", "min_body_fraction", "net_pnl_usd", f"Pullback | volume quantile x body fraction | Net PnL | range_atr={range_atr}", text_auto=".0f")

            lookback_rows = []
            for volume_lookback in PULLBACK_VOLUME_LOOKBACK_GRID:
                variant = make_pullback_variant(volume_lookback=int(volume_lookback))
                signal_df, result, full_curve, summary = run_pullback_variant(variant, retained_sizing)
                for _, row in summary.iterrows():
                    lookback_rows.append(
                        {
                            "scope": row["scope"],
                            "volume_lookback": int(volume_lookback),
                            "net_pnl_usd": float(row["net_pnl_usd"]),
                            "sharpe": float(row["sharpe"]),
                            "max_drawdown_usd": float(row["max_drawdown_usd"]),
                        }
                    )
            pullback_lookback_grid = pd.DataFrame(lookback_rows)
            plot_scope_heatmap(pullback_lookback_grid, "volume_lookback", "sharpe", "Pullback | volume lookback | Sharpe")
            """
        ),
        _code(
            """
            display(Markdown("## 4. Heatmaps IS/OOS - Modes d'entree et de sortie"))

            entry_rows = []
            for entry_mode, pullback_fraction, confirmation_window in PULLBACK_ENTRY_LABEL_GRID:
                variant = make_pullback_variant(
                    entry_mode=str(entry_mode),
                    pullback_fraction=pullback_fraction,
                    confirmation_window=confirmation_window,
                )
                label = f"{entry_mode}|pf={pullback_fraction}|cw={confirmation_window}"
                signal_df, result, full_curve, summary = run_pullback_variant(variant, retained_sizing)
                for _, row in summary.iterrows():
                    entry_rows.append(
                        {
                            "scope": row["scope"],
                            "entry_label": label,
                            "net_pnl_usd": float(row["net_pnl_usd"]),
                            "sharpe": float(row["sharpe"]),
                            "max_drawdown_usd": float(row["max_drawdown_usd"]),
                        }
                    )
            pullback_entry_grid = pd.DataFrame(entry_rows)
            plot_scope_heatmap(pullback_entry_grid, "entry_label", "sharpe", "Pullback | entry-mode family | Sharpe")

            atr_exit_rows = []
            for atr_target in PULLBACK_ATR_TARGET_GRID:
                for time_stop in PULLBACK_TIME_STOP_BARS_GRID:
                    variant = make_pullback_variant(
                        exit_mode="atr_fraction",
                        atr_target_multiple=float(atr_target),
                        time_stop_bars=int(time_stop),
                    )
                    signal_df, result, full_curve, summary = run_pullback_variant(variant, retained_sizing)
                    for _, row in summary.iterrows():
                        atr_exit_rows.append(
                            {
                                "scope": row["scope"],
                                "atr_target_multiple": float(atr_target),
                                "time_stop_bars": int(time_stop),
                                "net_pnl_usd": float(row["net_pnl_usd"]),
                                "sharpe": float(row["sharpe"]),
                                "max_drawdown_usd": float(row["max_drawdown_usd"]),
                            }
                        )
            pullback_atr_exit_grid = pd.DataFrame(atr_exit_rows)
            plot_is_oos_heatmaps(pullback_atr_exit_grid, "atr_target_multiple", "time_stop_bars", "sharpe", "Pullback | atr target x time stop | Sharpe")

            rr_exit_rows = []
            for rr_target in PULLBACK_RR_TARGET_GRID:
                for time_stop in PULLBACK_TIME_STOP_BARS_GRID:
                    variant = make_pullback_variant(
                        exit_mode="fixed_rr",
                        rr_target=float(rr_target),
                        time_stop_bars=int(time_stop),
                        atr_target_multiple=None,
                    )
                    signal_df, result, full_curve, summary = run_pullback_variant(variant, retained_sizing)
                    for _, row in summary.iterrows():
                        rr_exit_rows.append(
                            {
                                "scope": row["scope"],
                                "rr_target": float(rr_target),
                                "time_stop_bars": int(time_stop),
                                "net_pnl_usd": float(row["net_pnl_usd"]),
                                "sharpe": float(row["sharpe"]),
                                "max_drawdown_usd": float(row["max_drawdown_usd"]),
                            }
                        )
            pullback_rr_exit_grid = pd.DataFrame(rr_exit_rows)
            plot_is_oos_heatmaps(pullback_rr_exit_grid, "rr_target", "time_stop_bars", "sharpe", "Pullback | fixed RR target x time stop | Sharpe")

            mixed_rows = []
            for trailing_atr in PULLBACK_TRAILING_ATR_MULTIPLE_GRID:
                for time_stop in PULLBACK_TIME_STOP_BARS_GRID:
                    variant = make_pullback_variant(
                        exit_mode="mixed",
                        atr_target_multiple=None,
                        trailing_atr_multiple=float(trailing_atr),
                        time_stop_bars=int(time_stop),
                    )
                    signal_df, result, full_curve, summary = run_pullback_variant(variant, retained_sizing)
                    for _, row in summary.iterrows():
                        mixed_rows.append(
                            {
                                "scope": row["scope"],
                                "trailing_atr_multiple": float(trailing_atr),
                                "time_stop_bars": int(time_stop),
                                "net_pnl_usd": float(row["net_pnl_usd"]),
                                "sharpe": float(row["sharpe"]),
                                "max_drawdown_usd": float(row["max_drawdown_usd"]),
                            }
                        )
            pullback_mixed_grid = pd.DataFrame(mixed_rows)
            plot_is_oos_heatmaps(pullback_mixed_grid, "trailing_atr_multiple", "time_stop_bars", "sharpe", "Pullback | mixed trailing ATR x time stop | Sharpe")
            """
        ),
        _code(
            """
            display(Markdown("## 5. Heatmaps IS/OOS - Sizing et compounding"))

            sizing_rows = []
            for risk_pct in PULLBACK_RISK_PCT_GRID:
                for max_contracts in PULLBACK_MAX_CONTRACTS_GRID:
                    sizing = make_pullback_sizing(risk_pct=float(risk_pct), max_contracts=int(max_contracts))
                    signal_df, result, full_curve, summary = run_pullback_variant(retained_variant, sizing, signal_df=pullback_signal_df)
                    for _, row in summary.iterrows():
                        sizing_rows.append(
                            {
                                "scope": row["scope"],
                                "risk_pct": float(risk_pct),
                                "max_contracts": int(max_contracts),
                                "net_pnl_usd": float(row["net_pnl_usd"]),
                                "sharpe": float(row["sharpe"]),
                                "max_drawdown_usd": float(row["max_drawdown_usd"]),
                            }
                        )
            pullback_sizing_grid = pd.DataFrame(sizing_rows)
            plot_is_oos_heatmaps(pullback_sizing_grid, "max_contracts", "risk_pct", "sharpe", "Pullback | risk pct x max contracts | Sharpe")
            plot_is_oos_heatmaps(pullback_sizing_grid, "max_contracts", "risk_pct", "max_drawdown_usd", "Pullback | risk pct x max contracts | Max DD", text_auto=".0f", color_continuous_scale="RdYlGn_r")

            bool_rows = []
            for skip_small in PULLBACK_SKIP_TOO_SMALL_GRID:
                for compound_pnl in PULLBACK_COMPOUND_REALIZED_PNL_GRID:
                    sizing = make_pullback_sizing(skip_trade_if_too_small=bool(skip_small), compound_realized_pnl=bool(compound_pnl))
                    signal_df, result, full_curve, summary = run_pullback_variant(retained_variant, sizing, signal_df=pullback_signal_df)
                    for _, row in summary.iterrows():
                        bool_rows.append(
                            {
                                "scope": row["scope"],
                                "skip_trade_if_too_small": str(bool(skip_small)),
                                "compound_realized_pnl": str(bool(compound_pnl)),
                                "net_pnl_usd": float(row["net_pnl_usd"]),
                                "sharpe": float(row["sharpe"]),
                                "max_drawdown_usd": float(row["max_drawdown_usd"]),
                            }
                        )
            pullback_bool_grid = pd.DataFrame(bool_rows)
            plot_is_oos_heatmaps(pullback_bool_grid, "compound_realized_pnl", "skip_trade_if_too_small", "sharpe", "Pullback | skip-small x compound-PnL | Sharpe")
            """
        ),
        _code(
            """
            display(Markdown("## 6. Lecture finale"))
            oos_row = pullback_summary.loc[pullback_summary["scope"].eq("oos")].iloc[0]
            notes = [
                f"- Sleeve portefeuille par defaut: alpha `vq=0.95 / body=0.50 / range_atr=1.20 / atr_target=1.0 / time_stop=2`, sizing `{PULLBACK_RISK_PCT:.4f}` avec cap `{PULLBACK_MAX_CONTRACTS}` contrats.",
                f"- Pullback OOS du notebook par defaut: net `{fmt_money(oos_row['net_pnl_usd'])}`, Sharpe `{fmt_float(oos_row['sharpe'])}`, maxDD `{fmt_money(oos_row['max_drawdown_usd'])}`.",
                "- Le notebook separe bien alpha et sizing: on peut changer le setup sans toucher au moteur de risque, ou l'inverse.",
                "- Si l'objectif est la rigueur client, les premieres surfaces a lire sont : `volume_quantile / body_fraction / range_atr`, puis `entry-mode family`, puis `risk_pct / max_contracts`.",
            ]
            display(Markdown("\\n".join(notes)))
            """
        ),
    ]
    return nb


def _write_notebook(notebook: nbf.NotebookNode, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbf.writes(notebook), encoding="utf-8")
    return path


def _execute_notebook(input_path: Path, output_path: Path, timeout_seconds: int) -> Path:
    notebook = nbf.read(input_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name="python3",
        resources={"metadata": {"path": str(input_path.parent)}},
    )
    client.execute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")
    return output_path


def build_notebooks() -> dict[str, nbf.NotebookNode]:
    return {
        "orb": _orb_notebook(),
        "pullback": _pullback_notebook(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebooks = build_notebooks()

    orb_path = _write_notebook(notebooks["orb"], ORB_NOTEBOOK_PATH)
    pullback_path = _write_notebook(notebooks["pullback"], PULLBACK_NOTEBOOK_PATH)
    print(f"Notebook written to {orb_path}")
    print(f"Notebook written to {pullback_path}")

    if args.execute:
        orb_executed = _execute_notebook(orb_path, ORB_EXECUTED_NOTEBOOK_PATH, timeout_seconds=args.timeout_seconds)
        pullback_executed = _execute_notebook(pullback_path, PULLBACK_EXECUTED_NOTEBOOK_PATH, timeout_seconds=args.timeout_seconds)
        print(f"Executed notebook written to {orb_executed}")
        print(f"Executed notebook written to {pullback_executed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
