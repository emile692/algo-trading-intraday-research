"""Recompute signal sheet 01 with the frozen 18-model ATR vote.

The ensemble is the Cartesian product
ATR {14, 16, 18} x q_low {20, 25, 30} x q_high {80, 85}.
Each threshold is calibrated on IS only and the day is retained with at least
9 votes out of 18.  The script exports an auditable result bundle and replaces
only the equity-curve asset belonging to signal sheet 01.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.orb_research.campaign import _build_incremental_experiments, _split_sessions  # noqa: E402
from src.analytics.orb_research.evaluation import compute_extended_metrics  # noqa: E402
from src.analytics.orb_research.exits import run_exit_variant_backtest  # noqa: E402
from src.analytics.orb_research.features import (  # noqa: E402
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    build_signal_frame_for_backtest,
    calibrate_ensemble_thresholds,
    first_pass_signal_rows,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import BaselineEnsembleConfig, BaselineEntryConfig  # noqa: E402
from src.engine.execution_model import ExecutionModel  # noqa: E402


DATASET = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
OUTPUT = ROOT / "research" / "signal_sheet_01_atr18"
EQUITY_ASSET = ROOT / "docs" / "signal_sheets" / "assets" / "01_equity_curve.png"
ATR_PERIODS = (14, 16, 18)
Q_LOWS = (20, 25, 30)
Q_HIGHS = (80, 85)
VOTE_THRESHOLD = 0.50
MINIMUM_VOTES = 9


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _daily_path(trades: pd.DataFrame, sessions: list) -> pd.DataFrame:
    daily = pd.DataFrame({"session_date": pd.to_datetime(sessions)})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
        return daily
    pnl = trades.copy()
    pnl["session_date"] = pd.to_datetime(pnl["session_date"])
    pnl = pnl.groupby("session_date", as_index=False)["net_pnl_usd"].sum()
    return daily.merge(pnl, on="session_date", how="left").rename(columns={"net_pnl_usd": "daily_pnl_usd"}).fillna({"daily_pnl_usd": 0.0})


def _plot_equity(daily: pd.DataFrame, oos_start: pd.Timestamp) -> None:
    oos = daily.loc[daily["session_date"].ge(oos_start)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    axes[0].plot(daily["session_date"], daily["daily_pnl_usd"].cumsum(), color="#1769aa", linewidth=1.8, drawstyle="steps-post")
    axes[0].axvline(oos_start, color="#c45a4a", linestyle="--", linewidth=1.4, label=f"Début : {oos_start.date():%d/%m/%Y}")
    axes[0].axvspan(oos_start, daily["session_date"].max(), color="#dbe8f4", alpha=.45)
    axes[0].set_title("FS — Full Sample", weight="bold")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(oos["session_date"], oos["daily_pnl_usd"].cumsum(), color="#25855a", linewidth=1.8, drawstyle="steps-post")
    axes[1].set_title("OOS historique réutilisé", weight="bold")
    axes[1].text(
        .02,
        .04,
        f"PnL de période : {oos['daily_pnl_usd'].sum():,.2f} USD".replace(",", " "),
        transform=axes[1].transAxes,
        color="#102b4e",
        bbox={"facecolor": "white", "edgecolor": "#c5d0dc", "alpha": .9, "pad": 3},
    )
    for axis in axes:
        axis.axhline(0, color="#7b8794", linewidth=.8)
        axis.set_ylabel("PnL cumulé (USD)")
        axis.set_xlabel("Date")
        axis.grid(axis="y", alpha=.22)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle("Equity nette — 01 · MNQ ORB · ensemble ATR 18 sous-modèles", weight="bold")
    fig.text(.99, .012, "FS inclut l’OOS ; chaque courbe démarre à 0 USD", ha="right", fontsize=8, color="#7b8794")
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(EQUITY_ASSET, dpi=190, facecolor="white")
    plt.close(fig)


def main() -> None:
    entry = BaselineEntryConfig()
    minute = prepare_minute_dataset(DATASET, entry, ATR_PERIODS)
    minute = attach_daily_reference(minute, build_daily_reference(minute))
    candidates = build_candidate_universe(minute, baseline_entry=entry)
    sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(sessions, 0.70)

    selected_pre = first_pass_signal_rows(candidates, candidates["candidate_base_pass"].fillna(False).astype(bool))
    vote_columns: dict[str, pd.Series] = {}
    thresholds_export: list[dict[str, object]] = []
    for period in ATR_PERIODS:
        atr_col = f"atr_{period}"
        thresholds = calibrate_ensemble_thresholds(selected_pre, is_sessions, atr_col, Q_LOWS, Q_HIGHS)
        if len(thresholds) != len(Q_LOWS) * len(Q_HIGHS):
            raise RuntimeError(f"Expected 6 thresholds for ATR {period}, got {len(thresholds)}")
        for (q_low, q_high), (low_value, high_value) in zip(
            ((low, high) for low in Q_LOWS for high in Q_HIGHS), thresholds
        ):
            model = f"atr{period}_q{q_low}_q{q_high}"
            vote_columns[model] = pd.to_numeric(selected_pre[atr_col], errors="coerce").between(
                low_value, high_value, inclusive="both"
            ).fillna(False)
            thresholds_export.append(
                {
                    "model": model,
                    "atr_period": period,
                    "q_low_pct": q_low,
                    "q_high_pct": q_high,
                    "low_threshold_is": low_value,
                    "high_threshold_is": high_value,
                }
            )

    votes = pd.DataFrame(vote_columns, index=selected_pre.index)
    selected_pre = selected_pre.copy()
    selected_pre["atr_vote_count"] = votes.sum(axis=1).astype(int)
    selected_pre["atr_vote_score"] = selected_pre["atr_vote_count"] / float(len(vote_columns))
    selected_final = selected_pre.loc[selected_pre["atr_vote_count"].ge(MINIMUM_VOTES)].copy()

    experiment = _build_incremental_experiments(entry, BaselineEnsembleConfig())[0]
    signal_df = build_signal_frame_for_backtest(minute, selected_final)
    trades = run_exit_variant_backtest(
        signal_df=signal_df,
        execution_model=ExecutionModel(),
        baseline=entry,
        exit_cfg=experiment.exit,
    )

    is_set, oos_set = set(is_sessions), set(oos_sessions)
    trade_dates = pd.to_datetime(trades["session_date"]).dt.date
    scoped_trades = {
        "overall": trades,
        "is": trades.loc[trade_dates.isin(is_set)].copy(),
        "oos": trades.loc[trade_dates.isin(oos_set)].copy(),
    }
    scoped_sessions = {"overall": sessions, "is": is_sessions, "oos": oos_sessions}
    metrics = {
        scope: compute_extended_metrics(
            trades=scope_trades,
            signal_df=None,
            sessions=scoped_sessions[scope],
            initial_capital=entry.account_size_usd,
            bootstrap_paths=300,
            random_seed=42 + index * 17,
        )
        for index, (scope, scope_trades) in enumerate(scoped_trades.items())
    }
    daily = _daily_path(trades, sessions)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    EQUITY_ASSET.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(OUTPUT / "trades.csv", index=False)
    daily.to_csv(OUTPUT / "daily_results.csv", index=False)
    pd.DataFrame(thresholds_export).to_csv(OUTPUT / "thresholds_is.csv", index=False)
    selected_pre[["session_date", "timestamp", "atr_vote_count", "atr_vote_score"]].to_csv(
        OUTPUT / "candidate_votes.csv", index=False
    )
    payload = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "ensemble": {
            "atr_periods": ATR_PERIODS,
            "q_lows_pct": Q_LOWS,
            "q_highs_pct": Q_HIGHS,
            "n_submodels": len(vote_columns),
            "vote_threshold": VOTE_THRESHOLD,
            "minimum_votes": MINIMUM_VOTES,
        },
        "split": {
            "is_start": str(is_sessions[0]),
            "is_end": str(is_sessions[-1]),
            "oos_start": str(oos_sessions[0]),
            "oos_end": str(oos_sessions[-1]),
            "is_sessions": len(is_sessions),
            "oos_sessions": len(oos_sessions),
        },
        "metrics": metrics,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_value) + "\n", encoding="utf-8"
    )
    _plot_equity(daily, pd.Timestamp(oos_sessions[0]))

    for scope in ("is", "oos", "overall"):
        result = metrics[scope]
        print(
            scope,
            f"trades={int(result['n_trades'])}",
            f"pnl={float(result['cumulative_pnl']):.2f}",
            f"sharpe={float(result['sharpe_ratio']):.6f}",
            f"pf={float(result['profit_factor']):.6f}",
            f"maxdd={float(result['max_drawdown']):.2f}",
        )


if __name__ == "__main__":
    main()
