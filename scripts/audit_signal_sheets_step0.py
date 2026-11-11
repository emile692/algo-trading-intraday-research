"""Reproduce the 13 October 2026 signal-sheet figures without changing them.

This is an audit-only entry point.  It reads the frozen strategy artefacts,
re-runs the frozen MNQ/MGC engines, and writes separate Step-0 tables.  It does
not write under ``docs/signal_sheets`` and cannot change a lock or a strategy.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.baseline_backtests.baseline_engine import run_baseline_backtest
from research.locked_four_sleeve_portfolio.run_locked_four_sleeve_portfolio import run_alpha_sleeves
from research.mgc_baseline_backtests.run_mgc_baselines import prepare_mgc_data
from src.analytics.orb_research.features import prepare_minute_dataset
from src.analytics.orb_research.types import BaselineEntryConfig


OUT = ROOT / "research" / "signal_sheets_audit_20261014"
CONFIG_PATH = OUT / "step0_config.json"
MNQ_CONFIG = ROOT / "research" / "locked_four_sleeve_portfolio" / "locked_four_sleeve_config.json"
MGC_CONFIG = ROOT / "research" / "portfolio_integration_phase1" / "portfolio_integration_phase1_config.json"
ORB_TRADES = ROOT / "export" / "mnq_orb_vwap_atr_retained" / "trades.csv"
ORB_DATA = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
MBT_DAILY = ROOT / "research" / "mbt_three_sleeve_freeze_20261004" / "vote_ensemble" / "daily_ensemble_results.csv"
BOOK_DAILY = ROOT / "research" / "mbt_nine_sleeve_topstep_20261004" / "daily_nine_sleeves_nominal.csv"

SLEEVES = [
    "mnq_orb_vwap_atr",
    "mnq_volatility_breakout",
    "mnq_compression_breakout",
    "mnq_atr_trend_flip",
    "mgc_volatility_breakout",
    "mgc_gap_continuation",
    "mbt_rolling_range_breakout",
    "mbt_trend_pullback_ma_structure",
    "mbt_volatility_breakout",
]

SOURCE_FILES = {
    "mnq_orb_vwap_atr": "export/mnq_orb_vwap_atr_retained/trades.csv + data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet",
    "mnq_volatility_breakout": "research/locked_four_sleeve_portfolio/locked_four_sleeve_config.json + frozen engine",
    "mnq_compression_breakout": "research/locked_four_sleeve_portfolio/locked_four_sleeve_config.json + frozen engine",
    "mnq_atr_trend_flip": "research/locked_four_sleeve_portfolio/locked_four_sleeve_config.json + frozen engine",
    "mgc_volatility_breakout": "research/mgc_final_two_sleeve_lock_20260926/frozen_candidate_specs/volatility_breakout.json + frozen engine",
    "mgc_gap_continuation": "research/mgc_final_two_sleeve_lock_20260926/frozen_candidate_specs/gap_continuation.json + frozen engine",
    "mbt_rolling_range_breakout": "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/daily_ensemble_results.csv",
    "mbt_trend_pullback_ma_structure": "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/daily_ensemble_results.csv",
    "mbt_volatility_breakout": "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/daily_ensemble_results.csv",
}

DOCUMENT_CLAIMS = [
    ("S01", "is", 584, 12524.50, 0.962, -2061.50),
    ("S01", "oos", 239, 6013.00, 1.153, -928.50),
    ("S01", "total", 823, 18537.50, 1.016, -2061.50),
    ("S02", "total", 899, 18597.00, 0.903, -4655.50),
    ("S03", "validation", 110, 1078.50, None, None),
    ("S03", "total", 331, 2617.50, 0.493, -1190.00),
    ("S04", "validation", 659, 8274.00, None, None),
    ("S04", "total", 2119, 12377.50, 0.532, -4056.00),
    ("S07", "executable_total", 288, 1993.34, 0.673, -1285.80),
    ("S08", "executable_total", 693, 3465.24, 0.827, -1653.28),
    ("S09", "executable_total", 442, 2244.06, 0.565, -1916.40),
]


def _daily_metrics(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="raise").fillna(0.0).astype(float)
    equity = np.r_[0.0, values.cumsum().to_numpy()]
    drawdown = equity - np.maximum.accumulate(equity)
    volatility = float(values.std(ddof=1))
    return {
        "net_pnl_usd": float(values.sum()),
        "daily_volatility_usd": volatility,
        "sharpe_annualized": float(values.mean() / volatility * math.sqrt(252.0)) if volatility > 0 else np.nan,
        "worst_day_usd": float(values.min()),
        "max_drawdown_usd": float(drawdown.min()),
    }


def _period_row(
    sleeve: str,
    period: str,
    bounds: list[str],
    daily: pd.DataFrame,
    trade_count_column: str = "trade_count",
) -> dict[str, object]:
    start, end = map(pd.Timestamp, bounds)
    x = daily.loc[daily["session_date"].between(start, end, inclusive="both")].copy()
    if x.empty:
        return {
            "sleeve": sleeve,
            "period": period,
            "configured_start": bounds[0],
            "configured_end": bounds[1],
            "coverage_status": "missing",
        }
    coverage = "complete"
    if x["session_date"].min() > start + pd.Timedelta(days=7) or x["session_date"].max() < end - pd.Timedelta(days=7):
        coverage = "partial"
    metrics = _daily_metrics(x["daily_pnl_usd"])
    source_sharpe = metrics["sharpe_annualized"]
    source_basis = "net daily PnL, zero-trade days included, sqrt(252), ddof=1"
    if "source_equity_based" in x.columns and bool(x["source_equity_based"].iloc[0]):
        source_equity = 50_000.0 + pd.to_numeric(x["daily_pnl_usd"], errors="raise").fillna(0.0).cumsum()
        source_values = x["daily_pnl_usd"] / source_equity.shift(1).fillna(50_000.0)
        source_ddof = int(x["source_sharpe_ddof"].iloc[0])
        source_std = float(source_values.std(ddof=source_ddof))
        source_sharpe = float(source_values.mean() / source_std * math.sqrt(252.0)) if source_std > 0 else np.nan
        source_basis = f"equity-dependent daily return reset at period start, zero-trade days included, sqrt(252), ddof={source_ddof}"
    elif "source_daily_return" in x.columns:
        source_values = pd.to_numeric(x["source_daily_return"], errors="raise").fillna(0.0)
        source_ddof = int(x["source_sharpe_ddof"].iloc[0])
        source_std = float(source_values.std(ddof=source_ddof))
        source_sharpe = float(source_values.mean() / source_std * math.sqrt(252.0)) if source_std > 0 else np.nan
        source_basis = f"daily return used by source engine, zero-trade days included, sqrt(252), ddof={source_ddof}"
    return {
        "sleeve": sleeve,
        "period": period,
        "configured_start": bounds[0],
        "configured_end": bounds[1],
        "observed_start": x["session_date"].min().date().isoformat(),
        "observed_end": x["session_date"].max().date().isoformat(),
        "coverage_status": coverage,
        "sessions": int(len(x)),
        "trades": int(x[trade_count_column].sum()),
        **metrics,
        "sharpe_source_current": source_sharpe,
        "source_sharpe_basis": source_basis,
    }


def _orb_daily() -> pd.DataFrame:
    trades = pd.read_csv(ORB_TRADES, parse_dates=["session_date"])
    minute = prepare_minute_dataset(ORB_DATA, BaselineEntryConfig(), (10, 14, 20, 30))
    sessions = pd.Series(
        pd.to_datetime(minute["session_date"]).dt.normalize().unique(), name="session_date"
    ).sort_values()
    daily_trade = trades.groupby("session_date").agg(
        daily_pnl_usd=("net_pnl_usd", "sum"), trade_count=("trade_id", "count")
    )
    return pd.DataFrame({"session_date": sessions}).merge(
        daily_trade, how="left", on="session_date"
    ).fillna({"daily_pnl_usd": 0.0, "trade_count": 0}).assign(
        source_daily_return=lambda x: x["daily_pnl_usd"] / 50_000.0,
        source_sharpe_ddof=0,
    )


def _load_strategy_daily() -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {"mnq_orb_vwap_atr": _orb_daily()}

    mnq_config = json.loads(MNQ_CONFIG.read_text(encoding="utf-8"))
    mnq_runs, _, _ = run_alpha_sleeves(mnq_config)
    for mechanism in ("volatility_breakout", "compression_breakout", "atr_trend_flip"):
        frame = mnq_runs[mechanism].daily[["session_date", "daily_pnl_usd", "trade_count", "daily_return"]].copy()
        frame = frame.rename(columns={"daily_return": "source_daily_return"})
        frame["source_sharpe_ddof"] = 1
        frame["source_equity_based"] = 1
        frame["session_date"] = pd.to_datetime(frame["session_date"])
        output[f"mnq_{mechanism}"] = frame

    mgc_config = json.loads(MGC_CONFIG.read_text(encoding="utf-8"))
    mgc_start, mgc_end = date(2019, 5, 6), date(2026, 3, 19)
    mgc_data, _ = prepare_mgc_data(
        ROOT / mgc_config["source_locks"]["mgc_minute_data"]["path"], mgc_start, mgc_end
    )
    for sleeve, lock_key in (
        ("mgc_volatility_breakout", "mgc_volatility_spec"),
        ("mgc_gap_continuation", "mgc_gap_spec"),
    ):
        spec = json.loads((ROOT / mgc_config["source_locks"][lock_key]["path"]).read_text(encoding="utf-8"))
        run = run_baseline_backtest(spec, mgc_data, evaluation_start=mgc_start, evaluation_end=mgc_end)
        frame = run.daily[["session_date", "daily_pnl_usd", "trade_count", "daily_return"]].copy()
        frame = frame.rename(columns={"daily_return": "source_daily_return"})
        frame["source_sharpe_ddof"] = 1
        frame["source_equity_based"] = 1
        frame["session_date"] = pd.to_datetime(frame["session_date"])
        output[sleeve] = frame

    mbt = pd.read_csv(MBT_DAILY, parse_dates=["session_date"])
    mbt = mbt.loc[mbt["cost_scenario"].eq("nominal")].copy()
    for mechanism in ("rolling_range_breakout", "trend_pullback_ma_structure", "volatility_breakout"):
        frame = mbt.loc[mbt["sleeve"].eq(mechanism), ["session_date", "daily_pnl_usd", "trade_count"]].copy()
        output[f"mbt_{mechanism}"] = frame
    return output


def _book_outputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bounds = config["book_window"]
    data = pd.read_csv(BOOK_DAILY, parse_dates=["session_date"])
    data = data.loc[data["session_date"].between(bounds["start"], bounds["end"], inclusive="both")].copy()
    if len(data) != int(bounds["expected_sessions"]):
        raise RuntimeError(f"Book session count changed: {len(data)}")
    pnl = data[SLEEVES].fillna(0.0)
    full = pd.DataFrame([{"portfolio": "book_full", "sessions": len(data), **_daily_metrics(pnl.sum(axis=1))}])

    cov = pnl.cov(ddof=1).to_numpy(float)
    weights = np.ones(len(SLEEVES))
    portfolio_variance = float(weights @ cov @ weights)
    shares = weights * (cov @ weights) / portfolio_variance
    risk = pd.DataFrame(
        {
            "code": [f"S{i:02d}" for i in range(1, 10)],
            "sleeve": SLEEVES,
            "daily_volatility_usd": pnl.std(ddof=1).to_numpy(),
            "worst_day_usd": pnl.min().to_numpy(),
            "max_drawdown_usd": [_daily_metrics(pnl[column])["max_drawdown_usd"] for column in SLEEVES],
            "risk_contribution_pct": shares * 100.0,
        }
    )

    rows = []
    for code, omitted in zip(risk["code"], SLEEVES):
        metrics = _daily_metrics(pnl.drop(columns=omitted).sum(axis=1))
        rows.append(
            {
                "omitted_code": code,
                "omitted_sleeve": omitted,
                **metrics,
                "net_pnl_change_vs_full_usd": metrics["net_pnl_usd"] - full.iloc[0]["net_pnl_usd"],
                "max_drawdown_change_vs_full_usd": metrics["max_drawdown_usd"] - full.iloc[0]["max_drawdown_usd"],
            }
        )
    return full, risk, pd.DataFrame(rows)


def _comparison(periods: pd.DataFrame) -> pd.DataFrame:
    code_map = {f"S{i:02d}": sleeve for i, sleeve in enumerate(SLEEVES, 1)}
    rows = []
    for code, period, trades, pnl, sharpe, drawdown in DOCUMENT_CLAIMS:
        found = periods.loc[(periods["sleeve"] == code_map[code]) & (periods["period"] == period)]
        if len(found) != 1:
            rows.append({"code": code, "period": period, "status": "not_reproduced", "reason": "period row absent"})
            continue
        actual = found.iloc[0]
        item = {
            "code": code,
            "period": period,
            "document_trades": trades,
            "reproduced_trades": actual.get("trades"),
            "delta_trades": actual.get("trades") - trades,
            "document_net_pnl_usd": pnl,
            "reproduced_net_pnl_usd": actual.get("net_pnl_usd"),
            "delta_net_pnl_usd": actual.get("net_pnl_usd") - pnl,
            "document_sharpe": sharpe,
            "reproduced_source_sharpe": actual.get("sharpe_source_current"),
            "canonical_pnl_sharpe": actual.get("sharpe_annualized"),
            "delta_document_vs_source_sharpe": actual.get("sharpe_source_current") - sharpe if sharpe is not None else np.nan,
            "delta_document_vs_canonical_sharpe": actual.get("sharpe_annualized") - sharpe if sharpe is not None else np.nan,
            "document_max_drawdown_usd": drawdown,
            "reproduced_max_drawdown_usd": actual.get("max_drawdown_usd"),
            "delta_max_drawdown_usd": actual.get("max_drawdown_usd") - drawdown if drawdown is not None else np.nan,
        }
        numeric_deltas = [abs(float(item[k])) for k in ("delta_trades", "delta_net_pnl_usd")]
        if sharpe is not None:
            numeric_deltas.append(abs(float(item["delta_document_vs_source_sharpe"])))
        if drawdown is not None:
            numeric_deltas.append(abs(float(item["delta_max_drawdown_usd"])))
        item["status"] = "match" if max(numeric_deltas) <= 0.001 else "difference"
        rows.append(item)
    return pd.DataFrame(rows)


def _traceability(periods: pd.DataFrame, full: pd.DataFrame, risk: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in periods.iterrows():
        if row.get("coverage_status") == "missing":
            continue
        for metric in ("trades", "net_pnl_usd", "sharpe_annualized", "sharpe_source_current", "max_drawdown_usd"):
            rows.append(
                {
                    "figure": row.get(metric),
                    "section": f"{row['sleeve']} / {row['period']}",
                    "metric": metric,
                    "source_file": SOURCE_FILES[row["sleeve"]],
                    "function": "audit_signal_sheets_step0._period_row + _daily_metrics",
                }
            )
    for metric in ("net_pnl_usd", "daily_volatility_usd", "worst_day_usd", "max_drawdown_usd", "sharpe_annualized"):
        rows.append(
            {
                "figure": full.iloc[0][metric],
                "section": "book / common window",
                "metric": metric,
                "source_file": "research/mbt_nine_sleeve_topstep_20261004/daily_nine_sleeves_nominal.csv",
                "function": "audit_signal_sheets_step0._book_outputs + _daily_metrics",
            }
        )
    for _, row in risk.iterrows():
        rows.append(
            {
                "figure": row["risk_contribution_pct"],
                "section": f"book risk / {row['code']}",
                "metric": "risk_contribution_pct",
                "source_file": "research/mbt_nine_sleeve_topstep_20261004/daily_nine_sleeves_nominal.csv",
                "function": "audit_signal_sheets_step0._book_outputs (w_i (Sigma w)_i / w' Sigma w)",
            }
        )
    for _, row in loo.iterrows():
        for metric in ("net_pnl_usd", "sharpe_annualized", "max_drawdown_usd"):
            rows.append(
                {
                    "figure": row[metric],
                    "section": f"book leave-one-out / omit {row['omitted_code']}",
                    "metric": metric,
                    "source_file": "research/mbt_nine_sleeve_topstep_20261004/daily_nine_sleeves_nominal.csv",
                    "function": "audit_signal_sheets_step0._book_outputs + _daily_metrics",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    daily = _load_strategy_daily()
    rows = []
    for sleeve in SLEEVES:
        for period, bounds in config["periods"][sleeve].items():
            rows.append(_period_row(sleeve, period, bounds, daily[sleeve]))
    periods = pd.DataFrame(rows)
    full, risk, loo = _book_outputs(config)
    comparison = _comparison(periods)
    traceability = _traceability(periods, full, risk, loo)

    periods.to_csv(OUT / "sleeve_period_reproduction.csv", index=False)
    full.to_csv(OUT / "book_reproduction.csv", index=False)
    risk.to_csv(OUT / "book_risk_reproduction.csv", index=False)
    loo.to_csv(OUT / "book_leave_one_out_reproduction.csv", index=False)
    comparison.to_csv(OUT / "document_claim_comparison.csv", index=False)
    traceability.to_csv(OUT / "number_traceability.csv", index=False)
    print(periods.to_string(index=False))
    print("\nDOCUMENT COMPARISON\n", comparison.to_string(index=False))
    print("\nBOOK\n", full.to_string(index=False))


if __name__ == "__main__":
    main()
