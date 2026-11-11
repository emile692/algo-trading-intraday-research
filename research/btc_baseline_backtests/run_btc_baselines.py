"""Run the 16 frozen representative mechanisms once on CME BTC data.

The transfer keeps the frozen alpha parameters and uses a predeclared
09:30-16:00 New York comparison window. Only contract metadata and costs are
adapted. This is a first raw screen, not a robustness or OOS campaign.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import (
    INITIAL_CAPITAL_USD,
    NY,
    PreparedData,
    _resample_rth,
    _sha256,
    _timestamp,
    _true_range,
    _validate_minute_data,
    _wilder_rma,
    build_buy_and_hold,
    compute_metrics,
    load_frozen_specs,
    run_baseline_backtest,
    serialize_checks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
TRADE_DIR = OUTPUT_DIR / "btc_baseline_trade_level"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
CONFIG_PATH = OUTPUT_DIR / "btc_baseline_config.json"
DEFAULT_DATA = REPO_ROOT / "data" / "downloaded" / "BTC_v_0_1m_20171217_20261003.parquet"
START_DATE = date(2019, 5, 6)
END_DATE = date(2023, 12, 31)


def _canonical_json_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if not key.startswith("_")}
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_btc_data(source_path: Path, start: date, end: date) -> tuple[PreparedData, list[date]]:
    start_utc = pd.Timestamp(start, tz=NY).tz_convert("UTC")
    end_utc = pd.Timestamp(end + timedelta(days=1), tz=NY).tz_convert("UTC")
    raw = pd.read_parquet(
        source_path,
        columns=["open", "high", "low", "close", "volume", "instrument_id"],
        filters=[("ts_event", ">=", start_utc), ("ts_event", "<", end_utc)],
    )
    raw = raw.sort_index()
    _validate_minute_data(raw)
    raw.index = raw.index.tz_convert(NY)
    raw.index.name = "timestamp"
    raw = raw[(raw.index.date >= start) & (raw.index.date <= end)].copy()
    raw["session_date"] = raw.index.date

    session_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    roll_sessions: list[date] = []
    for day, group in raw.groupby("session_date", sort=True):
        if day.weekday() >= 5:
            continue
        rth_start = _timestamp(day, time(9, 30))
        close_bar = _timestamp(day, time(15, 59))
        hard_flatten = _timestamp(day, time(16, 8))
        required = (rth_start, close_bar, hard_flatten)
        if any(stamp not in group.index for stamp in required):
            warnings.append(f"incomplete_us_window:{day}")
            continue
        window = group[(group.index >= rth_start) & (group.index <= hard_flatten)]
        if window["instrument_id"].nunique() != 1:
            roll_sessions.append(day)
            warnings.append(f"intraday_roll_excluded:{day}")
            continue
        cash_close = _timestamp(day, time(16, 0))
        session_rows.append(
            {
                "session_date": day,
                "rth_start": rth_start,
                "cash_close": cash_close,
                "calendar_last_entry": _timestamp(day, time(15, 55)),
                "hard_flatten": hard_flatten,
                "early_close": False,
                "early_close_reason": "regular_observed_US_window",
            }
        )
        benchmark_rows.append(
            {
                "session_date": day,
                "close": float(group.loc[close_bar, "close"]),
                "close_timestamp": close_bar,
                "cash_close": cash_close,
                "early_close": False,
                "early_close_reason": "regular_observed_US_window",
            }
        )

    sessions = pd.DataFrame(session_rows).sort_values("session_date").reset_index(drop=True)
    benchmark_daily = pd.DataFrame(benchmark_rows).sort_values("session_date").reset_index(drop=True)
    valid_dates = set(sessions["session_date"])
    minute = raw[raw["session_date"].isin(valid_dates)].copy()
    starts = sessions.set_index("session_date")["rth_start"].to_dict()
    ends = sessions.set_index("session_date")["hard_flatten"].to_dict()
    keep = [starts[day] <= stamp <= ends[day] for stamp, day in zip(minute.index, minute["session_date"])]
    minute = minute.loc[keep, ["open", "high", "low", "close", "volume", "session_date"]]

    daily_rows: list[dict[str, Any]] = []
    minute_by_date = {day: frame for day, frame in minute.groupby("session_date", sort=False)}
    for session in sessions.itertuples(index=False):
        group = minute_by_date[session.session_date]
        rth = group[(group.index >= session.rth_start) & (group.index < session.cash_close)]
        daily_rows.append(
            {
                "session_date": session.session_date,
                "open": float(rth["open"].iloc[0]),
                "high": float(rth["high"].max()),
                "low": float(rth["low"].min()),
                "close": float(rth["close"].iloc[-1]),
                "close_timestamp": pd.Timestamp(rth.index[-1]),
                "volume": float(rth["volume"].sum()),
            }
        )
    daily = pd.DataFrame(daily_rows).sort_values("session_date").reset_index(drop=True)
    daily["true_range"] = _true_range(daily)
    daily["atr14"] = _wilder_rma(daily["true_range"], 14)
    daily["prior_close"] = daily["close"].shift(1)
    daily["prior_atr14"] = daily["atr14"].shift(1)
    bars = {interval: _resample_rth(minute, sessions, interval) for interval in (5, 15, 60)}
    return (
        PreparedData(
            minute=minute,
            sessions=sessions,
            bars=bars,
            daily=daily,
            benchmark_daily=benchmark_daily,
            source_path=source_path,
            source_sha256=_sha256(source_path),
            data_warnings=warnings,
        ),
        roll_sessions,
    )


def adapt_spec_to_btc(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(source)
    execution = spec["canonical_parameters"]["execution"]
    instrument = config["instrument"]
    session = config["session"]
    execution.update(
        {
            "tick_size": instrument["tick_size"],
            "tick_value_usd": instrument["tick_value_usd"],
            "point_value_usd": instrument["point_value_usd"],
            "commission_per_side_usd": instrument["commission_per_side_usd"],
            "slippage_ticks_per_side": instrument["slippage_ticks_per_side"],
            "session_timezone": session["timezone"],
            "rth_start": session["rth_start"],
            "cash_close": session["cash_close"],
            "last_entry_time": session["last_entry_time"],
            "forced_flatten_time": session["forced_flatten_time"],
            "early_close_calendar": session["calendar"],
        }
    )
    spec["canonical_parameters"]["risk_management"]["fixed_quantity"] = 1
    spec["canonical_parameters"]["risk_management"]["hard_flatten_time"] = session["forced_flatten_time"]
    mechanism = spec["mechanism_id"]
    relative = config["relative_time_adaptations"]
    if mechanism == "gap_continuation":
        execution["last_entry_time"] = relative["gap_continuation_last_entry"]
    elif mechanism == "gap_fade":
        execution["last_entry_time"] = relative["gap_fade_last_entry"]
        spec["canonical_parameters"]["risk_management"]["midday_time_exit"] = relative["gap_fade_midday_time_exit"]
    elif mechanism == "session_range_breakout":
        spec["canonical_parameters"]["structural"]["opening_range_start"] = relative["session_range_breakout_opening_range_start"]
    spec["_source_frozen_spec_sha256"] = source["_frozen_spec_sha256"]
    spec["_frozen_spec_sha256"] = source["_frozen_spec_sha256"]
    spec["_effective_btc_spec_sha256"] = _canonical_json_hash(spec)
    return spec


def adapt_benchmark_to_btc(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(source)
    spec["mechanism_id"] = "BTC_BUY_AND_HOLD"
    spec["strategy_id"] = "BTC_BUY_AND_HOLD"
    spec["hypothesis"]["statement"] = "Continuous BTC exposure sampled at the US-window close."
    spec["canonical_parameters"]["structural"]["cash_close"] = config["session"]["cash_close"]
    spec["_source_frozen_spec_sha256"] = source["_frozen_spec_sha256"]
    spec["_effective_btc_spec_sha256"] = _canonical_json_hash(spec)
    return spec


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value) or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):,.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)


def _daily_pnl_sharpe(daily: pd.DataFrame) -> float:
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    volatility = float(pnl.std(ddof=1))
    if not np.isfinite(volatility) or volatility == 0.0:
        return np.nan
    return float(pnl.mean() / volatility * np.sqrt(252.0))


def _yearly_rows(mechanism_id: str, run: Any) -> list[dict[str, Any]]:
    daily = run.daily.copy()
    daily["year"] = pd.to_datetime(daily["session_date"]).dt.year
    trades = run.trades.copy()
    if not trades.empty:
        trades["year"] = pd.to_datetime(trades["session_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for year, group in daily.groupby("year", sort=True):
        year_trades = trades[trades["year"] == year] if not trades.empty else trades
        wins = year_trades.loc[year_trades["net_pnl_usd"] > 0, "net_pnl_usd"] if not year_trades.empty else pd.Series(dtype=float)
        losses = year_trades.loc[year_trades["net_pnl_usd"] < 0, "net_pnl_usd"] if not year_trades.empty else pd.Series(dtype=float)
        gross_loss = abs(float(losses.sum()))
        rows.append(
            {
                "mechanism_id": mechanism_id,
                "year": int(year),
                "session_count": int(len(group)),
                "trade_count": int(len(year_trades)),
                "net_pnl_usd": float(group["daily_pnl_usd"].sum()),
                "daily_pnl_sharpe_annualized": _daily_pnl_sharpe(group),
                "profit_factor": float(wins.sum()) / gross_loss if gross_loss else np.nan,
            }
        )
    return rows


def _write_report(results: pd.DataFrame, prepared: PreparedData, config_hash: str, roll_sessions: list[date]) -> None:
    strategies = results[results["specification_type"] == "strategy_baseline"].sort_values("net_pnl_usd", ascending=False)
    benchmark = results[results["mechanism_id"] == "BTC_BUY_AND_HOLD"].iloc[0]
    positive = int((strategies["net_pnl_usd"] > 0).sum())
    lines = [
        "# BTC - premier test brut des 16 mecanismes representatifs",
        "",
        "## Protocole gele avant calcul",
        "",
        f"- Fenetre historique: `{START_DATE} -> {END_DATE}`; ce run est une replication IS, pas un OOS vierge.",
        "- Une configuration canonique par mecanisme; aucune grille, optimisation ou selection prealable.",
        "- Transfert de session: fenetre US `09:30-16:00 America/New_York`, puis hard flatten `16:08`.",
        "- Contrat: 1 BTC CME; tick 5 points = 25 USD; slippage 1 tick par cote.",
        "- Commission 1.25 USD par cote: hypothese de recherche, car BTC standard n'est pas un produit Topstep actuellement permis.",
        f"- Donnees: `{prepared.source_path.relative_to(REPO_ROOT).as_posix()}`; SHA-256 `{prepared.source_sha256}`.",
        f"- Configuration: `btc_baseline_config.json`; SHA-256 `{config_hash}`.",
        f"- Couverture: {len(prepared.sessions)} sessions completes, {len(prepared.minute):,} minutes; {len(roll_sessions)} sessions avec roll intraday exclues.",
        "",
        "## Resultats descriptifs",
        "",
        f"Mecanismes positifs apres couts nominaux: **{positive}/16**. Aucun statut de promotion n'est attribue sur ce seul run.",
        "",
        "| Mecanisme | Grade | Trades | PnL net USD | Expectancy | PF | Sharpe PnL | Max DD USD | Win rate | Valide |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in strategies.itertuples(index=False):
        lines.append(
            f"| `{row.mechanism_id}` | {row.complexity_grade} | {row.trade_count} | {_fmt(row.net_pnl_usd)} | "
            f"{_fmt(row.expectancy_per_trade_usd)} | {_fmt(row.profit_factor)} | {_fmt(row.daily_pnl_sharpe_annualized)} | "
            f"{_fmt(row.max_drawdown_usd)} | {_fmt(row.win_rate_pct)}% | {'oui' if row.backtest_valid else 'non'} |"
        )
    lines.extend(
        [
            "",
            "## Benchmark",
            "",
            f"`BTC_BUY_AND_HOLD` normalise a {INITIAL_CAPITAL_USD:,.0f} USD: PnL {_fmt(benchmark.net_pnl_usd)} USD, "
            f"Sharpe {_fmt(benchmark.sharpe_annualized)}, max drawdown {_fmt(benchmark.max_drawdown_usd)} USD.",
            "",
            "## Limites",
            "",
            "- BTC cote presque 23 heures; cette premiere lecture teste uniquement le transfert de la fenetre US, pas une session crypto optimisee.",
            "- Le Sharpe des strategies est calcule sur le PnL quotidien, car un contrat BTC est trop gros pour que des rendements composes sur un capital arbitraire de 50 kUSD restent interpretables.",
            "- La serie continue peut encore transmettre des discontinuities inter-session aux indicateurs et aux mecanismes de gap; les rolls intraday sont seulement exclus.",
            "- Les fills intraminute reposent sur OHLC 1 minute; stop prioritaire en cas de conflit stop/target.",
            "- Une performance brute positive ne suffit pas: robustesse, sous-periodes et OOS doivent venir ensuite.",
        ]
    )
    (OUTPUT_DIR / "BTC_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(data_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = _sha256(CONFIG_PATH)
    frozen_paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    before = {str(path): _sha256(path) for path in frozen_paths}
    source_specs, source_benchmark, manifest = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    specs = [adapt_spec_to_btc(spec, config) for spec in source_specs]
    benchmark_spec = adapt_benchmark_to_btc(source_benchmark, config)
    prepared, roll_sessions = prepare_btc_data(data_path, START_DATE, END_DATE)

    benchmark_run = build_buy_and_hold(benchmark_spec, prepared)
    benchmark_run.daily["mechanism_id"] = "BTC_BUY_AND_HOLD"
    benchmark_run.equity["mechanism_id"] = "BTC_BUY_AND_HOLD"
    all_runs: dict[str, Any] = {"BTC_BUY_AND_HOLD": benchmark_run}
    metric_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for number, spec in enumerate(specs, start=1):
        print(f"[{number:02d}/16] {spec['mechanism_id']}", flush=True)
        run = run_baseline_backtest(spec, prepared)
        all_runs[spec["mechanism_id"]] = run
        row = compute_metrics(spec, run, benchmark_run.daily)
        row["daily_pnl_sharpe_annualized"] = _daily_pnl_sharpe(run.daily)
        row["source_frozen_spec_sha256"] = spec["_source_frozen_spec_sha256"]
        row["effective_btc_spec_sha256"] = spec["_effective_btc_spec_sha256"]
        metric_rows.append(row)
        check_rows.append(serialize_checks(spec["mechanism_id"], run.checks))
        yearly_rows.extend(_yearly_rows(spec["mechanism_id"], run))
        run.trades.to_csv(TRADE_DIR / f"{spec['mechanism_id']}.csv", index=False)

    benchmark_row = compute_metrics(benchmark_spec, benchmark_run, benchmark_run.daily)
    benchmark_row["daily_pnl_sharpe_annualized"] = _daily_pnl_sharpe(benchmark_run.daily)
    benchmark_row["source_frozen_spec_sha256"] = benchmark_spec["_source_frozen_spec_sha256"]
    benchmark_row["effective_btc_spec_sha256"] = benchmark_spec["_effective_btc_spec_sha256"]
    metric_rows.append(benchmark_row)
    check_rows.append(serialize_checks("BTC_BUY_AND_HOLD", benchmark_run.checks))
    results = pd.DataFrame(metric_rows)
    results.to_csv(OUTPUT_DIR / "btc_baseline_results.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUTPUT_DIR / "btc_yearly_results.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUTPUT_DIR / "implementation_checks.csv", index=False)

    identities = {spec["mechanism_id"]: spec for spec in [*specs, benchmark_spec]}
    daily_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    for mechanism_id, run in all_runs.items():
        daily = run.daily.copy()
        daily.insert(1, "strategy_id", identities[mechanism_id]["strategy_id"])
        daily_frames.append(daily)
        equity = run.equity.copy()
        equity.insert(1, "strategy_id", identities[mechanism_id]["strategy_id"])
        equity_frames.append(equity)
        if not run.positions.empty:
            positions = run.positions.copy()
            positions.insert(1, "strategy_id", identities[mechanism_id]["strategy_id"])
            position_frames.append(positions)
    pd.concat(daily_frames, ignore_index=True).to_csv(OUTPUT_DIR / "btc_baseline_daily_returns.csv", index=False)
    pd.concat(equity_frames, ignore_index=True).to_csv(OUTPUT_DIR / "btc_baseline_equity_curves.csv", index=False)
    pd.concat(position_frames, ignore_index=True).to_csv(OUTPUT_DIR / "btc_baseline_position_intervals.csv", index=False)
    _write_report(results, prepared, config_hash, roll_sessions)

    if before != {str(path): _sha256(path) for path in frozen_paths}:
        raise RuntimeError("A source frozen baseline changed during BTC execution")
    strategies = results[results["specification_type"] == "strategy_baseline"]
    metadata = {
        "run_type": config["run_type"],
        "evaluation_window": config["evaluation_window"],
        "validation_used": False,
        "optimization_performed": False,
        "grid_search_performed": False,
        "variant_count_per_mechanism": 1,
        "strategy_count": 16,
        "benchmark": "BTC_BUY_AND_HOLD",
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": prepared.source_sha256,
        "data_rows_used": int(len(prepared.minute)),
        "session_count": int(len(prepared.sessions)),
        "intraday_roll_sessions_excluded": [str(day) for day in roll_sessions],
        "config_sha256": config_hash,
        "source_manifest_rows": int(len(manifest)),
        "source_frozen_files_unchanged": True,
        "backtest_valid_count": int(strategies["backtest_valid"].sum()),
        "positive_net_count": int((strategies["net_pnl_usd"] > 0).sum()),
        "data_warnings": prepared.data_warnings,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Sessions: {len(prepared.sessions)}")
    print(f"Strategies executed: {len(specs)}")
    print(f"Valid strategy backtests: {metadata['backtest_valid_count']}")
    print(f"Positive after nominal costs: {metadata['positive_net_count']}")
    print("Optimization/grid search: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    execute(args.data.resolve())


if __name__ == "__main__":
    main()
