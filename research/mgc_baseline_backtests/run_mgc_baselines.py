"""Run one raw MGC baseline for each of the 16 frozen MNQ mechanisms.

Only contract metadata, costs, and session clocks are adapted. Signal logic,
alpha parameters, and risk formulas remain those of the source frozen specs.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import json
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.easter import easter

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
TRADE_DIR = OUTPUT_DIR / "mgc_baseline_trade_level"
FROZEN_DIR = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO_ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
CONFIG_PATH = OUTPUT_DIR / "mgc_baseline_config.json"
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "parquet" / "MGC_c_0_1m_20260322_155729.parquet"
START_DATE = date(2019, 5, 6)
END_DATE = date(2023, 12, 31)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first_weekday, _ = calendar.monthrange(year, month)
    return date(year, month, 1 + ((weekday - first_weekday) % 7) + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    _, days = calendar.monthrange(year, month)
    result = date(year, month, days)
    while result.weekday() != weekday:
        result -= timedelta(days=1)
    return result


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def build_cme_metals_prop_calendar(start: date, end: date) -> pd.DataFrame:
    """Materialize shortened COMEX sessions relevant to the prop flatten.

    Regular MGC signal bars stop at the 13:30 ET core-session close, while the
    prop hard flatten is 16:08 ET. On listed U.S. holiday sessions, metals stop
    trading at or before the times below, so flatten is one minute earlier.
    """
    entries: dict[date, tuple[time, str]] = {}

    def add(day: date, close: time, reason: str) -> None:
        if start <= day <= end and day.weekday() < 5:
            current = entries.get(day)
            if current is None or close < current[0]:
                entries[day] = (close, reason)

    for year in range(start.year, end.year + 1):
        add(_nth_weekday(year, 1, calendar.MONDAY, 3), time(13, 30), "Martin Luther King Jr. Day")
        add(_nth_weekday(year, 2, calendar.MONDAY, 3), time(13, 30), "Presidents Day")
        add(easter(year) - timedelta(days=2), time(9, 15), "Good Friday")
        add(_last_weekday(year, 5, calendar.MONDAY), time(13, 30), "Memorial Day")
        if year >= 2022:
            add(_observed(date(year, 6, 19)), time(13, 30), "Juneteenth observed")
        independence = _observed(date(year, 7, 4))
        add(independence, time(13, 30), "Independence Day observed")
        july_third = date(year, 7, 3)
        if july_third.weekday() < 5 and july_third != independence:
            add(july_third, time(13, 45), "Independence Day eve")
        add(_nth_weekday(year, 9, calendar.MONDAY, 1), time(13, 30), "Labor Day")
        thanksgiving = _nth_weekday(year, 11, calendar.THURSDAY, 4)
        add(thanksgiving, time(13, 30), "Thanksgiving Day")
        add(thanksgiving + timedelta(days=1), time(13, 45), "Day after Thanksgiving")
        christmas_eve = date(year, 12, 24)
        if christmas_eve.weekday() < 5:
            add(christmas_eve, time(13, 45), "Christmas Eve")

    return pd.DataFrame(
        [
            {
                "session_date": day,
                "market_close_time": close.isoformat(),
                "reason": reason,
                "calendar": "deterministic_CME_COMEX_metals_holiday_calendar",
            }
            for day, (close, reason) in sorted(entries.items())
        ]
    )


def prepare_mgc_data(source_path: Path, start: date, end: date) -> tuple[PreparedData, pd.DataFrame]:
    raw = pd.read_parquet(
        source_path,
        columns=["open", "high", "low", "close", "volume"],
        filters=[
            ("timestamp", ">=", pd.Timestamp(start, tz=NY)),
            ("timestamp", "<", pd.Timestamp(end + timedelta(days=1), tz=NY)),
        ],
    )
    _validate_minute_data(raw)
    raw = raw.copy()
    raw.index = raw.index.tz_convert(NY)
    raw.index.name = "timestamp"
    raw = raw[(raw.index.date >= start) & (raw.index.date <= end)]
    raw["session_date"] = raw.index.date

    calendar_frame = build_cme_metals_prop_calendar(start, end)
    early_map = {
        row.session_date: (time.fromisoformat(row.market_close_time), row.reason)
        for row in calendar_frame.itertuples(index=False)
    }
    regular_open = time(8, 20)
    regular_cash_close = time(13, 30)
    regular_last_entry = time(13, 25)
    regular_flatten = time(16, 8)
    session_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for day, group in raw.groupby("session_date", sort=True):
        if day.weekday() >= 5:
            continue
        start_ts = _timestamp(day, regular_open)
        market_close_time, reason = early_map.get(day, (None, "regular"))
        is_shortened = market_close_time is not None
        cash_close_time = min(regular_cash_close, market_close_time) if is_shortened else regular_cash_close
        cash_close = _timestamp(day, cash_close_time)
        core = group[(group.index >= start_ts) & (group.index < cash_close)]
        if core.empty:
            warnings.append(f"empty_core_session:{day}")
            continue
        close_bar = pd.Timestamp(core.index[-1])
        if close_bar != cash_close - pd.Timedelta(minutes=1):
            warnings.append(f"cash_close_proxy:{day}:{close_bar.isoformat()}")
        if is_shortened:
            market_close = _timestamp(day, market_close_time)
            hard_flatten = market_close - pd.Timedelta(minutes=1)
            calendar_last_entry = min(cash_close - pd.Timedelta(minutes=5), market_close - pd.Timedelta(minutes=5))
        else:
            hard_flatten = _timestamp(day, regular_flatten)
            calendar_last_entry = _timestamp(day, regular_last_entry)
        session_rows.append(
            {
                "session_date": day,
                "rth_start": start_ts,
                "cash_close": cash_close,
                "calendar_last_entry": calendar_last_entry,
                "hard_flatten": hard_flatten,
                "early_close": is_shortened,
                "early_close_reason": reason,
            }
        )
        benchmark_rows.append(
            {
                "session_date": day,
                "close": float(group.loc[close_bar, "close"]),
                "close_timestamp": close_bar,
                "cash_close": cash_close,
                "early_close": is_shortened,
                "early_close_reason": reason,
            }
        )

    sessions = pd.DataFrame(session_rows).sort_values("session_date").reset_index(drop=True)
    benchmark_daily = pd.DataFrame(benchmark_rows).sort_values("session_date").reset_index(drop=True)
    valid_dates = set(sessions["session_date"])
    minute = raw[raw["session_date"].isin(valid_dates)].copy()
    starts = sessions.set_index("session_date")["rth_start"].to_dict()
    ends = sessions.set_index("session_date")["hard_flatten"].to_dict()
    mask = [starts[d] <= ts <= ends[d] for ts, d in zip(minute.index, minute["session_date"])]
    minute = minute.loc[mask, ["open", "high", "low", "close", "volume", "session_date"]]

    daily_rows: list[dict[str, Any]] = []
    minute_by_date = {day: frame for day, frame in minute.groupby("session_date", sort=False)}
    for session in sessions.itertuples(index=False):
        group = minute_by_date.get(session.session_date)
        if group is None:
            warnings.append(f"missing_session:{session.session_date}")
            continue
        rth = group[(group.index >= session.rth_start) & (group.index < session.cash_close)]
        if rth.empty:
            warnings.append(f"empty_rth:{session.session_date}")
            continue
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
        calendar_frame,
    )


def _canonical_json_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if not key.startswith("_")}
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adapt_spec_to_mgc(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(source)
    source_hash = source["_frozen_spec_sha256"]
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
    spec["_source_frozen_spec_sha256"] = source_hash
    spec["_frozen_spec_sha256"] = source_hash
    spec["_effective_mgc_spec_sha256"] = _canonical_json_hash(spec)
    return spec


def adapt_benchmark_to_mgc(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(source)
    spec["mechanism_id"] = "MGC_BUY_AND_HOLD"
    spec["strategy_id"] = "MGC_BUY_AND_HOLD"
    spec["hypothesis"]["statement"] = "Continuous long exposure to MGC represents the passive Gold futures market return over the fixed evaluation window."
    spec["canonical_parameters"]["structural"]["cash_close"] = config["session"]["cash_close"]
    spec["_source_frozen_spec_sha256"] = source["_frozen_spec_sha256"]
    spec["_effective_mgc_spec_sha256"] = _canonical_json_hash(spec)
    return spec


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value) or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    rendered = _fmt(value)
    return f"{rendered}%" if rendered != "n/a" else rendered


def _write_report(results: pd.DataFrame, specs: list[dict[str, Any]], benchmark: Any, prepared: PreparedData, config_hash: str) -> None:
    strategies = results[results["specification_type"] == "strategy_baseline"].sort_values("mechanism_id")
    benchmark_row = results[results["mechanism_id"] == "MGC_BUY_AND_HOLD"].iloc[0]
    proxy_count = sum(item.startswith("cash_close_proxy:") for item in prepared.data_warnings)
    empty_count = sum(item.startswith("empty_core_session:") for item in prepared.data_warnings)
    lines = [
        "# MGC — baseline brute des 16 mécanismes représentatifs",
        "",
        "## Protocole",
        "",
        f"- Période utilisée : `{START_DATE} → {END_DATE}` (IS uniquement).",
        "- Une seule configuration canonique par mécanisme; aucune grille, variante ou optimisation.",
        "- Les paramètres alpha et les formules de signal proviennent sans changement des 16 frozen baselines MNQ.",
        "- Adaptations limitées à MGC : contrat, coûts et horloge de session.",
        "- Signaux sur barres closes, exécution 1 minute, 1 MGC, aucun pyramiding.",
        "- Stop prioritaire si stop et target sont touchés pendant la même minute.",
        f"- Données : `{prepared.source_path.relative_to(REPO_ROOT).as_posix()}`; SHA-256 `{prepared.source_sha256}`.",
        f"- Configuration MGC figée : `mgc_baseline_config.json`; SHA-256 `{config_hash}`.",
        "",
        "## Adaptation du contrat et de la session",
        "",
        "- Tick : `0,10`; valeur du tick : `1 USD`; valeur du point : `10 USD`.",
        "- Commission : `1,25 USD/côté`; slippage : `1 tick/côté`.",
        "- Session de calcul MGC : `08:20–13:30 America/New_York`.",
        "- Dernière entrée standard : `13:25`; hard flatten prop : `16:08`.",
        "- Les horaires propres aux mécanismes d'ouverture sont translatés en conservant le même délai depuis l'ouverture MGC.",
        "- Les jours COMEX raccourcis imposent un flatten une minute avant la fermeture du produit.",
        "",
        "## Couverture des données",
        "",
        f"- `{len(prepared.sessions)}` séances MGC exploitables et `{len(prepared.minute):,}` barres minute dans la fenêtre IS.",
        f"- `{proxy_count}` séances n'impriment pas de transaction exactement à 13:29; leur dernier close observé avant 13:30 sert uniquement au close quotidien et au benchmark.",
        f"- `{empty_count}` jours présents dans le fichier ne contiennent aucune transaction pendant 08:20–13:30 et sont exclus.",
        "- Un fill de stratégie continue d'exiger la minute d'exécution exacte. Une minute absente entraîne un signal ignoré, jamais un fill synthétique.",
        "",
        "## Résultats",
        "",
        "Cette table est descriptive. Aucun mécanisme n'est éliminé ni classé automatiquement.",
        "",
        "| Mécanisme | Grade | Trades | PnL net USD | Expectancy | PF | Sharpe | Max DD USD | Win rate | Avg win | Avg loss | MAE | MFE | Holding min | Valide |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in strategies.itertuples(index=False):
        lines.append(
            "| " + " | ".join(
                [
                    f"`{row.mechanism_id}`", str(row.complexity_grade), str(row.trade_count),
                    _fmt(row.net_pnl_usd), _fmt(row.expectancy_per_trade_usd), _fmt(row.profit_factor),
                    _fmt(row.sharpe_annualized), _fmt(row.max_drawdown_usd), _fmt_pct(row.win_rate_pct),
                    _fmt(row.average_win_usd), _fmt(row.average_loss_usd), _fmt(row.average_mae_usd),
                    _fmt(row.average_mfe_usd), _fmt(row.average_holding_minutes), "oui" if row.backtest_valid else "non",
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Benchmark MGC_BUY_AND_HOLD",
            "",
            f"Courbe continue close-to-close, sans daily flatten ni coûts actifs : PnL normalisé `{_fmt(benchmark_row.net_pnl_usd)} USD`, "
            f"Sharpe `{_fmt(benchmark_row.sharpe_annualized)}`, max DD `{_fmt(benchmark_row.max_drawdown_usd)} USD`.",
            "",
            "## Contrôles et limites",
            "",
            f"- `{int(strategies['backtest_valid'].sum())}/16` backtests passent les contrôles de causalité, double entrée, overnight, cutoff et hard flatten.",
            "- La série est un contrat continu fourni; ses rolls peuvent créer des discontinuités dans les signaux et le benchmark.",
            "- Les fills intraminute sont reconstruits avec OHLC 1 minute; le chemin exact des prix reste inconnu.",
            "- La transposition de la session conserve les horizons relatifs à l'ouverture mais ne prétend pas valider leur pertinence économique sur Gold.",
            "- Aucun résultat de Validation 2024–2026 n'a été chargé ou calculé.",
            "",
            "### Diagnostics d'exécution par mécanisme",
            "",
            "`Valide` signifie que les contrôles de causalité et d'exécution passent; ce statut ne mesure ni la qualité des données ni la suffisance de l'échantillon.",
            "",
            "| Mécanisme | Diagnostic |",
            "|---|---|",
        ]
    )
    for row in strategies.itertuples(index=False):
        warning = row.implementation_warning if isinstance(row.implementation_warning, str) and row.implementation_warning else "Aucun avertissement."
        lines.append(f"| `{row.mechanism_id}` | {warning.replace('|', '/')} |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Chaque ligne de résultats conserve le SHA-256 du frozen baseline MNQ source et le SHA-256 de la spécification MGC effective construite en mémoire. Les frozen baselines sources ne sont jamais modifiées.",
        ]
    )
    (OUTPUT_DIR / "MGC_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(data_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_hash = _sha256(CONFIG_PATH)
    (OUTPUT_DIR / "mgc_baseline_config.sha256").write_text(f"{config_hash}  mgc_baseline_config.json\n", encoding="ascii")

    source_paths = sorted(FROZEN_DIR.glob("*.json")) + [FROZEN_MANIFEST]
    before = {str(path): _sha256(path) for path in source_paths}
    source_specs, source_benchmark, manifest = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    specs = [adapt_spec_to_mgc(spec, config) for spec in source_specs]
    benchmark_spec = adapt_benchmark_to_mgc(source_benchmark, config)
    prepared, calendar_frame = prepare_mgc_data(data_path, START_DATE, END_DATE)
    calendar_frame.to_csv(OUTPUT_DIR / "cme_comex_metals_early_closes.csv", index=False)

    benchmark_run = build_buy_and_hold(benchmark_spec, prepared)
    benchmark_run.daily["mechanism_id"] = "MGC_BUY_AND_HOLD"
    benchmark_run.equity["mechanism_id"] = "MGC_BUY_AND_HOLD"
    all_runs: dict[str, Any] = {"MGC_BUY_AND_HOLD": benchmark_run}
    metric_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []

    for number, spec in enumerate(specs, start=1):
        print(f"[{number:02d}/16] {spec['mechanism_id']}", flush=True)
        run = run_baseline_backtest(spec, prepared)
        all_runs[spec["mechanism_id"]] = run
        row = compute_metrics(spec, run, benchmark_run.daily)
        row["source_frozen_spec_sha256"] = spec["_source_frozen_spec_sha256"]
        row["effective_mgc_spec_sha256"] = spec["_effective_mgc_spec_sha256"]
        metric_rows.append(row)
        check_rows.append(serialize_checks(spec["mechanism_id"], run.checks))
        run.trades.to_csv(TRADE_DIR / f"{spec['mechanism_id']}.csv", index=False)

    benchmark_row = compute_metrics(benchmark_spec, benchmark_run, benchmark_run.daily)
    benchmark_row["source_frozen_spec_sha256"] = benchmark_spec["_source_frozen_spec_sha256"]
    benchmark_row["effective_mgc_spec_sha256"] = benchmark_spec["_effective_mgc_spec_sha256"]
    metric_rows.append(benchmark_row)
    check_rows.append(serialize_checks("MGC_BUY_AND_HOLD", benchmark_run.checks))
    results = pd.DataFrame(metric_rows)
    results.to_csv(OUTPUT_DIR / "mgc_baseline_results.csv", index=False)
    pd.DataFrame(check_rows).to_csv(OUTPUT_DIR / "implementation_checks.csv", index=False)

    daily_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    identities = {spec["mechanism_id"]: spec for spec in [*specs, benchmark_spec]}
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
    pd.concat(daily_frames, ignore_index=True).to_csv(OUTPUT_DIR / "mgc_baseline_daily_returns.csv", index=False)
    pd.concat(equity_frames, ignore_index=True).to_csv(OUTPUT_DIR / "mgc_baseline_equity_curves.csv", index=False)
    if position_frames:
        pd.concat(position_frames, ignore_index=True).to_csv(OUTPUT_DIR / "mgc_baseline_position_intervals.csv", index=False)

    _write_report(results, specs, benchmark_run, prepared, config_hash)
    after = {str(path): _sha256(path) for path in source_paths}
    if before != after:
        raise RuntimeError("A source frozen baseline changed during MGC execution")

    strategies = results[results["specification_type"] == "strategy_baseline"]
    metadata = {
        "run_type": config["run_type"],
        "evaluation_window": config["evaluation_window"],
        "validation_used": False,
        "optimization_performed": False,
        "grid_search_performed": False,
        "variant_count_per_mechanism": 1,
        "strategy_count": 16,
        "benchmark": "MGC_BUY_AND_HOLD",
        "data_path": str(data_path.relative_to(REPO_ROOT)),
        "data_sha256": prepared.source_sha256,
        "data_rows_used": int(len(prepared.minute)),
        "session_count": int(len(prepared.sessions)),
        "early_close_sessions_used": int(prepared.sessions["early_close"].sum()),
        "config_sha256": config_hash,
        "source_manifest_rows": int(len(manifest)),
        "source_frozen_files_unchanged": True,
        "backtest_valid_count": int(strategies["backtest_valid"].sum()),
        "data_warnings": prepared.data_warnings,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Sessions: {len(prepared.sessions)}")
    print(f"Strategies executed: {len(specs)}")
    print(f"Valid strategy backtests: {metadata['backtest_valid_count']}")
    print("Validation used: 0")
    print("Optimization/grid search: 0")
    print("Source frozen files modified: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    execute(args.data.resolve())


if __name__ == "__main__":
    main()
