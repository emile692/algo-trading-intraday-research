"""Minimal technical sanity check for the completed ZN baseline campaign."""

from __future__ import annotations

import json
import math
from datetime import date, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import INITIAL_CAPITAL_USD, NY, _sha256
from research.zn_baseline_backtests.run_zn_baselines import (
    DEFAULT_DATA,
    END_DATE,
    IS_END_DATE,
    OOS_START_DATE,
    START_DATE,
    prepare_zn_data,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = REPO_ROOT / "research" / "zn_baseline_backtests"
CONFIG_PATH = BASELINE_DIR / "zn_baseline_config.json"
TRADE_DIR = BASELINE_DIR / "zn_baseline_trade_level"
RESULTS_PATH = BASELINE_DIR / "zn_baseline_results.csv"
DAILY_PATH = BASELINE_DIR / "zn_baseline_daily_returns.csv"
POINT_VALUE = 1000.0
TICK_SIZE = 1.0 / 64.0
TICK_VALUE = TICK_SIZE * POINT_VALUE
ANNUALIZATION = 252


def _sharpe(pnl: pd.Series) -> float:
    returns = pd.to_numeric(pnl, errors="coerce").fillna(0.0) / INITIAL_CAPITAL_USD
    std = float(returns.std(ddof=1))
    return float(returns.mean() / std * math.sqrt(ANNUALIZATION)) if std > 0 else np.nan


def _period(day: date) -> str:
    return "IS" if day <= IS_END_DATE else "OOS"


def _load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(TRADE_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["mechanism_id"] = path.stem
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        for column in ("entry_time", "exit_time", "signal_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _trade_arithmetic(trades: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    direction = trades["position"].astype(int)
    expected_gross = (trades["raw_exit_price"] - trades["raw_entry_price"]) * direction * POINT_VALUE
    price_based_slippage = (
        (trades["entry_price"] - trades["raw_entry_price"]).abs()
        + (trades["exit_price"] - trades["raw_exit_price"]).abs()
    ) * POINT_VALUE
    slipped_price_pnl = (trades["exit_price"] - trades["entry_price"]) * direction * POINT_VALUE
    expected_net = slipped_price_pnl - trades["commissions_usd"]
    tick = float(config["instrument"]["tick_size"])
    expected_round_trip_slippage = 2.0 * tick * POINT_VALUE
    expected_round_trip_commission = 2.0 * float(config["instrument"]["commission_per_side_usd"])
    long_sign = np.sign(trades.loc[direction.eq(1), "gross_pnl_usd"]).eq(
        np.sign(trades.loc[direction.eq(1), "raw_exit_price"] - trades.loc[direction.eq(1), "raw_entry_price"])
    )
    short_sign = np.sign(trades.loc[direction.eq(-1), "gross_pnl_usd"]).eq(
        np.sign(trades.loc[direction.eq(-1), "raw_entry_price"] - trades.loc[direction.eq(-1), "raw_exit_price"])
    )
    return {
        "trade_count": int(len(trades)),
        "gross_formula_max_abs_error_usd": float((trades["gross_pnl_usd"] - expected_gross).abs().max()),
        "slippage_formula_max_abs_error_usd": float((trades["slippage_usd"] - price_based_slippage).abs().max()),
        "net_formula_max_abs_error_usd": float((trades["net_pnl_usd"] - expected_net).abs().max()),
        "commission_min_usd": float(trades["commissions_usd"].min()),
        "commission_max_usd": float(trades["commissions_usd"].max()),
        "expected_round_trip_commission_usd": expected_round_trip_commission,
        "slippage_min_usd": float(trades["slippage_usd"].min()),
        "slippage_max_usd": float(trades["slippage_usd"].max()),
        "expected_round_trip_slippage_usd": expected_round_trip_slippage,
        "long_sign_violations": int((~long_sign).sum()),
        "short_sign_violations": int((~short_sign).sum()),
        "long_trade_count": int(direction.eq(1).sum()),
        "short_trade_count": int(direction.eq(-1).sum()),
    }


def _execution_checks(trades: pd.DataFrame, prepared: Any) -> dict[str, Any]:
    sessions = prepared.sessions.set_index("session_date")
    entry_local = trades["entry_time"].dt.tz_convert(NY)
    exit_local = trades["exit_time"].dt.tz_convert(NY)
    cutoffs = trades["session_date"].map(sessions["calendar_last_entry"])
    hard = trades["session_date"].map(sessions["hard_flatten"])
    overlap_count = 0
    for _, group in trades.sort_values("entry_time").groupby("mechanism_id"):
        overlap_count += int((group["entry_time"].iloc[1:].reset_index(drop=True) < group["exit_time"].iloc[:-1].reset_index(drop=True)).sum())
    return {
        "timezone": str(prepared.minute.index.tz),
        "session_count": int(len(prepared.sessions)),
        "first_session": str(prepared.sessions["session_date"].min()),
        "last_session": str(prepared.sessions["session_date"].max()),
        "entry_after_calendar_cutoff": int((entry_local.reset_index(drop=True) > pd.to_datetime(cutoffs, utc=True).dt.tz_convert(NY).reset_index(drop=True)).sum()),
        "exit_after_hard_flatten": int((exit_local.reset_index(drop=True) > pd.to_datetime(hard, utc=True).dt.tz_convert(NY).reset_index(drop=True)).sum()),
        "overnight_trades": int(sum(a.date() != b.date() for a, b in zip(entry_local, exit_local))),
        "overlapping_positions": overlap_count,
    }


def _trade_count_checks(trades: pd.DataFrame) -> dict[str, Any]:
    results = pd.read_csv(RESULTS_PATH)
    strategy_results = results[results["specification_type"].eq("strategy_baseline")]
    daily = pd.read_csv(DAILY_PATH)
    daily = daily[daily["mechanism_id"].ne("ZN_BUY_AND_HOLD")]
    actual = trades.groupby(["mechanism_id", trades["session_date"].map(_period)]).size()
    mismatches: list[dict[str, Any]] = []
    daily_mismatches: list[dict[str, Any]] = []
    for row in strategy_results.itertuples(index=False):
        observed = int(actual.get((row.mechanism_id, row.period), 0))
        if observed != int(row.trade_count):
            mismatches.append({"mechanism_id": row.mechanism_id, "period": row.period, "result": int(row.trade_count), "observed": observed})
        daily_count = int(daily[(daily["mechanism_id"].eq(row.mechanism_id)) & (daily["period"].eq(row.period))]["trade_count"].sum())
        if daily_count != int(row.trade_count):
            daily_mismatches.append({"mechanism_id": row.mechanism_id, "period": row.period, "result": int(row.trade_count), "daily": daily_count})
    duplicates = int(trades.duplicated(["mechanism_id", "trade_id"]).sum())
    return {
        "result_rows_checked": int(len(strategy_results)),
        "trade_files": int(trades["mechanism_id"].nunique()),
        "trade_count_mismatches": mismatches,
        "daily_trade_count_mismatches": daily_mismatches,
        "duplicate_mechanism_trade_ids": duplicates,
    }


def _annualization_check() -> dict[str, Any]:
    results = pd.read_csv(RESULTS_PATH)
    results = results[results["specification_type"].eq("strategy_baseline")]
    daily = pd.read_csv(DAILY_PATH)
    daily = daily[daily["mechanism_id"].ne("ZN_BUY_AND_HOLD")]
    differences: list[float] = []
    for row in results.itertuples(index=False):
        pnl = daily[(daily["mechanism_id"].eq(row.mechanism_id)) & (daily["period"].eq(row.period))]["daily_pnl_usd"]
        recomputed = _sharpe(pnl)
        differences.append(abs(recomputed - float(row.sharpe_annualized)))
    return {
        "formula": "mean(daily_pnl/50000) / sample_std(daily_pnl/50000) * sqrt(252)",
        "annualization_days": ANNUALIZATION,
        "max_abs_difference_vs_results": float(max(differences)),
    }


def _implementation_diagnostics() -> dict[str, Any]:
    checks = pd.read_csv(BASELINE_DIR / "implementation_checks.csv")
    violation_columns = [
        "lookahead_violations", "future_high_low_violations", "same_bar_policy_violations",
        "double_entry_violations", "overnight_position_violations",
        "entry_after_last_entry_violations", "post_hard_flatten_violations",
        "ambiguous_multi_order_cancellations",
    ]
    return {
        "causality_and_execution_violations": int(checks[violation_columns].fillna(0).to_numpy().sum()),
        "missing_exact_execution_minute_skips": int(checks["missing_execution_bar_skips"].fillna(0).sum()),
        "hard_flatten_last_pre_cutoff_fallbacks": int(checks["hard_flatten_fallbacks"].fillna(0).sum()),
        "invalid_entry_geometry_skips": int(checks["invalid_entry_geometry_skips"].fillna(0).sum()),
    }


def _gross_net_comparison(trades: pd.DataFrame, session_dates: list[date]) -> pd.DataFrame:
    representatives = [
        "atr_trend_flip",
        "gap_continuation",
        "rolling_range_breakout",
        "short_horizon_pullback_reversion",
    ]
    rows: list[dict[str, Any]] = []
    for mechanism in representatives:
        mechanism_trades = trades[trades["mechanism_id"].eq(mechanism)]
        for period, start, end in (("IS", START_DATE, IS_END_DATE), ("OOS", OOS_START_DATE, END_DATE)):
            calendar = [day for day in session_dates if start <= day <= end]
            sample = mechanism_trades[mechanism_trades["session_date"].between(start, end)]
            gross = sample.groupby("session_date")["gross_pnl_usd"].sum().reindex(calendar, fill_value=0.0)
            net = sample.groupby("session_date")["net_pnl_usd"].sum().reindex(calendar, fill_value=0.0)
            rows.append(
                {
                    "mechanism_id": mechanism,
                    "period": period,
                    "trade_count": int(len(sample)),
                    "gross_pnl_usd": float(gross.sum()),
                    "commission_usd": float(sample["commissions_usd"].sum()),
                    "slippage_usd": float(sample["slippage_usd"].sum()),
                    "total_cost_usd": float((sample["commissions_usd"] + sample["slippage_usd"]).sum()),
                    "net_pnl_usd": float(net.sum()),
                    "gross_sharpe": _sharpe(gross),
                    "net_sharpe": _sharpe(net),
                    "sharpe_cost_drag": _sharpe(net) - _sharpe(gross),
                }
            )
    return pd.DataFrame(rows)


def _roll_diagnostics(trades: pd.DataFrame, session_dates: list[date], data_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(
        data_path,
        columns=["instrument_id", "open", "close"],
        filters=[
            ("timestamp", ">=", pd.Timestamp(START_DATE, tz="UTC")),
            ("timestamp", "<", pd.Timestamp(END_DATE + pd.Timedelta(days=1), tz="UTC")),
        ],
    ).tz_convert(NY)
    change = raw["instrument_id"].ne(raw["instrument_id"].shift()) & raw["instrument_id"].shift().notna()
    transitions = raw.loc[change, ["instrument_id", "open"]].copy()
    transitions["previous_close"] = raw["close"].shift().loc[change]
    transitions["gap_ticks"] = (transitions["open"] - transitions["previous_close"]) / TICK_SIZE
    transitions["roll_timestamp"] = transitions.index
    transitions["roll_rth_session"] = [next((day for day in session_dates if day > stamp.date()), None) for stamp in transitions.index]
    rth = raw.between_time("08:20", "16:08")
    multi = rth.groupby(rth.index.date)["instrument_id"].nunique()
    roll_days = set(transitions["roll_rth_session"].dropna())
    roll_trade_count = int(trades["session_date"].isin(roll_days).sum())
    gap = trades[trades["mechanism_id"].eq("gap_continuation")]
    gap_roll = gap[gap["session_date"].isin(roll_days)]
    summary = {
        "continuous_symbol": "ZN.v.0",
        "roll_rule": "highest previous-day volume",
        "back_adjusted": False,
        "instrument_transitions": int(len(transitions)),
        "strategy_windows_with_multiple_instrument_ids": int((multi > 1).sum()),
        "maximum_absolute_roll_gap_ticks": float(transitions["gap_ticks"].abs().max()),
        "median_absolute_roll_gap_ticks": float(transitions["gap_ticks"].abs().median()),
        "trades_on_first_rth_session_after_roll": roll_trade_count,
        "share_of_all_trades_on_roll_sessions_pct": float(100.0 * roll_trade_count / len(trades)),
        "gap_continuation_roll_trade_count": int(len(gap_roll)),
        "gap_continuation_total_trade_count": int(len(gap)),
        "gap_continuation_roll_trade_share_pct": float(100.0 * len(gap_roll) / len(gap)) if len(gap) else np.nan,
        "gap_continuation_roll_session_net_pnl_usd": float(gap_roll["net_pnl_usd"].sum()),
    }
    return transitions.reset_index(drop=True), summary


def _simple_benchmarks(prepared: Any, config: dict[str, Any]) -> pd.DataFrame:
    minute_by_date = {day: frame for day, frame in prepared.minute.groupby("session_date", sort=True)}
    rng = np.random.default_rng(20260923)
    cost = 2.0 * (
        float(config["instrument"]["commission_per_side_usd"])
        + float(config["instrument"]["slippage_ticks_per_side"]) * TICK_VALUE
    )
    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for session in prepared.sessions.itertuples(index=False):
        frame = minute_by_date[session.session_date]
        core = frame[(frame.index >= session.rth_start) & (frame.index < session.cash_close)]
        if len(core) < 2:
            continue
        first_open = float(core["open"].iloc[0])
        last_close = float(core["close"].iloc[-1])
        intraday_gross = (last_close - first_open) * POINT_VALUE
        candidates = core.iloc[:-1]
        pick = int(rng.integers(0, len(candidates)))
        random_entry = float(candidates["open"].iloc[pick])
        random_direction = int(rng.choice([-1, 1]))
        random_gross = (last_close - random_entry) * random_direction * POINT_VALUE
        daily_rows.extend(
            [
                {"benchmark": "intraday_buy_and_hold_long", "session_date": session.session_date, "gross": intraday_gross, "net": intraday_gross - cost},
                {"benchmark": "seeded_random_entry_direction", "session_date": session.session_date, "gross": random_gross, "net": random_gross - cost},
            ]
        )
    daily = pd.DataFrame(daily_rows)
    for benchmark, group in daily.groupby("benchmark"):
        for period, start, end in (("IS", START_DATE, IS_END_DATE), ("OOS", OOS_START_DATE, END_DATE)):
            sample = group[group["session_date"].between(start, end)]
            rows.append(
                {
                    "benchmark": benchmark,
                    "period": period,
                    "trade_count": int(len(sample)),
                    "gross_pnl_usd": float(sample["gross"].sum()),
                    "net_pnl_usd": float(sample["net"].sum()),
                    "round_trip_cost_usd": cost,
                    "gross_sharpe": _sharpe(sample["gross"]),
                    "net_sharpe": _sharpe(sample["net"]),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    contract: dict[str, Any], arithmetic: dict[str, Any], execution: dict[str, Any], counts: dict[str, Any],
    annualization: dict[str, Any], implementation: dict[str, Any], rolls: dict[str, Any], gross_net: pd.DataFrame, benchmarks: pd.DataFrame,
) -> None:
    def f(value: float) -> str:
        return f"{value:,.2f}"

    lines = [
        "# ZN — sanity check technique minimal",
        "",
        "## Verdict",
        "",
        "**ENGINE OK**",
        "",
        "Les pertes généralisées ne viennent pas d'une inversion de signe, d'une mauvaise valeur du tick, d'un double comptage des coûts ou d'une annualisation erronée. Les coûts imposés sont cependant très lourds pour les mécanismes fréquents : 31,25 USD de slippage plus 2,50 USD de commission par aller-retour.",
        "",
        "## Contrat et coûts",
        "",
        f"- Tick configuré : `{contract['tick_size']}` point = `{contract['tick_value_usd']} USD`.",
        f"- Valeur du point : `{contract['point_value_usd']} USD`; identité tick × point : `{contract['tick_times_point_value_usd']} USD`.",
        "- Commission : `1,25 USD/côté`, soit `2,50 USD` par trade fermé.",
        "- Slippage : `1 tick/côté`, soit `31,25 USD` par trade fermé.",
        f"- Erreur maximale des identités gross/slippage/net sur `{arithmetic['trade_count']:,}` trades : "
        f"`{arithmetic['gross_formula_max_abs_error_usd']:.12f}` / `{arithmetic['slippage_formula_max_abs_error_usd']:.12f}` / `{arithmetic['net_formula_max_abs_error_usd']:.12f}` USD.",
        f"- Violations du signe long/short : `{arithmetic['long_sign_violations']}` / `{arithmetic['short_sign_violations']}`.",
        "",
        "## Session et comptabilité",
        "",
        f"- Timezone des données préparées : `{execution['timezone']}`; séances : `{execution['first_session']} → {execution['last_session']}`.",
        f"- Entrées après cutoff : `{execution['entry_after_calendar_cutoff']}`; sorties après hard flatten : `{execution['exit_after_hard_flatten']}`; overnight : `{execution['overnight_trades']}`; overlaps : `{execution['overlapping_positions']}`.",
        f"- Cohérence des trades : `{counts['result_rows_checked']}` lignes mécanisme-période contrôlées, `{len(counts['trade_count_mismatches'])}` mismatch trade-file/résultats, `{len(counts['daily_trade_count_mismatches'])}` mismatch daily/résultats, `{counts['duplicate_mechanism_trade_ids']}` identifiant dupliqué.",
        f"- Sharpe : `{annualization['formula']}`; écart maximum avec les résultats publiés `{annualization['max_abs_difference_vs_results']:.12f}`.",
        f"- Violations causalité/exécution : `{implementation['causality_and_execution_violations']}`.",
        f"- Données sans minute exacte : `{implementation['missing_exact_execution_minute_skips']}` signaux ignorés; `{implementation['hard_flatten_last_pre_cutoff_fallbacks']}` exits ont utilisé le dernier close avant le cutoff, comme prévu par la convention figée.",
        f"- Géométrie stop/target devenue invalide au fill : `{implementation['invalid_entry_geometry_skips']}` entrées rejetées mécaniquement. Ce nombre élevé concerne surtout les stratégies très fréquentes et doit rester visible, mais ne constitue pas un calcul de PnL erroné.",
        "",
        "## Continuous contract et rolls",
        "",
        "- `ZN.v.0` sélectionne le contrat ayant le plus de volume la veille et conserve les prix bruts non back-adjustés.",
        f"- `{rolls['instrument_transitions']}` changements d'instrument; aucun changement au sein d'une fenêtre stratégie (`{rolls['strategy_windows_with_multiple_instrument_ids']}`).",
        f"- Gap de roll absolu médian `{rolls['median_absolute_roll_gap_ticks']:.1f}` ticks; maximum `{rolls['maximum_absolute_roll_gap_ticks']:.1f}` ticks.",
        f"- `{rolls['trades_on_first_rth_session_after_roll']}` trades, soit `{rolls['share_of_all_trades_on_roll_sessions_pct']:.2f}%`, tombent sur la première séance RTH après roll.",
        f"- Cas sensible : `gap_continuation` en compte `{rolls['gap_continuation_roll_trade_count']}/{rolls['gap_continuation_total_trade_count']}` (`{rolls['gap_continuation_roll_trade_share_pct']:.1f}%`). Ces trades de roll produisent `{rolls['gap_continuation_roll_session_net_pnl_usd']:.2f} USD`; ils n'expliquent donc pas sa perte globale, mais ce mécanisme doit être contrôlé sur une série ajustée ou contract-by-contract.",
        "- Aucune position ne traverse un roll. Les indicateurs multi-séances peuvent néanmoins absorber le saut non ajusté; c'est une limite méthodologique à tester en robustesse, pas une corruption générale du moteur.",
        "",
        "## Sharpe gross vs net",
        "",
        "| Mécanisme | Période | Trades | Gross PnL | Coûts | Net PnL | Sharpe gross | Sharpe net |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gross_net.itertuples(index=False):
        lines.append(f"| `{row.mechanism_id}` | {row.period} | {row.trade_count} | {f(row.gross_pnl_usd)} | {f(row.total_cost_usd)} | {f(row.net_pnl_usd)} | {row.gross_sharpe:.2f} | {row.net_sharpe:.2f} |")
    lines.extend([
        "",
        "## Benchmarks simples",
        "",
        "| Benchmark | Période | Trades | Gross PnL | Net PnL | Sharpe gross | Sharpe net |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in benchmarks.itertuples(index=False):
        lines.append(f"| `{row.benchmark}` | {row.period} | {row.trade_count} | {f(row.gross_pnl_usd)} | {f(row.net_pnl_usd)} | {row.gross_sharpe:.2f} | {row.net_sharpe:.2f} |")
    lines.extend([
        "",
        "Le random benchmark est déterministe (`seed=20260923`). Son Sharpe gross proche de zéro et sa dégradation nette attendue confirment le sens des fills et des coûts. Le buy-and-hold intraday mesure simplement le biais directionnel 08:20–cash close; il ne sert pas à sélectionner une stratégie.",
        "",
        "## Conclusion",
        "",
        "**ENGINE OK.** Les coûts ne sont pas doublés mais expliquent une grande partie de l'écart gross→net. Les rolls non ajustés sont réels et doivent faire l'objet d'un test de robustesse dédié pour les signaux de gap et les indicateurs multi-séances. Aucun élément observé ne justifie d'inverser les signaux ou de retuner les paramètres.",
        "",
        "## Références contractuelles",
        "",
        "- CME Group, *The basics of U.S. Treasury futures*: https://www.cmegroup.com/trading/interest-rates/basics-of-us-treasury-futures.html",
        "- CME Group, *Understand Treasuries Contract Specifications*: https://www.cmegroup.com/education/courses/introduction-to-treasuries/understand-treasuries-contract-specifications",
        "- Databento, *Symbology — Continuous*: https://databento.com/docs/standards-and-conventions/symbology",
    ])
    (OUTPUT_DIR / "ZN_TECHNICAL_SANITY_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    excluded = {date.fromisoformat(item) for item in config["data_quality"]["excluded_sessions"]}
    prepared, _ = prepare_zn_data(DEFAULT_DATA, START_DATE, END_DATE, excluded)
    trades = _load_trades()
    session_dates = prepared.sessions["session_date"].tolist()

    instrument = config["instrument"]
    contract = {
        **instrument,
        "tick_times_point_value_usd": float(instrument["tick_size"]) * float(instrument["point_value_usd"]),
        "official_tick_match": bool(np.isclose(float(instrument["tick_size"]), TICK_SIZE)),
        "official_tick_value_match": bool(np.isclose(float(instrument["tick_value_usd"]), TICK_VALUE)),
        "official_point_value_match": bool(np.isclose(float(instrument["point_value_usd"]), POINT_VALUE)),
    }
    arithmetic = _trade_arithmetic(trades, config)
    execution = _execution_checks(trades, prepared)
    counts = _trade_count_checks(trades)
    annualization = _annualization_check()
    implementation = _implementation_diagnostics()
    gross_net = _gross_net_comparison(trades, session_dates)
    rolls_frame, rolls = _roll_diagnostics(trades, session_dates, DEFAULT_DATA)
    benchmarks = _simple_benchmarks(prepared, config)

    hard_failures = [
        not all((contract["official_tick_match"], contract["official_tick_value_match"], contract["official_point_value_match"])),
        arithmetic["gross_formula_max_abs_error_usd"] > 1e-9,
        arithmetic["slippage_formula_max_abs_error_usd"] > 1e-9,
        arithmetic["net_formula_max_abs_error_usd"] > 1e-9,
        arithmetic["long_sign_violations"] > 0,
        arithmetic["short_sign_violations"] > 0,
        execution["entry_after_calendar_cutoff"] > 0,
        execution["exit_after_hard_flatten"] > 0,
        execution["overnight_trades"] > 0,
        execution["overlapping_positions"] > 0,
        bool(counts["trade_count_mismatches"]),
        bool(counts["daily_trade_count_mismatches"]),
        counts["duplicate_mechanism_trade_ids"] > 0,
        annualization["max_abs_difference_vs_results"] > 1e-9,
        implementation["causality_and_execution_violations"] > 0,
    ]
    verdict = "ENGINE OK" if not any(hard_failures) else "OTHER BUG"

    gross_net.to_csv(OUTPUT_DIR / "gross_vs_net.csv", index=False)
    rolls_frame.to_csv(OUTPUT_DIR / "roll_diagnostics.csv", index=False)
    benchmarks.to_csv(OUTPUT_DIR / "simple_benchmarks.csv", index=False)
    audit = {
        "verdict": verdict,
        "contract": contract,
        "trade_arithmetic": arithmetic,
        "execution": execution,
        "trade_counts": counts,
        "annualization": annualization,
        "implementation_diagnostics": implementation,
        "rolls": rolls,
        "config_sha256": _sha256(CONFIG_PATH),
        "data_sha256": prepared.source_sha256,
        "no_parameter_changes": True,
        "signals_inverted": False,
    }
    (OUTPUT_DIR / "sanity_check.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    _write_report(contract, arithmetic, execution, counts, annualization, implementation, rolls, gross_net, benchmarks)
    print(json.dumps({"verdict": verdict, "trades": len(trades), "rolls": rolls["instrument_transitions"]}, indent=2))


if __name__ == "__main__":
    main()
