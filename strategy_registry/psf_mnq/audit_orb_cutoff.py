"""Replay retained ORB exits through the declared 16:00 NY cutoff.

Uses the archived trade entries, quantities, stops and targets unchanged. This
isolates the exit-ordering bug from signal and sizing logic. The result is an
audit of the retained trade log, not a fresh independent strategy validation.
"""

from __future__ import annotations

from datetime import time
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.analytics.metrics import compute_metrics  # noqa: E402
from src.engine.execution_model import ExecutionModel  # noqa: E402


DATASET = REPO / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet"
OUTPUT = Path(__file__).resolve().parent / "orb_cutoff_audit"
CUTOFF = time(16, 0)
METRIC_KEYS = ("n_trades", "cumulative_pnl", "sharpe_ratio", "max_drawdown", "profit_factor", "win_rate", "expectancy")


def replay_one(row: pd.Series, bars: pd.DataFrame, model: ExecutionModel) -> dict | None:
    entry_time = pd.Timestamp(row["entry_time"]).tz_convert("America/New_York")
    if entry_time.time() >= CUTOFF:
        return None
    if entry_time.date() != pd.Timestamp(row["session_date"]).date():
        raise ValueError(f"Trade {row['trade_id']} has an entry outside its session date")
    day_end = entry_time.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    session = bars.loc[entry_time:day_end]
    if session.empty or session.index[0] != entry_time:
        raise ValueError(f"Missing entry bar for trade {row['trade_id']}")
    cutoff_candidates = session.index[session.index.time >= CUTOFF]
    cutoff_at = cutoff_candidates[0] if len(cutoff_candidates) else None
    active = session.loc[session.index.time < CUTOFF]
    if active.empty:
        raise ValueError(f"No active bar for trade {row['trade_id']}")

    direction = 1 if row["direction"] == "long" else -1
    stop = float(row["stop_price"])
    target = float(row["target_price"])
    if direction == 1:
        hits = active.loc[(active["low"] <= stop) | (active["high"] >= target)]
    else:
        hits = active.loc[(active["high"] >= stop) | (active["low"] <= target)]

    if not hits.empty:
        exit_time = hits.index[0]
        hit = hits.iloc[0]
        stop_hit = hit["low"] <= stop if direction == 1 else hit["high"] >= stop
        reason = "stop" if stop_hit else "target"
        raw_exit = stop if stop_hit else target
    elif cutoff_at is not None and cutoff_at.time() == CUTOFF:
        exit_time = cutoff_at
        raw_exit = float(session.loc[cutoff_at, "open"])
        reason = "time_exit"
    else:
        exit_time = active.index[-1]
        raw_exit = float(active.iloc[-1]["close"])
        reason = "time_exit" if cutoff_at is not None else "eod_exit"

    out = row.to_dict()
    exit_price = model.apply_slippage(raw_exit, direction, is_entry=False)
    quantity = int(row["quantity"])
    pnl_points = (exit_price - float(row["entry_price"])) * direction
    pnl_ticks = pnl_points / model.tick_size
    gross = pnl_ticks * 0.5 * quantity  # MNQ tick value from project settings.
    fees = model.round_trip_fees(quantity)
    out.update({
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": reason,
        "pnl_points": pnl_points,
        "pnl_ticks": pnl_ticks,
        "pnl_usd": gross,
        "fees": fees,
        "net_pnl_usd": gross - fees,
    })
    return out


def metrics_for(trades: pd.DataFrame, sessions: list) -> dict:
    values = compute_metrics(trades, session_dates=sessions, initial_capital=50_000)
    return {key: float(values[key]) if key != "n_trades" else int(values[key]) for key in METRIC_KEYS}


def main() -> None:
    archived_summary_path = OUTPUT / "archived_summary_before_fix.json"
    archived_trades_path = OUTPUT / "archived_trades_before_fix.csv"
    if not archived_summary_path.exists() or not archived_trades_path.exists():
        raise FileNotFoundError("Frozen legacy artifacts are missing; run rebuild_legacy_trades.py")
    source_summary = json.loads(archived_summary_path.read_text(encoding="utf-8"))
    archived = pd.read_csv(archived_trades_path)
    bars = pd.read_parquet(DATASET, columns=["timestamp", "open", "high", "low", "close"])
    if bars.index.name != "timestamp":
        bars = bars.set_index("timestamp")
    bars = bars.sort_index()
    sessions = sorted(set(bars.index.date))
    is_end = pd.Timestamp(source_summary["split"]["is_end"]).date()
    oos_start = pd.Timestamp(source_summary["split"]["oos_start"]).date()
    is_sessions = [day for day in sessions if day <= is_end]
    oos_sessions = [day for day in sessions if day >= oos_start]
    if len(is_sessions) != 1500 or len(oos_sessions) != 643:
        raise ValueError("Archived ORB session split does not match the dataset")

    model = ExecutionModel()
    corrected_rows = [replay_one(row, bars, model) for _, row in archived.iterrows()]
    corrected = pd.DataFrame([row for row in corrected_rows if row is not None])
    corrected_dates = pd.to_datetime(corrected["session_date"]).dt.date
    scopes = {
        "overall": (corrected, sessions),
        "is": (corrected.loc[corrected_dates.isin(is_sessions)], is_sessions),
        "oos": (corrected.loc[corrected_dates.isin(oos_sessions)], oos_sessions),
    }
    metrics = {scope: metrics_for(trades, days) for scope, (trades, days) in scopes.items()}

    old_exits = pd.to_datetime(archived["exit_time"], utc=True).dt.tz_convert("America/New_York")
    new_exits = pd.to_datetime(corrected["exit_time"], utc=True).dt.tz_convert("America/New_York")
    old_early = archived.loc[old_exits.dt.time < CUTOFF].set_index("trade_id")
    new_by_id = corrected.set_index("trade_id")
    same_early = old_early.index.intersection(new_by_id.index)
    early_mismatch = int((old_early.loc[same_early, "net_pnl_usd"].round(8) != new_by_id.loc[same_early, "net_pnl_usd"].round(8)).sum())
    if early_mismatch:
        raise ValueError(f"Replay changed {early_mismatch} trades exited before 16:00")
    if any(t.time() > CUTOFF for t in new_exits):
        raise ValueError("Corrected ORB still exits after 16:00")

    OUTPUT.mkdir(exist_ok=True)
    corrected.to_csv(OUTPUT / "corrected_trades.csv", index=False)
    daily = pd.DataFrame({"session_date": sessions})
    sums = corrected.groupby("session_date")["net_pnl_usd"].sum()
    daily["net_pnl_usd"] = daily["session_date"].astype(str).map(sums).fillna(0.0)
    daily["split"] = ["IS" if day <= is_end else "OOS" for day in sessions]
    daily.to_csv(OUTPUT / "corrected_daily_pnl.csv", index=False)

    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    payload = {
        "dataset_sha256": digest,
        "source_report": str(archived_summary_path.relative_to(REPO)),
        "archived_trades": len(archived),
        "corrected_trades": len(corrected),
        "dropped_entries_at_or_after_1600": len(archived) - len(corrected),
        "archived_exits_after_1600": int((old_exits.dt.time > CUTOFF).sum()),
        "corrected_exits_after_1600": int(sum(t.time() > CUTOFF for t in new_exits)),
        "unchanged_early_exit_mismatches": early_mismatch,
        "archived_metrics": source_summary["metrics"],
        "corrected_metrics": metrics,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Audit de la sortie horaire de l'ORB MNQ retenu",
        "",
        "Le backtester archivé cherchait un stop ou un objectif sur toutes les barres avant de considérer la sortie horaire. Un stop ou objectif atteint après 16:00 pouvait donc remplacer la sortie prévue à 16:00. La correction limite la recherche de stop/objectif aux barres antérieures au cutoff et sort à l'ouverture de 16:00, ou à la dernière clôture disponible avant 16:00 si cette barre manque. Le journal ancien et le moteur historique sont conservés dans ce dossier pour reproduire la comparaison.",
        "",
        f"- Sorties archivées après 16:00 New York : **{payload['archived_exits_after_1600']}**.",
        f"- Sorties corrigées après 16:00 : **{payload['corrected_exits_after_1600']}**.",
        f"- Entrées archivées à partir de 16:00 écartées : **{payload['dropped_entries_at_or_after_1600']}**.",
        f"- Trades initialement clôturés avant 16:00 dont le PnL change : **{early_mismatch}**.",
        "",
        "| Période | Trades anciens | PnL ancien ($) | Sharpe ancien | Trades corrigés | PnL corrigé ($) | Sharpe corrigé | DD corrigé ($) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, label in (("is", "IS"), ("oos", "OOS historique"), ("overall", "Ensemble")):
        old, new = source_summary["metrics"][scope], metrics[scope]
        lines.append(
            f"| {label} | {old['n_trades']} | {old['net_pnl']:,.2f} | {old['sharpe_ratio']:.3f} | "
            f"{new['n_trades']} | {new['cumulative_pnl']:,.2f} | {new['sharpe_ratio']:.3f} | {new['max_drawdown']:,.2f} |"
        )
    lines += [
        "",
        "La correction réutilise les entrées, stops, objectifs et quantités du journal archivé. Elle ne rejoue ni la génération des signaux ni une sélection de paramètres. L'OOS historique a déjà été consulté pendant la recherche; il ne constitue pas une nouvelle validation indépendante. Les anciennes métriques du rapport ne doivent plus servir de référence de performance.",
        "",
        "Le [rapport retenu](../../../export/mnq_orb_vwap_atr_retained/performance_report.md) a ensuite été entièrement recalculé avec le moteur corrigé et reproduit les métriques de cet audit. `audit.json` conserve les anciennes métriques; le journal courant du rapport est désormais corrigé.",
        "",
        "Reproduction : `python strategy_registry/psf_mnq/audit_orb_cutoff.py`. Le journal ancien peut être reconstruit par `rebuild_legacy_trades.py` à partir de `legacy_backtester.py`.",
        "",
        "Fichiers : `audit.json`, `archived_trades_before_fix.csv`, `corrected_trades.csv`, `corrected_daily_pnl.csv`.",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("audit complete", payload["archived_exits_after_1600"], metrics)


if __name__ == "__main__":
    main()
