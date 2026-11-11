"""Run the prespecified MNQ gap-fill baseline on development sessions only."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.analytics.metrics import compute_metrics  # noqa: E402
from src.engine.execution_model import ExecutionModel  # noqa: E402


DATASET = REPO / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet"
ORB_DAILY = REPO / "strategy_registry/psf_mnq/orb_cutoff_audit/corrected_daily_pnl.csv"
OUTPUT = Path(__file__).resolve().parent / "gap_fill_v0"
IS_END = pd.Timestamp("2024-02-27").date()
MIN_RTH_BARS = 300
TICK_SIZE = 0.25
POINT_VALUE_USD = 2.0


def daily_reference(rth: pd.DataFrame) -> pd.DataFrame:
    work = rth.assign(timestamp=rth.index, session_date=rth.index.date)
    daily = work.groupby("session_date", sort=True).agg(
        first_ts=("timestamp", "first"),
        last_ts=("timestamp", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        first_contract=("instrument_id", "first"),
        last_contract=("instrument_id", "last"),
        bar_count=("open", "size"),
    )
    daily["complete"] = (
        daily["first_ts"].map(lambda x: x.time().isoformat() == "09:30:00")
        & daily["last_ts"].map(lambda x: x.time().isoformat() == "15:59:00")
        & (daily["bar_count"] >= MIN_RTH_BARS)
        & (daily["first_contract"] == daily["last_contract"])
    )
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_contract"] = daily["last_contract"].shift(1)
    daily["prev_complete"] = daily["complete"].shift(1).eq(True)
    daily["contract_continuity"] = daily["first_contract"] == daily["prev_contract"]
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - daily["prev_close"]).abs(),
            (daily["low"] - daily["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.where(daily["complete"] & daily["prev_complete"] & daily["contract_continuity"])
    daily["atr14_prior"] = tr.rolling(14, min_periods=14).mean().shift(1)
    return daily


def run() -> None:
    bars = pd.read_parquet(
        DATASET, columns=["timestamp", "open", "high", "low", "close", "instrument_id"]
    )
    if bars.index.name != "timestamp":
        bars = bars.set_index("timestamp")
    bars = bars.sort_index()
    all_sessions = sorted({date for date in bars.index.date if date <= IS_END})
    rth = bars.between_time("09:30", "15:59")
    daily = daily_reference(rth)
    model = ExecutionModel()
    skips = Counter()
    trades = []

    for session_date, ref in daily.iterrows():
        if session_date > IS_END:
            continue
        if not bool(ref["complete"]) or not bool(ref["prev_complete"]):
            skips["incomplete_rth"] += 1
            continue
        if not bool(ref["contract_continuity"]):
            skips["contract_roll"] += 1
            continue
        atr = float(ref["atr14_prior"])
        if not np.isfinite(atr) or atr <= 0:
            skips["atr_unavailable"] += 1
            continue
        gap = float(ref["open"] - ref["prev_close"])
        if not (TICK_SIZE <= abs(gap) < atr):
            skips["gap_outside_rule"] += 1
            continue

        day = pd.Timestamp(session_date).tz_localize("America/New_York")
        first_time = day + pd.Timedelta(hours=9, minutes=30)
        entry_time = day + pd.Timedelta(hours=9, minutes=31)
        noon = day + pd.Timedelta(hours=12)
        first = rth.loc[first_time]
        if entry_time not in rth.index:
            skips["missing_0931"] += 1
            continue
        entry_bar = rth.loc[entry_time]
        direction = -1 if gap > 0 else 1
        target = float(ref["prev_close"])
        stop = float(ref["open"] + 0.5 * gap)
        if direction == -1:
            already_touched = float(first["low"]) <= target or float(first["high"]) >= stop
            entry_valid = target < float(entry_bar["open"]) < stop
        else:
            already_touched = float(first["high"]) >= target or float(first["low"]) <= stop
            entry_valid = stop < float(entry_bar["open"]) < target
        if already_touched:
            skips["touched_before_entry"] += 1
            continue
        if not entry_valid:
            skips["entry_outside_bracket"] += 1
            continue

        raw_entry = float(entry_bar["open"])
        entry_price = model.apply_slippage(raw_entry, direction, is_entry=True)
        active = rth.loc[entry_time:noon - pd.Timedelta(minutes=1)]
        if active.empty:
            skips["missing_active_bars"] += 1
            continue
        raw_exit = None
        exit_time = None
        exit_reason = None
        for ts, bar in active.iterrows():
            stop_hit = float(bar["low"]) <= stop if direction == 1 else float(bar["high"]) >= stop
            target_hit = float(bar["high"]) >= target if direction == 1 else float(bar["low"]) <= target
            if stop_hit:
                # A stop-market order can fill through its trigger at the bar open.
                raw_exit = min(stop, float(bar["open"])) if direction == 1 else max(stop, float(bar["open"]))
                exit_time, exit_reason = ts, "stop"
                break
            if target_hit:
                raw_exit, exit_time, exit_reason = target, ts, "target"
                break
        if exit_time is None:
            if noon in rth.index:
                raw_exit, exit_time = float(rth.loc[noon, "open"]), noon
            else:
                raw_exit, exit_time = float(active.iloc[-1]["close"]), active.index[-1]
            exit_reason = "time_exit"

        exit_price = model.apply_slippage(float(raw_exit), direction, is_entry=False)
        gross = (exit_price - entry_price) * direction * POINT_VALUE_USD
        fees = model.round_trip_fees(1)
        trades.append({
            "session_date": str(session_date),
            "direction": "long" if direction == 1 else "short",
            "quantity": 1,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "stop_price": stop,
            "target_price": target,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "gap_points": gap,
            "atr14_prior_points": atr,
            "fees": fees,
            "net_pnl_usd": gross - fees,
            "trade_risk_usd": abs((entry_price - stop) * POINT_VALUE_USD) + fees,
        })

    result = pd.DataFrame(trades).sort_values("session_date")
    metrics = compute_metrics(result, session_dates=all_sessions, initial_capital=50_000)
    metric_keys = ("n_trades", "cumulative_pnl", "sharpe_ratio", "max_drawdown", "profit_factor", "win_rate", "expectancy", "worst_day")
    selected_metrics = {key: float(metrics[key]) if key != "n_trades" else int(metrics[key]) for key in metric_keys}

    candidate_daily = pd.Series(0.0, index=pd.Index(all_sessions), dtype=float)
    trade_daily = result.groupby("session_date")["net_pnl_usd"].sum()
    candidate_daily.loc[pd.to_datetime(trade_daily.index).date] = trade_daily.to_numpy()
    orb = pd.read_csv(ORB_DAILY)
    orb = orb.loc[orb["split"] == "IS"].copy()
    orb.index = pd.to_datetime(orb["session_date"]).dt.date
    orb_daily = orb["net_pnl_usd"].reindex(all_sessions, fill_value=0.0).astype(float)
    down = orb_daily < 0
    comparison = {
        "daily_pnl_correlation_is": float(candidate_daily.corr(orb_daily)),
        "orb_losing_days_is": int(down.sum()),
        "candidate_mean_usd_on_orb_losing_days": float(candidate_daily.loc[down].mean()),
        "candidate_positive_fraction_on_orb_losing_days": float((candidate_daily.loc[down] > 0).mean()),
        "candidate_trade_days_on_orb_losing_days": int((candidate_daily.loc[down] != 0).sum()),
    }

    OUTPUT.mkdir(exist_ok=True)
    result.to_csv(OUTPUT / "trades_is.csv", index=False)
    pd.DataFrame({"session_date": all_sessions, "candidate_pnl_usd": candidate_daily.to_numpy(), "orb_pnl_usd": orb_daily.to_numpy()}).to_csv(OUTPUT / "daily_is.csv", index=False)
    payload = {
        "spec": "experiments/gap_fill_v0.md",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "is_end": str(IS_END),
        "rth_dates_examined": int(sum(date <= IS_END for date in daily.index)),
        "skip_counts": dict(sorted(skips.items())),
        "metrics_is": selected_metrics,
        "orb_comparison_is": comparison,
        "oos_evaluated": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Gap Fill MNQ v0 — baseline IS",
        "",
        "Règles figées dans [gap_fill_v0.md](../gap_fill_v0.md). Une seule position MNQ, coûts et slippage inclus. La période OOS n'a pas été évaluée pour cette candidate.",
        "",
        f"- Trades : **{selected_metrics['n_trades']}** sur {len(all_sessions)} dates du découpage IS.",
        f"- PnL net : **{selected_metrics['cumulative_pnl']:,.2f} $** ; espérance/trade : **{selected_metrics['expectancy']:,.2f} $**.",
        f"- Profit factor : **{selected_metrics['profit_factor']:.3f}** ; Sharpe quotidien : **{selected_metrics['sharpe_ratio']:.3f}** ; drawdown max : **{selected_metrics['max_drawdown']:,.2f} $**.",
        f"- Corrélation des PnL quotidiens avec l'ORB corrigé en IS : **{comparison['daily_pnl_correlation_is']:.3f}**.",
        f"- PnL moyen de cette candidate les {comparison['orb_losing_days_is']} jours où l'ORB perd : **{comparison['candidate_mean_usd_on_orb_losing_days']:,.2f} $** par jour, journées sans trade incluses.",
        "",
        "Les coûts fixes sont proportionnellement lourds pour un micro-contrat. Ces chiffres décrivent une baseline de recherche et ne constituent ni validation OOS ni prévision de payout.",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("gap fill IS complete", selected_metrics, comparison, dict(skips))


if __name__ == "__main__":
    run()
