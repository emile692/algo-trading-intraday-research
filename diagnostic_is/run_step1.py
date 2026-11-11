"""IS-only loss anatomy for the frozen gap and MGC climax signals.

Signal A inference is NQ only. MES, MNQ and M2K are exported separately as
robustness series. Thresholds are verified as nested before they are converted
to mutually exclusive normalized-gap buckets.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diagnostic_is.step0_inventory import (  # noqa: E402
    GAP_SOURCES,
    GAP_THRESHOLDS,
    _assert_is_only_bars,
    _assert_is_only_trades,
    _write_is_only_parquet,
)
from research.mean_reversion_cross_asset_20261006.run_campaign import (  # noqa: E402
    EXIT_SPECS,
    INSTRUMENTS as GAP_INSTRUMENTS,
    _load_bars,
    _prepare_features,
    _signals,
    backtest as gap_backtest,
)
from research.volume_auction_landscape_20261007.run_campaign import (  # noqa: E402
    INSTRUMENTS as CLIMAX_INSTRUMENTS,
    add_causal_volume_baselines,
    add_price_features,
    backtest as climax_backtest,
    load_bars,
    signals,
)


OUT = Path(__file__).resolve().parent
TABLES = OUT / "step1_tables"
PLOTS = OUT / "step1_plots"
TEMP = OUT / "_tmp"
SEED = 20261008
A_OOS_START = pd.Timestamp("2024-02-23")
B_OOS_START = pd.Timestamp("2023-03-17")
B_SOURCE = REPO / "data/processed/parquet/MGC_v_0_5m_20150101_20260923.parquet"
B_PARAMS = {
    "mechanism": "B", "mode": "confirmed_fade", "lookback": 1,
    "move": 1.5, "rvol": 2.0, "profile": 0, "confirm": 1,
}
A_BUCKETS = [0.25, 0.50, 0.75, 1.00, np.inf]
A_LABELS = ["[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00)", ">=1.00"]
B_CONFIG = "confirmed_fade_m1.5_rvol2.0"


def prior_daily_atr14(bars: pd.DataFrame) -> pd.Series:
    daily = bars.groupby("session_date", sort=True).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    prior_close = daily.close.shift(1)
    tr = pd.concat(
        [daily.high - daily.low, (daily.high - prior_close).abs(), (daily.low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    # Wilder ATR through t-1; shift makes the value known before day t opens.
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1)
    atr.index = pd.to_datetime(atr.index)
    return atr


def verify_nested(ledgers: dict[float, pd.DataFrame], symbol: str) -> list[dict[str, Any]]:
    keys = ["session_date", "signal_time", "entry_time", "exit_time", "direction"]
    numeric = ["entry_price", "exit_price", "gross_pnl", "nominal_cost", "net_pnl"]
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(GAP_THRESHOLDS[:-1], GAP_THRESHOLDS[1:]):
        lo = ledgers[lower].copy().set_index(keys).sort_index()
        hi = ledgers[upper].copy().set_index(keys).sort_index()
        missing = hi.index.difference(lo.index)
        common = lo.loc[hi.index] if len(missing) == 0 else pd.DataFrame()
        identical = len(missing) == 0 and all(
            np.allclose(common[c], hi[c], rtol=0.0, atol=1e-8) for c in numeric
        )
        if not identical:
            raise AssertionError(f"Non-identical nested trades for {symbol}: {upper} within {lower}")
        rows.append(
            {
                "asset": symbol,
                "lower_threshold": lower,
                "upper_threshold": upper,
                "lower_trades": len(lo),
                "upper_trades": len(hi),
                "missing_upper_trades": len(missing),
                "identical_on_common_keys": True,
            }
        )
    return rows


def add_normalizations(
    trades: pd.DataFrame,
    point_value: float,
    daily_atr: pd.Series,
    entry_col: str,
    gap_col: str | None = None,
) -> pd.DataFrame:
    out = trades.copy()
    dates = pd.to_datetime(out.session_date)
    out["prior_daily_atr14"] = dates.map(daily_atr)
    out["cost"] = out.gross_pnl - out.net_pnl
    out["gross_points"] = out.gross_pnl / point_value
    out["cost_points"] = out.cost / point_value
    notional = out[entry_col].abs() * point_value
    out["gross_bps"] = out.gross_pnl / notional.replace(0, np.nan) * 10_000
    out["cost_bps"] = out.cost / notional.replace(0, np.nan) * 10_000
    out["gross_daily_atr14"] = out.gross_points / out.prior_daily_atr14.replace(0, np.nan)
    out["cost_daily_atr14"] = out.cost_points / out.prior_daily_atr14.replace(0, np.nan)
    if gap_col is not None:
        gap = out[gap_col].abs().replace(0, np.nan)
        out["gross_gap_multiple"] = out.gross_points / gap
        out["cost_gap_multiple"] = out.cost_points / gap
    return out


def prepare_a(temp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fair_exit = next(spec for spec in EXIT_SPECS if spec[0] == "fair_value")
    all_trades: list[pd.DataFrame] = []
    nested_rows: list[dict[str, Any]] = []
    for symbol, source in GAP_SOURCES.items():
        temp_path = temp_dir / f"A_{symbol}_is.parquet"
        _write_is_only_parquet(source, temp_path, A_OOS_START)
        bars = _prepare_features(_load_bars(temp_path, GAP_INSTRUMENTS[symbol]))
        _assert_is_only_bars(bars, A_OOS_START)
        ledgers: dict[float, pd.DataFrame] = {}
        for threshold in GAP_THRESHOLDS:
            signal, fair = _signals(
                bars, "gap_reversion", {"lookback": 3, "threshold": threshold}
            )
            trades = gap_backtest(bars, signal, fair, GAP_INSTRUMENTS[symbol], fair_exit)
            _assert_is_only_trades(trades, A_OOS_START)
            ledgers[threshold] = trades
        nested_rows.extend(verify_nested(ledgers, symbol))

        base = ledgers[0.25].copy()
        session = bars.groupby("session_date", sort=False).first()[
            ["session_open", "prior_day_close", "prior_range"]
        ].copy()
        session.index = pd.to_datetime(session.index)
        dates = pd.to_datetime(base.session_date)
        base["session_open"] = dates.map(session.session_open)
        base["prior_close"] = dates.map(session.prior_day_close)
        base["prior_range"] = dates.map(session.prior_range)
        base["gap_points"] = base.session_open - base.prior_close
        base["gap_norm"] = base.gap_points.abs() / base.prior_range.replace(0, np.nan)
        base["bucket"] = pd.cut(
            base.gap_norm, A_BUCKETS, labels=A_LABELS, right=False, include_lowest=True
        ).astype("string")
        if base.bucket.isna().any():
            raise AssertionError(f"Unbucketed Signal A trades for {symbol}")
        base["configuration"] = base.bucket
        base["asset"] = symbol
        base["signal"] = "A"
        base["inference_role"] = "primary" if symbol == "NQ" else "robustness"
        inst = GAP_INSTRUMENTS[symbol]
        base = add_normalizations(
            base, inst.point_value, prior_daily_atr14(bars), "entry_price", "gap_points"
        )
        base["mfe_gap_fraction"] = (base.mfe_usd / inst.point_value) / base.gap_points.abs()
        base["mae_gap_fraction"] = (base.mae_usd / inst.point_value) / base.gap_points.abs()
        all_trades.append(base)
    return pd.concat(all_trades, ignore_index=True), pd.DataFrame(nested_rows)


def prepare_b(temp_dir: Path) -> pd.DataFrame:
    temp_path = temp_dir / "B_MGC_is.parquet"
    _write_is_only_parquet(B_SOURCE, temp_path, B_OOS_START)
    inst = CLIMAX_INSTRUMENTS["MGC"]
    bars = add_price_features(add_causal_volume_baselines(load_bars(temp_path, inst)))
    _assert_is_only_bars(bars, B_OOS_START)
    trades = climax_backtest(bars, signals(bars, B_PARAMS), inst)
    _assert_is_only_trades(trades, B_OOS_START)
    signal_atr = pd.Series(bars.atr.to_numpy(), index=pd.to_datetime(bars.index, utc=True))
    signal_times = pd.to_datetime(trades.signal_time, utc=True)
    trades["signal_atr"] = signal_times.map(signal_atr)
    trades["configuration"] = B_CONFIG
    trades["bucket"] = "all"
    trades["asset"] = "MGC"
    trades["signal"] = "B"
    trades["inference_role"] = "primary"
    trades["exit_reason"] = np.where(trades.holding_bars.lt(6), "session_end", "time_stop")
    trades = add_normalizations(trades, inst.point_value, prior_daily_atr14(bars), "entry")
    trades["mfe_ticks"] = trades.mfe / inst.tick_value
    trades["mae_ticks"] = trades.mae / inst.tick_value
    trades["mfe_intraday_atr"] = (trades.mfe / inst.point_value) / trades.signal_atr
    trades["mae_intraday_atr"] = (trades.mae / inst.point_value) / trades.signal_atr
    # Requested B2 replacement, saved now for later state-variable analysis.
    climax_times = signal_times.dt.tz_convert("America/New_York") - pd.Timedelta(minutes=5)
    trades["climax_time"] = climax_times
    minute = climax_times.dt.hour * 60 + climax_times.dt.minute
    trades["session_bucket"] = pd.cut(
        minute, [500, 570, 660, 805], right=False,
        labels=["08:20-09:30", "09:30-11:00", "11:00-13:25"],
    ).astype("string")
    if trades.session_bucket.isna().any():
        raise AssertionError("MGC trade outside the approved B2 session buckets")
    return trades


def group_pnl(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        trades.groupby(group_cols, dropna=False, observed=True)
        .agg(
            trades=("net_pnl", "size"),
            gross_pnl=("gross_pnl", "sum"),
            costs=("cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            gross_per_trade=("gross_pnl", "mean"),
            cost_per_trade=("cost", "mean"),
            net_per_trade=("net_pnl", "mean"),
        )
        .reset_index()
    )


def exit_decomposition(trades: pd.DataFrame) -> pd.DataFrame:
    grouped = group_pnl(
        trades, ["signal", "asset", "configuration", "year", "exit_reason"]
    )
    totals = grouped.groupby(
        ["signal", "asset", "configuration", "year"], observed=True
    ).trades.transform("sum")
    grouped["share_of_trades"] = grouped.trades / totals
    return grouped


def normalized_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["year"] = pd.to_datetime(out.session_date).dt.year
    columns = ["gross_bps", "cost_bps", "gross_daily_atr14", "cost_daily_atr14"]
    if out.signal.iloc[0] == "A":
        columns += ["gross_gap_multiple", "cost_gap_multiple"]
    named: dict[str, tuple[str, str]] = {"trades": ("net_pnl", "size")}
    for column in columns:
        named[f"valid_{column}"] = (column, "count")
        named[f"mean_{column}"] = (column, "mean")
        named[f"median_{column}"] = (column, "median")
    return out.groupby(["signal", "asset", "configuration", "year"], observed=True).agg(**named).reset_index()


def excursion_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["year"] = pd.to_datetime(out.session_date).dt.year
    columns = ["mfe_gap_fraction", "mae_gap_fraction"] if out.signal.iloc[0] == "A" else ["mfe_ticks", "mae_ticks", "mfe_intraday_atr", "mae_intraday_atr"]
    rows: list[dict[str, Any]] = []
    for keys, group in out.groupby(["signal", "asset", "configuration", "year"], observed=True):
        row: dict[str, Any] = dict(zip(["signal", "asset", "configuration", "year"], keys))
        row["trades"] = len(group)
        for column in columns:
            values = group[column].dropna()
            row[f"{column}_p25"] = values.quantile(0.25)
            row[f"{column}_median"] = values.median()
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_p75"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def concentration(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(["signal", "asset", "configuration"], observed=True):
        total = float(group.net_pnl.sum())
        loss_pool = float(-group.loc[group.net_pnl < 0, "net_pnl"].sum())
        daily = group.groupby("session_date").net_pnl.sum().sort_values()
        for kind, n, value in (
            ("worst_trades", 10, -group.nsmallest(10, "net_pnl").net_pnl.sum()),
            ("worst_trades", 20, -group.nsmallest(20, "net_pnl").net_pnl.sum()),
            ("worst_days", 10, -daily.head(10).sum()),
        ):
            rows.append(
                {
                    "signal": keys[0], "asset": keys[1], "configuration": keys[2],
                    "kind": kind, "n": n, "net_pnl": total,
                    "loss_amount_selected": value,
                    "share_of_net_loss": value / (-total) if total < 0 else np.nan,
                    "share_of_all_losing_trades": value / loss_pool if loss_pool > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def drawdown_episodes(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(["signal", "asset", "configuration"], observed=True):
        by_date = group.assign(date=pd.to_datetime(group.session_date)).groupby("date").net_pnl.sum().sort_index()
        curve = by_date.cumsum()
        running = curve.cummax().clip(lower=0.0)
        underwater = curve - running
        active = underwater.lt(-1e-12)
        episode_id = active.ne(active.shift(fill_value=False)).cumsum()
        episodes: list[dict[str, Any]] = []
        for _, part in underwater[active].groupby(episode_id[active]):
            start_pos = max(curve.index.get_loc(part.index[0]) - 1, 0)
            start = curve.index[start_pos]
            trough = part.idxmin()
            after = underwater.loc[part.index[-1]:]
            recovered = after[after.ge(-1e-12)]
            end = recovered.index[0] if len(recovered) else curve.index[-1]
            mask = pd.to_datetime(group.session_date).between(start, end)
            episodes.append(
                {
                    "start_date": start.date().isoformat(),
                    "trough_date": trough.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "duration_calendar_days": int((end - start).days),
                    "depth": float(part.min()),
                    "trades": int(mask.sum()),
                    "recovered": bool(len(recovered)),
                }
            )
        for rank, episode in enumerate(sorted(episodes, key=lambda z: z["depth"])[:5], 1):
            rows.append(
                {"signal": keys[0], "asset": keys[1], "configuration": keys[2], "rank": rank, **episode}
            )
    return pd.DataFrame(rows)


def equity_curves(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for keys, group in trades.groupby(["signal", "asset", "configuration"], observed=True):
        daily = group.assign(date=pd.to_datetime(group.session_date)).groupby("date").net_pnl.sum().sort_index()
        rows.append(
            pd.DataFrame(
                {
                    "signal": keys[0], "asset": keys[1], "configuration": keys[2],
                    "date": daily.index, "daily_net_pnl": daily.values,
                    "cumulative_net_pnl": daily.cumsum().values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def plot_equity(curves: pd.DataFrame, signal: str, asset: str, path: Path) -> None:
    subset = curves[(curves.signal == signal) & (curves.asset == asset)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for configuration, group in subset.groupby("configuration", observed=True):
        ax.plot(pd.to_datetime(group.date), group.cumulative_net_pnl, label=str(configuration))
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"{signal} {asset} — IS cumulative net PnL")
    ax.set_ylabel("USD per contract")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def priority_nq_bucket(a: pd.DataFrame) -> None:
    subset = a[(a.asset == "NQ") & (a.configuration == "[0.50,0.75)")].copy()
    subset["year"] = pd.to_datetime(subset.session_date).dt.year
    by_year = group_pnl(subset, ["signal", "asset", "configuration", "year"])
    worst_trades = subset.nsmallest(10, "net_pnl")[
        ["session_date", "signal_time", "entry_time", "exit_time", "direction", "gap_norm", "gross_pnl", "cost", "net_pnl"]
    ]
    worst_days = (
        subset.groupby("session_date", as_index=False)
        .agg(trades=("net_pnl", "size"), gross_pnl=("gross_pnl", "sum"), costs=("cost", "sum"), net_pnl=("net_pnl", "sum"))
        .nsmallest(10, "net_pnl")
    )
    total = -float(subset.net_pnl.sum())
    metadata = {
        "trades": len(subset),
        "gross_pnl": float(subset.gross_pnl.sum()),
        "costs": float(subset.cost.sum()),
        "net_pnl": float(subset.net_pnl.sum()),
        "worst_10_trades_loss": float(-worst_trades.net_pnl.sum()),
        "worst_10_trades_share_of_net_loss": float(-worst_trades.net_pnl.sum() / total) if total > 0 else None,
    }
    by_year.to_csv(TABLES / "priority_nq_050_075_by_year.csv", index=False)
    worst_trades.to_csv(TABLES / "priority_nq_050_075_worst_10_trades.csv", index=False)
    worst_days.to_csv(TABLES / "priority_nq_050_075_worst_10_days.csv", index=False)
    (TABLES / "priority_nq_050_075_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    try:
        a, nesting = prepare_a(TEMP)
        b = prepare_b(TEMP)
    finally:
        for path in TEMP.glob("*_is.parquet"):
            path.unlink(missing_ok=True)
        try:
            TEMP.rmdir()
        except OSError:
            pass

    all_trades = pd.concat([a, b], ignore_index=True, sort=False)
    all_trades["year"] = pd.to_datetime(all_trades.session_date).dt.year
    all_trades["quarter"] = pd.to_datetime(all_trades.session_date).dt.to_period("Q").astype(str)
    nesting.to_csv(TABLES / "signal_a_threshold_nesting.csv", index=False)
    a.to_csv(TABLES / "signal_a_exclusive_bucket_trades_is.csv", index=False)
    b.to_csv(TABLES / "signal_b_trades_is.csv", index=False)
    group_pnl(all_trades, ["signal", "asset", "configuration"]).to_csv(TABLES / "pnl_by_configuration.csv", index=False)
    group_pnl(all_trades, ["signal", "asset", "configuration", "year"]).to_csv(TABLES / "pnl_by_year.csv", index=False)
    group_pnl(all_trades, ["signal", "asset", "configuration", "quarter"]).to_csv(TABLES / "pnl_by_quarter.csv", index=False)
    normalized_yearly(a).to_csv(TABLES / "normalized_pnl_yearly_signal_a.csv", index=False)
    normalized_yearly(b).to_csv(TABLES / "normalized_pnl_yearly_signal_b.csv", index=False)
    excursion_yearly(a).to_csv(TABLES / "mfe_mae_yearly_signal_a.csv", index=False)
    excursion_yearly(b).to_csv(TABLES / "mfe_mae_yearly_signal_b.csv", index=False)
    exit_decomposition(all_trades).to_csv(TABLES / "exit_decomposition_yearly.csv", index=False)
    side = np.where(all_trades.direction.gt(0), "long", "short")
    group_pnl(all_trades.assign(side=side), ["signal", "asset", "configuration", "year", "side"]).to_csv(TABLES / "long_short_yearly.csv", index=False)
    concentration(all_trades).to_csv(TABLES / "loss_concentration.csv", index=False)
    drawdown_episodes(all_trades).to_csv(TABLES / "worst_5_drawdowns.csv", index=False)
    curves = equity_curves(all_trades)
    curves.to_csv(TABLES / "equity_curves_is.csv", index=False)
    plot_equity(curves, "A", "NQ", PLOTS / "signal_a_nq_equity_curves_is.png")
    plot_equity(curves, "B", "MGC", PLOTS / "signal_b_mgc_equity_curve_is.png")
    priority_nq_bucket(a)
    metadata = {
        "seed": SEED,
        "scope": "IS_ONLY",
        "signal_a": {"start": "2010-06-07", "end": "2024-02-22", "primary": "NQ", "robustness": ["MES", "MNQ", "M2K"]},
        "signal_b": {"start": "2015-01-02", "end": "2023-03-16", "asset": "MGC"},
        "post_is_statistics_computed": False,
        "atr14_definition": "Wilder EWM alpha=1/14 on daily true range, shifted one session",
        "b_intraday_atr_definition": "median true range of prior 20 completed same-session 5m bars, minimum 10",
        "b2_session_buckets": ["08:20-09:30", "09:30-11:00", "11:00-13:25"],
        "blocked_variables": ["A1 macro calendar", "A2 VIX/VIX3M", "B0b GC volume", "B1 announcement bars"],
    }
    (OUT / "step1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(group_pnl(all_trades, ["signal", "asset", "configuration"]).to_string(index=False))


if __name__ == "__main__":
    main()
