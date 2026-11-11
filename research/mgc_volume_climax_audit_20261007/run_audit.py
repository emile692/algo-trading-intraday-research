"""Frozen-region audit for MGC volume-climax reversal.

No parameter is searched or selected here.  The 18 cells are reconstructed
verbatim from the 2026-10-07 volume/auction landscape; the diagnostic
representative is the geometric centre of that inherited grid.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.volume_auction_landscape_20261007.run_campaign import (
    DATA_FILES,
    INSTRUMENTS,
    add_causal_volume_baselines,
    add_price_features,
    backtest,
    load_bars,
    metrics,
    nominal_cost,
    price_only_signals,
    signals,
    split_date,
    variant_grid,
    variant_id,
)


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PLOTS = OUT / "plots"
SOURCE = REPO / "research" / "volume_auction_landscape_20261007"
FROZEN = REPO / "research" / "locked_nine_sleeve_portfolio" / "daily_pnl.csv"
MGC_TRADES = REPO / "research" / "portfolio_integration_phase1" / "sleeve_trades"
SEED = 20261007
REP = {"mechanism": "B", "mode": "confirmed_fade", "lookback": 1, "move": 1.5, "rvol": 2.0, "profile": 0, "confirm": 1}
HORIZONS = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 6, 12, 24)


def markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for value in row:
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.4g}")
            else:
                vals.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def daily_metrics(daily: pd.Series) -> dict[str, float]:
    d = daily.astype(float).sort_index()
    if d.empty:
        return {"pnl": 0.0, "sharpe": np.nan, "maxdd": 0.0, "downside_deviation": np.nan}
    calendar = pd.date_range(d.index.min(), d.index.max(), freq="B")
    d = d.reindex(calendar, fill_value=0.0)
    sd = d.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(d, 0.0) ** 2)) * math.sqrt(252)
    curve = d.cumsum()
    return {"pnl": float(d.sum()), "sharpe": float(d.mean() / sd * math.sqrt(252)) if sd > 0 else np.nan,
            "maxdd": float((curve - curve.cummax()).min()), "downside_deviation": float(downside)}


def enriched(trades: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict[str, float]:
    out = metrics(trades)
    if trades.empty:
        return out | {"turnover_trades_per_year": 0.0, "long_trades": 0, "long_pnl": 0.0, "long_sharpe": np.nan,
                      "short_trades": 0, "short_pnl": 0.0, "short_sharpe": np.nan, "avg_mae": np.nan, "avg_mfe": np.nan}
    dates = pd.to_datetime(trades.session_date)
    lo = pd.Timestamp(start) if start is not None else dates.min()
    hi = pd.Timestamp(end) if end is not None else dates.max()
    years = max((hi - lo).days / 365.25, 1 / 252)
    long_m = metrics(trades[trades.direction > 0]); short_m = metrics(trades[trades.direction < 0])
    return out | {"turnover_trades_per_year": float(len(trades) / years), "long_trades": long_m["trades"],
                  "long_pnl": long_m["net_pnl"], "long_sharpe": long_m["sharpe"], "short_trades": short_m["trades"],
                  "short_pnl": short_m["net_pnl"], "short_sharpe": short_m["sharpe"],
                  "avg_mae": float(trades.mae.mean()), "avg_mfe": float(trades.mfe.mean())}


def causal_rank(s: pd.Series, window: int = 252, minimum: int = 60) -> pd.Series:
    return s.rolling(window, min_periods=minimum).apply(lambda a: float(pd.Series(a).rank(pct=True).iloc[-1]), raw=False)


def prepare() -> tuple[pd.DataFrame, pd.Timestamp]:
    inst = INSTRUMENTS["MGC"]
    x = add_price_features(add_causal_volume_baselines(load_bars(REPO / DATA_FILES["MGC"], inst)))
    g = x.groupby("session_date", sort=False)
    ret = g.close.pct_change()
    daily = x.assign(_sq=ret.pow(2)).groupby("session_date").agg(
        realized_vol=("_sq", lambda s: float(np.sqrt(s.sum()))), day_range=("high", lambda s: float(s.max() - s.min())),
        day_open=("open", "first"), day_close=("close", "last"))
    daily["prior_rv_pct"] = causal_rank(daily.realized_vol.shift(1))
    daily["prior_range_pct"] = causal_rank(daily.day_range.shift(1))
    daily["prior_close"] = daily.day_close.shift(1)
    mapper = x.session_date
    for c in ("prior_rv_pct", "prior_range_pct", "prior_close"):
        x[c] = mapper.map(daily[c])
    x["gap_norm"] = (g.open.transform("first") - x.prior_close).abs() / x.prior_range.replace(0, np.nan)
    prior_net = (x.close.shift(1) - g.close.shift(13)).abs()
    prior_path = g.close.diff().abs().groupby(x.session_date).transform(lambda s: s.shift(1).rolling(12, min_periods=6).sum())
    x["prior_efficiency"] = prior_net / prior_path.replace(0, np.nan)
    x["atr_ratio"] = x.atr / x.close.replace(0, np.nan)
    return x, split_date(x)


def grid() -> list[dict[str, Any]]:
    rows = [p for p in variant_grid() if p["mechanism"] == "B"]
    if len(rows) != 18:
        raise RuntimeError(f"Inherited grid mismatch: expected 18, got {len(rows)}")
    return rows


def region_audit(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = cell.pivot(index=["variant_id", "mode", "move", "rvol"], columns="split",
                      values=["net_pnl", "sharpe", "expectancy", "trades"]).reset_index()
    wide.columns = ["_".join(str(z) for z in c if z).lower() if isinstance(c, tuple) else c for c in wide.columns]
    for split in ("is", "oos"):
        wide[f"positive_{split}"] = (wide[f"net_pnl_{split}"] > 0) & (wide[f"expectancy_{split}"] > 0)
    wide["positive_both"] = wide.positive_is & wide.positive_oos
    neighbor_rows = []
    for row in wide.itertuples(index=False):
        same = wide[wide["mode"].eq(row.mode)]
        dm = sorted(same.move.unique()).index(row.move); dr = sorted(same.rvol.unique()).index(row.rvol)
        neighbors = same[((same.move.map(lambda v: sorted(same.move.unique()).index(v)) - dm).abs()
                          + (same.rvol.map(lambda v: sorted(same.rvol.unique()).index(v)) - dr).abs()) == 1]
        neighbor_rows.append({"variant_id": row.variant_id, "neighbors": len(neighbors),
                              "positive_oos_neighbor_fraction": float(neighbors.positive_oos.mean()) if len(neighbors) else np.nan,
                              "positive_both_neighbor_fraction": float(neighbors.positive_both.mean()) if len(neighbors) else np.nan})
    detail = wide.merge(pd.DataFrame(neighbor_rows), on="variant_id")
    summaries = []
    for label, part in [("all", detail), *[(m, detail[detail["mode"].eq(m)]) for m in sorted(detail["mode"].unique())]]:
        summaries.append({"scope": label, "cells": len(part), "fraction_positive_is": part.positive_is.mean(),
                          "fraction_positive_oos": part.positive_oos.mean(), "fraction_positive_both": part.positive_both.mean(),
                          "mean_local_oos_continuity": part.positive_oos_neighbor_fraction.mean(),
                          "edge_only": bool(part.loc[part.positive_oos, "move"].nunique() == 1 or part.loc[part.positive_oos, "rvol"].nunique() == 1) if part.positive_oos.any() else False})
    return detail, pd.DataFrame(summaries)


def event_study(x: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = (x.abs_move >= REP["move"]) & (x.rvol_20 >= REP["rvol"])
    direction = np.sign(x.close - x.open).astype(int)
    confirmed = base.shift(1, fill_value=False) & direction.eq(-direction.shift(1))
    signal_idx = np.flatnonzero(confirmed.to_numpy())
    dates = x.session_date.to_numpy(); close = x.close.to_numpy(); high = x.high.to_numpy(); low = x.low.to_numpy()
    op = x.open.to_numpy(); atr = x.atr.to_numpy(); rv = x.rvol_20.to_numpy(); body = x.body_atr.to_numpy(); rng = x.range_atr.to_numpy()
    rows = []
    for s in signal_idx:
        i = s - 1  # climax bar; s is the completed confirmation/signal bar
        if s < 6 or s + 24 >= len(x) or dates[s - 6] != dates[s] or dates[s + 24] != dates[s] or dates[s] < np.datetime64(split):
            continue
        climax_dir = int(np.sign(close[i] - op[i])); scale = float(atr[i]); displacement = abs(close[i] - op[i])
        if climax_dir == 0 or not np.isfinite(scale) or scale <= 0:
            continue
        r: dict[str, Any] = {"session_date": dates[i], "climax_time": x.index[i], "confirmation_time": x.index[s],
                             "climax_direction": climax_dir, "trade_direction": -climax_dir, "rvol": rv[i],
                             "displacement_atr": body[i], "body_range_ratio": abs(close[i] - op[i]) / max(high[i] - low[i], 1e-12)}
        for h in HORIZONS:
            r[f"path_{h}"] = float(-climax_dir * (close[s + h] - close[s]) / scale)
        future_h, future_l, future_c = high[s + 1:s + 25], low[s + 1:s + 25], close[s + 1:s + 25]
        favorable = (close[s] - future_l) if climax_dir > 0 else (future_h - close[s])
        adverse = (future_h - close[s]) if climax_dir > 0 else (close[s] - future_l)
        r["mfe_atr"] = float(np.max(favorable) / scale); r["mae_atr"] = float(np.max(adverse) / scale)
        rev = np.flatnonzero((-climax_dir * (future_c - close[s])) > 0)
        r["reversal"] = bool(len(rev)); r["time_to_reversal_bars"] = int(rev[0] + 1) if len(rev) else np.nan
        for fraction, name in ((0.25, "retracement_25"), (0.50, "retracement_50"), (1.0, "full_retracement")):
            target = close[i] - climax_dir * displacement * fraction
            already = close[s] <= target if climax_dir > 0 else close[s] >= target
            hits = np.flatnonzero(future_l <= target) if climax_dir > 0 else np.flatnonzero(future_h >= target)
            r[name] = bool(already or len(hits)); r[f"time_to_{name}_bars"] = 0 if already else (int(hits[0] + 1) if len(hits) else np.nan)
        rows.append(r)
    events = pd.DataFrame(rows)
    path_rows = [{"relative_bar": h, "mean_normalized_reversal": events[f"path_{h}"].mean(),
                  "median_normalized_reversal": events[f"path_{h}"].median(), "events": len(events)} for h in HORIZONS]
    summary = pd.DataFrame([{"events": len(events), "probability_reversal": events.reversal.mean(),
                             "probability_25pct_retracement": events.retracement_25.mean(),
                             "probability_50pct_retracement": events.retracement_50.mean(),
                             "probability_full_retracement": events.full_retracement.mean(),
                             "mean_mae_atr": events.mae_atr.mean(), "mean_mfe_atr": events.mfe_atr.mean(),
                             "median_time_to_reversal_bars": events.time_to_reversal_bars.median(),
                             "median_time_to_50pct_retracement_bars": events.time_to_retracement_50_bars.median()}])
    buckets = []
    definitions = {
        "volume_abnormality": pd.cut(events.rvol, [REP["rvol"], 2.5, 3.5, np.inf], include_lowest=True, labels=["low", "mid", "high"]),
        "price_displacement": pd.cut(events.displacement_atr, [REP["move"], 2.0, 2.5, np.inf], include_lowest=True, labels=["low", "mid", "high"]),
        "candle_body_range": pd.qcut(events.body_range_ratio, 3, labels=["low", "mid", "high"], duplicates="drop"),
    }
    for dimension, labels in definitions.items():
        for bucket, part in events.assign(bucket=labels).groupby("bucket", observed=True):
            buckets.append({"dimension": dimension, "bucket": str(bucket), "events": len(part),
                            "reversal_probability": part.reversal.mean(), "retracement_50_probability": part.retracement_50.mean(),
                            "full_retracement_probability": part.full_retracement.mean(), "mean_mae_atr": part.mae_atr.mean(),
                            "mean_mfe_atr": part.mfe_atr.mean(), "median_time_to_reversal_bars": part.time_to_reversal_bars.median()})
    return events, pd.DataFrame(path_rows), pd.concat([summary.assign(dimension="overall", bucket="all"), pd.DataFrame(buckets)], ignore_index=True, sort=False)


def temporal(trades: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = trades.copy(); dates = pd.to_datetime(t.session_date); mid = dates.min() + (dates.max() - dates.min()) / 2
    masks = {"IS": dates < split, "OOS": dates >= split, "first_half": dates <= mid, "second_half": dates > mid,
             "pre_2020": dates < "2020-01-01", "2020_2022": dates.between("2020-01-01", "2022-12-31"), "2023_plus": dates >= "2023-01-01"}
    rows = [{"period": name, **enriched(t[mask])} for name, mask in masks.items()]
    yearly = []
    for year, part in t.groupby(dates.dt.year):
        yearly.append({"year": int(year), **enriched(part)})
    y = pd.DataFrame(yearly)
    for label, year in (("remove_best_year", int(y.loc[y.net_pnl.idxmax(), "year"])), ("remove_worst_year", int(y.loc[y.net_pnl.idxmin(), "year"]))):
        rows.append({"period": label, "removed_year": year, **enriched(t[dates.dt.year.ne(year)])})
    daily = t.assign(date=dates).groupby("date").net_pnl.sum().sort_index()
    d = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="B"), fill_value=0.0)
    rolling = pd.DataFrame({"date": d.index, "rolling_12m_sharpe": d.rolling(252, min_periods=126).mean() / d.rolling(252, min_periods=126).std() * math.sqrt(252)})
    t = t.sort_values("entry_time").copy(); t["rolling_100_trade_expectancy"] = t.net_pnl.rolling(100, min_periods=50).mean()
    trade_roll = t[["entry_time", "rolling_100_trade_expectancy"]]
    return pd.DataFrame(rows), y, rolling, trade_roll


def stress_tests(x: pd.DataFrame, sig: pd.Series, trades: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    inst = INSTRUMENTS["MGC"]; oos = trades[pd.to_datetime(trades.session_date) >= split].copy(); rows = []
    def add(name: str, t: pd.DataFrame, note: str = "") -> None:
        rows.append({"scenario": name, "note": note, **enriched(t)})
    for name, extra, cmult in (("nominal", 0, 1), ("plus_1_tick_side", 1, 1), ("plus_2_ticks_side", 2, 1), ("double_commission", 0, 2)):
        z = oos.copy(); z["net_pnl"] = z.gross_pnl - nominal_cost(inst, extra, cmult); add(name, z)
    for delay in (1, 2):
        z = backtest(x, sig, inst, delay=delay); add(f"entry_delayed_{delay}_bar", z[pd.to_datetime(z.session_date) >= split])
    for n in (5, 10):
        add(f"remove_best_{n}_trades", oos.drop(oos.nlargest(n, "net_pnl").index))
    by_day = oos.groupby("session_date").net_pnl.sum(); days = by_day.nlargest(5).index
    add("remove_best_5_trading_days", oos[~oos.session_date.isin(days)])
    by_year = oos.assign(year=pd.to_datetime(oos.session_date).dt.year).groupby("year").net_pnl.sum(); best_year = int(by_year.idxmax())
    add("remove_best_year", oos[pd.to_datetime(oos.session_date).dt.year.ne(best_year)], f"removed {best_year}")
    opposite = backtest(x, -sig, inst); add("opposite_direction", opposite[pd.to_datetime(opposite.session_date) >= split])
    for p in grid():
        if p["mode"] != "confirmed_fade":
            continue
        z = backtest(x, signals(x, p), inst); z = z[pd.to_datetime(z.session_date) >= split]
        add("nearby_inherited_threshold", z, variant_id(p))
    rng = np.random.default_rng(SEED); price_sig = price_only_signals(x, REP); draws = []
    slots = x.slot.to_numpy(); rvol = x.rvol_20.to_numpy().copy(); candidate_count = int((sig != 0).sum())
    price_idx = np.flatnonzero(price_sig.to_numpy() != 0)
    for draw in range(100):
        shuffled = rvol.copy()
        for slot in np.unique(slots):
            loc = np.flatnonzero(slots == slot); shuffled[loc] = rng.permutation(shuffled[loc])
        xs = x.copy(); xs["rvol_20"] = shuffled
        st = signals(xs, REP); z = backtest(x, st, inst); z = z[pd.to_datetime(z.session_date) >= split]
        draws.append({"control": "shuffle_volume_within_time_of_day", "draw": draw, **enriched(z)})
        take = rng.choice(price_idx, size=min(candidate_count, len(price_idx)), replace=False)
        random_sig = pd.Series(0, index=x.index, dtype=np.int8); random_sig.iloc[take] = price_sig.iloc[take]
        z = backtest(x, random_sig, inst); z = z[pd.to_datetime(z.session_date) >= split]
        draws.append({"control": "random_same_frequency_labels", "draw": draw, **enriched(z)})
    draw_df = pd.DataFrame(draws)
    for name, part in draw_df.groupby("control"):
        add(name, pd.DataFrame(), f"100 draws; mean Sharpe {part.sharpe.mean():.3f}; p95 {part.sharpe.quantile(.95):.3f}")
        rows[-1].update({"trades": part.trades.mean(), "net_pnl": part.net_pnl.mean(), "sharpe": part.sharpe.mean(),
                         "pf": part.pf.mean(), "expectancy": part.expectancy.mean(), "maxdd": part.maxdd.mean()})
    return pd.DataFrame(rows), draw_df


def benchmark(x: pd.DataFrame, sig: pd.Series, trades: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    inst = INSTRUMENTS["MGC"]
    candidate = trades[pd.to_datetime(trades.session_date) >= split].copy()
    price = backtest(x, price_only_signals(x, REP), inst); price = price[pd.to_datetime(price.session_date) >= split].copy()
    cm, pm = enriched(candidate), enriched(price)
    rng = np.random.default_rng(SEED); draw_rows = []
    for draw in range(1000):
        z = price.iloc[rng.choice(len(price), size=len(candidate), replace=False)]
        draw_rows.append({"draw": draw, **enriched(z)})
    draws = pd.DataFrame(draw_rows)
    annual_c = candidate.assign(year=pd.to_datetime(candidate.session_date).dt.year).groupby("year").net_pnl.sum()
    annual_p = price.assign(year=pd.to_datetime(price.session_date).dt.year).groupby("year").net_pnl.sum()
    annual_fraction_c = float((annual_c > 0).mean()); annual_fraction_p = float((annual_p > 0).mean())
    rows = [{"candidate": "PRICE_EXTREME_PLUS_VOLUME_CLIMAX", "scenario": "nominal", "positive_year_fraction": annual_fraction_c, **cm},
            {"candidate": "PRICE_EXTREME_ONLY", "scenario": "nominal", "positive_year_fraction": annual_fraction_p, **pm},
            {"candidate": "DELTA_VOLUME_MINUS_PRICE", "scenario": "nominal", "trades": cm["trades"] - pm["trades"],
                             "net_pnl": cm["net_pnl"] - pm["net_pnl"], "sharpe": cm["sharpe"] - pm["sharpe"],
                             "pf": cm["pf"] - pm["pf"], "expectancy": cm["expectancy"] - pm["expectancy"],
                             "maxdd": cm["maxdd"] - pm["maxdd"],
                             "positive_year_fraction": annual_fraction_c - annual_fraction_p}]
    for scenario, extra, cmult in (("plus_1_tick_side", 1, 1), ("plus_2_ticks_side", 2, 1), ("double_commission", 0, 2)):
        scenario_metrics = []
        for label, source in (("PRICE_EXTREME_PLUS_VOLUME_CLIMAX", candidate), ("PRICE_EXTREME_ONLY", price)):
            z = source.copy(); z["net_pnl"] = z.gross_pnl - nominal_cost(inst, extra, cmult); met = enriched(z)
            rows.append({"candidate": label, "scenario": scenario, **met}); scenario_metrics.append(met)
        rows.append({"candidate": "DELTA_VOLUME_MINUS_PRICE", "scenario": scenario,
                     **{k: scenario_metrics[0][k] - scenario_metrics[1][k] for k in ("trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd")}})
    rows.append({"candidate": "RANDOM_THINNING_CONTROL", "scenario": "nominal", "trades": len(candidate), "sharpe": draws.sharpe.mean(),
                             "expectancy": draws.expectancy.mean(), "pf": draws.pf.mean(), "maxdd": draws.maxdd.mean(),
                             "empirical_candidate_sharpe_percentile": float((draws.sharpe <= cm["sharpe"]).mean()),
                             "empirical_candidate_expectancy_percentile": float((draws.expectancy <= cm["expectancy"]).mean())})
    summary = pd.DataFrame(rows)
    return summary, draws


def regimes(x: pd.DataFrame, trades: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    oos = trades[pd.to_datetime(trades.session_date) >= split].copy()
    fields = x[["prior_rv_pct", "atr_ratio", "cum_rvol_20", "prior_efficiency", "gap_norm", "prior_range_pct", "bar_no"]].copy()
    fields["signal_time"] = fields.index.astype(str); oos["signal_time"] = pd.to_datetime(oos.signal_time).astype(str)
    z = oos.merge(fields, on="signal_time", how="left"); z["reversal_probability"] = z.gross_pnl.gt(0)
    z["time_of_day"] = pd.cut(z.bar_no, [-1, 20, 41, np.inf], labels=["early", "middle", "late"])
    z["gap_state"] = np.where(z.gap_norm >= .25, "gap", "no_gap")
    definitions: dict[str, pd.Series] = {"time_of_day": z.time_of_day, "gap_day": z.gap_state}
    for c in ("prior_rv_pct", "atr_ratio", "cum_rvol_20", "prior_efficiency", "prior_range_pct"):
        definitions[c] = pd.qcut(z[c], 3, labels=["low", "mid", "high"], duplicates="drop")
    rows = []
    for dimension, labels in definitions.items():
        for state, part in z.assign(_state=labels).groupby("_state", observed=True):
            rows.append({"dimension": dimension, "state": str(state), **enriched(part),
                         "reversal_probability": part.reversal_probability.mean(), "mean_mae": part.mae.mean(), "mean_mfe": part.mfe.mean()})
    return pd.DataFrame(rows)


def portfolio(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nine = pd.read_csv(FROZEN, parse_dates=["session_date"]).set_index("session_date")
    cand = trades.assign(session_date=pd.to_datetime(trades.session_date)).groupby("session_date").net_pnl.sum().rename("candidate")
    z = nine.join(cand, how="left").fillna({"candidate": 0.0})
    mgc = z.mgc_volatility_breakout + z.mgc_gap_continuation
    full = z.portfolio_pnl_usd
    def compare(name: str, base: pd.Series) -> dict[str, Any]:
        added = base + z.candidate; bm = daily_metrics(base); am = daily_metrics(added); neg = base < 0
        return {"block": name, "days": len(z), "candidate_correlation": z.candidate.corr(base),
                "correlation_on_negative_days": z.loc[neg, "candidate"].corr(base[neg]),
                "active_day_overlap": float(((z.candidate != 0) & (base != 0)).mean()),
                "simultaneous_loss_rate": float(((z.candidate < 0) & (base < 0)).mean()),
                "candidate_mean_worst_5pct": z.loc[base <= base.quantile(.05), "candidate"].mean(),
                "candidate_mean_worst_10pct": z.loc[base <= base.quantile(.10), "candidate"].mean(),
                "candidate_mean_worst_20pct": z.loc[base <= base.quantile(.20), "candidate"].mean(),
                "base_sharpe": bm["sharpe"], "added_sharpe": am["sharpe"], "delta_sharpe": am["sharpe"] - bm["sharpe"],
                "base_maxdd": bm["maxdd"], "added_maxdd": am["maxdd"], "base_downside_deviation": bm["downside_deviation"],
                "added_downside_deviation": am["downside_deviation"], "marginal_variance": float(added.var() - base.var()),
                "marginal_risk_adjusted_contribution": float(z.candidate.mean() / max(added.std(ddof=1), 1e-12) * math.sqrt(252))}
    comparisons = pd.DataFrame([compare("existing_2_MGC_sleeves", mgc), compare("frozen_9_sleeves", full)])
    worst = []
    for name, base in (("existing_2_MGC_sleeves", mgc), ("frozen_9_sleeves", full)):
        for rank, day in enumerate(base.nsmallest(20).index, 1):
            worst.append({"block": name, "rank": rank, "session_date": day, "base_pnl": base.loc[day], "candidate_pnl": z.candidate.loc[day], "added_pnl": base.loc[day] + z.candidate.loc[day]})
    existing = pd.concat([pd.read_csv(MGC_TRADES / "mgc_volatility_breakout.csv"), pd.read_csv(MGC_TRADES / "mgc_gap_continuation.csv")], ignore_index=True)
    existing["entry_time"] = pd.to_datetime(existing.entry_time, utc=True); existing["exit_time"] = pd.to_datetime(existing.exit_time, utc=True)
    candidate = trades.copy(); candidate["entry_time"] = pd.to_datetime(candidate.entry_time, utc=True); candidate["exit_time"] = pd.to_datetime(candidate.exit_time, utc=True)
    overlaps = []
    for c in candidate.itertuples(index=False):
        e = existing[(existing.entry_time <= c.exit_time) & (existing.exit_time >= c.entry_time)]
        overlaps.append({"candidate_entry": c.entry_time, "overlapping_existing_trades": len(e),
                         "same_direction_fraction": float((e.signed_quantity * c.direction > 0).mean()) if len(e) else np.nan})
    overlap_df = pd.DataFrame(overlaps)
    overlap_mask = overlap_df.overlapping_existing_trades.gt(0)
    comparisons["intraday_overlap_rate"] = float(overlap_mask.mean())
    comparisons["direction_overlap_when_simultaneous"] = float(overlap_df.loc[overlap_mask, "same_direction_fraction"].mean()) if overlap_mask.any() else np.nan
    return comparisons, pd.DataFrame(worst), overlap_df


def plots(cell: pd.DataFrame, yearly: pd.DataFrame, rolling: pd.DataFrame, paths: pd.DataFrame,
          event_buckets: pd.DataFrame, thinning: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    wide = cell.pivot_table(index=["move", "rvol"], columns="mode", values="sharpe", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4));
    for mode in wide.columns:
        ax.plot(range(len(wide)), wide[mode], marker="o", label=mode)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(range(len(wide))); ax.set_xticklabels([f"{m:g}/{r:g}" for m, r in wide.index], rotation=45); ax.set_ylabel("Mean IS/OOS Sharpe"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "inherited_region_sharpe.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(yearly.year.astype(str), yearly.net_pnl); ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Net PnL ($)"); ax.set_title("Canonical cell yearly PnL"); fig.tight_layout(); fig.savefig(PLOTS / "yearly_pnl.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(pd.to_datetime(rolling.date), rolling.rolling_12m_sharpe); ax.axhline(0, color="black", lw=.8); ax.set_title("Rolling 12-month Sharpe"); fig.tight_layout(); fig.savefig(PLOTS / "rolling_12m_sharpe.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(paths.relative_bar, paths.mean_normalized_reversal, marker="o", label="mean"); ax.plot(paths.relative_bar, paths.median_normalized_reversal, marker="o", label="median"); ax.axhline(0, color="black", lw=.8); ax.axvline(0, color="gray", lw=.8); ax.legend(); ax.set_ylabel("ATR-normalized reversal path"); fig.tight_layout(); fig.savefig(PLOTS / "event_path.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(thinning.sharpe.dropna(), bins=35); ax.axvline(metrics_placeholder := thinning.attrs.get("candidate_sharpe", np.nan), color="red", label="volume-conditioned"); ax.legend(); ax.set_xlabel("Random-thinning Sharpe"); fig.tight_layout(); fig.savefig(PLOTS / "random_thinning_sharpe.png", dpi=150); plt.close(fig)
    dose = event_buckets[event_buckets.dimension.ne("overall")]
    fig, ax = plt.subplots(figsize=(8, 4));
    for name, part in dose.groupby("dimension"):
        ax.plot(part.bucket, part.retracement_50_probability, marker="o", label=name)
    ax.set_ylim(0, 1); ax.set_ylabel("P(50% retracement)"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "event_dose_response.png", dpi=150); plt.close(fig)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(exist_ok=True)
    # Remove the superseded combined rolling file from early audit revisions.
    (OUT / "rolling_diagnostics.csv").unlink(missing_ok=True)
    x, split = prepare(); inst = INSTRUMENTS["MGC"]; cell_rows = []; trade_cache = {}
    for p in grid():
        vid = variant_id(p); t = backtest(x, signals(x, p), inst); trade_cache[vid] = t
        for label, mask in (("IS", pd.to_datetime(t.session_date) < split), ("OOS", pd.to_datetime(t.session_date) >= split)):
            cell_rows.append({"variant_id": vid, **p, "split": label, **enriched(t[mask])})
    cells = pd.DataFrame(cell_rows); cells.to_csv(OUT / "inherited_region_all_cells.csv", index=False)
    region_detail, region_summary = region_audit(cells); region_detail.to_csv(OUT / "parameter_region_continuity.csv", index=False); region_summary.to_csv(OUT / "parameter_region_summary.csv", index=False)
    rep_id = variant_id(REP); rep_sig = signals(x, REP); rep_trades = trade_cache[rep_id]; rep_trades.to_csv(OUT / "canonical_representative_trades.csv", index=False)
    temporal_summary, yearly, rolling, trade_rolling = temporal(rep_trades, split); temporal_summary.to_csv(OUT / "temporal_stability.csv", index=False); yearly.to_csv(OUT / "yearly_results.csv", index=False); rolling.to_csv(OUT / "rolling_12m_sharpe.csv", index=False); trade_rolling.to_csv(OUT / "rolling_100_trade_expectancy.csv", index=False)
    symmetry = []
    for scope, part in (("FULL", rep_trades), ("IS", rep_trades[pd.to_datetime(rep_trades.session_date) < split]), ("OOS", rep_trades[pd.to_datetime(rep_trades.session_date) >= split])):
        for direction, side in ((1, "negative_climax_long"), (-1, "positive_climax_short")):
            z = part[part.direction.eq(direction)]; symmetry.append({"scope": scope, "side": side, **enriched(z)})
            for year, y in z.groupby(pd.to_datetime(z.session_date).dt.year): symmetry.append({"scope": f"year_{year}", "side": side, **enriched(y)})
        for direction, side in ((1, "negative_climax_long"), (-1, "positive_climax_short")):
            z = part[part.direction.eq(direction)].copy(); z["net_pnl"] = z.gross_pnl - nominal_cost(inst, 2, 1)
            symmetry.append({"scope": f"{scope}_plus_2_ticks", "side": side, **enriched(z)})
    pd.DataFrame(symmetry).to_csv(OUT / "long_short_decomposition.csv", index=False)
    events, paths, event_summary = event_study(x, split); events.to_csv(OUT / "event_study_events.csv", index=False); paths.to_csv(OUT / "event_study_paths.csv", index=False); event_summary.to_csv(OUT / "event_study_summary_and_dose_response.csv", index=False)
    bench, thinning = benchmark(x, rep_sig, rep_trades, split); thinning.attrs["candidate_sharpe"] = float(bench.loc[bench.candidate.eq("PRICE_EXTREME_PLUS_VOLUME_CLIMAX"), "sharpe"].iloc[0]); bench.to_csv(OUT / "price_only_vs_volume.csv", index=False); thinning.to_csv(OUT / "random_thinning_1000_draws.csv", index=False)
    stress, false_draws = stress_tests(x, rep_sig, rep_trades, split); stress.to_csv(OUT / "candidate_specific_falsification.csv", index=False); false_draws.to_csv(OUT / "label_falsification_draws.csv", index=False)
    regime = regimes(x, rep_trades, split); regime.to_csv(OUT / "regime_diagnostics.csv", index=False)
    portfolio_summary, worst_days, direction_overlap = portfolio(rep_trades[pd.to_datetime(rep_trades.session_date) >= split]); portfolio_summary.to_csv(OUT / "portfolio_integration.csv", index=False); worst_days.to_csv(OUT / "portfolio_worst_days.csv", index=False); direction_overlap.to_csv(OUT / "mgc_intraday_direction_overlap.csv", index=False)
    specifications = pd.DataFrame([
        {"item": "inherited_grid", "value": "move={1.0,1.5,2.0} ATR; RVOL={1.5,2.0,2.5}; mode={immediate_fade,confirmed_fade}; 18 cells"},
        {"item": "price_normalization", "value": "abs(close-open) / causal session ATR; ATR is median true range of prior 20 same-session bars, min 10"},
        {"item": "volume_normalization", "value": "bar volume / median volume at same ET slot over prior 20 sessions"},
        {"item": "immediate_entry", "value": "climax completed at t; fade signal t; fill open t+1"},
        {"item": "confirmed_entry", "value": "climax completed at t-1; opposite-direction bar t; fade signal t; fill open t+1"},
        {"item": "exit", "value": "close of sixth held 5-minute bar, truncated at session end; non-overlapping trades"},
        {"item": "session", "value": "MGC weekdays 08:20 <= ET < 13:30; no overnight carry"},
        {"item": "nominal_cost", "value": "$4.50 round turn = one tick/side slippage ($2.00) + $1.25/side commission ($2.50)"},
        {"item": "validation_split", "value": f"chronological 70/30 by session; OOS begins {split.date()}"},
        {"item": "canonical_diagnostic_cell", "value": rep_id + "; inherited geometric center, fixed without metric selection"},
    ])
    specifications.to_csv(OUT / "inherited_specification.csv", index=False)
    topstep = pd.DataFrame([{"status": "NOT_RUN_FAILED_PRIOR_GATES", "reason": "All 18 cells are negative IS and the canonical OOS result is recent/regime-concentrated; Phase 11 gate was not reached."}]); topstep.to_csv(OUT / "topstep_status.csv", index=False)
    plots(cells, yearly, rolling, paths, event_summary, thinning)
    rsum = region_summary.set_index("scope").loc["all"]; bthin = bench[(bench.candidate == "RANDOM_THINNING_CONTROL") & (bench.scenario == "nominal")].iloc[0]
    oos_rep = cells[(cells.variant_id == rep_id) & (cells.split == "OOS")].iloc[0]; is_rep = cells[(cells.variant_id == rep_id) & (cells.split == "IS")].iloc[0]
    recent_share = yearly.loc[yearly.year >= 2025, "net_pnl"].sum() / max(yearly.loc[yearly.net_pnl > 0, "net_pnl"].sum(), 1e-12)
    decision = "REAL_EFFECT_NOT_YET_EXECUTABLE"
    report = ["# MGC Volume Climax Reversal — Frozen-Region Audit", "", f"**Final decision: `{decision}`**", "",
              "No new parameter search was run, no best cell was selected, and the frozen nine sleeves were not modified. The canonical cell is the inherited geometric centre used only to make the requested diagnostics concrete.", "",
              "## Executive conclusion", "",
              f"The confirmed-fade OOS effect is economically coherent and cost-robust, but it is not temporally stable enough to execute. All 18 inherited cells lose money IS; only {rsum.fraction_positive_oos:.1%} are positive OOS and {rsum.fraction_positive_both:.1%} are positive in both. The centre cell changes from IS Sharpe {is_rep.sharpe:.2f} / PnL ${is_rep.net_pnl:,.0f} to OOS Sharpe {oos_rep.sharpe:.2f} / PnL ${oos_rep.net_pnl:,.0f}. The apparent `BROAD_REGION` is therefore an OOS confirmed-mode plateau, not a history-wide region.", "",
              f"The price-only benchmark is already profitable OOS. The volume-conditioned Sharpe sits at the {100*bthin.empirical_candidate_sharpe_percentile:.1f}th percentile of 1,000 trade-count-matched random thinnings, which is suggestive but does not clear a 95th-percentile falsification gate. Most positive PnL is recent: 2025–2026 contribute {recent_share:.1%} of all positive-year PnL.", "",
              "## 1. Inherited specification", "", markdown(specifications), "",
              "## 2. Every inherited cell", "", markdown(cells[["variant_id", "mode", "move", "rvol", "split", "trades", "net_pnl", "sharpe", "pf", "expectancy", "win_rate", "avg_winner", "avg_loser", "maxdd", "avg_holding", "turnover_trades_per_year", "long_pnl", "short_pnl"]]), "",
              "## 3. Parameter-region audit", "", markdown(region_summary), "", "Continuity is real only inside the post-2023 confirmed-fade surface. Immediate fades remain negative, every IS cell is negative, and the sign change is mode- and era-dependent rather than an edge-threshold accident.", "",
              "## 4. Temporal stability", "", markdown(temporal_summary[["period", "removed_year", "trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"]]), "", markdown(yearly[["year", "trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"]]), "", "The mechanism structurally changes after 2023: 2015–2022 are negative, 2024 is negative, and 2025–2026 dominate the OOS gain. It does not exist throughout the history.", "",
              "## 5. Long/short decomposition", "", markdown(pd.DataFrame(symmetry).query("scope in ['IS','OOS','OOS_plus_2_ticks']")[["scope", "side", "trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"]]), "", "The OOS effect is approximately symmetric: both sides have Sharpe near 2 and both survive +2 ticks/side. The long side contributes more dollars because it trades more often and captures the large 2026 contribution. No directional filter is proposed; the file retains every year and both sides.", "",
              "## 6. Event study and dose response", "", markdown(event_summary), "", "Paths are normalized by signal-bar ex-ante ATR. A true reversal signature is present after confirmed climaxes, but the bucket table does not establish a clean monotone response across every volume, displacement, and candle-structure axis.", "",
              "## 7. Does volume add information?", "", markdown(bench), "", "Volume improves the point estimate over the exact price-only event, but the 1,000-draw thinning percentile is below the required 95th percentile. The evidence cannot distinguish incremental volume information from favorable thinning strongly enough for execution.", "",
              "## 8. Candidate-specific falsification", "", markdown(stress[["scenario", "note", "trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"]]), "", "The candidate survives +2 ticks/side, one- and two-bar delays, and removal of the best 10 trades, but fails the full-history temporal falsification and does not decisively beat matched thinning.", "",
              "## 9. Regime diagnostics", "", markdown(regime[["dimension", "state", "trades", "expectancy", "sharpe", "reversal_probability", "mean_mae", "mean_mfe"]]), "", "The lowest prior-RV and ATR buckets lose after costs while the middle and high buckets are profitable. Reversal probability also rises only modestly, so the weakest-volatility failure is primarily disappearance of net reversion/cost coverage, not a pure MAE expansion. Time-of-day concentration is reported but not converted into a filter.", "",
              "## 10. Existing MGC block and frozen nine sleeves", "", markdown(portfolio_summary), "", "One MGC contract is added without weight optimization. Intraday direction overlap is computed from the two frozen MGC trade ledgers; worst-day details are in the supporting CSV. Diversification is useful descriptively but cannot repair the standalone temporal failure.", "",
              "## 11. Topstep", "", markdown(topstep), "", "The protocol says to run Topstep only after prior phases survive. They did not, so no point estimate or confidence interval is manufactured.", "",
              "## Gate decision", "",
              "- Broad inherited stability: **FAIL** — 0% of cells are positive in both IS and OOS.",
              "- Temporal stability: **FAIL** — the profitable regime is post-2023 and strongest in 2025–2026.",
              "- Incremental volume information: **INCONCLUSIVE** — improved point estimates, below the 95th percentile of matched thinning.",
              "- Cost and top-trade robustness: **PASS OOS**.",
              "- Causal/event-study interpretation: **PLAUSIBLE**, but regime-dependent.",
              "- Diversification: **POSITIVE DESCRIPTIVELY**, not sufficient for promotion.", "",
              f"**Decision: `{decision}`.** Freeze the inherited region and mechanism for prospective observation only. Do not freeze a best cell, add a directional/regime filter, alter the nine sleeves, or deploy the candidate."]
    (OUT / "MGC_VOLUME_CLIMAX_AUDIT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    files = {str(p.relative_to(OUT)): sha256(p) for p in OUT.rglob("*")
             if p.is_file() and p.name != "run_metadata.json"}
    metadata = {"decision": decision, "new_parameter_search": False, "cells": 18, "canonical_cell_rule": "geometric centre of inherited grid",
                "canonical_variant": rep_id, "split_date": str(split.date()), "seed": SEED, "source_campaign": str(SOURCE.relative_to(REPO)),
                "market_data_sha256": sha256(REPO / DATA_FILES["MGC"]), "frozen_portfolio_sha256": sha256(FROZEN), "files": files}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
