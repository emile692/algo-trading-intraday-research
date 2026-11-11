"""Audit the exact inherited price-move-without-volume-confirmation region.

This module does not search or optimize.  It replays the six inherited
representatives and diagnoses the complete pre-existing 3 x 2 x 2 region.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.mean_reversion_cross_asset_20261006.run_topstep_incremental import (
    build_baseline,
    candidate_intraday,
)
from research.mbt_nine_sleeve_topstep_20261004.run_topstep import topstep_campaign
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
SOURCE = REPO / "research" / "volume_auction_landscape_20261007" / "results"
FROZEN_DAILY = REPO / "research" / "locked_nine_sleeve_portfolio" / "daily_pnl.csv"
ASSETS = ("M2K", "MES", "MNQ", "NQ", "MGC", "MBT")
SEED = 20261007
THINNING_DRAWS = 1000

REPRESENTATIVES = {
    "M2K": {"mechanism": "E", "mode": "fade", "lookback": 1, "move": 1.5, "rvol": 1.0, "profile": 0, "confirm": 1},
    "MES": {"mechanism": "E", "mode": "fade", "lookback": 1, "move": 1.5, "rvol": 0.75, "profile": 0, "confirm": 1},
    "MNQ": {"mechanism": "E", "mode": "continuation_control", "lookback": 1, "move": 1.5, "rvol": 1.0, "profile": 0, "confirm": 1},
    "NQ": {"mechanism": "E", "mode": "fade", "lookback": 1, "move": 1.5, "rvol": 1.0, "profile": 0, "confirm": 1},
    "MGC": {"mechanism": "E", "mode": "continuation_control", "lookback": 1, "move": 1.5, "rvol": 1.0, "profile": 0, "confirm": 1},
    "MBT": {"mechanism": "E", "mode": "continuation_control", "lookback": 1, "move": 1.5, "rvol": 1.0, "profile": 0, "confirm": 1},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    x = df.copy()
    cols = [str(c) for c in x.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in x.itertuples(index=False, name=None):
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.4g}")
            else:
                vals.append(str(v).replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def prepare(asset: str) -> pd.DataFrame:
    path = REPO / DATA_FILES[asset]
    x = load_bars(path, INSTRUMENTS[asset])
    x = add_price_features(add_causal_volume_baselines(x))
    # Diagnostic percentile of the current bar versus the prior 20 completed
    # sessions at the same clock slot.  The current observation is compared
    # with, but never used to construct, its reference sample.
    x["volume_percentile_20"] = x.groupby("slot", sort=False).volume.transform(
        lambda s: s.rolling(21, min_periods=21).apply(lambda a: float(np.mean(a[:-1] <= a[-1])), raw=True)
    )
    return x


def trade_metrics(t: pd.DataFrame, pnl_col: str = "net_pnl") -> dict[str, float]:
    m = metrics(t, pnl_col)
    return {"trades": m["trades"], "net_pnl": m["net_pnl"], "sharpe": m["sharpe"], "pf": m["pf"],
            "expectancy": m["expectancy"], "win_rate": m["win_rate"], "maxdd": m["maxdd"]}


def apply_cost(t: pd.DataFrame, asset: str, extra_ticks: int = 0, commission_mult: float = 1.0) -> pd.DataFrame:
    z = t.copy()
    z["net_pnl"] = z.gross_pnl - nominal_cost(INSTRUMENTS[asset], extra_ticks, commission_mult)
    return z


def event_rows(x: pd.DataFrame, asset: str, p: dict[str, Any], split: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per OOS bar event and its 2x2 aggregate."""
    inst = INSTRUMENTS[asset]
    move, rv = float(p["move"]), float(p["rvol"])
    dates = x.session_date.to_numpy(); close = x.close.to_numpy(); high = x.high.to_numpy(); low = x.low.to_numpy()
    directions = np.sign(x.close - x.open).to_numpy(dtype=float)
    end_pos = x["_session_last_pos"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    valid = x.atr.notna() & x.rvol_20.notna() & (~x.last_bar) & x.session_date.ge(split)
    for i in np.flatnonzero(valid.to_numpy()):
        d = int(directions[i])
        if d == 0:
            continue
        large = bool(x.abs_move.iloc[i] >= move); high_volume = bool(x.rvol_20.iloc[i] > rv)
        group = ("A_large_high" if large and high_volume else "B_large_low_normal" if large else
                 "C_small_high" if high_volume else "D_small_low_normal")
        j = min(i + 6, int(end_pos[i]))
        if j <= i:
            continue
        path_h, path_l = high[i + 1:j + 1], low[i + 1:j + 1]
        base = float(close[i]); fwd = d * (float(close[j]) - base) * inst.point_value
        mfe = ((path_h.max() - base) if d > 0 else (base - path_l.min())) * inst.point_value
        mae = ((base - path_l.min()) if d > 0 else (path_h.max() - base)) * inst.point_value
        rets = np.diff(np.r_[base, close[i + 1:j + 1]]) / base
        rows.append({"asset": asset, "session_date": dates[i], "signal_time": x.index[i], "direction": d,
                     "price_side": "up" if d > 0 else "down", "group": group, "large_move": large,
                     "high_abnormal_volume": high_volume, "abs_move_atr": float(x.abs_move.iloc[i]),
                     "rvol": float(x.rvol_20.iloc[i]), "expected_volume": float(x.expected_median_20.iloc[i]),
                     "actual_volume": float(x.volume.iloc[i]), "volume_percentile_20": float(x.volume_percentile_20.iloc[i]), "forward_return_usd": fwd,
                     "reversal": fwd < 0, "continuation": fwd > 0, "mae_usd": float(mae), "mfe_usd": float(mfe),
                     "forward_realized_vol": float(np.sqrt(np.sum(rets ** 2))) if len(rets) else np.nan})
    events = pd.DataFrame(rows)
    agg = []
    for group, g in events.groupby("group", sort=True):
        agg.append({"record_type": "price_volume_2x2", "asset": asset, "scope": "OOS", "group": group,
                    "events": len(g), "forward_return_usd": g.forward_return_usd.mean(),
                    "reversal_probability": g.reversal.mean(), "continuation_probability": g.continuation.mean(),
                    "mae_usd": g.mae_usd.mean(), "mfe_usd": g.mfe_usd.mean(),
                    "forward_realized_vol": g.forward_realized_vol.mean()})
    return events, pd.DataFrame(agg)


def thinning(asset: str, observed: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    n = len(observed)
    if n == 0 or len(benchmark) < n:
        return {"asset": asset, "draws": 0, "observed_trades": n, "benchmark_trades": len(benchmark)}
    rng = np.random.default_rng(SEED + sum(map(ord, asset)))
    sh, ex, pf = np.empty(THINNING_DRAWS), np.empty(THINNING_DRAWS), np.empty(THINNING_DRAWS)
    for k in range(THINNING_DRAWS):
        sample = benchmark.iloc[rng.choice(len(benchmark), n, replace=False)]
        m = trade_metrics(sample)
        sh[k], ex[k], pf[k] = m["sharpe"], m["expectancy"], m["pf"]
    obs = trade_metrics(observed)
    percentile = lambda a, v: float(100 * np.mean(a <= v))
    exceed = lambda a, v: float(np.mean(a >= v))
    return {"asset": asset, "draws": THINNING_DRAWS, "observed_trades": n, "benchmark_trades": len(benchmark),
            "observed_sharpe": obs["sharpe"], "sharpe_percentile": percentile(sh, obs["sharpe"]),
            "prob_random_matches_exceeds_sharpe": exceed(sh, obs["sharpe"]),
            "observed_expectancy": obs["expectancy"], "expectancy_percentile": percentile(ex, obs["expectancy"]),
            "prob_random_matches_exceeds_expectancy": exceed(ex, obs["expectancy"]),
            "observed_pf": obs["pf"], "pf_percentile": percentile(pf, obs["pf"]),
            "prob_random_matches_exceeds_pf": exceed(pf, obs["pf"]),
            "random_sharpe_mean": float(np.nanmean(sh)), "random_sharpe_p95": float(np.nanpercentile(sh, 95)),
            "random_expectancy_mean": float(np.nanmean(ex)), "random_expectancy_p95": float(np.nanpercentile(ex, 95)),
            "random_pf_mean": float(np.nanmean(pf)), "random_pf_p95": float(np.nanpercentile(pf, 95))}


def rolling_sharpe(t: pd.DataFrame, asset: str) -> pd.DataFrame:
    d = t.assign(session_date=pd.to_datetime(t.session_date)).groupby("session_date").net_pnl.sum()
    idx = pd.date_range(d.index.min(), d.index.max(), freq="B")
    d = d.reindex(idx, fill_value=0.0)
    mean = d.rolling(252, min_periods=126).mean(); sd = d.rolling(252, min_periods=126).std(ddof=1)
    return pd.DataFrame({"record_type": "rolling_12m", "asset": asset, "period": idx,
                         "sharpe": mean / sd * math.sqrt(252)})


def temporal(asset: str, t: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    dt = pd.to_datetime(t.session_date); rows = []
    groups = {"IS": dt < split, "OOS": dt >= split, "pre_2020": dt < "2020-01-01",
              "2020_2022": (dt >= "2020-01-01") & (dt <= "2022-12-31"), "2023_plus": dt >= "2023-01-01"}
    for label, mask in groups.items():
        rows.append({"record_type": "period", "asset": asset, "period": label, **trade_metrics(t[mask])})
    for year, g in t.groupby(dt.dt.year):
        rows.append({"record_type": "year", "asset": asset, "period": int(year), **trade_metrics(g)})
    oos = t[dt >= split]
    for n in (5, 10):
        z = oos.drop(oos.nlargest(min(n, len(oos)), "net_pnl").index)
        rows.append({"record_type": "concentration", "asset": asset, "period": f"remove_best_{n}_trades", **trade_metrics(z)})
    if len(oos):
        yp = oos.groupby(pd.to_datetime(oos.session_date).dt.year).net_pnl.sum(); best = int(yp.idxmax())
        z = oos[pd.to_datetime(oos.session_date).dt.year != best]
        rows.append({"record_type": "concentration", "asset": asset, "period": "remove_best_year", "removed_year": best, **trade_metrics(z)})
    return pd.concat([pd.DataFrame(rows), rolling_sharpe(t, asset)], ignore_index=True, sort=False)


def volume_stability(asset: str, events: pd.DataFrame, x: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    q = events[events.group.eq("B_large_low_normal")].copy()
    q["year"] = pd.to_datetime(q.session_date).dt.year
    eligible = x.loc[x.session_date.ge(split), ["session_date"]].drop_duplicates()
    eligible["year"] = pd.to_datetime(eligible.session_date).dt.year
    sessions_by_year = eligible.groupby("year").size()
    rows = []
    for year, g in q.groupby("year"):
        rows.append({"record_type": "volume_signal_stability", "asset": asset, "year": int(year), "events": len(g),
                     "median_rvol": g.rvol.median(), "median_volume_percentile": g.volume_percentile_20.median(),
                     "median_expected_volume": g.expected_volume.median(), "signal_frequency_per_session": len(g) / max(1, int(sessions_by_year.get(year, 0)))})
    return pd.DataFrame(rows)


def time_direction(asset: str, x: pd.DataFrame, split: pd.Timestamp, p: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inst = INSTRUMENTS[asset]
    q = backtest(x, signals(x, p), inst)
    q = q[q.session_date >= split].copy()
    q["session_bucket"] = pd.cut(q.signal_bar_no * 5, [-1, 120, 240, 10_000], labels=["open_0_2h", "mid_2_4h", "late_4h_plus"])
    rows = []
    for bucket, g in q.groupby("session_bucket", observed=True):
        rows.append({"record_type": "time_of_day", "asset": asset, "bucket": str(bucket), **trade_metrics(g),
                     "reversal_probability": float((g.gross_pnl > 0).mean()) if p["mode"] == "fade" else float((g.gross_pnl < 0).mean())})
    dr = []
    fade = dict(p); fade["mode"] = "fade"; fs = signals(x, fade)
    for signal_direction, label in ((1, "fade_long_after_down"), (-1, "fade_short_after_up")):
        side_signal = fs.where(fs.eq(signal_direction), 0)
        g = backtest(x, side_signal, inst); g = g[g.session_date >= split]
        dr.append({"asset": asset, "direction": label, **trade_metrics(g),
                   "reversal_probability": float((g.gross_pnl > 0).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(dr)


def event_study(asset: str, x: pd.DataFrame, split: pd.Timestamp, p: dict[str, Any]) -> pd.DataFrame:
    move, rv = float(p["move"]), float(p["rvol"]); inst = INSTRUMENTS[asset]; rows = []
    valid = x.atr.notna() & x.rvol_20.notna() & x.session_date.ge(split) & (x.abs_move >= move)
    d = np.sign(x.close - x.open).to_numpy(); c = x.close.to_numpy(); dates = x.session_date.to_numpy()
    for group, mask in (("A_large_high", valid & (x.rvol_20 > rv)), ("B_large_low_normal", valid & (x.rvol_20 <= rv))):
        idx = np.flatnonzero(mask.to_numpy())
        for h in (1, 3, 6, 12):
            vals = [d[i] * (c[i + h] - c[i]) * inst.point_value for i in idx if i + h < len(x) and dates[i + h] == dates[i]]
            rows.append({"asset": asset, "group": group, "horizon_bars": h, "events": len(vals),
                         "mean_continuation_usd": np.mean(vals) if vals else np.nan,
                         "median_continuation_usd": np.median(vals) if vals else np.nan,
                         "continuation_probability": np.mean(np.asarray(vals) > 0) if vals else np.nan,
                         "reversal_probability": np.mean(np.asarray(vals) < 0) if vals else np.nan})
    return pd.DataFrame(rows)


def cross_dependence(events: dict[str, pd.DataFrame], trades: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(ASSETS):
        ea = events[a][events[a].group.eq("B_large_low_normal")].drop_duplicates("session_date")
        for b in ASSETS[i + 1:]:
            eb = events[b][events[b].group.eq("B_large_low_normal")].drop_duplicates("session_date")
            merged = ea[["session_date", "direction"]].merge(eb[["session_date", "direction"]], on="session_date", suffixes=("_a", "_b"))
            da, db = set(ea.session_date), set(eb.session_date); common = da & db; union = da | db
            pa = trades[a].groupby("session_date").net_pnl.sum(); pb = trades[b].groupby("session_date").net_pnl.sum(); cd = pa.index.intersection(pb.index)
            rows.append({"record_type": "pair_overlap", "asset_a": a, "asset_b": b, "common_event_dates": len(common),
                         "event_date_jaccard": len(common) / len(union) if union else np.nan,
                         "overlap_vs_smaller_set": len(common) / min(len(da), len(db)) if min(len(da), len(db)) else np.nan,
                         "direction_overlap": (merged.direction_a == merged.direction_b).mean() if len(merged) else np.nan,
                         "pnl_correlation": pa.loc[cd].corr(pb.loc[cd]) if len(cd) >= 10 else np.nan,
                         "unique_event_count": len(da ^ db), "related_equity_contracts": a in {"M2K", "MES", "MNQ", "NQ"} and b in {"M2K", "MES", "MNQ", "NQ"}})
    equity_sets = [set(events[a].loc[events[a].group.eq("B_large_low_normal"), "session_date"]) for a in ("M2K", "MES", "MNQ", "NQ")]
    rows.append({"record_type": "distinct_event_summary", "asset_a": "US_EQUITY_INDICES", "common_event_dates": len(set.intersection(*equity_sets)),
                 "unique_event_count": len(set.union(*equity_sets)), "asset_level_event_count": sum(map(len, equity_sets)),
                 "effective_distinct_event_ratio": len(set.union(*equity_sets)) / sum(map(len, equity_sets))})
    return pd.DataFrame(rows)


def diversification(trades: dict[str, pd.DataFrame], survivors: list[str]) -> pd.DataFrame:
    frozen = pd.read_csv(FROZEN_DAILY, parse_dates=["session_date"]).set_index("session_date")
    base = frozen.portfolio_pnl_usd; sleeves = [c for c in frozen if c != "portfolio_pnl_usd"]
    rows = []
    for asset in survivors:
        c = trades[asset].assign(session_date=pd.to_datetime(trades[asset].session_date)).groupby("session_date").net_pnl.sum()
        z = frozen.join(c.rename("candidate"), how="left").fillna({"candidate": 0.0}); p, q = z.portfolio_pnl_usd, z.candidate
        row = {"asset": asset, "corr_portfolio": q.corr(p), "downside_correlation": q[p < 0].corr(p[p < 0]),
               "worst_5pct_mean_candidate": q[p <= p.quantile(.05)].mean(), "worst_10pct_mean_candidate": q[p <= p.quantile(.10)].mean(),
               "worst_20pct_mean_candidate": q[p <= p.quantile(.20)].mean(), "active_overlap": ((q != 0) & (p != 0)).mean(),
               "marginal_variance_contribution": (p + q).var() - p.var(), "trend_failure_mean_candidate": q[p < 0].mean()}
        for sleeve in sleeves:
            row[f"corr_{sleeve}"] = q.corr(z[sleeve])
        rows.append(row)
    if not rows:
        return pd.DataFrame([{"status": "NOT_RUN_NO_STANDALONE_SURVIVOR"}])
    return pd.DataFrame(rows)


def paired_topstep(trades: dict[str, pd.DataFrame], executable: list[str]) -> pd.DataFrame:
    executable = [a for a in executable if a in {"M2K", "MES", "MNQ", "MGC"}]
    if not executable:
        return pd.DataFrame([{"status": "NOT_RUN_NO_EXECUTABLE_MICRO_SURVIVOR"}])
    timeline, base_daily, base_min, base_traded, bootstrap_traded = build_baseline(); calendar = base_daily.index
    baseline = {"daily": base_daily.to_numpy(float), "intraday_min": base_min.to_numpy(float), "traded": base_traded.to_numpy(bool)}
    variants = {"equal_contract__nominal__six": {**baseline, "traded": bootstrap_traded.to_numpy(bool)}, "nine_sleeves": baseline}
    dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)
    for asset in executable:
        t = trades[asset].copy(); t["mae_usd"] = t.mae; t["nominal_cost"] = nominal_cost(INSTRUMENTS[asset])
        _, bad, daily, traded = candidate_intraday(t, timeline, INSTRUMENTS[asset].point_value)
        cmin = bad.groupby(dates, sort=False).min(); cmin.index = pd.to_datetime(cmin.index)
        variants[asset] = {"daily": base_daily.add(daily.reindex(calendar).fillna(0)).to_numpy(float),
                           "intraday_min": base_min.add(cmin.reindex(calendar).fillna(0)).to_numpy(float),
                           "traded": (base_traded | traded.reindex(calendar).fillna(False)).to_numpy(bool)}
    runs, _ = topstep_campaign(variants)
    boot = runs[(runs.method.eq("moving_block_bootstrap")) & (runs.horizon.eq(60))]
    keys = ["block", "path_id"]; base = boot[boot.scenario.eq("nine_sleeves")].set_index(keys); rows = []
    rng = np.random.default_rng(SEED)
    for asset in executable:
        cand = boot[boot.scenario.eq(asset)].set_index(keys).loc[base.index]
        for outcome, label in (("PASS", "delta_pass"), ("MLL_FAIL", "delta_mll"), ("EXPIRY", "delta_expiry")):
            diff = cand.outcome.eq(outcome).astype(float).to_numpy() - base.outcome.eq(outcome).astype(float).to_numpy()
            draws = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(5000)])
            rows.append({"status": "RUN", "asset": asset, "metric": label, "paths": len(diff), "estimate": diff.mean(),
                         "ci95_low": np.quantile(draws, .025), "ci95_high": np.quantile(draws, .975),
                         "baseline_probability": base.outcome.eq(outcome).mean(), "candidate_probability": cand.outcome.eq(outcome).mean()})
    return pd.DataFrame(rows)


def classify(perf: pd.DataFrame, thin: pd.DataFrame, temporal_df: pd.DataFrame, two: pd.DataFrame, source_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in ASSETS:
        o = perf[(perf.asset == asset) & (perf.scope == "OOS") & (perf.scenario == "nominal")].iloc[0]
        c2 = perf[(perf.asset == asset) & (perf.scope == "OOS") & (perf.scenario == "plus_2_ticks_side")].iloc[0]
        th = thin[thin.asset == asset].iloc[0]
        conc = temporal_df[(temporal_df.asset == asset) & temporal_df.period.eq("remove_best_10_trades")].iloc[0]
        source = source_summary[(source_summary.asset == asset) & source_summary.mechanism.eq("E")].iloc[0]
        a = two[(two.asset == asset) & two.group.eq("A_large_high")].iloc[0]
        b = two[(two.asset == asset) & two.group.eq("B_large_low_normal")].iloc[0]
        volume_difference = abs(float(b.forward_return_usd) - float(a.forward_return_usd))
        beats_thinning = float(th.sharpe_percentile) >= 95 and float(th.expectancy_percentile) >= 95
        positive_year_fraction = float(source.positive_oos_year_fraction)
        cost = float(c2.net_pnl) > 0; stable = positive_year_fraction >= .50; concentrated = float(conc.net_pnl) > 0
        meaningful = volume_difference > .10 * max(1.0, abs(float(a.forward_return_usd)))
        supports_reversal = float(b.forward_return_usd) < float(a.forward_return_usd) and float(b.forward_return_usd) < 0
        opposite = float(b.forward_return_usd) > float(a.forward_return_usd) and float(b.forward_return_usd) > 0
        if opposite and meaningful:
            label = "OPPOSITE_EFFECT"
        elif float(o.expectancy) <= 0 or float(o.pf) <= 1:
            label = "NO_EFFECT"
        elif beats_thinning and cost and stable and concentrated and meaningful and supports_reversal:
            label = "SUPPORTS_EFFECT"
        elif beats_thinning and cost and meaningful and supports_reversal:
            label = "WEAK_SUPPORT"
        else:
            label = "NO_EFFECT"
        rows.append({"asset": asset, "classification": label, "beats_thinning": beats_thinning, "plus2_cost_positive": cost,
                     "positive_oos_year_fraction": positive_year_fraction, "remove_best_10_positive": concentrated,
                     "large_low_minus_high_forward_usd": float(b.forward_return_usd) - float(a.forward_return_usd), "meaningful_2x2_difference": meaningful})
    return pd.DataFrame(rows)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "event_studies").mkdir(exist_ok=True)
    source_summary = pd.read_csv(SOURCE / "mechanism_by_asset_summary.csv")
    source_cells = pd.read_csv(SOURCE / "all_parameter_results.csv")
    inherited = source_cells[(source_cells.mechanism.eq("E")) & source_cells.asset.isin(ASSETS)].copy()
    inherited.to_csv(OUT / "inherited_region_cells.csv", index=False)
    perf_rows: list[dict[str, Any]] = []; thin_rows = []; two_rows = []; temporal_frames = []; vol_frames = []; tod_frames = []; dir_frames = []
    events_by_asset: dict[str, pd.DataFrame] = {}; trades_by_asset: dict[str, pd.DataFrame] = {}; splits = {}; frames = {}
    for asset in ASSETS:
        p = REPRESENTATIVES[asset]; expected_vid = variant_id(p)
        saved = source_summary[(source_summary.asset == asset) & source_summary.mechanism.eq("E")].iloc[0]
        if saved.representative_variant != expected_vid:
            raise RuntimeError(f"Representative mismatch for {asset}: {saved.representative_variant} != {expected_vid}")
        x = prepare(asset); frames[asset] = x; split = split_date(x); splits[asset] = split
        sig = signals(x, p); price_sig = price_only_signals(x, p)
        t = backtest(x, sig, INSTRUMENTS[asset]); bench = backtest(x, price_sig, INSTRUMENTS[asset])
        trades_by_asset[asset] = t[t.session_date >= split].copy()
        for scope, mask in (("IS", t.session_date < split), ("OOS", t.session_date >= split)):
            part = t[mask]
            for scenario, z in (("nominal", part), ("plus_1_tick_side", apply_cost(part, asset, 1)),
                                ("plus_2_ticks_side", apply_cost(part, asset, 2)), ("double_commission", apply_cost(part, asset, 0, 2))):
                perf_rows.append({"record_type": "asset_performance", "asset": asset, "variant_id": expected_vid, "scope": scope,
                                  "scenario": scenario, "region_stability": saved.region_stability, **trade_metrics(z)})
        for delay in (1, 2):
            z = backtest(x, sig, INSTRUMENTS[asset], delay=delay); z = z[z.session_date >= split]
            perf_rows.append({"record_type": "asset_performance", "asset": asset, "variant_id": expected_vid, "scope": "OOS",
                              "scenario": f"delayed_{delay}_bar", "region_stability": saved.region_stability, **trade_metrics(z)})
        observed = t[t.session_date >= split]; bmo = bench[bench.session_date >= split]
        thin_rows.append(thinning(asset, observed, bmo))
        ev, agg = event_rows(x, asset, p, split); events_by_asset[asset] = ev; two_rows.append(agg)
        temporal_frames.append(temporal(asset, t, split)); vol_frames.append(volume_stability(asset, ev, x, split))
        tod, dr = time_direction(asset, x, split, p); tod_frames.append(tod); dir_frames.append(dr)
        event_study(asset, x, split, p).to_csv(OUT / "event_studies" / f"{asset}.csv", index=False)
    perf = pd.DataFrame(perf_rows); thin_df = pd.DataFrame(thin_rows); two = pd.concat(two_rows, ignore_index=True)
    temporal_df = pd.concat(temporal_frames, ignore_index=True); vol_df = pd.concat(vol_frames, ignore_index=True)
    tod_df = pd.concat(tod_frames, ignore_index=True); direction_df = pd.concat(dir_frames, ignore_index=True)
    dependence = cross_dependence(events_by_asset, trades_by_asset)
    classes = classify(perf, thin_df, temporal_df, two, source_summary)
    class_records = classes.copy(); class_records.insert(0, "record_type", "asset_classification")
    cross = pd.concat([two, perf, tod_df, dependence, class_records], ignore_index=True, sort=False)
    cross.to_csv(OUT / "cross_asset_event_analysis.csv", index=False)
    thin_df.to_csv(OUT / "matched_thinning_tests.csv", index=False)
    pd.concat([temporal_df, vol_df], ignore_index=True, sort=False).to_csv(OUT / "temporal_stability.csv", index=False)
    direction_df.to_csv(OUT / "directional_decomposition.csv", index=False)
    survivors = classes.loc[classes.classification.eq("SUPPORTS_EFFECT"), "asset"].tolist()
    portfolio = diversification(trades_by_asset, survivors); portfolio.to_csv(OUT / "portfolio_diagnostics.csv", index=False)
    topstep = paired_topstep(trades_by_asset, survivors); topstep.to_csv(OUT / "topstep_paired_bootstrap.csv", index=False)
    class_counts = classes.classification.value_counts().to_dict()
    distinct_support = [a for a in survivors if a not in {"NQ"}]
    if not survivors:
        global_decision = "REJECTED"
    elif len(distinct_support) >= 2 and any(a == "MGC" for a in survivors):
        global_decision = "REAL_CROSS_ASSET_EFFECT"
    elif all(a in {"M2K", "MES", "MNQ", "NQ"} for a in survivors):
        global_decision = "ASSET_CLASS_SPECIFIC_EFFECT"
    else:
        global_decision = "REAL_EFFECT_NOT_YET_EXECUTABLE"
    reps = pd.DataFrame([{"asset": a, "variant_id": variant_id(REPRESENTATIVES[a]), **REPRESENTATIVES[a]} for a in ASSETS])
    oos = perf[(perf.scope.eq("OOS")) & perf.scenario.eq("nominal")]
    two_ab = two[two.group.isin(["A_large_high", "B_large_low_normal"])]
    period = temporal_df[(temporal_df.record_type.eq("period")) & temporal_df.period.isin(["IS", "OOS", "pre_2020", "2020_2022", "2023_plus"])]
    interpretations = []
    for asset in ASSETS:
        es = pd.read_csv(OUT / "event_studies" / f"{asset}.csv")
        a = es[es.group.eq("A_large_high")].set_index("horizon_bars").mean_continuation_usd
        b = es[es.group.eq("B_large_low_normal")].set_index("horizon_bars").mean_continuation_usd
        if b.loc[1] < 0 and b.loc[6] < 0:
            label = "IMMEDIATE_REVERSAL_THROUGH_30M"
        elif b.loc[1] >= 0 and b.loc[6] < 0:
            label = "DELAYED_REVERSAL"
        elif b.loc[6] > a.loc[6] and b.loc[6] > 0:
            label = "STRONGER_LOW_VOLUME_CONTINUATION_OPPOSITE_HYPOTHESIS"
        elif b.loc[6] < a.loc[6]:
            label = "WEAKER_CONTINUATION"
        else:
            label = "NO_STABLE_DIRECTIONAL_DIFFERENCE"
        interpretations.append({"asset": asset, "economic_path": label, "A_high_volume_30m_usd": a.loc[6],
                                "B_low_normal_volume_30m_usd": b.loc[6], "B_minus_A_30m_usd": b.loc[6] - a.loc[6]})
    interpretation_df = pd.DataFrame(interpretations)
    report = ["# Price Move Without Volume Confirmation Audit", "", f"**Global economic-mechanism classification: `{global_decision}`**", "",
              "This is a frozen-region audit. No new parameter, asset, threshold, time filter, weight, or contract count was selected. The frozen nine sleeves were read only.", "",
              "## 1. Exact inherited mechanism", "",
              "- Five-minute regular-session OHLCV bars in America/New_York time.",
              "- Price move: `abs((close-open)/ATR)`, with ATR equal to the median true range of the prior 10-20 completed bars within that session (10-bar minimum).",
              "- Large-move region: 1.0, 1.5, and 2.0 ATR; inherited representatives use 1.5 ATR.",
              "- Expected volume: median volume of the prior 20 completed sessions at the same five-minute clock slot; `RVOL = volume / expected volume`.",
              "- No confirmation: RVOL <= 0.75 or <= 1.0; the representative threshold is asset-specific below.",
              "- Signal at completed-bar close; entry at next bar open; exit at the close of the sixth holding bar, truncated at session end; trades do not overlap.",
              "- Nominal cost: one adverse tick per side plus commission. Stress adds one or two further ticks per side or doubles commission.", "",
              markdown(reps[["asset", "variant_id", "mode", "move", "rvol"]]), "",
              f"All {len(inherited)} inherited IS/OOS rows are preserved in `inherited_region_cells.csv`: 12 exact variants per asset (3 move thresholds x 2 RVOL thresholds x fade/continuation control), each with IS and OOS rows. No cell was added.", "",
              "## 2. Price move x volume 2x2 (OOS)", "", markdown(two_ab[["asset", "group", "events", "forward_return_usd", "reversal_probability", "continuation_probability", "mae_usd", "mfe_usd", "forward_realized_vol"]]), "",
              "The forward return is signed in the direction of the triggering price bar: negative means reversal and positive means continuation. Thus the A/B comparison is independent of each representative's inherited trade direction.", "",
              "## 3. Matched price-only benchmark", "", markdown(thin_df[["asset", "draws", "observed_trades", "benchmark_trades", "observed_sharpe", "sharpe_percentile", "observed_expectancy", "expectancy_percentile", "observed_pf", "pf_percentile", "prob_random_matches_exceeds_sharpe"]]), "",
              "Random thinning is OOS, without replacement, and exactly matched to the volume-filtered trade count for 1,000 draws per asset.", "",
              "## 4. Cross-asset replication and classifications", "", markdown(oos[["asset", "variant_id", "trades", "sharpe", "pf", "expectancy", "maxdd", "region_stability"]].merge(classes, on="asset")), "",
              markdown(dependence), "",
              "MNQ and NQ are explicitly treated as related expressions, not independent confirmations. The distinct-event summary counts the union of M2K/MES/MNQ/NQ dates rather than summing contracts.", "",
              "## 5. Asset-class interpretation", "",
              f"- US equity indices: {', '.join(f'{r.asset}={r.classification}' for r in classes[classes.asset.isin(['M2K','MES','MNQ','NQ'])].itertuples())}.",
              f"- Gold: MGC={classes.set_index('asset').loc['MGC','classification']}.",
              f"- Bitcoin: MBT={classes.set_index('asset').loc['MBT','classification']}.", "",
              "The classes are kept separate; contracts are not pooled into a synthetic cross-asset Sharpe.", "",
              "## 6. Temporal stability", "", markdown(period[["asset", "period", "trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"]]), "",
              "Yearly, rolling-12-month, best-year removal, best-5-trade removal, best-10-trade removal, and annual volume-signal stability are in `temporal_stability.csv`.", "",
              "The positive OOS appearances are not persistent: M2K, MNQ, and MGC have negative IS histories; NQ is concentrated in 2020-2022 and is negative in 2023+; and every asset is negative after removing its ten best OOS trades.", "",
              "## 7. Volume-signal stability", "", markdown(vol_df), "",
              "The qualifying RVOL distribution and event frequency move materially through time, while expected-volume levels also shift with contract liquidity (especially MGC). The filter is causal, but its absolute liquidity context is not stationary.", "",
              "## 8. Time of day", "", markdown(tod_df), "",
              "The result is not uniformly a midday effect: M2K is strongest late, while MGC's inherited continuation result is concentrated near the open. Bucket signs vary sharply, which is further evidence against a common stable mechanism. These are diagnostics only; no time filter was introduced.", "",
              "## 9. Directional symmetry", "", markdown(direction_df), "",
              "## 10. Cost and latency stress", "", markdown(perf[(perf.scope.eq('OOS'))][["asset", "scenario", "trades", "net_pnl", "sharpe", "pf", "expectancy", "maxdd"]]), "",
              "## 11. Economic interpretation", "",
              markdown(interpretation_df), "",
              "The per-asset 1/3/6/12-bar A/B paths are in `event_studies/`. M2K, MES, and MNQ reverse immediately through 30 minutes; NQ shows delayed reversal. MGC and MBT instead show stronger continuation on low/normal volume, the opposite of the proposed reversal mechanism. Most reversal paths dissipate or flip by 60 minutes. This sign disagreement is why profitable asset-specific representatives do not establish one common cross-asset effect.", "",
              "## 12. Diversification", "", markdown(portfolio), "",
              "Diversification was run only for strict standalone survivors. An empty gated result is recorded explicitly rather than backfilling weaker assets.", "",
              "## 13. Topstep", "", markdown(topstep), "",
              "NQ is excluded as an execution candidate. The paired/common-path simulation is run only for strict executable micro survivors and does not optimize weights or contract counts.", "",
              "## Final decision", "", markdown(classes), "",
              f"Classification counts: `{class_counts}`.", "",
              f"**Final global classification: `{global_decision}`.**"]
    (OUT / "PRICE_MOVE_WITHOUT_VOLUME_CONFIRMATION_AUDIT.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {"decision": global_decision, "source_campaign": str(SOURCE.relative_to(REPO)).replace("\\", "/"),
                "representatives": {a: variant_id(REPRESENTATIVES[a]) for a in ASSETS}, "split_dates": {a: str(d.date()) for a, d in splits.items()},
                "thinning_draws": THINNING_DRAWS, "new_parameter_search": False, "frozen_nine_sleeves_modified": False,
                "data_hashes": {a: sha256(REPO / DATA_FILES[a]) for a in ASSETS},
                "outputs": ["PRICE_MOVE_WITHOUT_VOLUME_CONFIRMATION_AUDIT.md", "cross_asset_event_analysis.csv", "matched_thinning_tests.csv", "temporal_stability.csv", "directional_decomposition.csv", "event_studies/", "portfolio_diagnostics.csv", "topstep_paired_bootstrap.csv"]}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"decision": global_decision, "asset_classes": classes.set_index("asset").classification.to_dict(), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    run()
