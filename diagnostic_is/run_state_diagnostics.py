"""Available IS-only state diagnostics (Steps 2-4) with blocked inputs omitted."""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diagnostic_is.run_step1 import (  # noqa: E402
    A_LABELS,
    A_OOS_START,
    B_CONFIG,
    B_OOS_START,
    B_PARAMS,
    B_SOURCE,
    GAP_SOURCES,
    TEMP,
    _assert_is_only_bars,
    _write_is_only_parquet,
)
from research.mean_reversion_cross_asset_20261006.run_campaign import (  # noqa: E402
    INSTRUMENTS as GAP_INSTRUMENTS,
    _load_bars,
    _prepare_features,
)
from research.volume_auction_landscape_20261007.run_campaign import (  # noqa: E402
    INSTRUMENTS as CLIMAX_INSTRUMENTS,
    add_causal_volume_baselines,
    add_price_features,
    backtest as climax_backtest,
    load_bars,
    metrics as climax_metrics,
    price_only_signals,
    signals,
)


OUT = Path(__file__).resolve().parent
TABLES = OUT / "state_tables"
SEED = 20261008
BOOTSTRAPS = 2000


def tercile(values: pd.Series) -> tuple[pd.Series, list[float]]:
    valid = values.dropna()
    q = valid.quantile([1 / 3, 2 / 3]).to_list()
    labels = pd.cut(values, [-np.inf, q[0], q[1], np.inf], labels=["low", "mid", "high"], include_lowest=True)
    return labels.astype("string"), [float(q[0]), float(q[1])]


def load_cboe(path: Path, name: str) -> pd.Series:
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame.DATE)
    series = pd.Series(pd.to_numeric(frame.CLOSE).to_numpy(), index=dates, name=name).sort_index()
    return series[series.index < A_OOS_START]


def prior_market_value(dates: pd.Series, series: pd.Series) -> pd.Series:
    left = pd.DataFrame({"date": pd.to_datetime(dates), "order": np.arange(len(dates))}).sort_values("date")
    right = series.rename("value").rename_axis("obs_date").reset_index().sort_values("obs_date")
    merged = pd.merge_asof(left, right, left_on="date", right_on="obs_date", direction="backward", allow_exact_matches=False)
    return merged.sort_values("order").value.reset_index(drop=True)


def enrich_a() -> tuple[pd.DataFrame, dict[str, list[float]]]:
    trades = pd.read_csv(TABLES.parent / "step1_tables/signal_a_exclusive_bucket_trades_is.csv")
    vix = load_cboe(REPO / "data/raw/vix-cboe.csv", "vix")
    vix3m = load_cboe(REPO / "data/raw/vix3m-cboe.csv", "vix3m")
    enriched: list[pd.DataFrame] = []
    cuts: dict[str, list[float]] = {}
    for symbol, group in trades.groupby("asset", sort=False):
        temp_path = TEMP / f"states_A_{symbol}_is.parquet"
        _write_is_only_parquet(GAP_SOURCES[symbol], temp_path, A_OOS_START)
        bars = _prepare_features(_load_bars(temp_path, GAP_INSTRUMENTS[symbol]))
        _assert_is_only_bars(bars, A_OOS_START)
        daily = bars.groupby("session_date", sort=True).agg(
            close=("close", "last"), high=("high", "max"), low=("low", "min")
        )
        daily.index = pd.to_datetime(daily.index)
        daily["ret20"] = daily.close.pct_change(20).shift(1)
        daily["prior_close"] = daily.close.shift(1)
        daily["sma50"] = daily.close.rolling(50, min_periods=50).mean().shift(1)
        obs = bars[bars.bar_in_session.eq(3)].copy()
        obs.index = pd.to_datetime(obs.session_date)
        dates = pd.to_datetime(group.session_date).reset_index(drop=True)
        z = group.copy().reset_index(drop=True)
        z["ret20"] = dates.map(daily.ret20)
        z["prior_sma50"] = dates.map(daily.sma50)
        z["obs20_close"] = dates.map(obs.close)
        gap_sign = np.sign(z.gap_points)
        z["A3_ret20"] = np.where(gap_sign.eq(np.sign(z.ret20)), "with_trend", "against_trend")
        sma_sign = np.sign(z.prior_close - z.prior_sma50)
        z["A3_sma50"] = np.where(gap_sign.eq(sma_sign), "with_trend", "against_trend")
        z.loc[z.ret20.isna(), "A3_ret20"] = pd.NA
        z.loc[z.prior_sma50.isna(), "A3_sma50"] = pd.NA
        z["recovery_fraction_20m"] = np.sign(z.gap_points) * (z.session_open - z.obs20_close) / z.gap_points.abs()
        z["gap_atr14"] = z.gap_points.abs() / z.prior_daily_atr14
        z["A5_gap_bucket"] = z.configuration
        z["A6_direction"] = np.where(z.gap_points > 0, "gap_up", "gap_down")
        z["vix"] = prior_market_value(dates, vix)
        z["vix3m"] = prior_market_value(dates, vix3m)
        z["A2_term_structure"] = np.where(z.vix > z.vix3m, "backwardation", "contango")
        z.loc[z.vix.isna() | z.vix3m.isna(), "A2_term_structure"] = pd.NA
        # Terciles are estimated on the NQ inference sample and reused for robustness assets.
        enriched.append(z)
    out = pd.concat(enriched, ignore_index=True)
    nq = out.asset.eq("NQ")
    for field, variable in (("vix", "A2_vix_tercile"), ("recovery_fraction_20m", "A4_recovery_tercile"), ("gap_atr14", "A5_gap_atr_tercile")):
        if out.loc[nq, field].dropna().empty:
            raise AssertionError(f"No valid NQ observations for {field}")
        _, cut = tercile(out.loc[nq, field])
        if not cut[0] < cut[1]:
            raise AssertionError(f"Non-distinct terciles for {field}: {cut}")
        cuts[variable] = cut
        out[variable] = pd.cut(out[field], [-np.inf, cut[0], cut[1], np.inf], labels=["low", "mid", "high"], include_lowest=True).astype("string")
    if pd.to_datetime(out.session_date).max() >= A_OOS_START:
        raise AssertionError("Post-IS Signal A trade in state diagnostics")
    return out, cuts


def daily_gold_features(bars: pd.DataFrame, daily_instrument: pd.Series) -> pd.DataFrame:
    daily = bars.groupby("session_date", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    )
    daily.index = pd.to_datetime(daily.index)
    daily["instrument_id"] = daily.index.map(daily_instrument)
    daily["ret50"] = daily.close.pct_change(50).shift(1)
    daily["prior_close"] = daily.close.shift(1)
    daily["sma100"] = daily.close.rolling(100, min_periods=100).mean().shift(1)
    daily["rv20"] = np.log(daily.close / daily.close.shift(1)).rolling(20, min_periods=20).std(ddof=1).shift(1) * math.sqrt(252)
    daily["roll_date"] = daily.instrument_id.ne(daily.instrument_id.shift())
    if len(daily):
        daily.iloc[0, daily.columns.get_loc("roll_date")] = False
    roll_positions = np.flatnonzero(daily.roll_date.to_numpy())
    in_roll = np.zeros(len(daily), dtype=bool)
    for pos in roll_positions:
        in_roll[max(0, pos - 2): min(len(daily), pos + 3)] = True
    daily["roll_window_pm2_sessions"] = in_roll
    return daily


def enrich_b() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[float]], pd.DataFrame]:
    trades = pd.read_csv(TABLES.parent / "step1_tables/signal_b_trades_is.csv")
    temp_path = TEMP / "states_B_MGC_is.parquet"
    _write_is_only_parquet(B_SOURCE, temp_path, B_OOS_START)
    bars = add_price_features(add_causal_volume_baselines(load_bars(temp_path, CLIMAX_INSTRUMENTS["MGC"])))
    _assert_is_only_bars(bars, B_OOS_START)
    raw_id = pd.read_parquet(temp_path, columns=["instrument_id"])
    raw_id.index = pd.DatetimeIndex(raw_id.index).tz_convert("America/New_York")
    tod = raw_id.index.strftime("%H:%M")
    raw_id = raw_id[(tod >= "08:20") & (tod < "13:30") & (raw_id.index.dayofweek < 5)].copy()
    raw_id["session_date"] = pd.to_datetime(raw_id.index.date)
    daily_instrument = raw_id.groupby("session_date").instrument_id.last()
    daily = daily_gold_features(bars, daily_instrument)
    z = trades.copy().reset_index(drop=True)
    dates = pd.to_datetime(z.session_date)
    signal_idx = pd.DatetimeIndex(pd.to_datetime(z.signal_time, utc=True)).tz_convert("America/New_York")
    climax_idx = signal_idx - pd.Timedelta(minutes=5)
    fields = bars[["rvol_20", "abs_move"]].copy()
    fields.index = pd.DatetimeIndex(fields.index)
    signal_fields = fields.reindex(signal_idx)
    climax = fields.reindex(climax_idx)
    z["signal_rvol_recomputed"] = signal_fields.rvol_20.to_numpy()
    z["climax_rvol"] = climax.rvol_20.to_numpy()
    z["climax_abs_move_atr"] = climax.abs_move.to_numpy()
    z["ret50"] = dates.map(daily.ret50)
    z["prior_close_daily"] = dates.map(daily.prior_close)
    z["sma100"] = dates.map(daily.sma100)
    z["rv20_daily"] = dates.map(daily.rv20)
    z["roll_window"] = np.where(dates.map(daily.roll_window_pm2_sessions), "roll_pm2", "outside_roll")
    climax_direction = -z.direction
    z["B3_ret50"] = np.where(climax_direction.eq(np.sign(z.ret50)), "with_trend", "against_trend")
    trend_sma = np.sign(z.prior_close_daily - z.sma100)
    z["B3_sma100"] = np.where(climax_direction.eq(trend_sma), "with_trend", "against_trend")
    z.loc[z.ret50.isna(), "B3_ret50"] = pd.NA
    z.loc[z.sma100.isna(), "B3_sma100"] = pd.NA
    cuts: dict[str, list[float]] = {}
    for field, variable in (("rv20_daily", "B4_gold_rv20_tercile"), ("climax_rvol", "B5_volume_tercile"), ("climax_abs_move_atr", "B5_move_tercile")):
        labels, cut = tercile(z[field])
        z[variable] = labels
        cuts[variable] = cut
    if not np.allclose(z.signal_rvol_recomputed, z.signal_rvol, rtol=0, atol=1e-12):
        raise AssertionError("Stored signal RVOL does not match the canonical same-slot RVOL feature")
    liquidity = bars.assign(year=bars.index.year).groupby("year").agg(
        bars=("volume", "size"), median_bar_volume=("volume", "median"), pct_zero_volume=("volume", lambda s: float((s == 0).mean())), pct_volume_le_1=("volume", lambda s: float((s <= 1).mean()))
    ).reset_index()
    day_volume = bars.groupby("session_date").volume.sum().rename("daily_volume").reset_index()
    day_volume["year"] = pd.to_datetime(day_volume.session_date).dt.year
    daily_liq = day_volume.groupby("year", as_index=False).daily_volume.median().rename(columns={"daily_volume": "median_daily_volume"})
    liquidity = liquidity.merge(daily_liq, on="year", how="left")
    return z, bars, cuts, liquidity


def bucket_metrics(frame: pd.DataFrame, signal: str, asset: str, variable: str, tick_value: float) -> pd.DataFrame:
    rows = []
    total = float(frame.net_pnl.sum())
    for bucket, group in frame.dropna(subset=[variable]).groupby(variable, observed=True):
        pnl = group.net_pnl.astype(float)
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        sd = pnl.std(ddof=1)
        rows.append({
            "signal": signal, "asset": asset, "variable": variable, "bucket": str(bucket), "trades": len(group),
            "gross_pnl": group.gross_pnl.sum(), "costs": group.cost.sum(), "net_pnl": pnl.sum(),
            "avg_net_usd": pnl.mean(), "avg_net_ticks": pnl.mean() / tick_value,
            "hit_rate": (pnl > 0).mean(), "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) else np.inf,
            "t_stat": pnl.mean() / (sd / math.sqrt(len(pnl))) if len(pnl) > 1 and sd > 0 else np.nan,
            "share_of_total_net_loss": -pnl.sum() / (-total) if total < 0 else np.nan,
        })
    return pd.DataFrame(rows)


def clustered_difference(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, float]:
    combined = pd.concat([a.assign(_x=1.0), b.assign(_x=0.0)], ignore_index=True)
    y = combined.net_pnl.to_numpy(float); x = np.column_stack([np.ones(len(combined)), combined._x])
    inv = np.linalg.pinv(x.T @ x); beta = inv @ x.T @ y; resid = y - x @ beta
    meat = np.zeros((2, 2))
    for _, idx in combined.groupby("session_date").groups.items():
        score = x[list(idx)].T @ resid[list(idx)]; meat += np.outer(score, score)
    cov = inv @ meat @ inv
    se = math.sqrt(max(cov[1, 1], 0)); t = beta[1] / se if se else np.nan
    p = 2 * stats.norm.sf(abs(t)) if np.isfinite(t) else np.nan
    return float(beta[1]), float(t), float(p)


def bootstrap_difference(a: pd.DataFrame, b: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float]:
    dates = np.array(sorted(set(a.session_date) | set(b.session_date)))
    av = a.groupby("session_date").net_pnl.mean().reindex(dates).to_numpy()
    bv = b.groupby("session_date").net_pnl.mean().reindex(dates).to_numpy()
    n = len(dates); block = 5; draws = np.empty(BOOTSTRAPS)
    starts = np.arange(max(n - block + 1, 1))
    for i in range(BOOTSTRAPS):
        idx: list[int] = []
        while len(idx) < n:
            s = int(rng.choice(starts)); idx.extend(range(s, min(s + block, n)))
        take = np.array(idx[:n]); aa = av[take]; bb = bv[take]
        draws[i] = np.nanmean(aa) - np.nanmean(bb)
    return float(np.nanquantile(draws, .025)), float(np.nanquantile(draws, .975))


def tests_for_variable(frame: pd.DataFrame, signal: str, asset: str, variable: str, rng: np.random.Generator) -> pd.DataFrame:
    buckets = [str(x) for x in frame[variable].dropna().unique()]
    rows = []
    for first, second in combinations(sorted(buckets), 2):
        a = frame[frame[variable].astype(str).eq(first)]; b = frame[frame[variable].astype(str).eq(second)]
        diff, t, p = clustered_difference(a, b); lo, hi = bootstrap_difference(a, b, rng)
        rows.append({"signal": signal, "asset": asset, "variable": variable, "bucket_a": first, "bucket_b": second, "mean_difference_a_minus_b": diff, "cluster_t": t, "p_value": p, "bootstrap95_low": lo, "bootstrap95_high": hi})
    return pd.DataFrame(rows)


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(); valid = out.p_value.notna(); p = out.loc[valid, "p_value"].sort_values(); m = len(p)
    adjusted = pd.Series(index=p.index, dtype=float); running = 0.0
    for rank, (idx, value) in enumerate(p.items()):
        running = max(running, (m - rank) * value); adjusted.loc[idx] = min(running, 1.0)
    out["holm_p"] = adjusted
    out["holm_reject_5pct"] = out.holm_p <= .05
    return out


def year_matrix(frame: pd.DataFrame, signal: str, asset: str, variable: str) -> pd.DataFrame:
    z = frame.dropna(subset=[variable]).copy(); z["year"] = pd.to_datetime(z.session_date).dt.year
    return z.groupby(["year", variable], observed=True).agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum")).reset_index().rename(columns={variable: "bucket"}).assign(signal=signal, asset=asset, variable=variable)


def proxy_flags(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in matrix.groupby(["signal", "asset", "variable", "bucket"], observed=True):
        total = group.trades.sum(); share = group.nlargest(3, "trades").trades.sum() / total
        rows.append(dict(zip(["signal", "asset", "variable", "bucket"], keys)) | {"trades": total, "top_3_year_share": share, "proxy_flag_gt_60pct": share > .60})
    return pd.DataFrame(rows)


def subperiods(frame: pd.DataFrame, signal: str, asset: str, variables: list[str]) -> pd.DataFrame:
    z = frame.copy(); year = pd.to_datetime(z.session_date).dt.year
    z["subperiod"] = np.select([year <= 2019, year <= 2022], ["start-2019", "2020-2022"], default="2023-end")
    rows = []
    for variable in variables:
        r = z.dropna(subset=[variable]).groupby(["subperiod", variable], observed=True).agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), avg_net=("net_pnl", "mean")).reset_index().rename(columns={variable: "bucket"})
        r["signal"] = signal; r["asset"] = asset; r["variable"] = variable; rows.append(r)
    return pd.concat(rows, ignore_index=True)


def explanatory_score(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["signal", "asset", "variable"]):
        rows.append(dict(zip(["signal", "asset", "variable"], keys)) | {"mean_pnl_spread": group.avg_net_usd.max() - group.avg_net_usd.min(), "worst_bucket_net_pnl": group.net_pnl.min()})
    return pd.DataFrame(rows).sort_values(["signal", "asset", "mean_pnl_spread"], ascending=[True, True, False])


def interaction(frame: pd.DataFrame, signal: str, asset: str, variables: list[str]) -> pd.DataFrame:
    a, b = variables[:2]
    return frame.dropna(subset=[a, b]).groupby([a, b], observed=True).agg(trades=("net_pnl", "size"), gross_pnl=("gross_pnl", "sum"), costs=("cost", "sum"), net_pnl=("net_pnl", "sum"), avg_net=("net_pnl", "mean")).reset_index().rename(columns={a: "bucket_a", b: "bucket_b"}).assign(signal=signal, asset=asset, variable_a=a, variable_b=b)


def random_thinning_is(bars: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inst = CLIMAX_INSTRUMENTS["MGC"]
    price = climax_backtest(bars, price_only_signals(bars, B_PARAMS), inst)
    price = price[pd.to_datetime(price.session_date) < B_OOS_START]
    rng = np.random.default_rng(SEED); rows = []
    cm = climax_metrics(candidate)
    for draw in range(1000):
        sample = price.iloc[rng.choice(len(price), size=len(candidate), replace=False)]
        rows.append({"draw": draw, **climax_metrics(sample)})
    draws = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "version": "original_equals_B0c", "candidate_trades": len(candidate), "price_only_trades": len(price),
        "candidate_sharpe": cm["sharpe"], "candidate_expectancy": cm["expectancy"],
        "sharpe_percentile": (draws.sharpe <= cm["sharpe"]).mean(),
        "expectancy_percentile": (draws.expectancy <= cm["expectancy"]).mean(),
        "seed": SEED,
    }])
    return summary, draws


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True); TEMP.mkdir(exist_ok=True)
    try:
        a, a_cuts = enrich_a(); b, b_bars, b_cuts, liquidity = enrich_b()
    finally:
        for path in TEMP.glob("*_is.parquet"): path.unlink(missing_ok=True)
        try: TEMP.rmdir()
        except OSError: pass
    a.to_csv(TABLES / "signal_a_state_trades.csv", index=False); b.to_csv(TABLES / "signal_b_state_trades.csv", index=False)
    liquidity.to_csv(TABLES / "B0a_mgc_liquidity_yearly.csv", index=False)
    b.groupby("roll_window", as_index=False).agg(trades=("net_pnl", "size"), gross_pnl=("gross_pnl", "sum"), costs=("cost", "sum"), net_pnl=("net_pnl", "sum")).to_csv(TABLES / "B0d_roll_window.csv", index=False)
    pd.DataFrame([{"original_version": "rvol_20", "B0c_version": "same-slot median over prior 20 sessions", "identical": True, "trade_count": len(b), "net_pnl": b.net_pnl.sum()}]).to_csv(TABLES / "B0c_identity.csv", index=False)

    a_vars = ["A2_term_structure", "A2_vix_tercile", "A3_ret20", "A3_sma50", "A4_recovery_tercile", "A5_gap_bucket", "A5_gap_atr_tercile", "A6_direction"]
    b_vars = ["roll_window", "session_bucket", "B3_ret50", "B3_sma100", "B4_gold_rv20_tercile", "B5_volume_tercile", "B5_move_tercile"]
    metric_frames = []; test_frames = []; matrix_frames = []; rng = np.random.default_rng(SEED)
    # Inference: NQ only for A, MGC for B.
    for frame, signal, asset, variables, tick in ((a[a.asset.eq("NQ")], "A", "NQ", a_vars, 5.0), (b, "B", "MGC", b_vars, 1.0)):
        for variable in variables:
            metric_frames.append(bucket_metrics(frame, signal, asset, variable, tick))
            test_frames.append(tests_for_variable(frame, signal, asset, variable, rng))
            matrix_frames.append(year_matrix(frame, signal, asset, variable))
    metrics = pd.concat(metric_frames, ignore_index=True); tests = holm(pd.concat(test_frames, ignore_index=True)); matrices = pd.concat(matrix_frames, ignore_index=True)
    metrics.to_csv(TABLES / "bucket_metrics.csv", index=False); tests.to_csv(TABLES / "bucket_tests_holm.csv", index=False); matrices.to_csv(TABLES / "year_bucket_matrices.csv", index=False)
    proxy_flags(matrices).to_csv(TABLES / "temporal_proxy_flags.csv", index=False)
    pd.concat([subperiods(a[a.asset.eq("NQ")], "A", "NQ", a_vars), subperiods(b, "B", "MGC", b_vars)], ignore_index=True).to_csv(TABLES / "subperiod_stability.csv", index=False)
    scores = explanatory_score(metrics); scores.to_csv(TABLES / "variable_explanatory_ranking.csv", index=False)
    interactions = []
    for signal, asset, frame in (("A", "NQ", a[a.asset.eq("NQ")]), ("B", "MGC", b)):
        top = scores[(scores.signal == signal) & (scores.asset == asset)].variable.head(2).tolist()
        interactions.append(interaction(frame, signal, asset, top))
    pd.concat(interactions, ignore_index=True).to_csv(TABLES / "top2_interactions.csv", index=False)
    thinning_summary, thinning_draws = random_thinning_is(b_bars, b)
    thinning_summary.to_csv(TABLES / "B6_random_thinning_summary.csv", index=False); thinning_draws.to_csv(TABLES / "B6_random_thinning_draws.csv", index=False)
    # Micros appear only in robustness, using the same NQ-derived cut points.
    robustness = []
    for asset in ("MES", "MNQ", "M2K"):
        for variable in a_vars:
            robustness.append(bucket_metrics(a[a.asset.eq(asset)], "A", asset, variable, GAP_INSTRUMENTS[asset].tick_value))
    pd.concat(robustness, ignore_index=True).to_csv(TABLES / "signal_a_micros_robustness.csv", index=False)
    metadata = {
        "seed": SEED, "bootstrap_draws": BOOTSTRAPS, "bootstrap_block_dates": 5,
        "number_of_tests": int(len(tests)), "holm_family": "all available variable pairwise bucket contrasts",
        "a_tercile_cutpoints": a_cuts, "b_tercile_cutpoints": b_cuts,
        "blocked": ["A1 macro", "B0b GC volume", "B1 announcement bar"],
        "B0c_note": "The canonical signal already uses bar volume divided by the prior-20-session same-slot median; B0c is exactly identical to original.",
        "post_is_statistics_computed": False,
    }
    (OUT / "state_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"tests": len(tests), "holm_rejections": int(tests.holm_reject_5pct.sum()), "top_variables": scores.groupby("signal").head(2)[["signal", "variable"]].to_dict("records"), "thinning": thinning_summary.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
