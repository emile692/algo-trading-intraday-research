"""Phase-1 screen of same-slot intraday return periodicity in futures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "intraday_periodicity_phase1_20261003"
WINDOWS = {
    "A_0830_0900_CT": ("08:30", "09:00"),
    "B_0900_0930_CT": ("09:00", "09:30"),
    "C_1430_1500_CT": ("14:30", "15:00"),
}
LOOKBACKS = (10, 20, 40)
PRIMARY_LOOKBACK = 20

# Frozen before examining results. One representative per broad asset class is
# selected for cross-section; duplicates remain in the inventory only.
SPECS = {
    "NQ": dict(asset_class="equity_index", path="data/processed/parquet/NQ_c_0_1m_20260321_095036.parquet", point_value=20.0, tick_size=0.25, tick_value=5.0, commission_rt=3.78, selected=True, reason="selected; longest local Nasdaq history"),
    "MNQ": dict(asset_class="equity_index", path="data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet", point_value=2.0, tick_size=0.25, tick_value=0.5, commission_rt=1.22, selected=False, reason="excluded cross-section duplicate of NQ"),
    "MES": dict(asset_class="equity_index", path="data/processed/parquet/MES_c_0_1m_20260322_135702.parquet", point_value=5.0, tick_size=0.25, tick_value=1.25, commission_rt=1.22, selected=False, reason="excluded to avoid correlated equity-index duplicates"),
    "M2K": dict(asset_class="equity_index", path="data/processed/parquet/M2K_c_0_1m_20260322_134808.parquet", point_value=5.0, tick_size=0.10, tick_value=0.5, commission_rt=1.22, selected=False, reason="excluded to keep one equity-index representative"),
    "ZN": dict(asset_class="rates", path="data/processed/parquet/ZN_v_0_1m_20130517_20260920.parquet", point_value=1000.0, tick_size=1/64, tick_value=15.625, commission_rt=2.62, selected=True, reason="selected liquid 10-year rates representative"),
    "ZF": dict(asset_class="rates", path="data/processed/parquet/ZF_v_0_1m_20130517_20260920.parquet", point_value=1000.0, tick_size=1/128, tick_value=7.8125, commission_rt=2.32, selected=False, reason="excluded cross-section rates duplicate of ZN"),
    "6E": dict(asset_class="fx", path="data/processed/parquet/6E_v_0_1m_20150101_20260923.parquet", point_value=125000.0, tick_size=0.00005, tick_value=6.25, commission_rt=4.22, selected=True, reason="selected FX representative"),
    "6J": dict(asset_class="fx", path="data/downloaded/6J_v_0_1m_20150101_20261003.parquet", point_value=12_500_000.0, tick_size=0.0000005, tick_value=6.25, commission_rt=4.22, selected=False, reason="excluded to keep one FX representative; Gotobi remains separate"),
    "MGC": dict(asset_class="metals", path="data/processed/parquet/MGC_v_0_1m_20150101_20260923.parquet", point_value=10.0, tick_size=0.10, tick_value=1.0, commission_rt=1.92, selected=True, reason="selected metals representative"),
    "MCL": dict(asset_class="energy", path="data/downloaded/MCL_v_0_1m_20210712_20260924.parquet", point_value=100.0, tick_size=0.01, tick_value=1.0, commission_rt=1.52, selected=True, reason="selected energy representative; defines common start"),
    "NG": dict(asset_class="energy", path="data/downloaded/NG_v_0_1m_20200101_20261003.parquet", point_value=10000.0, tick_size=0.001, tick_value=10.0, commission_rt=4.22, selected=False, reason="excluded to keep one energy representative and avoid oversized tick risk"),
}


def hac_lag(n: int) -> int:
    return max(1, int(math.floor(4 * (n / 100) ** (2 / 9))))


def hac_ols_t(y: np.ndarray, x: np.ndarray, coefficient: int) -> float:
    """Newey-West t-statistic with Bartlett weights and no small-sample tweak."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(hac_lag(len(y)), len(y) - 1) + 1):
        weight = 1.0 - lag / (hac_lag(len(y)) + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(max(covariance[coefficient, coefficient], 0.0)))
    return float(beta[coefficient] / se) if se > 0 else np.nan


def hac_mean_t(values: pd.Series) -> float:
    y = pd.Series(values).dropna().astype(float)
    if len(y) < 5 or y.std(ddof=1) == 0:
        return np.nan
    return hac_ols_t(y.to_numpy(), np.ones((len(y), 1)), 0)


def hac_spread_t(frame: pd.DataFrame) -> float:
    use = frame[["ret", "score"]].dropna()
    if len(use) < 10 or use.score.gt(0).nunique() < 2:
        return np.nan
    dummy = use.score.gt(0).astype(float).to_numpy()
    x = np.column_stack([np.ones(len(use)), dummy])
    return hac_ols_t(use.ret.to_numpy(), x, 1)


def performance(values: pd.Series) -> dict[str, float]:
    x = pd.Series(values).dropna().astype(float)
    if x.empty:
        return dict(pnl=np.nan, sharpe=np.nan, pf=np.nan, maxdd=np.nan)
    std = x.std(ddof=1)
    equity = x.cumsum()
    dd = equity - equity.cummax().clip(lower=0)
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    return {
        "pnl": float(x.sum()),
        "sharpe": float(np.sqrt(252) * x.mean() / std) if std > 0 else np.nan,
        "pf": float(gains / losses) if losses > 0 else np.nan,
        "maxdd": float(dd.min()),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    columns = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join(lines)


def load_slots(symbol: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    path = ROOT / spec["path"]
    schema = pq.ParquetFile(path).schema.names
    columns = ["open"] + (["instrument_id"] if "instrument_id" in schema else [])
    bars = pd.read_parquet(path, columns=columns).sort_index()
    if bars.index.tz is None:
        raise ValueError(f"{symbol}: timezone-naive bars")
    ct = bars.index.tz_convert("America/Chicago")
    minute = pd.Index(ct.hour * 60 + ct.minute)
    wanted = {510, 540, 570, 870, 900}
    mask = minute.isin(wanted)
    small = bars.loc[mask].copy()
    small["session_date"] = pd.Index(ct[mask].date)
    small["minute"] = minute[mask]
    px = small.pivot_table(index="session_date", columns="minute", values="open", aggfunc="last")
    ids = None
    if "instrument_id" in small:
        ids = small.pivot_table(index="session_date", columns="minute", values="instrument_id", aggfunc="last")
    records = []
    complete_all = pd.Series(True, index=px.index)
    numeric_windows = dict(zip(WINDOWS, [(510, 540), (540, 570), (870, 900)]))
    for window, (start, end) in numeric_windows.items():
        valid = px.get(start, pd.Series(index=px.index, dtype=float)).notna() & px.get(end, pd.Series(index=px.index, dtype=float)).notna()
        if ids is not None and start in ids and end in ids:
            valid &= ids[start].eq(ids[end])
        complete_all &= valid
        entry = px.loc[valid, start]
        exit_ = px.loc[valid, end]
        frame = pd.DataFrame({
            "session_date": pd.to_datetime(entry.index),
            "instrument": symbol,
            "window": window,
            "entry_price": entry.to_numpy(),
            "exit_price": exit_.to_numpy(),
        })
        frame["ret"] = frame.exit_price / frame.entry_price - 1.0
        frame["usd_ret"] = (frame.exit_price - frame.entry_price) * spec["point_value"]
        records.append(frame)
    first = pd.Timestamp(px.index.min())
    last = pd.Timestamp(px.index.max())
    expected_weekdays = len(pd.bdate_range(first, last))
    coverage = float(complete_all.sum() / expected_weekdays) if expected_weekdays else np.nan
    audit = {
        "symbol": symbol,
        "asset_class": spec["asset_class"],
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "coverage_pct": 100 * coverage,
        "coverage_method": "complete_all_three_fixed_windows_over_weekdays",
        "common_period_coverage_pct": np.nan,
        "round_turn_cost_usd": spec["commission_rt"],
        "tick_size": spec["tick_size"],
        "tick_value_usd": spec["tick_value"],
        "point_value_usd": spec["point_value"],
        "selected_cross_section": spec["selected"],
        "selection_reason": spec["reason"],
        "source_path": spec["path"],
        "sha256": "not_computed_large_local_file",
    }
    return pd.concat(records, ignore_index=True), audit


def metadata_only_audit(symbol: str, spec: dict) -> dict:
    """Fast inventory for series deliberately excluded from the analysis."""
    path = ROOT / spec["path"]
    parquet = pq.ParquetFile(path)
    names = parquet.schema.names
    timestamp_name = "timestamp" if "timestamp" in names else "ts_event"
    timestamp_idx = names.index(timestamp_name)
    minima, maxima = [], []
    for i in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(i).column(timestamp_idx).statistics
        if stats and stats.has_min_max:
            minima.append(pd.Timestamp(stats.min)); maxima.append(pd.Timestamp(stats.max))
    first, last = min(minima), max(maxima)
    expected_weekdays = len(pd.bdate_range(first.tz_localize(None).normalize(), last.tz_localize(None).normalize()))
    # Conservative overall minute-density diagnostic for the near-23h Globex session.
    density = min(1.0, parquet.metadata.num_rows / max(expected_weekdays * 1380, 1))
    return {
        "symbol": symbol, "asset_class": spec["asset_class"], "first_date": str(first.date()),
        "last_date": str(last.date()), "coverage_pct": 100 * density,
        "coverage_method": "observed_minutes_over_1380x_weekdays_metadata_estimate",
        "common_period_coverage_pct": np.nan,
        "round_turn_cost_usd": spec["commission_rt"], "tick_size": spec["tick_size"],
        "tick_value_usd": spec["tick_value"], "point_value_usd": spec["point_value"],
        "selected_cross_section": False, "selection_reason": spec["reason"],
        "source_path": spec["path"], "sha256": "not_computed_large_local_file",
    }


def add_scores(panel: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (_, _), group in panel.groupby(["instrument", "window"], sort=False):
        group = group.sort_values("session_date").copy()
        for n in LOOKBACKS:
            mean = group.ret.rolling(n, min_periods=n).mean().shift(1)
            vol = group.ret.rolling(n, min_periods=n).std(ddof=1).shift(1)
            usd_vol = group.usd_ret.rolling(n, min_periods=n).std(ddof=1).shift(1)
            out = group.copy()
            out["lookback"] = n
            out["score"] = mean / vol.replace(0, np.nan)
            out["usd_vol"] = usd_vol
            out["score_lag2"] = group.ret.rolling(n, min_periods=n).mean().shift(2) / group.ret.rolling(n, min_periods=n).std(ddof=1).shift(2).replace(0, np.nan)
            parts.append(out)
    return pd.concat(parts, ignore_index=True)


def ts_row(frame: pd.DataFrame, instrument: str, window: str, n: int) -> dict:
    x = frame.dropna(subset=["score", "ret"])
    pos, neg = x[x.score > 0].ret, x[x.score < 0].ret
    return {
        "instrument": instrument,
        "window": window,
        "lookback": n,
        "predictive_corr": x.score.corr(x.ret),
        "mean_return_score_positive": pos.mean(),
        "mean_return_score_negative": neg.mean(),
        "positive_minus_negative_spread": pos.mean() - neg.mean(),
        "hac_t_spread": hac_spread_t(x),
        "directional_hit_rate": (np.sign(x.score) == np.sign(x.ret)).mean(),
        "n_observations": len(x),
    }


def cross_section_daily(scored: pd.DataFrame, instruments: list[str]) -> pd.DataFrame:
    use = scored[scored.instrument.isin(instruments)].dropna(subset=["score", "usd_vol"])
    keys = ["session_date", "window", "lookback"]
    complete = use.groupby(keys).instrument.transform("nunique").eq(len(instruments))
    use = use[complete].copy()
    use["rank"] = use.groupby(keys).score.rank(method="first")
    leg_n = math.ceil(len(instruments) / 3)
    use["side"] = np.select(
        [use["rank"] <= leg_n, use["rank"] > len(instruments) - leg_n],
        [-1.0, 1.0], default=0.0,
    )
    selected = use[use.side != 0].copy()
    selected["inv_vol"] = 1.0 / selected.usd_vol.replace(0, np.nan)
    selected["contract_equiv"] = selected.inv_vol / selected.groupby(keys + ["side"]).inv_vol.transform("mean")
    selected["weight"] = selected.side * selected.contract_equiv
    selected["gross_pnl"] = selected.weight * selected.usd_ret
    selected["commission"] = selected.contract_equiv * selected.instrument.map({k: v["commission_rt"] for k, v in SPECS.items()})
    selected["slippage_1tick"] = selected.contract_equiv * selected.instrument.map({k: 2 * v["tick_value"] for k, v in SPECS.items()})
    return selected[["session_date", "window", "lookback", "instrument", "side", "contract_equiv", "gross_pnl", "commission", "slippage_1tick"]]


def summarize_cross(legs: pd.DataFrame) -> pd.DataFrame:
    daily = legs.groupby(["session_date", "window", "lookback"], as_index=False).agg(
        gross_pnl=("gross_pnl", "sum"), commission=("commission", "sum"),
        slippage_1tick=("slippage_1tick", "sum"), turnover_contract_equiv=("contract_equiv", "sum"),
        positions=("instrument", "size"),
    )
    daily["commission_only"] = daily.gross_pnl - daily.commission
    daily["commission_plus_1tick_side"] = daily.gross_pnl - daily.commission - daily.slippage_1tick
    daily["commission_plus_2ticks_side"] = daily.gross_pnl - daily.commission - 2 * daily.slippage_1tick
    all_windows = daily.groupby(["session_date", "lookback"], as_index=False).agg({
        "gross_pnl": "sum", "commission": "sum", "slippage_1tick": "sum",
        "turnover_contract_equiv": "sum", "positions": "sum", "commission_only": "sum",
        "commission_plus_1tick_side": "sum", "commission_plus_2ticks_side": "sum",
    })
    all_windows["window"] = "ALL_WINDOWS"
    daily_all = pd.concat([daily, all_windows], ignore_index=True)
    rows = []
    scenarios = ["gross_pnl", "commission_only", "commission_plus_1tick_side", "commission_plus_2ticks_side"]
    for (window, n), group in daily_all.groupby(["window", "lookback"]):
        for scenario in scenarios:
            metrics = performance(group[scenario])
            rows.append({
                "window": window, "lookback": n, "cost_scenario": scenario,
                "pnl_usd": metrics["pnl"], "sharpe": metrics["sharpe"], "profit_factor": metrics["pf"],
                "max_drawdown_usd": metrics["maxdd"], "turnover_contract_equiv": group.turnover_contract_equiv.sum(),
                "average_positions": group.positions.mean(), "n_days": group.session_date.nunique(),
            })
    return pd.DataFrame(rows), daily_all


def pooled_ts_rows(scored: pd.DataFrame) -> list[dict]:
    rows = []
    for (window, n), group in scored.groupby(["window", "lookback"]):
        rows.append(ts_row(group, "POOLED", window, n))
    return rows


def temporal_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fee_map = {k: v["commission_rt"] + 2 * v["tick_value"] for k, v in SPECS.items()}
    for (window, n), group in scored.dropna(subset=["score"]).groupby(["window", "lookback"]):
        dates = np.array(sorted(group.session_date.unique()))
        blocks = np.array_split(dates, 3)
        slices = [(f"block_{i+1}", set(block)) for i, block in enumerate(blocks)]
        slices += [(str(year), set(g.session_date.unique())) for year, g in group.groupby(group.session_date.dt.year)]
        for label, date_set in slices:
            x = group[group.session_date.isin(date_set)].copy()
            x["gross_directional"] = np.sign(x.score) * x.usd_ret
            x["net_directional"] = x.gross_directional - x.instrument.map(fee_map)
            gross = x.groupby("session_date").gross_directional.sum()
            net = x.groupby("session_date").net_directional.sum()
            pos, neg = x[x.score > 0].ret, x[x.score < 0].ret
            rows.append({
                "window": window, "lookback": n, "period_type": "third" if label.startswith("block") else "year",
                "period": label, "start_date": x.session_date.min(), "end_date": x.session_date.max(),
                "predictive_corr": x.score.corr(x.ret), "positive_minus_negative_spread": pos.mean() - neg.mean(),
                "gross_sharpe": performance(gross)["sharpe"], "net_1tick_sharpe": performance(net)["sharpe"],
                "n_observations": len(x),
            })
    return pd.DataFrame(rows)


def placebo_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in LOOKBACKS:
        for window, group in scored[scored.lookback == n].groupby("window"):
            x = group.dropna(subset=["score", "ret"])
            rows.append({"placebo": "same_slot_reference", "source_window": window, "target_window": window, "lookback": n, "predictive_corr": x.score.corr(x.ret), "directional_mean_return": (np.sign(x.score) * x.ret).mean(), "hac_t_directional_mean": hac_mean_t(np.sign(x.score) * x.ret), "n_observations": len(x)})
            lag = group.dropna(subset=["score_lag2", "ret"])
            rows.append({"placebo": "extra_one_day_lag", "source_window": window, "target_window": window, "lookback": n, "predictive_corr": lag.score_lag2.corr(lag.ret), "directional_mean_return": (np.sign(lag.score_lag2) * lag.ret).mean(), "hac_t_directional_mean": hac_mean_t(np.sign(lag.score_lag2) * lag.ret), "n_observations": len(lag)})
        for source, target in [("A_0830_0900_CT", "B_0900_0930_CT"), ("B_0900_0930_CT", "C_1430_1500_CT")]:
            s = scored[(scored.lookback == n) & (scored.window == source)][["session_date", "instrument", "score"]]
            t = scored[(scored.lookback == n) & (scored.window == target)][["session_date", "instrument", "ret"]]
            x = s.merge(t, on=["session_date", "instrument"]).dropna()
            rows.append({"placebo": "previous_slot_predicts_other_slot", "source_window": source, "target_window": target, "lookback": n, "predictive_corr": x.score.corr(x.ret), "directional_mean_return": (np.sign(x.score) * x.ret).mean(), "hac_t_directional_mean": hac_mean_t(np.sign(x.score) * x.ret), "n_observations": len(x)})
    return pd.DataFrame(rows)


def cost_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (instrument, window, n), x in scored.dropna(subset=["score"]).groupby(["instrument", "window", "lookback"]):
        spec = SPECS[instrument]
        gross = np.sign(x.score) * x.usd_ret
        notional = x.entry_price.abs() * spec["point_value"]
        for scenario, cost in [
            ("commission_only", spec["commission_rt"]),
            ("commission_plus_1tick_side", spec["commission_rt"] + 2 * spec["tick_value"]),
            ("commission_plus_2ticks_side", spec["commission_rt"] + 4 * spec["tick_value"]),
        ]:
            rows.append({
                "instrument": instrument, "window": window, "lookback": n, "scenario": scenario,
                "mean_gross_pnl_per_trade_usd": gross.mean(), "mean_cost_per_trade_usd": cost,
                "mean_net_pnl_per_trade_usd": gross.mean() - cost,
                "mean_cost_bps_of_notional": float((10000 * cost / notional).mean()),
                "break_even_cost_usd": gross.mean(), "n_trades": len(x),
            })
    return pd.DataFrame(rows)


def concentration_rows(legs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = [k for k, v in SPECS.items() if v["selected"]]
    reruns = {}
    for omitted in selected:
        rerun = cross_section_daily(SCORED, [x for x in selected if x != omitted])
        rerun["net"] = rerun.gross_pnl - rerun.commission - rerun.slippage_1tick
        reruns[omitted] = rerun
    for (window, n), base in legs.groupby(["window", "lookback"]):
        base = base.copy()
        base["net"] = base.gross_pnl - base.commission - base.slippage_1tick
        total = base.net.sum()
        contribution = base.groupby("instrument").net.sum().sort_values(ascending=False)
        for instrument, pnl in contribution.items():
            rows.append({"window": window, "lookback": n, "analysis": "original_contribution", "instrument": instrument, "net_pnl_usd": pnl, "contribution_pct": 100 * pnl / total if total else np.nan, "leave_one_out_sharpe": np.nan})
        # Re-rank the cross-section after each omission, rather than merely deleting a leg.
        for omitted in selected:
            rerun = reruns[omitted]
            rerun = rerun[(rerun.window == window) & (rerun.lookback == n)]
            daily = rerun.groupby("session_date").net.sum()
            rows.append({"window": window, "lookback": n, "analysis": "leave_one_out", "instrument": omitted, "net_pnl_usd": daily.sum(), "contribution_pct": np.nan, "leave_one_out_sharpe": performance(daily)["sharpe"]})
        best = contribution.index[0]
        rerun = reruns[best]
        rerun = rerun[(rerun.window == window) & (rerun.lookback == n)]
        daily = rerun.groupby("session_date").net.sum()
        rows.append({"window": window, "lookback": n, "analysis": "without_best_instrument", "instrument": best, "net_pnl_usd": daily.sum(), "contribution_pct": np.nan, "leave_one_out_sharpe": performance(daily)["sharpe"]})
    return pd.DataFrame(rows)


def verdict_and_report(universe, ts, cross, costs, temporal, placebo, concentration, common_start, common_end) -> tuple[str, str]:
    all_net = cross[(cross.window == "ALL_WINDOWS") & (cross.cost_scenario == "commission_plus_1tick_side")].set_index("lookback")
    n20_2tick = cross[(cross.window == "ALL_WINDOWS") & (cross.lookback == 20) & (cross.cost_scenario == "commission_plus_2ticks_side")].iloc[0]
    pool = ts[ts.instrument == "POOLED"].groupby("lookback").agg(predictive_corr=("predictive_corr", "mean"))
    thirds = temporal[(temporal.lookback == 20) & (temporal.period_type == "third")]
    positive_thirds = int((thirds.net_1tick_sharpe > 0).sum())
    contrib20 = concentration[(concentration.lookback == 20) & (concentration.analysis == "original_contribution")]
    positive_instruments = int((contrib20.groupby("instrument").net_pnl_usd.sum() > 0).sum())
    same = placebo[(placebo.lookback == 20) & (placebo.placebo == "same_slot_reference")].directional_mean_return.mean()
    other = placebo[(placebo.lookback == 20) & (placebo.placebo != "same_slot_reference")].directional_mean_return.mean()
    promising = bool((all_net.pnl_usd > 0).all() and n20_2tick.pnl_usd > 0 and (pool.predictive_corr > 0).all() and positive_thirds >= 2 and positive_instruments >= 3 and same > other)
    visible = bool((all_net.pnl_usd > 0).any() or (pool.predictive_corr > 0).sum() >= 2)
    verdict = "PROMISING_FOR_VALIDATION" if promising else ("WEAK_EDGE" if visible else "REJECTED")
    main = cross[(cross.window == "ALL_WINDOWS") & (cross.lookback == 20)][["cost_scenario", "pnl_usd", "sharpe", "profit_factor", "max_drawdown_usd"]]
    ts_pool = ts[ts.instrument == "POOLED"]
    report = f"""# Périodicité intraday des rendements futures — Phase 1

## Verdict : {verdict}

La campagne est un screening figé : fenêtres 08:30–09:00, 09:00–09:30 et 14:30–15:00 CT; lookbacks 10/20/40; aucune recherche horaire, aucun Topstep, aucune modification des six sleeves ou de Gotobi.

## Univers et causalité

Le cross-section utilise NQ, ZN, 6E, MGC et MCL, un représentant par classe. Sa période commune est {common_start.date()}–{common_end.date()}. Le rendement d'un créneau est `open(fin)/open(début)-1`; le score du jour t utilise uniquement les N observations antérieures du même créneau, avec `shift(1)`. Les bascules de contrat observables à l'intérieur d'un créneau sont exclues.

Les mini/micro redondants et les autres séries locales figurent dans `universe_audit.csv`. Pour les cinq contrats retenus, la couverture est le pourcentage de jours ouvrés disposant des cinq prix-frontières nécessaires aux trois créneaux; `common_period_coverage_pct` applique cette mesure à 2021–2026. Pour les doublons exclus, `coverage_pct` est seulement une densité Globex issue des métadonnées, explicitement identifiée par `coverage_method`. Les fermetures de marché restent dans le dénominateur : ces mesures sont conservatrices.

## Diagnostic time-series

{markdown_table(ts_pool)}

`hac_t_spread` teste la différence rendement moyen score positif moins score négatif avec erreurs HAC. Les lignes `POOLED` agrègent les observations disponibles sans choisir un instrument ou un créneau gagnant.

## Cross-section canonique — toutes fenêtres, N=20

{markdown_table(main)}

Chaque date/créneau, les deux meilleurs scores sont achetés et les deux plus faibles vendus. Les poids inverse-volatilité sont normalisés à une moyenne d'un contrat équivalent par position, séparément sur chaque jambe. Il s'agit d'un diagnostic à risque comparable avec contrats fractionnaires, pas d'un mapping exécutable Topstep.

## Coûts

Les coûts round-turn TopstepX retenus sont ceux publiés le 28 juillet 2026 : NQ 3,78 USD, ZN 2,62 USD, 6E 4,22 USD, MGC 1,92 USD et MCL 1,52 USD. Le scénario central ajoute un tick défavorable à l'entrée et à la sortie; le stress en ajoute deux par côté. `cost_analysis.csv` compare l'espérance brute par trade au coût et au break-even pour chaque instrument.

Sources : [TopstepX commissions](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees), [NQ](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html), [ZN](https://www.cmegroup.com/education/whitepapers/insights-into-us-treasury-futures), [6E](https://www.cmegroup.com/markets/fx/g10/euro-fx/specs), [MGC](https://www.cmegroup.com/education/courses/event-contracts-underlying-markets/product-gold), [MCL](https://www.cmegroup.com/content/dam/cmegroup/market-regulation/rule-filings/2021/6/21-154.pdf).

## Stabilité, placebos et concentration

- Blocs chronologiques N=20 à Sharpe net positif : {positive_thirds}/{len(thirds)} lignes créneau-bloc.
- Instruments à contribution nette positive, agrégés sur les trois créneaux pour N=20 : {positive_instruments}/5.
- Rendement directionnel moyen même créneau N=20 : {same:.8f}; moyenne des placebos : {other:.8f}.
- Les détails par tiers et année sont dans `temporal_stability.csv`; les deux placebos dans `placebo_results.csv`; les contributions et reruns leave-one-out dans `concentration_analysis.csv`.

## Décision

Le verdict est appliqué au mécanisme complet, sans sélectionner le meilleur créneau ou lookback. `PROMISING_FOR_VALIDATION` exige : PnL net +1 tick positif pour 10/20/40 sur l'ensemble des fenêtres, N=20 encore positif à deux ticks, corrélation prédictive poolée positive sur les trois lookbacks, stabilité dans au moins deux tiers, contributions positives d'au moins trois instruments et effet même-créneau supérieur aux placebos. À défaut, un signal seulement partiel est `WEAK_EDGE`; absence d'effet robuste ou destruction par les coûts est `REJECTED`.

`orthogonality.csv` n'est produit que si une règle nette positive mérite ce diagnostic. Aucun paramètre ne sera retuné dans cette Phase 1.
"""
    return verdict, report


def main() -> None:
    global SCORED
    OUT.mkdir(parents=True, exist_ok=True)
    slot_frames, audits = [], []
    for symbol, spec in SPECS.items():
        if spec["selected"]:
            frame, audit = load_slots(symbol, spec)
            slot_frames.append(frame)
        else:
            audit = metadata_only_audit(symbol, spec)
        audits.append(audit)
    universe = pd.DataFrame(audits)

    panel = pd.concat(slot_frames, ignore_index=True)
    selected = [k for k, v in SPECS.items() if v["selected"]]
    ranges = panel.groupby("instrument").session_date.agg(["min", "max"])
    common_start, common_end = ranges["min"].max(), ranges["max"].min()
    panel = panel[(panel.session_date >= common_start) & (panel.session_date <= common_end)].copy()
    expected_common = len(pd.bdate_range(common_start, common_end))
    complete_common = panel.groupby(["instrument", "session_date"]).window.nunique().eq(len(WINDOWS))
    common_coverage = complete_common.groupby(level=0).sum().div(expected_common).mul(100)
    universe.loc[universe.selected_cross_section, "common_period_coverage_pct"] = universe.loc[universe.selected_cross_section, "symbol"].map(common_coverage)
    universe.to_csv(OUT / "universe_audit.csv", index=False)
    SCORED = add_scores(panel)

    ts_rows = []
    for (instrument, window, n), group in SCORED.groupby(["instrument", "window", "lookback"]):
        ts_rows.append(ts_row(group, instrument, window, n))
    ts_rows += pooled_ts_rows(SCORED)
    ts = pd.DataFrame(ts_rows)
    ts.to_csv(OUT / "timeseries_results.csv", index=False)

    legs = cross_section_daily(SCORED, selected)
    cross, daily = summarize_cross(legs)
    cross.to_csv(OUT / "cross_section_results.csv", index=False)
    costs = cost_rows(SCORED); costs.to_csv(OUT / "cost_analysis.csv", index=False)
    temporal = temporal_rows(SCORED); temporal.to_csv(OUT / "temporal_stability.csv", index=False)
    placebo = placebo_rows(SCORED); placebo.to_csv(OUT / "placebo_results.csv", index=False)
    concentration = concentration_rows(legs); concentration.to_csv(OUT / "concentration_analysis.csv", index=False)

    verdict, report = verdict_and_report(universe, ts, cross, costs, temporal, placebo, concentration, common_start, common_end)
    (OUT / "PHASE1_REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "study": "intraday_return_periodicity_phase1", "as_of": "2026-10-03", "verdict": verdict,
        "windows_ct": WINDOWS, "lookbacks": LOOKBACKS, "selected_instruments": selected,
        "common_start": str(common_start.date()), "common_end": str(common_end.date()),
        "cross_section_weighting": "inverse trailing same-slot USD volatility; mean one contract equivalent per selected position per side",
        "primary_cost": "current TopstepX round-turn fee plus one adverse tick per side",
        "frozen_sleeves_modified": False, "gotobi_modified": False, "topstep_simulation_run": False,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(cross[cross.window == "ALL_WINDOWS"].to_string(index=False))


if __name__ == "__main__":
    main()
