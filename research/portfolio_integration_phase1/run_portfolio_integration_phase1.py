"""Reproduce and diagnose the six frozen MNQ/MGC sleeves without retuning them."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = OUTPUT_DIR / "portfolio_integration_phase1_config.json"
TRADES_DIR = OUTPUT_DIR / "sleeve_trades"
FIGURES_DIR = OUTPUT_DIR / "figures"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from research.baseline_backtests.baseline_engine import run_baseline_backtest  # noqa: E402
from research.locked_four_sleeve_portfolio.run_erc_portfolio import (  # noqa: E402
    risk_details,
    solve_erc,
)
from research.locked_four_sleeve_portfolio.run_locked_four_sleeve_portfolio import (  # noqa: E402
    load_and_validate_config,
    run_alpha_sleeves,
    validate_locks,
)
from research.mgc_baseline_backtests.run_mgc_baselines import prepare_mgc_data  # noqa: E402
from src.analytics.mnq_orb_topstep_50k_simulation import (  # noqa: E402
    TopstepCampaignSpec,
    simulate_topstep_path,
)


SLEEVES = [
    "mnq_orb_vwap_atr",
    "mnq_volatility_breakout",
    "mnq_compression_breakout",
    "mnq_atr_trend_flip",
    "mgc_volatility_breakout",
    "mgc_gap_continuation",
]
INSTRUMENT = {s: ("MNQ" if s.startswith("mnq_") else "MGC") for s in SLEEVES}
SOURCE_IDS = {
    "mnq_orb_vwap_atr": "orb_vwap_atr",
    "mnq_volatility_breakout": "volatility_breakout",
    "mnq_compression_breakout": "compression_breakout",
    "mnq_atr_trend_flip": "atr_trend_flip",
    "mgc_volatility_breakout": "volatility_breakout",
    "mgc_gap_continuation": "gap_continuation",
}
INITIAL_CAPITAL = 50_000.0
ANNUALIZATION = 252


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_verify_phase_config() -> tuple[dict[str, Any], dict[str, str]]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not cfg.get("locked_before_execution") or [row["id"] for row in cfg["sleeves"]] != SLEEVES:
        raise RuntimeError("Phase-1 config is not locked or sleeve ordering changed")
    hashes: dict[str, str] = {}
    for key, source in cfg["source_locks"].items():
        path = REPO_ROOT / source["path"]
        actual = sha256(path)
        if actual != source["sha256"]:
            raise RuntimeError(f"Locked source hash mismatch: {key}")
        hashes[key] = actual
    return cfg, hashes


def normalize_trades(frame: pd.DataFrame, sleeve: str) -> pd.DataFrame:
    out = frame.copy()
    out["sleeve"] = sleeve
    out["instrument"] = INSTRUMENT[sleeve]
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.date
    out["entry_ts_utc"] = pd.to_datetime(out["entry_time"], utc=True)
    out["exit_ts_utc"] = pd.to_datetime(out["exit_time"], utc=True)
    out["quantity"] = pd.to_numeric(out["quantity"], errors="raise").astype(float)
    if "position" in out:
        out["signed_quantity"] = pd.to_numeric(out["position"], errors="raise").astype(float) * out["quantity"]
    else:
        direction = out["direction"].astype(str).str.lower().map({"long": 1.0, "short": -1.0})
        if direction.isna().any():
            raise RuntimeError(f"Unrecognized trade direction in {sleeve}")
        out["signed_quantity"] = direction * out["quantity"]
    out["net_pnl_usd"] = pd.to_numeric(out["net_pnl_usd"], errors="raise").astype(float)
    if not (out["exit_ts_utc"] >= out["entry_ts_utc"]).all():
        raise RuntimeError(f"Negative holding interval in {sleeve}")
    return out


def reproduce_sleeves(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    start = date.fromisoformat(cfg["warmup_start"])
    end = date.fromisoformat(cfg["primary_common_evaluation"]["end"])
    oos_start = date.fromisoformat(cfg["primary_common_evaluation"]["start"])

    mnq_cfg, _, mm_locks, _ = load_and_validate_config()
    validate_locks(mnq_cfg, mm_locks)
    mnq_runs, mnq_trades_raw, mnq_checks = run_alpha_sleeves(mnq_cfg)
    orb_path = REPO_ROOT / mnq_cfg["sleeves"]["orb_vwap_atr"]["source"]
    mnq_trades_raw["orb_vwap_atr"] = pd.read_csv(orb_path)

    trades: dict[str, pd.DataFrame] = {}
    daily: dict[str, pd.DataFrame] = {}
    for sleeve in SLEEVES[:4]:
        source = SOURCE_IDS[sleeve]
        normalized = normalize_trades(mnq_trades_raw[source], sleeve)
        normalized = normalized[normalized["session_date"].between(oos_start, end)].copy()
        trades[sleeve] = normalized
        run_daily = mnq_runs[source].daily[["session_date", "daily_pnl_usd"]].copy() if source != "orb_vwap_atr" else None
        if run_daily is None:
            calendar = mnq_runs["compression_breakout"].daily[["session_date"]].copy()
            pnl_map = normalized.groupby("session_date")["net_pnl_usd"].sum()
            run_daily = calendar.assign(daily_pnl_usd=calendar["session_date"].map(pnl_map).fillna(0.0))
        run_daily["session_date"] = pd.to_datetime(run_daily["session_date"]).dt.date
        daily[sleeve] = run_daily[run_daily["session_date"].between(oos_start, end)].copy()

    mgc_data_path = REPO_ROOT / cfg["source_locks"]["mgc_minute_data"]["path"]
    mgc_data, _ = prepare_mgc_data(mgc_data_path, start, end)
    mgc_checks: dict[str, Any] = {}
    for sleeve, lock_key in (
        ("mgc_volatility_breakout", "mgc_volatility_spec"),
        ("mgc_gap_continuation", "mgc_gap_spec"),
    ):
        spec = json.loads((REPO_ROOT / cfg["source_locks"][lock_key]["path"]).read_text(encoding="utf-8"))
        run = run_baseline_backtest(spec, mgc_data, evaluation_start=start, evaluation_end=end)
        violations = {k: v for k, v in run.checks.items() if k.endswith("violations") and v}
        if violations:
            raise RuntimeError(f"Implementation violations in {sleeve}: {violations}")
        normalized = normalize_trades(run.trades, sleeve)
        trades[sleeve] = normalized[normalized["session_date"].between(oos_start, end)].copy()
        run_daily = run.daily[["session_date", "daily_pnl_usd"]].copy()
        run_daily["session_date"] = pd.to_datetime(run_daily["session_date"]).dt.date
        daily[sleeve] = run_daily[run_daily["session_date"].between(oos_start, end)].copy()
        mgc_checks[sleeve] = run.checks

    return trades, daily, {"mnq": mnq_checks, "mgc": mgc_checks}


def build_daily_matrix(daily: dict[str, pd.DataFrame], trades: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendars = [set(frame["session_date"]) for frame in daily.values()]
    common = sorted(set.intersection(*calendars))
    if not common:
        raise RuntimeError("No common trading calendar")
    pnl = pd.DataFrame(index=pd.Index(common, name="session_date"))
    trade_counts = pd.DataFrame(index=pnl.index)
    for sleeve in SLEEVES:
        mapped = daily[sleeve].set_index("session_date")["daily_pnl_usd"].astype(float)
        pnl[sleeve] = mapped.reindex(pnl.index).fillna(0.0)
        counts = trades[sleeve].groupby("session_date").size()
        trade_counts[sleeve] = counts.reindex(pnl.index).fillna(0).astype(int)
        reproduced = trades[sleeve].groupby("session_date")["net_pnl_usd"].sum().reindex(pnl.index).fillna(0.0)
        if not np.allclose(pnl[sleeve], reproduced, atol=1e-7):
            delta = float((pnl[sleeve] - reproduced).abs().max())
            raise RuntimeError(f"Trade/daily PnL mismatch for {sleeve}: {delta}")
    return pnl, trade_counts


def build_timestamp_dataset(trades: dict[str, pd.DataFrame], common_dates: list[date]) -> pd.DataFrame:
    timestamp_parts = []
    for day in common_dates:
        start = pd.Timestamp(f"{day} 08:20", tz="America/New_York").tz_convert("UTC")
        end = pd.Timestamp(f"{day} 16:08", tz="America/New_York").tz_convert("UTC")
        timestamp_parts.append(pd.date_range(start, end, freq="1min"))
    index = timestamp_parts[0].append(timestamp_parts[1:]) if len(timestamp_parts) > 1 else timestamp_parts[0]
    frame = pd.DataFrame(index=index)
    frame.index.name = "timestamp_utc"

    for sleeve in SLEEVES:
        position = pd.Series(0.0, index=index)
        entries = pd.Series(0, index=index, dtype=int)
        exits = pd.Series(0, index=index, dtype=int)
        realized = pd.Series(0.0, index=index)
        for row in trades[sleeve].itertuples(index=False):
            entry = pd.Timestamp(row.entry_ts_utc).floor("min")
            exit_ = pd.Timestamp(row.exit_ts_utc).floor("min")
            if entry in position.index:
                entries.loc[entry] += 1
            if exit_ in position.index:
                exits.loc[exit_] += 1
                realized.loc[exit_] += float(row.net_pnl_usd)
            active = (position.index >= entry) & (position.index < exit_)
            position.loc[active] += float(row.signed_quantity)
        frame[f"position__{sleeve}"] = position
        frame[f"entries__{sleeve}"] = entries
        frame[f"exits__{sleeve}"] = exits
        frame[f"realized_pnl__{sleeve}"] = realized

    mnq_cols = [f"position__{s}" for s in SLEEVES if INSTRUMENT[s] == "MNQ"]
    mgc_cols = [f"position__{s}" for s in SLEEVES if INSTRUMENT[s] == "MGC"]
    position_cols = [f"position__{s}" for s in SLEEVES]
    entry_cols = [f"entries__{s}" for s in SLEEVES]
    frame["mnq_net_contracts"] = frame[mnq_cols].sum(axis=1)
    frame["mnq_gross_contracts"] = frame[mnq_cols].abs().sum(axis=1)
    frame["mgc_net_contracts"] = frame[mgc_cols].sum(axis=1)
    frame["mgc_gross_contracts"] = frame[mgc_cols].abs().sum(axis=1)
    frame["active_sleeves"] = frame[position_cols].ne(0).sum(axis=1)
    frame["active_mnq_sleeves"] = frame[mnq_cols].ne(0).sum(axis=1)
    frame["simultaneous_trade_entries"] = frame[entry_cols].sum(axis=1)
    mnq_signs = np.sign(frame[mnq_cols])
    frame["mnq_same_direction_collapse"] = (
        frame["active_mnq_sleeves"].ge(2)
        & ((mnq_signs.ge(0).all(axis=1)) | (mnq_signs.le(0).all(axis=1)))
    )
    frame["total_realized_pnl_usd"] = frame[[f"realized_pnl__{s}" for s in SLEEVES]].sum(axis=1)
    return frame.reset_index()


def pairwise_overlap(trades: dict[str, pd.DataFrame], timestamp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = timestamp.set_index("timestamp_utc")
    for i, left in enumerate(SLEEVES):
        left_days = set(trades[left]["session_date"])
        left_active = indexed[f"position__{left}"].ne(0)
        for right in SLEEVES[i + 1 :]:
            right_days = set(trades[right]["session_date"])
            union = left_days | right_days
            both_minutes = left_active & indexed[f"position__{right}"].ne(0)
            rows.append({
                "sleeve_left": left,
                "sleeve_right": right,
                "overlap_trade_days": len(left_days & right_days),
                "trade_day_jaccard": len(left_days & right_days) / len(union) if union else np.nan,
                "simultaneous_position_minutes": int(both_minutes.sum()),
                "same_direction_minutes": int((both_minutes & (indexed[f"position__{left}"] * indexed[f"position__{right}"]).gt(0)).sum()),
                "opposite_direction_minutes": int((both_minutes & (indexed[f"position__{left}"] * indexed[f"position__{right}"]).lt(0)).sum()),
            })
    return pd.DataFrame(rows)


def drawdown_frame(pnl: pd.DataFrame) -> pd.DataFrame:
    equity = INITIAL_CAPITAL + pnl.cumsum()
    return equity - equity.cummax().clip(lower=INITIAL_CAPITAL)


def pairwise_drawdown_overlap(drawdown: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, left in enumerate(SLEEVES):
        a = drawdown[left].lt(0)
        for right in SLEEVES[i + 1 :]:
            b = drawdown[right].lt(0)
            union = a | b
            rows.append({
                "sleeve_left": left,
                "sleeve_right": right,
                "both_underwater_days": int((a & b).sum()),
                "underwater_jaccard": float((a & b).sum() / union.sum()) if union.any() else np.nan,
                "conditional_right_underwater_given_left": float((a & b).sum() / a.sum()) if a.any() else np.nan,
            })
    return pd.DataFrame(rows)


def rolling_correlations(pnl: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    for i, left in enumerate(SLEEVES):
        for right in SLEEVES[i + 1 :]:
            values = pnl[left].rolling(window, min_periods=window // 2).corr(pnl[right])
            rows.extend({"session_date": day, "sleeve_left": left, "sleeve_right": right, "pearson_correlation": value}
                        for day, value in values.dropna().items())
    return pd.DataFrame(rows)


def dependence_outputs(pnl: pd.DataFrame, trades: dict[str, pd.DataFrame], timestamp: pd.DataFrame, window: int) -> dict[str, pd.DataFrame]:
    pearson = pnl.corr(method="pearson")
    spearman = pnl.corr(method="spearman")
    downside = pnl.clip(upper=0).corr(method="pearson")
    equal_loss_days = pnl.mean(axis=1).lt(0)
    losing = pnl.loc[equal_loss_days].corr(method="pearson")
    dd = drawdown_frame(pnl)
    overlap = pairwise_overlap(trades, timestamp)
    dd_overlap = pairwise_drawdown_overlap(dd)
    rolling = rolling_correlations(pnl, window)

    yearly_rows = []
    date_index = pd.to_datetime(pnl.index)
    for year in sorted(date_index.year.unique()):
        part = pnl.loc[date_index.year == year]
        corr = part.corr()
        for i, left in enumerate(SLEEVES):
            for right in SLEEVES[i + 1 :]:
                yearly_rows.append({"year": int(year), "sleeve_left": left, "sleeve_right": right,
                                    "pearson_correlation": float(corr.loc[left, right]), "sessions": len(part)})

    loss_rows = []
    equal_pnl = pnl.mean(axis=1)
    loss_dates = equal_pnl.lt(0)
    total_loss = float(-equal_pnl.loc[loss_dates].sum())
    for sleeve in SLEEVES:
        contribution = pnl.loc[loss_dates, sleeve] / len(SLEEVES)
        loss_rows.append({
            "sleeve": sleeve,
            "signed_pnl_on_portfolio_losing_days_usd": float(contribution.sum()),
            "negative_pnl_only_usd": float(contribution.clip(upper=0).sum()),
            "share_of_gross_negative_contributions": np.nan,
            "portfolio_net_loss_share": float(-contribution.sum() / total_loss) if total_loss > 0 else np.nan,
        })
    loss_contribution = pd.DataFrame(loss_rows)
    gross_negative = float(-loss_contribution["negative_pnl_only_usd"].sum())
    loss_contribution["share_of_gross_negative_contributions"] = -loss_contribution["negative_pnl_only_usd"] / gross_negative

    return {
        "pearson": pearson,
        "spearman": spearman,
        "downside": downside,
        "losing_days": losing,
        "drawdown": dd,
        "trade_overlap": overlap,
        "drawdown_overlap": dd_overlap,
        "rolling": rolling,
        "yearly": pd.DataFrame(yearly_rows),
        "loss_contribution": loss_contribution,
    }


def connected_clusters(correlation: pd.DataFrame, threshold: float) -> list[list[str]]:
    remaining = set(SLEEVES)
    clusters: list[list[str]] = []
    while remaining:
        seed = sorted(remaining)[0]
        component = {seed}
        frontier = [seed]
        while frontier:
            node = frontier.pop()
            neighbors = {other for other in remaining if other != node and correlation.loc[node, other] >= threshold}
            for neighbor in neighbors - component:
                component.add(neighbor)
                frontier.append(neighbor)
        remaining -= component
        clusters.append(sorted(component))
    return sorted(clusters, key=lambda values: (-len(values), values))


def allocation_weights(pnl: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns = pnl / INITIAL_CAPITAL
    covariance = returns.cov(ddof=1).to_numpy(float)
    equal = np.full(len(SLEEVES), 1.0 / len(SLEEVES))
    vol = np.sqrt(np.diag(covariance))
    inverse = (1.0 / vol) / (1.0 / vol).sum()
    erc, iterations, error = solve_erc(covariance)
    weights = pd.DataFrame({
        "sleeve": SLEEVES,
        "equal_sleeve_weight": equal,
        "inverse_volatility": inverse,
        "equal_risk_contribution": erc,
    })
    return weights, {"erc_iterations": iterations, "erc_max_budget_error": error}


def portfolio_metrics(name: str, daily_pnl: pd.Series) -> dict[str, Any]:
    equity = INITIAL_CAPITAL + daily_pnl.cumsum()
    prior = equity.shift(1).fillna(INITIAL_CAPITAL)
    returns = daily_pnl / prior
    std = float(returns.std(ddof=1))
    peak = equity.cummax().clip(lower=INITIAL_CAPITAL)
    dd = equity - peak
    gains = float(daily_pnl.clip(lower=0).sum())
    losses = float(-daily_pnl.clip(upper=0).sum())
    return {
        "allocation": name,
        "start_date": daily_pnl.index[0],
        "end_date": daily_pnl.index[-1],
        "sessions": len(daily_pnl),
        "net_pnl_usd": float(daily_pnl.sum()),
        "sharpe_annualized": float(returns.mean() / std * math.sqrt(ANNUALIZATION)) if std > 0 else np.nan,
        "daily_profit_factor": gains / losses if losses > 0 else np.nan,
        "max_drawdown_usd": float(dd.min()),
        "worst_day_usd": float(daily_pnl.min()),
        "annualized_volatility_usd": float(daily_pnl.std(ddof=1) * math.sqrt(ANNUALIZATION)),
    }


def sleeve_yearly_results(pnl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = pd.to_datetime(pnl.index).year
    for sleeve in SLEEVES:
        for year in sorted(years.unique()):
            part = pnl.loc[years == year, sleeve]
            metrics = portfolio_metrics(sleeve, part)
            rows.append({"year": int(year), "sleeve": sleeve, **{k: v for k, v in metrics.items() if k != "allocation"}})
    return pd.DataFrame(rows)


def build_allocations(pnl: pd.DataFrame, weights: pd.DataFrame, timestamp: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cov = (pnl / INITIAL_CAPITAL).cov(ddof=1).to_numpy(float)
    weight_map = weights.set_index("sleeve")
    portfolio = pd.DataFrame(index=pnl.index)
    metrics_rows, yearly_rows, risk_rows, instrument_rows, concentration_rows = [], [], [], [], []
    position_frame = timestamp.set_index("timestamp_utc")
    for allocation in weights.columns[1:]:
        w = weight_map[allocation].reindex(SLEEVES).to_numpy(float)
        components = pnl.mul(w, axis=1)
        portfolio[f"daily_pnl__{allocation}"] = components.sum(axis=1)
        portfolio[f"equity__{allocation}"] = INITIAL_CAPITAL + components.sum(axis=1).cumsum()
        for sleeve in SLEEVES:
            portfolio[f"contribution__{allocation}__{sleeve}"] = components[sleeve]
        metrics_rows.append(portfolio_metrics(allocation, components.sum(axis=1)))
        years = pd.to_datetime(pnl.index).year
        for year in sorted(years.unique()):
            yearly_rows.append({"year": int(year), **portfolio_metrics(allocation, components.sum(axis=1).loc[years == year])})
        _, marginal, component, shares = risk_details(w, cov)
        for idx, sleeve in enumerate(SLEEVES):
            risk_rows.append({"allocation": allocation, "sleeve": sleeve, "instrument": INSTRUMENT[sleeve],
                              "weight": w[idx], "marginal_volatility": marginal[idx],
                              "component_volatility": component[idx], "risk_contribution_pct": shares[idx] * 100.0})
        for instrument in ("MNQ", "MGC"):
            mask = np.array([INSTRUMENT[s] == instrument for s in SLEEVES])
            instrument_rows.append({"allocation": allocation, "instrument": instrument,
                                    "risk_contribution_pct": float(shares[mask].sum() * 100.0),
                                    "capital_weight_pct": float(w[mask].sum() * 100.0)})
        weighted_positions = pd.DataFrame(index=position_frame.index)
        for idx, sleeve in enumerate(SLEEVES):
            weighted_positions[sleeve] = position_frame[f"position__{sleeve}"] * w[idx]
        mnq = weighted_positions[[s for s in SLEEVES if INSTRUMENT[s] == "MNQ"]]
        mgc = weighted_positions[[s for s in SLEEVES if INSTRUMENT[s] == "MGC"]]
        concentration_rows.append({
            "allocation": allocation,
            "largest_sleeve_weight_pct": float(w.max() * 100.0),
            "weight_hhi": float(np.square(w).sum()),
            "effective_weighted_sleeves": float(1.0 / np.square(w).sum()),
            "max_weighted_mnq_gross_contracts": float(mnq.abs().sum(axis=1).max()),
            "max_weighted_mnq_net_contracts_abs": float(mnq.sum(axis=1).abs().max()),
            "max_weighted_mgc_gross_contracts": float(mgc.abs().sum(axis=1).max()),
            "max_weighted_mgc_net_contracts_abs": float(mgc.sum(axis=1).abs().max()),
            "max_simultaneous_active_sleeves": int(position_frame["active_sleeves"].max()),
            "largest_sleeve_risk_contribution_pct": float(shares.max() * 100.0),
        })
    portfolio.index.name = "session_date"
    return {
        "portfolio": portfolio.reset_index(),
        "metrics": pd.DataFrame(metrics_rows),
        "yearly": pd.DataFrame(yearly_rows),
        "risk": pd.DataFrame(risk_rows),
        "instrument_risk": pd.DataFrame(instrument_rows),
        "concentration": pd.DataFrame(concentration_rows),
    }


def topstep_diagnostics(allocations: dict[str, pd.DataFrame], trade_counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio = allocations["portfolio"].set_index("session_date")
    rulesets = TopstepCampaignSpec().rulesets
    rows, path_rows = [], []
    any_trade = trade_counts.sum(axis=1).gt(0).astype(int)
    for allocation in ("equal_sleeve_weight", "inverse_volatility", "equal_risk_contribution"):
        daily = pd.DataFrame({
            "session_date": portfolio.index,
            "daily_pnl_usd": portfolio[f"daily_pnl__{allocation}"].to_numpy(),
            "daily_trade_count": any_trade.reindex(portfolio.index).to_numpy(),
        })
        for ruleset in rulesets:
            traded_remaining = daily["daily_trade_count"].iloc[::-1].cumsum().iloc[::-1]
            valid_starts = daily.index[traded_remaining.ge(int(ruleset.max_traded_days or 0))]
            for idx in valid_starts:
                history, result = simulate_topstep_path(daily.iloc[idx:].reset_index(drop=True), ruleset)
                rows.append({"allocation": allocation, "ruleset": ruleset.name,
                             "start_session_date": daily.iloc[idx]["session_date"], **result})
                if idx == valid_starts[0]:
                    part = history.copy()
                    part.insert(0, "allocation", allocation)
                    part.insert(1, "ruleset", ruleset.name)
                    path_rows.append(part)
    runs = pd.DataFrame(rows)
    summary_rows = []
    for (allocation, ruleset), group in runs.groupby(["allocation", "ruleset"]):
        reason = group["status"].value_counts()
        summary_rows.append({
            "allocation": allocation,
            "ruleset": ruleset,
            "run_count": len(group),
            "pass_rate": float(group["pass"].mean()),
            "trailing_mll_fail_rate": float(group["trailing_mll_breached"].mean()),
            "expiry_rate": float(group["expire"].mean()),
            "economic_target_hit_rate": float(group["economic_target_hit"].mean()),
            "consistency_delay_rate": float(group["economic_target_without_immediate_validation"].mean()),
            "median_days_to_pass": float(group.loc[group["pass"], "days_to_pass"].median()),
            "most_common_terminal_status": str(reason.index[0]),
        })
    return pd.DataFrame(summary_rows), (pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame())


def stress_diagnostics(pnl: pd.DataFrame, trades: dict[str, pd.DataFrame], weights: pd.DataFrame,
                       timestamp: pd.DataFrame) -> pd.DataFrame:
    weight_map = weights.set_index("sleeve")
    rows = []
    ts = timestamp.set_index("timestamp_utc")
    for allocation in weights.columns[1:]:
        w = weight_map[allocation].reindex(SLEEVES)
        daily_portfolio = pnl.mul(w, axis=1).sum(axis=1)
        simultaneous_exit_pnl = sum(ts[f"realized_pnl__{s}"] * w[s] for s in SLEEVES)
        all_worst_trades = sum(float(trades[s]["net_pnl_usd"].min()) * w[s] for s in SLEEVES)
        empirical_day = pnl.mul(w, axis=1).sum(axis=1).min()
        for stress_name, loss in (
            ("worst_observed_portfolio_day", float(empirical_day)),
            ("worst_observed_simultaneous_exit_minute", float(simultaneous_exit_pnl.min())),
            ("all_sleeve_worst_trades_synchronous", float(all_worst_trades)),
            ("two_repeated_worst_portfolio_days", float(2.0 * daily_portfolio.min())),
        ):
            repetitions = int(math.ceil(2000.0 / abs(loss))) if loss < 0 else pd.NA
            rows.append({
                "allocation": allocation,
                "stress": stress_name,
                "loss_usd": loss,
                "trailing_mll_2000_breached_from_start": bool(loss <= -2000.0),
                "distance_to_initial_mll_usd": float(2000.0 + loss),
                "repetitions_to_breach_initial_mll": repetitions,
                "cumulative_loss_at_repetition_usd": float(repetitions * loss) if loss < 0 else np.nan,
                "first_binding_constraint": "trailing_mll" if loss <= -2000.0 else "none_immediate",
            })
    return pd.DataFrame(rows)


def save_matrix(frame: pd.DataFrame, filename: str) -> None:
    frame.rename_axis("sleeve").reset_index().to_csv(OUTPUT_DIR / filename, index=False)


def make_figures(dependence: dict[str, pd.DataFrame], allocations: dict[str, pd.DataFrame],
                 timestamp: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("pearson", "spearman", "downside", "losing_days"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(dependence[name], annot=True, fmt=".2f", center=0, cmap="RdYlBu_r", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"{name.replace('_', ' ').title()} PnL correlation")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{name}_correlation_heatmap.png", dpi=160)
        plt.close(fig)

    portfolio = allocations["portfolio"]
    fig, ax = plt.subplots(figsize=(14, 7))
    for allocation in ("equal_sleeve_weight", "inverse_volatility", "equal_risk_contribution"):
        ax.plot(pd.to_datetime(portfolio["session_date"]), portfolio[f"equity__{allocation}"], label=allocation)
    ax.axhline(INITIAL_CAPITAL, color="black", lw=0.8)
    ax.set_title("Naive diagnostic allocations — final common evaluation")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "allocation_equity_curves.png", dpi=160)
    plt.close(fig)

    rolling = dependence["rolling"]
    fig, ax = plt.subplots(figsize=(14, 8))
    for (left, right), group in rolling.groupby(["sleeve_left", "sleeve_right"]):
        if left.startswith("mnq_") and right.startswith("mnq_"):
            ax.plot(pd.to_datetime(group["session_date"]), group["pearson_correlation"], label=f"{left} / {right}", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_title("63-session rolling correlations — MNQ sleeve pairs")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mnq_rolling_correlations.png", dpi=160)
    plt.close(fig)

    ts = timestamp.copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    daily = ts.assign(session_date=pd.to_datetime(ts["timestamp_utc"]).dt.date).groupby("session_date").agg(
        max_active_sleeves=("active_sleeves", "max"), max_active_mnq=("active_mnq_sleeves", "max"),
        same_direction_mnq_minutes=("mnq_same_direction_collapse", "sum"))
    ax.plot(pd.to_datetime(daily.index), daily["max_active_sleeves"], label="all sleeves")
    ax.plot(pd.to_datetime(daily.index), daily["max_active_mnq"], label="MNQ sleeves")
    ax.set_title("Maximum simultaneous active sleeves by session")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simultaneous_exposure.png", dpi=160)
    plt.close(fig)


def write_report(cfg: dict[str, Any], pnl: pd.DataFrame, trades: dict[str, pd.DataFrame],
                 dependence: dict[str, pd.DataFrame], weights: pd.DataFrame,
                 allocations: dict[str, pd.DataFrame], clusters: list[list[str]], effective_n: float,
                 topstep: pd.DataFrame, stress: pd.DataFrame, timestamp: pd.DataFrame) -> None:
    metrics = allocations["metrics"].set_index("allocation")
    concentration = allocations["concentration"].set_index("allocation")
    risk = allocations["risk"]
    instrument_risk = allocations["instrument_risk"]
    corr = dependence["pearson"]
    mnq_names = [s for s in SLEEVES if INSTRUMENT[s] == "MNQ"]
    mnq_pairs = [float(corr.loc[a, b]) for i, a in enumerate(mnq_names) for b in mnq_names[i + 1 :]]
    ts = timestamp
    active_mnq = ts["active_mnq_sleeves"].ge(2)
    collapse_rate = float(ts.loc[active_mnq, "mnq_same_direction_collapse"].mean()) if active_mnq.any() else np.nan
    loss = dependence["loss_contribution"].sort_values("share_of_gross_negative_contributions", ascending=False)

    lines = [
        "# PORTFOLIO INTEGRATION — PHASE 1", "", "## Verdict", "",
        f"The six frozen sleeves behave like approximately **{effective_n:.2f} independent risk dimensions** by correlation-eigenvalue participation ratio, with **{len(clusters)} graph clusters** at the predeclared {cfg['cluster_correlation_threshold']:.2f} correlation threshold.",
        "The portfolio is **not yet ready for Phase 2 allocation research** if Phase 2 assumes pristine OOS evidence: the MGC lock explicitly labels 2024–2026 as adaptive/reused. It is ready only for a prospective shadow/live-forward collection stage and for implementation checks that do not retune weights.",
        "", "## Scope and provenance", "",
        f"- Primary common interval: **{cfg['primary_common_evaluation']['start']} → {cfg['primary_common_evaluation']['end']}**.",
        "- Four MNQ sleeves and two MGC sleeves were reproduced from their locked configs/specifications. No sleeve parameter, exit, sizing rule, or source file was changed.",
        "- The three allocations are static diagnostics. They use no mean-return forecast, ML, Sharpe objective, or allocation selection.",
        "- Methodological caveat: the two MGC sleeves are frozen, but their source manifest calls this interval `adaptive_reused_not_pristine_oos`.",
        "- Pearson and Spearman use daily net PnL. Downside correlation uses each series clipped at zero. Losing-day correlation is conditional on a negative equal-sleeve portfolio day. Trade-day overlap is Jaccard overlap; drawdown overlap uses simultaneous underwater days.",
        "", "## Reproduced sleeves", "",
        "| Sleeve | Instrument | Trades | PnL USD | Daily vol USD |", "|---|---|---:|---:|---:|",
    ]
    for sleeve in SLEEVES:
        lines.append(f"| `{sleeve}` | {INSTRUMENT[sleeve]} | {len(trades[sleeve])} | {pnl[sleeve].sum():,.2f} | {pnl[sleeve].std(ddof=1):,.2f} |")

    lines += ["", "## Dependence and effective alpha structure", "",
              f"- Mean pairwise correlation among the four MNQ sleeves: **{np.mean(mnq_pairs):.3f}**; maximum: **{np.max(mnq_pairs):.3f}**.",
              f"- When at least two MNQ sleeves are simultaneously active, they point in the same direction **{collapse_rate:.1%}** of those minutes.",
              f"- Maximum simultaneously active sleeves: **{int(ts['active_sleeves'].max())}/6**; maximum simultaneous MNQ sleeves: **{int(ts['active_mnq_sleeves'].max())}/4**.",
              "- Clusters: " + "; ".join("[" + ", ".join(group) + "]" for group in clusters) + ".",
              "- The Nasdaq sleeves are not automatically four independent bets: the timestamp overlap and same-direction diagnostics must be read together with daily correlations and drawdown overlap.",
              "", "Largest contributors to gross negative sleeve contributions on equal-weight portfolio losing days:", "",
              "| Sleeve | Share | Signed contribution USD |", "|---|---:|---:|" ]
    for row in loss.itertuples(index=False):
        lines.append(f"| `{row.sleeve}` | {row.share_of_gross_negative_contributions:.1%} | {row.signed_pnl_on_portfolio_losing_days_usd:,.2f} |")
    yearly_corr = dependence["yearly"]
    mnq_pair = yearly_corr[(yearly_corr["sleeve_left"] == "mnq_volatility_breakout") & (yearly_corr["sleeve_right"] == "mnq_atr_trend_flip")]
    mgc_pair = yearly_corr[(yearly_corr["sleeve_left"] == "mgc_volatility_breakout") & (yearly_corr["sleeve_right"] == "mgc_gap_continuation")]
    lines += ["", "Yearly dependence stability:", "",
              f"- MNQ volatility/ATR correlation ranges from **{mnq_pair['pearson_correlation'].min():.3f}** to **{mnq_pair['pearson_correlation'].max():.3f}**.",
              f"- MGC volatility/gap correlation ranges from **{mgc_pair['pearson_correlation'].min():.3f}** to **{mgc_pair['pearson_correlation'].max():.3f}**.",
              "- The cluster identities are useful diagnostics, but their strength is not stationary year to year. Drawdown overlap is also high because several low-frequency sleeves remain underwater for long stretches; it is not evidence of synchronized entries by itself."]

    lines += ["", "## Naive allocation benchmarks", "",
              "| Allocation | PnL | Sharpe | Daily PF | Max DD | Worst day | Largest weight | Largest risk contribution |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for allocation in weights.columns[1:]:
        row = metrics.loc[allocation]
        conc = concentration.loc[allocation]
        lines.append(f"| `{allocation}` | {row.net_pnl_usd:,.2f} | {row.sharpe_annualized:.3f} | {row.daily_profit_factor:.3f} | {row.max_drawdown_usd:,.2f} | {row.worst_day_usd:,.2f} | {conc.largest_sleeve_weight_pct:.1f}% | {conc.largest_sleeve_risk_contribution_pct:.1f}% |")
    lines += ["", "Yearly benchmark PnL:", "", "| Year | Equal | Inverse vol | ERC |", "|---:|---:|---:|---:|"]
    yearly_alloc = allocations["yearly"]
    for year in sorted(yearly_alloc["year"].unique()):
        sub = yearly_alloc[yearly_alloc["year"].eq(year)].set_index("allocation")
        lines.append(f"| {int(year)} | {sub.loc['equal_sleeve_weight', 'net_pnl_usd']:,.2f} | {sub.loc['inverse_volatility', 'net_pnl_usd']:,.2f} | {sub.loc['equal_risk_contribution', 'net_pnl_usd']:,.2f} |")
    lines += ["", "Static diagnostic weights:", "", "| Sleeve | Equal | Inverse vol | ERC |", "|---|---:|---:|---:|"]
    for row in weights.itertuples(index=False):
        lines.append(f"| `{row.sleeve}` | {row.equal_sleeve_weight:.2%} | {row.inverse_volatility:.2%} | {row.equal_risk_contribution:.2%} |")
    lines += ["", "Instrument risk contribution:", "", "| Allocation | MNQ | MGC |", "|---|---:|---:|"]
    for allocation in weights.columns[1:]:
        sub = instrument_risk[instrument_risk["allocation"].eq(allocation)].set_index("instrument")
        lines.append(f"| `{allocation}` | {sub.loc['MNQ', 'risk_contribution_pct']:.1f}% | {sub.loc['MGC', 'risk_contribution_pct']:.1f}% |")
    lines += ["", "Exposure and allocation concentration:", "",
              "| Allocation | Effective weighted sleeves | Max weighted MNQ gross | Max weighted MGC gross | Max simultaneous sleeves |",
              "|---|---:|---:|---:|---:|"]
    for allocation in weights.columns[1:]:
        row = concentration.loc[allocation]
        lines.append(f"| `{allocation}` | {row.effective_weighted_sleeves:.2f} | {row.max_weighted_mnq_gross_contracts:.3f} | {row.max_weighted_mgc_gross_contracts:.3f} | {int(row.max_simultaneous_active_sleeves)} |")

    lines += ["", "## TopstepX 50K constraints", "",
              "The simulation calls the repository's existing `TopstepCampaignSpec` verbatim: USD 50,000 start, USD 3,000 target, USD 2,000 trailing MLL, 50% best-day consistency, 35- and 60-traded-day horizons, no daily loss limit, and no breakeven lock. No contract cap is present in that ruleset, so none is invented here.",
              "", "| Allocation | Ruleset | Pass | MLL fail | Expiry | Target hit | Consistency delay | First empirical binder |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in topstep.itertuples(index=False):
        lines.append(f"| `{row.allocation}` | `{row.ruleset}` | {row.pass_rate:.1%} | {row.trailing_mll_fail_rate:.1%} | {row.expiry_rate:.1%} | {row.economic_target_hit_rate:.1%} | {row.consistency_delay_rate:.1%} | `{row.most_common_terminal_status}` |")
    sync = stress[stress["stress"].eq("all_sleeve_worst_trades_synchronous")]
    lines += ["", "Synchronous-loss stress (all six sleeves realize their individual worst trade together):", "",
              "| Allocation | Loss | Initial MLL breached | Repetitions to MLL | First immediate constraint |", "|---|---:|---|---:|---|"]
    for row in sync.itertuples(index=False):
        lines.append(f"| `{row.allocation}` | {row.loss_usd:,.2f} | {row.trailing_mll_2000_breached_from_start} | {int(row.repetitions_to_breach_initial_mll)} | `{row.first_binding_constraint}` |")
    lines += ["", "On observed rolling starts, **expiry binds first in every case**: none of the three unlevered static benchmarks reaches the USD 3,000 target inside 35 or 60 traded days, and none breaches the USD 2,000 MLL. Under repeated synchronous worst-trade stress, the first risk constraint that can bind is the trailing MLL; there is no repository Topstep daily-loss constraint to bind earlier."]

    lines += ["", "## Phase-2 readiness", "",
              "**Not ready for parameterized or dynamic Phase 2 allocation.** The correct next step is prospective collection with all six definitions unchanged. Static benchmark results can guide monitoring, but must not be used to choose a best-Sharpe allocation on this reused interval.",
              "", "## Reproducible artifacts", "",
              "- `common_timestamp_portfolio.csv`: minute positions, entries/exits, realized PnL, instrument exposure, and simultaneity flags.",
              "- `daily_sleeve_pnl.csv`, `daily_trade_counts.csv`, `sleeve_trades/`: reproduced source series.",
              "- Correlation, downside, loss-day, overlap, drawdown, rolling, yearly, and loss-contribution CSVs.",
              "- Allocation weights, metrics, yearly results, sleeve/instrument risk contributions, concentration, Topstep and stress CSVs.",
              "- `run_metadata.json` records hashes and implementation checks.", ""]
    (OUTPUT_DIR / "PORTFOLIO_INTEGRATION_PHASE1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def execute() -> None:
    for directory in (OUTPUT_DIR, TRADES_DIR, FIGURES_DIR, MPL_CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    cfg, input_hashes = load_and_verify_phase_config()
    config_hash = sha256(CONFIG_PATH)
    trades, daily, implementation_checks = reproduce_sleeves(cfg)
    for sleeve, frame in trades.items():
        frame.to_csv(TRADES_DIR / f"{sleeve}.csv", index=False)
    pnl, trade_counts = build_daily_matrix(daily, trades)
    pnl.reset_index().to_csv(OUTPUT_DIR / "daily_sleeve_pnl.csv", index=False)
    trade_counts.reset_index().to_csv(OUTPUT_DIR / "daily_trade_counts.csv", index=False)
    sleeve_yearly_results(pnl).to_csv(OUTPUT_DIR / "sleeve_yearly_results.csv", index=False)
    timestamp = build_timestamp_dataset(trades, pnl.index.tolist())
    timestamp.to_csv(OUTPUT_DIR / "common_timestamp_portfolio.csv", index=False)

    dependence = dependence_outputs(pnl, trades, timestamp, int(cfg["rolling_correlation_sessions"]))
    for key, filename in (
        ("pearson", "pearson_pnl_correlation.csv"), ("spearman", "spearman_pnl_correlation.csv"),
        ("downside", "downside_pnl_correlation.csv"), ("losing_days", "losing_day_pnl_correlation.csv"),
    ):
        save_matrix(dependence[key], filename)
    dependence["drawdown"].reset_index().to_csv(OUTPUT_DIR / "sleeve_drawdowns.csv", index=False)
    dependence["trade_overlap"].to_csv(OUTPUT_DIR / "trade_day_and_position_overlap.csv", index=False)
    dependence["drawdown_overlap"].to_csv(OUTPUT_DIR / "drawdown_overlap.csv", index=False)
    dependence["rolling"].to_csv(OUTPUT_DIR / "rolling_correlations_63d.csv", index=False)
    dependence["yearly"].to_csv(OUTPUT_DIR / "yearly_correlation_stability.csv", index=False)
    dependence["loss_contribution"].to_csv(OUTPUT_DIR / "sleeve_loss_contributions.csv", index=False)

    eigenvalues = np.linalg.eigvalsh(dependence["pearson"].to_numpy(float)).clip(min=0)
    effective_n = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    clusters = connected_clusters(dependence["pearson"], float(cfg["cluster_correlation_threshold"]))
    pd.DataFrame([{"cluster_id": idx + 1, "sleeve": sleeve} for idx, group in enumerate(clusters) for sleeve in group]).to_csv(
        OUTPUT_DIR / "effective_clusters.csv", index=False)

    weights, solver = allocation_weights(pnl)
    weights.to_csv(OUTPUT_DIR / "allocation_weights.csv", index=False)
    allocations = build_allocations(pnl, weights, timestamp)
    for key, filename in (
        ("portfolio", "allocation_daily_portfolio.csv"), ("metrics", "allocation_metrics.csv"),
        ("yearly", "allocation_yearly_results.csv"), ("risk", "allocation_sleeve_risk_contributions.csv"),
        ("instrument_risk", "allocation_instrument_risk_contributions.csv"),
        ("concentration", "allocation_concentration.csv"),
    ):
        allocations[key].to_csv(OUTPUT_DIR / filename, index=False)

    topstep, topstep_paths = topstep_diagnostics(allocations, trade_counts)
    topstep.to_csv(OUTPUT_DIR / "topstep_rolling_start_summary.csv", index=False)
    topstep_paths.to_csv(OUTPUT_DIR / "topstep_initial_paths.csv", index=False)
    rules = pd.DataFrame([asdict(item) for item in TopstepCampaignSpec().rulesets])
    rules.to_csv(OUTPUT_DIR / "topstep_rulesets_exact.csv", index=False)
    stress = stress_diagnostics(pnl, trades, weights, timestamp)
    stress.to_csv(OUTPUT_DIR / "simultaneous_loss_stress.csv", index=False)
    make_figures(dependence, allocations, timestamp)
    write_report(cfg, pnl, trades, dependence, weights, allocations, clusters, effective_n, topstep, stress, timestamp)

    if sha256(CONFIG_PATH) != config_hash:
        raise RuntimeError("Phase config changed during execution")
    for key, expected in input_hashes.items():
        if sha256(REPO_ROOT / cfg["source_locks"][key]["path"]) != expected:
            raise RuntimeError(f"Locked source changed during execution: {key}")
    metadata = {
        "phase": cfg["phase"],
        "config_sha256": config_hash,
        "input_sha256": input_hashes,
        "period": cfg["primary_common_evaluation"],
        "session_count": len(pnl),
        "sleeve_count": len(SLEEVES),
        "trade_count_by_sleeve": {s: len(trades[s]) for s in SLEEVES},
        "effective_independent_sleeves_participation_ratio": effective_n,
        "cluster_count": len(clusters),
        "clusters": clusters,
        "erc_solver": solver,
        "topstep_rulesets": [asdict(item) for item in TopstepCampaignSpec().rulesets],
        "parameter_optimization_performed": False,
        "ml_used": False,
        "sleeve_sources_modified": False,
        "implementation_checks": implementation_checks,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(allocations["metrics"].to_string(index=False), flush=True)
    print(f"Effective independent sleeves: {effective_n:.3f}; clusters: {clusters}", flush=True)


if __name__ == "__main__":
    execute()
