"""Causal cross-asset lead/lag event study and conditional strategy gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "cross_asset_lead_lag_20261002"
SPEC_PATH = OUT / "frozen_specification.json"
START, END = pd.Timestamp("2019-05-06"), pd.Timestamp("2026-03-19")
SEED = 20261002
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PATHS = {
    "MNQ": ROOT / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet",
    "MGC": ROOT / "data/processed/parquet/MGC_v_0_1m_20150101_20260923.parquet",
    "MES": ROOT / "data/processed/parquet/MES_c_0_1m_20260322_135702.parquet",
    "M2K": ROOT / "data/processed/parquet/M2K_c_0_1m_20260322_134808.parquet",
    "6E": ROOT / "data/processed/parquet/6E_v_0_1m_20150101_20260923.parquet",
    "ZN": ROOT / "data/processed/parquet/ZN_v_0_1m_20130517_20260920.parquet",
}
PAIRS = (("MES", "MNQ"), ("M2K", "MNQ"), ("6E", "MGC"), ("ZN", "MGC"))
SESSIONS = {"MNQ": ("09:30", "16:00"), "MGC": ("08:20", "13:30")}
EXECUTION = {"MNQ": {"tick": .25, "point_value": 2.0}, "MGC": {"tick": .1, "point_value": 10.0}}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, (pd.Timestamp, pd.Timedelta, Path)):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return None if not np.isfinite(x) else float(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def split_label(day: Any) -> str:
    d = pd.Timestamp(day)
    return "is" if d < pd.Timestamp("2023-01-01") else ("validation" if d < pd.Timestamp("2024-01-01") else "oos")


def load_bars(symbol: str) -> pd.DataFrame:
    x = pd.read_parquet(PATHS[symbol]).sort_index()
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    x.index = x.index.tz_convert("America/New_York")
    mask = (x.index.date >= START.date()) & (x.index.date <= END.date())
    return x.loc[mask]


def five_minute_rth(bars: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = bars.between_time(start, end, inclusive="left").copy()
    x["session_date"] = pd.to_datetime(x.index.date)
    parts: list[pd.DataFrame] = []
    rejected_roll = 0
    rejected_open = 0
    shortened = 0
    for day, frame in x.groupby("session_date", sort=True):
        day = pd.Timestamp(day)
        if day.weekday() >= 5:
            continue
        open_ts = pd.Timestamp(f"{day.date()} {start}", tz="America/New_York")
        end_ts = pd.Timestamp(f"{day.date()} {end}", tz="America/New_York")
        if open_ts not in frame.index or len(frame) < 30:
            rejected_open += 1
            continue
        if "instrument_id" in frame and frame.instrument_id.nunique() != 1:
            rejected_roll += 1
            continue
        shortened += int(frame.index[-1] < end_ts - pd.Timedelta(minutes=2))
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        if "instrument_id" in frame:
            agg["instrument_id"] = "last"
        five = frame.resample("5min", origin=open_ts, label="left", closed="left").agg(agg).dropna(subset=["open", "close"])
        five["session_date"] = day
        parts.append(five)
    out = pd.concat(parts).sort_index() if parts else pd.DataFrame()
    audit = {"sessions": int(out.session_date.nunique()) if len(out) else 0, "shortened_sessions": shortened, "sessions_rejected_missing_open": rejected_open, "sessions_rejected_intraday_roll": rejected_roll}
    return out, audit


def synchronize_pair(leader: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "session_date"]
    left = leader[[c for c in cols if c in leader]].add_prefix("leader_")
    right = target[[c for c in cols if c in target]].add_prefix("target_")
    out = left.join(right, how="inner")
    out = out[out.leader_session_date == out.target_session_date].copy()
    out["session_date"] = pd.to_datetime(out.target_session_date)
    return out


def causal_features(sync: pd.DataFrame, beta_window: int, impulse_bars: int) -> pd.DataFrame:
    """All estimates at row t exclude target return t and beta observations t."""
    out = sync.copy()
    same_previous_session = out.session_date.eq(out.session_date.shift(1))
    leader_ret = np.log(out.leader_close).diff().where(same_previous_session)
    target_ret = np.log(out.target_close).diff().where(same_previous_session)
    # Compress to valid intraday return pairs so the rolling window contains exactly
    # the requested number of observations while excluding every overnight gap.
    prior = pd.DataFrame({"leader": leader_ret.shift(1), "target": target_ret.shift(1)}, index=out.index).dropna()
    beta_valid = prior.target.rolling(beta_window, min_periods=beta_window).cov(prior.leader) / prior.leader.rolling(beta_window, min_periods=beta_window).var().replace(0, np.nan)
    sigma_valid = prior.leader.rolling(beta_window, min_periods=beta_window).std()
    out["beta"] = beta_valid.reindex(out.index).ffill()
    out["leader_sigma"] = sigma_valid.reindex(out.index).ffill()
    out["leader_impulse"] = np.log(out.leader_close) - np.log(out.leader_close.shift(impulse_bars))
    # This target response ends at t-1, so the simultaneous target bar can never gate a trade.
    out["target_lagged_response"] = np.log(out.target_close.shift(1)) - np.log(out.target_close.shift(impulse_bars + 1))
    impulse_same_session = out.session_date.eq(out.session_date.shift(impulse_bars))
    lagged_response_same_session = out.session_date.eq(out.session_date.shift(1)) & out.session_date.eq(out.session_date.shift(impulse_bars + 1))
    out["leader_impulse"] = out.leader_impulse.where(impulse_same_session)
    out["target_lagged_response"] = out.target_lagged_response.where(lagged_response_same_session)
    out["impulse_z"] = out.leader_impulse / (out.leader_sigma * math.sqrt(impulse_bars))
    out["expected_target_move"] = out.beta * out.leader_impulse
    out["residual"] = out.target_lagged_response - out.expected_target_move
    return out


def _event_pnl(frame: pd.DataFrame, loc: int, direction: int, horizon: int, tick: float, point_value: float, slip_ticks: int = 1, commission_mult: float = 1.0, delay_bars: int = 0) -> dict[str, float] | None:
    entry_loc = loc + 1 + delay_bars
    exit_loc = entry_loc + horizon // 5
    if exit_loc >= len(frame):
        return None
    if frame.session_date.iloc[entry_loc] != frame.session_date.iloc[loc] or frame.session_date.iloc[exit_loc] != frame.session_date.iloc[loc]:
        return None
    raw_entry = float(frame.target_open.iloc[entry_loc]); raw_exit = float(frame.target_open.iloc[exit_loc])
    gross = direction * (raw_exit - raw_entry) * point_value
    slippage = 2 * slip_ticks * tick * point_value
    commission = 2 * 1.25 * commission_mult
    return {"entry_ts": frame.index[entry_loc], "exit_ts": frame.index[exit_loc], "entry_price": raw_entry + direction * slip_ticks * tick, "exit_price": raw_exit - direction * slip_ticks * tick, "gross_pnl_usd": gross, "slippage_usd": slippage, "commissions_usd": commission, "net_pnl_usd": gross - slippage - commission, "signed_return_bps": direction * math.log(raw_exit / raw_entry) * 10000}


def build_events(leader: str, target: str, sync: pd.DataFrame) -> pd.DataFrame:
    costs = EXECUTION[target]
    records: list[dict[str, Any]] = []
    for beta_window in (60, 120):
        for impulse_minutes in (5, 15, 30):
            feat = causal_features(sync, beta_window, impulse_minutes // 5)
            for threshold in (1.5, 2.0, 2.5):
                for side_name, direction in (("long", 1), ("short", -1)):
                    eligible = (direction * feat.expected_target_move > 0) & (feat.impulse_z.abs() >= threshold) & (direction * feat.target_lagged_response < .5 * feat.expected_target_move.abs()) & (direction * feat.residual < 0)
                    candidates = feat.loc[eligible].copy()
                    if candidates.empty:
                        continue
                    candidates = candidates.loc[~candidates.session_date.duplicated(keep="first")]
                    for ts, row in candidates.iterrows():
                        loc = feat.index.get_loc(ts)
                        base: dict[str, Any] = {"leader": leader, "target": target, "pair": f"{leader}->{target}", "side": side_name, "direction": direction, "session_date": row.session_date, "split": split_label(row.session_date), "bar_start_ts": ts, "signal_ts": ts + pd.Timedelta(minutes=5), "beta_window_bars": beta_window, "impulse_minutes": impulse_minutes, "threshold_z": threshold, "beta": row.beta, "leader_impulse": row.leader_impulse, "impulse_z": row.impulse_z, "target_lagged_response": row.target_lagged_response, "expected_target_move": row.expected_target_move, "residual": row.residual}
                        ok = True
                        for horizon in (5, 15, 30, 60):
                            nominal = _event_pnl(feat, loc, direction, horizon, costs["tick"], costs["point_value"])
                            if nominal is None:
                                ok = False; break
                            base[f"fwd_{horizon}m_bps"] = nominal["signed_return_bps"]
                            base[f"gross_{horizon}m_usd"] = nominal["gross_pnl_usd"]
                            base[f"net_{horizon}m_usd"] = nominal["net_pnl_usd"]
                            if horizon == 30:
                                base.update({"entry_ts": nominal["entry_ts"], "exit_ts": nominal["exit_ts"], "entry_price": nominal["entry_price"], "exit_price": nominal["exit_price"], "gross_pnl_usd": nominal["gross_pnl_usd"], "slippage_usd": nominal["slippage_usd"], "commissions_usd": nominal["commissions_usd"], "net_pnl_usd": nominal["net_pnl_usd"]})
                                for stress, kwargs in {"plus1tick": {"slip_ticks": 2}, "plus2ticks": {"slip_ticks": 3}, "plus3ticks": {"slip_ticks": 4}, "double_commission": {"commission_mult": 2}, "delay_one_bar": {"delay_bars": 1}}.items():
                                    alt = _event_pnl(feat, loc, direction, horizon, costs["tick"], costs["point_value"], **kwargs)
                                    base[f"net_30m_{stress}_usd"] = alt["net_pnl_usd"] if alt else np.nan
                                    if stress == "delay_one_bar":
                                        base["fwd_30m_delay_bps"] = alt["signed_return_bps"] if alt else np.nan
                        if ok:
                            records.append(base)
    events = pd.DataFrame(records)
    if len(events):
        events["overlap_cells"] = events.groupby(["pair", "side", "session_date", "signal_ts"])["threshold_z"].transform("size")
    return events


def bootstrap_summary(events: pd.DataFrame, samples: int) -> pd.DataFrame:
    long = events.melt(id_vars=["pair", "leader", "target", "side", "session_date", "split", "beta_window_bars", "impulse_minutes", "threshold_z"], value_vars=[f"fwd_{h}m_bps" for h in (5, 15, 30, 60)], var_name="horizon", value_name="signed_return_bps")
    rng = np.random.default_rng(SEED)
    rows = []
    keys = ["pair", "leader", "target", "side", "beta_window_bars", "impulse_minutes", "threshold_z", "horizon"]
    for key, group in long.groupby(keys):
        for period in ("all", "is", "validation", "oos"):
            g = group if period == "all" else group[group.split == period]
            daily = g.groupby("session_date").signed_return_bps.mean().dropna().to_numpy(float)
            if not len(daily):
                continue
            boot = np.array([rng.choice(daily, len(daily), replace=True).mean() for _ in range(samples)])
            test = stats.ttest_1samp(daily, 0) if len(daily) > 1 else None
            rows.append(dict(zip(keys, key)) | {"period": period, "events": len(daily), "mean_bps": daily.mean(), "median_bps": np.median(daily), "ci_low_bps": np.quantile(boot, .025), "ci_high_bps": np.quantile(boot, .975), "p_value": float(test.pvalue) if test else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bonferroni"] = out.groupby(["pair", "side"])["p_value"].transform(lambda s: np.minimum(1.0, s * s.notna().sum()))
    return out


def annual_summary(events: pd.DataFrame) -> pd.DataFrame:
    x = events.copy(); x["year"] = pd.to_datetime(x.session_date).dt.year
    keys = ["pair", "leader", "target", "side", "beta_window_bars", "impulse_minutes", "threshold_z", "year"]
    return x.groupby(keys).agg(events=("session_date", "size"), mean_5m_bps=("fwd_5m_bps", "mean"), mean_15m_bps=("fwd_15m_bps", "mean"), mean_30m_bps=("fwd_30m_bps", "mean"), mean_60m_bps=("fwd_60m_bps", "mean"), gross_30m_usd=("gross_30m_usd", "mean"), net_30m_usd=("net_30m_usd", "mean")).reset_index()


def monotonicity_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pair, side, beta, impulse), g in events.groupby(["pair", "side", "beta_window_bars", "impulse_minutes"]):
        means = g.groupby("threshold_z").fwd_30m_bps.mean().reindex([1.5, 2.0, 2.5])
        available = means.dropna()
        rho, p = stats.spearmanr(available.index, available.values) if len(available) >= 3 else (np.nan, np.nan)
        monotone = len(available) == 3 and bool(np.all(np.diff(available.values) >= -.05))
        rows.append({"pair": pair, "side": side, "beta_window_bars": beta, "impulse_minutes": impulse, "mean_1p5_bps": means.loc[1.5], "mean_2p0_bps": means.loc[2.0], "mean_2p5_bps": means.loc[2.5], "spearman_rho": rho, "p_value": p, "monotone_non_decreasing": monotone})
    return pd.DataFrame(rows)


def decision_registry(events: pd.DataFrame, summary: pd.DataFrame, annual: pd.DataFrame, monotonicity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for leader, target in PAIRS:
        pair = f"{leader}->{target}"
        for side in ("long", "short"):
            e = events[(events.pair == pair) & (events.side == side) & (events.beta_window_bars == 120) & (events.impulse_minutes == 15) & (events.threshold_z == 2.0)]
            s = summary[(summary.pair == pair) & (summary.side == side) & (summary.beta_window_bars == 120) & (summary.impulse_minutes == 15) & (summary.threshold_z == 2.0) & (summary.horizon == "fwd_30m_bps")].set_index("period")
            m = monotonicity[(monotonicity.pair == pair) & (monotonicity.side == side) & (monotonicity.beta_window_bars == 120) & (monotonicity.impulse_minutes == 15)]
            years = annual[(annual.pair == pair) & (annual.side == side) & (annual.beta_window_bars == 120) & (annual.impulse_minutes == 15) & (annual.threshold_z == 2.0)]
            full_n = int(len(e)); oos_n = int((e.split == "oos").sum())
            split_means = {p: (float(s.at[p, "mean_bps"]) if p in s.index else np.nan) for p in ("is", "validation", "oos")}
            positive_year_fraction = float((years.mean_30m_bps > 0).mean()) if len(years) else 0.0
            monotone = bool(m.monotone_non_decreasing.iloc[0]) if len(m) else False
            all_s = s.loc["all"] if "all" in s.index else None
            inference = bool(all_s is not None and all_s.ci_low_bps > 0 and all_s.p_bonferroni <= .05)
            neighbor = events[(events.pair == pair) & (events.side == side)].groupby(["beta_window_bars", "impulse_minutes", "threshold_z"]).fwd_30m_bps.mean()
            neighbor_positive = float((neighbor > 0).mean()) if len(neighbor) else 0.0
            costs = {c: float(e[c].mean()) if len(e) else np.nan for c in ("gross_30m_usd", "net_30m_usd", "net_30m_plus1tick_usd", "net_30m_plus2ticks_usd", "net_30m_plus3ticks_usd", "net_30m_double_commission_usd", "net_30m_delay_one_bar_usd")}
            economic = all(np.isfinite(costs[c]) and costs[c] > 0 for c in costs if c != "gross_30m_usd")
            stable = full_n >= 80 and oos_n >= 25 and all(np.isfinite(v) and v > 0 for v in split_means.values()) and positive_year_fraction >= .6 and neighbor_positive >= .5
            passed = monotone and inference and economic and stable
            rows.append({"pair": pair, "leader": leader, "target": target, "side": side, "canonical_events": full_n, "oos_events": oos_n, "is_mean_30m_bps": split_means["is"], "validation_mean_30m_bps": split_means["validation"], "oos_mean_30m_bps": split_means["oos"], "positive_year_fraction": positive_year_fraction, "neighbor_positive_fraction": neighbor_positive, "monotone": monotone, "bootstrap_inference_pass": inference, "economic_cost_delay_pass": economic, **costs, "verdict": "STRATEGY_GATE_SURVIVOR" if passed else "REJECTED_NO_LEAD_LAG_EDGE"})
    return pd.DataFrame(rows)


def build_survivor_trades(events: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    survivors = decisions[decisions.verdict == "STRATEGY_GATE_SURVIVOR"]
    cols = ["pair", "leader", "target", "side", "session_date", "split", "entry_ts", "exit_ts", "entry_price", "exit_price", "gross_pnl_usd", "slippage_usd", "commissions_usd", "net_pnl_usd", "stress"]
    rows = []
    for _, d in survivors.iterrows():
        e = events[(events.pair == d.pair) & (events.side == d.side) & (events.beta_window_bars == 120) & (events.impulse_minutes == 15) & (events.threshold_z == 2.0)]
        for _, r in e.iterrows():
            base = {c: r[c] for c in cols[:-1]}
            rows.append(base | {"stress": "nominal"})
            for stress, col in (("plus1tick", "net_30m_plus1tick_usd"), ("plus2ticks", "net_30m_plus2ticks_usd"), ("plus3ticks", "net_30m_plus3ticks_usd"), ("double_commission", "net_30m_double_commission_usd"), ("delay_one_bar", "net_30m_delay_one_bar_usd")):
                rows.append(base | {"net_pnl_usd": r[col], "stress": stress})
    return pd.DataFrame(rows, columns=cols)


def trade_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["pair", "side", "stress", "period", "trades", "pnl_usd", "expectancy_usd", "sharpe", "profit_factor", "hit_rate", "max_drawdown_usd"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (pair, side, stress), g in trades.groupby(["pair", "side", "stress"]):
        for period in ("all", "is", "validation", "oos"):
            x = g if period == "all" else g[g.split == period]
            if x.empty:
                continue
            pnl = x.net_pnl_usd.astype(float); daily = x.groupby("session_date").net_pnl_usd.sum().sort_index(); sd = daily.std(); curve = daily.cumsum(); losses = -pnl[pnl < 0].sum()
            rows.append({"pair": pair, "side": side, "stress": stress, "period": period, "trades": len(x), "pnl_usd": pnl.sum(), "expectancy_usd": pnl.mean(), "sharpe": daily.mean() / sd * math.sqrt(252) if sd > 0 else np.nan, "profit_factor": pnl[pnl > 0].sum() / losses if losses > 0 else np.inf, "hit_rate": (pnl > 0).mean(), "max_drawdown_usd": (curve - curve.cummax()).min()})
    return pd.DataFrame(rows, columns=columns)


def plots(events: pd.DataFrame, summary: pd.DataFrame) -> None:
    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)
    for (pair, side), g in events.groupby(["pair", "side"]):
        canon = g[(g.beta_window_bars == 120) & (g.impulse_minutes == 15)]
        pivot = canon.pivot_table(index="threshold_z", values=["fwd_5m_bps", "fwd_15m_bps", "fwd_30m_bps", "fwd_60m_bps"], aggfunc="mean")
        if len(pivot):
            fig, ax = plt.subplots(figsize=(7, 4)); pivot.plot(marker="o", ax=ax); ax.axhline(0, color="black", lw=.8); ax.set_title(f"{pair} {side}: conditional target returns"); ax.set_ylabel("signed bps"); fig.tight_layout(); fig.savefig(figdir / f"conditional_{pair.replace('->','_')}_{side}.png", dpi=140); plt.close(fig)
        heat = g.pivot_table(index="impulse_minutes", columns="threshold_z", values="fwd_30m_bps", aggfunc="mean")
        if len(heat):
            fig, ax = plt.subplots(figsize=(6, 4)); sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax); ax.set_title(f"{pair} {side}: mean 30m bps"); fig.tight_layout(); fig.savefig(figdir / f"heatmap_{pair.replace('->','_')}_{side}.png", dpi=140); plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(empty)_"
    x = frame.copy()
    for c in x:
        x[c] = x[c].map(lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v))
    return "\n".join(["| " + " | ".join(x.columns) + " |", "|" + "|".join(["---"] * len(x.columns)) + "|"] + ["| " + " | ".join(row.astype(str)) + " |" for _, row in x.iterrows()])


def write_report(decisions: pd.DataFrame, audits: dict[str, Any], events: pd.DataFrame, trades: pd.DataFrame) -> None:
    survivors = decisions[decisions.verdict == "STRATEGY_GATE_SURVIVOR"]
    concise = decisions[["pair", "side", "canonical_events", "oos_events", "is_mean_30m_bps", "validation_mean_30m_bps", "oos_mean_30m_bps", "monotone", "bootstrap_inference_pass", "economic_cost_delay_pass", "verdict"]]
    lines = ["# Cross-asset lead/lag — final report", "", "## Résumé", "", f"**Verdict global: `{'STRATEGY_GATE_SURVIVOR' if len(survivors) else 'REJECTED_NO_LEAD_LAG_EDGE'}`.**", "", f"Les quatre paires et les deux directions ont été évaluées séparément. **{len(survivors)}** combinaison(s) franchissent simultanément monotonie, stabilité temporelle, bootstrap, coûts et délai. {'Une stratégie conditionnelle a été produite.' if len(survivors) else 'Aucune stratégie n’a été implémentée, conformément au gate préspécifié.'}", "", "## Registre de décision", "", markdown(concise), "", "## Causalité et exécution", "", "Les bêtas et volatilités à `t` n’utilisent que des rendements terminés à `t-1`. L’impulsion leader est connue à la clôture de la barre `t`; la réponse cible utilisée pour qualifier la sous-réaction se termine également à `t-1`, donc aucun rendement cible contemporain n’est tradable. Le premier prix d’entrée hypothétique est l’open exact de la barre suivante. Le stress de délai décale encore l’entrée de cinq minutes.", "", "## Données et synchronisation", "", "| Symbol | Rows | First | Last | Roll safety | SHA-256 |", "|---|---:|---|---|---|---|"]
    for symbol, a in audits.items():
        lines.append(f"| {symbol} | {a['rows']} | {a['first']} | {a['last']} | {a['roll_safety']} | `{a['sha256']}` |")
    lines += ["", f"Événements retenus sur toute la grille: **{len(events):,}**. Trades de stratégie: **{len(trades):,}**.", "", "## Portefeuille et Topstep", "", "Aucune comparaison marginale ni simulation Topstep n’est exécutée sans survivant standalone. Les six sleeves gelées restent inchangées." if not len(survivors) else "Les survivants doivent encore franchir le fichier de diagnostic portefeuille avant toute simulation Topstep.", "", "## Limites", "", "Le dataset 6E ne contient pas `instrument_id`; sa sûreté de roll intraday ne peut donc pas être vérifiée comme pour les cinq autres séries. Aucun téléchargement ou proxy n’a été utilisé. Les sessions raccourcies sont conservées seulement lorsque les timestamps exacts nécessaires existent; aucune barre manquante n’est imputée."]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(samples: int = 1000) -> None:
    if not SPEC_PATH.exists():
        raise FileNotFoundError("Frozen specification is required before calculations")
    spec_hash = sha256(SPEC_PATH)
    audits: dict[str, Any] = {}
    event_parts = []
    for leader, target in PAIRS:
        leader_raw, target_raw = load_bars(leader), load_bars(target)
        start, end = SESSIONS[target]
        leader_five, leader_session_audit = five_minute_rth(leader_raw, start, end)
        target_five, target_session_audit = five_minute_rth(target_raw, start, end)
        for symbol, raw, sa in ((leader, leader_raw, leader_session_audit), (target, target_raw, target_session_audit)):
            if symbol not in audits:
                audits[symbol] = {"path": str(PATHS[symbol].relative_to(ROOT)), "sha256": sha256(PATHS[symbol]), "rows": len(raw), "first": raw.index.min(), "last": raw.index.max(), "roll_safety": "available" if "instrument_id" in raw else "unavailable_instrument_id_missing", **sa}
        sync = synchronize_pair(leader_five, target_five)
        pair_events = build_events(leader, target, sync)
        event_parts.append(pair_events)
        pd.DataFrame({"pair": [f"{leader}->{target}"], "leader_5m_rows": [len(leader_five)], "target_5m_rows": [len(target_five)], "synchronized_rows": [len(sync)], "first": [sync.index.min()], "last": [sync.index.max()]}).to_csv(OUT / f"synchronization_{leader}_{target}.csv", index=False)
        del leader_raw, target_raw, leader_five, target_five, sync, pair_events
        gc.collect()
    events = pd.concat(event_parts, ignore_index=True)
    events.to_csv(OUT / "events.csv", index=False)
    summary = bootstrap_summary(events, samples); summary.to_csv(OUT / "event_summary.csv", index=False)
    annual = annual_summary(events); annual.to_csv(OUT / "annual_stability.csv", index=False)
    monotonicity = monotonicity_table(events); monotonicity.to_csv(OUT / "monotonicity.csv", index=False)
    decisions = decision_registry(events, summary, annual, monotonicity); decisions.to_csv(OUT / "decision_registry.csv", index=False)
    trades = build_survivor_trades(events, decisions); trades.to_csv(OUT / "trades.csv", index=False)
    metrics = trade_metrics(trades); metrics.to_csv(OUT / "strategy_metrics.csv", index=False)
    pd.DataFrame(columns=["pair", "portfolio_gate", "daily_correlation", "negative_day_contribution", "sharpe_delta", "maxdd_delta", "es95_delta"]).to_csv(OUT / "portfolio_marginal_value.csv", index=False)
    plots(events, summary)
    write_json(OUT / "data_audit.json", audits)
    write_report(decisions, audits, events, trades)
    inputs = [SPEC_PATH, Path(__file__), ROOT / "tests/test_cross_asset_lead_lag_campaign.py"] + list(PATHS.values())
    manifest = {"specification_sha256_before_and_after": [spec_hash, sha256(SPEC_PATH)], "seed": SEED, "bootstrap_samples": samples, "topstep_run": False, "inputs_and_scripts": {str(p.relative_to(ROOT)): sha256(p) for p in inputs}, "outputs": {}}
    for p in OUT.rglob("*"):
        if p.is_file() and p.name != "manifest.json":
            manifest["outputs"][str(p.relative_to(OUT))] = sha256(p)
    write_json(OUT / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--bootstrap-samples", type=int, default=1000); args = parser.parse_args(); run(args.bootstrap_samples)


if __name__ == "__main__":
    main()
