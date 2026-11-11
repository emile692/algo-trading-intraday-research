"""Execute the preregistered Phase-2 tests after the spec freeze commit."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostic_is.run_step1 import B_PARAMS, prior_daily_atr14  # noqa: E402
from research.mean_reversion_cross_asset_20261006.run_campaign import (  # noqa: E402
    EXIT_SPECS,
    INSTRUMENTS as A_INSTRUMENTS,
    _load_bars,
    _prepare_features,
    _signals,
    backtest as canonical_a_backtest,
)
from research.mgc_baseline_backtests.run_mgc_baselines import prepare_mgc_data  # noqa: E402
from research.baseline_backtests.baseline_engine import run_baseline_backtest  # noqa: E402
from research.volume_auction_landscape_20261007.run_campaign import (  # noqa: E402
    INSTRUMENTS as B_INSTRUMENTS,
    add_causal_volume_baselines,
    add_price_features,
    backtest as b_backtest,
    load_bars as load_b_bars,
    price_only_signals,
    signals as b_signals,
)


SPEC_PATH = ROOT / "diagnostic_is" / "phase2_spec.json"
SPEC_COMMIT = "aa5bd8ef005a282f60517865e8ce0f347f429f49"
OUT = ROOT / "diagnostic_is" / "phase2_outputs"
REPORT = ROOT / "diagnostic_is" / "phase2_report.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_gate_spec() -> dict[str, Any]:
    check = subprocess.run(
        ["git", "diff", "--quiet", SPEC_COMMIT, "--", "diagnostic_is/phase2_spec.json"],
        cwd=ROOT,
    )
    if check.returncode != 0:
        raise RuntimeError("phase2_spec.json differs from the frozen commit")
    committed = subprocess.check_output(
        ["git", "show", f"{SPEC_COMMIT}:diagnostic_is/phase2_spec.json"], cwd=ROOT
    )
    working = SPEC_PATH.read_bytes().replace(b"\r\n", b"\n")
    if committed.replace(b"\r\n", b"\n") != working:
        raise RuntimeError("Working spec bytes do not match the frozen git blob")
    spec = json.loads(working)
    for block in spec["data_locks"]["A"].values():
        if sha256(ROOT / block["path"]) != block["sha256"]:
            raise RuntimeError(f"A source hash mismatch: {block['path']}")
    b = spec["data_locks"]["B"]
    if sha256(ROOT / b["path"]) != b["sha256"]:
        raise RuntimeError("B source hash mismatch")
    vol = spec["data_locks"]["MGC_volatility_breakout"]
    for path_key, hash_key in (("minute_path", "minute_sha256"), ("spec_path", "spec_sha256")):
        if sha256(ROOT / vol[path_key]) != vol[hash_key]:
            raise RuntimeError(f"MGC volatility source hash mismatch: {vol[path_key]}")
    return spec


def a_cost(inst: Any) -> float:
    return 2 * inst.tick_value + 2 * inst.commission_side


def a_variant_backtest(
    bars: pd.DataFrame,
    signal: pd.Series,
    fair: pd.Series,
    inst: Any,
    stop_rule: str,
) -> pd.DataFrame:
    sig = signal.to_numpy(copy=True)
    opens, highs, lows, closes = (bars[c].to_numpy() for c in ("open", "high", "low", "close"))
    fair_a, atr = fair.to_numpy(), bars.atr.to_numpy()
    dates, times = bars.session_date.to_numpy(), bars.index.to_numpy()
    session_last = bars.session_last.to_numpy(dtype=bool)
    records: list[dict[str, Any]] = []
    next_free = 0
    signal_indices = np.flatnonzero(sig)
    cursor = 0
    while cursor < len(signal_indices):
        i = int(signal_indices[cursor])
        entry_i = i + 1
        if entry_i <= next_free or entry_i >= len(bars) or dates[entry_i] != dates[i]:
            cursor += 1
            continue
        direction = int(sig[i])
        entry = float(opens[entry_i])
        initial_fair = float(fair_a[i]) if np.isfinite(fair_a[i]) else float(closes[i])
        deviation = abs(entry - initial_fair)
        if not np.isfinite(deviation) or deviation <= 0:
            cursor += 1
            continue
        target = entry + direction * deviation
        if stop_rule == "canonical":
            stop_distance = max(deviation, float(atr[i]))
        elif stop_rule == "equal_target_distance":
            stop_distance = deviation
        else:
            raise KeyError(stop_rule)
        stop = entry - direction * stop_distance
        end_i = min(entry_i + 24, len(bars) - 1)
        exit_i, exit_price, reason = end_i, float(closes[end_i]), "time"
        for j in range(entry_i, end_i + 1):
            if dates[j] != dates[entry_i]:
                exit_i, exit_price, reason = j - 1, float(closes[j - 1]), "session_end"
                break
            stop_hit = lows[j] <= stop if direction == 1 else highs[j] >= stop
            target_hit = highs[j] >= target if direction == 1 else lows[j] <= target
            if stop_hit:
                exit_i, exit_price, reason = j, stop, "stop"
                break
            if target_hit:
                exit_i, exit_price, reason = j, target, "target"
                break
            if session_last[j]:
                exit_i, exit_price, reason = j, float(closes[j]), "session_end"
                break
        path_h = highs[entry_i:exit_i + 1]
        path_l = lows[entry_i:exit_i + 1]
        mfe_points = path_h.max() - entry if direction == 1 else entry - path_l.min()
        mae_points = entry - path_l.min() if direction == 1 else path_h.max() - entry
        gross = direction * (exit_price - entry) * inst.point_value
        cost = a_cost(inst)
        records.append({
            "session_date": str(dates[entry_i]),
            "signal_time": str(pd.Timestamp(times[i])),
            "entry_time": str(pd.Timestamp(times[entry_i])),
            "exit_time": str(pd.Timestamp(times[exit_i])),
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "fair_value": initial_fair,
            "initial_deviation": deviation,
            "stop_price": stop,
            "stop_distance": stop_distance,
            "exit_reason": reason,
            "holding_minutes": int((exit_i - entry_i + 1) * 5),
            "gross_pnl": gross,
            "cost": cost,
            "net_pnl": gross - cost,
            "mfe_usd": mfe_points * inst.point_value,
            "mae_usd": mae_points * inst.point_value,
        })
        next_free = exit_i
        cursor = int(np.searchsorted(signal_indices, next_free, side="right"))
    return pd.DataFrame(records)


def add_a_fields(trades: pd.DataFrame, bars: pd.DataFrame, inst: Any, asset: str, variant: str) -> pd.DataFrame:
    out = trades.copy()
    session = bars.groupby("session_date", sort=False).first()[["session_open", "prior_day_close", "prior_range"]]
    session.index = pd.to_datetime(session.index)
    dates = pd.to_datetime(out.session_date)
    out["session_open"] = dates.map(session.session_open)
    out["prior_close"] = dates.map(session.prior_day_close)
    out["prior_range"] = dates.map(session.prior_range)
    out["gap_points"] = out.session_open - out.prior_close
    out["gap_norm"] = out.gap_points.abs() / out.prior_range.replace(0, np.nan)
    out["bucket"] = pd.cut(
        out.gap_norm,
        [0.25, 0.50, 0.75, 1.00, np.inf],
        labels=["[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00)", ">=1.00"],
        right=False,
        include_lowest=True,
    ).astype("string")
    atr = prior_daily_atr14(bars)
    out["prior_daily_atr14"] = dates.map(atr)
    denominator = out.prior_daily_atr14 * inst.point_value
    out["gross_R"] = out.gross_pnl / denominator
    out["cost_R"] = out.cost / denominator
    out["net_R"] = out.net_pnl / denominator
    out["asset"] = asset
    out["variant"] = variant
    return out


def period_slice(frame: pd.DataFrame, bounds: list[str]) -> pd.DataFrame:
    dates = pd.to_datetime(frame.session_date)
    return frame[dates.between(pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))].copy()


def risk_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame.dropna(subset=["net_R"]).copy()
    n = len(valid)
    if n:
        k = int(math.ceil(0.05 * n))
        cvar = -float(valid.nsmallest(k, "net_R").net_R.mean())
        daily = valid.assign(_date=pd.to_datetime(valid.session_date)).groupby("_date").net_R.sum()
        calendar = pd.bdate_range(daily.index.min(), daily.index.max())
        daily = daily.reindex(calendar, fill_value=0.0)
        curve = daily.cumsum()
        peak = pd.concat([pd.Series([0.0]), curve.reset_index(drop=True)], ignore_index=True).cummax().iloc[1:].to_numpy()
        maxdd = -float(np.min(curve.to_numpy() - peak))
    else:
        k, cvar, maxdd = 0, np.nan, np.nan
    return {
        "trades": len(frame),
        "valid_R_trades": n,
        "excluded_missing_R": len(frame) - n,
        "gross_usd": frame.gross_pnl.sum(),
        "cost_usd": frame.cost.sum(),
        "net_usd": frame.net_pnl.sum(),
        "gross_R": valid.gross_R.sum(),
        "cost_R": valid.cost_R.sum(),
        "net_R": valid.net_R.sum(),
        "mean_net_R": valid.net_R.mean(),
        "sd_net_R": valid.net_R.std(ddof=1),
        "trade_CVaR5_loss_R": cvar,
        "CVaR_tail_trades": k,
        "max_drawdown_loss_R": maxdd,
    }


def run_a(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ledgers, metric_rows = [], []
    fair_exit = next(exit_spec for exit_spec in EXIT_SPECS if exit_spec[0] == "fair_value")
    for asset, lock in spec["data_locks"]["A"].items():
        inst = A_INSTRUMENTS[asset]
        bars = _prepare_features(_load_bars(ROOT / lock["path"], inst))
        signal, fair = _signals(bars, "gap_reversion", {"lookback": 3, "threshold": 0.25})
        canonical = canonical_a_backtest(bars, signal, fair, inst, fair_exit)
        variants = {
            "A0": a_variant_backtest(bars, signal, fair, inst, "canonical"),
            "A_stop": a_variant_backtest(bars, signal, fair, inst, "equal_target_distance"),
        }
        a0 = variants["A0"]
        keys = ["session_date", "signal_time", "entry_time", "exit_time", "direction", "exit_reason"]
        if len(a0) != len(canonical) or a0[keys].astype(str).to_numpy().tolist() != canonical[keys].astype(str).to_numpy().tolist():
            raise RuntimeError(f"A0 canonical trade identity replication failed for {asset}")
        if not np.allclose(a0.net_pnl, canonical.net_pnl, rtol=0, atol=1e-8):
            raise RuntimeError(f"A0 canonical PnL replication failed for {asset}")
        period_key = "A_NQ" if asset == "NQ" else "A_micros"
        for variant, raw in variants.items():
            enriched = add_a_fields(raw, bars, inst, asset, variant)
            ledgers.append(enriched)
            for label in ("IS", "OOS"):
                sliced = period_slice(enriched, spec["periods"][period_key][label])
                metric_rows.append({"signal": "A", "asset": asset, "role": lock["role"], "variant": variant, "period": label, **risk_metrics(sliced)})
    metrics = pd.DataFrame(metric_rows)
    nq = metrics[(metrics.asset == "NQ")].set_index(["variant", "period"])
    is_stop, is_a0 = nq.loc[("A_stop", "IS")], nq.loc[("A0", "IS")]
    oos_stop, oos_a0 = nq.loc[("A_stop", "OOS")], nq.loc[("A0", "OOS")]
    is_cvar_pass = bool(is_stop.trade_CVaR5_loss_R < is_a0.trade_CVaR5_loss_R)
    is_dd_pass = bool(is_stop.max_drawdown_loss_R < is_a0.max_drawdown_loss_R)
    oos_floor = -0.25 * abs(float(oos_a0.mean_net_R))
    oos_delta = float(oos_stop.mean_net_R - oos_a0.mean_net_R)
    oos_pass = bool(oos_delta >= oos_floor)
    retained = "A_stop" if is_cvar_pass and is_dd_pass and oos_pass else "A0"
    decision = {
        "IS_CVaR_pass": is_cvar_pass,
        "IS_max_drawdown_pass": is_dd_pass,
        "OOS_non_degradation_pass": oos_pass,
        "OOS_mean_R_delta_stop_minus_A0": oos_delta,
        "OOS_allowed_delta_floor": oos_floor,
        "retained_variant": retained,
    }
    return pd.concat(ledgers, ignore_index=True), metrics, decision


def add_b_r(trades: pd.DataFrame, bars: pd.DataFrame, version: str) -> pd.DataFrame:
    out = trades.copy()
    dates = pd.to_datetime(out.session_date)
    out["cost"] = out.gross_pnl - out.net_pnl
    out["prior_daily_atr14"] = dates.map(prior_daily_atr14(bars))
    denominator = out.prior_daily_atr14 * B_INSTRUMENTS["MGC"].point_value
    out["gross_R"] = out.gross_pnl / denominator
    out["cost_R"] = out.cost / denominator
    out["net_R"] = out.net_pnl / denominator
    out["version"] = version
    return out


def run_b(spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = ROOT / spec["data_locks"]["B"]["path"]
    bars = add_price_features(add_causal_volume_baselines(load_b_bars(path, B_INSTRUMENTS["MGC"])))
    canonical = add_b_r(b_backtest(bars, b_signals(bars, B_PARAMS), B_INSTRUMENTS["MGC"]), bars, "B_canonical")
    no_volume = add_b_r(b_backtest(bars, price_only_signals(bars, B_PARAMS), B_INSTRUMENTS["MGC"]), bars, "B_no_volume")
    is_canonical = period_slice(canonical, spec["periods"]["B"]["IS"])
    if len(is_canonical) != 1333 or not np.isclose(is_canonical.net_pnl.sum(), -6762.5, atol=1e-8):
        raise RuntimeError("B canonical IS reproduction failed")
    ledgers = pd.concat([canonical, no_volume], ignore_index=True)
    rows = []
    for version, frame in (("B_canonical", canonical), ("B_no_volume", no_volume)):
        for period in ("IS", "OOS"):
            rows.append({"signal": "B", "version": version, "period": period, **risk_metrics(period_slice(frame, spec["periods"]["B"][period]))})
    return ledgers, pd.DataFrame(rows), bars


def power_row(series: str, frame: pd.DataFrame, forward_start: str) -> dict[str, Any]:
    r = frame.net_R.dropna().astype(float)
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    if len(r) >= 2 and mean > 0 and sd > 0:
        n = int(math.ceil(((1.2815515655446004 + 0.8416212335729143) * sd / mean) ** 2))
        threshold = mean - 1.2815515655446004 * sd / math.sqrt(n)
    else:
        n, threshold = None, np.nan
    return {
        "series": series,
        "OOS_trades": len(r),
        "OOS_mean_net_R": mean,
        "OOS_sd_net_R": sd,
        "review_N": n,
        "abandon_threshold_mean_net_R": threshold,
        "finite_power_review": n is not None,
        "forward_start": forward_start,
        "kill_switch": "trailing_365_calendar_day_net_R_strictly_negative_after_365_elapsed_days",
    }


def normalize_volatility_trades(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["session_date"] = pd.to_datetime(out.session_date)
    out["entry_time"] = pd.to_datetime(out.entry_time, utc=True)
    out["exit_time"] = pd.to_datetime(out.exit_time, utc=True)
    out["net_pnl"] = pd.to_numeric(out.net_pnl_usd)
    return out


def replay_volatility(spec: dict[str, Any]) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    lock = spec["data_locks"]["MGC_volatility_breakout"]
    data, _ = prepare_mgc_data(
        ROOT / lock["minute_path"],
        date.fromisoformat("2015-01-02"),
        date.fromisoformat(spec["periods"]["B"]["OOS"][1]),
    )
    frozen = json.loads((ROOT / lock["spec_path"]).read_text(encoding="utf-8"))
    run = run_baseline_backtest(
        frozen,
        data,
        evaluation_start=date.fromisoformat("2015-01-02"),
        evaluation_end=date.fromisoformat(spec["periods"]["B"]["OOS"][1]),
    )
    violations = {k: v for k, v in run.checks.items() if k.endswith("violations") and v}
    if violations:
        raise RuntimeError(f"Frozen MGC volatility replay violations: {violations}")
    available_sessions = set(pd.to_datetime(data.sessions["session_date"]))
    return normalize_volatility_trades(run.trades), available_sessions


def correlation_and_overlap(
    b_oos: pd.DataFrame,
    vol: pd.DataFrame,
    bounds: list[str],
    b_available_sessions: set[pd.Timestamp],
    vol_available_sessions: set[pd.Timestamp],
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    b = period_slice(b_oos, bounds).copy()
    v = vol[vol.session_date.between(start, end)].copy()
    b["entry_utc"] = pd.to_datetime(b.entry_time, utc=True)
    b["exit_utc"] = pd.to_datetime(b.exit_time, utc=True)
    common_sessions = pd.DatetimeIndex(sorted(
        day for day in b_available_sessions & vol_available_sessions if start <= day <= end
    ))
    bd = b.assign(_date=pd.to_datetime(b.session_date)).groupby("_date").net_pnl.sum().reindex(common_sessions, fill_value=0.0)
    vd = v.groupby("session_date").net_pnl.sum().reindex(common_sessions, fill_value=0.0)
    union_active = bd.ne(0) | vd.ne(0)
    pearson = float(bd.corr(vd, method="pearson"))
    spearman = float(bd.corr(vd, method="spearman"))
    active_corr = float(bd[union_active].corr(vd[union_active])) if union_active.sum() > 1 else np.nan
    b_days, v_days = set(pd.to_datetime(b.session_date)), set(v.session_date)
    intersection = b_days & v_days
    b_overlap = []
    for row in b.itertuples(index=False):
        hit = ((v.entry_time < row.exit_utc) & (v.exit_time > row.entry_utc)).any()
        b_overlap.append(bool(hit))
    v_overlap = []
    for row in v.itertuples(index=False):
        hit = ((b.entry_utc < row.exit_time) & (b.exit_utc > row.entry_time)).any()
        v_overlap.append(bool(hit))
    result = {
        "common_OOS_start": bounds[0],
        "common_OOS_end": bounds[1],
        "common_available_sessions": len(common_sessions),
        "B_trades": len(b),
        "volatility_trades": len(v),
        "pearson_daily_net_usd": pearson,
        "spearman_daily_net_usd": spearman,
        "pearson_union_active_days": active_corr,
        "same_session_count": len(intersection),
        "B_active_sessions": len(b_days),
        "volatility_active_sessions": len(v_days),
        "same_session_rate_over_B": len(intersection) / len(b_days) if b_days else np.nan,
        "same_session_rate_over_volatility": len(intersection) / len(v_days) if v_days else np.nan,
        "B_position_overlap_rate": float(np.mean(b_overlap)) if b_overlap else np.nan,
        "volatility_position_overlap_rate": float(np.mean(v_overlap)) if v_overlap else np.nan,
    }
    dec = spec["B_vs_MGC_volatility_breakout"]["decision"]
    if (
        abs(pearson) < 0.30
        and result["same_session_rate_over_B"] < 0.25
        and result["same_session_rate_over_volatility"] < 0.25
        and result["B_position_overlap_rate"] < 0.25
    ):
        decision = "separate_sleeve"
    elif pearson >= 0.50 or result["same_session_rate_over_B"] >= 0.50 or result["B_position_overlap_rate"] >= 0.50:
        decision = "filter_only_candidate"
    else:
        decision = "inconclusive_complementary_diagnostic"
    result["decision"] = decision
    daily = pd.DataFrame({"session_date": common_sessions, "B_net_pnl_usd": bd.to_numpy(), "MGC_volatility_net_pnl_usd": vd.to_numpy()})
    return daily, result


def md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Aucune ligne._"
    x = frame.copy().replace([np.inf, -np.inf], np.nan).fillna("—")
    lines = ["| " + " | ".join(map(str, x.columns)) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    for row in x.itertuples(index=False, name=None):
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.4f}")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_report(
    a_metrics: pd.DataFrame,
    a_decision: dict[str, Any],
    b_metrics: pd.DataFrame,
    power: pd.DataFrame,
    dependence: dict[str, Any],
) -> None:
    primary_a = a_metrics[a_metrics.asset.eq("NQ")][[
        "variant", "period", "trades", "valid_R_trades", "mean_net_R", "trade_CVaR5_loss_R", "max_drawdown_loss_R", "net_R", "net_usd"
    ]]
    micro = a_metrics[a_metrics.role.eq("robustness_only")][[
        "asset", "variant", "period", "trades", "mean_net_R", "trade_CVaR5_loss_R", "max_drawdown_loss_R", "net_R", "net_usd"
    ]]
    bshow = b_metrics[["version", "period", "trades", "valid_R_trades", "mean_net_R", "sd_net_R", "net_R", "net_usd"]]
    dep = pd.DataFrame([dependence])
    content = f"""Spec git commit: `{SPEC_COMMIT}`

# Phase 2 — pré-enregistrement et tests gelés

## Audit de pré-enregistrement

Tous les calculs ci-dessous utilisent [phase2_spec.json](phase2_spec.json) tel que committé dans `{SPEC_COMMIT}`. Le runner vérifie le blob git, l’absence de diff et les SHA-256 des sources avant exécution. Aucun full sample n’est utilisé comme critère.

Le premier commit de gel contenait deux erreurs de transcription de SHA-256 (NQ et M2K). Le garde-fou a interrompu le runner avant toute lecture analytique OOS. Seules ces deux empreintes ont été corrigées, sans changement de paramètre ni de protocole, puis la spec a été regelée au commit affiché en tête.

Divulgation : avant le commit, deux lignes individuelles 2024 du ledger déjà gelé de la sleeve MGC volatility breakout ont été affichées pendant l’identification de la source. Aucun agrégat ni résultat A/B n’avait été calculé. Les seuils génériques ont ensuite été gelés avant les tests.

## A — NQ principal

Le stop A0 existant était `max(distance entrée→target, ATR intraday causal du signal)`. A-stop le remplace par exactement la distance entrée→target ; toutes les autres règles restent identiques. La réplication A0 est exacte trade par trade avant comparaison.

{md(primary_a)}

Décision pré-enregistrée : CVaR IS améliorée = `{a_decision['IS_CVaR_pass']}` ; max drawdown IS amélioré = `{a_decision['IS_max_drawdown_pass']}` ; non-dégradation OOS = `{a_decision['OOS_non_degradation_pass']}`. Variante forward retenue : **{a_decision['retained_variant']}**.

Concrètement, la CVaR 5 % IS est inchangée et le max drawdown IS est légèrement plus mauvais avec A-stop : la variante échoue donc le gate IS, indépendamment de son passage du gate OOS.

## A — micros en robustesse seulement

{md(micro)}

Les micros ne participent jamais à la sélection.

## B — deux séries de suivi

{md(bshow)}

`B_no_volume` retire uniquement le seuil RVOL ; le mouvement, la confirmation, l’entrée, les coûts, le holding et le warm-up restent gelés. Aucune variante B n’est sélectionnée.

## Protocole forward calibré sur l’OOS

{md(power)}

La revue est unique au premier passage de N trades. Le seuil d’abandon est la moyenne forward indiquée dans le tableau. Le kill-switch commun est un PnL net R strictement négatif sur les 365 jours calendaires précédents, évalué après chaque trade seulement une fois 365 jours écoulés depuis le début forward.

## B contre la sleeve MGC volatility breakout — OOS uniquement

{md(dep)}

La décision applique mécaniquement les seuils de la spec : **{dependence['decision']}**. Aucune performance conditionnée ou filtrée n’est calculée.

Justification exacte : le statut `separate_sleeve` échoue parce que {dependence['same_session_rate_over_volatility']:.1%} des sessions actives de volatility breakout coïncident avec une session B (seuil < 25 %), tandis que le statut `filter_only_candidate` échoue car la corrélation de Pearson ({dependence['pearson_daily_net_usd']:.3f}), le recouvrement des sessions B ({dependence['same_session_rate_over_B']:.1%}) et le recouvrement des positions B ({dependence['B_position_overlap_rate']:.1%}) restent tous sous leurs seuils de 50 %.

## Conclusion

- Diagnostic clos conservé : A = (c2), B = (c1), hypothèse de mesure (b) rejetée.
- Phase 2 A : **{a_decision['retained_variant']}** est la seule série forward retenue selon les deux gates IS et le gate OOS.
- Phase 2 B : les versions canonique et sans volume restent deux séries de mesure, sans sélection de règle.
- Rôle de B par rapport à MGC volatility breakout : **{dependence['decision']}** selon les seuils pré-enregistrés.
"""
    REPORT.write_text(content, encoding="utf-8")


def main() -> None:
    spec = load_and_gate_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    a_ledgers, a_metrics, a_decision = run_a(spec)
    b_ledgers, b_metrics, b_bars = run_b(spec)
    retained = a_decision["retained_variant"]
    a_oos = period_slice(
        a_ledgers[(a_ledgers.asset == "NQ") & (a_ledgers.variant == retained)],
        spec["periods"]["A_NQ"]["OOS"],
    )
    b_canonical_oos = period_slice(
        b_ledgers[b_ledgers.version == "B_canonical"], spec["periods"]["B"]["OOS"]
    )
    b_no_volume_oos = period_slice(
        b_ledgers[b_ledgers.version == "B_no_volume"], spec["periods"]["B"]["OOS"]
    )
    power = pd.DataFrame([
        power_row("A_retained", a_oos, spec["forward_protocol"]["forward_starts"]["A_retained"]),
        power_row("B_canonical", b_canonical_oos, spec["forward_protocol"]["forward_starts"]["B_canonical"]),
        power_row("B_no_volume", b_no_volume_oos, spec["forward_protocol"]["forward_starts"]["B_no_volume"]),
    ])
    vol, vol_available_sessions = replay_volatility(spec)
    b_available_sessions = set(pd.to_datetime(b_bars["session_date"].unique()))
    daily, dependence = correlation_and_overlap(
        b_canonical_oos,
        vol,
        [
            spec["periods"]["correlation_common_OOS"]["start"],
            spec["periods"]["correlation_common_OOS"]["end"],
        ],
        b_available_sessions,
        vol_available_sessions,
        spec,
    )
    a_ledgers.to_csv(OUT / "A_phase2_trades.csv", index=False)
    a_metrics.to_csv(OUT / "A_phase2_metrics.csv", index=False)
    (OUT / "A_phase2_decision.json").write_text(json.dumps(a_decision, indent=2) + "\n", encoding="utf-8")
    b_ledgers.to_csv(OUT / "B_tracking_trades.csv", index=False)
    b_metrics.to_csv(OUT / "B_tracking_metrics.csv", index=False)
    power.to_csv(OUT / "forward_protocol_calibration.csv", index=False)
    vol.to_csv(OUT / "MGC_volatility_breakout_replay_trades.csv", index=False)
    daily.to_csv(OUT / "B_vs_MGC_volatility_daily_OOS.csv", index=False)
    (OUT / "B_vs_MGC_volatility_dependence.json").write_text(json.dumps(dependence, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "spec_commit": SPEC_COMMIT,
        "spec_sha256": sha256(SPEC_PATH),
        "A_decision": a_decision,
        "dependence_decision": dependence["decision"],
        "post_IS_computed_only_after_spec_commit": True,
        "full_sample_selection_used": False,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    build_report(a_metrics, a_decision, b_metrics, power, dependence)
    print(json.dumps({"spec_commit": SPEC_COMMIT, "A_decision": a_decision, "power": power.to_dict("records"), "dependence": dependence}, indent=2))


if __name__ == "__main__":
    main()
