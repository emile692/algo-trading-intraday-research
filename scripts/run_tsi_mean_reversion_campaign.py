"""Run the pre-specified, bounded TSI intraday mean-reversion campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.tsi_mean_reversion_campaign import (  # noqa: E402
    CampaignSpec, THRESHOLDS, TSI_NEIGHBOURHOOD, add_indicators, build_event_study,
    collision_analysis, detect_events, matched_controls, moving_block_bootstrap,
    prepare_rth_bars, primary_gate, simulate_trades, spec_dict, stability_tables,
    summarize_event_study, trade_metrics,
)


INPUTS = {
    "MNQ": ROOT / "data/processed/parquet/MNQ_c_0_5m_20260321_094501.parquet",
    "MES": ROOT / "data/processed/parquet/MES_c_0_5m_20260322_135702.parquet",
    "M2K": ROOT / "data/processed/parquet/M2K_c_0_5m_20260322_134808.parquet",
}
LOCKED_DIR = ROOT / "research/locked_four_sleeve_portfolio"
SLEEVES = ("orb_vwap_atr", "volatility_breakout", "compression_breakout", "atr_trend_flip")
DEFAULT_OUTPUT = ROOT / "research/tsi_mean_reversion_20261002"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hashes() -> dict[str, str]:
    paths = [LOCKED_DIR / "locked_four_sleeve_config.json", LOCKED_DIR / "daily_pnl.csv"]
    paths += [LOCKED_DIR / "sleeve_trades" / f"{name}.csv" for name in SLEEVES]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def _paired_bootstrap_delta(primary: pd.DataFrame, control: pd.DataFrame, spec: CampaignSpec) -> dict[str, float]:
    n = min(len(primary), len(control))
    if n == 0:
        return {"n": 0, "delta_usd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    delta = primary.iloc[:n]["net_pnl_usd"].to_numpy(float) - control.iloc[:n]["net_pnl_usd"].to_numpy(float)
    rng = np.random.default_rng(spec.random_seed + 1)
    samples = np.array([rng.choice(delta, size=n, replace=True).mean() for _ in range(spec.bootstrap_samples)])
    return {"n": n, "delta_usd": float(delta.mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975))}


def _topstep_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Daily-resolution 50K diagnostic, only called after the robustness gate."""
    daily = trades.groupby("session_date")["net_pnl_usd"].sum().sort_index()
    rows = []
    for start in range(max(0, len(daily) - 59)):
        path = daily.iloc[start:start + 60].cumsum()
        outcome = "EXPIRY"
        target_hits, loss_hits = np.flatnonzero(path.to_numpy() >= 3000.0), np.flatnonzero(path.to_numpy() <= -2000.0)
        first_target = int(target_hits[0]) if len(target_hits) else 10**9
        first_loss = int(loss_hits[0]) if len(loss_hits) else 10**9
        if first_target < first_loss:
            outcome = "PASS"
        elif first_loss < first_target:
            outcome = "MLL"
        rows.append({"start_date": daily.index[start], "outcome": outcome, "ending_pnl_usd": float(path.iloc[-1])})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame([{"account": "50K", "attempts": 0, "pass_rate": np.nan, "mll_rate": np.nan, "expiry_rate": np.nan}])
    rates = detail["outcome"].value_counts(normalize=True)
    return pd.DataFrame([{"account": "50K", "attempts": len(detail), "pass_rate": rates.get("PASS", 0.0),
                          "mll_rate": rates.get("MLL", 0.0), "expiry_rate": rates.get("EXPIRY", 0.0),
                          "profit_target_usd": 3000.0, "maximum_loss_usd": 2000.0, "horizon_sessions": 60}])


def _write_skipped(path: Path, reason: str) -> None:
    pd.DataFrame([{"status": "SKIPPED_BY_PRIMARY_GATE", "reason": reason}]).to_csv(path, index=False)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown renderer for audit tables."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _report(verdict: str, gate: dict[str, object], metrics: pd.DataFrame, event_summary: pd.DataFrame,
            annual: pd.DataFrame, subperiod: pd.DataFrame, downstream: str, topstep_run: bool) -> str:
    primary = metrics.iloc[0]
    sixty = event_summary.loc[(event_summary.symbol == "MNQ") & (event_summary.direction == "long") & (event_summary.horizon_minutes == 60)]
    lines = [
        "# TSI intraday mean reversion — campagne strictement bornée", "", f"## Verdict : `{verdict}`", "",
        f"Le gate primaire est **{'ouvert' if gate['passed'] else 'fermé'}**. {downstream}", "",
        "## Résultat primaire MNQ long", "",
        f"- Trades : **{int(primary.n)}** ; expectancy nette : **{primary.net_expectancy_usd:+.2f} USD** ; PF : **{primary.profit_factor:.3f}**.",
        f"- Années positives : **{gate['positive_year_ratio']:.1%}** ; trois sous-périodes positives : **{'oui' if gate['all_subperiods_positive'] else 'non'}**.",
        f"- Monotonie −15/−25/−35 à 60 minutes : **{'oui' if gate['monotone_60m_thresholds'] else 'non'}**.", "",
        "| Seuil | n | Rendement brut moyen (bp) | PnL net moyen (USD) | MFE médiane (pts) | MAE médiane (pts) | Coût RT break-even (USD) |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sixty.sort_values("threshold").itertuples():
        lines.append(f"| −{row.threshold:g} | {row.n} | {row.mean_gross_return_bps:+.3f} | {row.mean_net_pnl_usd:+.2f} | {row.median_mfe_points:.2f} | {row.median_mae_points:.2f} | {row.break_even_round_trip_cost_usd:+.2f} |")
    lines += ["", "## Stabilité", "", "### Annuelle", "", _markdown_table(annual), "", "### Sous-périodes figées", "", _markdown_table(subperiod), "",
              "## Protocole", "",
              "- Barres 5 minutes RTH `[09:30,16:00)` America/New_York ; TSI canonique `25/13`, signal EMA `7`.",
              "- L'armement extrême et le croisement doivent appartenir à la même séance. Entrée au prochain open.",
              "- Sortie : stop `1,5 ATR(14)` prioritaire dans la barre, puis TSI zéro, 12 barres ou dernière clôture RTH.",
              "- Coûts : 1 tick défavorable et 1,25 USD par côté ; un seul trade ouvert par instrument.",
              "- Les seuils −15/−25/−35 ne servent qu'au test de monotonie ; aucun meilleur seuil n'est sélectionné.",
              "- Gate primaire : expectancy nette `−25` positive, monotonie non décroissante stricte des expectancies 60 min, ≥60 % d'années positives et 3/3 sous-périodes positives.",
              f"- Simulateur Topstep exécuté : **{'oui' if topstep_run else 'non'}**.", "",
              "Tous les artefacts aval non calculés portent explicitement le statut `SKIPPED_BY_PRIMARY_GATE`. Les ledgers des quatre sleeves MNQ ne sont chargés pour l'analyse qu'après survie du gate ; leurs fichiers sont seulement hashés avant/après pour prouver leur invariance."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    spec = CampaignSpec(bootstrap_samples=args.bootstrap_samples)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    hashes_before = frozen_hashes()

    raw = pd.read_parquet(INPUTS["MNQ"])
    bars = add_indicators(prepare_rth_bars(raw), atr_period=spec.atr_period)
    studies, ledgers, threshold_metrics = [], [], []
    event_cache: dict[tuple[float, int], pd.DataFrame] = {}
    for threshold in THRESHOLDS:
        for side in (1, -1):
            events = detect_events(bars, threshold=threshold, side=side)
            event_cache[(threshold, side)] = events
            studies.append(build_event_study(bars, events, symbol="MNQ"))
            trades = simulate_trades(bars, events, symbol="MNQ", max_holding_bars=spec.max_holding_bars, stop_atr=spec.stop_atr)
            ledgers.append(trades)
            threshold_metrics.append(trade_metrics(trades, symbol="MNQ", direction="long" if side == 1 else "short", threshold=threshold))
    event_ledger = pd.concat(studies, ignore_index=True)
    event_summary = summarize_event_study(event_ledger)
    trade_ledger = pd.concat(ledgers, ignore_index=True)
    metrics = pd.DataFrame(threshold_metrics)
    primary_trades = trade_ledger.loc[(trade_ledger.direction == "long") & (trade_ledger.threshold == spec.primary_threshold)].copy()
    annual, subperiod = stability_tables(primary_trades)
    passed, gate_details = primary_gate(event_summary, primary_trades, annual, subperiod, spec)
    gate = {"passed": passed, **gate_details}

    event_ledger.to_csv(out / "event_ledger.csv", index=False)
    event_summary.to_csv(out / "event_study_grid.csv", index=False)
    trade_ledger.to_csv(out / "trade_ledger.csv", index=False)
    metrics.to_csv(out / "threshold_strategy_grid.csv", index=False)
    annual.to_csv(out / "annual_stability.csv", index=False)
    subperiod.to_csv(out / "subperiod_stability.csv", index=False)
    primary_trades.to_csv(out / "primary_trade_ledger.csv", index=False)

    verdict, topstep_run = "REJECTED", False
    downstream = "Les validations aval et le simulateur Topstep n'ont pas été exécutés conformément à la règle d'arrêt."
    validation = {"primary_gate": gate}
    downstream_files = ["block_bootstrap.csv", "tsi_parameter_neighbourhood.csv", "external_validation.csv",
                        "matched_event_comparison.csv", "sleeve_collisions.csv", "sleeve_daily_correlations.csv", "topstep_summary.csv"]
    if not passed:
        reason = "primary signal failed positivity, monotonicity, or temporal stability"
        for name in downstream_files:
            _write_skipped(out / name, reason)
    else:
        bootstrap = moving_block_bootstrap(primary_trades, spec)
        bootstrap.to_csv(out / "block_bootstrap.csv", index=False)
        neighbourhood_rows = []
        for long_span, short_span, signal_span in TSI_NEIGHBOURHOOD:
            variant = add_indicators(prepare_rth_bars(raw), long_span=long_span, short_span=short_span, signal_span=signal_span, atr_period=spec.atr_period)
            events = detect_events(variant, threshold=spec.primary_threshold, side=1)
            trades = simulate_trades(variant, events, symbol="MNQ")
            neighbourhood_rows.append(trade_metrics(trades, long_span=long_span, short_span=short_span, signal_span=signal_span))
        neighbourhood = pd.DataFrame(neighbourhood_rows)
        neighbourhood.to_csv(out / "tsi_parameter_neighbourhood.csv", index=False)
        external_rows = []
        for symbol in ("MES", "M2K"):
            ext_bars = add_indicators(prepare_rth_bars(pd.read_parquet(INPUTS[symbol])))
            ext_events = detect_events(ext_bars, threshold=spec.primary_threshold, side=1)
            ext_trades = simulate_trades(ext_bars, ext_events, symbol=symbol)
            external_rows.append(trade_metrics(ext_trades, symbol=symbol))
        external = pd.DataFrame(external_rows)
        external.to_csv(out / "external_validation.csv", index=False)
        _, control_trades = matched_controls(bars, event_cache[(spec.primary_threshold, 1)], symbol="MNQ", seed=spec.random_seed)
        matched = _paired_bootstrap_delta(primary_trades.sort_values("entry_time"), control_trades.sort_values("entry_time"), spec)
        pd.DataFrame([matched]).to_csv(out / "matched_event_comparison.csv", index=False)
        sleeve_ledgers = {name: pd.read_csv(LOCKED_DIR / "sleeve_trades" / f"{name}.csv") for name in SLEEVES}
        collisions, correlations = collision_analysis(primary_trades, sleeve_ledgers, pd.read_csv(LOCKED_DIR / "daily_pnl.csv"))
        collisions.to_csv(out / "sleeve_collisions.csv", index=False)
        correlations.to_csv(out / "sleeve_daily_correlations.csv", index=False)
        robust = (float(bootstrap.iloc[0].ci95_low) > 0 and (neighbourhood.net_expectancy_usd > 0).mean() >= 7 / 9
                  and bool((external.net_expectancy_usd > 0).all()) and matched["ci95_low"] > 0)
        validation.update({"bootstrap_ci_low_positive": bool(float(bootstrap.iloc[0].ci95_low) > 0),
                           "positive_neighbourhood_ratio": float((neighbourhood.net_expectancy_usd > 0).mean()),
                           "both_external_positive": bool((external.net_expectancy_usd > 0).all()), "matched_control": matched,
                           "robust_zone_survived": bool(robust)})
        if robust:
            topstep = _topstep_summary(primary_trades)
            topstep.to_csv(out / "topstep_summary.csv", index=False)
            topstep_run = True
            compatible = float(topstep.iloc[0].pass_rate) >= .20 and float(topstep.iloc[0].mll_rate) <= .40
            verdict = "ROBUST_CANDIDATE" if compatible else "WEAK_NON_ACTIONABLE"
            downstream = "La zone robuste a survécu ; le diagnostic Topstep 50K a donc été exécuté."
        else:
            _write_skipped(out / "topstep_summary.csv", "no robust zone survived full validation")
            verdict = "WEAK_NON_ACTIONABLE"
            downstream = "Le gate primaire a survécu, mais pas la validation robuste complète ; Topstep reste fermé."

    hashes_after = frozen_hashes()
    frozen_unchanged = hashes_before == hashes_after
    if not frozen_unchanged:
        raise RuntimeError("A frozen four-sleeve artifact changed during the campaign")
    metadata = {"verdict": verdict, "primary_gate": gate, "validation": validation, "topstep_simulation_run": topstep_run,
                "frozen_sleeves_modified": False, "frozen_artifact_hashes": hashes_after, "specification": spec_dict(spec),
                "thresholds_diagnostic_only": list(THRESHOLDS), "tsi_neighbourhood": [list(x) for x in TSI_NEIGHBOURHOOD],
                "inputs": {symbol: {"path": str(path.resolve()), "sha256": sha256(path)} for symbol, path in INPUTS.items()}}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (out / "REPORT.md").write_text(_report(verdict, gate, metrics.loc[(metrics.direction == "long") & (metrics.threshold == spec.primary_threshold)], event_summary, annual, subperiod, downstream, topstep_run), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "primary_gate": passed, "topstep_simulation_run": topstep_run, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
