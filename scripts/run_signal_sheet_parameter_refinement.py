"""Map the principal parameter landscapes for signal sheets S03-S09.

This is a diagnostic campaign only.  It never edits frozen specifications,
portfolio configurations, or selects a winning cell.  Every candidate is a
deep copy of the current frozen/adapted specification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    load_frozen_specs,
    prepare_data,
    run_baseline_backtest,
)
from research.btc_baseline_backtests.run_btc_baselines import (  # noqa: E402
    adapt_spec_to_btc,
    prepare_btc_data,
)
from research.mbt_strategy_heatmaps_20261003.run_mbt_heatmaps import mbt_config  # noqa: E402
from research.mbt_three_sleeve_freeze_20261004.run_analysis import apply_cell  # noqa: E402
from research.mgc_baseline_backtests.run_mgc_baselines import (  # noqa: E402
    adapt_spec_to_mgc,
    prepare_mgc_data,
)
from research.money_management_lab.mm_engine import (  # noqa: E402
    build_overlay_context,
    run_exit_overlay_backtest,
    slice_result,
)


OUTPUT_ROOT = ROOT / "research" / "signal_sheet_parameter_refinement"
FROZEN_DIR = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = ROOT / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
MNQ_DATA = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
MGC_DATA = ROOT / "data" / "processed" / "parquet" / "MGC_v_0_1m_20150101_20260923.parquet"
MGC_CONFIG = ROOT / "research" / "mgc_baseline_backtests" / "mgc_baseline_config.json"
MBT_DATA = ROOT / "data" / "downloaded" / "MBT_v_0_1m_20210425_20261003.parquet"


CAMPAIGNS: dict[str, dict[str, Any]] = {
    "s03": {
        "sheet": "03_mnq_compression_breakout",
        "mechanism": "compression_breakout",
        "instrument": "MNQ",
        "start": "2019-05-06", "end": "2026-03-19", "is_end": "2023-12-31", "oos_start": "2024-01-01",
        "grids": {
            "nr_lookback": [6, 8, 10, 12, 14],
            "entry_buffer_ticks": [0, 1, 2],
            "target_setup_range_multiple": [1.5, 2.0, 2.5],
        },
        "reasons": {
            "nr_lookback": "Contrôle la rareté et la profondeur de la compression détectée.",
            "entry_buffer_ticks": "Arbitre entre faux breakouts et retard/coût d’entrée.",
            "target_setup_range_multiple": "Détermine directement le payoff et le taux de réussite des sorties.",
        },
    },
    "s04": {
        "sheet": "04_mnq_atr_trend_flip", "mechanism": "atr_trend_flip", "instrument": "MNQ",
        "start": "2019-05-06", "end": "2026-03-19", "is_end": "2023-12-31", "oos_start": "2024-01-01",
        "grids": {
            "atr_period": [6, 8, 10, 12, 14],
            "atr_multiplier": [1.25, 1.5, 1.75, 2.0, 2.25],
        },
        "reasons": {
            "atr_period": "Fixe la vitesse d’adaptation de la volatilité qui déplace les bandes.",
            "atr_multiplier": "Fixe directement la largeur des bandes et donc la fréquence des flips.",
        },
    },
    "s05": {
        "sheet": "05_mgc_volatility_breakout", "mechanism": "volatility_breakout", "instrument": "MGC",
        "start": "2019-05-06", "end": "2026-03-19", "is_end": "2023-12-31", "oos_start": "2024-01-01",
        "grids": {"daily_atr_period": [10, 20, 30, 40, 50], "entry_atr_multiple": [0.7, 0.9, 1.1, 1.3, 1.5]},
        "reasons": {
            "daily_atr_period": "Détermine le régime de volatilité de référence et la stabilité du seuil.",
            "entry_atr_multiple": "Détermine l’amplitude minimale du choc et la fréquence de déclenchement.",
        },
    },
    "s06": {
        "sheet": "06_mgc_gap_continuation", "mechanism": "gap_continuation", "instrument": "MGC",
        "start": "2019-05-06", "end": "2026-03-19", "is_end": "2023-12-31", "oos_start": "2024-01-01",
        "grids": {
            "minimum_gap_atr_multiple": [0.3, 0.4, 0.5, 0.6, 0.7],
            "volume_threshold_multiple": [0.4, 0.5, 0.6, 0.7, 0.8],
        },
        "reasons": {
            "minimum_gap_atr_multiple": "Filtre l’importance du gap relativement au régime de volatilité.",
            "volume_threshold_multiple": "Filtre la confirmation par participation et pilote le nombre de signaux.",
        },
    },
    "s07": {
        "sheet": "07_mbt_rolling_range_breakout", "mechanism": "rolling_range_breakout", "instrument": "MBT",
        "start": "2021-05-03", "end": "2026-10-02", "is_end": "2025-05-28", "oos_start": "2025-05-29",
        "grids": {"entry_channel_length": [16, 20, 24, 28, 32], "exit_channel_length": [5, 7, 10, 12, 15, 18]},
        "reasons": {
            "entry_channel_length": "Fixe la maturité du breakout exigé avant l’entrée.",
            "exit_channel_length": "Fixe la vitesse de restitution des gains et la durée des tendances captées.",
        },
    },
    "s08": {
        "sheet": "08_mbt_trend_pullback_ma_structure", "mechanism": "trend_pullback_ma_structure", "instrument": "MBT",
        "start": "2021-05-03", "end": "2026-10-02", "is_end": "2025-05-28", "oos_start": "2025-05-29",
        "grids": {"fast_pair_scale": [0.4, 0.5, 0.6, 0.75, 0.9], "slow_tail_scale": [0.6, 0.75, 1.0, 1.25, 1.5]},
        "reasons": {
            "fast_pair_scale": "Contrôle la réactivité des EMA rapides au pullback et à la reprise.",
            "slow_tail_scale": "Contrôle la persistance de la structure de tendance de fond.",
        },
    },
    "s09": {
        "sheet": "09_mbt_volatility_breakout", "mechanism": "volatility_breakout", "instrument": "MBT",
        "start": "2021-05-03", "end": "2026-10-02", "is_end": "2025-05-28", "oos_start": "2025-05-29",
        "grids": {"daily_atr_period": [7, 10, 14, 20, 30, 40], "entry_atr_multiple": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]},
        "reasons": {
            "daily_atr_period": "Fixe l’horizon du régime de volatilité utilisé pour normaliser le mouvement.",
            "entry_atr_multiple": "Fixe la distance de déclenchement et le compromis fréquence/sélectivité.",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (pd.Timestamp, date)): return str(value)
    return value


def metric_block(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="raise").fillna(0.0)
    std = float(pnl.std(ddof=1))
    sharpe = float(math.sqrt(252.0) * pnl.mean() / std) if std > 0 else float("nan")
    equity = np.r_[0.0, pnl.cumsum().to_numpy(float)]
    drawdown = equity - np.maximum.accumulate(equity)
    trade_pnl = pd.to_numeric(trades.get("net_pnl_usd", pd.Series(dtype=float)), errors="coerce").dropna()
    gains, losses = float(trade_pnl[trade_pnl > 0].sum()), float(-trade_pnl[trade_pnl < 0].sum())
    profit_factor = gains / losses if losses > 0 else (float("inf") if gains > 0 else float("nan"))
    return {
        "session_count": int(len(pnl)), "net_pnl_usd": float(pnl.sum()), "sharpe_annualized": sharpe,
        "profit_factor": profit_factor, "max_drawdown_usd": float(drawdown.min()),
        "trade_count": int(len(trades)), "worst_day_usd": float(pnl.min()),
    }


def scoped_metrics(run: Any, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    daily = run.daily.copy()
    trades = run.trades.copy()
    daily["session_date"] = pd.to_datetime(daily["session_date"])
    if not trades.empty:
        trades["session_date"] = pd.to_datetime(trades["session_date"])
    start, end = pd.Timestamp(campaign["start"]), pd.Timestamp(campaign["end"])
    is_end, oos_start = pd.Timestamp(campaign["is_end"]), pd.Timestamp(campaign["oos_start"])
    scopes = (("full", start, end), ("is", start, is_end), ("oos", oos_start, end))
    rows = []
    for scope, left, right in scopes:
        scoped_daily = daily[daily["session_date"].between(left, right)]
        scoped_trades = trades[trades["session_date"].between(left, right)] if not trades.empty else trades
        rows.append({"scope": scope, "start": str(left.date()), "end": str(right.date()), **metric_block(scoped_daily, scoped_trades)})
    return rows


def inventory(spec: dict[str, Any], campaign: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    canonical = spec["canonical_parameters"]
    for family, values in canonical.items():
        for name, value in values.items():
            principal = name in campaign["grids"]
            rows.append({
                "family": family, "name": name, "current_value": value,
                "principal": principal, "reason": campaign["reasons"].get(name),
                "tested_grid": campaign["grids"].get(name),
                "policy": "tested" if principal else "frozen_at_current_value",
            })
    if campaign["sheet"].startswith("04_"):
        rows.append({
            "family": "active_exit_overlay", "name": "stop_atr", "current_value": 1.0,
            "principal": False, "reason": "Sortie money-management active conservée pour isoler les paramètres du signal.",
            "tested_grid": None, "policy": "frozen_at_current_value",
        })
    return rows


def prepare_campaign(key: str, campaign: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    start, end = date.fromisoformat(campaign["start"]), date.fromisoformat(campaign["end"])
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    by_mechanism = {item["mechanism_id"]: item for item in specs}
    if campaign["instrument"] == "MNQ":
        prepared = prepare_data(MNQ_DATA, start_date=start, end_date=end)
        spec = copy.deepcopy(by_mechanism[campaign["mechanism"]])
        # Single-point locks the grids are centred on; superseded by frozen vote ensembles on 2026-11-01.
        reference_mnq_parameters = {
            "compression_breakout": {"nr_lookback": 10, "entry_buffer_ticks": 1},
            "atr_trend_flip": {"atr_period": 10, "atr_multiplier": 1.75},
        }
        spec["canonical_parameters"]["structural"].update(reference_mnq_parameters[campaign["mechanism"]])
        source = {"data_path": str(MNQ_DATA.relative_to(ROOT)), "data_sha256": sha256(MNQ_DATA)}
    elif campaign["instrument"] == "MGC":
        prepared, _ = prepare_mgc_data(MGC_DATA, start, end)
        mgc_config = json.loads(MGC_CONFIG.read_text(encoding="utf-8"))
        spec = adapt_spec_to_mgc(copy.deepcopy(by_mechanism[campaign["mechanism"]]), mgc_config)
        active_mgc_parameters = {
            "volatility_breakout": {"daily_atr_period": 30, "entry_atr_multiple": 1.10},
            "gap_continuation": {"minimum_gap_atr_multiple": 0.50, "volume_threshold_multiple": 0.60},
        }
        spec["canonical_parameters"]["structural"].update(active_mgc_parameters[campaign["mechanism"]])
        source_spec = FROZEN_DIR / f"{campaign['mechanism']}.json"
        source = {
            "data_path": str(MGC_DATA.relative_to(ROOT)), "data_sha256": sha256(MGC_DATA),
            "mgc_config_path": str(MGC_CONFIG.relative_to(ROOT)), "mgc_config_sha256": sha256(MGC_CONFIG),
            "source_spec_path": str(source_spec.relative_to(ROOT)), "source_spec_sha256": sha256(source_spec),
        }
    else:
        prepared, excluded_roll_sessions = prepare_btc_data(MBT_DATA, start, end)
        spec = adapt_spec_to_btc(copy.deepcopy(by_mechanism[campaign["mechanism"]]), mbt_config())
        source = {"data_path": str(MBT_DATA.relative_to(ROOT)), "data_sha256": sha256(MBT_DATA), "excluded_roll_sessions": [str(item) for item in excluded_roll_sessions]}
    return prepared, spec, source


def apply_parameters(spec: dict[str, Any], campaign: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(spec)
    mechanism = campaign["mechanism"]
    if campaign["instrument"] == "MBT":
        keys = list(campaign["grids"])
        if mechanism == "volatility_breakout":
            # apply_cell declares the MBT volatility axes in multiple, period order.
            candidate, _ = apply_cell(candidate, mechanism, params["entry_atr_multiple"], params["daily_atr_period"])
        else:
            candidate, _ = apply_cell(candidate, mechanism, params[keys[0]], params[keys[1]])
        return candidate
    structural = candidate["canonical_parameters"]["structural"]
    risk = candidate["canonical_parameters"]["risk_management"]
    for name, value in params.items():
        if name == "stop_atr":
            candidate["_refinement_stop_atr"] = float(value)
            continue
        if name in structural:
            structural[name] = scalar(value)
        elif name in risk:
            risk[name] = scalar(value)
        else:
            raise KeyError(f"Parameter {name} is not active in {mechanism}")
    return candidate


def run_candidate(key: str, candidate: dict[str, Any], prepared: Any, campaign: dict[str, Any]) -> Any:
    if key != "s04":
        return run_baseline_backtest(candidate, prepared)
    stop_atr = float(candidate.pop("_refinement_stop_atr", 1.0))
    timeframe = 15
    context = build_overlay_context(candidate, prepared, timeframe)
    run = run_exit_overlay_backtest(candidate, prepared, {"family": "atr_stop", "stop_atr": stop_atr}, timeframe, context=context)
    return slice_result(run, date.fromisoformat(campaign["start"]), date.fromisoformat(campaign["end"]))


def run_campaign(key: str) -> Path:
    campaign = CAMPAIGNS[key]
    output = OUTPUT_ROOT / key
    output.mkdir(parents=True, exist_ok=True)
    prepared, base_spec, source = prepare_campaign(key, campaign)
    parameter_names = list(campaign["grids"])
    combinations = list(itertools.product(*(campaign["grids"][name] for name in parameter_names)))
    checkpoint_path = output / "grid_metrics.partial.csv"
    checkpoint_checks = output / "cell_checks.partial.json"
    checkpoint_meta = output / "checkpoint_signature.txt"
    signature_payload = json.dumps({"campaign": campaign, "source": source}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    checkpoint_signature = hashlib.sha256(signature_payload).hexdigest()
    checkpoint_valid = checkpoint_meta.is_file() and checkpoint_meta.read_text(encoding="ascii").strip() == checkpoint_signature
    existing = pd.read_csv(checkpoint_path) if checkpoint_valid and checkpoint_path.is_file() else pd.DataFrame()
    rows: list[dict[str, Any]] = existing.to_dict("records")
    checks: list[dict[str, Any]] = json.loads(checkpoint_checks.read_text(encoding="utf-8")) if checkpoint_valid and checkpoint_checks.is_file() else []
    for index, values in enumerate(combinations, 1):
        params = dict(zip(parameter_names, values))
        if not existing.empty:
            mask = pd.Series(True, index=existing.index)
            for name, value in params.items():
                mask &= np.isclose(pd.to_numeric(existing[name]), float(value))
            if existing.loc[mask, "scope"].nunique() == 3:
                print(f"{key} {index}/{len(combinations)} checkpoint {params}", flush=True)
                continue
        candidate = apply_parameters(base_spec, campaign, params)
        run = run_candidate(key, candidate, prepared, campaign)
        violations = {name: value for name, value in run.checks.items() if name.endswith("violations") and value}
        if violations:
            raise RuntimeError(f"{key} {params}: {violations}")
        cell_id = f"{key}_{index:03d}"
        for metrics in scoped_metrics(run, campaign):
            rows.append({"cell_id": cell_id, **params, **metrics})
        checks.append({"cell_id": cell_id, "parameters": params, "checks": run.checks})
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        checkpoint_checks.write_text(json.dumps(checks, indent=2, ensure_ascii=False, default=scalar), encoding="utf-8")
        checkpoint_meta.write_text(checkpoint_signature, encoding="ascii")
        print(f"{key} {index}/{len(combinations)} {params}", flush=True)

    metrics_path = output / "grid_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    (output / "cell_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False, default=scalar), encoding="utf-8")
    inventory_rows = inventory(base_spec, campaign)
    (output / "parameter_inventory.json").write_text(json.dumps(inventory_rows, indent=2, ensure_ascii=False, default=scalar), encoding="utf-8")
    manifest = {
        "schema_version": "1.0", "purpose": "diagnostic_landscape_no_selection", "sheet": campaign["sheet"],
        "mechanism": campaign["mechanism"], "instrument": campaign["instrument"],
        "periods": {name: campaign[name] for name in ("start", "end", "is_end", "oos_start")},
        "principal_parameters": campaign["reasons"], "grids": campaign["grids"],
        "combination_count": len(combinations), "scope_count": 3,
        "all_other_parameters": "frozen_at_current_value", "selection_performed": False,
        "profit_factor_definition": "sum positive net trade PnL / absolute sum negative net trade PnL",
        "max_drawdown_definition": "minimum drawdown of cumulative daily net PnL including zero origin",
        "worst_day_definition": "minimum realized net daily PnL; prop-firm daily-loss diagnostic, not intraday trailing loss",
        "source": source,
        "outputs": {"grid_metrics": str(metrics_path.relative_to(ROOT)), "grid_metrics_sha256": sha256(metrics_path)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleeve", required=True, choices=sorted(CAMPAIGNS))
    args = parser.parse_args()
    print(run_campaign(args.sleeve))


if __name__ == "__main__":
    main()
