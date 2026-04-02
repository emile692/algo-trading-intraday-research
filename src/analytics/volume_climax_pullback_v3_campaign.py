"""Focused V3 campaign for the volume climax pullback standalone strategy."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.volume_climax_pullback_common import (
    filter_trades_by_sessions,
    load_latest_reference_run,
    load_symbol_data,
    pf_for_ranking,
    resample_rth_1h,
    safe_float,
    split_sessions,
    summarize_scope,
)
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    build_volume_climax_pullback_v3_variants,
    prepare_volume_climax_pullback_v2_features,
)

SYMBOLS = ("MNQ", "MES", "M2K", "MGC")
DEFAULT_OUTPUT_ROOT = Path("data/exports/volume_climax_pullback_v3_run")
DEFAULT_V2_REFERENCE_ROOT = Path("data/exports/volume_climax_pullback_v2_run")
DEFAULT_V2_REFERENCE_PREFIX = "volume_climax_pullback_v2_"
SMALL_SAMPLE_OOS_TRADES = 6

PRIMARY_V2_REFERENCE_BY_ASSET: dict[str, str] = {
    "MNQ": "dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2",
    "MES": "dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p2",
    "M2K": "dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2",
    "MGC": "regime_filtered_trend_ema50_medium_vq0p975_bf0p5_ra1p2",
}

MGC_DYNAMIC_V2_REFERENCE = "dynamic_exit_mixed_ts4_vq0p95_bf0p5_ra1p2"
EXPLICIT_DYNAMIC_V2_ANCHORS = {
    "dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2",
    "dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p2",
    "dynamic_exit_mixed_ts4_vq0p95_bf0p5_ra1p2",
}


def _float_or_nan(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float("nan")


def _variant_branch(symbol: str, variant: VolumeClimaxPullbackV2Variant) -> str:
    if str(symbol).upper() == "MGC" and variant.family == "regime_filtered":
        return "mgc_dynamic_exit_plus_regime"
    return "dynamic_exit_only"


def _ema_filter_label(variant: VolumeClimaxPullbackV2Variant) -> str:
    if variant.trend_ema_window is None or variant.ema_slope_threshold is None:
        return "off"
    if int(variant.trend_ema_window) == 50 and abs(float(variant.ema_slope_threshold) - 0.06) <= 1e-9:
        return "mild"
    return f"custom_ema{int(variant.trend_ema_window)}_{variant.ema_slope_threshold}"


def _atr_band_label(variant: VolumeClimaxPullbackV2Variant) -> str:
    if variant.atr_percentile_low is None or variant.atr_percentile_high is None:
        return "off"
    low = int(round(float(variant.atr_percentile_low) * 100))
    high = int(round(float(variant.atr_percentile_high) * 100))
    if (low, high) == (20, 80):
        return "20_80"
    if (low, high) == (30, 70):
        return "30_70"
    return f"{low}_{high}"


def _compression_filter_label(variant: VolumeClimaxPullbackV2Variant) -> str:
    if variant.compression_ratio_max is None:
        return "off"
    if abs(float(variant.compression_ratio_max) - 0.90) <= 1e-9:
        return "mild"
    return f"custom_{variant.compression_ratio_max}"


def _variant_descriptor(symbol: str, variant: VolumeClimaxPullbackV2Variant) -> dict[str, Any]:
    descriptor = asdict(variant)
    descriptor["variant_name"] = variant.name
    descriptor["generation"] = "v3"
    descriptor["branch"] = _variant_branch(symbol, variant)
    descriptor["ema_slope_filter"] = _ema_filter_label(variant)
    descriptor["atr_percentile_band"] = _atr_band_label(variant)
    descriptor["compression_filter"] = _compression_filter_label(variant)
    descriptor["is_explicit_v2_dynamic_anchor"] = bool(variant.name in EXPLICIT_DYNAMIC_V2_ANCHORS)
    return descriptor


def _stability_ratio(is_sharpe: Any, oos_sharpe: Any) -> float:
    is_value = _float_or_nan(is_sharpe)
    oos_value = _float_or_nan(oos_sharpe)
    if not np.isfinite(is_value) or abs(is_value) <= 1e-9 or not np.isfinite(oos_value):
        return float("nan")
    return float(oos_value / is_value)


def _resolve_reference_row(reference_summary: pd.DataFrame, symbol: str, target_variant_name: str) -> pd.Series:
    symbol_frame = reference_summary.loc[reference_summary["symbol"] == symbol].copy()
    if symbol_frame.empty:
        raise ValueError(f"No V2 reference rows available for {symbol}.")

    explicit_match = symbol_frame.loc[symbol_frame["variant_name"] == target_variant_name].copy()
    if not explicit_match.empty:
        return explicit_match.iloc[0]

    fallback = symbol_frame.sort_values(
        ["selection_score", "oos_sharpe", "oos_net_pnl"],
        ascending=[False, False, False],
    )
    return fallback.iloc[0]


def _build_reference_views(reference_summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    primary_rows: list[dict[str, Any]] = []
    secondary_rows: dict[str, pd.Series] = {}

    for symbol in SYMBOLS:
        primary_reference = _resolve_reference_row(reference_summary, symbol, PRIMARY_V2_REFERENCE_BY_ASSET[symbol])
        primary_rows.append(
            {
                "symbol": symbol,
                "v2_reference_variant_name": str(primary_reference["variant_name"]),
                "v2_reference_family": str(primary_reference.get("family", "")),
                "v2_reference_oos_net_pnl": _float_or_nan(primary_reference.get("oos_net_pnl")),
                "v2_reference_oos_profit_factor": _float_or_nan(primary_reference.get("oos_profit_factor")),
                "v2_reference_oos_sharpe": _float_or_nan(primary_reference.get("oos_sharpe")),
                "v2_reference_oos_expectancy": _float_or_nan(primary_reference.get("oos_expectancy")),
                "v2_reference_oos_nb_trades": int(_float_or_nan(primary_reference.get("oos_nb_trades"))),
                "v2_reference_stability_is_oos_sharpe_ratio": _float_or_nan(
                    primary_reference.get("stability_is_oos_sharpe_ratio")
                ),
                "v2_reference_selection_score": _float_or_nan(primary_reference.get("selection_score")),
            }
        )

    secondary_rows["MGC_dynamic_exit"] = _resolve_reference_row(reference_summary, "MGC", MGC_DYNAMIC_V2_REFERENCE)
    return pd.DataFrame(primary_rows), secondary_rows


def _selection_score(row: pd.Series) -> float:
    if int(row.get("oos_nb_trades", 0)) <= 0:
        return -99.0

    stability = _float_or_nan(row.get("stability_is_oos_sharpe_ratio"))
    score = float(
        1.25 * np.tanh(safe_float(row.get("oos_sharpe")) / 1.5)
        + 0.45 * np.tanh(pf_for_ranking(row.get("oos_profit_factor")) - 1.0)
        + 0.35 * np.tanh(safe_float(row.get("oos_net_pnl")) / 500.0)
        + 0.30 * np.tanh(safe_float(row.get("oos_expectancy")) / 20.0)
        + 0.25 * np.tanh(safe_float(row.get("delta_oos_sharpe_vs_v2")) / 0.75)
        + 0.20 * np.tanh(safe_float(row.get("delta_oos_net_pnl_vs_v2")) / 400.0)
        + 0.15 * min(safe_float(row.get("oos_nb_trades")) / 12.0, 1.0)
    )
    if np.isfinite(stability):
        score += float(0.20 * np.tanh(stability / 1.0))
    if bool(row.get("anomaly_small_oos_sample")):
        score -= 0.55
    if bool(row.get("anomaly_profit_factor_inf")):
        score -= 0.35
    if bool(row.get("anomaly_undefined_stability")):
        score -= 0.30
    return score


def _status_label(row: pd.Series) -> str:
    if int(row.get("oos_nb_trades", 0)) == 0:
        return "dead"
    if (
        safe_float(row.get("oos_net_pnl")) <= 0.0
        or safe_float(row.get("oos_profit_factor")) <= 1.0
        or safe_float(row.get("oos_sharpe")) <= 0.0
    ):
        return "dead"
    if (
        bool(row.get("anomaly_small_oos_sample"))
        or bool(row.get("anomaly_profit_factor_inf"))
        or bool(row.get("anomaly_undefined_stability"))
    ):
        return "fragile"
    return "survivor"


def _status_rank(status: str) -> int:
    if status == "survivor":
        return 0
    if status == "fragile":
        return 1
    return 2


def _spec_label(row: pd.Series) -> str:
    return (
        f"{row['variant_name']} | exit={row['exit_mode']} ts={int(row['time_stop_bars'])} "
        f"| vq={safe_float(row['volume_quantile']):.3f} bf={safe_float(row['min_body_fraction']):.1f} "
        f"ra={safe_float(row['min_range_atr']):.1f} | regime={row['ema_slope_filter']}/{row['atr_percentile_band']}/{row['compression_filter']}"
    )


def _asset_verdict(asset_frame: pd.DataFrame) -> dict[str, Any]:
    ordered = asset_frame.sort_values(
        ["status_rank", "selection_score", "oos_sharpe", "delta_oos_sharpe_vs_v2", "oos_net_pnl"],
        ascending=[True, False, False, False, False],
    ).copy()
    live = ordered.loc[ordered["variant_status"] != "dead"].copy()
    clean = ordered.loc[ordered["variant_status"] == "survivor"].copy()

    if live.empty:
        return {
            "symbol": str(asset_frame.iloc[0]["symbol"]),
            "verdict": "non_recommandee",
            "reason": "Aucune variante V3 n'est vivante OOS.",
            "recommended_row": None,
            "clean_survivor_count": 0,
            "live_variant_count": 0,
        }

    recommended = clean.iloc[0] if not clean.empty else live.iloc[0]
    delta_sharpe = safe_float(recommended.get("delta_oos_sharpe_vs_v2"))
    delta_pnl = safe_float(recommended.get("delta_oos_net_pnl_vs_v2"))

    if str(recommended["variant_status"]) == "survivor" and delta_sharpe >= -0.15 and delta_pnl >= -150.0:
        verdict = "recommandee"
        reason = "Survivant propre OOS avec profil encore comparable a la reference V2."
    elif str(recommended["variant_status"]) == "survivor":
        verdict = "survivante_sous_reference_v2"
        reason = "Survivant propre OOS, mais encore sous la reference V2 retenue."
    else:
        verdict = "survivant_fragile"
        reason = "Variant vivant mais encore penalise par les anomalies ou un echantillon OOS trop mince."

    return {
        "symbol": str(recommended["symbol"]),
        "verdict": verdict,
        "reason": reason,
        "recommended_row": recommended,
        "clean_survivor_count": int(len(clean)),
        "live_variant_count": int(len(live)),
    }


def _mgc_regime_value(asset_frame: pd.DataFrame, secondary_reference: pd.Series) -> dict[str, str]:
    dynamic = asset_frame.loc[asset_frame["family"] == "dynamic_exit"].copy()
    regime = asset_frame.loc[asset_frame["family"] == "regime_filtered"].copy()
    if dynamic.empty or regime.empty:
        return {"verdict": "inconclusive", "reason": "Une des deux branches MGC est absente."}

    dynamic_best = dynamic.sort_values(
        ["status_rank", "selection_score", "oos_sharpe", "oos_net_pnl"],
        ascending=[True, False, False, False],
    ).iloc[0]
    regime_best = regime.sort_values(
        ["status_rank", "selection_score", "oos_sharpe", "oos_net_pnl"],
        ascending=[True, False, False, False],
    ).iloc[0]

    if str(regime_best["variant_status"]) == "dead":
        return {"verdict": "negative", "reason": "La branche regime_filtered ne survit pas en OOS sur MGC."}

    if safe_float(regime_best["selection_score"]) > safe_float(dynamic_best["selection_score"]) + 0.15:
        return {
            "verdict": "positive",
            "reason": "Le filtre de regime ajoute une vraie valeur nette sur MGC face au meilleur dynamic_exit pur.",
        }

    dynamic_drawdown = abs(safe_float(dynamic_best.get("oos_max_drawdown")))
    regime_drawdown = abs(safe_float(regime_best.get("oos_max_drawdown")))
    dynamic_pnl = safe_float(dynamic_best.get("oos_net_pnl"))
    regime_pnl = safe_float(regime_best.get("oos_net_pnl"))
    if regime_drawdown < dynamic_drawdown and regime_pnl >= 0.85 * dynamic_pnl:
        return {
            "verdict": "defensive_only",
            "reason": "Le filtre de regime aide surtout en defense sur MGC, sans vrai saut offensif net.",
        }

    return {
        "verdict": "cosmetic",
        "reason": (
            "Le filtre de regime reste surtout cosmetique sur MGC par rapport au meilleur dynamic_exit "
            f"V2 `{secondary_reference['variant_name']}` et au meilleur V3 dynamique."
        ),
    }


def _build_final_report(
    run_dir: Path,
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    reference_dir: Path,
    secondary_reference_rows: dict[str, pd.Series],
) -> dict[str, Any]:
    asset_payloads = []
    for symbol in SYMBOLS:
        asset_frame = summary.loc[summary["symbol"] == symbol].copy()
        asset_payloads.append(_asset_verdict(asset_frame))

    verdict_by_symbol = {payload["symbol"]: payload for payload in asset_payloads}
    ready_assets = sum(payload["verdict"] == "recommandee" for payload in asset_payloads)
    viable_assets = sum(payload["verdict"] != "non_recommandee" for payload in asset_payloads)
    clean_survivors_total = int((summary["variant_status"] == "survivor").sum())

    if ready_assets >= 3 and viable_assets == len(SYMBOLS) and clean_survivors_total >= 4:
        global_conclusion = "strategie prete pour paper trading cible"
    else:
        global_conclusion = "strategie encore trop fragile"

    mgc_payload = summary.loc[summary["symbol"] == "MGC"].copy()
    mgc_regime_value = _mgc_regime_value(mgc_payload, secondary_reference_rows["MGC_dynamic_exit"])

    dynamic_anchor_rows = summary.loc[summary["is_explicit_v2_dynamic_anchor"]].copy()
    dynamic_anchor_survivors = int((dynamic_anchor_rows["variant_status"] != "dead").sum())
    dynamic_anchor_total = int(len(dynamic_anchor_rows))

    answers = {
        "stabilize_dynamic_exit": "yes"
        if ready_assets >= 2 and int((summary["delta_stability_vs_v2"] >= -0.10).sum()) >= 2
        else "partial_or_no",
        "best_exit_time_stop_by_asset": {
            payload["symbol"]: None
            if payload["recommended_row"] is None
            else f"{payload['recommended_row']['exit_mode']} / ts{int(payload['recommended_row']['time_stop_bars'])}"
            for payload in asset_payloads
        },
        "mgc_regime_value": mgc_regime_value["verdict"],
        "v2_dynamic_anchors_survive": f"{dynamic_anchor_survivors}/{dynamic_anchor_total}",
        "few_competing_variants": "yes"
        if all(payload["clean_survivor_count"] <= 3 for payload in asset_payloads if payload["recommended_row"] is not None)
        else "no",
    }

    anomaly_zero_trade = summary.loc[summary["anomaly_oos_zero_trade"]].copy()
    anomaly_pf_inf = summary.loc[summary["anomaly_profit_factor_inf"]].copy()
    anomaly_small_sample = summary.loc[summary["anomaly_small_oos_sample"]].copy()
    anomaly_undefined_stability = summary.loc[summary["anomaly_undefined_stability"]].copy()

    lines = [
        "# Volume Climax Pullback V3 - Final Report",
        "",
        "## Scope",
        "- Universe: `MNQ`, `MES`, `M2K`, `MGC`.",
        "- Timeframe: `1h` RTH only, one open position max, flat end of session, leak-free convention inherited from V2.",
        f"- Tested V3 rows: `{len(summary)}`.",
        f"- V2 reference run: `{reference_dir}`.",
        "",
        "## Verdicts By Asset",
    ]

    for payload in asset_payloads:
        symbol = payload["symbol"]
        lines.append(f"### {symbol}")
        lines.append(f"- Verdict: `{payload['verdict']}`.")
        lines.append(f"- Reason: {payload['reason']}")
        if payload["recommended_row"] is None:
            lines.append("- Recommended spec: `none`.")
            continue

        row = payload["recommended_row"]
        lines.extend(
            [
                f"- Recommended spec: `{_spec_label(row)}`.",
                f"- OOS Sharpe / PF / Net PnL / Trades: `{safe_float(row['oos_sharpe']):.3f}` / `{safe_float(row['oos_profit_factor']):.3f}` / `{safe_float(row['oos_net_pnl']):.2f}` / `{int(row['oos_nb_trades'])}`.",
                f"- Stability IS/OOS: `{safe_float(row['stability_is_oos_sharpe_ratio'], default=np.nan):.3f}`.",
                f"- Vs V2 `{row['v2_reference_variant_name']}`: delta Sharpe `{safe_float(row['delta_oos_sharpe_vs_v2']):.3f}`, delta Net PnL `{safe_float(row['delta_oos_net_pnl_vs_v2']):.2f}`.",
                f"- Clean survivors / live variants: `{payload['clean_survivor_count']}` / `{payload['live_variant_count']}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Research Answers",
            f"1. Dynamic exit stabilization without losing the core PnL: `{answers['stabilize_dynamic_exit']}`.",
            (
                "2. Best exit_mode / time_stop by asset: "
                + ", ".join(
                    f"`{symbol}={value or 'n/a'}`" for symbol, value in answers["best_exit_time_stop_by_asset"].items()
                )
                + "."
            ),
            f"3. MGC regime filter value: `{answers['mgc_regime_value']}`. {mgc_regime_value['reason']}",
            (
                "4. V2 survivors under the stricter grid: "
                f"`{answers['v2_dynamic_anchors_survive']}` explicit dynamic anchors remain alive. "
                "The historical MGC fixed-RR regime winner is kept as comparison reference only."
            ),
            f"5. Recommended specs with few concurrent contenders: `{answers['few_competing_variants']}`.",
            "",
            "## Anomalies / Faux Survivants",
            f"- 0 trade OOS: `{len(anomaly_zero_trade)}`.",
            f"- Profit factor inf: `{len(anomaly_pf_inf)}`.",
            f"- Echantillons trop petits (<{SMALL_SAMPLE_OOS_TRADES} trades OOS): `{len(anomaly_small_sample)}`.",
            f"- Ratios de stabilite non definis: `{len(anomaly_undefined_stability)}`.",
        ]
    )

    if not anomaly_zero_trade.empty:
        zero_examples = anomaly_zero_trade.sort_values(["symbol", "variant_name"]).head(5)
        lines.append(
            "- Examples 0 trade OOS: "
            + ", ".join(f"`{row.symbol}:{row.variant_name}`" for row in zero_examples.itertuples())
            + "."
        )
    if not anomaly_pf_inf.empty:
        pf_examples = anomaly_pf_inf.sort_values(["selection_score", "symbol"], ascending=[False, True]).head(5)
        lines.append(
            "- Examples profit_factor inf: "
            + ", ".join(f"`{row.symbol}:{row.variant_name}`" for row in pf_examples.itertuples())
            + "."
        )
    if not anomaly_small_sample.empty:
        sample_examples = anomaly_small_sample.sort_values(["selection_score", "symbol"], ascending=[False, True]).head(5)
        lines.append(
            "- Examples small OOS samples: "
            + ", ".join(
                f"`{row.symbol}:{row.variant_name} ({int(row.oos_nb_trades)} trades)`" for row in sample_examples.itertuples()
            )
            + "."
        )
    if not anomaly_undefined_stability.empty:
        stability_examples = anomaly_undefined_stability.sort_values(["selection_score", "symbol"], ascending=[False, True]).head(5)
        lines.append(
            "- Examples undefined stability: "
            + ", ".join(f"`{row.symbol}:{row.variant_name}`" for row in stability_examples.itertuples())
            + "."
        )

    lines.extend(
        [
            "",
            "## Global Conclusion",
            f"- Conclusion: `{global_conclusion}`.",
            "- Paper-trading readiness rule: at least three assets recommended, zero asset fully rejected, and at least four clean survivors overall.",
        ]
    )

    report_path = run_dir / "final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "global_conclusion": global_conclusion,
        "ready_assets": ready_assets,
        "viable_assets": viable_assets,
        "clean_survivors_total": clean_survivors_total,
        "mgc_regime_value": mgc_regime_value["verdict"],
        "verdict_by_symbol": {
            symbol: {
                "verdict": verdict_by_symbol[symbol]["verdict"],
                "recommended_variant_name": None
                if verdict_by_symbol[symbol]["recommended_row"] is None
                else str(verdict_by_symbol[symbol]["recommended_row"]["variant_name"]),
            }
            for symbol in SYMBOLS
        },
    }
    (run_dir / "final_verdict.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _normalize_max_workers(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    return 1


def _chunk_variants(
    variants: list[VolumeClimaxPullbackV2Variant],
    *,
    max_workers: int,
) -> list[list[VolumeClimaxPullbackV2Variant]]:
    if max_workers <= 1 or len(variants) <= 1:
        return [variants]
    target_tasks = max_workers * 2
    chunk_size = max(1, math.ceil(len(variants) / target_tasks))
    return [variants[idx : idx + chunk_size] for idx in range(0, len(variants), chunk_size)]


def _evaluate_symbol_variant_chunk(
    *,
    symbol: str,
    features: pd.DataFrame,
    is_sessions: list,
    oos_sessions: list,
    variants: list[VolumeClimaxPullbackV2Variant],
) -> list[dict[str, Any]]:
    execution_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")
    rows: list[dict[str, Any]] = []

    for variant in variants:
        signal_df = build_volume_climax_pullback_v2_signal_frame(features, variant)
        trades = run_volume_climax_pullback_v2_backtest(signal_df, variant, execution_model, instrument).trades

        is_signal = signal_df.loc[pd.to_datetime(signal_df["session_date"]).dt.date.isin(is_sessions)].copy()
        oos_signal = signal_df.loc[pd.to_datetime(signal_df["session_date"]).dt.date.isin(oos_sessions)].copy()
        is_trades = filter_trades_by_sessions(trades, is_sessions)
        oos_trades = filter_trades_by_sessions(trades, oos_sessions)
        is_metrics = summarize_scope(is_trades, is_signal, is_sessions)
        oos_metrics = summarize_scope(oos_trades, oos_signal, oos_sessions)

        rows.append(
            {
                "symbol": symbol,
                **_variant_descriptor(symbol, variant),
                **{f"is_{key}": value for key, value in is_metrics.items()},
                **{f"oos_{key}": value for key, value in oos_metrics.items()},
                "stability_is_oos_sharpe_ratio": _stability_ratio(is_metrics["sharpe"], oos_metrics["sharpe"]),
            }
        )

    return rows


def run_campaign(
    output_root: Path,
    *,
    symbols: tuple[str, ...] = SYMBOLS,
    variants_by_symbol: dict[str, list[VolumeClimaxPullbackV2Variant]] | None = None,
    input_paths: dict[str, Path] | None = None,
    v2_reference_dir: Path | None = None,
    max_workers: int | None = None,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"volume_climax_pullback_v3_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_reference_dir = (
        Path(v2_reference_dir)
        if v2_reference_dir is not None
        else load_latest_reference_run(DEFAULT_V2_REFERENCE_ROOT, DEFAULT_V2_REFERENCE_PREFIX)
    )
    reference_summary = pd.read_csv(resolved_reference_dir / "summary_variants.csv")
    reference_view, secondary_reference_rows = _build_reference_views(reference_summary)

    rows: list[dict[str, Any]] = []
    variant_count_by_symbol: dict[str, int] = {}
    symbol_contexts: list[dict[str, Any]] = []
    resolved_max_workers = _normalize_max_workers(max_workers)

    for symbol in symbols:
        symbol_variants = (variants_by_symbol or {}).get(symbol) or build_volume_climax_pullback_v3_variants(symbol)
        variant_count_by_symbol[symbol] = len(symbol_variants)

        raw = load_symbol_data(symbol, input_paths=input_paths)
        bars = resample_rth_1h(raw)
        if bars.empty:
            raise ValueError(f"No RTH 1h bars available for {symbol}.")

        session_frame = bars.copy()
        session_frame["timestamp"] = pd.to_datetime(session_frame["timestamp"], errors="coerce")
        session_frame["session_date"] = session_frame["timestamp"].dt.date
        is_sessions, oos_sessions = split_sessions(session_frame)

        symbol_contexts.append(
            {
                "symbol": symbol,
                "features": prepare_volume_climax_pullback_v2_features(bars),
                "is_sessions": is_sessions,
                "oos_sessions": oos_sessions,
                "variants": symbol_variants,
            }
        )

    if resolved_max_workers <= 1:
        for context in symbol_contexts:
            rows.extend(
                _evaluate_symbol_variant_chunk(
                    symbol=context["symbol"],
                    features=context["features"],
                    is_sessions=context["is_sessions"],
                    oos_sessions=context["oos_sessions"],
                    variants=context["variants"],
                )
            )
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=resolved_max_workers) as executor:
            for context in symbol_contexts:
                for chunk in _chunk_variants(context["variants"], max_workers=resolved_max_workers):
                    futures.append(
                        executor.submit(
                            _evaluate_symbol_variant_chunk,
                            symbol=context["symbol"],
                            features=context["features"],
                            is_sessions=context["is_sessions"],
                            oos_sessions=context["oos_sessions"],
                            variants=chunk,
                        )
                    )
            for future in as_completed(futures):
                rows.extend(future.result())

    summary = pd.DataFrame(rows)
    summary = summary.merge(reference_view, on="symbol", how="left")

    summary["delta_oos_net_pnl_vs_v2"] = pd.to_numeric(summary["oos_net_pnl"], errors="coerce") - pd.to_numeric(
        summary["v2_reference_oos_net_pnl"], errors="coerce"
    )
    summary["delta_oos_profit_factor_vs_v2"] = pd.to_numeric(summary["oos_profit_factor"], errors="coerce") - pd.to_numeric(
        summary["v2_reference_oos_profit_factor"], errors="coerce"
    )
    summary["delta_oos_sharpe_vs_v2"] = pd.to_numeric(summary["oos_sharpe"], errors="coerce") - pd.to_numeric(
        summary["v2_reference_oos_sharpe"], errors="coerce"
    )
    summary["delta_oos_expectancy_vs_v2"] = pd.to_numeric(summary["oos_expectancy"], errors="coerce") - pd.to_numeric(
        summary["v2_reference_oos_expectancy"], errors="coerce"
    )
    summary["delta_stability_vs_v2"] = pd.to_numeric(
        summary["stability_is_oos_sharpe_ratio"], errors="coerce"
    ) - pd.to_numeric(summary["v2_reference_stability_is_oos_sharpe_ratio"], errors="coerce")
    summary["relative_oos_net_pnl_vs_v2"] = np.where(
        pd.to_numeric(summary["v2_reference_oos_net_pnl"], errors="coerce").abs() > 1e-9,
        pd.to_numeric(summary["oos_net_pnl"], errors="coerce") / pd.to_numeric(summary["v2_reference_oos_net_pnl"], errors="coerce"),
        np.nan,
    )

    oos_pf = pd.to_numeric(summary["oos_profit_factor"], errors="coerce")
    summary["anomaly_oos_zero_trade"] = pd.to_numeric(summary["oos_nb_trades"], errors="coerce").fillna(0).astype(int) == 0
    summary["anomaly_profit_factor_inf"] = np.isinf(oos_pf.to_numpy(dtype=float))
    summary["anomaly_small_oos_sample"] = pd.to_numeric(summary["oos_nb_trades"], errors="coerce").fillna(0).astype(int) < SMALL_SAMPLE_OOS_TRADES
    summary["anomaly_undefined_stability"] = ~np.isfinite(
        pd.to_numeric(summary["stability_is_oos_sharpe_ratio"], errors="coerce").to_numpy(dtype=float)
    )
    summary["anomaly_count"] = (
        summary["anomaly_oos_zero_trade"].astype(int)
        + summary["anomaly_profit_factor_inf"].astype(int)
        + summary["anomaly_small_oos_sample"].astype(int)
        + summary["anomaly_undefined_stability"].astype(int)
    )
    summary["selection_score"] = summary.apply(_selection_score, axis=1)
    summary["variant_status"] = summary.apply(_status_label, axis=1)
    summary["status_rank"] = summary["variant_status"].map(_status_rank).fillna(3).astype(int)
    summary["is_clean_survivor"] = summary["variant_status"].eq("survivor")

    summary = summary.sort_values(
        ["symbol", "branch", "family", "selection_score", "variant_name"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    summary.to_csv(run_dir / "summary_variants.csv", index=False)

    ranking = summary.loc[summary["variant_status"] != "dead"].copy()
    ranking = ranking.sort_values(
        ["is_clean_survivor", "selection_score", "oos_sharpe", "delta_oos_sharpe_vs_v2", "oos_net_pnl"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    ranking.to_csv(run_dir / "ranking_oos.csv", index=False)
    ranking.sort_values(
        ["symbol", "is_clean_survivor", "selection_score", "oos_sharpe"],
        ascending=[True, False, False, False],
    ).to_csv(run_dir / "ranking_oos_by_asset.csv", index=False)

    comparison = summary.sort_values(
        ["symbol", "is_clean_survivor", "selection_score", "delta_oos_sharpe_vs_v2", "delta_oos_net_pnl_vs_v2"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    comparison.to_csv(run_dir / "comparison_vs_v2.csv", index=False)

    verdict = _build_final_report(run_dir, summary, ranking, resolved_reference_dir, secondary_reference_rows)
    metadata = {
        "run_dir": str(run_dir),
        "symbols": list(symbols),
        "variant_count_by_symbol": variant_count_by_symbol,
        "generated_at_utc": datetime.utcnow().isoformat(),
        "v2_reference_dir": str(resolved_reference_dir),
        "max_workers": resolved_max_workers,
        "verdict": verdict,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the focused V3 volume climax pullback campaign.")
    parser.add_argument("--output-dir", default=None, help="Legacy output root flag.")
    parser.add_argument("--output-root", default=None, help="Output root directory.")
    parser.add_argument("--v2-reference-dir", default=None, help="Explicit V2 reference run directory.")
    parser.add_argument("--max-workers", type=int, default=None, help="Worker count for chunked variant evaluation.")
    args = parser.parse_args()

    output_root = Path(args.output_root or args.output_dir or DEFAULT_OUTPUT_ROOT)
    v2_reference_dir = None if args.v2_reference_dir is None else Path(args.v2_reference_dir)
    out = run_campaign(output_root, v2_reference_dir=v2_reference_dir, max_workers=args.max_workers)
    print(out)


if __name__ == "__main__":
    main()
