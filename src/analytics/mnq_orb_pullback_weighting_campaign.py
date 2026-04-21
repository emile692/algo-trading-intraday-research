"""MNQ ORB + volume pullback portfolio weighting campaign.

The campaign keeps both alpha engines fixed and only tests transparent
portfolio-combination rules on daily sleeve returns.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_REPO_ROOT))

from src.analytics.orb_multi_asset_campaign import resolve_processed_dataset
from src.analytics.orb_research.campaign import _evaluate_experiment, _experiment_from_json, _split_sessions
from src.analytics.orb_research.features import (
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import (
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignContext,
    CompressionConfig,
    DynamicThresholdConfig,
    ExitConfig,
    ExperimentConfig,
)
from src.analytics.volume_climax_pullback_common import (
    load_symbol_data,
    resample_rth_1h,
    split_sessions,
)
from src.config.paths import EXPORTS_DIR, REPO_ROOT, ensure_directories
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.risk.position_sizing import RiskPercentPositionSizing
from src.strategy.volume_climax_pullback_v2 import (
    VolumeClimaxPullbackV2Variant,
    build_volume_climax_pullback_v2_signal_frame,
    prepare_volume_climax_pullback_v2_features,
)


DEFAULT_SYMBOL = "MNQ"
DEFAULT_INITIAL_CAPITAL_USD = 50_000.0
DEFAULT_IS_FRACTION = 0.70
DEFAULT_OUTPUT_PREFIX = "mnq_orb_pullback_weighting_"
DEFAULT_ORB_CONFIG_NAME = "full_reopt__seed__pair__comp_dynamic__weak_close__noise_area_gate"
DEFAULT_PULLBACK_VARIANT_NAME = "dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2"
DEFAULT_PULLBACK_CAMPAIGN_VARIANT_NAME = "risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true"
DEFAULT_WEIGHT_PAIRS: tuple[tuple[float, float], ...] = (
    (0.20, 0.80),
    (0.25, 0.75),
    (0.33, 0.67),
    (0.40, 0.60),
    (0.50, 0.50),
    (0.60, 0.40),
    (0.67, 0.33),
    (0.75, 0.25),
    (0.80, 0.20),
)


@dataclass(frozen=True)
class WeightingCampaignConfig:
    symbol: str = DEFAULT_SYMBOL
    initial_capital_usd: float = DEFAULT_INITIAL_CAPITAL_USD
    is_fraction: float = DEFAULT_IS_FRACTION
    output_dir: Path | None = None
    dataset_path: Path | None = None
    daily_source_path: Path | None = None
    force_rebuild_daily_source: bool = False
    weight_pairs: tuple[tuple[float, float], ...] = DEFAULT_WEIGHT_PAIRS
    risk_scale_min: float = 0.25
    risk_scale_max: float = 2.0
    use_pullback_daily_export: bool = True


def _serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    clean = {key: _serialize_value(value) for key, value in payload.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _validate_weight_pairs(weight_pairs: tuple[tuple[float, float], ...]) -> None:
    if not weight_pairs:
        raise ValueError("At least one weight pair is required.")
    for w_orb, w_pullback in weight_pairs:
        if float(w_orb) < 0.0 or float(w_pullback) < 0.0:
            raise ValueError(f"Negative weights are not allowed: {(w_orb, w_pullback)}")
        if not math.isclose(float(w_orb) + float(w_pullback), 1.0, abs_tol=1e-9):
            raise ValueError(f"Weights must sum to 1: {(w_orb, w_pullback)}")


def _weight_variant_name(w_orb: float, w_pullback: float) -> str:
    return f"orb{int(round(float(w_orb) * 100)):02d}_pull{int(round(float(w_pullback) * 100)):02d}"


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EXPORTS_DIR / f"{DEFAULT_OUTPUT_PREFIX}{stamp}"


def _normalize_dates(values: Any) -> pd.Series:
    return pd.to_datetime(pd.Series(values).reset_index(drop=True), errors="coerce").dt.normalize()


def _split_common_sessions(session_dates: pd.Series, is_fraction: float) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    sessions = sorted(pd.to_datetime(pd.Series(session_dates), errors="coerce").dropna().dt.normalize().unique())
    if len(sessions) < 2:
        raise ValueError("Need at least two common sessions for IS/OOS split.")
    split_idx = int(len(sessions) * float(is_fraction))
    split_idx = max(1, min(len(sessions) - 1, split_idx))
    return [pd.Timestamp(x) for x in sessions[:split_idx]], [pd.Timestamp(x) for x in sessions[split_idx:]]


def _daily_from_trades(trades: pd.DataFrame, sessions: list[Any], initial_capital: float) -> pd.DataFrame:
    dates = _normalize_dates(sessions).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    daily = pd.DataFrame({"session_date": dates})
    if trades.empty:
        daily["daily_pnl_usd"] = 0.0
        daily["daily_trade_count"] = 0
    else:
        view = trades.copy().reset_index(drop=True)
        view["session_date"] = _normalize_dates(view["session_date"])
        grouped = (
            view.groupby("session_date", as_index=False)
            .agg(daily_pnl_usd=("net_pnl_usd", "sum"), daily_trade_count=("trade_id", "count"))
        )
        daily = daily.merge(grouped, on="session_date", how="left")
        daily["daily_pnl_usd"] = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
        daily["daily_trade_count"] = pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0).astype(int)

    pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
    prev_equity = float(initial_capital) + pnl.cumsum().shift(1, fill_value=0.0)
    daily["daily_return"] = np.where(prev_equity != 0.0, pnl / prev_equity, 0.0)
    daily["equity"] = float(initial_capital) + pnl.cumsum()
    daily["peak_equity"] = daily["equity"].cummax()
    daily["drawdown_usd"] = daily["equity"] - daily["peak_equity"]
    return daily


def _load_orb_research_experiment(initial_capital: float) -> tuple[ExperimentConfig, str]:
    path = REPO_ROOT / "export" / "orb_research_campaign" / "top_configs_prop_score.csv"
    if path.exists():
        top = pd.read_csv(path)
        matched = top.loc[top["name"].astype(str).eq(DEFAULT_ORB_CONFIG_NAME)].copy()
        if not matched.empty:
            exp = _experiment_from_json(str(matched.iloc[0]["config_json"]))
            entry = replace(exp.baseline_entry, account_size_usd=float(initial_capital))
            return replace(exp, baseline_entry=entry), f"loaded_config_json:{path}"

    entry = BaselineEntryConfig(
        or_minutes=15,
        opening_time="09:30:00",
        direction="long",
        one_trade_per_day=True,
        entry_buffer_ticks=2,
        stop_buffer_ticks=2,
        target_multiple=2.0,
        vwap_confirmation=True,
        vwap_column="continuous_session_vwap",
        time_exit="16:00:00",
        account_size_usd=float(initial_capital),
        risk_per_trade_pct=0.5,
        tick_size=0.25,
        entry_on_next_open=True,
    )
    exp = ExperimentConfig(
        name=DEFAULT_ORB_CONFIG_NAME,
        stage="full_reopt",
        family="full_reopt",
        baseline_entry=entry,
        baseline_ensemble=BaselineEnsembleConfig(
            atr_window=14,
            q_lows_pct=(20, 25, 30),
            q_highs_pct=(90, 95),
            vote_threshold=0.5,
        ),
        compression=CompressionConfig(mode="weak_close", usage="soft_vote_bonus", soft_bonus_votes=1.0),
        exit=ExitConfig(mode="baseline"),
        dynamic_threshold=DynamicThresholdConfig(
            mode="noise_area_gate",
            noise_lookback=30,
            noise_vm=1.0,
            threshold_style="max_or_high_noise",
            noise_k=0.0,
            atr_k=0.0,
            confirm_bars=1,
            schedule="continuous_on_bar_close",
        ),
    )
    return exp, "hardcoded_fallback"


def _rebuild_orb_daily(dataset_path: Path, config: WeightingCampaignConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    exp, config_source = _load_orb_research_experiment(config.initial_capital_usd)
    atr_windows = sorted({10, 14, 20, 30, int(exp.baseline_ensemble.atr_window)})
    minute_df = prepare_minute_dataset(dataset_path=dataset_path, baseline_entry=exp.baseline_entry, atr_windows=atr_windows)
    daily_reference = build_daily_reference(minute_df)
    minute_df = attach_daily_reference(minute_df, daily_reference)
    candidate_base = build_candidate_universe(minute_df, baseline_entry=exp.baseline_entry)
    all_sessions = sorted(pd.to_datetime(minute_df["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(all_sessions, config.is_fraction)
    context = CampaignContext(
        all_sessions=all_sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        minute_df=minute_df,
        candidate_base_df=candidate_base,
        daily_patterns=daily_reference,
    )
    row, detail = _evaluate_experiment(
        experiment=exp,
        context=context,
        bootstrap_paths=300,
        random_seed=42,
        keep_details=True,
    )
    if detail is None:
        raise RuntimeError(f"ORB sleeve rebuild failed: {row}")
    trades = detail["trades"].copy()
    daily = _daily_from_trades(trades, all_sessions, config.initial_capital_usd)
    meta = {
        "source": "rebuilt_from_minute_data",
        "config_source": config_source,
        "experiment": exp.name,
        "config": {
            "baseline_entry": asdict(exp.baseline_entry),
            "baseline_ensemble": asdict(exp.baseline_ensemble),
            "compression": asdict(exp.compression),
            "dynamic_threshold": asdict(exp.dynamic_threshold),
            "exit": asdict(exp.exit),
        },
        "trades": int(len(trades)),
        "sessions": int(len(daily)),
    }
    return daily, meta


def _latest_matching_dir(root: Path, prefix: str) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _load_pullback_daily_export(initial_capital: float) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    root = EXPORTS_DIR
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("volume_climax_pullback_mnq_risk_sizing_refinement_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(
        sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith("volume_climax_pullback_mnq_risk_sizing_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )
    for directory in candidates:
        summary_path = directory / "summary_by_variant.csv"
        daily_path = directory / "daily_equity_by_variant.csv"
        if not summary_path.exists() or not daily_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        matched = summary.loc[
            summary["campaign_variant_name"].astype(str).eq(DEFAULT_PULLBACK_CAMPAIGN_VARIANT_NAME)
            & summary["alpha_variant_name"].astype(str).eq(DEFAULT_PULLBACK_VARIANT_NAME)
        ].copy()
        if matched.empty:
            continue

        daily = pd.read_csv(daily_path)
        daily = daily.loc[
            daily["campaign_variant_name"].astype(str).eq(DEFAULT_PULLBACK_CAMPAIGN_VARIANT_NAME)
            & daily["alpha_variant_name"].astype(str).eq(DEFAULT_PULLBACK_VARIANT_NAME)
            & daily["scope"].astype(str).eq("full")
        ].copy().reset_index(drop=True)
        if daily.empty:
            continue
        daily["session_date"] = _normalize_dates(daily["session_date"])
        daily = daily.sort_values("session_date").reset_index(drop=True)
        pnl = pd.to_numeric(daily["daily_pnl_usd"], errors="coerce").fillna(0.0)
        prev_equity = float(initial_capital) + pnl.cumsum().shift(1, fill_value=0.0)
        daily["daily_return"] = np.where(prev_equity != 0.0, pnl / prev_equity, 0.0)
        keep = ["session_date", "daily_pnl_usd", "daily_trade_count", "daily_return", "equity", "peak_equity", "drawdown_usd"]
        for column in keep:
            if column not in daily.columns:
                daily[column] = 0.0
        meta = {
            "source": "loaded_daily_equity_export",
            "directory": str(directory),
            "campaign_variant_name": DEFAULT_PULLBACK_CAMPAIGN_VARIANT_NAME,
            "alpha_variant_name": DEFAULT_PULLBACK_VARIANT_NAME,
            "sessions": int(len(daily)),
        }
        return daily[keep].copy(), meta
    return None


def _rebuild_pullback_daily(dataset_path: Path, config: WeightingCampaignConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_1m = load_symbol_data(config.symbol, input_paths={config.symbol: dataset_path})
    bars_1h = resample_rth_1h(raw_1m)
    bars_1h["timestamp"] = pd.to_datetime(bars_1h["timestamp"], errors="coerce")
    bars_1h["session_date"] = bars_1h["timestamp"].dt.date
    sessions = sorted(pd.to_datetime(bars_1h["session_date"]).dt.date.unique())
    _is_sessions, _oos_sessions = split_sessions(bars_1h[["session_date"]].copy(), ratio=config.is_fraction)
    variant = VolumeClimaxPullbackV2Variant(
        name="client_pullback_core",
        family="dynamic_exit",
        timeframe="1h",
        volume_quantile=0.95,
        volume_lookback=50,
        min_body_fraction=0.50,
        min_range_atr=1.20,
        trend_ema_window=None,
        ema_slope_threshold=None,
        atr_percentile_low=None,
        atr_percentile_high=None,
        compression_ratio_max=None,
        entry_mode="next_open",
        pullback_fraction=None,
        confirmation_window=None,
        exit_mode="atr_fraction",
        rr_target=1.0,
        atr_target_multiple=1.00,
        time_stop_bars=2,
        trailing_atr_multiple=0.50,
        session_overlay="all_rth",
    )
    features = prepare_volume_climax_pullback_v2_features(bars_1h)
    signal_df = build_volume_climax_pullback_v2_signal_frame(features, variant)
    execution, instrument = build_execution_model_for_profile(symbol=config.symbol, profile_name="repo_realistic")
    sizing = RiskPercentPositionSizing(
        initial_capital_usd=config.initial_capital_usd,
        risk_pct=0.0025,
        max_contracts=6,
        skip_trade_if_too_small=True,
        compound_realized_pnl=False,
    )
    result = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution,
        instrument=instrument,
        position_sizing=sizing,
    )
    daily = _daily_from_trades(result.trades.copy(), sessions, config.initial_capital_usd)
    meta = {
        "source": "rebuilt_from_minute_data",
        "variant": asdict(variant),
        "sizing": {
            "risk_pct": 0.0025,
            "max_contracts": 6,
            "skip_trade_if_too_small": True,
            "compound_realized_pnl": False,
        },
        "trades": int(len(result.trades)),
        "sessions": int(len(daily)),
    }
    return daily, meta


def _benchmark_daily(dataset_path: Path, symbol: str, common_dates: pd.Series) -> pd.DataFrame:
    raw = load_symbol_data(symbol, input_paths={symbol: dataset_path})
    timestamps = pd.to_datetime(raw["timestamp"], errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_localize(None)
    raw["session_date"] = timestamps.dt.normalize()
    daily_close = raw.groupby("session_date", as_index=False)["close"].last()
    out = pd.DataFrame({"session_date": _normalize_dates(common_dates)})
    out = out.merge(daily_close, on="session_date", how="left").sort_values("session_date").reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce").ffill()
    out["benchmark_return"] = out["close"].pct_change().fillna(0.0)
    return out[["session_date", "benchmark_return"]]


def _build_daily_source(config: WeightingCampaignConfig, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_path = Path(config.dataset_path) if config.dataset_path is not None else resolve_processed_dataset(config.symbol, timeframe="1m")
    orb_daily, orb_meta = _rebuild_orb_daily(dataset_path, config)

    pullback_loaded = _load_pullback_daily_export(config.initial_capital_usd) if config.use_pullback_daily_export else None
    if pullback_loaded is None:
        pullback_daily, pullback_meta = _rebuild_pullback_daily(dataset_path, config)
    else:
        pullback_daily, pullback_meta = pullback_loaded

    orb = orb_daily.rename(
        columns={
            "daily_pnl_usd": "orb_daily_pnl_usd",
            "daily_return": "orb_return",
            "daily_trade_count": "orb_daily_trade_count",
        }
    )[["session_date", "orb_daily_pnl_usd", "orb_return", "orb_daily_trade_count"]]
    pull = pullback_daily.rename(
        columns={
            "daily_pnl_usd": "pullback_daily_pnl_usd",
            "daily_return": "pullback_return",
            "daily_trade_count": "pullback_daily_trade_count",
        }
    )[["session_date", "pullback_daily_pnl_usd", "pullback_return", "pullback_daily_trade_count"]]
    source = orb.merge(pull, on="session_date", how="inner").sort_values("session_date").reset_index(drop=True)
    is_sessions, oos_sessions = _split_common_sessions(source["session_date"], config.is_fraction)
    is_set = set(is_sessions)
    source["phase"] = np.where(source["session_date"].isin(is_set), "is", "oos")

    benchmark = _benchmark_daily(dataset_path, config.symbol, source["session_date"])
    source = source.merge(benchmark, on="session_date", how="left")
    source["benchmark_return"] = pd.to_numeric(source["benchmark_return"], errors="coerce").fillna(0.0)

    source_path = output_dir / "source_daily_returns.csv"
    source.to_csv(source_path, index=False)
    metadata = {
        "source": "rebuilt_or_loaded_sleeves",
        "source_path": str(source_path),
        "dataset_path": str(dataset_path),
        "orb": orb_meta,
        "pullback": pullback_meta,
        "common_sessions": int(len(source)),
        "is_sessions": int((source["phase"] == "is").sum()),
        "oos_sessions": int((source["phase"] == "oos").sum()),
    }
    return source, metadata


def _load_daily_source(path: Path, config: WeightingCampaignConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(path)
    required = {
        "session_date",
        "orb_return",
        "pullback_return",
        "orb_daily_trade_count",
        "pullback_daily_trade_count",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Daily source is missing required columns: {missing}")
    source["session_date"] = _normalize_dates(source["session_date"])
    source = source.dropna(subset=["session_date"]).sort_values("session_date").reset_index(drop=True)
    if "phase" not in source.columns:
        is_sessions, _ = _split_common_sessions(source["session_date"], config.is_fraction)
        is_set = set(is_sessions)
        source["phase"] = np.where(source["session_date"].isin(is_set), "is", "oos")
    if "benchmark_return" not in source.columns:
        source["benchmark_return"] = 0.0
    if "orb_daily_pnl_usd" not in source.columns:
        source["orb_daily_pnl_usd"] = pd.to_numeric(source["orb_return"], errors="coerce").fillna(0.0) * config.initial_capital_usd
    if "pullback_daily_pnl_usd" not in source.columns:
        source["pullback_daily_pnl_usd"] = pd.to_numeric(source["pullback_return"], errors="coerce").fillna(0.0) * config.initial_capital_usd
    return source, {"source": "loaded_daily_source", "source_path": str(path)}


def _daily_curve_from_returns(
    session_dates: pd.Series,
    daily_returns: pd.Series,
    initial_capital: float,
    daily_trade_count: pd.Series | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame({"session_date": _normalize_dates(session_dates)})
    out["daily_return"] = pd.to_numeric(pd.Series(daily_returns).reset_index(drop=True), errors="coerce").fillna(0.0)
    if (out["daily_return"] <= -1.0).any():
        worst = float(out["daily_return"].min())
        raise ValueError(f"Daily return <= -100% after portfolio construction ({worst:.2%}).")
    out["equity"] = float(initial_capital) * (1.0 + out["daily_return"]).cumprod()
    out["daily_pnl_usd"] = out["equity"].diff().fillna(out["equity"].iloc[0] - float(initial_capital))
    out["peak_equity"] = out["equity"].cummax()
    out["drawdown_usd"] = out["equity"] - out["peak_equity"]
    out["drawdown_pct"] = np.where(out["peak_equity"] > 0.0, (out["equity"] / out["peak_equity"] - 1.0) * 100.0, 0.0)
    if daily_trade_count is None:
        out["daily_trade_count"] = 0
    else:
        out["daily_trade_count"] = pd.to_numeric(pd.Series(daily_trade_count).reset_index(drop=True), errors="coerce").fillna(0).astype(int)
    return out


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return default
    return float(numerator / denominator)


def _scope_metrics(
    *,
    curve: pd.DataFrame,
    initial_capital: float,
    correlation: float,
    method: str,
    weight_variant_name: str,
    variant_name: str,
    nominal_orb_weight: float,
    nominal_pullback_weight: float,
    effective_orb_weight: float,
    effective_pullback_weight: float,
    orb_scale: float,
    pullback_scale: float,
    calibration_scope: str,
    scope: str,
) -> dict[str, Any]:
    ordered = curve.sort_values("session_date").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("Cannot compute metrics on an empty curve.")
    returns = pd.to_numeric(ordered["daily_return"], errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(ordered["daily_pnl_usd"], errors="coerce").fillna(0.0)
    equity = pd.to_numeric(ordered["equity"], errors="coerce").fillna(float(initial_capital))
    final_equity = float(equity.iloc[-1])
    net_profit = float(final_equity - float(initial_capital))
    return_pct = _safe_ratio(net_profit, float(initial_capital)) * 100.0
    years = max(((pd.Timestamp(ordered["session_date"].iloc[-1]) - pd.Timestamp(ordered["session_date"].iloc[0])).days + 1) / 365.25, 1 / 365.25)
    cagr = float(((final_equity / float(initial_capital)) ** (1.0 / years) - 1.0) * 100.0) if final_equity > 0 else float("nan")
    vol = float(returns.std(ddof=0))
    sharpe = _safe_ratio(float(returns.mean()), vol) * math.sqrt(252.0) if vol > 0 else 0.0
    downside = returns.loc[returns < 0.0]
    downside_std = float(np.sqrt(np.mean(np.square(downside)))) if not downside.empty else 0.0
    sortino = _safe_ratio(float(returns.mean()), downside_std) * math.sqrt(252.0) if downside_std > 0 else 0.0
    wins = pnl.loc[pnl > 0.0]
    losses = pnl.loc[pnl < 0.0]
    profit_factor = _safe_ratio(float(wins.sum()), abs(float(losses.sum())), default=float("inf")) if not losses.empty else float("inf")
    max_drawdown_usd = abs(float(pd.to_numeric(ordered["drawdown_usd"], errors="coerce").min()))
    max_drawdown_pct = abs(float(pd.to_numeric(ordered["drawdown_pct"], errors="coerce").min()))
    max_daily_drawdown_usd = abs(float(min(pnl.min(), 0.0)))
    calmar = _safe_ratio(cagr, max_drawdown_pct)
    return_over_drawdown = _safe_ratio(net_profit, max_drawdown_usd)
    days_traded = int(pd.to_numeric(ordered["daily_trade_count"], errors="coerce").fillna(0).gt(0).sum())
    monthly = ordered.assign(month=ordered["session_date"].dt.to_period("M").dt.to_timestamp()).groupby("month")["daily_pnl_usd"].sum()
    positive_month_rate = float((monthly > 0.0).mean()) if len(monthly) else 0.0
    stability_score = 0.5 * positive_month_rate + 0.5 * (1.0 - min(max_drawdown_pct / 10.0, 1.0))
    composite_score = (
        1.20 * math.tanh(sharpe / 2.0)
        + 0.70 * math.tanh(sortino / 3.0)
        + 0.55 * math.tanh(return_over_drawdown / 4.0)
        + 0.40 * math.tanh(calmar / 2.0)
        + 0.35 * stability_score
        - 0.45 * math.tanh(max_drawdown_pct / 6.0)
        - 0.25 * math.tanh((max_daily_drawdown_usd / float(initial_capital)) / 0.02)
    )
    return {
        "method": method,
        "weight_variant_name": weight_variant_name,
        "variant_name": variant_name,
        "scope": scope,
        "calibration_scope": calibration_scope,
        "nominal_orb_weight": float(nominal_orb_weight),
        "nominal_pullback_weight": float(nominal_pullback_weight),
        "effective_orb_weight": float(effective_orb_weight),
        "effective_pullback_weight": float(effective_pullback_weight),
        "orb_scale": float(orb_scale),
        "pullback_scale": float(pullback_scale),
        "start_date": ordered["session_date"].iloc[0].date().isoformat(),
        "end_date": ordered["session_date"].iloc[-1].date().isoformat(),
        "n_days": int(len(ordered)),
        "net_profit_usd": net_profit,
        "return_pct": return_pct,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor_daily": profit_factor,
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "max_daily_drawdown_usd": max_daily_drawdown_usd,
        "calmar": calmar,
        "days_traded": days_traded,
        "orb_pullback_daily_corr": float(correlation) if math.isfinite(float(correlation)) else 0.0,
        "return_over_drawdown": return_over_drawdown,
        "positive_month_rate": positive_month_rate,
        "stability_score": stability_score,
        "composite_score": float(composite_score),
    }


def _calibrate_inputs(source: pd.DataFrame, config: WeightingCampaignConfig) -> dict[str, float | str]:
    is_rows = source.loc[source["phase"].astype(str).eq("is")].copy()
    if is_rows.empty:
        raise ValueError("No IS rows available for weighting calibration.")
    orb = pd.to_numeric(is_rows["orb_return"], errors="coerce").fillna(0.0)
    pull = pd.to_numeric(is_rows["pullback_return"], errors="coerce").fillna(0.0)
    orb_vol = float(orb.std(ddof=0))
    pull_vol = float(pull.std(ddof=0))
    eps = 1e-12
    target_vol = float(np.mean([x for x in (orb_vol, pull_vol) if x > eps])) if max(orb_vol, pull_vol) > eps else 0.0
    raw_orb_scale = _safe_ratio(target_vol, orb_vol, default=1.0) if orb_vol > eps else 1.0
    raw_pull_scale = _safe_ratio(target_vol, pull_vol, default=1.0) if pull_vol > eps else 1.0
    return {
        "calibration_scope": "is",
        "risk_measure": "daily_return_volatility",
        "orb_is_vol": orb_vol,
        "pullback_is_vol": pull_vol,
        "target_is_vol": target_vol,
        "risk_scaled_orb_scale": float(np.clip(raw_orb_scale, config.risk_scale_min, config.risk_scale_max)),
        "risk_scaled_pullback_scale": float(np.clip(raw_pull_scale, config.risk_scale_min, config.risk_scale_max)),
        "raw_risk_scaled_orb_scale": raw_orb_scale,
        "raw_risk_scaled_pullback_scale": raw_pull_scale,
    }


def _portfolio_returns_for_variant(
    source: pd.DataFrame,
    *,
    method: str,
    w_orb: float,
    w_pullback: float,
    calibration: dict[str, float | str],
) -> tuple[pd.Series, dict[str, float]]:
    orb_ret = pd.to_numeric(source["orb_return"], errors="coerce").fillna(0.0)
    pull_ret = pd.to_numeric(source["pullback_return"], errors="coerce").fillna(0.0)

    if method == "static":
        effective_orb = float(w_orb)
        effective_pull = float(w_pullback)
        orb_scale = 1.0
        pull_scale = 1.0
        returns = effective_orb * orb_ret + effective_pull * pull_ret
    elif method == "risk_scaled":
        effective_orb = float(w_orb)
        effective_pull = float(w_pullback)
        orb_scale = float(calibration["risk_scaled_orb_scale"])
        pull_scale = float(calibration["risk_scaled_pullback_scale"])
        returns = effective_orb * orb_scale * orb_ret + effective_pull * pull_scale * pull_ret
    elif method == "inverse_risk":
        orb_risk = max(float(calibration["orb_is_vol"]), 1e-12)
        pull_risk = max(float(calibration["pullback_is_vol"]), 1e-12)
        raw_orb = float(w_orb) / orb_risk
        raw_pull = float(w_pullback) / pull_risk
        denom = raw_orb + raw_pull
        effective_orb = raw_orb / denom if denom > 0 else float(w_orb)
        effective_pull = raw_pull / denom if denom > 0 else float(w_pullback)
        orb_scale = 1.0
        pull_scale = 1.0
        returns = effective_orb * orb_ret + effective_pull * pull_ret
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return returns, {
        "effective_orb_weight": effective_orb,
        "effective_pullback_weight": effective_pull,
        "orb_scale": orb_scale,
        "pullback_scale": pull_scale,
    }


def _evaluate_variant_on_scope(
    source: pd.DataFrame,
    *,
    returns: pd.Series,
    initial_capital: float,
    method: str,
    weight_variant_name: str,
    variant_name: str,
    nominal_orb_weight: float,
    nominal_pullback_weight: float,
    effective_orb_weight: float,
    effective_pullback_weight: float,
    orb_scale: float,
    pullback_scale: float,
    calibration_scope: str,
    scope: str,
) -> dict[str, Any]:
    trade_count = (
        pd.to_numeric(source["orb_daily_trade_count"], errors="coerce").fillna(0).astype(int)
        + pd.to_numeric(source["pullback_daily_trade_count"], errors="coerce").fillna(0).astype(int)
    )
    curve = _daily_curve_from_returns(source["session_date"], returns, initial_capital, trade_count)
    corr = pd.to_numeric(source["orb_return"], errors="coerce").fillna(0.0).corr(
        pd.to_numeric(source["pullback_return"], errors="coerce").fillna(0.0)
    )
    return _scope_metrics(
        curve=curve,
        initial_capital=initial_capital,
        correlation=float(corr) if pd.notna(corr) else 0.0,
        method=method,
        weight_variant_name=weight_variant_name,
        variant_name=variant_name,
        nominal_orb_weight=nominal_orb_weight,
        nominal_pullback_weight=nominal_pullback_weight,
        effective_orb_weight=effective_orb_weight,
        effective_pullback_weight=effective_pullback_weight,
        orb_scale=orb_scale,
        pullback_scale=pullback_scale,
        calibration_scope=calibration_scope,
        scope=scope,
    )


def _subperiod_masks(oos_source: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    masks: list[tuple[str, pd.Series]] = []
    if oos_source.empty:
        return masks
    ordered = oos_source.sort_values("session_date").reset_index(drop=True)
    idx = ordered.index.to_series()
    split_idx = len(ordered) // 2
    first_dates = set(ordered.loc[idx < split_idx, "session_date"])
    second_dates = set(ordered.loc[idx >= split_idx, "session_date"])
    masks.append(("oos_full", oos_source["session_date"].isin(set(ordered["session_date"]))))
    masks.append(("oos_first_half", oos_source["session_date"].isin(first_dates)))
    masks.append(("oos_second_half", oos_source["session_date"].isin(second_dates)))
    for year in sorted(oos_source["session_date"].dt.year.dropna().unique()):
        masks.append((f"oos_year_{int(year)}", oos_source["session_date"].dt.year.eq(int(year))))
    return masks


def _evaluate_all_variants(
    source: pd.DataFrame,
    config: WeightingCampaignConfig,
    calibration: dict[str, float | str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_weight_pairs(config.weight_pairs)
    source = source.sort_values("session_date").reset_index(drop=True)
    scope_frames = {
        "is": source.loc[source["phase"].astype(str).eq("is")].copy(),
        "oos": source.loc[source["phase"].astype(str).eq("oos")].copy(),
    }
    metrics_rows: list[dict[str, Any]] = []
    subperiod_rows: list[dict[str, Any]] = []

    variant_specs: list[dict[str, Any]] = []
    for method in ("static", "risk_scaled", "inverse_risk"):
        for w_orb, w_pullback in config.weight_pairs:
            weight_name = _weight_variant_name(w_orb, w_pullback)
            returns, meta = _portfolio_returns_for_variant(
                source,
                method=method,
                w_orb=float(w_orb),
                w_pullback=float(w_pullback),
                calibration=calibration,
            )
            variant_specs.append(
                {
                    "method": method,
                    "weight_variant_name": weight_name,
                    "variant_name": f"{method}__{weight_name}",
                    "nominal_orb_weight": float(w_orb),
                    "nominal_pullback_weight": float(w_pullback),
                    "returns": returns,
                    **meta,
                }
            )

    baseline_specs = [
        {
            "method": "baseline",
            "weight_variant_name": "orb100_pull00",
            "variant_name": "baseline__orb_standalone",
            "nominal_orb_weight": 1.0,
            "nominal_pullback_weight": 0.0,
            "effective_orb_weight": 1.0,
            "effective_pullback_weight": 0.0,
            "orb_scale": 1.0,
            "pullback_scale": 1.0,
            "returns": pd.to_numeric(source["orb_return"], errors="coerce").fillna(0.0),
        },
        {
            "method": "baseline",
            "weight_variant_name": "orb00_pull100",
            "variant_name": "baseline__pullback_standalone",
            "nominal_orb_weight": 0.0,
            "nominal_pullback_weight": 1.0,
            "effective_orb_weight": 0.0,
            "effective_pullback_weight": 1.0,
            "orb_scale": 1.0,
            "pullback_scale": 1.0,
            "returns": pd.to_numeric(source["pullback_return"], errors="coerce").fillna(0.0),
        },
        {
            "method": "baseline",
            "weight_variant_name": "benchmark",
            "variant_name": "baseline__benchmark",
            "nominal_orb_weight": 0.0,
            "nominal_pullback_weight": 0.0,
            "effective_orb_weight": 0.0,
            "effective_pullback_weight": 0.0,
            "orb_scale": 0.0,
            "pullback_scale": 0.0,
            "returns": pd.to_numeric(source["benchmark_return"], errors="coerce").fillna(0.0),
        },
    ]
    variant_specs.extend(baseline_specs)

    for spec in variant_specs:
        for scope, frame in scope_frames.items():
            if frame.empty:
                continue
            returns = spec["returns"].loc[frame.index]
            metrics_rows.append(
                _evaluate_variant_on_scope(
                    frame,
                    returns=returns,
                    initial_capital=config.initial_capital_usd,
                    method=spec["method"],
                    weight_variant_name=spec["weight_variant_name"],
                    variant_name=spec["variant_name"],
                    nominal_orb_weight=spec["nominal_orb_weight"],
                    nominal_pullback_weight=spec["nominal_pullback_weight"],
                    effective_orb_weight=spec["effective_orb_weight"],
                    effective_pullback_weight=spec["effective_pullback_weight"],
                    orb_scale=spec["orb_scale"],
                    pullback_scale=spec["pullback_scale"],
                    calibration_scope=str(calibration["calibration_scope"]),
                    scope=scope,
                )
            )

        oos = scope_frames["oos"]
        for subperiod, mask in _subperiod_masks(oos):
            frame = oos.loc[mask].copy()
            if frame.empty:
                continue
            returns = spec["returns"].loc[frame.index]
            row = _evaluate_variant_on_scope(
                frame,
                returns=returns,
                initial_capital=config.initial_capital_usd,
                method=spec["method"],
                weight_variant_name=spec["weight_variant_name"],
                variant_name=spec["variant_name"],
                nominal_orb_weight=spec["nominal_orb_weight"],
                nominal_pullback_weight=spec["nominal_pullback_weight"],
                effective_orb_weight=spec["effective_orb_weight"],
                effective_pullback_weight=spec["effective_pullback_weight"],
                orb_scale=spec["orb_scale"],
                pullback_scale=spec["pullback_scale"],
                calibration_scope=str(calibration["calibration_scope"]),
                scope=subperiod,
            )
            subperiod_rows.append(row)

    return pd.DataFrame(metrics_rows), pd.DataFrame(subperiod_rows)


def _pairwise_correlation(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    oos_source = source.loc[source["phase"].astype(str).eq("oos")].copy()
    scopes = [
        ("is", source["phase"].astype(str).eq("is")),
        ("oos", source["phase"].astype(str).eq("oos")),
    ]
    for subperiod, mask in _subperiod_masks(oos_source):
        dates = set(oos_source.loc[mask, "session_date"])
        scopes.append((subperiod, source["session_date"].isin(dates)))
    for scope, mask in scopes:
        frame = source.loc[mask].copy()
        if frame.empty:
            continue
        orb = pd.to_numeric(frame["orb_return"], errors="coerce").fillna(0.0)
        pull = pd.to_numeric(frame["pullback_return"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "scope": scope,
                "start_date": frame["session_date"].min().date().isoformat(),
                "end_date": frame["session_date"].max().date().isoformat(),
                "n_days": int(len(frame)),
                "orb_pullback_daily_corr": float(orb.corr(pull)) if len(frame) > 1 and pd.notna(orb.corr(pull)) else 0.0,
                "orb_daily_vol": float(orb.std(ddof=0)),
                "pullback_daily_vol": float(pull.std(ddof=0)),
                "calibration_used": scope == "is",
            }
        )
    return pd.DataFrame(rows)


def _ranking(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = metrics.loc[metrics["scope"].astype(str).eq(scope)].copy()
    return out.sort_values(
        ["composite_score", "sharpe", "max_drawdown_usd", "net_profit_usd"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def _select_recommendations(oos: pd.DataFrame, subperiods: pd.DataFrame) -> dict[str, pd.Series]:
    tradable = oos.loc[oos["method"].astype(str).ne("baseline")].copy()
    if tradable.empty:
        raise ValueError("No tradable weighting variants available.")
    equal = oos.loc[oos["variant_name"].astype(str).eq("static__orb50_pull50")]
    equal_row = equal.iloc[0] if not equal.empty else tradable.iloc[0]

    sub = subperiods.loc[subperiods["scope"].isin(["oos_first_half", "oos_second_half"])].copy()
    stable_names = set()
    if not sub.empty:
        pivot = sub.pivot_table(index="variant_name", columns="scope", values="net_profit_usd", aggfunc="first")
        stable_names = set(pivot.loc[(pivot.get("oos_first_half", -np.inf) > 0.0) & (pivot.get("oos_second_half", -np.inf) > 0.0)].index)

    main_pool = tradable.loc[
        (tradable["sharpe"] >= float(equal_row["sharpe"]) * 0.95)
        & (tradable["max_drawdown_usd"] <= float(equal_row["max_drawdown_usd"]) * 1.25)
    ].copy()
    if stable_names:
        stable_pool = main_pool.loc[main_pool["variant_name"].isin(stable_names)].copy()
        if not stable_pool.empty:
            main_pool = stable_pool
    if main_pool.empty:
        main_pool = tradable.copy()

    prop_pool = tradable.loc[
        (tradable["net_profit_usd"] > 0.0)
        & (tradable["max_drawdown_usd"] <= float(equal_row["max_drawdown_usd"]) * 1.05)
    ].copy()
    if prop_pool.empty:
        prop_pool = tradable.copy()

    aggressive_pool = tradable.loc[
        (tradable["sharpe"] >= max(0.75, float(equal_row["sharpe"]) * 0.75))
        & (tradable["max_drawdown_usd"] <= float(equal_row["max_drawdown_usd"]) * 1.75)
    ].copy()
    if aggressive_pool.empty:
        aggressive_pool = tradable.copy()

    return {
        "equal_weight": equal_row,
        "main": main_pool.sort_values(["composite_score", "sharpe", "max_drawdown_usd"], ascending=[False, False, True]).iloc[0],
        "prop_safe": prop_pool.sort_values(["max_drawdown_usd", "max_daily_drawdown_usd", "sharpe"], ascending=[True, True, False]).iloc[0],
        "aggressive": aggressive_pool.sort_values(["net_profit_usd", "sharpe", "composite_score"], ascending=[False, False, False]).iloc[0],
    }


def _top_table(frame: pd.DataFrame, n: int = 12) -> str:
    cols = [
        "variant_name",
        "net_profit_usd",
        "cagr_pct",
        "sharpe",
        "sortino",
        "max_drawdown_usd",
        "max_daily_drawdown_usd",
        "calmar",
        "return_over_drawdown",
        "composite_score",
    ]
    available = [c for c in cols if c in frame.columns]
    return frame[available].head(n).round(3).to_string(index=False)


def _build_final_report(
    *,
    output_dir: Path,
    ranking_is: pd.DataFrame,
    ranking_oos: pd.DataFrame,
    subperiods: pd.DataFrame,
    correlation: pd.DataFrame,
    recommendations: dict[str, pd.Series],
    calibration: dict[str, float | str],
    source_meta: dict[str, Any],
) -> None:
    equal = recommendations["equal_weight"]
    main = recommendations["main"]
    prop_safe = recommendations["prop_safe"]
    aggressive = recommendations["aggressive"]

    overweight_pull = ranking_oos.loc[
        (ranking_oos["method"].astype(str).ne("baseline"))
        & (pd.to_numeric(ranking_oos["effective_pullback_weight"], errors="coerce") > 0.50)
    ].copy()
    best_over_pull = overweight_pull.iloc[0] if not overweight_pull.empty else None
    overweight_answer = (
        "yes"
        if best_over_pull is not None and float(best_over_pull["composite_score"]) > float(equal["composite_score"])
        else "no"
    )
    equal_ok = (
        "yes"
        if float(equal["sharpe"]) >= 0.90 * float(main["sharpe"])
        and float(equal["max_drawdown_usd"]) <= 1.25 * float(main["max_drawdown_usd"])
        else "acceptable but not best"
    )

    sub_top_names = {main["variant_name"], prop_safe["variant_name"], aggressive["variant_name"], equal["variant_name"]}
    sub_table = subperiods.loc[subperiods["variant_name"].isin(sub_top_names)].copy()
    sub_table = sub_table.sort_values(["variant_name", "scope"])
    sub_cols = ["variant_name", "scope", "net_profit_usd", "sharpe", "max_drawdown_usd", "composite_score"]

    lines = [
        "# MNQ ORB + Pullback Weighting Campaign",
        "",
        "## Protocol",
        "",
        "- Sleeves are fixed: ORB research configuration and volume climax pullback configuration are not optimized here.",
        "- Execution assumptions are inherited from the audited sleeve implementations.",
        "- Portfolio calibration uses IS rows only: risk scaling factors and inverse-risk weights use IS daily volatility.",
        "- OOS metrics are computed strictly after the common-calendar IS/OOS split.",
        "- Weighting methods are simple and static in OOS: no adaptive recalibration.",
        "",
        "## Variants",
        "",
        "- Static weights: direct blend of sleeve daily returns.",
        "- Risk-scaled weights: each sleeve is scaled to the average IS daily volatility, then fixed weights are applied.",
        "- Inverse-risk weights: nominal weights are adjusted by inverse IS volatility, then frozen.",
        "",
        "Nominal weight grid:",
        "",
        ", ".join(_weight_variant_name(w_orb, w_pull) for w_orb, w_pull in DEFAULT_WEIGHT_PAIRS),
        "",
        "## Calibration",
        "",
        "```json",
        json.dumps({k: _serialize_value(v) for k, v in calibration.items()}, indent=2),
        "```",
        "",
        "## Source Data",
        "",
        "```json",
        json.dumps({k: _serialize_value(v) for k, v in source_meta.items()}, indent=2)[:6000],
        "```",
        "",
        "## IS Ranking",
        "",
        "```text",
        _top_table(ranking_is, n=15),
        "```",
        "",
        "## OOS Ranking",
        "",
        "```text",
        _top_table(ranking_oos, n=15),
        "```",
        "",
        "## OOS Subperiods",
        "",
        "```text",
        sub_table[sub_cols].round(3).to_string(index=False) if not sub_table.empty else "No subperiod rows.",
        "```",
        "",
        "## Pairwise Correlation",
        "",
        "```text",
        correlation.round(4).to_string(index=False),
        "```",
        "",
        "## Conclusion",
        "",
        f"- 50/50 acceptable: {equal_ok}. Static 50/50 OOS Sharpe={float(equal['sharpe']):.3f}, maxDD={float(equal['max_drawdown_usd']):.1f}, score={float(equal['composite_score']):.3f}.",
        f"- Pullback overweight improves composite vs 50/50: {overweight_answer}.",
        f"- Main production recommendation: `{main['variant_name']}` (OOS Sharpe={float(main['sharpe']):.3f}, maxDD={float(main['max_drawdown_usd']):.1f}, score={float(main['composite_score']):.3f}).",
        f"- Conservative / prop-safe recommendation: `{prop_safe['variant_name']}` (maxDD={float(prop_safe['max_drawdown_usd']):.1f}, max daily loss={float(prop_safe['max_daily_drawdown_usd']):.1f}).",
        f"- Aggressive but still defendable recommendation: `{aggressive['variant_name']}` (net={float(aggressive['net_profit_usd']):.1f}, Sharpe={float(aggressive['sharpe']):.3f}, maxDD={float(aggressive['max_drawdown_usd']):.1f}).",
        "",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_campaign(config: WeightingCampaignConfig) -> dict[str, Path]:
    ensure_directories()
    output_dir = Path(config.output_dir) if config.output_dir is not None else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.daily_source_path is not None and not config.force_rebuild_daily_source:
        source, source_meta = _load_daily_source(Path(config.daily_source_path), config)
    else:
        reusable_source = output_dir / "source_daily_returns.csv"
        if reusable_source.exists() and not config.force_rebuild_daily_source:
            source, source_meta = _load_daily_source(reusable_source, config)
        else:
            source, source_meta = _build_daily_source(config, output_dir)

    source["session_date"] = _normalize_dates(source["session_date"])
    numeric_cols = [
        "orb_return",
        "pullback_return",
        "benchmark_return",
        "orb_daily_trade_count",
        "pullback_daily_trade_count",
        "orb_daily_pnl_usd",
        "pullback_daily_pnl_usd",
    ]
    for column in numeric_cols:
        if column in source.columns:
            source[column] = pd.to_numeric(source[column], errors="coerce").fillna(0.0)

    calibration = _calibrate_inputs(source, config)
    metrics, subperiods = _evaluate_all_variants(source, config, calibration)
    correlation = _pairwise_correlation(source)
    ranking_is = _ranking(metrics, "is")
    ranking_oos = _ranking(metrics, "oos")
    recommendations = _select_recommendations(ranking_oos, subperiods)

    ranking_oos.to_csv(output_dir / "ranking_weights_oos.csv", index=False)
    ranking_is.to_csv(output_dir / "ranking_weights_is.csv", index=False)
    subperiods.to_csv(output_dir / "weighting_subperiods.csv", index=False)
    correlation.to_csv(output_dir / "weighting_pairwise_correlation.csv", index=False)

    best_summary = {
        "protocol": "static portfolio weighting calibrated on IS only",
        "output_dir": str(output_dir),
        "calibration": calibration,
        "source": source_meta,
        "equal_weight": recommendations["equal_weight"].to_dict(),
        "main_recommendation": recommendations["main"].to_dict(),
        "prop_safe_recommendation": recommendations["prop_safe"].to_dict(),
        "aggressive_recommendation": recommendations["aggressive"].to_dict(),
    }
    _json_dump(output_dir / "best_config_summary.json", best_summary)
    _json_dump(
        output_dir / "run_metadata.json",
        {
            "config": asdict(config),
            "created_at": datetime.now(),
            "exports": [
                "ranking_weights_oos.csv",
                "ranking_weights_is.csv",
                "weighting_subperiods.csv",
                "weighting_pairwise_correlation.csv",
                "best_config_summary.json",
                "final_report.md",
            ],
        },
    )
    _build_final_report(
        output_dir=output_dir,
        ranking_is=ranking_is,
        ranking_oos=ranking_oos,
        subperiods=subperiods,
        correlation=correlation,
        recommendations=recommendations,
        calibration=calibration,
        source_meta=source_meta,
    )

    return {
        "output_dir": output_dir,
        "ranking_weights_oos": output_dir / "ranking_weights_oos.csv",
        "ranking_weights_is": output_dir / "ranking_weights_is.csv",
        "weighting_subperiods": output_dir / "weighting_subperiods.csv",
        "weighting_pairwise_correlation": output_dir / "weighting_pairwise_correlation.csv",
        "best_config_summary": output_dir / "best_config_summary.json",
        "final_report": output_dir / "final_report.md",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--daily-source-path", type=Path, default=None)
    parser.add_argument("--force-rebuild-daily-source", action="store_true")
    parser.add_argument("--no-pullback-daily-export", action="store_true")
    parser.add_argument("--initial-capital-usd", type=float, default=DEFAULT_INITIAL_CAPITAL_USD)
    parser.add_argument("--is-fraction", type=float, default=DEFAULT_IS_FRACTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WeightingCampaignConfig(
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        daily_source_path=args.daily_source_path,
        force_rebuild_daily_source=bool(args.force_rebuild_daily_source),
        use_pullback_daily_export=not bool(args.no_pullback_daily_export),
        initial_capital_usd=float(args.initial_capital_usd),
        is_fraction=float(args.is_fraction),
    )
    artifacts = run_campaign(config)
    print(f"Output directory: {artifacts['output_dir']}")
    print(f"Final report: {artifacts['final_report']}")


if __name__ == "__main__":
    main()
