"""Prop-firm challenge simulation for the validated MNQ ORB baseline and sizing_3state variant.

This module intentionally stays simple:
- it reuses the audited daily/trade exports from the regime/sizing campaign,
- it evaluates only the two sanctioned variants,
- it works at the daily level because the strategy is one trade per day,
- it uses stylized prop rulesets rather than claiming commercial-rule replication.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.mnq_orb_prop_survivability_campaign import _rebuild_daily_results_from_trades
from src.config.paths import EXPORTS_DIR, ensure_directories


DEFAULT_SOURCE_RUN_GLOB = "mnq_orb_regime_filter_sizing_*"
DEFAULT_VARIANT_ORDER = (
    "nominal",
    "sizing_3state_realized_vol_ratio_15_60",
)
DEFAULT_PRIMARY_SCOPE = "oos"
DEFAULT_BOOTSTRAP_PATHS = 2000
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 5
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class PropChallengeRuleset:
    name: str
    family: str
    resembles: str
    description: str
    account_size_usd: float = 50_000.0
    profit_target_usd: float = 3_000.0
    max_traded_days: int | None = None
    daily_loss_limit_usd: float | None = None
    static_max_loss_usd: float | None = None
    trailing_drawdown_usd: float | None = None
    near_limit_buffer_frac: float = 0.20
    near_fail_buffer_frac: float = 0.10
    half_target_frac: float = 0.50
    notes: str = ""


@dataclass
class VariantInput:
    variant_name: str
    label: str
    source_root: Path
    trades: pd.DataFrame
    daily_results: pd.DataFrame
    controls: pd.DataFrame
    reference_account_size_usd: float
    source_summary_row: dict[str, Any]


@dataclass(frozen=True)
class CampaignSpec:
    source_run_root: Path | None = None
    primary_scope: str = DEFAULT_PRIMARY_SCOPE
    bootstrap_paths: int = DEFAULT_BOOTSTRAP_PATHS
    bootstrap_block_size: int = DEFAULT_BOOTSTRAP_BLOCK_SIZE
    random_seed: int = DEFAULT_RANDOM_SEED
    variant_names: tuple[str, ...] = DEFAULT_VARIANT_ORDER
    rulesets: tuple[PropChallengeRuleset, ...] = field(
        default_factory=lambda: (
            PropChallengeRuleset(
                name="classic_static_30d",
                family="classic_fixed_drawdown",
                resembles="Topstep-like fixed target challenge",
                description="Fixed target with static max loss, realized daily limit, and 30 traded-day expiry.",
                account_size_usd=50_000.0,
                profit_target_usd=3_000.0,
                max_traded_days=30,
                daily_loss_limit_usd=1_000.0,
                static_max_loss_usd=2_000.0,
                notes="Stylized fixed-drawdown evaluation; daily rule is enforced on realized daily PnL only.",
            ),
            PropChallengeRuleset(
                name="trailing_strict_35d",
                family="trailing_drawdown",
                resembles="Trailing-DD evaluation family",
                description="Fixed target with end-of-day trailing drawdown, daily limit, and 35 traded-day expiry.",
                account_size_usd=50_000.0,
                profit_target_usd=3_000.0,
                max_traded_days=35,
                daily_loss_limit_usd=1_000.0,
                trailing_drawdown_usd=2_000.0,
                notes="Trailing floor = peak equity minus trailing allowance, updated on end-of-day realized equity.",
            ),
            PropChallengeRuleset(
                name="static_permissive_45d",
                family="static_drawdown",
                resembles="Static-DD permissive evaluation family",
                description="Fixed target with wider static max loss, no daily limit, and 45 traded-day expiry.",
                account_size_usd=50_000.0,
                profit_target_usd=3_000.0,
                max_traded_days=45,
                static_max_loss_usd=2_500.0,
                notes="Useful to test whether the hierarchy changes once daily stop pressure is relaxed.",
            ),
        )
    )
    output_root: Path | None = None


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
    if isinstance(value, (np.floating,)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    clean = {key: _serialize_value(value) for key, value in payload.items()}
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else float(default)


def _nan_median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else float("nan")


def _nan_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else float("nan")


def _find_latest_source_run(root: Path = EXPORTS_DIR) -> Path:
    candidates = []
    for path in sorted(root.glob(DEFAULT_SOURCE_RUN_GLOB)):
        if not path.is_dir():
            continue
        if not (path / "run_metadata.json").exists():
            continue
        required = all((path / "variants" / variant).exists() for variant in DEFAULT_VARIANT_ORDER)
        if required:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No regime/sizing export with nominal and sizing_3state variants was found.")
    return candidates[-1]


def _read_run_metadata(source_root: Path) -> dict[str, Any]:
    path = source_root / "run_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_is_fraction(metadata: dict[str, Any]) -> float:
    spec_payload = metadata.get("spec", {})
    raw = spec_payload.get("is_fraction", 0.70)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.70
    return value if 0.0 < value < 1.0 else 0.70


def _summary_row_map(source_root: Path) -> dict[str, dict[str, Any]]:
    path = source_root / "summary_variants.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "variant_name" not in df.columns:
        return {}
    return {
        str(row["variant_name"]): {key: _serialize_value(value) for key, value in row.items()}
        for _, row in df.iterrows()
    }


def _normalize_daily_results(daily_results: pd.DataFrame) -> pd.DataFrame:
    daily = daily_results.copy()
    daily["session_date"] = pd.to_datetime(daily["session_date"]).dt.date
    for column in (
        "daily_pnl_usd",
        "daily_gross_pnl_usd",
        "daily_fees_usd",
        "daily_trade_count",
        "daily_loss_count",
    ):
        if column not in daily.columns:
            daily[column] = 0.0
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
    return daily.sort_values("session_date").reset_index(drop=True)


def _load_variant_input(
    source_root: Path,
    variant_name: str,
    summary_rows: dict[str, dict[str, Any]],
) -> VariantInput:
    variant_root = source_root / "variants" / variant_name
    if not variant_root.exists():
        raise FileNotFoundError(f"Variant directory not found: {variant_root}")

    trades_path = variant_root / "trades.csv"
    daily_path = variant_root / "daily_results.csv"
    controls_path = variant_root / "controls.csv"
    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    controls = pd.read_csv(controls_path) if controls_path.exists() else pd.DataFrame()
    account_series = (
        pd.to_numeric(trades.get("account_size_usd"), errors="coerce").dropna()
        if not trades.empty and "account_size_usd" in trades.columns
        else pd.Series(dtype=float)
    )
    reference_account_size = float(account_series.iloc[0]) if not account_series.empty else 50_000.0

    if daily_path.exists():
        daily_results = pd.read_csv(daily_path)
    else:
        base_daily_path = source_root / "variants" / DEFAULT_VARIANT_ORDER[0] / "daily_results.csv"
        if not base_daily_path.exists():
            raise FileNotFoundError(f"Missing daily_results fallback for variant {variant_name}.")
        base_daily = pd.read_csv(base_daily_path)
        all_sessions = pd.to_datetime(base_daily["session_date"]).dt.date.tolist()
        daily_results = _rebuild_daily_results_from_trades(
            trades=trades,
            all_sessions=all_sessions,
            initial_capital=reference_account_size,
        )

    label_map = {
        "nominal": "nominal_mnq_orb",
        "sizing_3state_realized_vol_ratio_15_60": "sizing_3state_realized_vol_ratio_15_60",
    }
    return VariantInput(
        variant_name=variant_name,
        label=label_map.get(variant_name, variant_name),
        source_root=variant_root,
        trades=trades,
        daily_results=_normalize_daily_results(daily_results),
        controls=controls,
        reference_account_size_usd=reference_account_size,
        source_summary_row=summary_rows.get(variant_name, {}),
    )


def _scope_daily_results(daily_results: pd.DataFrame, is_fraction: float, scope: str) -> pd.DataFrame:
    if scope not in {"overall", "oos"}:
        raise ValueError(f"Unsupported scope '{scope}'.")
    ordered = _normalize_daily_results(daily_results)
    if scope == "overall":
        return ordered.reset_index(drop=True)
    split_idx = int(len(ordered) * float(is_fraction))
    split_idx = max(1, min(len(ordered) - 1, split_idx))
    return ordered.iloc[split_idx:].copy().reset_index(drop=True)


def _scale_daily_results_for_ruleset(daily_results: pd.DataFrame, scale: float) -> pd.DataFrame:
    if abs(scale - 1.0) < 1e-12:
        return daily_results.copy()
    scaled = daily_results.copy()
    for column in ("daily_pnl_usd", "daily_gross_pnl_usd", "daily_fees_usd"):
        if column in scaled.columns:
            scaled[column] = pd.to_numeric(scaled[column], errors="coerce").fillna(0.0) * float(scale)
    return scaled


def _active_global_floor(
    ruleset: PropChallengeRuleset,
    start_equity: float,
    peak_equity: float,
) -> tuple[float, str | None, float | None]:
    candidates: list[tuple[float, str, float]] = []
    if ruleset.static_max_loss_usd is not None:
        candidates.append((start_equity - float(ruleset.static_max_loss_usd), "static_max_loss", float(ruleset.static_max_loss_usd)))
    if ruleset.trailing_drawdown_usd is not None:
        candidates.append((peak_equity - float(ruleset.trailing_drawdown_usd), "trailing_drawdown", float(ruleset.trailing_drawdown_usd)))
    if not candidates:
        return float("-inf"), None, None
    floor_value, floor_name, allowance = max(candidates, key=lambda item: item[0])
    return float(floor_value), floor_name, allowance


def simulate_challenge_path(
    daily_results: pd.DataFrame,
    ruleset: PropChallengeRuleset,
    reference_account_size_usd: float = 50_000.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = _normalize_daily_results(daily_results)
    scale = _safe_div(ruleset.account_size_usd, reference_account_size_usd, default=1.0)
    ordered = _scale_daily_results_for_ruleset(ordered, scale=scale)

    start_equity = float(ruleset.account_size_usd)
    equity = start_equity
    peak_equity = start_equity
    max_favorable_excursion = 0.0
    max_adverse_excursion = 0.0
    max_drawdown_usd = 0.0
    min_global_buffer_ratio = float("inf")
    min_daily_buffer_ratio = float("inf")
    near_limit_days = 0
    traded_days = 0
    calendar_days = 0
    half_target_hit = False
    daily_limit_before_half_target = False
    global_limit_before_half_target = False
    history_rows: list[dict[str, Any]] = []

    status = "open"
    failure_reason = ""
    days_to_pass = float("nan")
    days_to_fail = float("nan")
    target_threshold = float(ruleset.profit_target_usd)
    half_target_threshold = float(ruleset.profit_target_usd) * float(ruleset.half_target_frac)
    daily_limit_breached_any = False
    static_breach_any = False
    trailing_breach_any = False

    for _, row in ordered.iterrows():
        calendar_days += 1
        session_date = pd.to_datetime(row["session_date"]).date()
        daily_pnl = float(pd.to_numeric(pd.Series([row.get("daily_pnl_usd", 0.0)]), errors="coerce").iloc[0])
        traded = bool(float(pd.to_numeric(pd.Series([row.get("daily_trade_count", 0.0)]), errors="coerce").iloc[0]) > 0.0)
        if traded:
            traded_days += 1

        equity += daily_pnl
        peak_equity = max(peak_equity, equity)
        global_floor, global_floor_name, active_global_allowance = _active_global_floor(
            ruleset=ruleset,
            start_equity=start_equity,
            peak_equity=peak_equity,
        )
        global_buffer_usd = float("inf") if not math.isfinite(global_floor) else float(equity - global_floor)
        current_drawdown = float(equity - peak_equity)
        max_drawdown_usd = min(max_drawdown_usd, current_drawdown)
        daily_limit_breached = bool(
            ruleset.daily_loss_limit_usd is not None and daily_pnl <= -float(ruleset.daily_loss_limit_usd)
        )
        global_limit_breached = bool(math.isfinite(global_floor) and equity <= global_floor)

        if not half_target_hit and (equity - start_equity) >= half_target_threshold:
            half_target_hit = True

        if not half_target_hit and daily_limit_breached:
            daily_limit_before_half_target = True
        if not half_target_hit and global_limit_breached:
            global_limit_before_half_target = True

        if ruleset.daily_loss_limit_usd is not None:
            daily_buffer_ratio = _safe_div(
                float(ruleset.daily_loss_limit_usd) - abs(min(daily_pnl, 0.0)),
                float(ruleset.daily_loss_limit_usd),
                default=float("inf"),
            )
            min_daily_buffer_ratio = min(min_daily_buffer_ratio, daily_buffer_ratio)
        else:
            daily_buffer_ratio = float("inf")

        if active_global_allowance is not None:
            global_buffer_ratio = _safe_div(global_buffer_usd, active_global_allowance, default=float("inf"))
            min_global_buffer_ratio = min(min_global_buffer_ratio, global_buffer_ratio)
        else:
            global_buffer_ratio = float("inf")

        near_daily = bool(
            ruleset.daily_loss_limit_usd is not None
            and abs(min(daily_pnl, 0.0)) >= float(ruleset.daily_loss_limit_usd) * (1.0 - float(ruleset.near_limit_buffer_frac))
        )
        near_global = bool(
            active_global_allowance is not None
            and global_buffer_usd <= float(active_global_allowance) * float(ruleset.near_limit_buffer_frac)
        )
        near_limit = bool(near_daily or near_global)
        near_limit_days += int(near_limit)

        run_pnl = equity - start_equity
        max_favorable_excursion = max(max_favorable_excursion, run_pnl)
        max_adverse_excursion = min(max_adverse_excursion, run_pnl)

        if daily_limit_breached:
            daily_limit_breached_any = True
        if global_limit_breached and global_floor_name == "static_max_loss":
            static_breach_any = True
        if global_limit_breached and global_floor_name == "trailing_drawdown":
            trailing_breach_any = True

        history_rows.append(
            {
                "session_date": session_date,
                "daily_pnl_usd": daily_pnl,
                "daily_trade_count": int(float(row.get("daily_trade_count", 0.0))),
                "equity": equity,
                "peak_equity": peak_equity,
                "global_floor_usd": global_floor if math.isfinite(global_floor) else np.nan,
                "global_floor_name": global_floor_name,
                "global_buffer_usd": global_buffer_usd if math.isfinite(global_buffer_usd) else np.nan,
                "daily_limit_breached": daily_limit_breached,
                "global_limit_breached": global_limit_breached,
                "near_limit": near_limit,
                "traded_days_elapsed": traded_days,
                "calendar_days_elapsed": calendar_days,
            }
        )

        if daily_limit_breached:
            status = "fail"
            failure_reason = "daily_loss_limit"
            days_to_fail = float(traded_days)
            break

        if global_limit_breached:
            status = "fail"
            failure_reason = str(global_floor_name or "global_limit")
            days_to_fail = float(traded_days)
            break

        if run_pnl >= target_threshold:
            status = "pass"
            days_to_pass = float(traded_days)
            break

        if ruleset.max_traded_days is not None and traded_days >= int(ruleset.max_traded_days):
            status = "expire"
            break

    if status == "open":
        status = "expire" if ruleset.max_traded_days is not None else "fail"
        if status == "fail":
            failure_reason = "insufficient_history"
            days_to_fail = float(traded_days)

    history = pd.DataFrame(history_rows)
    min_global_buffer_ratio = float("nan") if min_global_buffer_ratio == float("inf") else float(min_global_buffer_ratio)
    min_daily_buffer_ratio = float("nan") if min_daily_buffer_ratio == float("inf") else float(min_daily_buffer_ratio)
    near_fail = bool(
        status != "fail"
        and (
            (pd.notna(min_global_buffer_ratio) and min_global_buffer_ratio <= float(ruleset.near_fail_buffer_frac))
            or (pd.notna(min_daily_buffer_ratio) and min_daily_buffer_ratio <= float(ruleset.near_fail_buffer_frac))
        )
    )
    final_pnl_usd = float(equity - start_equity)

    result = {
        "status": status,
        "pass": bool(status == "pass"),
        "fail": bool(status == "fail"),
        "expire": bool(status == "expire"),
        "failure_reason": failure_reason,
        "days_to_pass": days_to_pass,
        "days_to_fail": days_to_fail,
        "days_traded": int(traded_days),
        "calendar_days": int(calendar_days),
        "final_pnl_usd": final_pnl_usd,
        "max_favorable_excursion_usd": float(max_favorable_excursion),
        "max_adverse_excursion_usd": float(max_adverse_excursion),
        "max_drawdown_usd": float(max_drawdown_usd),
        "daily_loss_limit_breached": bool(daily_limit_breached_any),
        "global_max_loss_breached": bool(static_breach_any or trailing_breach_any),
        "static_max_loss_breached": bool(static_breach_any),
        "trailing_drawdown_breached": bool(trailing_breach_any),
        "half_target_reached": bool(half_target_hit),
        "daily_limit_before_half_target": bool(daily_limit_before_half_target),
        "global_limit_before_half_target": bool(global_limit_before_half_target),
        "near_fail": near_fail,
        "time_near_limit_share": _safe_div(near_limit_days, max(calendar_days, 1), default=0.0),
        "min_global_buffer_ratio": min_global_buffer_ratio,
        "min_daily_buffer_ratio": min_daily_buffer_ratio,
    }
    return history, result


def _eligible_rolling_start_indices(
    daily_results: pd.DataFrame,
    max_traded_days: int | None,
) -> list[int]:
    if max_traded_days is None:
        return list(range(len(daily_results)))
    traded_mask = pd.to_numeric(daily_results["daily_trade_count"], errors="coerce").fillna(0.0).gt(0).astype(int)
    remaining_traded = traded_mask.iloc[::-1].cumsum().iloc[::-1]
    return remaining_traded.loc[remaining_traded >= int(max_traded_days)].index.tolist()


def _eligible_rolling_start_dates(
    daily_results: pd.DataFrame,
    max_traded_days: int | None,
) -> list:
    ordered = _normalize_daily_results(daily_results)
    return [ordered.iloc[idx]["session_date"] for idx in _eligible_rolling_start_indices(ordered, max_traded_days=max_traded_days)]


def run_rolling_start_simulations(
    daily_results: pd.DataFrame,
    variant: VariantInput,
    ruleset: PropChallengeRuleset,
    start_dates: list | None = None,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    allowed_dates = None if start_dates is None else set(pd.to_datetime(pd.Index(start_dates)).date)
    rows: list[dict[str, Any]] = []
    for idx in _eligible_rolling_start_indices(ordered, max_traded_days=ruleset.max_traded_days):
        if allowed_dates is not None and ordered.iloc[idx]["session_date"] not in allowed_dates:
            continue
        subset = ordered.iloc[idx:].copy().reset_index(drop=True)
        _, result = simulate_challenge_path(
            daily_results=subset,
            ruleset=ruleset,
            reference_account_size_usd=variant.reference_account_size_usd,
        )
        rows.append(
            {
                "simulation_method": "rolling_start",
                "variant_name": variant.variant_name,
                "variant_label": variant.label,
                "ruleset_name": ruleset.name,
                "run_id": f"rolling_{idx}",
                "start_session_date": subset.iloc[0]["session_date"],
                "source_session_count": int(len(subset)),
                **result,
            }
        )
    return pd.DataFrame(rows)


def _sample_block_bootstrap_daily(
    daily_results: pd.DataFrame,
    ruleset: PropChallengeRuleset,
    block_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    if ordered.empty:
        return ordered.copy()

    effective_block = max(1, min(int(block_size), len(ordered)))
    max_start = max(len(ordered) - effective_block, 0)
    sampled_rows: list[dict[str, Any]] = []
    traded_days = 0
    synthetic_idx = 0
    max_rows = len(ordered) * 4

    while True:
        start_idx = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        block = ordered.iloc[start_idx : start_idx + effective_block]
        for _, row in block.iterrows():
            synthetic_idx += 1
            row_dict = row.to_dict()
            row_dict["source_session_date"] = row_dict["session_date"]
            row_dict["session_date"] = synthetic_idx
            sampled_rows.append(row_dict)
            if float(row.get("daily_trade_count", 0.0)) > 0:
                traded_days += 1
            if ruleset.max_traded_days is not None and traded_days >= int(ruleset.max_traded_days):
                return pd.DataFrame(sampled_rows)
            if ruleset.max_traded_days is None and len(sampled_rows) >= len(ordered):
                return pd.DataFrame(sampled_rows)
            if len(sampled_rows) >= max_rows:
                return pd.DataFrame(sampled_rows)


def run_bootstrap_simulations(
    daily_results: pd.DataFrame,
    variant: VariantInput,
    ruleset: PropChallengeRuleset,
    n_paths: int,
    block_size: int,
    random_seed: int,
) -> pd.DataFrame:
    ordered = _normalize_daily_results(daily_results)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    for path_idx in range(int(n_paths)):
        sampled = _sample_block_bootstrap_daily(
            daily_results=ordered,
            ruleset=ruleset,
            block_size=block_size,
            rng=rng,
        )
        _, result = simulate_challenge_path(
            daily_results=sampled,
            ruleset=ruleset,
            reference_account_size_usd=variant.reference_account_size_usd,
        )
        rows.append(
            {
                "simulation_method": f"bootstrap_block_{int(block_size)}",
                "variant_name": variant.variant_name,
                "variant_label": variant.label,
                "ruleset_name": ruleset.name,
                "run_id": f"bootstrap_{path_idx + 1}",
                "start_session_date": pd.NA,
                "source_session_count": int(len(sampled)),
                **result,
            }
        )
    return pd.DataFrame(rows)


def aggregate_simulation_runs(runs: pd.DataFrame) -> dict[str, Any]:
    if runs.empty:
        return {
            "run_count": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "expire_rate": 0.0,
            "median_days_to_pass": float("nan"),
            "mean_days_to_pass": float("nan"),
            "median_days_to_fail": float("nan"),
            "mean_days_to_fail": float("nan"),
            "median_final_pnl_usd": float("nan"),
            "mean_final_pnl_usd": float("nan"),
            "p25_final_pnl_usd": float("nan"),
            "p75_final_pnl_usd": float("nan"),
            "daily_loss_violation_rate": 0.0,
            "global_max_loss_violation_rate": 0.0,
            "static_max_loss_violation_rate": 0.0,
            "trailing_drawdown_violation_rate": 0.0,
            "daily_limit_before_half_target_rate": 0.0,
            "global_limit_before_half_target_rate": 0.0,
            "half_target_reached_rate": 0.0,
            "near_fail_rate": 0.0,
            "mean_time_near_limit_share": 0.0,
            "median_mfe_usd": float("nan"),
            "median_mae_usd": float("nan"),
            "worst_mae_usd": float("nan"),
            "median_max_drawdown_usd": float("nan"),
            "worst_max_drawdown_usd": float("nan"),
            "median_days_traded": float("nan"),
            "mean_days_traded": float("nan"),
        }

    pass_mask = runs["pass"].astype(bool)
    fail_mask = runs["fail"].astype(bool)
    expire_mask = runs["expire"].astype(bool)
    return {
        "run_count": int(len(runs)),
        "pass_rate": float(pass_mask.mean()),
        "fail_rate": float(fail_mask.mean()),
        "expire_rate": float(expire_mask.mean()),
        "median_days_to_pass": _nan_median(runs.loc[pass_mask, "days_to_pass"]),
        "mean_days_to_pass": _nan_mean(runs.loc[pass_mask, "days_to_pass"]),
        "median_days_to_fail": _nan_median(runs.loc[fail_mask, "days_to_fail"]),
        "mean_days_to_fail": _nan_mean(runs.loc[fail_mask, "days_to_fail"]),
        "median_final_pnl_usd": _nan_median(runs["final_pnl_usd"]),
        "mean_final_pnl_usd": _nan_mean(runs["final_pnl_usd"]),
        "p25_final_pnl_usd": float(pd.to_numeric(runs["final_pnl_usd"], errors="coerce").quantile(0.25)),
        "p75_final_pnl_usd": float(pd.to_numeric(runs["final_pnl_usd"], errors="coerce").quantile(0.75)),
        "daily_loss_violation_rate": float(runs["daily_loss_limit_breached"].astype(bool).mean()),
        "global_max_loss_violation_rate": float(runs["global_max_loss_breached"].astype(bool).mean()),
        "static_max_loss_violation_rate": float(runs["static_max_loss_breached"].astype(bool).mean()),
        "trailing_drawdown_violation_rate": float(runs["trailing_drawdown_breached"].astype(bool).mean()),
        "daily_limit_before_half_target_rate": float(runs["daily_limit_before_half_target"].astype(bool).mean()),
        "global_limit_before_half_target_rate": float(runs["global_limit_before_half_target"].astype(bool).mean()),
        "half_target_reached_rate": float(runs["half_target_reached"].astype(bool).mean()),
        "near_fail_rate": float(runs["near_fail"].astype(bool).mean()),
        "mean_time_near_limit_share": _nan_mean(runs["time_near_limit_share"]),
        "median_mfe_usd": _nan_median(runs["max_favorable_excursion_usd"]),
        "median_mae_usd": _nan_median(runs["max_adverse_excursion_usd"]),
        "worst_mae_usd": float(pd.to_numeric(runs["max_adverse_excursion_usd"], errors="coerce").min()),
        "median_max_drawdown_usd": _nan_median(runs["max_drawdown_usd"]),
        "worst_max_drawdown_usd": float(pd.to_numeric(runs["max_drawdown_usd"], errors="coerce").min()),
        "median_days_traded": _nan_median(runs["days_traded"]),
        "mean_days_traded": _nan_mean(runs["days_traded"]),
    }


def _prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def compare_ruleset_pair(summary_by_variant: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ruleset_name, pair_df in summary_by_variant.groupby("ruleset_name", sort=True):
        if len(pair_df) < 2:
            continue
        ordered = pair_df.sort_values("variant_name").reset_index(drop=True)
        nominal = ordered.loc[ordered["variant_name"] == "nominal"]
        sizing = ordered.loc[ordered["variant_name"] == "sizing_3state_realized_vol_ratio_15_60"]
        if nominal.empty or sizing.empty:
            continue
        nominal_row = nominal.iloc[0]
        sizing_row = sizing.iloc[0]

        pass_edge = float(sizing_row["bootstrap_pass_rate"] - nominal_row["bootstrap_pass_rate"])
        rolling_edge = float(sizing_row["rolling_start_pass_rate"] - nominal_row["rolling_start_pass_rate"])
        fail_edge = float(sizing_row["bootstrap_fail_rate"] - nominal_row["bootstrap_fail_rate"])
        survival_edge = float(
            nominal_row["bootstrap_global_max_loss_violation_rate"] - sizing_row["bootstrap_global_max_loss_violation_rate"]
        )
        speed_edge = float(
            pd.to_numeric(pd.Series([nominal_row["bootstrap_median_days_to_pass"]]), errors="coerce").fillna(np.nan).iloc[0]
            - pd.to_numeric(pd.Series([sizing_row["bootstrap_median_days_to_pass"]]), errors="coerce").fillna(np.nan).iloc[0]
        )

        if pass_edge >= 0.03 and rolling_edge >= 0.03 and fail_edge <= 0.02:
            verdict = "sizing_3state meilleur candidat prop"
        elif pass_edge <= -0.03 and rolling_edge <= -0.03 and fail_edge >= -0.02:
            verdict = "nominal meilleur candidat prop"
        elif survival_edge >= 0.05 and pass_edge >= -0.05:
            verdict = "sizing_3state plus survivant mais plus lent"
        elif survival_edge <= -0.05 and pass_edge <= 0.05:
            verdict = "nominal plus robuste sans sacrifice clair"
        elif abs(pass_edge) < 0.03 and abs(survival_edge) < 0.05:
            verdict = "hierarchie proche / dependant du critere"
        elif pass_edge > 0 and fail_edge > 0:
            verdict = "sizing_3state gagne en pass rate mais paye en risque"
        elif pass_edge < 0 and fail_edge < 0:
            verdict = "nominal gagne en pass rate mais paye en risque"
        else:
            verdict = "hierarchie depend du ruleset"

        rows.append(
            {
                "ruleset_name": ruleset_name,
                "nominal_bootstrap_pass_rate": nominal_row["bootstrap_pass_rate"],
                "sizing_bootstrap_pass_rate": sizing_row["bootstrap_pass_rate"],
                "nominal_bootstrap_fail_rate": nominal_row["bootstrap_fail_rate"],
                "sizing_bootstrap_fail_rate": sizing_row["bootstrap_fail_rate"],
                "nominal_bootstrap_median_days_to_pass": nominal_row["bootstrap_median_days_to_pass"],
                "sizing_bootstrap_median_days_to_pass": sizing_row["bootstrap_median_days_to_pass"],
                "nominal_bootstrap_global_violation_rate": nominal_row["bootstrap_global_max_loss_violation_rate"],
                "sizing_bootstrap_global_violation_rate": sizing_row["bootstrap_global_max_loss_violation_rate"],
                "pass_rate_edge_sizing_minus_nominal": pass_edge,
                "rolling_pass_rate_edge_sizing_minus_nominal": rolling_edge,
                "fail_rate_edge_sizing_minus_nominal": fail_edge,
                "survival_edge_nominal_minus_sizing_global_violation": survival_edge,
                "speed_edge_nominal_minus_sizing_days_to_pass": speed_edge,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _overall_verdict(comparison_table: pd.DataFrame, summary_by_variant: pd.DataFrame) -> str:
    if summary_by_variant.empty:
        return "aucune conclusion exploitable"

    best_pass = float(pd.to_numeric(summary_by_variant["bootstrap_pass_rate"], errors="coerce").max())
    best_rolling = float(pd.to_numeric(summary_by_variant["rolling_start_pass_rate"], errors="coerce").max())
    if best_pass < 0.35 and best_rolling < 0.35:
        return "aucune des deux n'a une geometrie suffisamment solide pour un challenge exigeant"

    verdicts = comparison_table["verdict"].tolist() if not comparison_table.empty else []
    nominal_wins = sum("nominal" in verdict and "sizing_3state" not in verdict for verdict in verdicts)
    sizing_wins = sum("sizing_3state" in verdict and "nominal" not in verdict for verdict in verdicts)
    sizing_survival_wins = sum("plus survivant" in verdict for verdict in verdicts)

    if nominal_wins >= 2 and sizing_wins == 0:
        return "le nominal reste superieur meme sous contraintes prop"
    if sizing_wins >= 2 and nominal_wins == 0:
        return "la version sizing_3state devient la meilleure candidate dans le cadre prop teste"
    if nominal_wins >= 1 and sizing_survival_wins >= 1:
        return "la version sizing_3state est inferieure en rendement brut mais devient plus defendable sous rulesets stricts"
    if sizing_wins > 0 and nominal_wins > 0:
        return "la hierarchie depend du ruleset"
    if any("plus survivant" in verdict for verdict in verdicts):
        return "la version sizing_3state est inferieure en rendement brut mais devient plus defendable sous rulesets stricts"
    return "la hierarchie depend du ruleset"


def _variant_summary_rows(
    variant_inputs: list[VariantInput],
    scoped_daily_map: dict[str, pd.DataFrame],
    rulesets: tuple[PropChallengeRuleset, ...],
    bootstrap_paths: int,
    bootstrap_block_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    run_frames: list[pd.DataFrame] = []

    for ruleset_idx, ruleset in enumerate(rulesets):
        common_start_dates: set | None = None
        for variant in variant_inputs:
            variant_dates = set(_eligible_rolling_start_dates(scoped_daily_map[variant.variant_name], max_traded_days=ruleset.max_traded_days))
            common_start_dates = variant_dates if common_start_dates is None else (common_start_dates & variant_dates)
        ordered_common_dates = sorted(common_start_dates) if common_start_dates else []

        for variant in variant_inputs:
            scoped_daily = scoped_daily_map[variant.variant_name]
            rolling_runs = run_rolling_start_simulations(
                daily_results=scoped_daily,
                variant=variant,
                ruleset=ruleset,
                start_dates=ordered_common_dates,
            )
            bootstrap_runs = run_bootstrap_simulations(
                daily_results=scoped_daily,
                variant=variant,
                ruleset=ruleset,
                n_paths=bootstrap_paths,
                block_size=bootstrap_block_size,
                random_seed=random_seed + (ruleset_idx * 1000),
            )
            combined_runs = pd.concat([rolling_runs, bootstrap_runs], ignore_index=True)
            run_frames.append(combined_runs)

            rolling_metrics = aggregate_simulation_runs(rolling_runs)
            bootstrap_metrics = aggregate_simulation_runs(bootstrap_runs)
            daily = _normalize_daily_results(scoped_daily)
            summary_rows.append(
                {
                    "ruleset_name": ruleset.name,
                    "ruleset_family": ruleset.family,
                    "variant_name": variant.variant_name,
                    "variant_label": variant.label,
                    "analysis_scope_session_count": int(len(daily)),
                    "analysis_scope_traded_day_count": int(pd.to_numeric(daily["daily_trade_count"], errors="coerce").fillna(0.0).gt(0).sum()),
                    "analysis_scope_start_date": daily["session_date"].iloc[0] if not daily.empty else pd.NA,
                    "analysis_scope_end_date": daily["session_date"].iloc[-1] if not daily.empty else pd.NA,
                    "source_oos_net_pnl": variant.source_summary_row.get("oos_net_pnl"),
                    "source_oos_sharpe": variant.source_summary_row.get("oos_sharpe"),
                    "source_oos_profit_factor": variant.source_summary_row.get("oos_profit_factor"),
                    "source_oos_max_drawdown": variant.source_summary_row.get("oos_max_drawdown"),
                    **_prefixed_metrics("rolling_start", rolling_metrics),
                    **_prefixed_metrics("bootstrap", bootstrap_metrics),
                }
            )

    run_table = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    summary_table = pd.DataFrame(summary_rows)
    return summary_table, run_table


def _ruleset_table(rulesets: tuple[PropChallengeRuleset, ...]) -> pd.DataFrame:
    rows = []
    for ruleset in rulesets:
        rows.append(
            {
                "ruleset_name": ruleset.name,
                "family": ruleset.family,
                "resembles": ruleset.resembles,
                "description": ruleset.description,
                "account_size_usd": ruleset.account_size_usd,
                "profit_target_usd": ruleset.profit_target_usd,
                "max_traded_days": ruleset.max_traded_days,
                "daily_loss_limit_usd": ruleset.daily_loss_limit_usd,
                "static_max_loss_usd": ruleset.static_max_loss_usd,
                "trailing_drawdown_usd": ruleset.trailing_drawdown_usd,
                "notes": ruleset.notes,
            }
        )
    return pd.DataFrame(rows)


def _build_markdown_summary(
    spec: CampaignSpec,
    source_root: Path,
    metadata: dict[str, Any],
    ruleset_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
) -> str:
    overall_verdict = _overall_verdict(comparison_table=comparison_table, summary_by_variant=summary_table)
    lines = [
        "# MNQ ORB Prop Challenge Simulation",
        "",
        "## Perimetre",
        "",
        f"- Source export: `{source_root}`",
        f"- Scope principal: `{spec.primary_scope}`",
        f"- Bootstrap: `{spec.bootstrap_paths}` paths en block bootstrap `{spec.bootstrap_block_size}` jours",
        f"- Seed: `{spec.random_seed}`",
        "- Variantes comparees strictement: `nominal` vs `sizing_3state_realized_vol_ratio_15_60`",
        "",
        "## Audit Data",
        "",
        f"- Source run timestamp: `{metadata.get('run_timestamp')}`",
        f"- Dataset: `{metadata.get('dataset_path')}`",
        f"- Aggregation rule: `{metadata.get('selected_aggregation_rule')}`",
        "- Primary decision evidence uses the OOS slice of the existing regime/sizing export.",
        "",
        "## Rulesets",
        "",
        "```text",
        ruleset_table.to_string(index=False),
        "```",
        "",
        "## Comparative Summary",
        "",
        "```text",
        comparison_table.to_string(index=False) if not comparison_table.empty else "No comparison rows.",
        "```",
        "",
        "## Variant Summary",
        "",
        "```text",
        summary_table[
            [
                "ruleset_name",
                "variant_name",
                "rolling_start_pass_rate",
                "rolling_start_fail_rate",
                "rolling_start_expire_rate",
                "rolling_start_median_days_to_pass",
                "bootstrap_pass_rate",
                "bootstrap_fail_rate",
                "bootstrap_expire_rate",
                "bootstrap_median_days_to_pass",
                "bootstrap_global_max_loss_violation_rate",
                "bootstrap_daily_loss_violation_rate",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Assumptions",
        "",
        "- Daily loss and trailing drawdown are enforced on realized end-of-day PnL, not intraday mark-to-market.",
        "- Expiry is measured in traded days only, which matches the stated objective better than wall-clock sessions for a one-trade-per-day strategy.",
        "- The bootstrap keeps path dependence simple by resampling contiguous daily blocks instead of individual days.",
        "",
        "## Verdict Final",
        "",
        f"- `{overall_verdict}`",
        "",
    ]
    return "\n".join(lines)


def run_campaign(spec: CampaignSpec) -> dict[str, Path]:
    ensure_directories()
    source_root = spec.source_run_root or _find_latest_source_run()
    metadata = _read_run_metadata(source_root)
    is_fraction = _source_is_fraction(metadata)
    summary_rows = _summary_row_map(source_root)
    variant_inputs = [
        _load_variant_input(source_root=source_root, variant_name=variant_name, summary_rows=summary_rows)
        for variant_name in spec.variant_names
    ]
    scoped_daily_map = {
        variant.variant_name: _scope_daily_results(variant.daily_results, is_fraction=is_fraction, scope=spec.primary_scope)
        for variant in variant_inputs
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = spec.output_root or (EXPORTS_DIR / f"mnq_orb_prop_challenge_simulation_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    ruleset_table = _ruleset_table(spec.rulesets)
    summary_table, simulation_runs = _variant_summary_rows(
        variant_inputs=variant_inputs,
        scoped_daily_map=scoped_daily_map,
        rulesets=spec.rulesets,
        bootstrap_paths=spec.bootstrap_paths,
        bootstrap_block_size=spec.bootstrap_block_size,
        random_seed=spec.random_seed,
    )
    comparison_table = compare_ruleset_pair(summary_table)

    summary_rulesets_path = output_root / "summary_rulesets.csv"
    summary_variants_path = output_root / "summary_variants.csv"
    simulation_runs_path = output_root / "simulation_runs.csv"
    rolling_starts_path = output_root / "rolling_starts.csv"
    bootstrap_runs_path = output_root / "bootstrap_runs.csv"
    comparison_path = output_root / "comparison_table.csv"
    markdown_path = output_root / "prop_challenge_summary.md"
    metadata_path = output_root / "run_metadata.json"

    ruleset_table.to_csv(summary_rulesets_path, index=False)
    summary_table.to_csv(summary_variants_path, index=False)
    simulation_runs.to_csv(simulation_runs_path, index=False)
    simulation_runs.loc[simulation_runs["simulation_method"] == "rolling_start"].to_csv(rolling_starts_path, index=False)
    simulation_runs.loc[simulation_runs["simulation_method"].str.startswith("bootstrap_")].to_csv(bootstrap_runs_path, index=False)
    comparison_table.to_csv(comparison_path, index=False)
    markdown_path.write_text(
        _build_markdown_summary(
            spec=spec,
            source_root=source_root,
            metadata=metadata,
            ruleset_table=ruleset_table,
            summary_table=summary_table,
            comparison_table=comparison_table,
        ),
        encoding="utf-8",
    )

    _json_dump(
        metadata_path,
        {
            "run_timestamp": datetime.now().isoformat(),
            "source_run_root": str(source_root),
            "primary_scope": spec.primary_scope,
            "source_is_fraction": is_fraction,
            "bootstrap_paths": spec.bootstrap_paths,
            "bootstrap_block_size": spec.bootstrap_block_size,
            "random_seed": spec.random_seed,
            "rulesets": [asdict(ruleset) for ruleset in spec.rulesets],
            "overall_verdict": _overall_verdict(comparison_table=comparison_table, summary_by_variant=summary_table),
        },
    )

    return {
        "summary_rulesets_csv": summary_rulesets_path,
        "summary_variants_csv": summary_variants_path,
        "simulation_runs_csv": simulation_runs_path,
        "rolling_starts_csv": rolling_starts_path,
        "bootstrap_runs_csv": bootstrap_runs_path,
        "comparison_table_csv": comparison_path,
        "summary_markdown": markdown_path,
        "run_metadata_json": metadata_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNQ ORB prop-firm challenge simulation.")
    parser.add_argument(
        "--source-run-root",
        type=Path,
        default=None,
        help="Optional regime/sizing export root. Defaults to the latest audited matching export.",
    )
    parser.add_argument(
        "--primary-scope",
        choices=("overall", "oos"),
        default=DEFAULT_PRIMARY_SCOPE,
        help="Use the full history or only the OOS slice from the source export.",
    )
    parser.add_argument(
        "--bootstrap-paths",
        type=int,
        default=DEFAULT_BOOTSTRAP_PATHS,
        help="Number of bootstrap paths per ruleset and per variant.",
    )
    parser.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=DEFAULT_BOOTSTRAP_BLOCK_SIZE,
        help="Contiguous daily block size used for bootstrap resampling.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Bootstrap random seed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional explicit export directory.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_campaign(
        CampaignSpec(
            source_run_root=args.source_run_root,
            primary_scope=str(args.primary_scope),
            bootstrap_paths=int(args.bootstrap_paths),
            bootstrap_block_size=int(args.bootstrap_block_size),
            random_seed=int(args.random_seed),
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
