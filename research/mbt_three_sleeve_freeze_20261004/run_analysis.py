"""Freeze and diagnose the three retained MBT parameter regions.

The frozen object is a parameter region, not a selected best cell.  For portfolio
diagnostics, each MBT sleeve is therefore the equal-weight daily-PnL average of
all tested cells inside its frozen region.  This is a research composite, not an
integer-contract execution rule.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.baseline_backtests.baseline_engine import load_frozen_specs, run_baseline_backtest
from research.btc_baseline_backtests.run_btc_baselines import adapt_spec_to_btc, prepare_btc_data
from research.mbt_strategy_heatmaps_20261003.run_mbt_heatmaps import _ribbon, mbt_config

FROZEN_BOUNDS = {
    "rolling_range_breakout": {"entry_channel_length": [20, 24], "exit_channel_length": [7, 15]},
    "trend_pullback_ma_structure": {"fast_pair_scale": [0.50, 0.75], "slow_tail_scale": [0.75, 1.25]},
    "volatility_breakout": {"entry_atr_multiple": [0.30, 0.60], "daily_atr_period": [10, 30]},
}


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "downloaded" / "MBT_v_0_1m_20210425_20261003.parquet"
SIX = REPO / "research" / "portfolio_integration_phase1" / "daily_sleeve_pnl.csv"
FROZEN_DIR = REPO / "strategy_registry" / "psf_mnq" / "frozen_baselines"
FROZEN_MANIFEST = REPO / "strategy_registry" / "psf_mnq" / "frozen_baselines_manifest.csv"
START = date(2021, 5, 3)
END = date(2026, 10, 2)
IS_END = pd.Timestamp("2025-05-28")
OOS_START = pd.Timestamp("2025-05-29")

REGIONS = {
    "rolling_range_breakout": {
        "entry_channel_length": [20, 24],
        "exit_channel_length": [7, 10, 12, 15],
    },
    "trend_pullback_ma_structure": {
        "fast_pair_scale": [0.50, 0.60, 0.75],
        "slow_tail_scale": [0.75, 1.00, 1.25],
    },
    "volatility_breakout": {
        "entry_atr_multiple": [0.30, 0.40, 0.60],
        "daily_atr_period": [10, 14, 20, 30],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sharpe(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(values.std(ddof=1))
    return float(values.mean() / std * math.sqrt(252.0)) if std > 0 else float("nan")


def metrics(series: pd.Series) -> dict[str, float | int | str]:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    cumulative = values.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        "start": str(values.index.min().date()),
        "end": str(values.index.max().date()),
        "sessions": int(len(values)),
        "net_pnl_usd": float(values.sum()),
        "mean_daily_pnl_usd": float(values.mean()),
        "daily_vol_usd": float(values.std(ddof=1)),
        "sharpe_annualized": sharpe(values),
        "max_drawdown_usd": float(drawdown.min()),
        "worst_day_usd": float(values.min()),
    }


def normalized_equal_weight(frame: pd.DataFrame) -> pd.Series:
    vols = frame.std(ddof=1).replace(0.0, np.nan)
    return frame.divide(vols, axis=1).mean(axis=1)


def effective_dimensions(frame: pd.DataFrame) -> float:
    corr = frame.corr().fillna(0.0).to_numpy(float)
    np.fill_diagonal(corr, 1.0)
    eig = np.linalg.eigvalsh(corr)
    return float(eig.sum() ** 2 / np.square(eig).sum())


def apply_cell(spec: dict, mechanism: str, a: float, b: float) -> tuple[dict, dict]:
    structural = spec["canonical_parameters"]["structural"]
    if mechanism == "rolling_range_breakout":
        structural.update(entry_channel_length=int(a), exit_channel_length=int(b))
        params = {"entry_channel_length": int(a), "exit_channel_length": int(b)}
    elif mechanism == "trend_pullback_ma_structure":
        _ribbon(structural, float(a), float(b))
        params = {"fast_pair_scale": float(a), "slow_tail_scale": float(b), "ema_periods": structural["ema_periods"]}
    else:
        # Region declaration is multiple first, period second; implementation uses named fields.
        structural.update(entry_atr_multiple=float(a), daily_atr_period=int(b))
        params = {"entry_atr_multiple": float(a), "daily_atr_period": int(b)}
    return spec, params


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    prepared, roll_sessions = prepare_btc_data(DATA, START, END)
    specs, _, _ = load_frozen_specs(FROZEN_DIR, FROZEN_MANIFEST)
    specs = {item["mechanism_id"]: item for item in specs}
    config = mbt_config()
    cell_series: dict[str, pd.Series] = {}
    cell_rows: list[dict] = []

    for mechanism, region in REGIONS.items():
        keys = list(region)
        for a in region[keys[0]]:
            for b in region[keys[1]]:
                spec = copy.deepcopy(adapt_spec_to_btc(specs[mechanism], config))
                spec, params = apply_cell(spec, mechanism, a, b)
                run = run_baseline_backtest(spec, prepared)
                daily = run.daily[["session_date", "daily_pnl_usd"]].copy()
                daily["session_date"] = pd.to_datetime(daily["session_date"])
                series = daily.set_index("session_date")["daily_pnl_usd"].sort_index()
                name = mechanism + "__" + "__".join(f"{key}={params[key]}" for key in keys)
                cell_series[name] = series
                for scope, scoped in {
                    "full": series,
                    "legacy_is": series.loc[:IS_END],
                    "legacy_oos": series.loc[OOS_START:],
                }.items():
                    cell_rows.append({"mechanism_id": mechanism, "cell_id": name, **params, "scope": scope, **metrics(scoped)})

    cells = pd.concat(cell_series, axis=1).fillna(0.0)
    composites = pd.DataFrame(index=cells.index)
    for mechanism in REGIONS:
        columns = [column for column in cells if column.startswith(mechanism + "__")]
        composites[mechanism] = cells[columns].mean(axis=1)
    composites["three_mbt_equal_weight_raw"] = composites[list(REGIONS)].mean(axis=1)

    portfolio_rows: list[dict] = []
    for scope, scoped in {
        "full": composites,
        "legacy_is": composites.loc[:IS_END],
        "legacy_oos": composites.loc[OOS_START:],
    }.items():
        for column in composites.columns:
            portfolio_rows.append({"scope": scope, "series": column, "allocation": "raw_equal_weight_cells", **metrics(scoped[column])})
        risk_book = normalized_equal_weight(scoped[list(REGIONS)])
        portfolio_rows.append({"scope": scope, "series": "three_mbt_equal_risk", "allocation": "full_window_vol_normalized", **metrics(risk_book)})

    six = pd.read_csv(SIX, parse_dates=["session_date"]).set_index("session_date").sort_index()
    six.columns = [f"current_{column}" for column in six.columns]
    common = six.join(composites[list(REGIONS)], how="inner")
    common = common.loc[(common.index >= six.index.min()) & (common.index <= six.index.max())]
    new_cols = list(REGIONS)
    old_cols = list(six.columns)
    common["current_six_equal_weight_raw"] = common[old_cols].mean(axis=1)
    common["new_three_equal_weight_raw"] = common[new_cols].mean(axis=1)
    common["nine_equal_weight_raw"] = common[old_cols + new_cols].mean(axis=1)
    common["current_six_equal_risk"] = normalized_equal_weight(common[old_cols])
    common["new_three_equal_risk"] = normalized_equal_weight(common[new_cols])
    common["nine_equal_risk"] = normalized_equal_weight(common[old_cols + new_cols])

    integration_rows: list[dict] = []
    for scope, frame in {"common_full": common, "common_legacy_oos": common.loc[OOS_START:]}.items():
        for column in [
            "current_six_equal_weight_raw", "new_three_equal_weight_raw", "nine_equal_weight_raw",
            "current_six_equal_risk", "new_three_equal_risk", "nine_equal_risk",
        ]:
            integration_rows.append({"scope": scope, "series": column, **metrics(frame[column])})

    corr = common[old_cols + new_cols + ["current_six_equal_weight_raw"]].corr()
    corr_oos = common.loc[OOS_START:, old_cols + new_cols + ["current_six_equal_weight_raw"]].corr()
    pairwise_rows: list[dict] = []
    for new in new_cols:
        for old in old_cols + ["current_six_equal_weight_raw"]:
            pairwise_rows.append({
                "new_sleeve": new,
                "comparison": old,
                "pearson_common_full": float(corr.loc[new, old]),
                "pearson_common_legacy_oos": float(corr_oos.loc[new, old]),
            })

    # Linear spanning diagnostic. This is descriptive and in-sample, not a prediction test.
    x = common[old_cols].to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    uniqueness_rows = []
    for new in new_cols:
        y = common[new].to_numpy(float)
        fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        denom = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - float(np.square(y - fitted).sum()) / denom if denom > 0 else float("nan")
        uniqueness_rows.append({"new_sleeve": new, "r2_explained_by_current_six": r2, "variance_unexplained": 1.0 - r2})

    pd.DataFrame(cell_rows).to_csv(HERE / "cell_metrics.csv", index=False)
    cells.to_csv(HERE / "daily_cell_pnl.csv", index_label="session_date")
    composites.to_csv(HERE / "daily_three_sleeve_pnl.csv", index_label="session_date")
    common.to_csv(HERE / "daily_aligned_nine_sleeves.csv", index_label="session_date")
    pd.DataFrame(portfolio_rows).to_csv(HERE / "three_sleeve_metrics.csv", index=False)
    pd.DataFrame(integration_rows).to_csv(HERE / "portfolio_integration_metrics.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(HERE / "orthogonality_vs_six.csv", index=False)
    pd.DataFrame(uniqueness_rows).to_csv(HERE / "linear_uniqueness.csv", index=False)
    corr.to_csv(HERE / "correlation_matrix_common.csv")
    corr_oos.to_csv(HERE / "correlation_matrix_common_legacy_oos.csv")

    metadata = {
        "run_date": "2026-10-04",
        "data_path": str(DATA.relative_to(REPO)),
        "data_sha256": sha256(DATA),
        "six_sleeve_path": str(SIX.relative_to(REPO)),
        "six_sleeve_sha256": sha256(SIX),
        "frozen_bounds": FROZEN_BOUNDS,
        "diagnostic_grid_values_within_bounds": REGIONS,
        "region_cell_counts": {m: int(np.prod([len(v) for v in region.values()])) for m, region in REGIONS.items()},
        "research_composite_rule": "within each mechanism, equal-weight average of daily PnL across every tested cell in the frozen region",
        "execution_caveat": "range composites use fractional averaging and are not a one-contract executable rule",
        "legacy_split": {"is_end": str(IS_END.date()), "oos_start": str(OOS_START.date())},
        "independence_caveat": "legacy OOS was inspected when these regions were chosen and is not independent validation",
        "common_window": [str(common.index.min().date()), str(common.index.max().date())],
        "common_oos_window": [str(common.loc[OOS_START:].index.min().date()), str(common.loc[OOS_START:].index.max().date())],
        "intraday_roll_sessions_excluded": [str(item) for item in roll_sessions],
        "effective_dimensions_current_six": effective_dimensions(common[old_cols]),
        "effective_dimensions_new_three": effective_dimensions(common[new_cols]),
        "effective_dimensions_all_nine": effective_dimensions(common[old_cols + new_cols]),
        "momentum_confirmed_breakout": {
            "status": "not_frozen_not_included",
            "candidate_region": {"adx_threshold": [15, 20], "breakout_lookback": [20, 40]},
            "required_next_step": "independent data validation before retention",
        },
    }
    (HERE / "freeze_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(portfolio_rows).query("series.str.contains('three_mbt')", engine="python").to_string(index=False))
    print(pd.DataFrame(integration_rows).to_string(index=False))
    print(pd.DataFrame(pairwise_rows).to_string(index=False))
    print(pd.DataFrame(uniqueness_rows).to_string(index=False))


if __name__ == "__main__":
    main()
