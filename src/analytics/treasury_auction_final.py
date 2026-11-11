"""Final, non-optimizing Treasury auction pressure replication on ZF and ZN."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analytics.event_flow_studies import _bootstrap_path_summary, _event_matrix, load_bars


MAPPINGS = (
    ("5-Year", "ZF", "primary"),
    ("7-Year", "ZF", "primary"),
    ("7-Year", "ZN", "secondary"),
    ("10-Year", "ZN", "primary"),
)
SHIFTS = (-60, -30, 0, 30, 60)
SUBSAMPLES = (
    ("full", None, None),
    ("pre_2020", None, "2020-01-01"),
    ("2020_plus", "2020-01-01", None),
    ("2021_plus", "2021-01-01", None),
    ("2024_plus", "2024-01-01", None),
)
PRE = (-180, -1)
# A bar is labelled by its start. The close of T+179 is the price known at T+180.
POST = (5, 179)
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260927


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")


def _bootstrap_ci(values: pd.Series, samples: int, seed: int) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if clean.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for idx in range(samples):
        means[idx] = rng.choice(clean, len(clean), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _block_bootstrap_ci(
    values: pd.Series, samples: int, seed: int, block_size: int = 5
) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if clean.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    n = len(clean)
    for idx in range(samples):
        sampled: list[float] = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend(clean[(start + np.arange(block_size)) % n].tolist())
        means[idx] = np.mean(sampled[:n])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _stats_row(
    frame: pd.DataFrame,
    mapping: str,
    sample: str,
    subsample: str,
    seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mapping": mapping,
        "sample": sample,
        "subsample": subsample,
        "events": len(frame),
    }
    for column, prefix in (
        ("pre_return_bps", "pre"),
        ("post_return_bps", "post"),
        ("combined_effect_bps", "combined"),
    ):
        clean = pd.to_numeric(frame[column], errors="coerce").dropna()
        ci_low, ci_high = _bootstrap_ci(clean, BOOTSTRAP_SAMPLES, seed + len(prefix))
        test = stats.ttest_1samp(clean, 0.0) if len(clean) > 1 else None
        row.update(
            {
                f"{prefix}_n": len(clean),
                f"{prefix}_mean_bps": float(clean.mean()) if len(clean) else math.nan,
                f"{prefix}_median_bps": float(clean.median()) if len(clean) else math.nan,
                f"{prefix}_hit_rate": float((clean > 0).mean()) if len(clean) else math.nan,
                f"{prefix}_t_stat": float(test.statistic) if test is not None else math.nan,
                f"{prefix}_p_value": float(test.pvalue) if test is not None else math.nan,
                f"{prefix}_bootstrap_ci_low_bps": ci_low,
                f"{prefix}_bootstrap_ci_high_bps": ci_high,
            }
        )
    block_low, block_high = _block_bootstrap_ci(
        frame["combined_effect_bps"], BOOTSTRAP_SAMPLES, seed + 99
    )
    row["combined_block_bootstrap_ci_low_bps"] = block_low
    row["combined_block_bootstrap_ci_high_bps"] = block_high
    return row


def _add_returns(matrix: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    out = details.copy()
    out["pre_return_bps"] = (matrix[PRE[1]] - matrix[PRE[0]]).to_numpy()
    out["post_return_bps"] = (matrix[POST[1]] - matrix[POST[0]]).to_numpy()
    out["combined_effect_bps"] = -out["pre_return_bps"] + out["post_return_bps"]
    out["canonical_hit"] = (out["pre_return_bps"] < 0) & (out["post_return_bps"] > 0)
    return out


def _roll_flag(frame: pd.DataFrame, event_ts: pd.Series) -> list[bool]:
    transitions = frame.index[frame["instrument_id"].ne(frame["instrument_id"].shift())]
    transition_ns = transitions.asi8
    flags: list[bool] = []
    for value in event_ts:
        stamp = pd.Timestamp(value).tz_convert("UTC")
        lo = (stamp - pd.Timedelta(minutes=240)).value
        hi = (stamp + pd.Timedelta(minutes=240)).value
        left = np.searchsorted(transition_ns, lo, side="left")
        right = np.searchsorted(transition_ns, hi, side="right")
        flags.append(bool(right > left))
    return flags


def _mapping_events(calendar: pd.DataFrame, tenor: str, instrument: str) -> pd.DataFrame:
    events = calendar.loc[calendar["tenor"].eq(tenor)].copy().reset_index(drop=True)
    events["instrument"] = instrument
    events["mapping"] = f"{tenor}->{instrument}"
    events["reference_type"] = "competitive_bid_deadline"
    return events


def _shifted_events(events: pd.DataFrame, shift: int) -> pd.DataFrame:
    out = events.copy()
    out["event_ts_utc"] = pd.to_datetime(out["bid_deadline_ts_utc"], utc=True) + pd.Timedelta(minutes=shift)
    out["shift_minutes"] = shift
    return out


def _placebo_events(events: pd.DataFrame, local_time: str, label: str) -> pd.DataFrame:
    out = events.copy()
    dates = pd.to_datetime(out["auction_date"])
    naive = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + local_time)
    out["event_ts_utc"] = naive.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    out["placebo"] = label
    return out


def _control_events(calendar: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    all_dates = set(pd.to_datetime(calendar["auction_date"]).dt.date)
    records: list[dict[str, Any]] = []
    for event_id, row in events.iterrows():
        event_local = pd.Timestamp(row["bid_deadline_ts_utc"]).tz_convert("America/New_York")
        auction_date = pd.Timestamp(row["auction_date"])
        for label, days, direction in (("week_before", -7, -1), ("week_after", 7, 1)):
            candidate = auction_date + pd.Timedelta(days=days)
            while candidate.weekday() >= 5 or candidate.date() in all_dates:
                candidate += pd.Timedelta(days=direction)
            local = pd.Timestamp.combine(candidate.date(), event_local.time()).tz_localize(
                "America/New_York"
            )
            record = row.to_dict()
            record.update(
                {
                    "matched_event_id": event_id,
                    "control_label": label,
                    "control_date": candidate.date(),
                    "event_ts_utc": local.tz_convert("UTC"),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def _subsample(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    dates = pd.to_datetime(frame["auction_date"])
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates < pd.Timestamp(end)
    return frame.loc[mask].copy()


def _plot_path(summary: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = summary["event_minute"].to_numpy(float)
    ax.plot(x, summary["mean_bps"], label="Mean", color="#145DA0")
    ax.plot(x, summary["median_bps"], label="Median", color="#F28E2B")
    ax.fill_between(x, summary["ci_low_bps"], summary["ci_high_bps"], alpha=0.18)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(title=title, xlabel="Minutes from competitive bid deadline", ylabel="Cumulative price return (bps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _concentration(canonical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mapping, group in canonical.groupby("mapping"):
        values = group["combined_effect_bps"].dropna().sort_values(ascending=False)
        absolute = group["combined_effect_bps"].dropna().abs().sort_values(ascending=False)
        total = float(values.sum())
        absolute_total = float(absolute.sum())
        rows.append(
            {
                "mapping": mapping,
                "events": len(values),
                "total_combined_effect_bps": total,
                "best_5_share_of_signed_total": float(values.head(5).sum() / total) if total else math.nan,
                "best_10_share_of_signed_total": float(values.head(10).sum() / total) if total else math.nan,
                "largest_5_abs_share": float(absolute.head(5).sum() / absolute_total) if absolute_total else math.nan,
                "largest_10_abs_share": float(absolute.head(10).sum() / absolute_total) if absolute_total else math.nan,
            }
        )
    return pd.DataFrame(rows)


def run(output_dir: Path, bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    global BOOTSTRAP_SAMPLES
    BOOTSTRAP_SAMPLES = bootstrap_samples
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {
        "ZF": REPO_ROOT / "data/processed/parquet/ZF_v_0_1m_20130517_20260920.parquet",
        "ZN": REPO_ROOT / "data/processed/parquet/ZN_v_0_1m_20130517_20260920.parquet",
    }
    bars = {symbol: load_bars(path) for symbol, path in data_paths.items()}
    calendar = pd.read_csv(REPO_ROOT / "data/raw/treasury_auction_calendar.csv")
    calendar["auction_date"] = pd.to_datetime(calendar["auction_date"])
    calendar["bid_deadline_ts_utc"] = pd.to_datetime(calendar["bid_deadline_ts_utc"], utc=True)

    canonical_frames: list[pd.DataFrame] = []
    path_summaries: list[pd.DataFrame] = []
    shift_rows: list[dict[str, Any]] = []
    placebo_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    subsample_rows: list[dict[str, Any]] = []
    loyo_rows: list[dict[str, Any]] = []

    for mapping_idx, (tenor, instrument, priority) in enumerate(MAPPINGS):
        mapping = f"{tenor}->{instrument}"
        events = _mapping_events(calendar, tenor, instrument)
        canonical_events = _shifted_events(events, 0)
        matrix, details = _event_matrix(bars[instrument], canonical_events, "event_ts_utc", -240, 240)
        details = _add_returns(matrix, details)
        details["priority"] = priority
        details["roll_transition_in_path"] = _roll_flag(
            bars[instrument], details["event_ts_utc"]
        )
        canonical_frames.append(details)

        path_summary = _bootstrap_path_summary(matrix, bootstrap_samples, BOOTSTRAP_SEED + mapping_idx)
        path_summary.insert(0, "mapping", mapping)
        path_summaries.append(path_summary)
        _plot_path(path_summary, output_dir / f"{tenor.lower().replace('-', '_')}_{instrument.lower()}_deadline_path.png", mapping)

        for subsample_name, start, end in SUBSAMPLES:
            subset = _subsample(details, start, end)
            subsample_rows.append(
                _stats_row(subset, mapping, "auction", subsample_name, BOOTSTRAP_SEED + mapping_idx * 100)
            )
        for year in sorted(pd.to_datetime(details["auction_date"]).dt.year.unique()):
            subset = details.loc[pd.to_datetime(details["auction_date"]).dt.year.eq(year)]
            yearly_rows.append(
                _stats_row(subset, mapping, "auction", str(year), BOOTSTRAP_SEED + int(year))
            )
        years = sorted(pd.to_datetime(details["auction_date"]).dt.year.unique())
        for year in years:
            subset = details.loc[~pd.to_datetime(details["auction_date"]).dt.year.eq(year)]
            row = _stats_row(subset, mapping, "auction", f"exclude_{year}", BOOTSTRAP_SEED + int(year))
            row["excluded_year"] = year
            loyo_rows.append(row)

        for shift in SHIFTS:
            shifted = _shifted_events(events, shift)
            shift_matrix, shift_details = _event_matrix(bars[instrument], shifted, "event_ts_utc", -240, 240)
            shift_details = _add_returns(shift_matrix, shift_details)
            row = _stats_row(shift_details, mapping, "shift", "full", BOOTSTRAP_SEED + shift + 500)
            row["shift_minutes"] = shift
            shift_rows.append(row)

        for local_time, label in (("11:00", "same_day_1100_et"), ("14:00", "same_day_1400_et")):
            placebo = _placebo_events(events, local_time, label)
            placebo_matrix, placebo_details = _event_matrix(bars[instrument], placebo, "event_ts_utc", -240, 240)
            placebo_details = _add_returns(placebo_matrix, placebo_details)
            placebo_rows.append(
                _stats_row(placebo_details, mapping, label, "full", BOOTSTRAP_SEED + len(label))
            )

        controls = _control_events(calendar, events)
        control_matrix, control_details = _event_matrix(bars[instrument], controls, "event_ts_utc", -240, 240)
        control_details = _add_returns(control_matrix, control_details)
        control_rows.append(
            _stats_row(control_details, mapping, "matched_non_auction", "full", BOOTSTRAP_SEED + 700)
        )

    canonical = pd.concat(canonical_frames, ignore_index=True)
    condition_path = output_dir / "databento_dataset_conditions.json"
    degraded_dates: set[str] = set()
    if condition_path.exists():
        condition_rows = json.loads(condition_path.read_text(encoding="utf-8"))
        degraded_dates = {
            str(row["date"])
            for row in condition_rows
            if str(row.get("condition", "")).lower() == "degraded"
        }
    canonical["databento_dataset_degraded_date"] = (
        pd.to_datetime(canonical["auction_date"]).dt.strftime("%Y-%m-%d").isin(degraded_dates)
    )
    canonical.to_csv(output_dir / "event_level_final.csv", index=False)
    pd.concat(path_summaries, ignore_index=True).to_csv(output_dir / "event_path_summary_final.csv", index=False)
    subsamples = pd.DataFrame(subsample_rows)
    shifts = pd.DataFrame(shift_rows)
    placebos = pd.DataFrame(placebo_rows)
    controls = pd.DataFrame(control_rows)
    yearly = pd.DataFrame(yearly_rows)
    loyo = pd.DataFrame(loyo_rows)
    concentration = _concentration(canonical)
    subsamples.to_csv(output_dir / "temporal_stability.csv", index=False)
    shifts.to_csv(output_dir / "event_time_shifts.csv", index=False)
    placebos.to_csv(output_dir / "placebo_event_times.csv", index=False)
    controls.to_csv(output_dir / "matched_control_results.csv", index=False)
    yearly.to_csv(output_dir / "yearly_metrics_final.csv", index=False)
    loyo.to_csv(output_dir / "leave_one_year_out.csv", index=False)
    concentration.to_csv(output_dir / "effect_concentration.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for mapping, group in shifts.groupby("mapping"):
        ax.plot(group["shift_minutes"], group["combined_mean_bps"], marker="o", label=mapping)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(title="Canonical combined effect by event-time shift", xlabel="Shift from bid deadline (minutes)", ylabel="-pre + post (bps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "event_time_shift_landscape.png", dpi=160)
    plt.close(fig)

    roll_sensitivity_rows: list[dict[str, Any]] = []
    quality_sensitivity_rows: list[dict[str, Any]] = []
    for mapping, group in canonical.groupby("mapping"):
        for label, subset in (
            ("all", group),
            ("exclude_roll_transition_paths", group.loc[~group["roll_transition_in_path"]]),
        ):
            roll_sensitivity_rows.append(
                _stats_row(subset, mapping, label, "full", BOOTSTRAP_SEED + 800)
            )
        for label, subset in (
            ("all", group),
            (
                "exclude_databento_degraded_dates",
                group.loc[~group["databento_dataset_degraded_date"]],
            ),
        ):
            quality_sensitivity_rows.append(
                _stats_row(subset, mapping, label, "full", BOOTSTRAP_SEED + 900)
            )
    pd.DataFrame(roll_sensitivity_rows).to_csv(output_dir / "roll_transition_sensitivity.csv", index=False)
    pd.DataFrame(quality_sensitivity_rows).to_csv(output_dir / "data_quality_sensitivity.csv", index=False)

    result = {
        "mappings": [f"{tenor}->{instrument}" for tenor, instrument, _ in MAPPINGS],
        "event_counts": canonical.groupby("mapping").size().to_dict(),
        "canonical_full": subsamples.loc[subsamples["subsample"].eq("full")].to_dict("records"),
        "controls": controls.to_dict("records"),
        "placebos": placebos.to_dict("records"),
        "bootstrap_samples": bootstrap_samples,
        "result_timestamp_coverage": 0,
        "reference_used": "competitive_bid_deadline",
        "databento_degraded_dates_in_dataset": len(degraded_dates),
        "auction_mapping_rows_on_degraded_dates": int(
            canonical["databento_dataset_degraded_date"].sum()
        ),
    }
    _write_json(output_dir / "final_event_study_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research/treasury_auction_pressure",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    result = run(args.output_dir, args.bootstrap_samples)
    print(json.dumps(result["event_counts"], indent=2))


if __name__ == "__main__":
    main()
