"""Add standalone MR candidates one at a time to the frozen nine-sleeve Topstep study."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from research.mbt_nine_sleeve_topstep_20261004.run_topstep import (
    ALL,
    END,
    START,
    build_mappings,
    load_mbt,
    load_mbt_minute,
    load_six,
    portfolio_daily,
    portfolio_path,
    six_intraday_paths,
    summarize,
    topstep_campaign,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SIX_DIR = HERE.parent / "portfolio_integration_phase1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_intraday(
    trades: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    point_value: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Build normal/adverse within-day paths from stored trade MAE and timestamps.

    The candidate engine is five-minute, so the adverse path places the observed
    trade MAE at entry and carries realized PnL after exit. This is deliberately
    conservative and leaves the existing nine-sleeve minute path untouched.
    """
    dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)
    normal = pd.Series(0.0, index=timeline); adverse = pd.Series(0.0, index=timeline)
    daily = pd.Series(0.0, index=pd.DatetimeIndex(sorted(dates.unique()))); traded = pd.Series(False, index=daily.index)
    if trades.empty:
        return normal, adverse, daily, traded
    t = trades.copy(); t["session_date"] = pd.to_datetime(t.session_date)
    t["entry_time"] = pd.to_datetime(t.entry_time, utc=True); t["exit_time"] = pd.to_datetime(t.exit_time, utc=True)
    for day, group in t.groupby("session_date"):
        if day < START or day > END or day not in daily.index:
            continue
        day_idx = timeline[dates.eq(day.normalize()).to_numpy()]
        if not len(day_idx):
            continue
        base = np.zeros(len(day_idx)); bad = np.zeros(len(day_idx)); realized = 0.0
        for row in group.sort_values("entry_time").itertuples(index=False):
            start_i = int(day_idx.searchsorted(pd.Timestamp(row.entry_time), side="left")); end_i = int(day_idx.searchsorted(pd.Timestamp(row.exit_time), side="left"))
            if start_i >= len(day_idx):
                continue
            # Linear normal mark is less influential than the conservative MAE
            # mark; exact candidate EOD is forced to its audited trade net PnL.
            end_i = min(end_i, len(day_idx) - 1)
            if end_i > start_i:
                path = np.linspace(0.0, float(row.net_pnl), end_i - start_i, endpoint=False)
                base[start_i:end_i] = realized + path
                bad[start_i:end_i] = realized + np.minimum(path, -float(row.mae_usd) - float(row.nominal_cost) / 2)
            realized += float(row.net_pnl)
            base[end_i:] = realized; bad[end_i:] = realized
        normal.loc[day_idx] = base; adverse.loc[day_idx] = bad
        daily.loc[day] = realized; traded.loc[day] = True
    return normal, adverse, daily, traded


def build_baseline() -> tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series, pd.Series]:
    six_pnl, six_counts, six_trades = load_six(); calendar = six_pnl.index
    mbt_nominal, _, mbt_counts, _ = load_mbt(calendar)
    nominal = six_pnl.join(mbt_nominal[[c for c in ALL if c.startswith("mbt_")]], how="left").fillna(0.0)
    counts = six_counts.join(mbt_counts[[c for c in ALL if c.startswith("mbt_")]], how="left").fillna(0)
    timeline_source = pd.read_csv(SIX_DIR / "common_timestamp_portfolio.csv", usecols=["timestamp_utc"], parse_dates=["timestamp_utc"])
    timeline = pd.DatetimeIndex(timeline_source.timestamp_utc).tz_convert("UTC")
    timeline_dates = pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline)
    six_path, six_adverse, _ = six_intraday_paths(timeline, six_trades)
    _, mbt_path, mbt_adverse = load_mbt_minute(timeline, "nominal")
    normal_path = six_path.join(mbt_path, how="left").fillna(0.0)
    adverse_path = six_adverse.join(mbt_adverse, how="left").fillna(0.0)
    mapping = build_mappings(nominal)["equal_contract"]["nine"]
    daily = portfolio_daily(nominal, mapping)
    intraday = portfolio_path(adverse_path, mapping).groupby(timeline_dates, sort=False).min()
    intraday.index = pd.to_datetime(intraday.index)
    traded = counts[ALL].sum(axis=1).gt(0)
    bootstrap_traded = six_counts.gt(0).any(axis=1)
    return timeline, daily, intraday.reindex(calendar).fillna(0.0), traded, bootstrap_traded


def run() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS / "mechanism_by_asset_summary.csv")
    chosen = summary[summary.verdict.eq("CANDIDATE_FOR_PROSPECTIVE_FREEZE")].copy()
    timeline, base_daily, base_min, base_traded, bootstrap_traded = build_baseline()
    baseline_variant = {"daily": base_daily.to_numpy(float), "intraday_min": base_min.to_numpy(float), "traded": base_traded.to_numpy(bool)}
    # The frozen helper keys bootstrap eligibility from this historical name.
    bootstrap_variant = {**baseline_variant, "traded": bootstrap_traded.to_numpy(bool)}
    variants = {"equal_contract__nominal__six": bootstrap_variant, "nine_sleeves": baseline_variant}
    calendar = base_daily.index
    for row in chosen.itertuples(index=False):
        trades = pd.read_csv(RESULTS / "candidate_trades" / f"{row.candidate_id}.csv")
        _, candidate_bad, candidate_daily, candidate_traded = candidate_intraday(trades, timeline, point_value=1.0)
        candidate_min = candidate_bad.groupby(pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date), index=timeline), sort=False).min()
        candidate_min.index = pd.to_datetime(candidate_min.index)
        added_daily = base_daily + candidate_daily.reindex(calendar).fillna(0.0)
        added_min = base_min + candidate_min.reindex(calendar).fillna(0.0)
        variants[f"nine_plus__{row.candidate_id}"] = {"daily": added_daily.to_numpy(float), "intraday_min": added_min.to_numpy(float), "traded": (base_traded | candidate_traded.reindex(calendar).fillna(False)).to_numpy(bool)}
    _, topstep = topstep_campaign(variants)
    pooled = topstep[(topstep.method == "moving_block_bootstrap") & (topstep.block == "pooled")].copy()
    baseline = pooled[pooled.scenario.eq("nine_sleeves")].set_index("horizon")
    rows = []
    for row in pooled.itertuples(index=False):
        base = baseline.loc[row.horizon]
        rows.append({
            "scenario": row.scenario, "horizon": row.horizon, "paths": row.paths,
            "pass_probability": row.pass_probability, "mll_probability": row.mll_probability, "expiry_probability": row.expiry_probability,
            "median_days_to_pass": row.median_days_to_pass, "median_max_drawdown_usd": row.median_max_drawdown_usd, "p05_max_drawdown_usd": row.p05_max_drawdown_usd,
            "delta_pass": row.pass_probability - base.pass_probability, "delta_mll": row.mll_probability - base.mll_probability,
            "delta_expiry": row.expiry_probability - base.expiry_probability,
        })
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "topstep_incremental_results.csv", index=False)
    metadata = {"framework": "research.mbt_nine_sleeve_topstep_20261004.run_topstep", "seed": 20261004, "blocks": [3,5,10,20], "paths_per_block": 2000, "candidate_intraday_note": "five-minute candidate MAE added conservatively to unchanged minute nine-sleeve adverse path"}
    (RESULTS / "topstep_incremental_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = (RESULTS / "MEAN_REVERSION_MASTER_REPORT.md").read_text(encoding="utf-8")
    table = out[out.horizon.eq(60)].to_csv(index=False)
    replacement = "## I. Topstep incremental results\n\nSimulation moving-block identique au study nine-sleeve (8 000 chemins, blocs 3/5/10/20, seed 20261004). Le chemin adverse des neuf sleeves reste minute; le candidat cinq minutes est ajouté conservativement via son MAE observé.\n\n```csv\n" + table + "```"
    start = report.index("## I. Topstep incremental results")
    end = report.index("## J. Recommended next actions")
    report = report[:start] + replacement + "\n\n" + report[end:]
    (RESULTS / "MEAN_REVERSION_MASTER_REPORT.md").write_text(report, encoding="utf-8")
    run_metadata_path = RESULTS / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    run_metadata["topstep_incremental"] = metadata
    run_metadata["files"].update({
        "MEAN_REVERSION_MASTER_REPORT.md": _sha256(RESULTS / "MEAN_REVERSION_MASTER_REPORT.md"),
        "topstep_incremental_results.csv": _sha256(RESULTS / "topstep_incremental_results.csv"),
        "topstep_incremental_metadata.json": _sha256(RESULTS / "topstep_incremental_metadata.json"),
    })
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run()
