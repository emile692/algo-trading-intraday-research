from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.mean_reversion_campaign import run_mean_reversion_campaign
from src.config.mean_reversion_campaign import MeanReversionCampaignSpec, TimeframeDefinition
from src.config.vwap_campaign import build_default_prop_constraints


def _synthetic_1m_dataset(path: Path) -> Path:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for day_offset, day in enumerate(pd.date_range("2024-01-02", periods=4, freq="D")):
        base = 100.0 + day_offset
        timestamps = pd.date_range(day.strftime("%Y-%m-%d 09:30:00"), periods=120, freq="1min")
        for i, ts in enumerate(timestamps):
            close = base + np.sin(i / 8.0) * 1.2 + (0.9 if i % 25 == 0 else 0.0) - (0.9 if i % 40 == 0 else 0.0)
            open_price = close - 0.1
            high = close + 0.2
            low = close - 0.2
            rows.append(
                {
                    "timestamp": ts,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 100 + (i % 10) * 5,
                }
            )
    frame = pd.DataFrame(rows).set_index("timestamp")
    frame.to_parquet(path)
    return path


def test_mean_reversion_campaign_smoke_run_generates_exports(tmp_path: Path) -> None:
    dataset_path = _synthetic_1m_dataset(tmp_path / "MNQ_c_0_1m_synth.parquet")
    output_dir = tmp_path / "exports"
    notebook_path = tmp_path / "mean_reversion.ipynb"
    spec = MeanReversionCampaignSpec(
        datasets_by_symbol={"MNQ": dataset_path},
        timeframes=(TimeframeDefinition(label="5m", bar_minutes=5, resample_rule="5min"),),
        prop_constraints=build_default_prop_constraints(),
        include_orb_baseline=False,
        max_validation_survivors=2,
        max_portfolio_candidates=2,
        output_root=output_dir,
    )

    artifacts = run_mean_reversion_campaign(
        spec=spec,
        output_dir=output_dir,
        phase="notebook",
        notebook_path=notebook_path,
    )

    assert (output_dir / "screening" / "screening_results.csv").exists()
    assert (output_dir / "validation" / "survivor_validation_summary.csv").exists()
    assert (output_dir / "portfolio" / "portfolio_summary.csv").exists()
    assert (output_dir / "portfolio" / "portfolio_summary.md").exists()
    assert notebook_path.exists()
    assert "notebook" in artifacts


def test_mean_reversion_campaign_with_orb_reference_handles_empty_portfolio(tmp_path: Path) -> None:
    dataset_path = _synthetic_1m_dataset(tmp_path / "MNQ_c_0_1m_synth.parquet")
    output_dir = tmp_path / "exports_orb"
    spec = MeanReversionCampaignSpec(
        datasets_by_symbol={"MNQ": dataset_path},
        timeframes=(TimeframeDefinition(label="5m", bar_minutes=5, resample_rule="5min"),),
        prop_constraints=build_default_prop_constraints(),
        include_orb_baseline=True,
        max_validation_survivors=2,
        max_portfolio_candidates=2,
        output_root=output_dir,
    )

    artifacts = run_mean_reversion_campaign(
        spec=spec,
        output_dir=output_dir,
        phase="portfolio",
    )

    assert (output_dir / "portfolio" / "portfolio_selection.csv").exists()
    assert (output_dir / "portfolio" / "orb_reference_summary.csv").exists()
    assert (output_dir / "portfolio" / "portfolio_summary.csv").exists()
    assert (output_dir / "portfolio" / "portfolio_summary.md").exists()
    assert "portfolio_summary_md" in artifacts
