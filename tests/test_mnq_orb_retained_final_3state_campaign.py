from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import src.analytics.mnq_orb_retained_final_3state_campaign as campaign
from src.analytics.audit_mnq_orb_retained_vs_3state import RetainedConfig
from src.analytics.mnq_orb_regime_filter_sizing_campaign import BucketCalibration


def test_run_retained_final_3state_campaign_wires_retained_baseline(tmp_path: Path, monkeypatch) -> None:
    sessions = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]).date
    trades = pd.DataFrame(
        [
            {
                "trade_id": idx + 1,
                "session_date": session,
                "direction": "long",
                "quantity": 1,
                "entry_time": pd.Timestamp(session) + pd.Timedelta(hours=14, minutes=35),
                "entry_price": 20000.0 + idx,
                "stop_price": 19950.0 + idx,
                "target_price": 20100.0 + idx,
                "exit_time": pd.Timestamp(session) + pd.Timedelta(hours=15, minutes=10),
                "exit_price": 20025.0 + idx,
                "exit_reason": "time_exit",
                "account_size_usd": 50000.0,
                "risk_per_trade_pct": 0.5,
                "risk_budget_usd": 250.0,
                "risk_per_contract_usd": 100.0,
                "actual_risk_usd": 100.0,
                "trade_risk_usd": 100.0,
                "notional_usd": 40000.0,
                "leverage_used": 0.8,
                "pnl_points": 12.5,
                "pnl_ticks": 50.0 + idx,
                "pnl_usd": 25.0 + idx,
                "fees": 2.5,
                "net_pnl_usd": 22.5 + idx,
            }
            for idx, session in enumerate(sessions)
        ]
    )

    fake_retained = {
        "config": RetainedConfig(),
        "trades": trades,
        "all_sessions": list(sessions),
        "is_sessions": list(sessions[:2]),
        "oos_sessions": list(sessions[2:]),
    }

    regime_df = pd.DataFrame(
        [
            {
                "session_date": session,
                "phase": "is" if idx < 2 else "oos",
                "realized_vol_ratio_15_60": value,
                "atr_ratio_10_30": value,
                "weekday_name": "thursday",
                "trade_id": idx + 1,
                "entry_time": pd.Timestamp(session) + pd.Timedelta(hours=14, minutes=35),
                "exit_time": pd.Timestamp(session) + pd.Timedelta(hours=15, minutes=10),
                "direction": "long",
                "quantity": 1,
                "net_pnl_usd": 22.5 + idx,
                "trade_risk_usd": 100.0,
                "fees": 2.5,
                "exit_reason": "time_exit",
                "opening_range_width_pts": 10.0,
                "gap_abs_atr20": 0.5,
                "signal_vwap_distance_atr20": 0.2,
                "signal_extension_over_or": 0.3,
                "nominal_selected": True,
            }
            for idx, (session, value) in enumerate(zip(sessions, [0.8, 0.9, 1.1, 1.2]))
        ]
    )

    conditional_df = pd.DataFrame(
        [
            {
                "feature_name": "realized_vol_ratio_15_60",
                "bucket_label": "low",
                "bucket_position": 1,
                "lower_bound": 0.0,
                "upper_bound": 0.85,
                "is_composite_score": -1.0,
                "is_expectancy": -10.0,
                "is_profit_factor": 0.8,
                "is_n_obs": 2,
            },
            {
                "feature_name": "realized_vol_ratio_15_60",
                "bucket_label": "mid",
                "bucket_position": 2,
                "lower_bound": 0.85,
                "upper_bound": 1.05,
                "is_composite_score": 0.0,
                "is_expectancy": 0.0,
                "is_profit_factor": 1.0,
                "is_n_obs": 2,
            },
            {
                "feature_name": "realized_vol_ratio_15_60",
                "bucket_label": "high",
                "bucket_position": 3,
                "lower_bound": 1.05,
                "upper_bound": 2.0,
                "is_composite_score": 1.0,
                "is_expectancy": 10.0,
                "is_profit_factor": 1.2,
                "is_n_obs": 2,
            },
        ]
    )
    feature_score_df = pd.DataFrame(
        [
            {
                "feature_name": "realized_vol_ratio_15_60",
                "family": "volatility",
                "bucket_kind": "quantile",
                "valid_for_overlay": True,
                "feature_selection_score": 1.0,
                "min_bucket_obs_is": 2,
                "worst_bucket_is": "low",
            }
        ]
    )
    assignments = {
        "realized_vol_ratio_15_60": pd.Series(["low", "mid", "mid", "high"], dtype="string"),
    }
    calibrations = {
        "realized_vol_ratio_15_60": BucketCalibration(
            feature_name="realized_vol_ratio_15_60",
            bucket_kind="quantile",
            labels=("low", "mid", "high"),
            bins=(0.0, 0.85, 1.05, 2.0),
        )
    }

    monkeypatch.setattr(campaign, "_latest_mnq_dataset", lambda: tmp_path / "fake.parquet")
    monkeypatch.setattr(campaign, "_rebuild_retained_final", lambda dataset_path: fake_retained)
    monkeypatch.setattr(campaign, "_build_retained_regime_dataset", lambda retained, cfg: regime_df.copy())
    monkeypatch.setattr(
        campaign,
        "build_conditional_bucket_analysis",
        lambda regime_df, nominal_trades, initial_capital, feature_specs, min_bucket_obs_is: (
            conditional_df.copy(),
            feature_score_df.copy(),
            assignments.copy(),
            calibrations.copy(),
        ),
    )

    artifacts = campaign.run_retained_final_3state_campaign(
        campaign.RetainedFinal3StateCampaignSpec(
            output_root=tmp_path / "exports",
            min_bucket_obs_is=2,
        )
    )

    summary = pd.read_csv(artifacts["summary"])
    metadata = json.loads(Path(artifacts["metadata"]).read_text(encoding="utf-8"))

    assert Path(artifacts["summary"]).exists()
    assert Path(artifacts["markdown"]).exists()
    assert {"nominal", "sizing_3state_realized_vol_ratio_15_60"}.issubset(set(summary["variant_name"]))
    assert metadata["campaign_type"] == "retained_final_3state"
    assert metadata["retained_config"]["or_minutes"] == 15
    assert metadata["retained_config"]["direction"] == "long"
    assert metadata["retained_config"]["risk_per_trade_pct"] == 0.5
