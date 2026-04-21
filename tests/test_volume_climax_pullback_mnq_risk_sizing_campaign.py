from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from src.analytics.volume_climax_pullback_mnq_risk_sizing_campaign import run_campaign
from src.engine.execution_model import ExecutionModel
from src.engine.volume_climax_pullback_v2_backtester import run_volume_climax_pullback_v2_backtest
from src.engine.vwap_backtester import InstrumentDetails
from src.risk.position_sizing import FixedContractPositionSizing, RiskPercentPositionSizing, resolve_position_size
from src.strategy.volume_climax_pullback_v2 import VolumeClimaxPullbackV2Variant, build_volume_climax_pullback_v3_variants


def _base_variant(**overrides: object) -> VolumeClimaxPullbackV2Variant:
    payload: dict[str, object] = {
        "name": "risk_sizing_test_variant",
        "family": "dynamic_exit",
        "timeframe": "1h",
        "volume_quantile": 0.95,
        "volume_lookback": 50,
        "min_body_fraction": 0.5,
        "min_range_atr": 1.2,
        "trend_ema_window": None,
        "ema_slope_threshold": None,
        "atr_percentile_low": None,
        "atr_percentile_high": None,
        "compression_ratio_max": None,
        "entry_mode": "next_open",
        "pullback_fraction": None,
        "confirmation_window": None,
        "exit_mode": "fixed_rr",
        "rr_target": 1.0,
        "atr_target_multiple": None,
        "time_stop_bars": 3,
        "trailing_atr_multiple": 0.5,
        "session_overlay": "all_rth",
    }
    payload.update(overrides)
    return VolumeClimaxPullbackV2Variant(**payload)


def _instrument() -> InstrumentDetails:
    return InstrumentDetails(
        symbol="MNQ",
        asset_class="futures",
        tick_size=0.25,
        tick_value_usd=0.5,
        point_value_usd=2.0,
        commission_per_side_usd=1.25,
        slippage_ticks=1,
    )


def _single_trade_signal_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(["2024-01-02 09:30:00", "2024-01-02 10:30:00"])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": [timestamps[0].date()] * 2,
            "open": [100.0, 99.8],
            "high": [100.0, 100.0],
            "low": [100.0, 98.5],
            "close": [100.0, 99.0],
            "signal": [1, 0],
            "setup_reference_close": [99.5, pd.NA],
            "setup_reference_range": [2.0, pd.NA],
            "setup_stop_reference_long": [99.25, pd.NA],
            "setup_stop_reference_short": [pd.NA, pd.NA],
            "setup_reference_atr": [2.0, pd.NA],
            "setup_reference_vwap": [101.0, pd.NA],
            "setup_signal_time": [timestamps[0] - pd.Timedelta(hours=1), pd.NaT],
        }
    )


def _write_synthetic_minute_dataset(path: Path, sessions: int = 12) -> None:
    rows: list[dict[str, object]] = []
    for day_index, session_date in enumerate(pd.bdate_range("2024-01-02", periods=sessions)):
        session_open = pd.Timestamp(session_date.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        previous_close = 100.0 + day_index * 0.02
        for minute_index in range(390):
            timestamp = session_open + pd.Timedelta(minutes=minute_index)
            phase = minute_index % 90
            drift = 0.006 * phase if phase < 45 else 0.27 - 0.008 * (phase - 45)
            close = 100.0 + day_index * 0.02 + drift
            if minute_index in {59, 149, 239, 329}:
                close -= 0.30
            open_price = previous_close
            high = max(open_price, close) + 0.03
            low = min(open_price, close) - 0.03
            volume = 120.0 + (minute_index % 25)
            if minute_index in {59, 149, 239, 329}:
                volume = 6_000.0 + day_index * 10.0
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
            previous_close = close
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_resolve_position_size_nominal_case() -> None:
    decision = resolve_position_size(
        config=RiskPercentPositionSizing(initial_capital_usd=50_000.0, risk_pct=0.005, max_contracts=10, skip_trade_if_too_small=True),
        capital_before_trade_usd=50_000.0,
        entry_price=100.0,
        initial_stop_price=95.0,
        point_value_usd=2.0,
    )

    assert decision.contracts == 10
    assert decision.contracts_raw == pytest.approx(25.0)
    assert decision.risk_budget_usd == pytest.approx(250.0)
    assert decision.risk_per_contract_usd == pytest.approx(10.0)
    assert decision.actual_risk_usd == pytest.approx(100.0)


def test_resolve_position_size_handles_small_size_force_and_cap() -> None:
    skip_decision = resolve_position_size(
        config=RiskPercentPositionSizing(initial_capital_usd=50_000.0, risk_pct=0.0025, max_contracts=5, skip_trade_if_too_small=True),
        capital_before_trade_usd=50_000.0,
        entry_price=100.0,
        initial_stop_price=-100.0,
        point_value_usd=2.0,
    )
    force_one = resolve_position_size(
        config=RiskPercentPositionSizing(initial_capital_usd=50_000.0, risk_pct=0.0025, max_contracts=5, skip_trade_if_too_small=False),
        capital_before_trade_usd=50_000.0,
        entry_price=100.0,
        initial_stop_price=-100.0,
        point_value_usd=2.0,
    )
    capped = resolve_position_size(
        config=RiskPercentPositionSizing(initial_capital_usd=50_000.0, risk_pct=0.01, max_contracts=3, skip_trade_if_too_small=True),
        capital_before_trade_usd=50_000.0,
        entry_price=100.0,
        initial_stop_price=99.0,
        point_value_usd=2.0,
    )

    assert skip_decision.skipped is True
    assert skip_decision.contracts == 0
    assert skip_decision.skip_reason == "contracts_below_one"

    assert force_one.skipped is False
    assert force_one.contracts == 1
    assert force_one.contracts_raw < 1.0

    assert capped.contracts == 3
    assert capped.contracts_raw == pytest.approx(250.0)


def test_resolve_position_size_skips_non_positive_stop_distance() -> None:
    decision = resolve_position_size(
        config=RiskPercentPositionSizing(initial_capital_usd=50_000.0, risk_pct=0.005, max_contracts=5, skip_trade_if_too_small=True),
        capital_before_trade_usd=50_000.0,
        entry_price=100.0,
        initial_stop_price=100.0,
        point_value_usd=2.0,
    )

    assert decision.skipped is True
    assert decision.contracts == 0
    assert decision.skip_reason == "non_positive_risk_per_contract"


def test_backtester_multiplies_pnl_fees_and_slippage_by_contract_count() -> None:
    result = run_volume_climax_pullback_v2_backtest(
        signal_df=_single_trade_signal_frame(),
        variant=_base_variant(),
        execution_model=ExecutionModel(commission_per_side_usd=1.25, slippage_ticks=1, tick_size=0.25),
        instrument=_instrument(),
        position_sizing=RiskPercentPositionSizing(
            initial_capital_usd=1_000.0,
            risk_pct=0.005,
            max_contracts=10,
            skip_trade_if_too_small=True,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert int(trade["quantity"]) == 2
    assert float(trade["risk_budget_usd"]) == pytest.approx(5.0)
    assert float(trade["risk_per_contract_usd"]) == pytest.approx(2.0)
    assert float(trade["trade_risk_usd"]) == pytest.approx(4.0)
    assert float(trade["pnl_usd"]) == pytest.approx(-5.0)
    assert float(trade["fees"]) == pytest.approx(5.0)
    assert float(trade["net_pnl_usd"]) == pytest.approx(-10.0)


def test_fixed_1_contract_baseline_matches_default_behavior() -> None:
    signal_df = _single_trade_signal_frame()
    variant = _base_variant()
    execution_model = ExecutionModel(commission_per_side_usd=0.0, slippage_ticks=0, tick_size=0.25)
    instrument = InstrumentDetails(
        symbol="MNQ",
        asset_class="futures",
        tick_size=0.25,
        tick_value_usd=0.5,
        point_value_usd=2.0,
        commission_per_side_usd=0.0,
        slippage_ticks=0,
    )

    default_result = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
    )
    explicit_fixed = run_volume_climax_pullback_v2_backtest(
        signal_df=signal_df,
        variant=variant,
        execution_model=execution_model,
        instrument=instrument,
        position_sizing=FixedContractPositionSizing(fixed_contracts=1),
    )

    pdt.assert_frame_equal(
        default_result.trades.reset_index(drop=True),
        explicit_fixed.trades.reset_index(drop=True),
        check_dtype=False,
    )
    pdt.assert_frame_equal(
        default_result.sizing_decisions.reset_index(drop=True),
        explicit_fixed.sizing_decisions.reset_index(drop=True),
        check_dtype=False,
    )


def test_mnq_risk_sizing_campaign_smoke_outputs_requested_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "MNQ_c_0_1m_smoke.parquet"
    _write_synthetic_minute_dataset(dataset_path)
    base_variant_name = build_volume_climax_pullback_v3_variants("MNQ")[0].name

    output_dir = run_campaign(
        output_root=tmp_path / "exports",
        input_path=dataset_path,
        initial_capital_usd=50_000.0,
        risk_pcts=(0.0050,),
        max_contracts_grid=(3,),
        skip_flags=(True, False),
        base_variant_name=base_variant_name,
    )

    assert (output_dir / "summary_by_variant.csv").exists()
    assert (output_dir / "trades_by_variant.csv").exists()
    assert (output_dir / "daily_equity_by_variant.csv").exists()
    assert (output_dir / "prop_constraints_summary.csv").exists()
    assert (output_dir / "final_report.md").exists()

    summary = pd.read_csv(output_dir / "summary_by_variant.csv")
    assert len(summary) == 3
    assert set(summary["campaign_variant_name"]) == {
        "fixed_1_contract",
        "risk_pct_0p0050__max_contracts_3__skip_trade_if_too_small_true",
        "risk_pct_0p0050__max_contracts_3__skip_trade_if_too_small_false",
    }
    prop = pd.read_csv(output_dir / "prop_constraints_summary.csv")
    assert set(prop["scope"]) == {"full", "oos"}
