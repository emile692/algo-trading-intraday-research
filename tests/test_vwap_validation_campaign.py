import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.vwap_validation import (
    ChallengeScenario,
    StressScenario,
    _apply_cost_stress_overlay,
    _apply_trade_controls_overlay,
    _drawdown_episode_table,
    _heatmap_topology_readout,
    _parse_phases,
    _simulate_challenge_path,
    _split_sessions,
    generate_validation_notebook,
)
from src.engine.vwap_backtester import build_execution_model_for_profile


def _sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": [1, 2, 3, 4],
            "session_date": ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"],
            "entry_time": pd.to_datetime(
                [
                    "2024-01-02 09:35:00",
                    "2024-01-02 10:00:00",
                    "2024-01-02 11:00:00",
                    "2024-01-03 09:40:00",
                ]
            ),
            "exit_time": pd.to_datetime(
                [
                    "2024-01-02 09:50:00",
                    "2024-01-02 10:10:00",
                    "2024-01-02 11:20:00",
                    "2024-01-03 10:00:00",
                ]
            ),
            "entry_price": [100.0, 101.0, 102.0, 103.0],
            "exit_price": [99.0, 100.0, 103.0, 104.0],
            "stop_price": [99.0, 100.0, 101.0, 102.0],
            "quantity": [1, 1, 1, 1],
            "direction": ["long", "long", "long", "long"],
            "pnl_usd": [-50.0, -60.0, 120.0, 80.0],
            "gross_pnl_usd": [-50.0, -60.0, 120.0, 80.0],
            "fees": [2.5, 2.5, 2.5, 2.5],
            "net_pnl_usd": [-52.5, -62.5, 117.5, 77.5],
            "trade_risk_usd": [100.0, 100.0, 100.0, 100.0],
            "holding_minutes": [15.0, 10.0, 20.0, 20.0],
            "exit_reason": ["stop", "stop", "session_close", "session_close"],
        }
    )


def test_apply_trade_controls_overlay_respects_daily_limits() -> None:
    trades = _sample_trades()

    filtered = _apply_trade_controls_overlay(
        trades,
        max_losses_per_day=1,
        daily_stop_threshold_usd=55.0,
    )

    assert filtered["session_date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert filtered["trade_id"].tolist() == [1, 2]


def test_apply_cost_stress_overlay_adds_extra_costs() -> None:
    trades = _sample_trades().iloc[[0]].copy()
    trades["quantity"] = 2
    execution_model, instrument = build_execution_model_for_profile("MNQ", "repo_realistic")

    stressed = _apply_cost_stress_overlay(
        trades,
        scenario=StressScenario(
            name="stress",
            commission_multiplier=1.50,
            slippage_multiplier=2.0,
        ),
        instrument=instrument,
        execution_model=execution_model,
        session_start="09:30",
    )

    assert stressed.iloc[0]["net_pnl_usd"] == pytest.approx(trades.iloc[0]["net_pnl_usd"] - 4.5)
    assert stressed.iloc[0]["fees"] == pytest.approx(trades.iloc[0]["fees"] + 2.5)


def test_simulate_challenge_path_respects_stop_after_losses() -> None:
    trades = _sample_trades()
    scenario = ChallengeScenario(
        name="test",
        label="test",
        risk_per_trade_pct=10.0,
        max_contracts=1,
        stop_after_losses_in_day=2,
        daily_loss_limit_usd=500.0,
        trailing_drawdown_limit_usd=500.0,
        profit_target_usd=1000.0,
        horizon_days=5,
        deleverage_after_red_days=99,
        deleverage_factor=0.5,
    )

    _, daily_path, summary = _simulate_challenge_path(
        trades,
        scenario=scenario,
        account_size_usd=1_000.0,
    )

    assert int(daily_path.iloc[0]["daily_trade_count"]) == 2
    assert bool(summary["success"]) is False


def test_heatmap_topology_readout_detects_stable_plateau() -> None:
    rows = []
    for x in ["0.00", "0.01", "0.02"]:
        for y in ["0.20", "0.30", "0.40"]:
            rows.append(
                {
                    "x_value": x,
                    "y_value": y,
                    "x_value_sort": float(x),
                    "y_value_sort": float(y),
                    "oos_profit_factor": 1.1,
                    "oos_net_pnl": 10.0,
                    "oos_sharpe_ratio": 0.5,
                }
            )
    df = pd.DataFrame(rows)

    readout = _heatmap_topology_readout(df, x_col="x_value", y_col="y_value", ref_x="0.01", ref_y="0.30")

    assert readout["verdict"] == "stable localement"


def test_parse_phases_validates_and_split_sessions_is_chronological() -> None:
    assert _parse_phases("stress,local") == ("stress", "local")
    with pytest.raises(ValueError):
        _parse_phases("unknown")

    is_sessions, oos_sessions = _split_sessions(list(pd.date_range("2024-01-01", periods=10).date), 0.7)
    assert max(is_sessions) < min(oos_sessions)


def test_generate_validation_notebook_emits_valid_code_cells_and_absolute_output_dir(tmp_path: Path) -> None:
    notebook_path = tmp_path / "validation.ipynb"
    output_dir = tmp_path / "exports" / "run_001"

    generate_validation_notebook(notebook_path=notebook_path, output_dir=output_dir)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all("outputs" in cell and cell["outputs"] == [] for cell in code_cells)
    assert all("execution_count" in cell and cell["execution_count"] is None for cell in code_cells)
    assert "OUTPUT_DIR = Path(r\"" + str(output_dir.resolve()) + "\")" in "".join(code_cells[1]["source"])


def test_drawdown_episode_table_empty_still_exposes_expected_columns() -> None:
    table = _drawdown_episode_table(pd.DataFrame(), initial_capital=50_000.0)

    assert table.empty
    assert "duration_sessions" in table.columns
