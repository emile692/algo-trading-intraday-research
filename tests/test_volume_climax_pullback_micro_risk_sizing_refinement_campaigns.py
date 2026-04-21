from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.analytics import volume_climax_pullback_mnq_risk_sizing_campaign as mnq_core
from src.analytics import volume_climax_pullback_mnq_risk_sizing_refinement_campaign as mnq_refinement
from src.analytics import volume_climax_pullback_m2k_risk_sizing_refinement_campaign as m2k_refinement
from src.analytics import volume_climax_pullback_mes_risk_sizing_refinement_campaign as mes_refinement
from src.analytics import volume_climax_pullback_mgc_risk_sizing_refinement_campaign as mgc_refinement
from src.analytics.volume_climax_pullback_symbol_risk_sizing_refinement_campaign import (
    SymbolRefinementConfig,
    temporary_refinement_defaults,
)
from src.strategy.volume_climax_pullback_v2 import build_volume_climax_pullback_v3_variants


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


def test_temporary_refinement_defaults_restores_globals() -> None:
    original_core_symbol = mnq_core.DEFAULT_SYMBOL
    original_ref_symbol = mnq_refinement.DEFAULT_SYMBOL
    original_output_prefix = mnq_refinement.DEFAULT_OUTPUT_PREFIX
    original_winner_name = mnq_refinement.BEST_PREVIOUS_WINNER_NAME

    config = SymbolRefinementConfig(
        symbol="MES",
        output_prefix="volume_climax_pullback_mes_risk_sizing_refinement_",
        risk_pcts=(0.0015, 0.0020),
        max_contracts_grid=(2, 3),
        best_previous_winner_name="risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true",
        best_previous_winner_risk_pct=0.0025,
        best_previous_winner_max_contracts=3,
    )

    with temporary_refinement_defaults(config):
        assert mnq_core.DEFAULT_SYMBOL == "MES"
        assert mnq_refinement.DEFAULT_SYMBOL == "MES"
        assert mnq_refinement.DEFAULT_OUTPUT_PREFIX == "volume_climax_pullback_mes_risk_sizing_refinement_"
        assert mnq_refinement.BEST_PREVIOUS_WINNER_NAME == config.best_previous_winner_name

    assert mnq_core.DEFAULT_SYMBOL == original_core_symbol
    assert mnq_refinement.DEFAULT_SYMBOL == original_ref_symbol
    assert mnq_refinement.DEFAULT_OUTPUT_PREFIX == original_output_prefix
    assert mnq_refinement.BEST_PREVIOUS_WINNER_NAME == original_winner_name


@pytest.mark.parametrize(
    ("symbol", "module"),
    [
        ("MES", mes_refinement),
        ("M2K", m2k_refinement),
        ("MGC", mgc_refinement),
    ],
)
def test_micro_refinement_campaign_smoke_outputs_requested_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    module,
) -> None:
    dataset_path = tmp_path / f"{symbol}_c_0_1m_smoke.parquet"
    _write_synthetic_minute_dataset(dataset_path)
    base_variant_name = build_volume_climax_pullback_v3_variants(symbol)[0].name

    compact_config = replace(
        module.CONFIG,
        risk_pcts=(module.CONFIG.risk_pcts[0], module.CONFIG.risk_pcts[min(1, len(module.CONFIG.risk_pcts) - 1)]),
        max_contracts_grid=(2, 3),
    )
    monkeypatch.setattr(module, "CONFIG", compact_config)

    output_dir = module.run_campaign(
        output_root=tmp_path / f"{symbol.lower()}_refinement_exports",
        input_path=dataset_path,
        initial_capital_usd=50_000.0,
        base_variant_name=base_variant_name,
    )

    expected_files = {
        "summary_by_variant.csv",
        "summary_oos_only.csv",
        "trades_by_variant.csv",
        "daily_equity_by_variant.csv",
        "prop_constraints_summary.csv",
        "heatmap_metrics.csv",
        "robustness_zone_summary.csv",
        "final_report.md",
        "heatmap_oos_net_pnl.png",
        "heatmap_oos_sharpe.png",
        "heatmap_oos_maxdd_usd.png",
        "heatmap_oos_prop_score.png",
        "robustness_cluster_map.png",
        "final_verdict.json",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    summary = pd.read_csv(output_dir / "summary_by_variant.csv")
    expected_count = 1 + len(compact_config.risk_pcts) * len(compact_config.max_contracts_grid) + 1
    assert len(summary) == expected_count
    assert set(summary["variant_role"]) == {"baseline", "grid", "best_previous_winner"}
    assert summary["alpha_variant_name"].nunique() == 1

    heatmap = pd.read_csv(output_dir / "heatmap_metrics.csv")
    assert len(heatmap) == len(compact_config.risk_pcts) * len(compact_config.max_contracts_grid)
    assert set(heatmap["max_contracts"]) == {2, 3}
