from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics import volume_climax_pullback_mnq_risk_sizing_campaign as mnq_core
from src.analytics.volume_climax_pullback_m2k_risk_sizing_campaign import run_campaign as run_m2k_campaign
from src.analytics.volume_climax_pullback_mgc_risk_sizing_campaign import run_campaign as run_mgc_campaign
from src.analytics.volume_climax_pullback_mes_risk_sizing_campaign import run_campaign as run_mes_campaign
from src.analytics.volume_climax_pullback_symbol_risk_sizing_campaign import temporary_symbol_defaults
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


def test_temporary_symbol_defaults_restores_core_globals() -> None:
    original_symbol = mnq_core.DEFAULT_SYMBOL
    original_prefix = mnq_core.DEFAULT_OUTPUT_PREFIX

    with temporary_symbol_defaults(symbol="MES", output_prefix="volume_climax_pullback_mes_risk_sizing_"):
        assert mnq_core.DEFAULT_SYMBOL == "MES"
        assert mnq_core.DEFAULT_OUTPUT_PREFIX == "volume_climax_pullback_mes_risk_sizing_"

    assert mnq_core.DEFAULT_SYMBOL == original_symbol
    assert mnq_core.DEFAULT_OUTPUT_PREFIX == original_prefix


@pytest.mark.parametrize(
    ("symbol", "runner"),
    [
        ("MES", run_mes_campaign),
        ("M2K", run_m2k_campaign),
        ("MGC", run_mgc_campaign),
    ],
)
def test_micro_symbol_risk_sizing_campaign_smoke_outputs_requested_files(
    tmp_path: Path,
    symbol: str,
    runner,
) -> None:
    dataset_path = tmp_path / f"{symbol}_c_0_1m_smoke.parquet"
    _write_synthetic_minute_dataset(dataset_path)
    base_variant_name = build_volume_climax_pullback_v3_variants(symbol)[0].name

    output_dir = runner(
        output_root=tmp_path / f"{symbol.lower()}_exports",
        input_path=dataset_path,
        initial_capital_usd=50_000.0,
        risk_pcts=(0.0050,),
        max_contracts_grid=(3,),
        skip_flags=(True, False),
        base_variant_name=base_variant_name,
    )

    for name in [
        "summary_by_variant.csv",
        "trades_by_variant.csv",
        "daily_equity_by_variant.csv",
        "prop_constraints_summary.csv",
        "final_report.md",
        "run_metadata.json",
    ]:
        assert (output_dir / name).exists(), f"Missing {name} for {symbol}"

    summary = pd.read_csv(output_dir / "summary_by_variant.csv")
    assert len(summary) == 3
    assert set(summary["campaign_variant_name"]) == {
        "fixed_1_contract",
        "risk_pct_0p0050__max_contracts_3__skip_trade_if_too_small_true",
        "risk_pct_0p0050__max_contracts_3__skip_trade_if_too_small_false",
    }

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["symbol"] == symbol
    assert metadata["variant_count"] == 3

    report_text = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert f"Volume Climax Pullback {symbol} Risk Sizing" in report_text
