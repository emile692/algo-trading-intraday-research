import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research" / "locked_eight_sleeve_portfolio_20261103"
NINE_LOCK = ROOT / "research" / "locked_nine_sleeve_portfolio_20261101"
SHADOW = "mnq_atr_trend_flip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config() -> dict:
    return json.loads((LOCK / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))


def test_eight_sleeve_equal_contract_reference_is_locked() -> None:
    config = _config()
    assert config["status"] == "LOCKED_ACTIVE_REFERENCE"
    assert config["portfolio_id"] == "eight_sleeve_equal_contract_reference_20261103"
    assert config["supersedes_active_reference"] == "nine_sleeve_equal_contract_reference_20261101"
    assert len(config["mapping"]) == 8 and set(config["mapping"].values()) == {1}
    assert SHADOW not in config["mapping"]
    assert config["shadow_sleeves"][SHADOW]["allocation"] == 0
    assert config["comparison_with_superseded"]["nine_20261101"]["horizon_60d"]["pass_probability"] == 0.3895
    assert config["governance"]["no_retuning"]


def test_locked_outputs_match_hash_manifest_and_daily_sum() -> None:
    manifest = json.loads((LOCK / "lock_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        assert _sha256(ROOT / relative) == expected
    daily = pd.read_csv(LOCK / "daily_pnl.csv")
    sleeve_columns = [column for column in daily if column not in {"session_date", "portfolio_pnl_usd"}]
    assert sorted(sleeve_columns) == sorted(_config()["mapping"])
    assert np.allclose(daily[sleeve_columns].sum(axis=1), daily["portfolio_pnl_usd"], atol=1e-7)


def test_remaining_sleeves_are_unchanged_from_nine_sleeve_lock() -> None:
    nine = pd.read_csv(NINE_LOCK / "daily_pnl.csv")
    eight = pd.read_csv(LOCK / "daily_pnl.csv")
    nine_sources = json.loads((NINE_LOCK / "locked_nine_sleeve_config.json").read_text(encoding="utf-8"))["sleeve_sources"]
    for sleeve, source in _config()["sleeve_sources"].items():
        assert source == nine_sources[sleeve]
        assert _sha256(ROOT / source["daily_source"]) == source["daily_source_sha256"]
        assert np.allclose(eight[sleeve], nine[sleeve], atol=1e-9)


def test_stop_rule_thresholds_cover_all_nine_sleeves() -> None:
    thresholds = pd.read_csv(LOCK / "sleeve_stop_rule_thresholds.csv").set_index("sleeve")
    assert len(thresholds) == 9
    assert (thresholds["threshold_usd"] < 0).all()
    assert thresholds.loc[SHADOW, "status"] == "SHADOW_FORWARD_ONLY"


def test_registry_records_this_lock_as_superseded_reference() -> None:
    # Superseded on 2026-11-15 by eight_sleeve_equal_contract_reference_20261115 (see SUPERSEDED.md);
    # the active-pointer test moved to tests/test_locked_eight_sleeve_portfolio_20261115.py.
    registry = json.loads((ROOT / "strategy_registry" / "forward_monitoring" / "registry.json").read_text(encoding="utf-8"))
    superseded = registry["active_reference"]["superseded_reference"]
    assert superseded["path"] == "research/locked_eight_sleeve_portfolio_20261103/locked_eight_sleeve_config.json"
    assert _sha256(ROOT / superseded["path"]) == superseded["sha256"]


def test_active_reference_pointer_targets_eight_sleeves() -> None:
    metadata = json.loads((ROOT / "docs" / "cleanup_manifest_metadata_2026.json").read_text(encoding="utf-8"))
    assert metadata["active_reference"] == "docs/ACTIVE_EIGHT_SLEEVES.md"
    assert metadata["previous_active_reference"] == "docs/ACTIVE_NINE_SLEEVES.md"
