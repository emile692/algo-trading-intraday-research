"""Lock eight_sleeve_equal_contract_reference_20261115: S01 = 18-model ATR ensemble, union calendar."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research" / "locked_eight_sleeve_portfolio_20261115"
OLD = ROOT / "research" / "locked_eight_sleeve_portfolio_20261103"
RULE = ROOT / "strategy_registry" / "execution" / "netting_rule_20261115_r2.json"
OLD_RULE = ROOT / "strategy_registry" / "execution" / "netting_rule_20261115.json"
S01 = "mnq_orb_vwap_atr"
SHADOW = "mnq_atr_trend_flip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config() -> dict:
    return json.loads((LOCK / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))


def test_lock_identity_and_mapping() -> None:
    config = _config()
    assert config["status"] == "LOCKED_ACTIVE_REFERENCE"
    assert config["portfolio_id"] == "eight_sleeve_equal_contract_reference_20261115"
    assert config["supersedes_active_reference"] == "eight_sleeve_equal_contract_reference_20261103"
    assert len(config["mapping"]) == 8 and set(config["mapping"].values()) == {1}
    assert SHADOW not in config["mapping"] and config["shadow_sleeves"][SHADOW]["allocation"] == 0
    assert config["sleeve_sources"][S01]["type"] == "frozen_atr_vote_ensemble"
    assert config["calendar"]["sessions"] == 572 and len(config["calendar"]["added_vs_mnq_calendar"]) == 8
    assert config["governance"]["no_retuning"]


def test_locked_outputs_match_hash_manifest_and_daily_sum() -> None:
    manifest = json.loads((LOCK / "lock_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        assert _sha256(ROOT / relative) == expected
    daily = pd.read_csv(LOCK / "daily_pnl.csv")
    cols = [c for c in daily if c not in {"session_date", "portfolio_pnl_usd"}]
    assert sorted(cols) == sorted(_config()["mapping"])
    assert np.allclose(daily[cols].sum(axis=1), daily["portfolio_pnl_usd"], atol=1e-7)
    assert abs(daily["portfolio_pnl_usd"].sum() - 30857.96) < 1e-6


def test_s01_is_the_signal_sheet_ensemble() -> None:
    source = _config()["sleeve_sources"][S01]
    for key in ("daily_source", "trades", "thresholds_is", "summary", "candidate_votes", "rebuild_script"):
        assert _sha256(ROOT / source[key]) == source[f"{key}_sha256"]
    trades = pd.read_csv(ROOT / source["trades"])
    assert len(trades) == 701 and abs(trades["net_pnl_usd"].sum() - 19416.5) < 1e-6
    thresholds = pd.read_csv(ROOT / source["thresholds_is"])
    assert len(thresholds) == 18
    assert set(thresholds["atr_period"]) == {14, 16, 18}
    assert set(thresholds["q_low_pct"]) == {20, 25, 30} and set(thresholds["q_high_pct"]) == {80, 85}
    sheet = pd.read_csv(ROOT / source["daily_source"], parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
    daily = pd.read_csv(LOCK / "daily_pnl.csv", parse_dates=["session_date"]).set_index("session_date")
    assert np.allclose(daily[S01], sheet.reindex(daily.index).fillna(0.0), atol=1e-9)


def test_other_sleeves_unchanged_from_previous_lock() -> None:
    old_config = json.loads((OLD / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))
    new, old = pd.read_csv(LOCK / "daily_pnl.csv"), pd.read_csv(OLD / "daily_pnl.csv")
    common = new.merge(old, on="session_date", suffixes=("", "_old"))
    assert len(common) == len(old) == 564
    for sleeve, source in _config()["sleeve_sources"].items():
        if sleeve == S01:
            continue
        assert source == old_config["sleeve_sources"][sleeve]
        assert _sha256(ROOT / source["daily_source"]) == source["daily_source_sha256"]
        assert _sha256(ROOT / source["ensemble_manifest"]) == source["ensemble_manifest_sha256"]
        assert np.allclose(common[sleeve], common[f"{sleeve}_old"], atol=1e-9)
    added = new[~new["session_date"].isin(old["session_date"])]
    assert (added[[S01, "mnq_volatility_breakout", "mnq_compression_breakout"]] == 0).all().all()


def test_stop_rule_thresholds_unchanged() -> None:
    new = pd.read_csv(LOCK / "sleeve_stop_rule_thresholds.csv")
    old = pd.read_csv(OLD / "sleeve_stop_rule_thresholds.csv")
    assert list(new["sleeve"]) == list(old["sleeve"]) and len(new) == 9
    assert np.allclose(new["threshold_usd"], old["threshold_usd"], atol=1e-9)


def test_forward_registry_records_this_lock_as_superseded_and_shadows_s04() -> None:
    # Superseded on 2026-11-21 by eight_sleeve_equal_contract_reference_20261121 (MNQ v.0, see SUPERSEDED.md);
    # the active-pointer test moved to tests/test_locked_eight_sleeve_portfolio_20261121.py.
    registry = json.loads((ROOT / "strategy_registry" / "forward_monitoring" / "registry.json").read_text(encoding="utf-8"))
    active = registry["active_reference"]
    superseded = active["superseded_reference"]
    assert superseded["portfolio_id"] == _config()["portfolio_id"]
    assert superseded["path"] == "research/locked_eight_sleeve_portfolio_20261115/locked_eight_sleeve_config.json"
    assert _sha256(ROOT / superseded["path"]) == superseded["sha256"]
    assert SHADOW not in active["members"] and sorted(active["members"]) == sorted(_config()["mapping"])
    shadow = next(entry for entry in registry["forward_monitoring"] if entry["id"] == SHADOW)
    assert shadow["status"] == "SHADOW_FORWARD_ONLY" and shadow["allocation"] == 0
    assert _sha256(ROOT / shadow["spec_path"]) == shadow["spec_sha256"]


def test_netting_rule_r2_applies_to_this_lock() -> None:
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    assert rule["status"] == "ACTIVE_EXECUTION_RULE"
    assert rule["applies_to_portfolio"] == _config()["portfolio_id"]
    assert rule["supersedes_rule"]["sha256"] == _sha256(OLD_RULE)
    assert rule["rule"] == json.loads(OLD_RULE.read_text(encoding="utf-8"))["rule"]
    for path, expected in rule["frozen_files"].items():
        assert _sha256(ROOT / path) == expected, f"{path} changed since the r2 netting rule was frozen"
    check = json.loads((LOCK / "netting_check.json").read_text(encoding="utf-8"))
    assert check["sessions"] == 572 and abs(check["separate_net_usd"] - 30857.96) < 1e-6
