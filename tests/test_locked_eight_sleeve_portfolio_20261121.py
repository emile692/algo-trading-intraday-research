"""Lock eight_sleeve_equal_contract_reference_20261121: the MNQ sleeves run on MNQ.v.0 back-adjusted instead of c.0."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research" / "locked_eight_sleeve_portfolio_20261121"
PREV = ROOT / "research" / "locked_eight_sleeve_portfolio_20261115"
REGISTRY = ROOT / "strategy_registry" / "forward_monitoring" / "registry.json"
R3 = ROOT / "strategy_registry" / "execution" / "netting_rule_20261121_r3.json"
R2 = ROOT / "strategy_registry" / "execution" / "netting_rule_20261115_r2.json"
STOP = ROOT / "strategy_registry" / "forward_monitoring" / "stop_rule_20261121.json"
CAT = ROOT / "strategy_registry" / "execution" / "catastrophe_stop_20261121.json"
MNQ_V0 = "data/processed/parquet/MNQ_v_0_adj_1m_20190505_20260320.parquet"
S01 = "mnq_orb_vwap_atr"
SHADOW = "mnq_atr_trend_flip"
MNQ_ACTIVE = (S01, "mnq_volatility_breakout", "mnq_compression_breakout")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config() -> dict:
    return _json(LOCK / "locked_eight_sleeve_config.json")


def test_lock_identity_and_mapping() -> None:
    config = _config()
    assert config["status"] == "LOCKED_ACTIVE_REFERENCE"
    assert config["portfolio_id"] == "eight_sleeve_equal_contract_reference_20261121"
    assert config["supersedes_active_reference"] == "eight_sleeve_equal_contract_reference_20261115"
    assert len(config["mapping"]) == 8 and set(config["mapping"].values()) == {1}
    assert SHADOW not in config["mapping"] and config["shadow_sleeves"][SHADOW]["allocation"] == 0
    assert config["calendar"]["sessions"] == 572
    assert config["governance"]["no_retuning"]


def test_locked_outputs_match_hash_manifest_and_daily_sum() -> None:
    for relative, expected in _json(LOCK / "lock_manifest.json")["files"].items():
        assert _sha256(ROOT / relative) == expected
    daily = pd.read_csv(LOCK / "daily_pnl.csv")
    cols = [c for c in daily if c not in {"session_date", "portfolio_pnl_usd"}]
    assert sorted(cols) == sorted(_config()["mapping"])
    assert np.allclose(daily[cols].sum(axis=1), daily["portfolio_pnl_usd"], atol=1e-7)
    assert abs(daily["portfolio_pnl_usd"].sum() - 31639.46) < 1e-6


def test_mnq_data_is_the_frozen_v0_file() -> None:
    data = _config()["mnq_data"]
    assert data["path"] == MNQ_V0 and _sha256(ROOT / MNQ_V0) == data["sha256"]
    meta = _json((ROOT / MNQ_V0).with_suffix(".metadata.json"))
    assert meta["sha256"] == data["sha256"] and meta["last"] == "2026-03-20 09:29:00-04:00"
    for sleeve in (*MNQ_ACTIVE, SHADOW):
        source = _config()["sleeve_sources"].get(sleeve) or _config()["shadow_sleeves"][sleeve]
        assert source["data"] == MNQ_V0
        assert _sha256(ROOT / source["daily_source"]) == source["daily_source_sha256"]
        assert "_v0/" in source["daily_source"]


def test_mnq_sleeves_rebuilt_on_v0_with_frozen_definitions() -> None:
    sources = _config()["sleeve_sources"]
    expected = {S01: (707, 18616.5), "mnq_volatility_breakout": (1110, 22747.0), "mnq_compression_breakout": (332, 2612.0)}
    for sleeve, (n, pnl) in expected.items():
        trades = pd.read_csv(ROOT / sources[sleeve]["trades"])
        assert _sha256(ROOT / sources[sleeve]["trades"]) == sources[sleeve]["trades_sha256"]
        assert len(trades) == n and abs(trades["net_pnl_usd"].sum() - pnl) < 1e-6
        if "ensemble_manifest" in sources[sleeve]:
            assert _sha256(ROOT / sources[sleeve]["ensemble_manifest"]) == sources[sleeve]["ensemble_manifest_sha256"]
    # S01 keeps the IS thresholds calibrated on c.0, byte for byte.
    assert _sha256(ROOT / sources[S01]["thresholds_is"]) == _sha256(ROOT / "research" / "signal_sheet_01_atr18" / "thresholds_is.csv")


def test_mgc_mbt_sleeves_unchanged_from_previous_lock() -> None:
    prev = _json(PREV / "locked_eight_sleeve_config.json")
    new, old = pd.read_csv(LOCK / "daily_pnl.csv"), pd.read_csv(PREV / "daily_pnl.csv")
    assert list(new["session_date"]) == list(old["session_date"])
    for sleeve, source in _config()["sleeve_sources"].items():
        if sleeve.startswith("mnq_"):
            continue
        assert source == prev["sleeve_sources"][sleeve]
        assert np.allclose(new[sleeve], old[sleeve], atol=1e-9)


def test_stop_rule_thresholds() -> None:
    new = pd.read_csv(LOCK / "sleeve_stop_rule_thresholds.csv").set_index("sleeve")
    old = pd.read_csv(PREV / "sleeve_stop_rule_thresholds.csv").set_index("sleeve")
    assert list(new.index) == list(old.index) and len(new) == 9
    for sleeve in new.index:
        if sleeve.startswith("mnq_"):
            assert new.loc[sleeve, "threshold_usd_c0_20261103"] == old.loc[sleeve, "threshold_usd"]
        else:
            assert abs(new.loc[sleeve, "threshold_usd"] - old.loc[sleeve, "threshold_usd"]) < 1e-9


def test_registry_points_to_this_lock() -> None:
    ref = _json(REGISTRY)["active_reference"]
    assert ref["portfolio_id"] == _config()["portfolio_id"]
    assert ref["canonical_lock"]["path"] == "research/locked_eight_sleeve_portfolio_20261121/locked_eight_sleeve_config.json"
    assert _sha256(ROOT / ref["canonical_lock"]["path"]) == ref["canonical_lock"]["sha256"]
    assert ref["execution_rule"] == "strategy_registry/execution/netting_rule_20261121_r3.json"
    assert ref["stop_rule"] == "strategy_registry/forward_monitoring/stop_rule_20261121.json"
    assert ref["protective_orders"] == ["strategy_registry/execution/catastrophe_stop_20261121.json"]
    assert ref["superseded_reference"]["portfolio_id"] == "eight_sleeve_equal_contract_reference_20261115"
    assert _sha256(ROOT / ref["superseded_reference"]["path"]) == ref["superseded_reference"]["sha256"]
    assert SHADOW not in ref["members"] and sorted(ref["members"]) == sorted(_config()["mapping"])


def test_netting_rule_r3_applies_to_this_lock() -> None:
    rule, r2 = _json(R3), _json(R2)
    assert rule["status"] == "ACTIVE_EXECUTION_RULE" and rule["applies_to_portfolio"] == _config()["portfolio_id"]
    assert rule["rule"] == r2["rule"]
    assert rule["supersedes_rule"]["sha256"] == _sha256(R2)
    for path, expected in rule["frozen_files"].items():
        assert _sha256(ROOT / path) == expected, f"{path} changed since the r3 netting rule was frozen"
    assert MNQ_V0 in rule["frozen_files"]
    check = _json(LOCK / "netting_check.json")
    assert check["sessions"] == 572 and abs(check["separate_net_usd"] - 31639.46) < 1e-6
    assert abs(check["gross_usd"] - (check["separate_net_usd"] + check["separate_costs_usd"])) < 1e-6


def test_stop_rule_20261121_is_wired_and_recomputed_on_v0() -> None:
    rule, old = _json(STOP), _json(ROOT / "strategy_registry" / "forward_monitoring" / "stop_rule_20261115.json")
    assert rule["status"] == "ACTIVE_STOP_RULE" and rule["applies_to_portfolio"] == _config()["portfolio_id"]
    assert rule["rules"]["R2_shadow"]["valid_up_to_h"] == old["rules"]["R2_shadow"]["valid_up_to_h"]
    for path, expected in rule["evidence_sha256_lf"].items():
        assert _sha256_lf(ROOT / path) == expected, f"{path} changed since the stop rule was frozen"
    r0 = pd.read_csv(ROOT / rule["rules"]["R0_alert"]["thresholds_file"]).set_index("sleeve")["threshold_usd"]
    curve = pd.read_csv(ROOT / rule["rules"]["R2_shadow"]["threshold_curve"]).set_index("h")
    for code, t in rule["thresholds"].items():
        assert t["R0_126_usd"] == r0[t["sleeve"]]
        assert abs(curve.loc[250, code] - t["R2_maxdd_p01_at_250_usd"]) <= 0.01 + 1e-9  # both rounded to the cent
        if not t["sleeve"].startswith("mnq_"):
            assert {k: t[k] for k in ("R3_120_p05_usd", "R2_maxdd_p01_at_250_usd")} == \
                   {k: old["thresholds"][code][k] for k in ("R3_120_p05_usd", "R2_maxdd_p01_at_250_usd")}
    statuses = rule["retroactive_application"]["statuses_as_of_2026_11_15"]
    assert {c for c, v in statuses.items() if v["status"] == "SHADOW_FORWARD_ONLY"} == {"S04"}


def test_catastrophe_stop_20261121_unchanged_order_rechecked_on_v0() -> None:
    rule, old = _json(CAT), _json(ROOT / "strategy_registry" / "execution" / "catastrophe_stop_20261115.json")
    assert rule["status"] == "ACTIVE_PROTECTIVE_ORDER" and rule["rule"] == old["rule"] and rule["rule"]["k"] == 3.0
    assert rule["applies_to_portfolio"] == _config()["portfolio_id"]
    assert rule["execution_rule"] == "strategy_registry/execution/netting_rule_20261121_r3.json"
    for path, expected in rule["evidence_sha256_lf"].items():
        assert _sha256_lf(ROOT / path) == expected
    check = _json(LOCK / "catastrophe_stop_check_v0.json")
    k3 = next(r for r in check["summary"] if r["k"] == 3.0)
    assert k3["triggers_MNQ"] == k3["triggers_MGC"] == k3["triggers_MBT"] == 0
    assert rule["evidence"]["v0_recheck"]["triggers_k3"] == {"MNQ": 0, "MGC": 0, "MBT": 0}
