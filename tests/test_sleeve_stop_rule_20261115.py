"""The sleeve stop rule decided on 2026-11-15 (R2 alone -> shadow, R0/R3 -> review) matches the audit evidence."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "strategy_registry" / "forward_monitoring" / "stop_rule_20261115.json"
REGISTRY = ROOT / "strategy_registry" / "forward_monitoring" / "registry.json"


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _rule() -> dict:
    return json.loads(RULE.read_text(encoding="utf-8"))


def test_rule_is_wired_into_the_registry():
    rule, registry = _rule(), json.loads(REGISTRY.read_text(encoding="utf-8"))
    ref = registry["active_reference"]
    assert rule["status"] == "ACTIVE_STOP_RULE"
    assert rule["applies_to_portfolio"] == ref["portfolio_id"]
    assert ref["stop_rule"] == "strategy_registry/forward_monitoring/stop_rule_20261115.json"
    under_review = {v["sleeve"] for v in rule["retroactive_application"]["statuses_as_of_2026_11_15"].values()
                    if v["status"] == "ACTIVE_UNDER_REVIEW"}
    assert under_review == set(ref["under_review"])
    assert under_review <= set(ref["members"])


def test_evidence_files_are_unchanged():
    for path, expected in _rule()["evidence_sha256_lf"].items():
        assert _sha256_lf(ROOT / path) == expected, f"{path} changed since the stop rule was frozen"


def test_thresholds_match_the_audit_and_the_lock():
    rule = _rule()
    th = json.loads((ROOT / "audit_20261115" / "outputs" / "05_thresholds.json").read_text(encoding="utf-8"))
    r0 = pd.read_csv(ROOT / rule["rules"]["R0_alert"]["thresholds_file"]).set_index("sleeve")["threshold_usd"]
    curve = pd.read_csv(ROOT / rule["rules"]["R2_shadow"]["threshold_curve"]).set_index("h")
    assert len(curve) == rule["rules"]["R2_shadow"]["valid_up_to_h"]
    for code, t in rule["thresholds"].items():
        assert t["R0_126_usd"] == r0[t["sleeve"]]
        assert t["R3_120_p05_usd"] == round(th[code]["R3_pnl120_p05"], 2)
        assert t["R2_maxdd_p01_at_250_usd"] == round(th[code]["R2_dd_p01_250"], 2)
        assert abs(curve.loc[250, code] - th[code]["R2_dd_p01_250"]) <= 0.01  # curve rounded to the cent
        assert curve[code].is_monotonic_decreasing


def test_only_s04_is_in_shadow_after_the_retroactive_application():
    statuses = _rule()["retroactive_application"]["statuses_as_of_2026_11_15"]
    assert {c for c, v in statuses.items() if v["status"] == "SHADOW_FORWARD_ONLY"} == {"S04"}
