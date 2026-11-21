"""The catastrophe stop decided on 2026-11-15 (k = 3) matches its evidence and is wired into the forward registry."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "strategy_registry" / "execution" / "catastrophe_stop_20261115.json"
REGISTRY = ROOT / "strategy_registry" / "forward_monitoring" / "registry.json"


def _sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _rule() -> dict:
    return json.loads(RULE.read_text(encoding="utf-8"))


def test_rule_is_k3_and_carried_over_to_the_active_portfolio():
    # Since 2026-11-21 the same order is attached to the MNQ v.0 lock by catastrophe_stop_20261121.json.
    rule, registry = _rule(), json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert rule["status"] == "ACTIVE_PROTECTIVE_ORDER"
    assert rule["rule"]["k"] == 3.0
    assert rule["applies_to_portfolio"] == registry["active_reference"]["superseded_reference"]["portfolio_id"]
    active = json.loads((ROOT / registry["active_reference"]["protective_orders"][0]).read_text(encoding="utf-8"))
    assert active["rule"] == rule["rule"] and active["supersedes"]["rule_id"] == rule["rule_id"]
    assert active["applies_to_portfolio"] == registry["active_reference"]["portfolio_id"]
    assert active["execution_rule"] == registry["active_reference"]["execution_rule"]


def test_evidence_files_are_unchanged():
    for path, expected in _rule()["evidence_sha256_lf"].items():
        assert _sha256_lf(ROOT / path) == expected, f"{path} changed since the catastrophe stop was frozen"


def test_k3_never_triggered_on_the_history():
    summary = json.loads((ROOT / "audit_20261115" / "outputs" / "10_summary.json").read_text(encoding="utf-8"))["summary"]
    k3 = next(r for r in summary if r["k"] == 3.0)
    rule = _rule()
    for inst, n in rule["evidence"]["triggers_k3"].items():
        assert k3[f"triggers_{inst}"] == n == 0
    assert round(k3["bound_p99"], 2) == rule["evidence"]["open_position_bound_k3_usd"]["p99"]
