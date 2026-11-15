"""The active netting rule of the eight-sleeve book stays byte-identical to what was frozen on 2026-11-15."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "strategy_registry" / "execution" / "netting_rule_20261115.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_netting_rule_files_are_unchanged():
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    assert rule["status"] == "ACTIVE_EXECUTION_RULE"
    for path, expected in rule["frozen_files"].items():
        assert _sha256(ROOT / path) == expected, f"{path} changed since the netting rule was frozen"


def test_netting_rule_matches_spec_lock_and_oos_run():
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    campaign = ROOT / "research" / "book_aggregation_risk_r1_20261112"
    lock = json.loads((campaign / "SPEC_LOCK.json").read_text(encoding="utf-8"))
    assert rule["frozen_files"]["research/book_aggregation_risk_r1_20261112/FROZEN_SPECIFICATION.json"] == lock["spec_sha256"]
    oos = json.loads((campaign / "results" / "oos" / "oos_results.json").read_text(encoding="utf-8"))
    assert oos["verdict"] == "ADOPT_AGGREGATION_ONLY"
    for path in ("src/portfolio/aggregator.py", "src/analytics/topstep_lifecycle/simulator.py",
                 "research/book_aggregation_risk_r1_20261112/run_study.py"):
        assert oos["code_sha256"][path] == rule["frozen_files"][path]


def test_netting_rule_does_not_touch_the_portfolio_lock():
    rule = json.loads(RULE.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "research" / "locked_eight_sleeve_portfolio_20261103" / "locked_eight_sleeve_config.json").read_text(encoding="utf-8"))
    assert rule["applies_to_portfolio"] == config["portfolio_id"]
    assert set(config["mapping"].values()) == {1} and len(config["mapping"]) == 8
