import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "research" / "locked_nine_sleeve_portfolio_20261101"
SUPERSEDED_LOCK = ROOT / "research" / "locked_nine_sleeve_portfolio"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_nine_sleeve_equal_contract_reference_is_locked() -> None:
    config = json.loads((LOCK / "locked_nine_sleeve_config.json").read_text(encoding="utf-8"))
    assert config["status"] == "LOCKED_ACTIVE_REFERENCE"
    assert config["portfolio_id"] == "nine_sleeve_equal_contract_reference_20261101"
    assert config["supersedes_active_reference"] == "nine_sleeve_equal_contract_reference_20261004"
    assert config["allocation_family"] == "equal_contract"
    assert len(config["mapping"]) == 9
    assert set(config["mapping"].values()) == {1}
    assert config["topstep_reference"]["horizon_60d"]["pass_probability"] == 0.3895
    assert config["comparison_with_superseded"]["superseded_reproduced_pass_60d"] == 0.452375
    assert config["governance"]["no_retuning"]


def test_locked_outputs_match_hash_manifest_and_daily_sum() -> None:
    manifest = json.loads((LOCK / "lock_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        assert _sha256(ROOT / relative) == expected
    daily = pd.read_csv(LOCK / "daily_pnl.csv")
    sleeve_columns = [column for column in daily if column not in {"session_date", "portfolio_pnl_usd"}]
    assert len(sleeve_columns) == 9
    assert np.allclose(daily[sleeve_columns].sum(axis=1), daily["portfolio_pnl_usd"], atol=1e-7)


def test_vote_sleeves_point_to_hashed_frozen_ensembles() -> None:
    config = json.loads((LOCK / "locked_nine_sleeve_config.json").read_text(encoding="utf-8"))
    votes = {name: source for name, source in config["sleeve_sources"].items() if source["type"] == "frozen_vote_ensemble"}
    assert len(votes) == 8
    for source in votes.values():
        assert _sha256(ROOT / source["ensemble_manifest"]) == source["ensemble_manifest_sha256"]
        assert _sha256(ROOT / source["daily_source"]) == source["daily_source_sha256"]


def test_superseded_reference_retained_unchanged() -> None:
    manifest = json.loads((SUPERSEDED_LOCK / "lock_manifest.json").read_text(encoding="utf-8"))
    assert manifest["portfolio_id"] == "nine_sleeve_equal_contract_reference_20261004"
    for relative, expected in manifest["files"].items():
        assert _sha256(ROOT / relative) == expected
