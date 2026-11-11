"""Materialize the inherited 7x7 grid before executing MCL landscapes."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "research/mgc_wide_parameter_landscape/mgc_wide_parameter_grids.json"
TARGET = HERE / "mcl_wide_parameter_grids.json"

payload = json.loads(SOURCE.read_text(encoding="utf-8"))
payload["selection_basis"] = (
    "Inherited unchanged from the pre-existing MGC 7x7 grids before any MCL landscape result; "
    "only instrument/session metadata and evaluation dates are adapted."
)
payload["periods"] = {"is": ["2021-07-12", "2023-12-31"], "validation": ["2024-01-01", "2026-03-19"]}
payload["common_rules"]["instrument"] = "MCL"
payload["common_rules"]["costs_and_session"] = "research/mcl_baseline_backtests/mcl_baseline_config.json"
TARGET.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
(HERE / "mcl_wide_parameter_grids.sha256").write_text(f"{digest}  {TARGET.name}\n", encoding="ascii")
print(digest)
