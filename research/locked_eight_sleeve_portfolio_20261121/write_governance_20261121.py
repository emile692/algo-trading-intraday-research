"""Versioned governance files for the 2026-11-21 lock (MNQ v.0), and the registry pointer.

New files (the 2026-11-15 ones are never overwritten):
  strategy_registry/execution/netting_rule_20261121_r3.json      same rule logic as r2, applies to the new lock
  strategy_registry/forward_monitoring/stop_rule_20261121.json   same rules; MNQ and book thresholds recomputed on v.0
  strategy_registry/execution/catastrophe_stop_20261121.json     same order (k = 3); evidence re-checked on v.0
and strategy_registry/forward_monitoring/registry.json: active_reference re-pointed to the new lock, the 2026-11-15
lock recorded as superseded (earlier superseded references kept in a list).
Run after relock_eight_sleeves_20261121.py, stop_rule_thresholds_v0.py and catastrophe_stop_check_v0.py.
Command: python -m research.locked_eight_sleeve_portfolio_20261121.write_governance_20261121
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REG = ROOT / "strategy_registry"
LOCK_ID = "eight_sleeve_equal_contract_reference_20261121"
PREV_ID = "eight_sleeve_equal_contract_reference_20261115"
DATE = "2026-11-21"
AUTH = "explicit_user decision of 2026-11-21 ('freeze le mnq v.0 au lieu de c.0'; back-adjusted v.0, same window, stop-rule thresholds recomputed on v.0)"
NETTING_R2 = REG / "execution" / "netting_rule_20261115_r2.json"
NETTING_R3 = REG / "execution" / "netting_rule_20261121_r3.json"
STOP_OLD = REG / "forward_monitoring" / "stop_rule_20261115.json"
STOP_NEW = REG / "forward_monitoring" / "stop_rule_20261121.json"
CAT_OLD = REG / "execution" / "catastrophe_stop_20261115.json"
CAT_NEW = REG / "execution" / "catastrophe_stop_20261121.json"
REGISTRY = REG / "forward_monitoring" / "registry.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, data: dict) -> None:
    if path.exists():
        raise SystemExit(f"{rel(path)} exists: versioned files are never overwritten")
    with path.open("w", encoding="utf-8", newline="\n") as f:  # LF: strategy_registry files are hashed on LF bytes
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    config_path = HERE / "locked_eight_sleeve_config.json"
    config = load(config_path)
    assert config["portfolio_id"] == LOCK_ID

    # --- netting rule r3
    r2 = load(NETTING_R2)
    r3 = copy.deepcopy(r2)
    frozen = {k: v for k, v in r2["frozen_files"].items() if "locked_eight_sleeve_portfolio_20261115" not in k
              and k != "research/signal_sheet_01_atr18/trades.csv"}
    for p in (config_path, HERE / "netting_check.json", ROOT / config["mnq_data"]["path"],
              *(ROOT / config["sleeve_sources"][s]["trades"] for s in ("mnq_orb_vwap_atr", "mnq_volatility_breakout", "mnq_compression_breakout"))):
        frozen[rel(p)] = sha(p)
    r3.update({
        "rule_id": "instrument_netting_eight_sleeves_20261121_r3", "status": "ACTIVE_EXECUTION_RULE", "decision_date": DATE,
        "decision_authority": AUTH,
        "supersedes_rule": {"rule_id": r2["rule_id"], "path": rel(NETTING_R2), "sha256": sha(NETTING_R2),
                            "reason": "the superseded rule applies to eight_sleeve_equal_contract_reference_20261115 (MNQ c.0); the rule logic is unchanged"},
        "applies_to_portfolio": LOCK_ID,
        "s01_order_note": "S01 is the 18-model ATR vote of signal sheet 01 rebuilt on MNQ v.0 (research/signal_sheet_01_atr18_v0): same entry, stop, target, 16:00 exit and native 1-4 contract sizing; frozen IS thresholds unchanged",
        "frozen_files": frozen,
        "open_points": [p for p in r2["open_points"] if "c.0" not in p],
    })
    r3["evidence"] = dict(r2["evidence"], netting_reconciliation_new_book=rel(HERE / "netting_check.json"))
    dump(NETTING_R3, r3)

    # --- stop rule
    old = load(STOP_OLD)
    th = pd.read_csv(HERE / "stop_rule_thresholds_v0.csv").set_index("series")
    new = copy.deepcopy(old)
    new.update({"rule_id": "sleeve_stop_rule_r2_shadow_20261121", "decision_date": DATE, "decision_authority": AUTH,
                "applies_to_portfolio": LOCK_ID,
                "supersedes": {"rule_id": old["rule_id"], "path": rel(STOP_OLD), "sha256": sha(STOP_OLD),
                               "reason": "same rules; the R0, R2 and R3 thresholds of the MNQ sleeves and of the book are recomputed on the MNQ v.0 histories of the 2026-11-21 lock"}})
    new["rules"] = copy.deepcopy(old["rules"])
    new["rules"]["R2_shadow"]["threshold_curve"] = rel(HERE / "r2_threshold_curve_v0.csv")
    new["rules"]["R2_shadow"]["trigger"] = old["rules"]["R2_shadow"]["trigger"] + "; MNQ and book curves recomputed on v.0 (research/locked_eight_sleeve_portfolio_20261121/stop_rule_thresholds_v0.py), MGC/MBT curves unchanged"
    new["rules"]["R0_alert"]["thresholds_file"] = rel(HERE / "sleeve_stop_rule_thresholds.csv")
    for code, t in new["thresholds"].items():
        row = th.loc[code]
        t.update({"R0_126_usd": float(row["R0_126_usd"]), "R3_120_p05_usd": float(row["R3_pnl120_p05_usd"]),
                  "R2_maxdd_p01_at_250_usd": float(row["R2_dd_p01_250_usd"]), "source": row["source"]})
    book = th.loc["Book 8"]
    new["book_thresholds"] = {"R3_120_p05_usd": float(book["R3_pnl120_p05_usd"]), "R2_maxdd_p01_at_250_usd": float(book["R2_dd_p01_250_usd"]),
                              "source": book["source"]}
    new["retroactive_application"] = dict(old["retroactive_application"],
                                          note_20261121="statuses decided on 2026-11-15 from the c.0 replay window 2026-03-20 -> 2026-10-02; kept unchanged (that window is not re-read on v.0)")
    new["evidence_sha256_lf"] = {**old["evidence_sha256_lf"],
                                 **{rel(HERE / n): sha_lf(HERE / n) for n in ("r2_threshold_curve_v0.csv", "stop_rule_thresholds_v0.csv", "stop_rule_thresholds_v0.py")}}
    new["limits"] = old["limits"] + "; MNQ thresholds now computed on v.0 back-adjusted histories (same sessions plus the quarterly expiry Fridays)"
    dump(STOP_NEW, new)

    # --- catastrophe stop
    cat = load(CAT_OLD)
    chk = json.loads((HERE / "catastrophe_stop_check_v0.json").read_text(encoding="utf-8"))
    by_k = {r["k"]: r for r in chk["summary"]}
    if any(by_k[3.0][f"triggers_{i}"] for i in ("MNQ", "MGC", "MBT")):
        raise RuntimeError("k = 3 triggers on the v.0 history: the catastrophe stop needs a new decision")
    newcat = copy.deepcopy(cat)
    newcat.update({"rule_id": "catastrophe_stop_k3_20261121", "decision_date": DATE, "decision_authority": AUTH + "; the order itself is unchanged (k = 3, decided 2026-11-15)",
                   "applies_to_portfolio": LOCK_ID, "execution_rule": rel(NETTING_R3),
                   "supersedes": {"rule_id": cat["rule_id"], "path": rel(CAT_OLD), "sha256": sha(CAT_OLD),
                                  "reason": "same order; re-attached to the MNQ v.0 lock and its evidence re-checked on v.0"}})
    newcat["evidence"] = dict(cat["evidence"], c0_evidence="kept above (audit_20261115 phase 10, MNQ c.0)",
                              v0_recheck={"script": rel(HERE / "catastrophe_stop_check_v0.py"), "output": rel(HERE / "catastrophe_stop_check_v0.json"),
                                          "sessions": chk["sessions"],
                                          **{f"triggers_k{str(k).replace('.', '_').replace('_0', '')}": {i: by_k[k][f"triggers_{i}"] for i in ("MNQ", "MGC", "MBT")} for k in (1.5, 2.0, 3.0)},
                                          "open_position_bound_k3_usd": {"p50": round(by_k[3.0]["bound_p50"], 2), "p99": round(by_k[3.0]["bound_p99"], 2),
                                                                          "max": round(by_k[3.0]["bound_max"], 2)}})
    newcat["evidence_sha256_lf"] = {**cat["evidence_sha256_lf"],
                                    **{rel(HERE / n): sha_lf(HERE / n) for n in ("catastrophe_stop_check_v0.py", "catastrophe_stop_check_v0.json")}}
    dump(CAT_NEW, newcat)

    # --- registry (living file, updated in place as in previous relocks)
    reg = load(REGISTRY)
    ref = reg["active_reference"]
    earlier = ref.get("earlier_superseded_references", [])
    earlier.append(ref["superseded_reference"])
    ref.update({"portfolio_id": LOCK_ID, "canonical_lock": {"path": rel(config_path), "sha256": sha(config_path)},
                "execution_rule": rel(NETTING_R3), "protective_orders": [rel(CAT_NEW)], "stop_rule": rel(STOP_NEW),
                "superseded_reference": {"portfolio_id": PREV_ID,
                                         "path": "research/locked_eight_sleeve_portfolio_20261115/locked_eight_sleeve_config.json",
                                         "sha256": sha(ROOT / "research/locked_eight_sleeve_portfolio_20261115/locked_eight_sleeve_config.json")},
                "earlier_superseded_references": earlier,
                "mnq_data": config["mnq_data"]["path"]})
    with REGISTRY.open("w", encoding="utf-8", newline="\n") as f:  # the registry is LF
        f.write(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
    print("written:", rel(NETTING_R3), rel(STOP_NEW), rel(CAT_NEW), "and registry re-pointed")


if __name__ == "__main__":
    main()
