"""Confirm and apply the user-selected MGC gap 0.50/0.60 freeze amendment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.mgc_final_two_sleeve_lock_20260926 import run_final_two_sleeve_lock as final


OUTPUT_DIR = Path(__file__).resolve().parent
OVERRIDE_PATH = OUTPUT_DIR / "gap_user_override_0_50_0_60.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = final.load_config()
    data_path = final.FINAL_DATA.resolve()
    data_hash = digest(data_path)
    if data_hash != final.EXPECTED_DATA_SHA256:
        raise RuntimeError(f"MGC v0 hash mismatch: {data_hash}")
    override = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
    parameters = override["locked_parameters"]
    final.init_worker(str(data_path), config)
    output = final.worker({
        "mechanism_id": "gap_continuation",
        "point_id": "gap_user_0.5_0.6",
        "p1": parameters["minimum_gap_atr_multiple"],
        "p2": parameters["volume_threshold_multiple"],
        "i": 4,
        "j": 4,
    })
    period = pd.DataFrame(output["period_metrics"]).set_index("period")
    years = pd.DataFrame(output["yearly"])
    complete_years = set(config["freeze_gates"]["complete_years"])
    positive_complete_years = int(years[years.year.isin(complete_years)].sharpe_annualized.gt(0).sum())
    gates = config["freeze_gates"]
    passed = bool(
        period.trade_count.ge(gates["minimum_trades_each_period"]).all()
        and period.sharpe_annualized.gt(gates["minimum_sharpe_each_period_exclusive"]).all()
        and period.profit_factor.gt(gates["minimum_profit_factor_each_period_exclusive"]).all()
        and positive_complete_years >= gates["minimum_positive_complete_years"]
    )
    diagnostic = {
        "override_sha256": digest(OVERRIDE_PATH),
        "data_sha256": data_hash,
        "passed_precommitted_freeze_gates": passed,
        "positive_complete_years": positive_complete_years,
        "period_metrics": output["period_metrics"],
        "yearly_metrics": output["yearly"],
        "rolling_metrics": output["rolling"],
    }
    diagnostic_path = OUTPUT_DIR / "gap_user_override_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("User-selected gap point failed the precommitted freeze gates; existing freeze was not changed")

    config_hash = digest(final.CONFIG_PATH)
    spec = final.effective_spec("gap_continuation", 0.5, 0.6, "FREEZE_CANDIDATE", config_hash)
    spec["freeze_metadata"].update({
        "selection": "explicit_user_choice_inside_confirmed_plateau",
        "override_path": str(OVERRIDE_PATH.relative_to(final.REPO_ROOT)).replace("\\", "/"),
        "override_sha256": digest(OVERRIDE_PATH),
        "exact_point_confirmed": True,
    })
    spec_path = final.SPEC_DIR / "gap_continuation.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_path = OUTPUT_DIR / "frozen_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gap_entry = next(item for item in manifest["candidates"] if item["mechanism_id"] == "gap_continuation")
    gap_entry.update({
        "parameter_1_value": 0.5,
        "parameter_2_value": 0.6,
        "locked_parameters": parameters,
        "selection": "explicit_user_choice_inside_confirmed_plateau",
        "exact_point_confirmed": True,
        "override_sha256": digest(OVERRIDE_PATH),
        "specification_sha256": digest(spec_path),
    })
    manifest["amended_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["amendment"] = "gap_continuation representative changed from geometric medoid 0.50/0.50 to user-selected interior point 0.50/0.60"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_hash = digest(manifest_path)
    (OUTPUT_DIR / "frozen_candidate_manifest.sha256").write_text(
        f"{manifest_hash}  {manifest_path.name}\n", encoding="ascii"
    )

    csv_path = OUTPUT_DIR / "frozen_candidate_manifest.csv"
    csv = pd.read_csv(csv_path)
    mask = csv.mechanism_id.eq("gap_continuation")
    csv.loc[mask, "parameter_1_value"] = 0.5
    csv.loc[mask, "parameter_2_value"] = 0.6
    csv.loc[mask, "specification_sha256"] = digest(spec_path)
    csv.to_csv(csv_path, index=False)

    metadata_path = OUTPUT_DIR / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "manifest_sha256": manifest_hash,
        "gap_user_override": parameters,
        "gap_user_override_sha256": digest(OVERRIDE_PATH),
        "gap_user_override_confirmed": True,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    amendment = [
        "", "## Amendement utilisateur — gap_continuation", "",
        "- Représentant final remplacé par `minimum_gap_atr_multiple=0.50`, `volume_threshold_multiple=0.60`.",
        "- Choix explicite à l'intérieur du plateau 81/81; aucune recherche ni comparaison supplémentaire.",
        f"- Contrôle exact : Sharpe IS/Validation `{period.loc['is','sharpe_annualized']:.3f}/{period.loc['validation','sharpe_annualized']:.3f}`; "
        f"PF `{period.loc['is','profit_factor']:.3f}/{period.loc['validation','profit_factor']:.3f}`; "
        f"trades `{int(period.loc['is','trade_count'])}/{int(period.loc['validation','trade_count'])}`; "
        f"DD `{period.loc['is','max_drawdown_usd']:.0f}/{period.loc['validation','max_drawdown_usd']:.0f} USD`.",
        f"- Années complètes positives : `{positive_complete_years}/6`; gates de freeze : **PASS**.",
        f"- Nouveau SHA-256 du manifeste : `{manifest_hash}`.", "",
    ]
    report_path = OUTPUT_DIR / "FINAL_TWO_SLEEVE_LOCK_REPORT.md"
    report_path.write_text(report_path.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(amendment), encoding="utf-8")
    print(json.dumps({"passed": passed, "manifest_sha256": manifest_hash, "period_metrics": output["period_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
