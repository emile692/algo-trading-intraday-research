import json
from pathlib import Path

import pandas as pd

from src.analytics.vwap_reranking import (
    RerankingSpec,
    VariantEvaluation,
    _final_verdict_payload,
    _merge_reranking_tables,
    build_default_reranking_spec,
    generate_reranking_notebook,
)
from src.config.vwap_campaign import build_default_vwap_reranking_variants, resolve_vwap_variant
from src.engine.vwap_backtester import VWAPBacktestResult


def test_default_reranking_universe_is_compact_and_excludes_invalidated_pullback() -> None:
    variants = build_default_vwap_reranking_variants()
    names = [variant.name for variant in variants]

    assert names[0] == "paper_vwap_baseline"
    assert "vwap_pullback_continuation" not in names
    assert "baseline_futures_adapted" in names
    assert len(names) <= 8


def _dummy_evaluation(variant_name: str, oos_net: float, oos_pf: float, oos_sharpe: float, oos_dd: float) -> VariantEvaluation:
    variant = resolve_vwap_variant(variant_name)
    summary = pd.DataFrame(
        [
            {
                "scope": "overall",
                "net_pnl": oos_net + 100.0,
                "profit_factor": max(oos_pf, 1.0),
                "sharpe_ratio": oos_sharpe,
                "max_drawdown": oos_dd,
                "expectancy_per_trade": 1.0,
                "total_trades": 25,
                "worst_daily_loss_usd": -200.0,
                "daily_loss_limit_breach_freq": 0.0,
                "trailing_drawdown_breach_freq": 0.0,
                "top_3_day_contribution_pct": 0.30,
                "top_5_day_contribution_pct": 0.45,
                "pnl_excluding_top_3_days": 50.0,
                "pnl_excluding_top_5_days": 25.0,
                "mean_trades_per_day": 2.0,
                "max_trades_per_day": 4,
                "worst_losing_trades_streak": 2,
                "worst_losing_days_streak": 1,
            },
            {
                "scope": "is",
                "net_pnl": oos_net / 2.0,
                "profit_factor": max(oos_pf, 1.0),
                "sharpe_ratio": oos_sharpe,
                "max_drawdown": oos_dd / 2.0,
                "expectancy_per_trade": 1.0,
                "total_trades": 10,
                "worst_daily_loss_usd": -150.0,
                "daily_loss_limit_breach_freq": 0.0,
                "trailing_drawdown_breach_freq": 0.0,
                "top_3_day_contribution_pct": 0.20,
                "top_5_day_contribution_pct": 0.30,
                "pnl_excluding_top_3_days": 40.0,
                "pnl_excluding_top_5_days": 30.0,
                "mean_trades_per_day": 2.0,
                "max_trades_per_day": 4,
                "worst_losing_trades_streak": 2,
                "worst_losing_days_streak": 1,
            },
            {
                "scope": "oos",
                "net_pnl": oos_net,
                "profit_factor": oos_pf,
                "sharpe_ratio": oos_sharpe,
                "max_drawdown": oos_dd,
                "expectancy_per_trade": 1.0,
                "total_trades": 15,
                "worst_daily_loss_usd": -200.0,
                "daily_loss_limit_breach_freq": 0.0,
                "trailing_drawdown_breach_freq": 0.0,
                "top_3_day_contribution_pct": 0.30,
                "top_5_day_contribution_pct": 0.45,
                "pnl_excluding_top_3_days": 50.0,
                "pnl_excluding_top_5_days": 25.0,
                "mean_trades_per_day": 2.0,
                "max_trades_per_day": 4,
                "worst_losing_trades_streak": 2,
                "worst_losing_days_streak": 1,
            },
        ]
    )
    empty_result = VWAPBacktestResult(trades=pd.DataFrame(), bar_results=pd.DataFrame(), daily_results=pd.DataFrame())
    return VariantEvaluation(
        variant=variant,
        signal_df=pd.DataFrame(),
        result=empty_result,
        trades=pd.DataFrame(),
        daily_results=pd.DataFrame(),
        bar_results=pd.DataFrame(),
        instrument=None,  # type: ignore[arg-type]
        execution_model=None,  # type: ignore[arg-type]
        all_sessions=[],
        is_sessions=[],
        oos_sessions=[],
        summary_by_scope=summary,
        tables_by_scope={},
    )


def test_reranking_summary_keeps_baselines_and_promotes_only_true_survivor() -> None:
    spec = build_default_reranking_spec()
    spec = RerankingSpec(
        dataset_path=spec.dataset_path,
        variant_names=("paper_vwap_baseline", "baseline_futures_adapted", "vwap_reclaim"),
        prop_constraints=spec.prop_constraints,
    )
    evaluations = {
        "paper_vwap_baseline": _dummy_evaluation("paper_vwap_baseline", 100.0, 1.02, 0.10, -500.0),
        "baseline_futures_adapted": _dummy_evaluation("baseline_futures_adapted", 50.0, 1.01, 0.05, -600.0),
        "vwap_reclaim": _dummy_evaluation("vwap_reclaim", 200.0, 1.10, 0.30, -400.0),
    }
    stress_df = pd.DataFrame(
        [
            {"strategy_id": "paper_vwap_baseline", "role": "paper_baseline_reference", "pnl_nominal": 100.0, "pnl_slip_x2": 80.0, "pf_nominal": 1.02, "pf_slip_x2": 1.01, "sharpe_nominal": 0.10, "sharpe_slip_x2": 0.08, "dd_nominal": -500.0, "dd_slip_x2": -550.0, "delta_pnl_nominal_vs_slip_x2": -20.0, "pass_fail_cost_stress": True},
            {"strategy_id": "baseline_futures_adapted", "role": "realistic_baseline_reference", "pnl_nominal": 50.0, "pnl_slip_x2": 10.0, "pf_nominal": 1.01, "pf_slip_x2": 1.00, "sharpe_nominal": 0.05, "sharpe_slip_x2": 0.01, "dd_nominal": -600.0, "dd_slip_x2": -700.0, "delta_pnl_nominal_vs_slip_x2": -40.0, "pass_fail_cost_stress": False},
            {"strategy_id": "vwap_reclaim", "role": "candidate", "pnl_nominal": 200.0, "pnl_slip_x2": 120.0, "pf_nominal": 1.10, "pf_slip_x2": 1.05, "sharpe_nominal": 0.30, "sharpe_slip_x2": 0.20, "dd_nominal": -400.0, "dd_slip_x2": -450.0, "delta_pnl_nominal_vs_slip_x2": -80.0, "pass_fail_cost_stress": True},
        ]
    )
    split_df = pd.DataFrame(
        [
            {"strategy_id": "paper_vwap_baseline", "positive_oos_splits": 2, "total_splits": 4, "mean_oos_net_pnl": 100.0, "mean_oos_profit_factor": 1.02, "mean_oos_sharpe_ratio": 0.10, "worst_oos_split_net_pnl": -50.0, "best_oos_split_net_pnl": 120.0, "pass_fail_splits": True},
            {"strategy_id": "baseline_futures_adapted", "positive_oos_splits": 1, "total_splits": 4, "mean_oos_net_pnl": 20.0, "mean_oos_profit_factor": 1.00, "mean_oos_sharpe_ratio": 0.02, "worst_oos_split_net_pnl": -100.0, "best_oos_split_net_pnl": 60.0, "pass_fail_splits": False},
            {"strategy_id": "vwap_reclaim", "positive_oos_splits": 3, "total_splits": 4, "mean_oos_net_pnl": 150.0, "mean_oos_profit_factor": 1.08, "mean_oos_sharpe_ratio": 0.25, "worst_oos_split_net_pnl": -20.0, "best_oos_split_net_pnl": 180.0, "pass_fail_splits": True},
        ]
    )
    prop_df = pd.DataFrame(
        [
            {"strategy_id": "paper_vwap_baseline", "role": "paper_baseline_reference", "worst_daily_loss_usd": -200.0, "daily_loss_limit_breach_freq": 0.0, "trailing_drawdown_breach_freq": 0.0, "avg_trades_per_day": 2.0, "max_trades_per_day": 4, "challenge_success_rate_standard": 0.05, "prop_verdict": "potentiellement compatible sous contraintes prudentes"},
            {"strategy_id": "baseline_futures_adapted", "role": "realistic_baseline_reference", "worst_daily_loss_usd": -300.0, "daily_loss_limit_breach_freq": 0.05, "trailing_drawdown_breach_freq": 0.12, "avg_trades_per_day": 2.0, "max_trades_per_day": 4, "challenge_success_rate_standard": 0.02, "prop_verdict": "trop fragile"},
            {"strategy_id": "vwap_reclaim", "role": "candidate", "worst_daily_loss_usd": -150.0, "daily_loss_limit_breach_freq": 0.0, "trailing_drawdown_breach_freq": 0.0, "avg_trades_per_day": 1.5, "max_trades_per_day": 3, "challenge_success_rate_standard": 0.25, "prop_verdict": "prop-compatible"},
        ]
    )
    concentration_df = pd.DataFrame(
        [
            {"strategy_id": "paper_vwap_baseline", "role": "paper_baseline_reference", "top_3_day_contribution_pct": 0.30, "top_5_day_contribution_pct": 0.45, "pnl_excluding_top_3_days": 50.0, "pnl_excluding_top_5_days": 25.0, "concentration_verdict": "distribution saine"},
            {"strategy_id": "baseline_futures_adapted", "role": "realistic_baseline_reference", "top_3_day_contribution_pct": 0.60, "top_5_day_contribution_pct": 0.90, "pnl_excluding_top_3_days": -10.0, "pnl_excluding_top_5_days": -20.0, "concentration_verdict": "forte dependance aux meilleurs jours"},
            {"strategy_id": "vwap_reclaim", "role": "candidate", "top_3_day_contribution_pct": 0.30, "top_5_day_contribution_pct": 0.45, "pnl_excluding_top_3_days": 50.0, "pnl_excluding_top_5_days": 25.0, "concentration_verdict": "distribution saine"},
        ]
    )

    summary = _merge_reranking_tables(spec, evaluations, stress_df, split_df, prop_df, concentration_df)

    assert summary.iloc[0]["strategy_id"] == "paper_vwap_baseline"
    assert "baseline_futures_adapted" in summary["strategy_id"].tolist()
    reclaim_row = summary.loc[summary["strategy_id"] == "vwap_reclaim"].iloc[0]
    assert bool(reclaim_row["survives_primary_filter"]) is True
    assert reclaim_row["final_bucket"] == "candidat prioritaire pour validation complementaire"


def test_final_verdict_payload_recommends_abandon_when_no_candidate_survives() -> None:
    reranking_df = pd.DataFrame(
        [
            {"strategy_id": "paper_vwap_baseline", "role": "paper_baseline_reference", "oos_net_pnl": 100.0, "oos_profit_factor": 1.02, "pf_slip_x2": 1.01, "survives_primary_filter": False, "rank_within_survivors": pd.NA, "final_bucket": "reference_officielle", "elimination_reason": "ref", "prop_verdict": "potentiellement compatible sous contraintes prudentes", "concentration_verdict": "distribution saine"},
            {"strategy_id": "baseline_futures_adapted", "role": "realistic_baseline_reference", "oos_net_pnl": -10.0, "oos_profit_factor": 0.99, "pf_slip_x2": 0.95, "survives_primary_filter": False, "rank_within_survivors": pd.NA, "final_bucket": "baseline_realiste_de_reference", "elimination_reason": "ref", "prop_verdict": "trop fragile", "concentration_verdict": "forte dependance aux meilleurs jours"},
            {"strategy_id": "vwap_reclaim", "role": "candidate", "oos_net_pnl": 20.0, "oos_profit_factor": 1.01, "pf_slip_x2": 0.98, "survives_primary_filter": False, "rank_within_survivors": pd.NA, "final_bucket": "interessante mais trop fragile", "elimination_reason": "fragile", "prop_verdict": "trop fragile", "concentration_verdict": "distribution saine"},
        ]
    )

    verdict = _final_verdict_payload(reranking_df)

    assert verdict["survivor_count"] == 0
    assert verdict["top_candidate"] is None
    assert verdict["answers"]["at_least_one_candidate_deserves_deeper_validation"] is False
    assert "abandonner" in verdict["answers"]["recommended_next_action"]


def test_generate_reranking_notebook_emits_valid_code_cells_and_absolute_output_dir(tmp_path: Path) -> None:
    notebook_path = tmp_path / "reranking.ipynb"
    output_dir = tmp_path / "exports" / "reranking_run"

    generate_reranking_notebook(notebook_path=notebook_path, output_dir=output_dir)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all("outputs" in cell and cell["outputs"] == [] for cell in code_cells)
    assert all("execution_count" in cell and cell["execution_count"] is None for cell in code_cells)
    assert "OUTPUT_DIR = Path(r\"" + str(output_dir.resolve()) + "\")" in "".join(code_cells[1]["source"])
