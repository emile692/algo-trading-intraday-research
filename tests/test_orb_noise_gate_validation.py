from src.analytics.orb_research.noise_gate_validation import build_noise_gate_grid_experiments
from src.analytics.orb_research.types import BaselineEnsembleConfig, BaselineEntryConfig


def test_noise_gate_grid_expected_size_and_modes() -> None:
    baseline_entry = BaselineEntryConfig()
    baseline_ensemble = BaselineEnsembleConfig()

    grid = build_noise_gate_grid_experiments(
        baseline_entry=baseline_entry,
        baseline_ensemble=baseline_ensemble,
    )

    # 3 lookbacks * 3 vm * (1 max-style + 3 k-style) * 2 confirm * 2 schedule
    assert len(grid) == 144
    assert {exp.stage for exp in grid} == {"grid"}
    assert {exp.family for exp in grid} == {"noise_gate"}


def test_noise_gate_grid_preserves_baseline_components() -> None:
    baseline_entry = BaselineEntryConfig()
    baseline_ensemble = BaselineEnsembleConfig()

    grid = build_noise_gate_grid_experiments(
        baseline_entry=baseline_entry,
        baseline_ensemble=baseline_ensemble,
    )

    assert all(exp.exit.mode == "baseline" for exp in grid)
    assert all(exp.compression.mode == "none" for exp in grid)
    assert all(exp.compression.usage == "hard_filter" for exp in grid)
    assert all(exp.baseline_ensemble == baseline_ensemble for exp in grid)
