from src.config.orb_campaign import build_focused_atr_regimes, build_focused_orb_experiments


def test_focused_campaign_grid_matches_requested_dimensions() -> None:
    experiments = build_focused_orb_experiments()

    assert len(experiments) == 3 * 4 * 3 * 4
    assert {experiment.side_mode for experiment in experiments} == {"long_only"}
    assert {experiment.direction_filter_mode for experiment in experiments} == {"ema_only"}
    assert {experiment.target_multiple for experiment in experiments} == {3.0, 4.0, 5.0}
    assert {experiment.ema_length for experiment in experiments} == {30, 50, 70, 100}
    assert {experiment.atr_regime for experiment in experiments} == {"none", "moderate_band", "restrictive_band"}
    assert {experiment.risk_per_trade_pct for experiment in experiments} == {0.10, 0.15, 0.20, 0.25}


def test_focused_atr_regimes_are_monotonic() -> None:
    regimes = build_focused_atr_regimes()

    assert regimes["none"].lower_quantile is None
    assert regimes["moderate_band"].lower_quantile == 0.50
    assert regimes["moderate_band"].upper_quantile == 1.0
    assert regimes["restrictive_band"].lower_quantile == 2.0 / 3.0
    assert regimes["restrictive_band"].upper_quantile == 1.0
