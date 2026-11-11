from research.mgc_volume_climax_audit_20261007.run_audit import REP, grid


def test_inherited_grid_is_exact_and_canonical_is_central():
    rows = grid()
    assert len(rows) == 18
    assert {p["move"] for p in rows} == {1.0, 1.5, 2.0}
    assert {p["rvol"] for p in rows} == {1.5, 2.0, 2.5}
    assert {p["mode"] for p in rows} == {"immediate_fade", "confirmed_fade"}
    assert REP["move"] == 1.5 and REP["rvol"] == 2.0 and REP["mode"] == "confirmed_fade"
