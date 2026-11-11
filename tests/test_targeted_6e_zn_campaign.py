from datetime import date

import pandas as pd

from research.targeted_6e_zn_campaign_20260926.run_targeted_campaign import (
    BERLIN,
    LONDON,
    execution_price,
    local_stamp,
    make_trade,
)


def test_dst_conversion_uses_each_region_civil_clock():
    # During the March 2024 US/EU DST mismatch, 08:00 Berlin is 03:00 NY;
    # after Europe changes clock it is 02:00 NY.
    assert str(local_stamp(date(2024, 3, 20), "08:00:00", BERLIN).time()) == "03:00:00"
    assert str(local_stamp(date(2024, 4, 3), "08:00:00", BERLIN).time()) == "02:00:00"
    assert str(local_stamp(date(2024, 3, 20), "16:00:00", LONDON).time()) == "12:00:00"


def test_execution_slippage_is_adverse_on_all_four_sides():
    tick = 0.25
    assert execution_price(100, 1, True, tick, 1) == 100.25
    assert execution_price(100, 1, False, tick, 1) == 99.75
    assert execution_price(100, -1, True, tick, 1) == 99.75
    assert execution_price(100, -1, False, tick, 1) == 100.25


def test_round_trip_applies_two_sides_commission_and_slippage():
    idx = pd.DatetimeIndex(["2024-01-02 08:00", "2024-01-02 09:00"], tz="America/New_York")
    data = pd.DataFrame({"open": [100.0, 101.0], "trade_date": [date(2024, 1, 2)] * 2}, index=idx)
    spec = {"tick_size": 0.25, "point_value_usd": 10.0, "commission_per_side_usd": 1.25}
    trade = make_trade("X", "f", "canonical", "leg", 1, idx[0], idx[1], data, spec)
    assert trade["gross_pnl"] == 10.0
    assert trade["net_pnl"] == 2.5  # gross 10 - slip 5 - commissions 2.5
