from __future__ import annotations

import pandas as pd
import pytest

from src.features.volatility import add_rolling_std


def test_add_rolling_std_is_trailing_and_has_no_lookahead() -> None:
    frame = pd.DataFrame({"close": [100.0, 101.0, 103.0, 102.0, 250.0]})

    base = add_rolling_std(frame, window=3)

    mutated = frame.copy()
    mutated.loc[4, "close"] = 10_000.0
    mutated_out = add_rolling_std(mutated, window=3)

    pd.testing.assert_series_equal(
        base.loc[:3, "vol_std_3"],
        mutated_out.loc[:3, "vol_std_3"],
        check_names=False,
    )

    expected = frame["close"].pct_change().iloc[1:4].std()
    assert base.loc[3, "vol_std_3"] == pytest.approx(expected)
