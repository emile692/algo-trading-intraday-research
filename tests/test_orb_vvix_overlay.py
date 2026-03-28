from __future__ import annotations

import pandas as pd

from src.analytics.orb_vvix_overlay import (
    assign_bucket_labels_from_export,
    resolve_vvix_overlay_spec,
)


def test_assign_bucket_labels_from_export_uses_bucket_bounds() -> None:
    bucket_rows = pd.DataFrame(
        [
            {"bucket_label": "low", "bucket_position": 1, "bucket_kind": "quantile", "upper_bound": 0.25},
            {"bucket_label": "mid", "bucket_position": 2, "bucket_kind": "quantile", "upper_bound": 0.75},
            {"bucket_label": "high", "bucket_position": 3, "bucket_kind": "quantile", "upper_bound": 1.0},
        ]
    )
    values = pd.Series([0.10, 0.40, 0.90, None])

    labels = assign_bucket_labels_from_export(values, bucket_rows)

    assert labels.astype("string").tolist() == ["low", "mid", "high", pd.NA]


def test_resolve_vvix_overlay_spec_reads_variant_metadata(tmp_path) -> None:
    export_root = tmp_path / "mnq_orb_vix_vvix_validation_20260327_run"
    export_root.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "variant_name": "filter_drop_low__vvix_pct_63_t1",
                "feature_name": "vvix_pct_63_t1",
                "kept_buckets": "mid,high",
            }
        ]
    ).to_csv(export_root / "validation_summary.csv", index=False)

    pd.DataFrame(
        [
            {"feature_name": "vvix_pct_63_t1", "bucket_label": "low", "bucket_position": 1, "bucket_kind": "quantile", "lower_bound": 0.0, "upper_bound": 0.25},
            {"feature_name": "vvix_pct_63_t1", "bucket_label": "mid", "bucket_position": 2, "bucket_kind": "quantile", "lower_bound": 0.25, "upper_bound": 0.75},
            {"feature_name": "vvix_pct_63_t1", "bucket_label": "high", "bucket_position": 3, "bucket_kind": "quantile", "lower_bound": 0.75, "upper_bound": 1.0},
        ]
    ).to_csv(export_root / "regime_summary.csv", index=False)

    spec = resolve_vvix_overlay_spec(export_root=export_root)

    assert spec.variant_name == "filter_drop_low__vvix_pct_63_t1"
    assert spec.feature_name == "vvix_pct_63_t1"
    assert spec.kept_buckets == ("mid", "high")
    assert spec.bucket_rows["bucket_label"].tolist() == ["low", "mid", "high"]
