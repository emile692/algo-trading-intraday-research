"""Helpers to add the audited VVIX filter as a notebook comparison overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import EXPORTS_DIR
from src.features.implied_volatility import (
    DEFAULT_VIX_DAILY_PATH,
    DEFAULT_VVIX_DAILY_PATH,
    load_vix_vvix_daily_features,
)


DEFAULT_EXPORT_PREFIX = "mnq_orb_vix_vvix_validation"
DEFAULT_VARIANT_NAME = "filter_drop_low__vvix_pct_63_t1"


@dataclass(frozen=True)
class VvixOverlaySpec:
    export_root: Path
    variant_name: str
    feature_name: str
    kept_buckets: tuple[str, ...]
    bucket_rows: pd.DataFrame


def find_latest_export(prefix: str = DEFAULT_EXPORT_PREFIX, exports_root: Path = EXPORTS_DIR) -> Path:
    candidates = [path for path in exports_root.glob(f"{prefix}_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No export folder found for prefix {prefix!r} under {exports_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_kept_buckets(raw: object) -> tuple[str, ...]:
    if raw is None or pd.isna(raw):
        return ()
    return tuple(label.strip() for label in str(raw).split(",") if label and label.strip())


def resolve_vvix_overlay_spec(
    export_root: Path | None = None,
    variant_name: str = DEFAULT_VARIANT_NAME,
    feature_name: str | None = None,
    kept_buckets_override: tuple[str, ...] = (),
) -> VvixOverlaySpec:
    resolved_root = export_root if export_root is not None else find_latest_export()
    if not resolved_root.exists():
        raise FileNotFoundError(f"VVIX validation export root not found: {resolved_root}")

    validation_summary = pd.read_csv(resolved_root / "validation_summary.csv")
    regime_summary = pd.read_csv(resolved_root / "regime_summary.csv")

    variant_rows = validation_summary.loc[validation_summary["variant_name"].astype(str).eq(str(variant_name))].copy()
    if variant_rows.empty:
        raise ValueError(f"Variant {variant_name!r} not found in {resolved_root / 'validation_summary.csv'}.")
    variant_row = variant_rows.iloc[0]

    resolved_feature_name = str(feature_name or variant_row.get("feature_name") or "")
    if not resolved_feature_name:
        raise ValueError(f"Could not resolve feature name for variant {variant_name!r}.")

    kept_buckets = tuple(str(value) for value in kept_buckets_override) if kept_buckets_override else _parse_kept_buckets(
        variant_row.get("kept_buckets")
    )
    if not kept_buckets:
        raise ValueError(f"Could not resolve kept buckets for variant {variant_name!r}.")

    bucket_rows = (
        regime_summary.loc[regime_summary["feature_name"].astype(str).eq(resolved_feature_name)]
        .sort_values("bucket_position")
        .reset_index(drop=True)
    )
    if bucket_rows.empty:
        raise ValueError(
            f"Feature {resolved_feature_name!r} not found in {resolved_root / 'regime_summary.csv'}."
        )

    return VvixOverlaySpec(
        export_root=resolved_root,
        variant_name=str(variant_name),
        feature_name=resolved_feature_name,
        kept_buckets=kept_buckets,
        bucket_rows=bucket_rows,
    )


def assign_bucket_labels_from_export(values: pd.Series, bucket_rows: pd.DataFrame) -> pd.Series:
    ordered = bucket_rows.sort_values("bucket_position").reset_index(drop=True)
    labels = ordered["bucket_label"].astype(str).tolist()
    if not labels:
        return pd.Series(pd.NA, index=values.index, dtype="string")

    bucket_kind = str(ordered.iloc[0].get("bucket_kind", "quantile"))
    if bucket_kind == "categorical":
        clean = pd.Series(values, index=values.index, dtype="object").astype("string")
        return clean.where(clean.isin(set(labels))).astype("string")

    if len(labels) == 1:
        out = pd.Series(pd.NA, index=values.index, dtype="string")
        numeric = pd.to_numeric(values, errors="coerce")
        out.loc[numeric.notna()] = labels[0]
        return out

    upper_bounds = pd.to_numeric(ordered["upper_bound"], errors="coerce").tolist()
    bins = [-np.inf]
    for bound in upper_bounds[:-1]:
        bins.append(float(bound))
    bins.append(np.inf)

    bucketed = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return pd.Series(bucketed, index=values.index, dtype="string")


def build_vvix_filter_controls(
    session_dates: list | tuple | pd.Series | pd.Index,
    export_root: Path | None = None,
    variant_name: str = DEFAULT_VARIANT_NAME,
    feature_name: str | None = None,
    kept_buckets_override: tuple[str, ...] = (),
    vix_path: Path = DEFAULT_VIX_DAILY_PATH,
    vvix_path: Path = DEFAULT_VVIX_DAILY_PATH,
) -> tuple[VvixOverlaySpec, pd.DataFrame]:
    spec = resolve_vvix_overlay_spec(
        export_root=export_root,
        variant_name=variant_name,
        feature_name=feature_name,
        kept_buckets_override=kept_buckets_override,
    )

    sessions = pd.Index(pd.to_datetime(pd.Index(session_dates), errors="coerce").date).dropna().unique().tolist()
    controls = pd.DataFrame({"session_date": sessions})

    daily_features = load_vix_vvix_daily_features(vix_path=vix_path, vvix_path=vvix_path).copy()
    daily_features["session_date"] = pd.to_datetime(daily_features["session_date"], errors="coerce").dt.date

    controls = controls.merge(
        daily_features[["session_date", spec.feature_name]],
        on="session_date",
        how="left",
        validate="one_to_one",
    )
    controls["bucket_label"] = assign_bucket_labels_from_export(controls[spec.feature_name], spec.bucket_rows)
    controls["selected"] = controls["bucket_label"].isin(set(spec.kept_buckets))
    controls["skip_trade"] = ~controls["selected"]
    controls["feature_name"] = spec.feature_name
    controls["kept_buckets"] = ",".join(spec.kept_buckets)
    return spec, controls.sort_values("session_date").reset_index(drop=True)
