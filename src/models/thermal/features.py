"""Shared inputs of the thermal models: history features, turbine keys,
row filters and per-target model specs."""

from collections.abc import Callable
from typing import Any

import pandas as pd


def turbine_key(farm_code: str, turbine_id: str) -> str:
    """Turbine id unique across farms, e.g. KLM_T01 (models shared by both
    farms are keyed by it)."""
    return f"{farm_code}_{turbine_id}"


def history_columns(inputs: list[str], windows_min: list[int]) -> list[str]:
    """Names of the rolling-mean columns, e.g. power_mean_60min."""
    return [f"{c}_mean_{w}min" for c in inputs for w in windows_min]


def feature_columns(thermal: dict[str, Any]) -> list[str]:
    """The current inputs followed by their rolling means."""
    inputs = thermal["history_inputs"]
    return [*inputs, *history_columns(inputs, thermal["history_windows_min"])]


def add_history_features(
    frame: pd.DataFrame, inputs: list[str], windows_min: list[int]
) -> pd.DataFrame:
    """Add time-window rolling means of the inputs (rows within the last
    `w` minutes, gaps respected). `frame` must be sorted by timestamp."""
    indexed = frame.set_index("timestamp")[inputs]
    columns = {
        f"{c}_mean_{w}min": indexed[c]
        .rolling(f"{w}min", min_periods=1)
        .mean()
        .to_numpy()
        for c in inputs
        for w in windows_min
    }
    return frame.assign(**columns)


def normal_target_rows(target: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Row filter: normal rows where the target is measured."""

    def rows(frame: pd.DataFrame) -> pd.Series:
        return frame["is_normal"] & frame[target].notna()

    return rows


def target_spec(spec: dict[str, Any], target: str, component: str) -> dict[str, Any]:
    """A thermal model spec for one target: the target is also the IRI
    qualifier, as one component has several temperatures."""
    return spec | {"target": target, "component": component, "qualifier": target}
