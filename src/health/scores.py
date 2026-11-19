"""Residual problem scores of every model of a farm (spec §7.10).

For each power, pitch and thermal model: residual = measured - expected on
normal rows (temperatures only while producing at least `min_power_kw`),
a rolling time-window mean, optionally made fleet-relative (minus the
median of the other turbines of the farm for the same model), then turned
so that larger is worse (`problem_score`). Shared by alerts.py and
curves.py.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.modelling import prediction_column

ModelKey = tuple[str, str, str]  # model class, target, variant


def smoothed_residual(
    frame: pd.DataFrame,
    target: str,
    column: str,
    window: str,
    min_samples: int,
    min_power: float = 0.0,
) -> pd.Series:
    """Rolling time-window mean of measured - expected over normal rows
    with both values (and power >= min_power when it is positive), indexed
    by timestamp; windows with fewer than `min_samples` rows dropped."""
    rows = frame["is_normal"] & frame[target].notna() & frame[column].notna()
    if min_power > 0:
        rows &= frame["power"] >= min_power
    residual = pd.Series(
        (frame.loc[rows, target] - frame.loc[rows, column]).to_numpy(),
        index=pd.DatetimeIndex(frame.loc[rows, "timestamp"]),
    )
    return residual.rolling(window, min_periods=min_samples).mean().dropna()


def problem_score(residual: pd.Series, direction: str) -> pd.Series:
    """Residual turned so that larger is worse: `low` (e.g. power below
    expected), `high` (temperature above) or `both` (either side)."""
    if direction == "low":
        return -residual
    if direction == "high":
        return residual
    if direction == "both":
        return residual.abs()
    raise ValueError(f"unknown alert direction {direction!r}")


def fleet_relative(
    residuals: Mapping[str, pd.Series], min_turbines: int
) -> dict[str, pd.Series]:
    """Each turbine's series minus the median of the other turbines' at the
    same timestamp; only where at least `min_turbines` others have a value."""
    wide = pd.DataFrame(dict(residuals))
    out = {}
    for tid in wide.columns:
        others = wide.drop(columns=tid)
        enough = others.notna().sum(axis=1) >= min_turbines
        reference = others.median(axis=1).where(enough)
        out[tid] = (wide[tid] - reference).dropna()
    return out


def residual_models(models: pd.DataFrame) -> pd.DataFrame:
    """The models that have a prediction column (all but yaw)."""
    return models[models["model_class"] != "YawModel"]


def needed_columns(models: pd.DataFrame) -> list[str]:
    """Parquet columns that the residual models of one turbine need."""
    residual = residual_models(models)
    columns = ["timestamp", "is_normal", "power", *residual["target"]]
    columns += [prediction_column(m.target, m.variant) for m in residual.itertuples()]
    return list(dict.fromkeys(columns))


def farm_scores(
    config: dict[str, Any],
    models: pd.DataFrame,
    paths: Mapping[str, Path],
    fleet: bool,
) -> dict[str, pd.Series]:
    """Problem score per model id of one farm (one turbine read at a time)."""
    settings = config["alerts"]
    residuals: dict[ModelKey, dict[str, pd.Series]] = {}
    ids: dict[tuple[ModelKey, str], str] = {}
    for tid, path in paths.items():
        mine = residual_models(models[models["turbine_id"] == tid])
        frame = pd.read_parquet(path, columns=needed_columns(mine))
        for model in mine.itertuples():
            key = (model.model_class, model.target, model.variant)
            residuals.setdefault(key, {})[tid] = smoothed_residual(
                frame,
                model.target,
                prediction_column(model.target, model.variant),
                settings["window"],
                settings["min_samples"],
                settings["min_power_kw"].get(model.model_class, 0.0),
            )
            ids[(key, tid)] = model.model.split("#")[-1]
    scores = {}
    for key, per_turbine in residuals.items():
        if fleet:
            per_turbine = fleet_relative(per_turbine, settings["fleet_min_turbines"])
        for tid, series in per_turbine.items():
            direction = settings["direction"][key[0]]
            scores[ids[(key, tid)]] = problem_score(series, direction)
    return scores
