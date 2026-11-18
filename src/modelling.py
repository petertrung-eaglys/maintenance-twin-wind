"""Shared modelling conventions (spec §7.6) for every model variant.

- Train on normal rows up to `train_end`, test on normal rows after it.
- Every variant implements `NormalBehaviourModel`: `fit` takes a mapping
  turbine id -> DataFrame, so data is never pooled out of sight of which
  turbine it came from (the federated-learning hook, spec §10).
- Metrics per turbine and variant go to models/metrics.json.
"""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd
from rdflib import Graph

from src.graph_utils import (
    add_model,
    get_sensors,
    get_turbines,
    local_name,
    model_iri,
    parquet_path_of,
)


class NormalBehaviourModel(Protocol):
    """What every model variant implements."""

    def fit(self, train_frames: Mapping[str, pd.DataFrame]) -> Self:
        """Fit on healthy training rows, one frame per turbine id."""
        ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected value of the target for every row of one turbine."""
        ...


def add_direction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add sin and cos of the wind direction (degrees)."""
    radians = np.deg2rad(frame["wind_direction"])
    return frame.assign(wind_dir_sin=np.sin(radians), wind_dir_cos=np.cos(radians))


def split_masks(
    frame: pd.DataFrame, train_end: str, required: list[str]
) -> tuple[pd.Series, pd.Series]:
    """Train and test masks: normal rows with all `required` columns present,
    up to and including train_end, and after it."""
    usable = frame["is_normal"] & frame[required].notna().all(axis=1)
    before = frame["timestamp"] <= pd.Timestamp(train_end, tz="UTC")
    return usable & before, usable & ~before


def evaluate(y_true: pd.Series, y_pred: np.ndarray, scale: float) -> dict[str, float]:
    """RMSE, MAE (target unit) and RMSE / scale."""
    error = np.asarray(y_pred) - np.asarray(y_true)
    rmse = float(np.sqrt(np.mean(error**2)))
    return {"rmse": rmse, "mae": float(np.mean(np.abs(error))), "nrmse": rmse / scale}


def prediction_column(target: str, variant: str) -> str:
    """Parquet column holding a variant's expected target, e.g.
    expected_power_baseline."""
    return f"expected_{target}_{variant}"


def turbine_paths(g: Graph, farm_code: str) -> dict[str, Path]:
    """Turbine id -> Parquet path of every turbine of a farm, via the graph."""
    turbines = get_turbines(g, farm_code)["turbine_id"]
    return {t: parquet_path_of(get_sensors(g, farm_code, t)) for t in turbines}


def with_turbine_specs(frame: pd.DataFrame, spec: pd.Series) -> pd.DataFrame:
    """Add the turbine's rotor diameter and rated power as constant columns,
    so physics-based models can read them from the frame they predict."""
    return frame.assign(
        rotor_diameter_m=spec["rotor_diameter_m"],
        rated_power_kw=spec["rated_power_kw"],
    )


def model_id_of(spec: dict[str, Any], farm_code: str, turbine_id: str) -> str:
    """Model id (IRI local name), e.g. PowerModel_KLM_T01_baseline."""
    iri = model_iri(spec["model_class"], farm_code, turbine_id, spec["variant"])
    return local_name(iri)


def metrics_entry(
    spec: dict[str, Any], farm_code: str, turbine_id: str, metrics: dict
) -> dict[str, Any]:
    """One models/metrics.json record."""
    keys = ("component", "variant", "target")
    return (
        {"farm_code": farm_code, "turbine_id": turbine_id}
        | {k: spec[k] for k in keys}
        | metrics
    )


def register_model(
    g: Graph,
    spec: dict[str, Any],
    farm_code: str,
    turbine_id: str,
    metrics: dict[str, float],
    model_file: Path,
) -> tuple[str, dict[str, Any]]:
    """Add the model node to the graph; return its id and metrics record."""
    add_model(
        g,
        spec["model_class"],
        farm_code,
        turbine_id,
        spec["component"],
        spec["variant"],
        spec["inputs"],
        spec["target"],
        metrics,
        datetime.now(UTC),
        str(model_file),
    )
    entry = metrics_entry(spec, farm_code, turbine_id, metrics)
    return model_id_of(spec, farm_code, turbine_id), entry


def merge_metrics(path: Path, entries: dict[str, dict[str, Any]]) -> dict:
    """Merge entries (keyed by model id) into the metrics JSON file."""
    existing = json.loads(path.read_text()) if path.exists() else {}
    merged = existing | entries
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    return merged
