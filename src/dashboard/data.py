"""Data access for the dashboard: only the columns a view needs, one
turbine at a time (a full turbine file is ~200 MB in memory)."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.dashboard.health import compact
from src.graph.graph_utils import local_name
from src.health.scores import farm_scores, residual_models
from src.models.modelling import prediction_column


def with_model_ids(models: pd.DataFrame) -> pd.DataFrame:
    """Models with their id (the IRI's local name, as in alerts.json)."""
    return models.assign(model_id=models["model"].map(local_name))


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """The listed columns that exist in a Parquet file (others skipped),
    sorted by timestamp."""
    present = set(pq.read_schema(path).names)
    frame = pd.read_parquet(path, columns=[c for c in columns if c in present])
    return frame.sort_values("timestamp", ignore_index=True)


def farm_normal(paths: Mapping[str, Path]) -> pd.DataFrame:
    """timestamp, turbine_id and is_normal of every turbine of a farm."""
    return pd.concat(
        [
            read_columns(path, ["timestamp", "is_normal"]).assign(turbine_id=tid)
            for tid, path in paths.items()
        ],
        ignore_index=True,
    )


def target_columns(target: str, variants: list[str]) -> list[str]:
    """A signal and its expected values for the given variants."""
    return [target, *(prediction_column(target, v) for v in variants)]


def detail_columns(
    power_variants: list[str], target: str | None, thermal_variants: list[str]
) -> list[str]:
    """Columns of the replay charts: power, wind and one temperature."""
    columns = ["timestamp", "is_normal", "wind_speed"]
    columns += target_columns("power", power_variants)
    if target is not None:
        columns += target_columns(target, thermal_variants)
    return columns


def component_scores(
    config: dict[str, Any], models: pd.DataFrame, paths: Mapping[str, Path]
) -> dict[str, pd.Series]:
    """Problem score per model id (`models`: one component of the whole
    farm), fleet-relative as configured, kept at the dashboard's
    resolution."""
    residual = residual_models(models)
    scores = farm_scores(config, residual, paths, config["alerts"]["fleet_relative"])
    rule = config["dashboard"]["health_resample"]
    return {model_id: compact(score, rule) for model_id, score in scores.items()}
