"""Tests for the shared modelling conventions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdflib import Graph

from src.modelling import (
    add_direction_features,
    evaluate,
    merge_metrics,
    prediction_column,
    split_masks,
    turbine_paths,
)


def test_direction_features_on_unit_circle() -> None:
    frame = add_direction_features(pd.DataFrame({"wind_direction": [0.0, 90.0]}))

    assert frame[["wind_dir_sin", "wind_dir_cos"]].round(6).values.tolist() == [
        [0.0, 1.0],
        [1.0, 0.0],
    ]


def test_split_masks_use_train_end_normal_and_required_columns() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2021-12-31 23:50",
                    "2021-12-31 23:40",
                    "2022-01-01 00:00",
                    "2022-01-01 00:10",
                ],
                utc=True,
            ),
            "is_normal": [True, True, True, False],
            "x": [1.0, np.nan, 2.0, 3.0],
        }
    )

    train, test = split_masks(frame, "2021-12-31 23:50", ["x"])

    assert (train.tolist(), test.tolist()) == (
        [True, False, False, False],
        [False, False, True, False],
    )


def test_evaluate_rmse_mae_and_normalised_rmse() -> None:
    metrics = evaluate(pd.Series([0.0, 0.0]), np.array([30.0, -40.0]), 1000.0)

    assert metrics == pytest.approx(
        {"rmse": np.sqrt(1250.0), "mae": 35.0, "nrmse": np.sqrt(1250.0) / 1000.0}
    )


def test_prediction_column_names_target_and_variant() -> None:
    assert prediction_column("power", "baseline") == "expected_power_baseline"


def test_merge_metrics_keeps_other_models(tmp_path: Path) -> None:
    path = tmp_path / "models" / "metrics.json"
    merge_metrics(path, {"A": {"rmse": 1.0}, "B": {"rmse": 2.0}})

    merge_metrics(path, {"B": {"rmse": 3.0}})

    assert json.loads(path.read_text()) == {"A": {"rmse": 1.0}, "B": {"rmse": 3.0}}


def test_turbine_paths_come_from_the_graph(graph: Graph) -> None:
    assert turbine_paths(graph, "KLM") == {
        "T01": Path("data/processed/kelmarsh/T01.parquet"),
        "T02": Path("data/processed/kelmarsh/T02.parquet"),
    }
