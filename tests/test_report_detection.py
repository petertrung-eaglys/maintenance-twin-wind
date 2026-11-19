"""Tests for the report's detection-vs-chance helpers."""

import pandas as pd
import pytest

from src.report.detection import detection_points, mean_lift


def curves() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["component", "component", "signal", "component"],
            "fleet": [True, True, True, False],
            "episodes": [10, 2, 10, 10],
            "detection_rate": [0.5, 1.0, 0.5, 0.5],
            "chance_detection_rate": [0.3, 0.0, 0.3, 0.3],
        }
    )


def test_detection_points_keep_one_scope_scoring_and_enough_episodes() -> None:
    points = detection_points(curves(), "component", True, min_episodes=5)

    assert points.index.tolist() == [0]


def test_mean_lift_is_detection_minus_chance() -> None:
    frame = pd.DataFrame(
        {"detection_rate": [0.5, 0.1], "chance_detection_rate": [0.3, 0.2]}
    )

    assert mean_lift(frame) == pytest.approx(0.05)
