"""Tests for the cross-site comparison helpers."""

import numpy as np
import pandas as pd
import pytest

from src.experiments.cross_site import compare, load_part, save_part, summarise


def test_compare_scores_in_site_and_cross_site_on_the_same_rows() -> None:
    measured = pd.Series([10.0, 10.0, 10.0, 10.0])
    rows = pd.Series([True, True, True, False])
    in_site = {"physics": np.array([11.0, 9.0, np.nan, 0.0])}
    cross = {"physics": np.array([12.0, 8.0, 10.0, 0.0])}

    records = compare(measured, rows, in_site, cross)

    # row 2 lacks an in-site prediction and row 3 is not a test row
    assert records == [
        {"variant": "physics", "rmse_in": 1.0, "rmse_cross": 2.0, "n": 2}
    ]


def test_compare_uses_rows_where_every_variant_predicts() -> None:
    measured = pd.Series([10.0, 10.0])
    rows = pd.Series([True, True])
    in_site = {"a": np.array([10.0, 10.0]), "b": np.array([10.0, np.nan])}
    cross = {"a": np.array([11.0, 13.0]), "b": np.array([11.0, 11.0])}

    records = compare(measured, rows, in_site, cross)

    assert [r["n"] for r in records] == [1, 1]


def test_summarise_reports_the_increase_per_direction_signal_and_variant() -> None:
    records = pd.DataFrame(
        {
            "train": "KLM",
            "test": "PEN",
            "signal": "power",
            "variant": ["baseline", "baseline", "physics"],
            "rmse_in": [50.0, 70.0, 60.0],
            "rmse_cross": [80.0, 100.0, 66.0],
        }
    )

    summary = summarise(records).set_index("variant")

    assert summary.loc["baseline", ["rmse_in", "rmse_cross", "increase"]].tolist() == [
        60.0,
        90.0,
        pytest.approx(0.5),
    ]


def test_saved_part_is_loaded_back_and_missing_part_is_none(tmp_path) -> None:
    records = [{"variant": "physics", "rmse_in": 1.0, "rmse_cross": 2.0, "n": 2}]

    save_part(tmp_path / "parts", "power", records)

    assert load_part(tmp_path / "parts", "power") == records
    assert load_part(tmp_path / "parts", "stator_temp") is None
    assert not list((tmp_path / "parts").glob("*.tmp"))
