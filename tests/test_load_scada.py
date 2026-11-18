"""Tests for the SCADA loader on small synthetic Greenbyte-style files."""

from pathlib import Path

import pandas as pd
import pytest

from src.load_scada import (
    all_signals,
    load_turbine,
    missing_percent,
    rows_per_year,
    tidy,
)

SIGNALS = {"wind_speed": "Wind speed (m/s)"}
COMPONENT_SIGNALS = {"gear_oil_temp": "Gear oil temperature (°C)"}
COLUMNS = ["Date and time", "Wind speed (m/s)", "Gear oil temperature (°C)"]
HEADER = (
    "# Exported by Greenbyte\r\n"
    '# Date and time,Wind speed (m/s),"Wind speed, Min (m/s)",'
    "Gear oil temperature (°C)\r\n"
)
FILE_2021 = HEADER + (
    "2020-12-31 23:50:00,1.0,0.5,40.0\r\n"
    "2021-12-31 23:50:00,4.0,3.0,41.0\r\n"
    "2022-01-01 00:00:00,5.0,4.0,42.0\r\n"
)
FILE_2022 = HEADER + (
    "2022-01-01 00:00:00,9.9,9.0,99.0\r\n"
    "2022-01-01 00:10:00,5.3,4.4,NaN\r\n"
    "2023-01-01 00:00:00,6.0,5.0,43.0\r\n"
)
CONFIG = {
    "scada": {"timestamp_column": "Date and time"},
    "signals": SIGNALS,
    "component_signals": COMPONENT_SIGNALS,
    "years": [2021, 2022],
}


@pytest.fixture
def yearly_files(tmp_path: Path) -> list[Path]:
    paths = []
    for year, text in [(2021, FILE_2021), (2022, FILE_2022)]:
        folder = tmp_path / f"scada_{year}"
        folder.mkdir()
        path = folder / f"Turbine_Data_Penmanshiel_01_{year}.csv"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


@pytest.fixture
def loaded(yearly_files: list[Path]) -> pd.DataFrame:
    return load_turbine("T01", yearly_files, CONFIG)


def test_all_signals_merges_core_and_component_signals() -> None:
    assert all_signals(CONFIG) == SIGNALS | COMPONENT_SIGNALS


def test_load_turbine_renames_to_config_keys_with_turbine_id(
    loaded: pd.DataFrame,
) -> None:
    assert (list(loaded.columns), set(loaded["turbine_id"])) == (
        ["timestamp", "turbine_id", "wind_speed", "gear_oil_temp"],
        {"T01"},
    )


def test_load_turbine_combines_years_and_drops_other_years(
    loaded: pd.DataFrame,
) -> None:
    assert list(loaded["timestamp"].astype(str)) == [
        "2021-12-31 23:50:00+00:00",
        "2022-01-01 00:00:00+00:00",
        "2022-01-01 00:10:00+00:00",
    ]


def test_load_turbine_keeps_first_file_row_at_year_boundary(
    loaded: pd.DataFrame,
) -> None:
    boundary = loaded["timestamp"] == pd.Timestamp("2022-01-01 00:00", tz="UTC")

    assert loaded.loc[boundary, "gear_oil_temp"].tolist() == [42.0]


def test_load_turbine_parses_nan_as_missing(loaded: pd.DataFrame) -> None:
    assert loaded["gear_oil_temp"].isna().tolist() == [False, False, True]


def test_rows_per_year_counts_by_timestamp_year(loaded: pd.DataFrame) -> None:
    assert rows_per_year(loaded) == {2021: 1, 2022: 2}


def test_tidy_prefers_complete_row_over_earlier_nan_copy() -> None:
    raw = pd.DataFrame(
        {
            COLUMNS[0]: ["2023-01-01 00:00:00"] * 3,
            COLUMNS[1]: [None, 5.0, 6.0],
            COLUMNS[2]: [None, None, 30.0],
        }
    )

    frame = tidy(raw, COLUMNS[0], SIGNALS | COMPONENT_SIGNALS, "T01", [2023])

    assert frame[["wind_speed", "gear_oil_temp"]].values.tolist() == [[6.0, 30.0]]


def test_missing_percent_per_column() -> None:
    frame = pd.DataFrame({"a": [1.0, None, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})

    assert missing_percent(frame).to_dict() == {"a": 50.0, "b": 0.0}
