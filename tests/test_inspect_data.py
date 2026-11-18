"""Tests for the raw-file inspection helpers."""

from pathlib import Path

import pandas as pd

from src.data.inspect_data import (
    event_counts,
    files_missing_count,
    filled_fraction,
    missing_columns,
    read_table,
)


def test_missing_columns_lists_absent_wanted_columns_in_order() -> None:
    columns = ["Date and time", "Power (kW)"]

    assert missing_columns(columns, ["Wind (m/s)", "Power (kW)", "Pitch"]) == [
        "Wind (m/s)",
        "Pitch",
    ]


def test_files_missing_count_counts_files_lacking_each_column() -> None:
    columns_by_file = {"a.csv": ["t", "p"], "b.csv": ["t"], "c.csv": []}

    assert files_missing_count(columns_by_file, ["t", "p", "x"]) == {
        "t": 1,
        "p": 2,
        "x": 3,
    }


def test_filled_fraction_counts_timestamps_filled_in_any_copy() -> None:
    nan = float("nan")
    frame = pd.DataFrame(
        {
            "t": ["00:00", "00:00", "00:10", "00:20"],
            "a": [nan, 1.0, 2.0, nan],  # 00:00 filled by its second copy
            "b": [nan, nan, nan, 3.0],
        }
    )

    assert filled_fraction(frame, "t").to_dict() == {"a": 2 / 3, "b": 1 / 3}


def event(turbine: str, start: str, status: str, message: str) -> dict:
    return {
        "turbine": turbine,
        "Timestamp start": start,
        "Status": status,
        "Code": "1",
        "Message": message,
    }


def test_event_counts_dedupes_filters_and_sorts() -> None:
    events = pd.DataFrame(
        [
            event("T01", "s1", "Stop", "Pitch"),
            event("T01", "s1", "Stop", "Pitch"),  # repeated export row
            event("T02", "s1", "Stop", "Pitch"),
            event("T01", "s2", "Warning", "Gear"),
            event("T01", "s3", "Informational", "Run"),
        ]
    )

    counts = event_counts(events, ["Stop", "Warning"])

    assert counts.values.tolist() == [["Stop", "Pitch", 2], ["Warning", "Gear", 1]]


def test_read_table_reads_csv_with_bom_and_xlsx(tmp_path: Path) -> None:
    frame = pd.DataFrame({"Title": ["T01"], "Rotor": [82]})
    csv_path, xlsx_path = tmp_path / "s.csv", tmp_path / "m.xlsx"
    csv_path.write_bytes("﻿Title,Rotor\nT01,82\n".encode())
    frame.to_excel(xlsx_path, index=False)

    assert [read_table(p).to_dict("list") for p in (csv_path, xlsx_path)] == [
        frame.to_dict("list")
    ] * 2
