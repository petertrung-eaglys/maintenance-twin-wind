"""Tests for the raw-file inspection helpers."""

from pathlib import Path

import pandas as pd

from src.inspect_data import (
    event_counts,
    file_columns,
    files_missing_count,
    filled_fraction,
    missing_columns,
    read_signals,
    read_status,
    read_table,
)

STATUS_FILE = (
    "# Exported by Greenbyte\r\n"
    "# Turbine: Penmanshiel 02\r\n"
    "Timestamp start,Timestamp end,Duration,Status,Code,Message,\r\n"
    "2022-01-01 00:00:00,2022-01-01 01:00:00,01:00:00,Stop,10,Low oil,\r\n"
    "2022-01-02 00:00:00,-,-,Informational,1,Run,\r\n"
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


def test_file_columns_reads_commented_header(tmp_path: Path) -> None:
    path = tmp_path / "Turbine_Data_K_1_x.csv"
    path.write_bytes(b"# meta\r\n# Date and time,Power (kW)\r\n2022,1\r\n")

    assert file_columns(path) == ["Date and time", "Power (kW)"]


def test_read_signals_reads_only_given_columns(tmp_path: Path) -> None:
    path = tmp_path / "Turbine_Data_K_1_x.csv"
    path.write_bytes(b"# meta\r\n# t,a,b\r\n1,2,3\r\n4,5,6\r\n")

    assert read_signals(path, ["t", "b"]).to_dict("list") == {"t": [1, 4], "b": [3, 6]}


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


def test_read_status_parses_plain_header_and_adds_turbine(tmp_path: Path) -> None:
    path = tmp_path / "Status_Penmanshiel_02_x.csv"
    path.write_bytes(STATUS_FILE.encode())

    frame = read_status(path, "Penmanshiel")

    assert frame[["Status", "Message", "turbine"]].values.tolist() == [
        ["Stop", "Low oil", "T02"],
        ["Informational", "Run", "T02"],
    ]


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
