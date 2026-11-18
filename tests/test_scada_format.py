"""Tests for the SCADA export format helpers."""

from pathlib import Path

import pytest

from src.data.scada_format import (
    classify_file,
    file_columns,
    format_turbine_id,
    group_by_turbine,
    group_files,
    header_line_index,
    parse_columns,
    read_head,
    read_signals,
    read_status,
    status_files,
    status_header_index,
    turbine_data_files,
    turbine_id_from_filename,
)

TURBINE_SAMPLE = [
    "# Exported by Greenbyte",
    "# Turbine: Kelmarsh 1",
    '# Date and time,Wind speed (m/s),"Wind speed, Minimum (m/s)",Power (kW)',
    "2022-01-01 00:00:00,5.1,4.2,300.0",
]

STATUS_SAMPLE = [
    "# Exported by Greenbyte",
    "# Turbine: Kelmarsh 1",
    "Timestamp start,Timestamp end,Duration,Status,Code,Message",
    "2022-01-01 00:00:00,-,-,Informational,1,Run",
]


def test_classify_file_by_prefix() -> None:
    names = [
        "Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv",
        "Status_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv",
        "readme.txt",
    ]

    assert [classify_file(n) for n in names] == ["turbine_data", "status", "other"]


def test_group_files_sorts_within_groups() -> None:
    names = ["Turbine_Data_K_2_x.csv", "Status_K_1_x.csv", "Turbine_Data_K_1_x.csv"]

    assert group_files(names) == {
        "status": ["Status_K_1_x.csv"],
        "turbine_data": ["Turbine_Data_K_1_x.csv", "Turbine_Data_K_2_x.csv"],
    }


def test_read_head_strips_crlf_and_bom(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_bytes("﻿a,b\r\n1,2\r\n3,4\r\n".encode())

    assert read_head(path, 2) == ["a,b", "1,2"]


def test_header_line_index_is_last_comment_line() -> None:
    assert header_line_index(TURBINE_SAMPLE) == 2


def test_header_line_index_without_comments_is_minus_one() -> None:
    assert header_line_index(["2022-01-01,1"]) == -1


def test_status_header_index_is_first_plain_line() -> None:
    assert status_header_index(STATUS_SAMPLE) == 2


def test_parse_columns_strips_comment_and_respects_quotes() -> None:
    assert parse_columns(TURBINE_SAMPLE[2]) == [
        "Date and time",
        "Wind speed (m/s)",
        "Wind speed, Minimum (m/s)",
        "Power (kW)",
    ]


def test_format_turbine_id_zero_pads() -> None:
    assert [format_turbine_id(n) for n in (1, 6, 12)] == ["T01", "T06", "T12"]


@pytest.mark.parametrize(
    ("name", "scada_name", "expected"),
    [
        ("Turbine_Data_Kelmarsh_3_2022-01-01_-_2023-01-01_230.csv", "Kelmarsh", "T03"),
        (
            "Status_Penmanshiel_01_2023-02-01_-_2023-03-01_1042.csv",
            "Penmanshiel",
            "T01",
        ),
        (
            "Turbine_Data_Penmanshiel_15_2024-01-01_-_2025-01-01_1056.csv",
            "Penmanshiel",
            "T15",
        ),
    ],
)
def test_turbine_id_from_filename_padded_or_not(
    name: str, scada_name: str, expected: str
) -> None:
    assert turbine_id_from_filename(name, scada_name) == expected


def test_turbine_id_from_filename_rejects_other_farm() -> None:
    with pytest.raises(ValueError, match="No Penmanshiel turbine number"):
        turbine_id_from_filename("Turbine_Data_Kelmarsh_3_x.csv", "Penmanshiel")


def make_files(folder: Path, names: list[str]) -> list[Path]:
    folder.mkdir()
    for name in names:
        (folder / name).write_text("")
    return [folder / n for n in names]


def test_data_and_status_files_keep_folder_order_and_split_by_type(
    tmp_path: Path,
) -> None:
    late = make_files(tmp_path / "scada_2022", ["Turbine_Data_K_1_b.csv"])
    early = make_files(
        tmp_path / "scada_2021", ["Turbine_Data_K_1_a.csv", "Status_K_1_a.csv"]
    )
    folders = [early[0].parent, late[0].parent]

    assert (turbine_data_files(folders), status_files(folders)) == (
        [early[0], late[0]],
        [early[1]],
    )


STATUS_FILE = (
    "# Exported by Greenbyte\r\n"
    "# Turbine: Penmanshiel 02\r\n"
    "Timestamp start,Timestamp end,Duration,Status,Code,Message,\r\n"
    "2022-01-01 00:00:00,2022-01-01 01:00:00,01:00:00,Stop,10,Low oil,\r\n"
    "2022-01-02 00:00:00,-,-,Informational,1,Run,\r\n"
)


def test_file_columns_reads_commented_header(tmp_path: Path) -> None:
    path = tmp_path / "Turbine_Data_K_1_x.csv"
    path.write_bytes(b"# meta\r\n# Date and time,Power (kW)\r\n2022,1\r\n")

    assert file_columns(path) == ["Date and time", "Power (kW)"]


def test_read_signals_reads_only_given_columns(tmp_path: Path) -> None:
    path = tmp_path / "Turbine_Data_K_1_x.csv"
    path.write_bytes(b"# meta\r\n# t,a,b\r\n1,2,3\r\n4,5,6\r\n")

    assert read_signals(path, ["t", "b"]).to_dict("list") == {"t": [1, 4], "b": [3, 6]}


def test_read_status_parses_plain_header_and_adds_turbine(tmp_path: Path) -> None:
    path = tmp_path / "Status_Penmanshiel_02_x.csv"
    path.write_bytes(STATUS_FILE.encode())

    frame = read_status(path, "Penmanshiel")

    assert frame[["Status", "Message", "turbine"]].values.tolist() == [
        ["Stop", "Low oil", "T02"],
        ["Informational", "Run", "T02"],
    ]


def test_group_by_turbine_keeps_file_order_per_turbine() -> None:
    paths = [
        Path("2021/Turbine_Data_Penmanshiel_02_a.csv"),
        Path("2021/Turbine_Data_Penmanshiel_01_a.csv"),
        Path("2022/Turbine_Data_Penmanshiel_01_b.csv"),
    ]

    assert group_by_turbine(paths, "Penmanshiel") == {
        "T01": [paths[1], paths[2]],
        "T02": [paths[0]],
    }
