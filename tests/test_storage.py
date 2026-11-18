"""Tests for crash-safe Parquet writes."""

from pathlib import Path

import pandas as pd
import pytest

from src.storage import write_parquet


def test_write_parquet_replaces_the_file(tmp_path: Path) -> None:
    path = tmp_path / "T01.parquet"
    write_parquet(pd.DataFrame({"x": [1]}), path)

    write_parquet(pd.DataFrame({"x": [2, 3]}), path)

    assert pd.read_parquet(path)["x"].tolist() == [2, 3]


def test_failed_write_keeps_the_old_file_and_no_temporary(tmp_path: Path) -> None:
    path = tmp_path / "T01.parquet"
    write_parquet(pd.DataFrame({"x": [1]}), path)
    unwritable = pd.DataFrame({"x": [object()]})  # pyarrow cannot store it

    with pytest.raises(Exception):  # noqa: B017 - pyarrow's error type varies
        write_parquet(unwritable, path)

    assert (pd.read_parquet(path)["x"].tolist(), sorted(tmp_path.iterdir())) == (
        [1],
        [path],
    )
