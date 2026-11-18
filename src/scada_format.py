"""Knowledge of the Greenbyte SCADA export format (file names and headers)."""

import csv
import re
from itertools import islice
from pathlib import Path

COMMENT = "#"
FILE_TYPES = {"Turbine_Data_": "turbine_data", "Status_": "status"}
TURBINE_FILE_GLOB = "Turbine_Data_*.csv"
STATUS_FILE_GLOB = "Status_*.csv"


def format_turbine_id(number: int) -> str:
    """Canonical turbine id used everywhere, e.g. 1 -> 'T01'."""
    return f"T{number:02d}"


def turbine_id_from_filename(name: str, scada_name: str) -> str:
    """Turbine id from a file name like '..._Kelmarsh_3_...' or '..._X_03_...'.

    Turbine numbers may or may not be zero-padded (Penmanshiel pads them).
    """
    match = re.search(rf"{re.escape(scada_name)}_(\d+)_", name)
    if match is None:
        raise ValueError(f"No {scada_name} turbine number in file name: {name}")
    return format_turbine_id(int(match.group(1)))


def files_matching(folders: list[Path], pattern: str) -> list[Path]:
    """Files matching a glob in the given folders, folder then name order."""
    return [p for folder in folders for p in sorted(folder.glob(pattern))]


def turbine_data_files(folders: list[Path]) -> list[Path]:
    """Turbine data files in the given folders, folder then name order."""
    return files_matching(folders, TURBINE_FILE_GLOB)


def status_files(folders: list[Path]) -> list[Path]:
    """Status (event log) files in the given folders, folder then name order."""
    return files_matching(folders, STATUS_FILE_GLOB)


def group_by_turbine(paths: list[Path], scada_name: str) -> dict[str, list[Path]]:
    """Map turbine id to its files, keeping the given order within each id."""
    ids = [turbine_id_from_filename(p.name, scada_name) for p in paths]
    return {
        tid: [p for p, i in zip(paths, ids, strict=True) if i == tid]
        for tid in sorted(set(ids))
    }


def classify_file(name: str) -> str:
    """Return the file type from its name: turbine_data, status or other."""
    return next(
        (kind for prefix, kind in FILE_TYPES.items() if name.startswith(prefix)),
        "other",
    )


def group_files(names: list[str]) -> dict[str, list[str]]:
    """Group file names by type, each group sorted."""
    kinds = sorted({classify_file(n) for n in names})
    return {k: sorted(n for n in names if classify_file(n) == k) for k in kinds}


def read_head(path: Path, n: int) -> list[str]:
    """Return the first n lines of a text file, without line endings."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [line.rstrip("\r\n") for line in islice(f, n)]


def first_plain_line_index(lines: list[str]) -> int:
    """Index of the first line that is not a comment (len(lines) if none)."""
    return next(
        (i for i, line in enumerate(lines) if not line.startswith(COMMENT)),
        len(lines),
    )


def header_line_index(lines: list[str]) -> int:
    """Turbine data header: the last comment line before the data."""
    return first_plain_line_index(lines) - 1


def status_header_index(lines: list[str]) -> int:
    """Status file header: the first plain line after the metadata comments."""
    return first_plain_line_index(lines)


def parse_columns(header_line: str) -> list[str]:
    """Split a (possibly commented) CSV header line into column names."""
    return next(csv.reader([header_line.lstrip(COMMENT).strip()]))
