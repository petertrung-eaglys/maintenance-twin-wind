"""Print the structure of each farm's raw files (spec §7.3 decision gate).

Output is printed (not logged) because the spec asks for printed inspection.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import farm_names, farm_raw_dir, load_config, scada_dirs
from src.scada_format import (
    HEADER_SEARCH_LINES,
    file_columns,
    group_files,
    read_head,
    read_signals,
    read_status,
    status_files,
    status_header_index,
)

HEAD_LINES = 12
LINE_PREVIEW_CHARS = 200
EVENT_KEY = ["turbine", "Timestamp start", "Status", "Code", "Message"]


def print_section(title: str) -> None:
    """Print a section heading."""
    print(f"\n=== {title} ===")


def missing_columns(columns: list[str], wanted: list[str]) -> list[str]:
    """The wanted column names that are not among the columns, in wanted order."""
    return [c for c in wanted if c not in columns]


def files_missing_count(
    columns_by_file: dict[str, list[str]], wanted: list[str]
) -> dict[str, int]:
    """For each wanted column, the number of files that lack it."""
    return {w: sum(w not in cols for cols in columns_by_file.values()) for w in wanted}


def filled_fraction(frame: pd.DataFrame, timestamp_column: str) -> pd.Series:
    """Per column, the share of distinct timestamps with a value.

    Exports may repeat timestamps with all-NaN copies, so a timestamp counts
    as filled when any of its rows has a value.
    """
    values = frame.drop(columns=timestamp_column).notna()
    return values.groupby(frame[timestamp_column]).any().mean()


def event_counts(events: pd.DataFrame, statuses: list[str]) -> pd.DataFrame:
    """Distinct events per (Status, Message) for the given statuses.

    Exports may repeat rows, so events are de-duplicated on turbine, start,
    status, code and message first. Sorted by count, most frequent first.
    """
    distinct = events[events["Status"].isin(statuses)].drop_duplicates(EVENT_KEY)
    counts = distinct.groupby(["Status", "Message"]).size().rename("count")
    return counts.reset_index().sort_values(
        ["count", "Status", "Message"], ascending=[False, True, True]
    )


def read_table(path: Path) -> pd.DataFrame:
    """A static or mapping table, from CSV (BOM-aware) or Excel."""
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def print_file_groups(scada_dir: Path) -> dict[str, list[str]]:
    """Print the SCADA files grouped by type and return the groups."""
    groups = group_files([p.name for p in scada_dir.iterdir() if p.is_file()])
    print_section(f"Files in {scada_dir}")
    for kind, names in groups.items():
        print(f"{kind} ({len(names)}):")
        for name in names:
            print(f"  {name}")
    return groups


def print_turbine_file(path: Path) -> None:
    """Print the raw head and the full column list of one turbine data file."""
    print_section(f"First {HEAD_LINES} raw lines of {path.name}")
    for line in read_head(path, HEAD_LINES):
        cut = line[:LINE_PREVIEW_CHARS]
        print(cut + (" ..." if len(line) > LINE_PREVIEW_CHARS else ""))
    columns = file_columns(path)
    print_section(f"{len(columns)} columns of {path.name}")
    for i, column in enumerate(columns):
        print(f"  {i:3d}  {column}")


def print_missing(columns_by_file: dict[str, list[str]], wanted: list[str]) -> None:
    """Print, per file, the configured columns it lacks."""
    print_section("Configured columns missing per turbine data file")
    for name, columns in columns_by_file.items():
        print(f"  {name}: {missing_columns(columns, wanted) or 'none'}")


def print_coverage(
    farm: str, columns_by_file: dict[str, list[str]], wanted: list[str]
) -> None:
    """Print how many of the farm's files lack each configured column."""
    total = len(columns_by_file)
    print_section(f"{farm}: files missing each configured column (of {total})")
    for column, count in files_missing_count(columns_by_file, wanted).items():
        print(f"  {count:3d}  {column}")


def print_fill(
    farm: str, paths: list[Path], signals: dict[str, str], timestamp_column: str
) -> None:
    """Print rows, distinct timestamps and filled share per signal and file."""
    rows = []
    for path in paths:
        frame = read_signals(path, [timestamp_column, *signals.values()])
        filled = filled_fraction(frame, timestamp_column)
        rows.append(
            {
                "folder": path.parent.name,
                "rows": len(frame),
                "timestamps": frame[timestamp_column].nunique(),
                **{key: filled[column] for key, column in signals.items()},
            }
        )
    print_section(f"{farm}: share of timestamps with a value (first file per year)")
    print("\n".join(f"  {p.parent.name}: {p.name}" for p in paths))
    print(pd.DataFrame(rows).set_index("folder").round(3).T.to_string())


def print_table(path: Path) -> None:
    """Print the columns and rows of a small static or mapping table."""
    frame = read_table(path)
    print_section(f"{path.name}: {len(frame)} rows")
    print(f"Columns: {list(frame.columns)}")
    print(frame.to_string())


def print_events(
    farm: str, paths: list[Path], scada_name: str, statuses: list[str]
) -> None:
    """Print the status header and the distinct event counts of a farm."""
    lines = read_head(paths[0], HEADER_SEARCH_LINES)
    print_section(f"{farm}: status header ({len(paths)} status files)")
    print(lines[status_header_index(lines)])
    events = pd.concat([read_status(p, scada_name) for p in paths])
    counts = event_counts(events, statuses)
    print_section(f"{farm}: distinct {'/'.join(statuses)} events per message")
    print(counts.to_string(index=False))


def inspect_farm(config: dict[str, Any], farm: str) -> None:
    """Print file groups, columns, coverage, tables and events of one farm."""
    farm_config = config["farms"][farm]
    wanted = [
        config["scada"]["timestamp_column"],
        *config["signals"].values(),
        *config["component_signals"].values(),
    ]
    columns_by_file: dict[str, list[str]] = {}
    first_files = []
    for scada_dir in scada_dirs(config, farm):
        groups = print_file_groups(scada_dir)
        paths = [scada_dir / name for name in groups["turbine_data"]]
        print_turbine_file(paths[0])
        year_columns = {p.name: file_columns(p) for p in paths}
        print_missing(year_columns, wanted)
        columns_by_file |= year_columns
        first_files.append(paths[0])
    print_coverage(farm, columns_by_file, wanted)
    print_fill(
        farm,
        first_files,
        config["signals"] | config["component_signals"],
        config["scada"]["timestamp_column"],
    )
    raw_dir = farm_raw_dir(config, farm)
    for f in farm_config["downloads"]:
        if "extract_to" not in f:
            print_table(raw_dir / f["name"])
    print_events(
        farm,
        status_files(scada_dirs(config, farm)),
        farm_config["scada_name"],
        config["events"]["fault_statuses"],
    )


def main() -> None:
    """Inspect the raw files of every configured farm."""
    config = load_config()
    for farm in farm_names(config):
        print_section(f"FARM {farm}")
        inspect_farm(config, farm)


if __name__ == "__main__":
    main()
