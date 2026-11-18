"""Crash-safe file writes.

Pipeline steps rewrite turbine Parquet files in place; a crash during a
plain write leaves a truncated file (it happened: a reboot left a 0-byte
turbine file). Writing to a temporary file, flushing it to disk and then
renaming it over the target keeps either the old or the new file.
"""

import os
from pathlib import Path

import pandas as pd


def write_parquet(frame: pd.DataFrame, path: Path | str) -> None:
    """Write `frame` (without index) to `path` atomically."""
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        with open(temporary, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
