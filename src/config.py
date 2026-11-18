"""Load the project configuration from config.yaml and resolve farm paths."""

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config file and return it as a dict."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def farm_names(config: dict[str, Any]) -> list[str]:
    """The configured farm keys (e.g. "kelmarsh"), in config order."""
    return list(config["farms"])


def farm_raw_dir(config: dict[str, Any], farm: str) -> Path:
    """The folder a farm's raw files are downloaded into: <raw>/<farm>."""
    return Path(config["paths"]["raw"]) / farm


def farm_processed_dir(config: dict[str, Any], farm: str) -> Path:
    """The folder a farm's processed Parquet files go to: <processed>/<farm>."""
    return Path(config["paths"]["processed"]) / farm


def scada_archives(config: dict[str, Any], farm: str) -> list[dict[str, str]]:
    """A farm's download entries that are SCADA archives (with extract_to)."""
    return [f for f in config["farms"][farm]["downloads"] if "extract_to" in f]


def scada_dirs(config: dict[str, Any], farm: str) -> list[Path]:
    """A farm's SCADA folders in config order, each listed once.

    Several archives may unzip into the same folder (Penmanshiel splits a
    year by turbine group), so shared folders are de-duplicated.
    """
    raw = farm_raw_dir(config, farm)
    folders = (f["extract_to"] for f in scada_archives(config, farm))
    return [raw / name for name in dict.fromkeys(folders)]
