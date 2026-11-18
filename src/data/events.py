"""Parse each farm's status files into one events table per farm.

Fault events per component are printed, as spec §7.4 asks.
"""

import logging
from typing import Any

import pandas as pd

from src.config import farm_names, farm_processed_dir, load_config, scada_dirs
from src.data.scada_format import read_status, status_files

logger = logging.getLogger(__name__)

EVENT_COLUMNS = {
    "turbine": "turbine_id",
    "Timestamp start": "start",
    "Timestamp end": "end",
    "Status": "status",
    "Code": "code",
    "Message": "message",
    "IEC category": "iec_category",
    "source_file": "source_file",
}
EVENT_KEY = ["turbine_id", "start", "status", "code", "message"]


def message_components(fault_labels: dict[str, list[str]]) -> dict[str, str]:
    """Invert the config labels: message -> component."""
    return {m: component for component, ms in fault_labels.items() for m in ms}


def tidy_events(
    raw: pd.DataFrame, fault_labels: dict[str, list[str]], statuses: list[str]
) -> pd.DataFrame:
    """Rename, parse UTC times, dedupe and label fault events with a component.

    `raw` holds status-file rows with `turbine` and `source_file` (the status
    file name). `end` is missing ("-") for informational rows. Only rows
    whose status is one of `statuses` get a component; unmatched messages
    stay missing.
    """
    frame = raw.rename(columns=EVENT_COLUMNS)[list(EVENT_COLUMNS.values())]
    frame = frame.assign(
        start=pd.to_datetime(frame["start"], utc=True),
        end=pd.to_datetime(frame["end"], utc=True, errors="coerce"),
    ).drop_duplicates(EVENT_KEY)
    component = frame["message"].map(message_components(fault_labels))
    return (
        frame.assign(component=component.where(frame["status"].isin(statuses)))
        .sort_values(["turbine_id", "start"], kind="stable")
        .reset_index(drop=True)
    )


def load_events(config: dict[str, Any], farm: str) -> pd.DataFrame:
    """All status rows of a farm as a tidy events table."""
    scada_name = config["farms"][farm]["scada_name"]
    paths = status_files(scada_dirs(config, farm))
    raw = pd.concat(
        [read_status(p, scada_name).assign(source_file=p.name) for p in paths],
        ignore_index=True,
    )
    return tidy_events(raw, config["fault_labels"], config["events"]["fault_statuses"])


def fault_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Labelled fault events per component and status."""
    labelled = events.dropna(subset=["component"])
    return labelled.groupby(["component", "status"]).size().unstack(fill_value=0)


def main() -> None:
    """Write <processed>/<farm>/events.parquet for every farm."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    for farm in farm_names(config):
        events = load_events(config, farm)
        out_dir = farm_processed_dir(config, farm)
        out_dir.mkdir(parents=True, exist_ok=True)
        events.to_parquet(out_dir / "events.parquet", index=False)
        logger.info("Wrote %d events for %s", len(events), farm)
        print(f"\n{farm}: {len(events)} events, fault events per component:")
        print(fault_counts(events).to_string())


if __name__ == "__main__":
    main()
