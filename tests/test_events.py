"""Tests for parsing and labelling status events."""

import pandas as pd

from src.events import fault_counts, message_components, tidy_events

LABELS = {"gearbox": ["Low gearbox oil pressure"], "yaw": ["Yaw error"]}
STATUSES = ["Stop", "Warning"]


def raw_row(start: str, end: str, status: str, message: str, turbine="T01") -> dict:
    return {
        "Timestamp start": start,
        "Timestamp end": end,
        "Duration": "",
        "Status": status,
        "Code": "1",
        "Message": message,
        "Comment": "",
        "Service contract category": "",
        "IEC category": "Forced outage",
        "turbine": turbine,
        "source_file": f"Status_X_{turbine}.csv",
    }


RAW = pd.DataFrame(
    [
        raw_row("2022-01-02 00:00:00", "2022-01-02 01:00:00", "Stop", "Yaw error"),
        raw_row("2022-01-01 00:00:00", "-", "Informational", "Yaw error"),
        raw_row(
            "2022-01-01 06:00:00",
            "2022-01-01 07:00:00",
            "Warning",
            "Low gearbox oil pressure",
        ),
        raw_row(
            "2022-01-01 06:00:00",
            "2022-01-01 07:00:00",
            "Warning",
            "Low gearbox oil pressure",
        ),
        raw_row("2022-01-03 00:00:00", "2022-01-03 00:30:00", "Stop", "Grid loss"),
    ]
)


def test_message_components_inverts_labels() -> None:
    assert message_components(LABELS) == {
        "Low gearbox oil pressure": "gearbox",
        "Yaw error": "yaw",
    }


def test_tidy_events_renames_dedupes_and_sorts() -> None:
    events = tidy_events(RAW, LABELS, STATUSES)

    assert (list(events.columns), events["message"].tolist()) == (
        [
            "turbine_id",
            "start",
            "end",
            "status",
            "code",
            "message",
            "iec_category",
            "source_file",
            "component",
        ],
        ["Yaw error", "Low gearbox oil pressure", "Yaw error", "Grid loss"],
    )


def test_tidy_events_parses_utc_and_dash_end_as_missing() -> None:
    events = tidy_events(RAW, LABELS, STATUSES)

    assert (events["start"].iloc[0], events["end"].isna().tolist()) == (
        pd.Timestamp("2022-01-01 00:00", tz="UTC"),
        [True, False, False, False],
    )


def test_tidy_events_labels_only_fault_statuses() -> None:
    components = tidy_events(RAW, LABELS, STATUSES)["component"]

    assert components.fillna("-").tolist() == [
        "-",  # informational row with a labelled message
        "gearbox",
        "yaw",
        "-",  # unmatched message
    ]


def test_fault_counts_per_component_and_status() -> None:
    counts = fault_counts(tidy_events(RAW, LABELS, STATUSES))

    assert counts.to_dict("index") == {
        "gearbox": {"Stop": 0, "Warning": 1},
        "yaw": {"Stop": 1, "Warning": 0},
    }
