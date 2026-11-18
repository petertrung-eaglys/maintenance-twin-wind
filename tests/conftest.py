"""Shared fixtures: a small synthetic two-farm graph."""

import pandas as pd
import pytest
from rdflib import Graph

from src.build_ontology import FarmSources, build_graph

SIGNALS = {"power": "Power (kW)", "wind_speed": "Wind speed (m/s)"}
COMPONENT_SIGNALS = {"gear_oil_temp": "Gear oil temperature (°C)"}


def farm_config(code: str, name: str, record: str) -> dict:
    files = [f"{name}_WT_static.csv", f"{name}_WT_dataSignalMapping.csv"]
    archive = f"{name}_SCADA_2022.zip"
    return {
        "code": code,
        "scada_name": name,
        "static_file": files[0],
        "mapping_file": files[1],
        "dataset": {
            "title": f"{name} wind farm data",
            "doi": f"10.5281/zenodo.{record}",
            "publisher": "Cubico Sustainable Investments Ltd",
            "license": "CC-BY-4.0",
        },
        "downloads": [
            {"name": n, "url": f"https://zenodo.org/{n}", "md5": "0" * 32}
            for n in files
        ]
        + [
            {
                "name": archive,
                "url": f"https://zenodo.org/{archive}",
                "md5": "1" * 32,
                "extract_to": "scada_2022",
            }
        ],
    }


CONFIG = {
    "paths": {"raw": "data/raw", "processed": "data/processed"},
    "farms": {
        "kelmarsh": farm_config("KLM", "Kelmarsh", "16807551"),
        "penmanshiel": farm_config("PEN", "Penmanshiel", "16807304"),
    },
    "signals": SIGNALS,
    "component_signals": COMPONENT_SIGNALS,
}


def static(name: str, numbers: list[int], model: str, rotor: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "turbine_id": f"T{n:02d}",
                "title": f"{name} {n}",
                "manufacturer": "Senvion",
                "model": model,
                "rated_power_kw": 2050,
                "hub_height_m": 78.5,
                "rotor_diameter_m": rotor,
                "latitude": 52.4 + n / 100,
                "longitude": -0.9,
                "elevation_m": 150.0,
                "commercial_operations_date": pd.Timestamp("2016-04-15", tz="UTC"),
            }
            for n in numbers
        ]
    ).set_index("turbine_id")


# Mapping without an entry for wind speed
MAPPING = pd.DataFrame(
    {
        "signal_id": [5, 178],
        "greenbyte_title": ["Power", "Gear oil temperature"],
        "manufacturer_title": ["Active power", "Gear oil temperature"],
        "unit": ["kWh", "degC"],
    }
)


def event(tid: str, start: str, end: str | None, message: str, component: str):
    return {
        "turbine_id": tid,
        "start": pd.Timestamp(start, tz="UTC"),
        "end": pd.Timestamp(end, tz="UTC") if end else pd.NaT,
        "status": "Stop",
        "code": "10",
        "message": message,
        "iec_category": "Forced outage",
        "source_file": f"Status_Kelmarsh_{tid[1:]}_2022.csv",
        "component": component,
    }


def data_file(name: str, number: str) -> str:
    return f"Turbine_Data_{name}_{number}_2022-01-01_-_2023-01-01_1.csv"


SOURCES = {
    "kelmarsh": FarmSources(
        static=static("Kelmarsh", [1, 2], "MM92", 92.0),
        mapping=MAPPING,
        data_files={
            "T01": [data_file("Kelmarsh", "1")],
            "T02": [data_file("Kelmarsh", "2")],
        },
        archive_of={
            name: "Kelmarsh_SCADA_2022.zip"
            for name in (
                data_file("Kelmarsh", "1"),
                data_file("Kelmarsh", "2"),
                "Status_Kelmarsh_01_2022.csv",
            )
        },
        events=pd.DataFrame(
            [
                event(
                    "T01", "2022-05-01 10:00", "2022-05-01 12:00", "Low oil", "gearbox"
                ),
                event("T01", "2022-03-01 08:00", None, "Yaw error", "yaw"),
            ]
        ),
    ),
    "penmanshiel": FarmSources(
        static=static("Penmanshiel", [1], "MM82", 82.0),
        mapping=MAPPING,
        data_files={"T01": [data_file("Penmanshiel", "01")]},
        archive_of={data_file("Penmanshiel", "01"): "Penmanshiel_SCADA_2022.zip"},
        events=pd.DataFrame(columns=list(event("T01", "2022", None, "", ""))),
    ),
}


@pytest.fixture
def graph() -> Graph:
    return build_graph(CONFIG, SOURCES)
