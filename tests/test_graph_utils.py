"""Tests for graph persistence and the SPARQL lookups."""

from pathlib import Path

import pandas as pd
import pytest
from rdflib import Graph

from src.graph_utils import (
    KT,
    PROV,
    get_fault_events,
    get_lineage,
    get_schema,
    get_sensors,
    load_graph,
    parquet_path_of,
    save_graph,
    turbine_iri,
)


def short(iri: str) -> str:
    return iri.split("#")[-1]


def test_save_and_load_round_trip(graph: Graph, tmp_path: Path) -> None:
    path = tmp_path / "out" / "g.ttl"
    save_graph(graph, path)

    assert len(load_graph(path)) == len(graph)


def test_saved_file_starts_with_schema_and_uses_prefixes(
    graph: Graph, tmp_path: Path
) -> None:
    path = tmp_path / "g.ttl"
    save_graph(graph, path)
    text = path.read_text(encoding="utf-8")

    assert (
        text.index("rdfs:Class") < text.index("kt:KLM_T01 "),
        "@prefix geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> ." in text,
    ) == (True, True)


def test_parquet_path_of_single_turbine_file(graph: Graph) -> None:
    sensors = get_sensors(graph, "KLM", "T02")

    assert parquet_path_of(sensors) == Path("data/processed/kelmarsh/T02.parquet")


def test_parquet_path_of_rejects_mixed_paths() -> None:
    sensors = pd.DataFrame({"parquet_path": ["a.parquet", "b.parquet"]})

    with pytest.raises(LookupError, match="one Parquet path"):
        parquet_path_of(sensors)


def test_get_fault_events_all_and_per_turbine(graph: Graph) -> None:
    events = get_fault_events(graph)

    assert (
        events[["farm_code", "turbine_id", "component_key", "message"]].to_dict(
            "records"
        ),
        len(get_fault_events(graph, "KLM", "T02")),
    ) == (
        [
            {
                "farm_code": "KLM",
                "turbine_id": "T01",
                "component_key": "yaw",
                "message": "Yaw error",
            },
            {
                "farm_code": "KLM",
                "turbine_id": "T01",
                "component_key": "gearbox",
                "message": "Low oil",
            },
        ],
        0,
    )


def test_get_fault_events_parses_utc_times_and_missing_end(graph: Graph) -> None:
    events = get_fault_events(graph, "KLM", "T01")

    assert (events["start"].iloc[1], events["end"].isna().tolist()) == (
        pd.Timestamp("2022-05-01 10:00", tz="UTC"),
        [True, False],
    )


def test_lineage_covers_turbine_and_sources_only(graph: Graph) -> None:
    nodes = {short(s) for s in get_lineage(graph, "KLM", "T01")["s"]}

    assert (
        {
            "KLM",
            "KLM_T01",
            "KLM_T01_gearbox",
            "KLM_T01_power",
            "Source_Kelmarsh_WT_static_csv",
            "Source_Kelmarsh_SCADA_2022_zip",
            "Dataset_KLM",
        }
        <= nodes,
        {n for n in nodes if "T02" in n or "PEN" in n},
    ) == (True, set())


def test_lineage_hides_links_to_other_turbines(graph: Graph) -> None:
    lineage = get_lineage(graph, "KLM", "T01")
    farm_links = lineage[lineage["s"] == str(KT.KLM)]

    assert str(turbine_iri("KLM", "T02")) not in set(farm_links["o"])


def test_schema_has_spec_classes(graph: Graph) -> None:
    classes, properties = get_schema(graph)
    parents = classes.set_index("class")["parent"].dropna().map(short).to_dict()

    assert (
        {short(c): p for c, p in parents.items() if p == "Model"},
        parents[str(KT.Dataset)],
        str(PROV.wasDerivedFrom) in set(properties["property"]),
    ) == (
        {
            "PowerModel": "Model",
            "ThermalModel": "Model",
            "PitchModel": "Model",
            "YawModel": "Model",
        },
        "Entity",
        True,
    )
