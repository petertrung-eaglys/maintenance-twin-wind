"""Tests for the ontology builder, checked through the SPARQL helpers."""

import zipfile
from datetime import date
from pathlib import Path

from rdflib import RDF, RDFS, Graph, Literal

from src.build_ontology import (
    COMPONENTS,
    SIGNAL_SEMANTICS,
    add_schema,
    archive_members,
    mapping_entry,
    strip_unit,
)
from src.graph_utils import (
    KT,
    PROV,
    component_iri,
    dataset_iri,
    fault_event_iri,
    get_components,
    get_farms,
    get_sensors,
    get_turbines,
    new_graph,
    sensor_iri,
    source_iri,
    turbine_iri,
)
from tests.conftest import MAPPING, data_file


def test_strip_unit_drops_only_trailing_unit() -> None:
    assert [
        strip_unit("Power (kW)"),
        strip_unit("Ambient temperature (converter) (°C)"),
        strip_unit("Capacity factor"),
    ] == ["Power", "Ambient temperature (converter)", "Capacity factor"]


def test_mapping_entry_matches_title_without_unit_or_none() -> None:
    assert (
        mapping_entry(MAPPING, "Power (kW)")["signal_id"],
        mapping_entry(MAPPING, "Wind speed (m/s)"),
    ) == (5, None)


def test_every_signal_monitors_a_known_component() -> None:
    assert {c for c, _, _ in SIGNAL_SEMANTICS.values()} <= set(COMPONENTS)


def test_schema_terms_all_have_label_and_comment() -> None:
    g = new_graph()
    add_schema(g)
    terms = set(g.subjects(RDF.type, RDFS.Class)) | set(
        g.subjects(RDF.type, RDF.Property)
    )

    unlabelled = [
        t
        for t in terms
        if (t, RDFS.label, None) not in g or (t, RDFS.comment, None) not in g
    ]
    assert unlabelled == []


def test_graph_has_both_farms_with_dataset_metadata(graph: Graph) -> None:
    farms = get_farms(graph)

    assert farms[["farm_code", "label", "doi"]].to_dict("records") == [
        {"farm_code": "KLM", "label": "Kelmarsh", "doi": "10.5281/zenodo.16807551"},
        {"farm_code": "PEN", "label": "Penmanshiel", "doi": "10.5281/zenodo.16807304"},
    ]


def test_turbines_are_farm_scoped_with_specs(graph: Graph) -> None:
    turbines = get_turbines(graph)

    assert turbines[["farm_code", "turbine_id", "rotor_diameter_m"]].to_dict(
        "records"
    ) == [
        {"farm_code": "KLM", "turbine_id": "T01", "rotor_diameter_m": 92.0},
        {"farm_code": "KLM", "turbine_id": "T02", "rotor_diameter_m": 92.0},
        {"farm_code": "PEN", "turbine_id": "T01", "rotor_diameter_m": 82.0},
    ]


def test_turbine_specs_traced_to_static_file(graph: Graph) -> None:
    node = turbine_iri("KLM", "T01")

    assert (
        str(graph.value(node, KT.turbineModel)),
        graph.value(node, KT.commissioningDate).toPython(),
        set(graph.objects(node, PROV.wasDerivedFrom)),
    ) == ("MM92", date(2016, 4, 15), {source_iri("Kelmarsh_WT_static.csv")})


def test_each_turbine_has_all_components(graph: Graph) -> None:
    keys = get_components(graph, "PEN", "T01")["component_key"]

    assert sorted(keys) == sorted(COMPONENTS)


def test_sensors_monitor_their_components_and_parquet(graph: Graph) -> None:
    sensors = get_sensors(graph, "PEN", "T01").set_index("signal_key")

    assert sensors[["component_key", "parquet_path"]].to_dict("index") == {
        "gear_oil_temp": {
            "component_key": "gearbox",
            "parquet_path": "data/processed/penmanshiel/T01.parquet",
        },
        "power": {
            "component_key": "generator",
            "parquet_path": "data/processed/penmanshiel/T01.parquet",
        },
        "wind_speed": {
            "component_key": "nacelle",
            "parquet_path": "data/processed/penmanshiel/T01.parquet",
        },
    }


def test_mapping_title_recorded_only_where_matched(graph: Graph) -> None:
    titles = [
        graph.value(sensor_iri("KLM", "T01", key), KT.manufacturerTitle)
        for key in ("power", "wind_speed")
    ]

    assert titles == [Literal("Active power"), None]


def test_sensor_traced_to_raw_file_zip_and_dataset(graph: Graph) -> None:
    raw = source_iri(data_file("Kelmarsh", "2"))
    archive = graph.value(raw, KT.partOf)
    sensor = sensor_iri("KLM", "T02", "power")

    assert (
        raw in set(graph.objects(sensor, PROV.wasDerivedFrom)),
        archive,
        graph.value(archive, KT.partOf),
    ) == (True, source_iri("Kelmarsh_SCADA_2022.zip"), dataset_iri("KLM"))


def test_fault_events_numbered_by_start_and_linked(graph: Graph) -> None:
    first, second = fault_event_iri("KLM", "T01", 1), fault_event_iri("KLM", "T01", 2)

    assert (
        str(graph.value(first, KT.eventMessage)),
        graph.value(first, KT.affects),
        graph.value(first, KT.endTime),
        graph.value(second, KT.affects),
        graph.value(second, PROV.wasDerivedFrom),
    ) == (
        "Yaw error",
        component_iri("KLM", "T01", "yaw"),
        None,
        component_iri("KLM", "T01", "gearbox"),
        source_iri("Status_Kelmarsh_01_2022.csv"),
    )


def test_archive_members_maps_files_to_their_zip(tmp_path: Path) -> None:
    for name, members in [("a.zip", ["x.csv", "dir/"]), ("b.zip", ["y.csv"])]:
        with zipfile.ZipFile(tmp_path / name, "w") as zf:
            for m in members:
                zf.writestr(m, "")
    archives = [{"name": "a.zip"}, {"name": "b.zip"}]

    assert archive_members(tmp_path, archives) == {"x.csv": "a.zip", "y.csv": "b.zip"}


def test_empty_events_add_no_fault_events(graph: Graph) -> None:
    pen_events = [
        e for e in graph.subjects(RDF.type, KT.FaultEvent) if "_PEN_" in str(e)
    ]

    assert pen_events == []
