"""Tests for graph persistence and the SPARQL lookups."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from rdflib import Graph

from src.graph.graph_utils import (
    KT,
    PROV,
    add_model,
    get_fault_events,
    get_lineage,
    get_model_inputs,
    get_models,
    get_schema,
    get_sensors,
    load_graph,
    local_name,
    model_iri,
    parquet_path_of,
    save_graph,
    turbine_iri,
)

METRICS = {"rmse": 60.0, "mae": 40.0, "nrmse": 0.03, "n_train": 100, "n_test": 50}
TRAINED_AT = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


def add_power_model(g: Graph, rmse: float = 60.0) -> None:
    add_model(
        g,
        "PowerModel",
        "KLM",
        "T01",
        "generator",
        "baseline",
        ["wind_speed"],
        "power",
        METRICS | {"rmse": rmse},
        TRAINED_AT,
        "models/kelmarsh/x.txt",
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


def test_model_iri_and_local_name() -> None:
    iri = model_iri("ThermalModel", "KLM", "T01", "hybrid")

    assert local_name(iri) == "ThermalModel_KLM_T01_hybrid"


def test_model_iri_qualifier_separates_models_of_one_component() -> None:
    iri = model_iri("ThermalModel", "KLM", "T01", "hybrid", "gear_oil_temp")

    assert local_name(iri) == "ThermalModel_KLM_T01_gear_oil_temp_hybrid"


def test_add_model_with_qualifier_keeps_both_models(graph: Graph) -> None:
    for qualifier in ("a", "b"):
        add_model(
            graph,
            "ThermalModel",
            "KLM",
            "T01",
            "gearbox",
            "physics",
            ["power"],
            "gear_oil_temp",
            METRICS,
            TRAINED_AT,
            qualifier=qualifier,
        )

    assert sorted(get_models(graph)["model"].map(local_name)) == [
        "ThermalModel_KLM_T01_a_physics",
        "ThermalModel_KLM_T01_b_physics",
    ]


def test_add_model_is_returned_by_get_models(graph: Graph) -> None:
    add_power_model(graph)

    models = get_models(graph, "KLM")
    assert models.drop(columns=["model", "generated_at"]).to_dict("records") == [
        {
            "model_class": "PowerModel",
            "farm_code": "KLM",
            "turbine_id": "T01",
            "component_key": "generator",
            "variant": "baseline",
            "target": "power",
            "rmse": 60.0,
            "mae": 40.0,
            "nrmse": 0.03,
            "n_train": 100,
            "n_test": 50,
            "model_file": "models/kelmarsh/x.txt",
        }
    ]


def test_add_model_replaces_previous_node(graph: Graph) -> None:
    add_power_model(graph, rmse=60.0)
    add_power_model(graph, rmse=55.0)

    assert get_models(graph)["rmse"].tolist() == [55.0]


def test_get_model_inputs_lists_input_sensors(graph: Graph) -> None:
    add_power_model(graph)
    model = model_iri("PowerModel", "KLM", "T01", "baseline")

    assert get_model_inputs(graph, str(model)) == ["wind_speed"]
