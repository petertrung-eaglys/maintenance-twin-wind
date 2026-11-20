"""Tests for the dashboard's Graphviz rendering of the graph."""

import shutil
import subprocess
from datetime import UTC, datetime

import pandas as pd
import pytest
from rdflib import Graph

from src.dashboard.ontology_view import (
    ACTIVE_ALERT_FILL,
    CLASS_FILLS,
    component_keys,
    dot_id,
    for_components,
    html_label,
    lineage_dot,
    main_type,
    qname,
    schema_dot,
)
from src.graph.graph_utils import add_model, get_lineage, get_schema, model_iri

METRICS = {"rmse": 1.0, "mae": 1.0, "nrmse": 1.0, "n_train": 6, "n_test": 6}


def with_models(graph: Graph) -> Graph:
    """The synthetic graph plus a gearbox and a generator model of KLM T01."""
    for component, target in (("gearbox", "gear_oil_temp"), ("generator", "power")):
        add_model(
            graph,
            "ThermalModel",
            "KLM",
            "T01",
            component,
            "physics",
            ["power"],
            target,
            METRICS,
            datetime(2026, 11, 20, tzinfo=UTC),
            qualifier=target,
        )
    return graph


def subjects(triples: pd.DataFrame) -> set[str]:
    return {qname(s) for s in triples["s"]}


def test_qname_uses_the_longest_matching_prefix() -> None:
    assert qname("https://example.org/wind-twin#Turbine") == "kt:Turbine"
    assert qname("urn:other") == "urn:other"


def test_dot_id_escapes_quotes_and_backslashes() -> None:
    assert dot_id('a"b\\c') == '"a\\"b\\\\c"'


def test_html_label_escapes_values() -> None:
    label = html_label("<T01>", "kt:Turbine", [("a&b", "1")], "#fff")

    assert "&lt;T01&gt;" in label and "a&amp;b" in label


def test_main_type_prefers_a_specific_class() -> None:
    assert main_type(["kt:PowerModel", "prov:Entity"]) == "kt:PowerModel"
    assert main_type(["prov:Entity"]) == "prov:Entity"


def test_for_components_drops_other_components_with_their_sensors_and_models(
    graph: Graph,
) -> None:
    triples = get_lineage(with_models(graph), "KLM", "T01")

    kept = for_components(triples, {"gearbox"})

    names = subjects(kept)
    assert "gearbox" in component_keys(kept) and "generator" not in component_keys(kept)
    assert "kt:ThermalModel_KLM_T01_gear_oil_temp_physics" in names
    assert "kt:ThermalModel_KLM_T01_power_physics" not in names
    assert not any("power" in n and "Sensor" in n for n in names)


def test_lineage_dot_colours_models_with_an_active_alert(graph: Graph) -> None:
    g = with_models(graph)
    model = str(model_iri("ThermalModel", "KLM", "T01", "physics", "gear_oil_temp"))

    dot = lineage_dot(get_lineage(g, "KLM", "T01"), {model})

    assert ACTIVE_ALERT_FILL in dot
    assert CLASS_FILLS["kt:Turbine"] in dot


def test_schema_dot_has_subclass_and_relation_edges(graph: Graph) -> None:
    dot = schema_dot(*get_schema(graph))

    assert 'label="subClassOf"' in dot
    assert 'label="kt:hasTurbine"' in dot


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz not installed")
def test_generated_dot_is_valid_graphviz(graph: Graph) -> None:
    g = with_models(graph)
    for dot in (
        lineage_dot(get_lineage(g, "KLM", "T01"), set()),
        schema_dot(*get_schema(g)),
    ):
        result = subprocess.run(
            ["dot", "-Tsvg"], input=dot, capture_output=True, text=True, check=False
        )
        assert (result.returncode, result.stderr) == (0, "")
