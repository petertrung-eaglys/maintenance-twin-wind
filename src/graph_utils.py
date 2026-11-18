"""The single access point to the wind twin RDF graph.

Lookups use SPARQL so that structure comes from the graph, not from config.
IRIs are deterministic and farm-scoped (kt:KLM_T01_gearbox), so re-running
a step replaces nodes instead of duplicating them.
"""

import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.query import Result

KT = Namespace("https://example.org/wind-twin#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
PROV = Namespace("http://www.w3.org/ns/prov#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
PREFIXES = {
    "kt": KT,
    "sosa": SOSA,
    "prov": PROV,
    "qudt": QUDT,
    "unit": UNIT,
    "geo": GEO,
    "rdf": RDF,
    "rdfs": RDFS,
    "xsd": XSD,
}
SCHEMA_TYPES = {RDFS.Class, RDF.Property}

FARMS_QUERY = """
SELECT ?farm_code ?label ?dataset ?doi ?publisher ?license
WHERE {
  ?farm a kt:WindFarm ;
        kt:farmCode ?farm_code ;
        rdfs:label ?label ;
        prov:wasDerivedFrom ?dataset .
  ?dataset kt:doi ?doi ;
           kt:publisher ?publisher ;
           kt:license ?license .
}
ORDER BY ?farm_code
"""

TURBINES_QUERY = """
SELECT ?farm_code ?turbine_id ?label ?lat ?long ?rated_power_kw
       ?rotor_diameter_m ?hub_height_m ?turbine_model ?commissioning_date
WHERE {
  ?farm kt:farmCode ?farm_code ;
        kt:hasTurbine ?turbine .
  ?turbine kt:turbineId ?turbine_id ;
           rdfs:label ?label ;
           geo:lat ?lat ;
           geo:long ?long ;
           kt:ratedPowerKW ?rated_power_kw ;
           kt:rotorDiameterM ?rotor_diameter_m ;
           kt:hubHeightM ?hub_height_m ;
           kt:turbineModel ?turbine_model ;
           kt:commissioningDate ?commissioning_date .
}
ORDER BY ?farm_code ?turbine_id
"""

COMPONENTS_QUERY = """
SELECT ?component_key ?component_class
WHERE {
  ?farm kt:farmCode ?farm_code ;
        kt:hasTurbine ?turbine .
  ?turbine kt:turbineId ?turbine_id ;
           kt:hasComponent ?component .
  ?component kt:componentKey ?component_key ;
             a ?component_class .
  ?component_class rdfs:subClassOf kt:Component .
}
ORDER BY ?component_key
"""

SENSORS_QUERY = """
SELECT ?signal_key ?scada_column ?unit ?parquet_path ?component_key ?observes
WHERE {
  ?farm kt:farmCode ?farm_code ;
        kt:hasTurbine ?turbine .
  ?turbine kt:turbineId ?turbine_id ;
           kt:hasSensor ?sensor .
  ?sensor kt:signalKey ?signal_key ;
          kt:scadaColumn ?scada_column ;
          qudt:unit ?unit ;
          kt:storedIn ?parquet_path ;
          sosa:observes ?observes ;
          kt:monitors ?component .
  ?component kt:componentKey ?component_key .
}
ORDER BY ?signal_key
"""

FAULT_EVENTS_QUERY = """
SELECT ?event ?farm_code ?turbine_id ?component_key ?start ?end ?status
       ?message
WHERE {
  ?event a kt:FaultEvent ;
         kt:affects ?component ;
         kt:startTime ?start ;
         kt:eventStatus ?status ;
         kt:eventMessage ?message .
  OPTIONAL { ?event kt:endTime ?end }
  ?component kt:componentKey ?component_key .
  ?turbine kt:hasComponent ?component ;
           kt:turbineId ?turbine_id .
  ?farm kt:hasTurbine ?turbine ;
        kt:farmCode ?farm_code .
}
ORDER BY ?farm_code ?turbine_id ?start
"""

# SPARQL evaluates UNION branches bottom-up, so each branch repeats the
# turbine pattern (?t) instead of referring to the outer ?turbine.
_T = "?f kt:farmCode ?farm_code ; kt:hasTurbine ?t . ?t kt:turbineId ?turbine_id ."
LINEAGE_QUERY = f"""
SELECT DISTINCT ?s ?p ?o (isIRI(?o) AS ?o_is_iri)
WHERE {{
  ?farm kt:farmCode ?farm_code ;
        kt:hasTurbine ?turbine .
  ?turbine kt:turbineId ?turbine_id .
  {{
    {{ ?s kt:farmCode ?farm_code ; kt:hasTurbine/kt:turbineId ?turbine_id }}
    UNION {{ {_T} BIND (?t AS ?s) }}
    UNION {{ {_T} ?t kt:hasComponent ?s }}
    UNION {{ {_T} ?t kt:hasSensor ?s }}
    UNION {{ {_T} ?t kt:hasComponent ?c . ?s kt:modelOf ?c }}
    UNION {{ {_T} ?t prov:wasDerivedFrom/kt:partOf* ?s }}
    UNION {{ {_T} ?t kt:hasSensor/prov:wasDerivedFrom/kt:partOf* ?s }}
  }}
  ?s ?p ?o .
  FILTER (?p != rdfs:comment)
  FILTER NOT EXISTS {{ ?o a kt:Turbine . FILTER (?o != ?turbine) }}
}}
"""

SCHEMA_CLASSES_QUERY = """
SELECT ?class ?label ?parent
WHERE {
  ?class a rdfs:Class ;
         rdfs:label ?label .
  OPTIONAL { ?class rdfs:subClassOf ?parent }
}
ORDER BY ?class
"""

SCHEMA_PROPERTIES_QUERY = """
SELECT ?property ?label ?domain ?range
WHERE {
  ?property a rdf:Property ;
            rdfs:label ?label ;
            rdfs:domain ?domain ;
            rdfs:range ?range .
}
ORDER BY ?property
"""


def farm_iri(farm_code: str) -> URIRef:
    """IRI of a wind farm, e.g. kt:KLM."""
    return KT[farm_code]


def dataset_iri(farm_code: str) -> URIRef:
    """IRI of a farm's published dataset, e.g. kt:Dataset_KLM."""
    return KT[f"Dataset_{farm_code}"]


def turbine_iri(farm_code: str, turbine_id: str) -> URIRef:
    """IRI of a turbine, e.g. kt:KLM_T01."""
    return KT[f"{farm_code}_{turbine_id}"]


def component_iri(farm_code: str, turbine_id: str, component_key: str) -> URIRef:
    """IRI of a turbine component, e.g. kt:KLM_T01_gearbox."""
    return KT[f"{farm_code}_{turbine_id}_{component_key}"]


def sensor_iri(farm_code: str, turbine_id: str, signal_key: str) -> URIRef:
    """IRI of a turbine sensor, e.g. kt:KLM_T01_gear_oil_temp."""
    return KT[f"{farm_code}_{turbine_id}_{signal_key}"]


def fault_event_iri(farm_code: str, turbine_id: str, number: int) -> URIRef:
    """IRI of a fault event, e.g. kt:FaultEvent_KLM_T01_0001."""
    return KT[f"FaultEvent_{farm_code}_{turbine_id}_{number:04d}"]


def source_iri(file_name: str) -> URIRef:
    """IRI of a raw source file, e.g. kt:Source_Kelmarsh_WT_static_csv."""
    return KT["Source_" + re.sub(r"[^A-Za-z0-9_-]", "_", file_name)]


def decimal(value: float) -> Literal:
    """An xsd:decimal literal."""
    return Literal(Decimal(str(value)), datatype=XSD.decimal)


def datetime_literal(value: datetime) -> Literal:
    """An xsd:dateTime literal from a datetime or pandas Timestamp."""
    return Literal(pd.Timestamp(value).to_pydatetime(), datatype=XSD.dateTime)


def new_graph() -> Graph:
    """An empty graph with the project prefixes bound (overriding defaults)."""
    g = Graph()
    for prefix, namespace in PREFIXES.items():
        g.bind(prefix, namespace, override=True, replace=True)
    return g


def load_graph(path: Path | str) -> Graph:
    """Parse a Turtle file into a graph with the project prefixes bound."""
    g = new_graph()
    g.parse(Path(path), format="turtle")
    return g


def save_graph(g: Graph, path: Path | str) -> None:
    """Write the graph as Turtle with the schema block first."""
    schema_subjects = {s for s, o in g.subject_objects(RDF.type) if o in SCHEMA_TYPES}
    schema, data = new_graph(), new_graph()
    for triple in g:
        (schema if triple[0] in schema_subjects else data).add(triple)
    text = (
        "# Schema\n"
        + schema.serialize(format="turtle")
        + "\n# Instances\n"
        + data.serialize(format="turtle")
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def to_python(term: object) -> object:
    """Convert an rdflib term to a plain Python value."""
    value = term.toPython() if term is not None else None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, URIRef):
        return str(value)
    return value


def result_frame(result: Result) -> pd.DataFrame:
    """Turn a SPARQL SELECT result into a DataFrame."""
    columns = [str(v) for v in result.vars]
    rows = [[to_python(term) for term in row] for row in result]
    return pd.DataFrame(rows, columns=columns)


def query(g: Graph, sparql: str, **bindings: str | None) -> pd.DataFrame:
    """Run a SELECT query with plain-literal bindings (None = unbound)."""
    init = {k: Literal(v) for k, v in bindings.items() if v is not None}
    return result_frame(g.query(sparql, initNs=PREFIXES, initBindings=init))


def get_farms(g: Graph) -> pd.DataFrame:
    """Farms with code, label and their dataset's DOI, publisher, license."""
    return query(g, FARMS_QUERY)


def get_turbines(g: Graph, farm_code: str | None = None) -> pd.DataFrame:
    """Turbines (optionally of one farm) with location, specs and model."""
    return query(g, TURBINES_QUERY, farm_code=farm_code).assign(
        farm_code=lambda f: f["farm_code"] if farm_code is None else farm_code
    )


def get_components(g: Graph, farm_code: str, turbine_id: str) -> pd.DataFrame:
    """Components of one turbine: component key and class IRI."""
    return query(g, COMPONENTS_QUERY, farm_code=farm_code, turbine_id=turbine_id)


def get_sensors(g: Graph, farm_code: str, turbine_id: str) -> pd.DataFrame:
    """Sensors of one turbine: signal key, SCADA column, unit, Parquet path,
    monitored component and observed property."""
    return query(g, SENSORS_QUERY, farm_code=farm_code, turbine_id=turbine_id)


def parquet_path_of(sensors: pd.DataFrame) -> Path:
    """The single Parquet file all sensors of a turbine are stored in."""
    paths = sensors["parquet_path"].unique()
    if len(paths) != 1:
        raise LookupError(f"Expected one Parquet path, got {list(paths)}")
    return Path(paths[0])


def get_fault_events(
    g: Graph, farm_code: str | None = None, turbine_id: str | None = None
) -> pd.DataFrame:
    """Fault events (optionally of one farm/turbine) with UTC start and end."""
    frame = query(g, FAULT_EVENTS_QUERY, farm_code=farm_code, turbine_id=turbine_id)
    return frame.assign(
        farm_code=frame["farm_code"] if farm_code is None else farm_code,
        turbine_id=frame["turbine_id"] if turbine_id is None else turbine_id,
        start=pd.to_datetime(frame["start"], utc=True),
        end=pd.to_datetime(frame["end"], utc=True),
    )


def get_lineage(g: Graph, farm_code: str, turbine_id: str) -> pd.DataFrame:
    """Triples (s, p, o, o_is_iri) of a turbine and everything it traces to."""
    return query(g, LINEAGE_QUERY, farm_code=farm_code, turbine_id=turbine_id)


def get_schema(g: Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Schema classes (with superclass) and properties (with domain, range)."""
    return query(g, SCHEMA_CLASSES_QUERY), query(g, SCHEMA_PROPERTIES_QUERY)
