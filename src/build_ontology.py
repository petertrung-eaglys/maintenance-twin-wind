"""Build the wind twin RDF graph: schema, provenance, farms, turbines,
components, sensors and labelled fault events (spec §7.5).

Overwrites the ontology file, so steps that add models or alerts must run
again afterwards.
"""

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdflib import RDF, RDFS, XSD, Graph, Literal, URIRef

from src.config import (
    farm_names,
    farm_processed_dir,
    farm_raw_dir,
    load_config,
    scada_archives,
    scada_dirs,
)
from src.graph_utils import (
    GEO,
    KT,
    PROV,
    QUDT,
    SOSA,
    UNIT,
    component_iri,
    dataset_iri,
    datetime_literal,
    decimal,
    farm_iri,
    fault_event_iri,
    new_graph,
    save_graph,
    sensor_iri,
    source_iri,
    turbine_iri,
)
from src.scada_format import (
    group_by_turbine,
    status_files,
    turbine_data_files,
)
from src.static_data import read_mapping, read_static

logger = logging.getLogger(__name__)

UNIT_SUFFIX = re.compile(r"\s*\([^()]*\)$")

# component key -> class; keys match config fault_labels
COMPONENTS: dict[str, URIRef] = {
    "gearbox": KT.Gearbox,
    "generator": KT.Generator,
    "main_bearing": KT.MainBearing,
    "pitch": KT.PitchSystem,
    "yaw": KT.YawSystem,
    "nacelle": KT.Nacelle,
    "rotor": KT.Rotor,
}

# signal key -> (component key, observable property, QUDT unit)
SIGNAL_SEMANTICS: dict[str, tuple[str, str, URIRef]] = {
    "wind_speed": ("nacelle", "WindSpeed", UNIT["M-PER-SEC"]),
    "wind_direction": ("nacelle", "WindDirection", UNIT.DEG),
    "ambient_temp": ("nacelle", "AmbientTemperature", UNIT.DEG_C),
    "nacelle_temp": ("nacelle", "NacelleTemperature", UNIT.DEG_C),
    "nacelle_direction": ("yaw", "NacelleDirection", UNIT.DEG),
    "pitch": ("pitch", "PitchAngle", UNIT.DEG),
    "rotor_speed": ("rotor", "RotorSpeed", UNIT["REV-PER-MIN"]),
    "power": ("generator", "ActivePower", UNIT.KiloW),
    "generator_bearing_front_temp": ("generator", "BearingTemperature", UNIT.DEG_C),
    "generator_bearing_rear_temp": ("generator", "BearingTemperature", UNIT.DEG_C),
    "stator_temp": ("generator", "StatorTemperature", UNIT.DEG_C),
    "gear_oil_temp": ("gearbox", "OilTemperature", UNIT.DEG_C),
    "gear_oil_inlet_temp": ("gearbox", "OilTemperature", UNIT.DEG_C),
    "front_bearing_temp": ("main_bearing", "BearingTemperature", UNIT.DEG_C),
    "rear_bearing_temp": ("main_bearing", "BearingTemperature", UNIT.DEG_C),
}

# (class, superclass or None, label, comment)
CLASSES: list[tuple[URIRef, URIRef | None, str, str]] = [
    (KT.Dataset, PROV.Entity, "Dataset", "A published dataset of one farm."),
    (KT.SourceFile, PROV.Entity, "Source file", "A raw file of a dataset."),
    (KT.WindFarm, None, "Wind farm", "A wind farm site."),
    (KT.Turbine, None, "Turbine", "One wind turbine."),
    (KT.Component, None, "Component", "Superclass for turbine parts."),
    (KT.Gearbox, KT.Component, "Gearbox", "The drivetrain gearbox."),
    (KT.Generator, KT.Component, "Generator", "The electrical generator."),
    (KT.MainBearing, KT.Component, "Main bearing", "The rotor shaft bearing."),
    (KT.PitchSystem, KT.Component, "Pitch system", "Blade pitch drives."),
    (KT.YawSystem, KT.Component, "Yaw system", "Nacelle yaw drives."),
    (KT.Nacelle, KT.Component, "Nacelle", "The nacelle and its sensors."),
    (KT.Rotor, KT.Component, "Rotor", "The rotor with hub and blades."),
    (KT.Sensor, SOSA.Sensor, "Sensor", "One SCADA signal of a turbine."),
    (KT.Model, PROV.Entity, "Model", "A normal-behaviour model."),
    (KT.PowerModel, KT.Model, "Power model", "Predicts turbine power."),
    (KT.ThermalModel, KT.Model, "Thermal model", "Predicts a temperature."),
    (KT.PitchModel, KT.Model, "Pitch model", "Predicts the pitch angle."),
    (KT.YawModel, KT.Model, "Yaw model", "Estimates yaw misalignment."),
    (KT.Alert, None, "Alert", "A residual-based component health alert."),
    (KT.FaultEvent, None, "Fault event", "A labelled Stop/Warning event."),
]

LITERAL = RDFS.Literal
# (property, domain, range, label, comment)
PROPERTIES: list[tuple[URIRef, URIRef, URIRef, str, str]] = [
    (KT.hasTurbine, KT.WindFarm, KT.Turbine, "has turbine", "A farm turbine."),
    (KT.hasComponent, KT.Turbine, KT.Component, "has component", "A part."),
    (KT.hasSensor, KT.Turbine, KT.Sensor, "has sensor", "A turbine sensor."),
    (KT.monitors, KT.Sensor, KT.Component, "monitors", "Component observed."),
    (SOSA.observes, KT.Sensor, SOSA.ObservableProperty, "observes", "Quantity."),
    (QUDT.unit, KT.Sensor, QUDT.Unit, "unit", "QUDT unit of the signal."),
    (KT.signalKey, KT.Sensor, LITERAL, "signal key", "Parquet column name."),
    (KT.scadaColumn, KT.Sensor, LITERAL, "SCADA column", "Raw column name."),
    (KT.storedIn, KT.Sensor, LITERAL, "stored in", "Parquet file path."),
    (KT.manufacturerTitle, KT.Sensor, LITERAL, "manufacturer title", "Mapping."),
    (KT.greenbyteSignalId, KT.Sensor, LITERAL, "Greenbyte signal id", "Mapping."),
    (KT.farmCode, KT.WindFarm, LITERAL, "farm code", "Short code, e.g. KLM."),
    (KT.turbineId, KT.Turbine, LITERAL, "turbine id", "Id in its farm, e.g. T01."),
    (KT.componentKey, KT.Component, LITERAL, "component key", "e.g. gearbox."),
    (KT.ratedPowerKW, KT.Turbine, LITERAL, "rated power (kW)", "Rated power."),
    (KT.rotorDiameterM, KT.Turbine, LITERAL, "rotor diameter (m)", "Diameter."),
    (KT.hubHeightM, KT.Turbine, LITERAL, "hub height (m)", "Hub height."),
    (KT.manufacturer, KT.Turbine, LITERAL, "manufacturer", "Maker."),
    (KT.turbineModel, KT.Turbine, LITERAL, "turbine model", "e.g. MM92."),
    (KT.commissioningDate, KT.Turbine, LITERAL, "commissioning date", "COD."),
    (GEO.lat, KT.Turbine, LITERAL, "latitude", "WGS84 latitude."),
    (GEO.long, KT.Turbine, LITERAL, "longitude", "WGS84 longitude."),
    (KT.modelOf, KT.Model, KT.Component, "model of", "Modelled component."),
    (
        KT.modelVariant,
        KT.Model,
        LITERAL,
        "model variant",
        "baseline, physics, hybrid or pinn.",
    ),
    (KT.hasInput, KT.Model, KT.Sensor, "has input", "An input signal."),
    (KT.hasTarget, KT.Model, KT.Sensor, "has target", "The predicted signal."),
    (KT.modelFile, KT.Model, LITERAL, "model file", "Saved model path."),
    (KT.raisedBy, KT.Alert, KT.Model, "raised by", "Model behind the alert."),
    (KT.concerns, KT.Alert, KT.Component, "concerns", "Component alerted on."),
    (KT.precedes, KT.Alert, KT.FaultEvent, "precedes", "Anticipated event."),
    (KT.leadTimeHours, KT.Alert, LITERAL, "lead time (h)", "Alert to event."),
    (KT.affects, KT.FaultEvent, KT.Component, "affects", "Faulty component."),
    (KT.startTime, RDFS.Resource, LITERAL, "start time", "Start (UTC)."),
    (KT.endTime, RDFS.Resource, LITERAL, "end time", "End (UTC)."),
    (KT.eventStatus, KT.FaultEvent, LITERAL, "status", "Stop or Warning."),
    (KT.eventCode, KT.FaultEvent, LITERAL, "code", "Controller event code."),
    (KT.eventMessage, KT.FaultEvent, LITERAL, "message", "Event message."),
    (KT.iecCategory, KT.FaultEvent, LITERAL, "IEC category", "IEC 61400-26."),
    (PROV.wasDerivedFrom, PROV.Entity, PROV.Entity, "was derived from", "Source."),
    (KT.partOf, KT.SourceFile, PROV.Entity, "part of", "Containing archive."),
    (KT.fileName, KT.SourceFile, LITERAL, "file name", "Raw file name."),
    (KT.downloadUrl, KT.SourceFile, LITERAL, "download URL", "Zenodo URL."),
    (KT.md5, KT.SourceFile, LITERAL, "MD5", "Checksum of the file."),
    (KT.doi, KT.Dataset, LITERAL, "DOI", "Digital object identifier."),
    (KT.publisher, KT.Dataset, LITERAL, "publisher", "Data owner."),
    (KT.license, KT.Dataset, LITERAL, "license", "Dataset license."),
]


@dataclass(frozen=True)
class FarmSources:
    """Everything read from disk that one farm's graph part is built from."""

    static: pd.DataFrame  # read_static: indexed by turbine id
    mapping: pd.DataFrame  # read_mapping
    data_files: dict[str, list[str]]  # turbine id -> raw data file names
    archive_of: dict[str, str]  # raw file name -> SCADA zip name
    events: pd.DataFrame  # events.parquet rows with a component


def add_schema(g: Graph) -> None:
    """Add classes and properties with labels and comments."""
    for cls, parent, label, comment in CLASSES:
        g.add((cls, RDF.type, RDFS.Class))
        g.add((cls, RDFS.label, Literal(label)))
        g.add((cls, RDFS.comment, Literal(comment)))
        if parent is not None:
            g.add((cls, RDFS.subClassOf, parent))
    for prop, domain, range_, label, comment in PROPERTIES:
        g.add((prop, RDF.type, RDF.Property))
        g.add((prop, RDFS.domain, domain))
        g.add((prop, RDFS.range, range_))
        g.add((prop, RDFS.label, Literal(label)))
        g.add((prop, RDFS.comment, Literal(comment)))


def add_observable_properties(g: Graph) -> None:
    """Declare the observable properties the sensors observe."""
    for _, observable, _ in SIGNAL_SEMANTICS.values():
        g.add((KT[observable], RDF.type, SOSA.ObservableProperty))
        g.add((KT[observable], RDFS.label, Literal(observable)))


def strip_unit(column: str) -> str:
    """Drop a trailing '(unit)' from a SCADA column name."""
    return UNIT_SUFFIX.sub("", column)


def mapping_entry(mapping: pd.DataFrame, column: str) -> dict | None:
    """The mapping row whose Greenbyte title matches the column, if any."""
    hits = mapping[mapping["greenbyte_title"] == strip_unit(column)]
    return None if hits.empty else hits.iloc[0].to_dict()


def add_source_file(g: Graph, name: str, part_of: URIRef, **details: str) -> URIRef:
    """Add a raw source file (optional download URL and MD5) and its container."""
    node = source_iri(name)
    g.add((node, RDF.type, KT.SourceFile))
    g.add((node, RDF.type, PROV.Entity))
    g.add((node, RDFS.label, Literal(name)))
    g.add((node, KT.fileName, Literal(name)))
    g.add((node, KT.partOf, part_of))
    for key, prop in [("url", KT.downloadUrl), ("md5", KT.md5)]:
        if key in details:
            g.add((node, prop, Literal(details[key])))
    return node


def add_dataset(g: Graph, farm_code: str, farm: dict[str, Any]) -> None:
    """Add a farm's dataset, its downloads and the farm node."""
    dataset, info = dataset_iri(farm_code), farm["dataset"]
    g.add((dataset, RDF.type, KT.Dataset))
    g.add((dataset, RDF.type, PROV.Entity))
    g.add((dataset, RDFS.label, Literal(info["title"])))
    g.add((dataset, KT.doi, Literal(info["doi"])))
    g.add((dataset, KT.publisher, Literal(info["publisher"])))
    g.add((dataset, KT.license, Literal(info["license"])))
    for f in farm["downloads"]:
        add_source_file(g, f["name"], dataset, url=f["url"], md5=f["md5"])
    node = farm_iri(farm_code)
    g.add((node, RDF.type, KT.WindFarm))
    g.add((node, RDFS.label, Literal(farm["scada_name"])))
    g.add((node, KT.farmCode, Literal(farm_code)))
    g.add((node, PROV.wasDerivedFrom, dataset))


def add_turbine(
    g: Graph, farm_code: str, turbine_id: str, spec: pd.Series, static_file: str
) -> URIRef:
    """Add a turbine with its specs (from the static file) and components."""
    node = turbine_iri(farm_code, turbine_id)
    g.add((farm_iri(farm_code), KT.hasTurbine, node))
    g.add((node, RDF.type, KT.Turbine))
    g.add((node, KT.turbineId, Literal(turbine_id)))
    g.add((node, RDFS.label, Literal(spec["title"])))
    g.add((node, GEO.lat, decimal(spec["latitude"])))
    g.add((node, GEO.long, decimal(spec["longitude"])))
    g.add((node, KT.ratedPowerKW, decimal(spec["rated_power_kw"])))
    g.add((node, KT.rotorDiameterM, decimal(spec["rotor_diameter_m"])))
    g.add((node, KT.hubHeightM, decimal(spec["hub_height_m"])))
    g.add((node, KT.manufacturer, Literal(spec["manufacturer"])))
    g.add((node, KT.turbineModel, Literal(spec["model"])))
    cod = spec["commercial_operations_date"].date()
    g.add((node, KT.commissioningDate, Literal(cod, datatype=XSD.date)))
    g.add((node, PROV.wasDerivedFrom, source_iri(static_file)))
    for key, cls in COMPONENTS.items():
        part = component_iri(farm_code, turbine_id, key)
        g.add((node, KT.hasComponent, part))
        g.add((part, RDF.type, cls))
        g.add((part, KT.componentKey, Literal(key)))
        g.add((part, RDFS.label, Literal(f"{key} {farm_code} {turbine_id}")))
    return node


def add_sensor(
    g: Graph,
    farm_code: str,
    turbine_id: str,
    key: str,
    column: str,
    parquet_path: str,
    raw_files: list[str],
    mapping: tuple[str, pd.DataFrame],
) -> None:
    """Add one sensor, the component it monitors and its raw sources."""
    component, observable, unit = SIGNAL_SEMANTICS[key]
    sensor = sensor_iri(farm_code, turbine_id, key)
    g.add((turbine_iri(farm_code, turbine_id), KT.hasSensor, sensor))
    g.add((sensor, RDF.type, KT.Sensor))
    g.add((sensor, RDFS.label, Literal(f"{key} {farm_code} {turbine_id}")))
    g.add((sensor, KT.monitors, component_iri(farm_code, turbine_id, component)))
    g.add((sensor, SOSA.observes, KT[observable]))
    g.add((sensor, QUDT.unit, unit))
    g.add((sensor, KT.signalKey, Literal(key)))
    g.add((sensor, KT.scadaColumn, Literal(column)))
    g.add((sensor, KT.storedIn, Literal(parquet_path)))
    for raw_file in raw_files:
        g.add((sensor, PROV.wasDerivedFrom, source_iri(raw_file)))
    mapping_file, table = mapping
    entry = mapping_entry(table, column)
    if entry is not None:
        g.add((sensor, PROV.wasDerivedFrom, source_iri(mapping_file)))
        g.add((sensor, KT.manufacturerTitle, Literal(entry["manufacturer_title"])))
        g.add((sensor, KT.greenbyteSignalId, Literal(int(entry["signal_id"]))))


def add_fault_events(g: Graph, farm_code: str, events: pd.DataFrame) -> int:
    """Add labelled events, numbered per turbine in start order."""
    ordered = events.sort_values(["turbine_id", "start"], kind="stable")
    numbers = ordered.groupby("turbine_id").cumcount() + 1
    for (_, event), number in zip(ordered.iterrows(), numbers, strict=True):
        tid = event["turbine_id"]
        node = fault_event_iri(farm_code, tid, number)
        g.add((node, RDF.type, KT.FaultEvent))
        g.add((node, RDFS.label, Literal(f"{event['message']} {farm_code} {tid}")))
        g.add((node, KT.affects, component_iri(farm_code, tid, event["component"])))
        g.add((node, KT.startTime, datetime_literal(event["start"])))
        if pd.notna(event["end"]):
            g.add((node, KT.endTime, datetime_literal(event["end"])))
        g.add((node, KT.eventStatus, Literal(event["status"])))
        g.add((node, KT.eventCode, Literal(event["code"])))
        g.add((node, KT.eventMessage, Literal(event["message"])))
        if pd.notna(event["iec_category"]):
            g.add((node, KT.iecCategory, Literal(event["iec_category"])))
        g.add((node, PROV.wasDerivedFrom, source_iri(event["source_file"])))
    return len(ordered)


def add_farm(g: Graph, config: dict[str, Any], farm: str, src: FarmSources) -> None:
    """Add one farm: dataset, raw files, turbines, sensors and fault events."""
    farm_config = config["farms"][farm]
    code = farm_config["code"]
    add_dataset(g, code, farm_config)
    for raw_file, archive in src.archive_of.items():
        add_source_file(g, raw_file, source_iri(archive))
    processed = farm_processed_dir(config, farm)
    signals = config["signals"] | config["component_signals"]
    mapping = (farm_config["mapping_file"], src.mapping)
    for tid, spec in src.static.iterrows():
        add_turbine(g, code, tid, spec, farm_config["static_file"])
        parquet = str(processed / f"{tid}.parquet")
        for key, column in signals.items():
            add_sensor(g, code, tid, key, column, parquet, src.data_files[tid], mapping)
    add_fault_events(g, code, src.events)


def build_graph(config: dict[str, Any], sources: dict[str, FarmSources]) -> Graph:
    """Build the full graph from each farm's sources and the config."""
    g = new_graph()
    add_schema(g)
    add_observable_properties(g)
    for farm, src in sources.items():
        add_farm(g, config, farm, src)
    return g


def archive_members(raw_dir: Path, archives: list[dict[str, str]]) -> dict[str, str]:
    """Map each file inside the farm's SCADA zips to its zip name."""
    return {
        member: a["name"]
        for a in archives
        for member in zipfile.ZipFile(raw_dir / a["name"]).namelist()
        if not member.endswith("/")
    }


def read_farm_sources(config: dict[str, Any], farm: str) -> FarmSources:
    """Read the static, mapping, file lists and labelled events of a farm."""
    farm_config = config["farms"][farm]
    raw_dir = farm_raw_dir(config, farm)
    folders = scada_dirs(config, farm)
    files = group_by_turbine(turbine_data_files(folders), farm_config["scada_name"])
    events = pd.read_parquet(farm_processed_dir(config, farm) / "events.parquet")
    archive_of = archive_members(raw_dir, scada_archives(config, farm))
    used = {p.name for p in turbine_data_files(folders) + status_files(folders)}
    return FarmSources(
        static=read_static(raw_dir / farm_config["static_file"]),
        mapping=read_mapping(raw_dir / farm_config["mapping_file"]),
        data_files={t: [p.name for p in ps] for t, ps in files.items()},
        archive_of={f: a for f, a in archive_of.items() if f in used},
        events=events.dropna(subset=["component"]),
    )


def main() -> None:
    """Build the graph from data/raw and data/processed and save it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    sources = {farm: read_farm_sources(config, farm) for farm in farm_names(config)}
    g = build_graph(config, sources)
    save_graph(g, config["paths"]["ontology"])
    logger.info("Wrote %d triples to %s", len(g), config["paths"]["ontology"])


if __name__ == "__main__":
    main()
