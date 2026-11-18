"""Tests for the config loader, farm path helpers and the shipped config.yaml."""

from pathlib import Path
from urllib.parse import quote

from src.config import (
    farm_names,
    farm_processed_dir,
    farm_raw_dir,
    load_config,
    scada_archives,
    scada_dirs,
)

CORE_SIGNAL_KEYS = {
    "wind_speed",
    "power",
    "wind_direction",
    "nacelle_direction",
    "ambient_temp",
    "pitch",
    "rotor_speed",
}

SYNTHETIC = {
    "paths": {"raw": "raw"},
    "farms": {
        "beta": {
            "downloads": [
                {"name": "static.csv"},
                {"name": "y1_a.zip", "extract_to": "scada_y1"},
                {"name": "y1_b.zip", "extract_to": "scada_y1"},
                {"name": "y2.zip", "extract_to": "scada_y2"},
            ]
        },
        "alpha": {"downloads": []},
    },
}


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("years: [2016]\nalerts:\n  window: 144\n", encoding="utf-8")

    assert load_config(path) == {"years": [2016], "alerts": {"window": 144}}


def test_farm_names_keep_config_order() -> None:
    assert farm_names(SYNTHETIC) == ["beta", "alpha"]


def test_farm_raw_dir_is_farm_folder_under_raw() -> None:
    assert farm_raw_dir(SYNTHETIC, "beta") == Path("raw/beta")


def test_farm_processed_dir_is_farm_folder_under_processed() -> None:
    config = {"paths": {"processed": "processed"}}

    assert farm_processed_dir(config, "beta") == Path("processed/beta")


def test_scada_archives_are_entries_with_extract_to() -> None:
    names = [f["name"] for f in scada_archives(SYNTHETIC, "beta")]

    assert names == ["y1_a.zip", "y1_b.zip", "y2.zip"]


def test_scada_dirs_list_shared_folders_once() -> None:
    assert scada_dirs(SYNTHETIC, "beta") == [
        Path("raw/beta/scada_y1"),
        Path("raw/beta/scada_y2"),
    ]


def test_shipped_config_has_both_farms_with_codes() -> None:
    farms = load_config()["farms"]

    assert {name: f["code"] for name, f in farms.items()} == {
        "kelmarsh": "KLM",
        "penmanshiel": "PEN",
    }


def test_shipped_config_download_counts_per_farm() -> None:
    farms = load_config()["farms"]

    assert {name: len(f["downloads"]) for name, f in farms.items()} == {
        "kelmarsh": 11,
        "penmanshiel": 22,
    }


def test_shipped_downloads_have_md5_and_url_of_their_zenodo_record() -> None:
    for farm in load_config()["farms"].values():
        record = farm["dataset"]["doi"].split("zenodo.")[1]
        for f in farm["downloads"]:
            expected = (
                f"https://zenodo.org/records/{record}/files/"
                f"{quote(f['name'])}?download=1"
            )
            assert (f["url"], len(f["md5"])) == (expected, 32), f["name"]


def test_shipped_scada_folders_match_years_for_every_farm() -> None:
    config = load_config()
    expected = [f"scada_{y}" for y in config["years"]]

    for farm in farm_names(config):
        assert [d.name for d in scada_dirs(config, farm)] == expected, farm


def test_shipped_datasets_are_cc_by_from_cubico() -> None:
    datasets = [f["dataset"] for f in load_config()["farms"].values()]

    assert {(d["publisher"], d["license"]) for d in datasets} == {
        ("Cubico Sustainable Investments Ltd", "CC-BY-4.0")
    }


def test_shipped_config_declares_core_signals() -> None:
    assert set(load_config()["signals"]) == CORE_SIGNAL_KEYS


def test_shipped_fault_labels_map_each_message_to_one_component() -> None:
    labels = load_config()["fault_labels"]
    messages = [m for ms in labels.values() for m in ms]

    assert (set(labels), len(messages)) == (
        {"gearbox", "generator", "pitch", "yaw"},
        len(set(messages)),
    )


def test_shipped_config_trains_up_to_end_of_2021() -> None:
    assert load_config()["train_test_split"]["train_end"] == "2021-12-31 23:50"
