"""Tests for the download helpers (no network access)."""

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from src.data.download import (
    download_farm,
    ensure_file,
    extract,
    md5_of,
    needs_download,
)

CONTENT = b"turbine"
CONTENT_MD5 = hashlib.md5(CONTENT).hexdigest()


def write_fetcher(contents: dict[str, bytes], calls: list[str]):
    """Return a fake fetcher that records URLs and writes contents[url]."""

    def fetcher(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(contents[url])

    return fetcher


def zip_bytes(members: dict[str, str]) -> bytes:
    """An in-memory zip archive holding the given name -> text members."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return buffer.getvalue()


def test_md5_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(CONTENT)

    assert md5_of(path) == CONTENT_MD5


def test_needs_download_when_file_missing(tmp_path: Path) -> None:
    assert needs_download(tmp_path / "missing.csv", CONTENT_MD5)


def test_needs_download_when_md5_differs(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_bytes(b"corrupted")

    assert needs_download(path, CONTENT_MD5)


def test_no_download_needed_when_md5_matches(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_bytes(CONTENT)

    assert not needs_download(path, CONTENT_MD5)


def test_ensure_file_downloads_once_then_skips(tmp_path: Path) -> None:
    calls: list[str] = []
    fetcher = write_fetcher({"http://x/f": CONTENT}, calls)

    first = ensure_file("f.csv", "http://x/f", CONTENT_MD5, tmp_path, fetcher)
    second = ensure_file("f.csv", "http://x/f", CONTENT_MD5, tmp_path, fetcher)

    assert (first, second, calls) == (True, False, ["http://x/f"])


def test_ensure_file_raises_on_md5_mismatch(tmp_path: Path) -> None:
    fetcher = write_fetcher({"http://x/f": b"wrong"}, [])

    with pytest.raises(ValueError, match="MD5 mismatch"):
        ensure_file("f.csv", "http://x/f", CONTENT_MD5, tmp_path, fetcher)


def test_extract_unzips_nested_members(tmp_path: Path) -> None:
    archive = tmp_path / "x.zip"
    archive.write_bytes(zip_bytes({"sub/a.csv": "a", "b.csv": "b"}))

    assert extract(archive, tmp_path / "out")
    assert (tmp_path / "out/sub/a.csv").read_text() == "a"


def test_extract_skips_when_already_extracted(tmp_path: Path) -> None:
    archive = tmp_path / "x.zip"
    archive.write_bytes(zip_bytes({"a.csv": "a"}))
    extract(archive, tmp_path / "out")

    assert not extract(archive, tmp_path / "out")


def farm_config(raw: Path) -> tuple[dict, dict[str, bytes]]:
    """A one-farm config with a CSV and two archives sharing one folder."""
    contents = {
        "u/static": b"static",
        "u/a": zip_bytes({"T01.csv": "1"}),
        "u/b": zip_bytes({"T11.csv": "11"}),
    }
    downloads = [
        {"name": "static.csv", "url": "u/static"},
        {"name": "a.zip", "url": "u/a", "extract_to": "scada_2016"},
        {"name": "b.zip", "url": "u/b", "extract_to": "scada_2016"},
    ]
    for d in downloads:
        d["md5"] = hashlib.md5(contents[d["url"]]).hexdigest()
    config = {"paths": {"raw": str(raw)}, "farms": {"pen": {"downloads": downloads}}}
    return config, contents


def test_download_farm_fills_farm_folder_and_shared_scada_folder(
    tmp_path: Path,
) -> None:
    config, contents = farm_config(tmp_path)

    counts = download_farm(config, "pen", write_fetcher(contents, []))

    assert counts == (3, 2)
    assert sorted(p.name for p in (tmp_path / "pen/scada_2016").iterdir()) == [
        "T01.csv",
        "T11.csv",
    ]


def test_download_farm_rerun_fetches_and_extracts_nothing(tmp_path: Path) -> None:
    config, contents = farm_config(tmp_path)
    calls: list[str] = []
    download_farm(config, "pen", write_fetcher(contents, calls))

    assert download_farm(config, "pen", write_fetcher(contents, calls)) == (0, 0)
    assert len(calls) == 3
