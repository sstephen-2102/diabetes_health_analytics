"""Tests for src/data/loader.py. Downloads are faked; no test touches the network."""
import sys
import types

import pandas as pd
import pytest

import src.data.loader as loader
from src.common.config import FEATURE_COLUMNS, VARIABLE_METADATA
from src.common.exceptions import DataLoadError
from src.data.loader import download_kaggle_dataset, download_uci_dataset, load_dataset


def _valid_value(column):
    meta = VARIABLE_METADATA[column]
    if meta.get("allowed_values"):
        return meta["allowed_values"][0]
    return meta["minimum"] if meta.get("minimum") is not None else 1


def create_uci_frames(n_rows=20):
    """Small features/targets frames shaped like ucimlrepo output (schema-valid, not reference-sized)."""
    features = pd.DataFrame({c: [_valid_value(c)] * n_rows for c in FEATURE_COLUMNS})
    targets = pd.DataFrame({"Diabetes_binary": [i % 2 for i in range(n_rows)]})
    return features, targets


def install_fake_ucimlrepo(monkeypatch, features=None, targets=None, error=None):
    """Put a fake ucimlrepo module in sys.modules; return the list of recorded fetch calls."""
    if features is None:
        features, targets = create_uci_frames()
    calls = []

    def fetch_ucirepo(name=None, id=None):
        calls.append({"name": name, "id": id})
        if error is not None:
            raise error
        return types.SimpleNamespace(data=types.SimpleNamespace(features=features, targets=targets))

    module = types.ModuleType("ucimlrepo")
    module.fetch_ucirepo = fetch_ucirepo
    monkeypatch.setitem(sys.modules, "ucimlrepo", module)
    return calls


def pass_validation(monkeypatch):
    """The real reference check needs all 253,680 rows; stub it for the success-path tests."""
    monkeypatch.setattr(loader, "validate_dataset",
                        lambda data, cols, target: {"is_valid": True, "matches_reference": True})


# ---------------------------------------------------------------- download_uci_dataset

def test_uci_download_saves_features_and_target(tmp_path, monkeypatch):
    calls = install_fake_ucimlrepo(monkeypatch)
    pass_validation(monkeypatch)
    output = tmp_path / "cdc_diabetes.csv"

    result = download_uci_dataset(str(output))

    assert result == str(output)
    saved = pd.read_csv(output)
    assert list(saved.columns) == FEATURE_COLUMNS + ["Diabetes_binary"]
    assert len(saved) == 20
    assert calls == [{"name": None, "id": 891}]


def test_uci_download_passes_dataset_id_by_keyword(tmp_path, monkeypatch):
    calls = install_fake_ucimlrepo(monkeypatch)
    pass_validation(monkeypatch)

    download_uci_dataset(str(tmp_path / "data.csv"), dataset_id=123)

    assert calls[0]["id"] == 123
    assert calls[0]["name"] is None


def test_uci_download_creates_missing_parent_folders(tmp_path, monkeypatch):
    install_fake_ucimlrepo(monkeypatch)
    pass_validation(monkeypatch)
    output = tmp_path / "data" / "raw" / "cdc_diabetes.csv"

    download_uci_dataset(str(output))

    assert output.is_file()


def test_uci_download_output_loads_with_load_dataset(tmp_path, monkeypatch):
    install_fake_ucimlrepo(monkeypatch)
    pass_validation(monkeypatch)
    output = tmp_path / "cdc_diabetes.csv"

    data = load_dataset(download_uci_dataset(str(output)))

    assert data.shape == (20, 22)
    assert data.attrs["dataset_variant"] == "raw_v1"


def test_uci_existing_file_is_kept_and_not_downloaded(tmp_path, monkeypatch):
    calls = install_fake_ucimlrepo(monkeypatch)
    output = tmp_path / "cdc_diabetes.csv"
    output.write_text("original contents")

    result = download_uci_dataset(str(output))

    assert result == str(output)
    assert output.read_text() == "original contents"
    assert calls == []


def test_uci_reference_mismatch_raises_and_writes_nothing(tmp_path, monkeypatch):
    install_fake_ucimlrepo(monkeypatch)
    output = tmp_path / "cdc_diabetes.csv"

    with pytest.raises(DataLoadError, match="reference"):
        download_uci_dataset(str(output))

    assert not output.exists()


def test_uci_missing_column_raises_and_writes_nothing(tmp_path, monkeypatch):
    features, targets = create_uci_frames()
    install_fake_ucimlrepo(monkeypatch, features.drop(columns=["BMI"]), targets)
    output = tmp_path / "cdc_diabetes.csv"

    with pytest.raises(DataLoadError):
        download_uci_dataset(str(output))

    assert not output.exists()


def test_uci_fetch_failure_is_wrapped(tmp_path, monkeypatch):
    install_fake_ucimlrepo(monkeypatch, error=ConnectionError("network down"))

    with pytest.raises(DataLoadError, match="UCI acquisition failed") as info:
        download_uci_dataset(str(tmp_path / "data.csv"))

    assert isinstance(info.value.__cause__, ConnectionError)


def test_uci_missing_library_is_wrapped(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ucimlrepo", None)

    with pytest.raises(DataLoadError, match="UCI acquisition failed"):
        download_uci_dataset(str(tmp_path / "data.csv"))


def test_uci_save_failure_is_wrapped(tmp_path, monkeypatch):
    install_fake_ucimlrepo(monkeypatch)
    pass_validation(monkeypatch)
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")

    with pytest.raises(DataLoadError, match="Cannot save"):
        download_uci_dataset(str(blocker / "data.csv"))


# ---------------------------------------------------------------- load_dataset

def test_load_dataset_reads_numeric_csv_and_tags_variant(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("Diabetes_binary,BMI\n0,25\n1,31\n")

    data = load_dataset(str(path))

    assert data.shape == (2, 2)
    assert data.attrs == {"dataset_variant": "raw_v1", "source_file": "sample.csv"}


def test_load_dataset_strips_excel_byte_order_mark(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("Diabetes_binary,BMI\n0,25\n".encode("utf-8-sig"))

    assert list(load_dataset(str(path)).columns) == ["Diabetes_binary", "BMI"]


@pytest.mark.parametrize("contents, reason", [
    ("Diabetes_binary,BMI,BMI\n0,25,26\n", "duplicate column names"),
    ("Diabetes_binary,BMI\n0,abc\n", "text column"),
    ("Diabetes_binary,BMI\n", "header only"),
    ("Diabetes_binary\n0\n1\n", "single column"),
    ("Diabetes_binary,BMI\n0,25\n1,30,99\n", "row with too many fields"),
    ("", "empty file"),
])
def test_load_dataset_rejects_bad_csv(tmp_path, contents, reason):
    path = tmp_path / "bad.csv"
    path.write_text(contents)

    with pytest.raises(DataLoadError):
        load_dataset(str(path))


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(DataLoadError, match="does not exist"):
        load_dataset(str(tmp_path / "missing.csv"))


def test_load_dataset_folder_path_raises(tmp_path):
    with pytest.raises(DataLoadError):
        load_dataset(str(tmp_path))


# ---------------------------------------------------------------- download_kaggle_dataset

def install_fake_kagglehub(monkeypatch, source_dir):
    module = types.ModuleType("kagglehub")
    module.dataset_download = lambda dataset_id: str(source_dir)
    monkeypatch.setitem(sys.modules, "kagglehub", module)


def test_kaggle_download_copies_csv_files(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.csv").write_text("x,y\n1,2\n")
    (cache / "notes.txt").write_text("ignored")
    install_fake_kagglehub(monkeypatch, cache)
    destination = tmp_path / "raw"

    result = download_kaggle_dataset("owner/dataset", str(destination))

    assert result == str(destination)
    assert (destination / "a.csv").read_text() == "x,y\n1,2\n"
    assert not (destination / "notes.txt").exists()


def test_kaggle_download_skips_identical_existing_file(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.csv").write_text("x,y\n1,2\n")
    install_fake_kagglehub(monkeypatch, cache)
    destination = tmp_path / "raw"
    destination.mkdir()
    (destination / "a.csv").write_text("x,y\n1,2\n")

    download_kaggle_dataset("owner/dataset", str(destination))

    assert (destination / "a.csv").read_text() == "x,y\n1,2\n"


def test_kaggle_download_refuses_to_overwrite_changed_file(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.csv").write_text("x,y\n1,2\n")
    install_fake_kagglehub(monkeypatch, cache)
    destination = tmp_path / "raw"
    destination.mkdir()
    (destination / "a.csv").write_text("edited locally")

    with pytest.raises(DataLoadError, match="Refusing to overwrite"):
        download_kaggle_dataset("owner/dataset", str(destination))

    assert (destination / "a.csv").read_text() == "edited locally"


def test_kaggle_download_without_csv_files_raises(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    install_fake_kagglehub(monkeypatch, cache)

    with pytest.raises(DataLoadError, match="no CSV"):
        download_kaggle_dataset("owner/dataset", str(tmp_path / "raw"))


@pytest.mark.parametrize("bad_id", ["no-slash", "too/many/slashes", 891])
def test_kaggle_download_rejects_bad_identifier(tmp_path, bad_id):
    with pytest.raises(DataLoadError, match="owner/dataset"):
        download_kaggle_dataset(bad_id, str(tmp_path))
