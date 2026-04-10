"""Tests for TabularDataset error paths and edge cases."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from mixins import TimeableMixin

from MEDS_tabular_automl.tabular_dataset import TabularDataset


def _make_minimal_dataset(tmp_path, labels=None, num_shards=1):
    """Helper to create a minimal TabularDataset via __new__ + manual attribute setup.

    This bypasses __init__ to test individual methods without the full pipeline.
    """
    ds = TabularDataset.__new__(TabularDataset)
    ds.cfg = MagicMock()
    ds.split = "train"
    ds._data_shards = [f"shard_{i}" for i in range(num_shards)]
    ds.codes_set = {0, 1, 2}
    ds.code_masks = {"code/count": [True, True, True]}
    ds.num_features = 3
    ds.valid_event_ids = None
    ds.labels = labels
    ds.imputer = None
    ds.scaler = None
    # TimeableMixin uses properties; initialize via the mixin's own init
    TimeableMixin.__init__(ds)
    return ds


# ============================================================================
# __init__ error paths
# ============================================================================


def test_init_no_labels_directory(tmp_path):
    """Test ValueError when label cache directory has no parquet files."""
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)

    cfg = MagicMock()
    cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    with pytest.raises(ValueError, match="No labels found"):
        TabularDataset(cfg, "train")


def test_init_empty_labels(tmp_path):
    """Test ValueError when loaded labels are empty."""
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    # Create a parquet file so shards are found
    pl.DataFrame(
        {"subject_id": [1], "boolean_value": [True], "prediction_time": ["2021-01-01"]}
    ).write_parquet(label_dir / "0.parquet")

    cfg = MagicMock()
    cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    # Mock _get_code_set and _set_scaler/_set_imputer to bypass setup
    with (
        patch.object(TabularDataset, "_get_code_set", return_value=({0}, {"code/count": [True]}, 1)),
        patch.object(TabularDataset, "_set_scaler"),
        patch.object(TabularDataset, "_set_imputer"),
        patch.object(TabularDataset, "_load_ids_and_labels", return_value=({}, {})),
        pytest.raises(ValueError, match="No labels found"),
    ):
        TabularDataset(cfg, "train")


# ============================================================================
# _load_ids_and_labels paths
# ============================================================================


def test_load_labels_only(tmp_path):
    """Test _load_labels helper which calls _load_ids_and_labels(load_ids=False)."""
    ds = _make_minimal_dataset(tmp_path)

    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1], "boolean_value": [True]}).write_parquet(label_dir / "shard_0.parquet")

    ds.cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    result = ds._load_labels()
    assert "shard_0" in result


def test_load_event_ids_only(tmp_path):
    """Test _load_event_ids helper which calls _load_ids_and_labels(load_labels=False)."""
    ds = _make_minimal_dataset(tmp_path)

    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1], "boolean_value": [True]}).write_parquet(label_dir / "shard_0.parquet")

    ds.cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    result = ds._load_event_ids()
    assert "shard_0" in result


# ============================================================================
# _get_approximate_correlation_per_feature
# ============================================================================


def test_correlation_single_class_labels():
    """Test ValueError when labels have only one unique value."""
    ds = _make_minimal_dataset(Path("/tmp"))

    X = sp.csc_matrix(np.array([[1, 2], [3, 4], [5, 6]]))
    y = np.array([1, 1, 1])  # all same class

    with pytest.raises(ValueError, match="Labels have only one unique value"):
        ds._get_approximate_correlation_per_feature(X, y)


def test_correlation_valid():
    """Test correlation calculation with valid data."""
    ds = _make_minimal_dataset(Path("/tmp"))

    X = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    y = np.array([1, 0, 1, 0])

    corrs = ds._get_approximate_correlation_per_feature(X, y)
    assert corrs.shape == (2,)
    assert abs(corrs[0]) > 0.5  # strong positive correlation
    assert abs(corrs[1]) > 0.5  # strong negative correlation


# ============================================================================
# _set_imputer / _set_scaler
# ============================================================================


def test_set_imputer_no_fit_method(tmp_path):
    """Test ValueError when imputer has neither fit nor partial_fit."""
    ds = _make_minimal_dataset(tmp_path)
    ds.cfg.data_loading_params.imputer.imputer_target = object()  # no fit or partial_fit

    with pytest.raises(ValueError, match="Imputer must have a fit or partial_fit method"):
        ds._set_imputer()


def test_set_scaler_no_fit_method(tmp_path):
    """Test ValueError when scaler has neither fit nor partial_fit."""
    ds = _make_minimal_dataset(tmp_path)
    ds.cfg.data_loading_params.normalization.normalizer = object()  # no fit or partial_fit

    with pytest.raises(ValueError, match="Scaler must have a fit or partial_fit method"):
        ds._set_scaler()


def test_set_imputer_with_fit(tmp_path):
    """Test that imputer with fit method is set correctly."""
    ds = _make_minimal_dataset(tmp_path)
    mock_imputer = MagicMock()
    mock_imputer.fit = MagicMock()
    del mock_imputer.partial_fit  # ensure partial_fit doesn't exist
    ds.cfg.data_loading_params.imputer.imputer_target = mock_imputer
    ds._get_shard_by_index = MagicMock(return_value=(sp.csc_matrix(np.eye(3)), np.array([1, 0, 1])))

    ds._set_imputer()
    assert ds.imputer is mock_imputer
    mock_imputer.fit.assert_called_once()


def test_set_scaler_with_partial_fit(tmp_path):
    """Test that scaler with partial_fit is called for each shard."""
    ds = _make_minimal_dataset(tmp_path, num_shards=2)
    mock_scaler = MagicMock()
    mock_scaler.partial_fit = MagicMock()
    ds.cfg.data_loading_params.normalization.normalizer = mock_scaler
    ds._get_shard_by_index = MagicMock(return_value=(sp.csc_matrix(np.eye(3)), np.array([1, 0, 1])))

    ds._set_scaler()
    assert ds.scaler is mock_scaler
    assert mock_scaler.partial_fit.call_count == 2


# ============================================================================
# _impute_and_scale_data
# ============================================================================


def test_impute_and_scale_data_both():
    """Test that imputer and scaler are both applied when set."""
    ds = _make_minimal_dataset(Path("/tmp"))
    ds.imputer = MagicMock()
    ds.imputer.transform = MagicMock(return_value=sp.csc_matrix(np.eye(3)))
    ds.scaler = MagicMock()
    ds.scaler.transform = MagicMock(return_value=sp.csc_matrix(np.eye(3)))

    data = sp.csc_matrix(np.eye(3))
    ds._impute_and_scale_data(data)

    ds.imputer.transform.assert_called_once()
    ds.scaler.transform.assert_called_once()


def test_impute_and_scale_data_neither():
    """Test passthrough when neither imputer nor scaler is set."""
    ds = _make_minimal_dataset(Path("/tmp"))
    data = sp.csc_matrix(np.eye(3))
    result = ds._impute_and_scale_data(data)
    assert (result.toarray() == data.toarray()).all()


# ============================================================================
# _get_dynamic_shard_by_index
# ============================================================================


def test_get_dynamic_shard_missing_files(tmp_path):
    """Test ValueError when required shard files don't exist."""
    ds = _make_minimal_dataset(tmp_path)
    # Return paths that don't exist
    ds.cfg.path = MagicMock()
    fake_files = [tmp_path / "nonexistent_1.npz", tmp_path / "nonexistent_2.npz"]

    with (
        patch("MEDS_tabular_automl.tabular_dataset.get_model_files", return_value=fake_files),
        pytest.raises(ValueError, match="Not all files exist"),
    ):
        ds._get_dynamic_shard_by_index(0)


# ============================================================================
# _get_shard_by_index - labels reload
# ============================================================================


def test_get_shard_reloads_labels_when_none(tmp_path):
    """Test that labels are reloaded when self.labels is None."""
    ds = _make_minimal_dataset(tmp_path)
    ds.labels = None
    ds._load_labels = MagicMock(return_value={"shard_0": np.array([1, 0])})
    ds._get_dynamic_shard_by_index = MagicMock(return_value=sp.csc_matrix(np.eye(2)))

    ds._get_shard_by_index(0)
    ds._load_labels.assert_called_once()


# ============================================================================
# get_data_shards
# ============================================================================


def test_get_data_shards_empty():
    """Test ValueError when no data in shards."""
    ds = _make_minimal_dataset(Path("/tmp"))
    ds._data_shards = []

    with pytest.raises((ValueError, IndexError)):
        ds.get_data_shards([])


# ============================================================================
# get_classes
# ============================================================================


def test_get_classes():
    """Test get_classes returns unique labels."""
    ds = _make_minimal_dataset(Path("/tmp"))
    ds.labels = {"shard_0": [1, 0, 1], "shard_1": [0, 0]}

    classes = ds.get_classes()
    np.testing.assert_array_equal(sorted(classes), [0, 1])


# ============================================================================
# _filter_shard_on_codes_and_freqs
# ============================================================================


def test_filter_shard_no_codes_set():
    """Test passthrough when codes_set is None."""
    ds = _make_minimal_dataset(Path("/tmp"))
    ds.codes_set = None

    data = sp.csc_matrix(np.eye(3))
    result = ds._filter_shard_on_codes_and_freqs("code/count", data)
    assert (result.toarray() == data.toarray()).all()


def test_filter_shard_with_mask():
    """Test that code mask correctly filters columns."""
    ds = _make_minimal_dataset(Path("/tmp"))
    ds.code_masks = {"code/count": [True, False, True]}

    data = sp.csc_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
    result = ds._filter_shard_on_codes_and_freqs("code/count", data)
    assert result.shape == (2, 2)
    assert (result.toarray() == np.array([[1, 3], [4, 6]])).all()
