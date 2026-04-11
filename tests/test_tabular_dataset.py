"""Tests for TabularDataset correctness and error handling.

Tests that verify actual behavior rather than just that mocks were called.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from mixins import TimeableMixin

from MEDS_tabular_automl.tabular_dataset import TabularDataset


def _make_dataset(tmp_path=None, labels=None, num_shards=1):
    """Create a minimal TabularDataset bypassing __init__ for unit testing."""
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
    TimeableMixin.__init__(ds)
    return ds


# ============================================================================
# __init__ validation
# ============================================================================


def test_init_no_labels_directory(tmp_path):
    """Empty label directory raises ValueError with path info."""
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)

    cfg = MagicMock()
    cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    with pytest.raises(ValueError, match="No labels found"):
        TabularDataset(cfg, "train")


# ============================================================================
# _load_ids_and_labels correctness
# ============================================================================


def test_load_labels_returns_correct_values(tmp_path):
    """_load_labels returns actual label values from parquet, not just keys."""
    ds = _make_dataset(tmp_path)
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    pl.DataFrame({"subject_id": [1, 2], "boolean_value": [True, False]}).write_parquet(
        label_dir / "shard_0.parquet"
    )
    ds.cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    labels = ds._load_labels()
    assert labels["shard_0"].to_list() == [True, False]


def test_load_event_ids_adds_row_index_when_missing(tmp_path):
    """When event_id column is absent, _load_ids_and_labels adds a row index."""
    ds = _make_dataset(tmp_path)
    label_dir = tmp_path / "labels" / "train"
    label_dir.mkdir(parents=True)
    # No event_id column
    pl.DataFrame({"subject_id": [1, 2], "boolean_value": [True, False]}).write_parquet(
        label_dir / "shard_0.parquet"
    )
    ds.cfg.path.input_label_cache_dir = str(tmp_path / "labels")

    event_ids = ds._load_event_ids()
    # Should have auto-generated 0-based event_ids
    assert event_ids["shard_0"].to_list() == [0, 1]


# ============================================================================
# _set_imputer / _set_scaler — verify the actual fitting happens correctly
# ============================================================================


def test_set_imputer_with_fit_receives_correct_data(tmp_path):
    """Imputer.fit is called with the actual sparse matrix from shard 0."""
    ds = _make_dataset(tmp_path)
    test_matrix = sp.csc_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    ds._get_shard_by_index = MagicMock(return_value=(test_matrix, np.array([0, 1])))

    mock_imputer = MagicMock()
    mock_imputer.fit = MagicMock()
    del mock_imputer.partial_fit
    ds.cfg.data_loading_params.imputer.imputer_target = mock_imputer

    ds._set_imputer()

    # Verify fit was called with the actual test matrix, not some other data
    call_args = mock_imputer.fit.call_args
    assert (call_args[0][0].toarray() == test_matrix.toarray()).all()


def test_set_scaler_partial_fit_iterates_all_shards(tmp_path):
    """Scaler.partial_fit is called once per shard with shard-specific data."""
    shard_matrices = [
        sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])),
        sp.csc_matrix(np.array([[5.0, 6.0], [7.0, 8.0]])),
    ]
    ds = _make_dataset(tmp_path, num_shards=2)
    ds._get_shard_by_index = MagicMock(
        side_effect=[(shard_matrices[0], np.array([0, 1])), (shard_matrices[1], np.array([1, 0]))]
    )

    mock_scaler = MagicMock()
    mock_scaler.partial_fit = MagicMock()
    ds.cfg.data_loading_params.normalization.normalizer = mock_scaler

    ds._set_scaler()

    assert mock_scaler.partial_fit.call_count == 2
    # Verify each call received the right matrix
    first_call_matrix = mock_scaler.partial_fit.call_args_list[0][0][0]
    second_call_matrix = mock_scaler.partial_fit.call_args_list[1][0][0]
    assert (first_call_matrix.toarray() == shard_matrices[0].toarray()).all()
    assert (second_call_matrix.toarray() == shard_matrices[1].toarray()).all()


def test_set_imputer_no_fit_method_raises(tmp_path):
    """Imputer without fit or partial_fit raises ValueError."""
    ds = _make_dataset(tmp_path)
    ds.cfg.data_loading_params.imputer.imputer_target = object()
    with pytest.raises(ValueError, match="Imputer must have a fit or partial_fit method"):
        ds._set_imputer()


def test_set_scaler_no_fit_method_raises(tmp_path):
    """Scaler without fit or partial_fit raises ValueError."""
    ds = _make_dataset(tmp_path)
    ds.cfg.data_loading_params.normalization.normalizer = object()
    with pytest.raises(ValueError, match="Scaler must have a fit or partial_fit method"):
        ds._set_scaler()


# ============================================================================
# _impute_and_scale_data — verify data flows correctly through pipeline
# ============================================================================


def test_impute_and_scale_chains_correctly():
    """Output of imputer.transform is passed as input to scaler.transform."""
    ds = _make_dataset()

    input_data = sp.csc_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    imputed_data = sp.csc_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))  # after imputation
    scaled_data = sp.csc_matrix(np.array([[0.5, 1.0], [1.5, 2.0]]))  # after scaling

    ds.imputer = MagicMock()
    ds.imputer.transform = MagicMock(return_value=imputed_data)
    ds.scaler = MagicMock()
    ds.scaler.transform = MagicMock(return_value=scaled_data)

    result = ds._impute_and_scale_data(input_data)

    # Imputer receives the raw input
    ds.imputer.transform.assert_called_once_with(input_data)
    # Scaler receives the imputer's output, not the raw input
    ds.scaler.transform.assert_called_once_with(imputed_data)
    # Final result is the scaler's output
    assert (result.toarray() == scaled_data.toarray()).all()


def test_impute_and_scale_passthrough_when_none():
    """Without imputer or scaler, data passes through unchanged."""
    ds = _make_dataset()
    data = sp.csc_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    result = ds._impute_and_scale_data(data)
    assert (result.toarray() == data.toarray()).all()


# ============================================================================
# _get_dynamic_shard_by_index
# ============================================================================


def test_get_dynamic_shard_missing_files_lists_which(tmp_path):
    """ValueError message includes the specific missing file paths."""
    ds = _make_dataset(tmp_path)
    missing = tmp_path / "nonexistent.npz"
    existing = tmp_path / "exists.npz"
    existing.touch()

    with (
        patch("MEDS_tabular_automl.tabular_dataset.get_model_files", return_value=[existing, missing]),
        pytest.raises(ValueError, match="nonexistent.npz"),
    ):
        ds._get_dynamic_shard_by_index(0)


# ============================================================================
# _filter_shard_on_codes_and_freqs — verify column filtering is correct
# ============================================================================


def test_filter_shard_selects_correct_columns():
    """Code mask [True, False, True] keeps columns 0 and 2, drops column 1."""
    ds = _make_dataset()
    ds.code_masks = {"code/count": [True, False, True]}

    data = sp.csc_matrix(np.array([[10, 20, 30], [40, 50, 60]]))
    result = ds._filter_shard_on_codes_and_freqs("code/count", data)

    expected = np.array([[10, 30], [40, 60]])
    assert result.shape == (2, 2)
    assert (result.toarray() == expected).all()


def test_filter_shard_passthrough_when_no_codes():
    """When codes_set is None, all columns pass through."""
    ds = _make_dataset()
    ds.codes_set = None

    data = sp.csc_matrix(np.eye(3))
    result = ds._filter_shard_on_codes_and_freqs("code/count", data)
    assert (result.toarray() == data.toarray()).all()


# ============================================================================
# get_classes — verify correctness of label aggregation
# ============================================================================


def test_get_classes_aggregates_across_shards():
    """get_classes returns unique labels from all shards combined."""
    ds = _make_dataset(num_shards=2)
    ds.labels = {"shard_0": [0, 1, 1], "shard_1": [2, 0]}

    classes = ds.get_classes()
    np.testing.assert_array_equal(sorted(classes), [0, 1, 2])


def test_get_classes_single_class():
    """get_classes with uniform labels returns a single value."""
    ds = _make_dataset()
    ds.labels = {"shard_0": [1, 1, 1]}

    classes = ds.get_classes()
    np.testing.assert_array_equal(classes, [1])
