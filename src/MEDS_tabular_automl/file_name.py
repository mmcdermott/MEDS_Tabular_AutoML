"""Help functions for getting file names and paths for MEDS tabular automl tasks."""
from pathlib import Path

from omegaconf import DictConfig


class FileNameResolver:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.meds_dir = Path(cfg.MEDS_cohort_dir)
        self.tabularize_dir = Path(cfg.tabularized_data_dir)

    def get_meds_dir(self):
        return self.meds_dir / "final_cohort"

    def get_static_dir(self):
        return self.tabularize_dir / "static"

    def get_ts_dir(self):
        return self.tabularize_dir / "ts"

    def get_sparse_dir(self):
        return self.tabularize_dir / "sparse"

    def get_feature_columns_fp(self):
        return self.tabularize_dir / "feature_columns.json"

    def get_feature_freqs_fp(self):
        return self.tabularize_dir / "feature_freqs.json"

    def get_config_path(self):
        return self.tabularize_dir / "config.yaml"

    def get_meds_shard(self, split: str, shard_num: int):
        # Given a shard number, return the MEDS format data
        return self.get_meds_dir() / split / f"{shard_num}.parquet"

    def get_flat_static_rep(self, split: str, shard_num: int, agg: str):
        # Given a shard number, returns the static representation path
        return self.get_static_dir() / split / f"{shard_num}" / f"{agg}.npz"

    def get_flat_ts_rep(self, split: str, shard_num: int, window_size: int, agg: str):
        # Given a shard number, returns the time series representation path
        return self.get_ts_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"

    def get_flat_sparse_rep(self, split: str, shard_num: int, window_size: int, agg: str):
        # Given a shard number, returns the sparse representation path
        return self.get_sparse_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"

    def list_meds_files(self, split=None):
        # List all MEDS files
        if split:
            return sorted(list(self.get_meds_dir().glob(f"{split}/*.parquet")))
        return sorted(list(self.get_meds_dir().glob("*/*.parquet")))

    def list_static_files(self, split=None):
        # List all static files
        if split:
            return sorted(list(self.get_static_dir().glob(f"{split}/*/*.npz")))
        return sorted(list(self.get_static_dir().glob("*/*/*.npz")))

    def list_ts_files(self, split=None):
        # List all time series files
        if split:
            return sorted(list(self.get_ts_dir().glob(f"{split}/*/*/*/*.npz")))
        return sorted(list(self.get_ts_dir().glob("*/*/*/*/*.npz")))

    def list_sparse_files(self, split=None):
        # List all sparse files
        if split:
            return sorted(list(self.get_sparse_dir().glob(f"{split}/*/*.npz")))
        return sorted(list(self.get_sparse_dir().glob("*/*/*.npz")))