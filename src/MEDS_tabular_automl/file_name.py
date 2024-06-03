"""Help functions for getting file names and paths for MEDS tabular automl tasks."""
from pathlib import Path

from omegaconf import DictConfig

from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    VALUE_AGGREGATIONS,
)


class FileNameResolver:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @property
    def meds_dir(self):
        return Path(self.cfg.MEDS_cohort_dir)

    @property
    def tabularize_dir(self):
        return Path(self.cfg.tabularized_data_dir)

    @property
    def cache_dir(self):
        return Path(self.cfg.cache_dir)

    def get_meds_dir(self):
        return self.meds_dir / "final_cohort"

    def get_static_dir(self):
        return self.tabularize_dir / "static"

    def get_ts_dir(self):
        return self.tabularize_dir / "ts"

    def get_sparse_dir(self):
        return self.tabularize_dir / "sparse"

    def get_label_dir(self):
        return Path(self.cfg.task_dir)

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
        agg_name = agg.split("/")[-1]
        return self.get_static_dir() / split / f"{shard_num}" / f"{agg_name}.npz"

    def get_flat_ts_rep(self, split: str, shard_num: int, window_size: int, agg: str):
        # Given a shard number, returns the time series representation path
        return self.get_ts_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"

    def get_flat_sparse_rep(self, split: str, shard_num: int, window_size: int, agg: str):
        # Given a shard number, returns the sparse representation path
        return self.get_sparse_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"

    def get_label(self, split: str, shard_num: int):
        # Given a shard number, returns the label path
        return self.get_label_dir() / split / f"{shard_num}.parquet"

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

    def list_label_files(self, split=None):
        # List all label files
        if split:
            return sorted(list(self.get_label_dir().glob(f"{split}/*.parquet")))
        return sorted(list(self.get_label_dir().glob("*/*.parquet")))

    def get_cache_dir(self):
        return self.cache_dir

    def get_model_files(self, window_sizes, aggs, split, shard_num: int):
        # Given a shard number, returns the model files
        model_files = []
        for window_size in window_sizes:
            for agg in aggs:
                if agg.startswith("static"):
                    continue
                else:
                    model_files.append(self.get_task_specific_path(split, shard_num, window_size, agg))
        for agg in aggs:
            if agg.startswith("static"):
                window_size = None
                model_files.append(self.get_task_specific_path(split, shard_num, window_size, agg))
        return sorted(model_files)

    def parse_ts_file_path(self, data_fp):
        agg = f"{data_fp.parent.stem}/{data_fp.stem}"
        if not agg in CODE_AGGREGATIONS + VALUE_AGGREGATIONS:
            raise ValueError(f"Invalid aggregation: {agg}")
        window_size = data_fp.parts[-3]
        shard_num = data_fp.parts[-4]
        split = data_fp.parts[-5]
        return split, shard_num, window_size, agg

    def parse_static_file_path(self, data_fp):
        # parse as static agg
        agg = f"{data_fp.parent.parent.parent.stem}/{data_fp.stem}"
        if not agg in [STATIC_VALUE_AGGREGATION, STATIC_CODE_AGGREGATION]:
            raise ValueError(f"Invalid aggregation: {agg}")
        shard_num = data_fp.parent.stem
        split = data_fp.parts[-3]
        return split, shard_num, agg

    def get_task_specific_path(self, split, shard_num, window_size, agg):
        if window_size:
            return self.get_label_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"
        else:
            return self.get_label_dir() / split / f"{shard_num}" / f"{agg}.npz"
