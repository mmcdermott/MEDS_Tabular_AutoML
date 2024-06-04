"""Help functions for getting file names and paths for MEDS tabular automl tasks."""
from pathlib import Path

from MEDS_tabular_automl.utils import (
    CODE_AGGREGATIONS,
    STATIC_CODE_AGGREGATION,
    STATIC_VALUE_AGGREGATION,
    VALUE_AGGREGATIONS,
)


def get_meds_dir(cfg):
    return cfg.meds_dir / "final_cohort"


def get_static_dir(cfg):
    return cfg.tabularize_dir / "static"


def get_ts_dir(cfg):
    return cfg.tabularize_dir / "ts"


def get_sparse_dir(cfg):
    return cfg.tabularize_dir / "sparse"


def get_label_dir(cfg):
    return Path(cfg.task_dir)


def get_feature_columns_fp(cfg):
    return cfg.tabularize_dir / "feature_columns.json"


def get_feature_freqs_fp(cfg):
    return cfg.tabularize_dir / "feature_freqs.json"


def get_config_path(cfg):
    return cfg.tabularize_dir / "config.yaml"


def get_meds_shard(cfg, split: str, shard_num: int):
    # Given a shard number, return the MEDS format data
    return get_meds_dir(cfg) / split / f"{shard_num}.parquet"


def get_flat_static_rep(cfg, split: str, shard_num: int, agg: str):
    # Given a shard number, returns the static representation path
    agg_name = agg.split("/")[-1]
    return cfg.get_static_dir() / split / f"{shard_num}" / f"{agg_name}.npz"


def get_flat_ts_rep(cfg, split: str, shard_num: int, window_size: int, agg: str):
    # Given a shard number, returns the time series representation path
    return cfg.get_ts_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"


def get_flat_sparse_rep(cfg, split: str, shard_num: int, window_size: int, agg: str):
    # Given a shard number, returns the sparse representation path
    return cfg.get_sparse_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"


def get_label(cfg, split: str, shard_num: int):
    # Given a shard number, returns the label path
    return cfg.get_label_dir() / split / f"{shard_num}.parquet"


def list_meds_files(meds_dir: Path, split=None):
    # List all MEDS files
    if split:
        return sorted(list(meds_dir.glob(f"{split}/*.parquet")))
    return sorted(list(meds_dir.glob("**/*.parquet")))


def list_subdir_files(dir: [Path | str], fmt: str):
    return sorted(list(Path(dir).glob(f"**/*.{fmt}")))


def list_static_files(cfg, split=None):
    # List all static files
    if split:
        return sorted(list(cfg.get_static_dir().glob(f"{split}/*/*.npz")))
    return sorted(list(cfg.get_static_dir().glob("*/*/*.npz")))


def list_ts_files(cfg, split=None):
    # List all time series files
    if split:
        return sorted(list(cfg.get_ts_dir().glob(f"{split}/*/*/*/*.npz")))
    return sorted(list(cfg.get_ts_dir().glob("*/*/*/*/*.npz")))


def list_sparse_files(cfg, split=None):
    # List all sparse files
    if split:
        return sorted(list(cfg.get_sparse_dir().glob(f"{split}/*/*.npz")))
    return sorted(list(cfg.get_sparse_dir().glob("*/*/*.npz")))


def list_label_files(cfg, split=None):
    # List all label files
    if split:
        return sorted(list(cfg.get_label_dir().glob(f"{split}/*.parquet")))
    return sorted(list(cfg.get_label_dir().glob("*/*.parquet")))


def get_cache_dir(cfg):
    return cfg.cache_dir


def get_model_files(cfg, window_sizes, aggs, split, shard_num: int):
    # Given a shard number, returns the model files
    model_files = []
    for window_size in window_sizes:
        for agg in aggs:
            if agg.startswith("static"):
                continue
            else:
                model_files.append(cfg.get_task_specific_path(split, shard_num, window_size, agg))
    for agg in aggs:
        if agg.startswith("static"):
            window_size = None
            model_files.append(cfg.get_task_specific_path(split, shard_num, window_size, agg))
    return sorted(model_files)


def parse_ts_file_path(cfg, data_fp):
    agg = f"{data_fp.parent.stem}/{data_fp.stem}"
    if agg not in CODE_AGGREGATIONS + VALUE_AGGREGATIONS:
        raise ValueError(f"Invalid aggregation: {agg}")
    window_size = data_fp.parts[-3]
    shard_num = data_fp.parts[-4]
    split = data_fp.parts[-5]
    return split, shard_num, window_size, agg


def parse_static_file_path(cfg, data_fp):
    # parse as static agg
    agg = f"{data_fp.parent.parent.parent.stem}/{data_fp.stem}"
    if agg not in [STATIC_VALUE_AGGREGATION, STATIC_CODE_AGGREGATION]:
        raise ValueError(f"Invalid aggregation: {agg}")
    shard_num = data_fp.parent.stem
    split = data_fp.parts[-3]
    return split, shard_num, agg


def get_task_specific_path(cfg, split, shard_num, window_size, agg):
    if window_size:
        return cfg.get_label_dir() / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"
    else:
        return cfg.get_label_dir() / split / f"{shard_num}" / f"{agg}.npz"
