"""Helper functions for getting file names and paths for MEDS tabular automl tasks."""
from pathlib import Path


def list_subdir_files(dir: Path | str, fmt: str) -> list[Path]:
    """List files in subdirectories of a directory with a given extension.

    Args:
        dir: Path to the directory.
        fmt: File extension to filter files.

    Returns:
        An alphabetically sorted list of Path objects to files matching the extension in any level of
        subdirectories of the given directory.
    """

    return sorted(list(Path(dir).glob(f"**/*.{fmt}")))


def get_task_specific_path(cfg, split, shard_num, window_size, agg):
    return Path(cfg.input_dir) / split / f"{shard_num}" / f"{window_size}" / f"{agg}.npz"


def get_model_files(cfg, split: str, shard_num: int):
    window_sizes = cfg.tabularization.window_sizes
    aggs = cfg.tabularization.aggs
    # Given a shard number, returns the model files
    model_files = []
    for window_size in window_sizes:
        for agg in aggs:
            if agg.startswith("static"):
                continue
            else:
                model_files.append(get_task_specific_path(cfg, split, shard_num, window_size, agg))
    for agg in aggs:
        if agg.startswith("static"):
            window_size = "none"
            model_files.append(get_task_specific_path(cfg, split, shard_num, window_size, agg))
    return sorted(model_files)
