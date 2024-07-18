"""Helper functions for getting file names and paths for MEDS tabular automl tasks."""
from pathlib import Path

from omegaconf import DictConfig


def list_subdir_files(root: Path | str, ext: str) -> list[Path]:
    """List files in subdirectories of a directory with a given extension.

    Args:
        root: Path to the directory.
        ext: File extension to filter files.

    Returns:
        An alphabetically sorted list of Path objects to files matching the extension in any level of
        subdirectories of the given directory.

    Examples:
        >>> import tempfile
        >>> tmpdir = tempfile.TemporaryDirectory()
        >>> root = Path(tmpdir.name)
        >>> subdir_1 = root / "subdir_1"
        >>> subdir_1.mkdir()
        >>> subdir_2 = root / "subdir_2"
        >>> subdir_2.mkdir()
        >>> subdir_1_A = subdir_1 / "A"
        >>> subdir_1_A.mkdir()
        >>> (root / "1.csv").touch()
        >>> (root / "foo.parquet").touch()
        >>> (root / "2.csv").touch()
        >>> (root / "subdir_1" / "3.csv").touch()
        >>> (root / "subdir_2" / "4.csv").touch()
        >>> (root / "subdir_1" / "A" / "5.csv").touch()
        >>> (root / "subdir_1" / "A" / "15.csv.gz").touch()
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "csv")] # doctest: +NORMALIZE_WHITESPACE
        [PosixPath('1.csv'),
         PosixPath('2.csv'),
         PosixPath('subdir_1/3.csv'),
         PosixPath('subdir_1/A/5.csv'),
         PosixPath('subdir_2/4.csv')]
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "parquet")]
        [PosixPath('foo.parquet')]
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "csv.gz")]
        [PosixPath('subdir_1/A/15.csv.gz')]
        >>> [fp.relative_to(root) for fp in list_subdir_files(root, "json")]
        []
        >>> list_subdir_files(root / "nonexistent", "csv")
        []
        >>> tmpdir.cleanup()
    """

    return sorted(list(Path(root).glob(f"**/*.{ext}")))


def get_model_files(cfg: DictConfig, split: str, shard: str) -> list[Path]:
    """Get the tabularized npz files for a given split and shard number.

    TODO: Rename function to get_tabularized_input_files or something.

    Args:
        cfg: `OmegaConf.DictConfig` object with the configuration. It must have the following keys:
            - input_dir: Path to the directory with the tabularized npz files.
            - tabularization: Tabularization configuration, as a nested `DictConfig` object with keys:
              - window_sizes: List of window sizes.
              - aggs: List of aggregation functions.
              - window_inclusion (optional): Dict of {f'param_{window_size}': 0 or 1} to filter window sizes.
              - agg_inclusion (optional): Dict of {f'param_{agg}': 0 or 1} to filter aggregations.

        split: Split name to reference the files stored on disk.
        shard: The shard within the split to reference the files stored on disk.

    Note:
        If `window_inclusion` is present in the config, only window sizes with a value of 1 (or not present)
        will be included. Similarly, if `agg_inclusion` is present, only aggregations with a value of 1
        (or not present) will be included.

    Returns:
        An alphabetically sorted list of Path objects to the tabularized npz files for the given split and
        shard. These files will take the form ``{cfg.input_dir}/{split}/{shard}/{window_size}/{agg}.npz``. For
        static aggregations, the window size will be "none" as these features are not time-varying.

    Examples:
        # >>> cfg = DictConfig({
        # ...     "input_dir": "data",
        # ...     "tabularization": {
        # ...         "window_sizes": ["1d", "7d"],
        # ...         "aggs": ["code/count", "value/sum", "static/present"],
        # ...     }
        # ... })
        # >>> get_model_files(cfg, "train", "0") # doctest: +NORMALIZE_WHITESPACE
        # [PosixPath('data/train/0/1d/code/count.npz'),
        #  PosixPath('data/train/0/1d/value/sum.npz'),
        #  PosixPath('data/train/0/7d/code/count.npz'),
        #  PosixPath('data/train/0/7d/value/sum.npz'),
        #  PosixPath('data/train/0/none/static/present.npz')]
        # >>> get_model_files(cfg, "test/IID", "3/0") # doctest: +NORMALIZE_WHITESPACE
        # [PosixPath('data/test/IID/3/0/1d/code/count.npz'),
        #  PosixPath('data/test/IID/3/0/1d/value/sum.npz'),
        #  PosixPath('data/test/IID/3/0/7d/code/count.npz'),
        #  PosixPath('data/test/IID/3/0/7d/value/sum.npz'),
        #  PosixPath('data/test/IID/3/0/none/static/present.npz')]
        >>> cfg = DictConfig({
        ...     "input_dir": "data",
        ...     "tabularization": {
        ...         "window_sizes": ["1d", "7d"],
        ...         "aggs": ["code/count", "value/sum", "static/present"],
        ...         "window_inclusion": {'param_1d': 0, 'param_7d': 1}
        ...     }
        ... })
        >>> get_model_files(cfg, "train", "0") # doctest: +NORMALIZE_WHITESPACE
         [PosixPath('data/train/0/7d/code/count.npz'),
         PosixPath('data/train/0/7d/value/sum.npz'),
         PosixPath('data/train/0/none/static/present.npz')]
        >>> cfg["tabularization"]["agg_inclusion"] = {
        ...    'param_code_count': 0, 'param_value_sum': 1, 'param_static_present': 1}
        >>> get_model_files(cfg, "test/IID", "3/0") # doctest: +NORMALIZE_WHITESPACE
         [PosixPath('data/test/IID/3/0/7d/value/sum.npz'),
         PosixPath('data/test/IID/3/0/none/static/present.npz')]
    """
    window_sizes = cfg.tabularization.window_sizes
    if "window_inclusion" in cfg.tabularization:
        window_sizes = [
            w for w in window_sizes if cfg.tabularization.window_inclusion.get(f"param_{w}", 1) == 1
        ]
    aggs = cfg.tabularization.aggs
    if "agg_inclusion" in cfg.tabularization:
        aggs = [
            a for a in aggs if cfg.tabularization.agg_inclusion.get(f'param_{a.replace("/", "_")}', 1) == 1
        ]
    shard_dir = Path(cfg.input_dir) / split / shard
    # Given a shard number, returns the model files
    model_files = []
    for window_size in window_sizes:
        for agg in aggs:
            if agg.startswith("static"):
                continue
            else:
                model_files.append(shard_dir / window_size / f"{agg}.npz")
    for agg in aggs:
        if agg.startswith("static"):
            window_size = "none"
            model_files.append(shard_dir / window_size / f"{agg}.npz")
    return sorted(model_files)
