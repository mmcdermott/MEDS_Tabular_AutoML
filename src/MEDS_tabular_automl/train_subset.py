"""Train on a reproducible subset of the training split (``train_subset``), e.g. for scaling studies.

``train_subset.by`` chooses what is counted:

- ``subjects``: keep ``n`` (or a ``fraction``) of the training subjects that have labels, with all of their
    labels;
- ``labels``: keep ``n`` (or a ``fraction``) of the training label rows.

Subjects are taken in the order of ``sha256("<salt>:<subject_id>")`` and label rows in the order of
``sha256("<salt>:<subject_id>:<prediction_time in microseconds>")``. A subset is therefore the same on any
machine and library version, and with one salt a smaller subset is contained in a larger one (nested subsets
for learning curves); other salts give independent repeats. Tuning and held-out labels are kept whole, so
every subset is evaluated on the same subjects.

Every stage applies the subset the same way, given the same ``train_subset.*`` overrides: :func:`apply`
writes the cropped labels once to ``<output_dir>/train_subset/<id>/labels`` (``<id>`` names the options and
fingerprints the source labels) and points ``input_label_dir`` at them. ``meds-tab-describe`` then fits the
code frequencies on the selected subjects only, tabularization computes features for them only, and the
model trains on them only. Next to the labels, ``subjects.parquet`` lists the selected training subjects and
``subset.json`` records the options and the counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterable

MODES = ("subjects", "labels")
TRAIN_SPLIT = "train"
SUBSET_DIRNAME = "train_subset"


def options(cfg: DictConfig) -> dict | None:
    """The validated ``train_subset`` options, or None when no subset is requested.

    Examples:
        >>> options(OmegaConf.create({"train_subset": {"by": None}})) is None
        True
        >>> options(OmegaConf.create({"train_subset": {"by": "subjects", "n": 5, "salt": 0}}))
        {'by': 'subjects', 'n': 5, 'fraction': None, 'salt': '0'}
        >>> options(OmegaConf.create({"train_subset": {"by": "labels", "fraction": 0.25}}))
        {'by': 'labels', 'n': None, 'fraction': 0.25, 'salt': '0'}
        >>> options(OmegaConf.create({"train_subset": {"by": "patients", "n": 5}}))
        Traceback (most recent call last):
            ...
        ValueError: train_subset.by must be null, 'subjects' or 'labels', got 'patients'
        >>> options(OmegaConf.create({"train_subset": {"by": "subjects", "n": 5, "fraction": 0.5}}))
        Traceback (most recent call last):
            ...
        ValueError: train_subset needs exactly one of n and fraction
        >>> options(OmegaConf.create({"train_subset": {"by": "subjects", "fraction": 1.5}}))
        Traceback (most recent call last):
            ...
        ValueError: train_subset.fraction must be in (0, 1], got 1.5
    """
    subset = cfg.get("train_subset")
    if subset is None or subset.get("by") is None:
        return None
    by = subset.get("by")
    if by not in MODES:
        raise ValueError(f"train_subset.by must be null, 'subjects' or 'labels', got {by!r}")
    n, fraction = subset.get("n"), subset.get("fraction")
    if (n is None) == (fraction is None):
        raise ValueError("train_subset needs exactly one of n and fraction")
    if n is not None and (isinstance(n, bool) or int(n) != n or n < 1):
        raise ValueError(f"train_subset.n must be a positive integer, got {n!r}")
    if fraction is not None and not 0 < float(fraction) <= 1:
        raise ValueError(f"train_subset.fraction must be in (0, 1], got {fraction!r}")
    return {
        "by": by,
        "n": None if n is None else int(n),
        "fraction": None if fraction is None else float(fraction),
        "salt": str(subset.get("salt", "0")),
    }


def _digest(salt: str, *parts) -> str:
    return hashlib.sha256(":".join([salt, *(str(p) for p in parts)]).encode()).hexdigest()


def rank_subjects(subject_ids: Iterable[int], salt: str) -> list[int]:
    """Distinct subjects in the order they are selected for ``salt``.

    Examples:
        >>> rank_subjects([3, 1, 2, 1, 4, 5], "0")
        [4, 3, 2, 5, 1]
        >>> rank_subjects([3, 1, 2, 4, 5], "1")
        [4, 5, 2, 3, 1]

    The order does not depend on the input order or on which other subjects are present, so subsets of one
    salt are nested:

        >>> rank_subjects([5, 4, 3, 2, 1], "0")[:3] == rank_subjects([1, 2, 3, 4, 5], "0")[:3]
        True
        >>> [s for s in rank_subjects([1, 2, 3, 4, 5], "0") if s != 3] == rank_subjects([1, 2, 4, 5], "0")
        True
    """
    return sorted({int(s) for s in subject_ids}, key=lambda s: (_digest(salt, s), s))


def subset_size(total: int, n: int | None, fraction: float | None, what: str) -> int:
    """How many of ``total`` items to keep.

    Examples:
        >>> subset_size(10, 3, None, "subjects"), subset_size(10, None, 0.25, "subjects")
        (3, 2)
        >>> subset_size(10, None, 0.01, "subjects")
        1
        >>> subset_size(10, 11, None, "subjects")
        Traceback (most recent call last):
            ...
        ValueError: train_subset.n is 11, but the training labels have only 10 subjects
    """
    if n is not None:
        if n > total:
            raise ValueError(f"train_subset.n is {n}, but the training labels have only {total} {what}")
        return n
    return max(1, round(fraction * total))


def select(train: dict[str, pl.DataFrame], opts: dict) -> tuple[dict[str, pl.DataFrame], list[int]]:
    """Crops the training label shards; returns them (rows in their original order) and the kept subjects.

    Examples:
        >>> from datetime import datetime
        >>> def labels(subjects):
        ...     return pl.DataFrame({
        ...         "subject_id": subjects,
        ...         "prediction_time": [datetime(2020, 1, d) for d in range(1, len(subjects) + 1)],
        ...         "boolean_value": [True] * len(subjects),
        ...     })
        >>> train = {"0": labels([1, 1, 2]), "1": labels([3, 4, 4, 5])}
        >>> opts = {"by": "subjects", "n": 2, "fraction": None, "salt": "0"}
        >>> kept, subjects = select(train, opts)
        >>> subjects, kept["0"]["subject_id"].to_list(), kept["1"]["subject_id"].to_list()
        ([4, 3], [], [3, 4, 4])
        >>> kept, subjects = select(train, {**opts, "by": "labels", "n": 3})
        >>> sum(len(f) for f in kept.values()), kept["0"].columns == train["0"].columns
        (3, True)
    """
    salt = opts["salt"]
    if opts["by"] == "subjects":
        ranked = rank_subjects((s for f in train.values() for s in f["subject_id"].to_list()), salt)
        subjects = ranked[: subset_size(len(ranked), opts["n"], opts["fraction"], "subjects")]
        keep = pl.Series(subjects, dtype=pl.Int64).implode()
        return {
            name: frame.filter(pl.col("subject_id").cast(pl.Int64).is_in(keep))
            for name, frame in train.items()
        }, subjects

    keys = []
    for name in sorted(train):
        frame = train[name]
        times = frame.select(pl.col("prediction_time").dt.epoch("us"))["prediction_time"].to_list()
        for row, (subject, time) in enumerate(zip(frame["subject_id"].to_list(), times, strict=True)):
            keys.append((_digest(salt, subject, time), name, row))
    keys.sort()
    rows: dict[str, set[int]] = {name: set() for name in train}
    for _, name, row in keys[: subset_size(len(keys), opts["n"], opts["fraction"], "label rows")]:
        rows[name].add(row)
    kept = {
        name: frame.with_row_index("__row").filter(pl.col("__row").is_in(sorted(rows[name]))).drop("__row")
        for name, frame in train.items()
    }
    subjects = rank_subjects((s for f in kept.values() for s in f["subject_id"].to_list()), salt)
    return kept, subjects


def _subset_dir(cfg: DictConfig, opts: dict, source: Path, files: list[Path]) -> Path:
    identity = {
        **opts,
        "source": source.as_posix(),
        "files": [[f.relative_to(source).as_posix(), f.stat().st_size, f.stat().st_mtime_ns] for f in files],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    size = opts["n"] if opts["n"] is not None else opts["fraction"]
    salt = re.sub(r"[^A-Za-z0-9._-]", "_", opts["salt"])
    return Path(cfg.output_dir) / SUBSET_DIRNAME / f"{opts['by']}-{size}-salt{salt}-{key}"


def materialize(cfg: DictConfig, opts: dict) -> Path:
    """Writes the cropped labels for ``cfg.input_label_dir`` (once) and returns their directory.

    Concurrent stages or workers may call this at the same time: each writes to a temporary directory and
    renames it into place; whoever loses the race uses the winner's copy (same content).
    """
    source = Path(cfg.input_label_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"input_label_dir: {source} is not a directory.")
    files = sorted(source.rglob("*.parquet"))
    train_files = [f for f in files if f.relative_to(source).parts[0] == TRAIN_SPLIT]
    if not train_files:
        raise ValueError(f"train_subset: no training labels under {source / TRAIN_SPLIT}")
    out = _subset_dir(cfg, opts, source, files)
    if (out / "subset.json").is_file():
        return out / "labels"

    train = {f.relative_to(source).as_posix(): pl.read_parquet(f) for f in train_files}
    kept, subjects = select(train, opts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        for f in files:
            rel = f.relative_to(source).as_posix()
            target = tmp / "labels" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if rel in kept:
                kept[rel].write_parquet(target)
            else:
                shutil.copyfile(f, target)
        pl.DataFrame({"subject_id": subjects}, schema={"subject_id": pl.Int64}).write_parquet(
            tmp / "subjects.parquet"
        )
        record = {
            **opts,
            "source_label_dir": source.as_posix(),
            "train_subjects": {"total": len({s for f in train.values() for s in f["subject_id"].to_list()})},
            "train_label_rows": {"total": sum(len(f) for f in train.values())},
        }
        record["train_subjects"]["kept"] = len(subjects)
        record["train_label_rows"]["kept"] = sum(len(f) for f in kept.values())
        (tmp / "subset.json").write_text(json.dumps(record, indent=2))
        try:
            os.replace(tmp, out)
        except OSError:
            if not (out / "subset.json").is_file():
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return out / "labels"


def apply(cfg: DictConfig) -> dict | None:
    """Points ``cfg.input_label_dir`` at the cropped labels when ``train_subset`` is set; returns the options.

    Idempotent: a label directory that already is a subset of this ``output_dir`` is left as it is.
    """
    opts = options(cfg)
    if opts is None:
        return None
    if (
        "input_label_dir" not in cfg
        or OmegaConf.is_missing(cfg, "input_label_dir")
        or not cfg.input_label_dir
    ):
        raise ValueError(
            "train_subset needs input_label_dir: the task's labels define the training subjects."
        )
    subsets = (Path(cfg.output_dir) / SUBSET_DIRNAME).resolve()
    if subsets in Path(cfg.input_label_dir).resolve().parents:
        return opts
    label_dir = materialize(cfg, opts)
    record = json.loads((label_dir.parent / "subset.json").read_text())
    subjects, rows = record["train_subjects"], record["train_label_rows"]
    logger.info(
        f"train_subset: {subjects['kept']:,} of {subjects['total']:,} training subjects, {rows['kept']:,} of "
        f"{rows['total']:,} training label rows (by={opts['by']}, salt={opts['salt']}); labels: {label_dir}"
    )
    cfg.input_label_dir = str(label_dir)
    return opts


def selected_subjects(cfg: DictConfig) -> pl.LazyFrame | None:
    """The selected training subjects (``subject_id``) after :func:`apply`, or None without a subset."""
    if options(cfg) is None:
        return None
    return pl.scan_parquet(Path(cfg.input_label_dir).parent / "subjects.parquet")
