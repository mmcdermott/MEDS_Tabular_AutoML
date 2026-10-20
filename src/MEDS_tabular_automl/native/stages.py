"""The native backend of meds-tab-tabularize-static, -time-series and -cache-task.

Each of those scripts first calls ``use_native(cfg)``; when it returns True, the script hands its Hydra
configuration to the matching function here and returns. These functions write the same files as the
Python backend, but schedule the work differently:

* A shard's static and time-series blocks are computed together, from one read of the shard. Whichever
  tabularization script runs first writes every configured block; the other finds them complete.
* One process does the work, on ``native.threads`` threads. In a Hydra multirun over ``worker``, workers
  other than 0 return immediately.
* Outputs are reused when their recorded identity (input files, configuration, feature metadata, engine
  build) matches; ``do_overwrite=True`` recomputes them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from . import engine
from .contract import SUPPORTED_AGGS, TabularConfig
from .pipeline import Pipeline

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

BACKENDS = ("python", "native")


def use_native(cfg: DictConfig) -> bool:
    """Whether cfg selects the native backend (``backend=native``; the default is ``python``).

    Raises RuntimeError when the native backend is selected but cannot run: the engine is not built, or
    the configuration uses something it does not implement.
    """
    choice = cfg.get("backend", "python")
    if choice not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {choice!r}.")
    if choice == "python":
        return False
    try:
        unsupported = sorted(set(cfg.tabularization.aggs) - set(SUPPORTED_AGGS))
        if unsupported:
            raise ValueError(f"The native backend does not implement the aggregations {unsupported}.")
        tabular_config(cfg)
        engine.library()
    except (ValueError, OSError, RuntimeError) as error:
        raise RuntimeError(f"backend=native cannot run: {error} Use backend=python.") from error
    return True


def tabular_config(cfg: DictConfig) -> TabularConfig:
    tab = cfg.tabularization
    allowed = tab.get("allowed_codes")
    return TabularConfig(
        window_sizes=tuple(tab.window_sizes),
        aggs=tuple(tab.aggs),
        allowed_codes=None if allowed is None else tuple(allowed),
        min_code_inclusion_count=tab.get("min_code_inclusion_count"),
        min_code_inclusion_frequency=tab.get("min_code_inclusion_frequency"),
        max_included_codes=tab.get("max_included_codes"),
        do_compress=bool(tab.get("do_compress", False)),
        label_column=cfg.get("label_column", "boolean_value"),
    )


def _threads(cfg: DictConfig) -> int:
    configured = cfg.native.threads
    if configured is not None:
        return int(configured)
    if hasattr(os, "sched_getaffinity"):  # Respects the CPUs a Linux job was given (e.g. by Slurm).
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _pipeline(cfg: DictConfig, config: TabularConfig) -> Pipeline:
    return Pipeline(
        cfg.input_dir,
        cfg.output_dir,
        config,
        code_metadata=cfg.tabularization.filtered_code_metadata_fp,
        threads=_threads(cfg),
        reuse=not cfg.do_overwrite,
    )


def _require_default(cfg: DictConfig, key: str, expected: Path):
    if Path(cfg[key]).resolve() != expected.resolve():
        raise ValueError(
            f"The native backend writes {key} to its default location {expected}; got {cfg[key]}. "
            "Use the default or backend=python."
        )


def _is_primary_worker(cfg: DictConfig, stage: str) -> bool:
    if cfg.get("worker", 0) != 0:
        logger.info(f"Native {stage}: worker 0 does all the work; worker {cfg.worker} has none.")
        return False
    return True


def tabularize(cfg: DictConfig, stage: str):
    """Every configured static and time-series block of every shard, at event times or label times."""
    if not _is_primary_worker(cfg, stage):
        return
    _require_default(cfg, "output_tabularized_dir", Path(cfg.output_dir) / "tabularize")
    run = _pipeline(cfg, tabular_config(cfg))
    report = run.tabularize_at_labels(cfg.input_label_dir) if cfg.input_label_dir else run.tabularize()
    _log(f"Native {stage}", report, run.threads)


def cache_task(cfg: DictConfig):
    """The task's rows of every event-time block, and its label cache."""
    if not _is_primary_worker(cfg, "cache_task"):
        return
    output = Path(cfg.output_dir)
    _require_default(cfg, "input_tabularized_dir", output / "tabularize")
    _require_default(cfg, "output_tabularized_cache_dir", output / cfg.task_name / "task_cache")
    _require_default(cfg, "output_label_cache_dir", output / cfg.task_name / "labels")
    run = _pipeline(cfg, tabular_config(cfg))
    report = run.cache_task(cfg.task_name, cfg.input_label_dir)
    computed = [name for name, shard in run.report["tabularize"].items() if not shard["reused"]]
    if computed:
        logger.warning(
            f"Native cache_task tabularized {len(computed)} shards first: their blocks were missing or "
            "were not written by the native backend with this configuration."
        )
    unknown = report.pop("unknown_subject_labels")
    if unknown:
        logger.warning(f"{unknown} labels belong to subjects that are in no shard; they are skipped.")
    _log("Native cache_task", report, run.threads)


def _log(stage: str, report: dict, threads: int):
    computed = [shard for shard in report.values() if not shard["reused"]]
    seconds = sum(shard["seconds"] for shard in computed)
    entries = sum(shard["entries"] for shard in computed)
    logger.info(
        f"{stage}: {len(computed)} shards computed ({entries:,} stored entries, {seconds:.2f} s on {threads} "
        f"threads), {len(report) - len(computed)} reused."
    )
