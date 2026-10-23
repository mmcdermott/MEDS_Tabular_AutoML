"""The native (C++) backend of MEDS-Tab's tabularization and task-caching scripts.

Select it with ``backend=native``; the default, ``backend=python``, never loads it. It writes the same
files as the Python backend, which ``TabularDataset`` and the model launchers read unchanged. See
``docs/native_backend.md``.
"""

from .stages import cache_task, tabularize, use_native

__all__ = ["cache_task", "tabularize", "use_native"]
