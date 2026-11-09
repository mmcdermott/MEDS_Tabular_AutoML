"""The native (C++) backend of MEDS-Tab's tabularization scripts.

Select it with ``backend=native``; the default, ``backend=python``, never loads it. It writes the same
files as the Python backend, which ``TabularDataset`` and the model launchers read unchanged. See
``docs/native_backend.md``.
"""

from .stages import tabularize, use_native

__all__ = ["tabularize", "use_native"]
