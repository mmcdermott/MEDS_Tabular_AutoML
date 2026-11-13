"""The native (C++) backend of MEDS-Tab's tabularization scripts.

The default, ``backend=auto``, uses it whenever it is built and supports the configuration;
``backend=native`` requires it and ``backend=python`` never loads it. It writes the same files as the Python
backend, which ``TabularDataset`` and the model launchers read unchanged. See ``docs/native_backend.md``.
"""

from .stages import tabularize, use_native

__all__ = ["tabularize", "use_native"]
