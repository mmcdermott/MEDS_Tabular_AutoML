"""The configurations the native backend supports, and the identity of its outputs.

The native backend writes the same rows, columns, values and labels as the Python backend
(``tests/test_native_backend.py`` checks every block). Any change that can alter a stored row, column,
value or label must bump SEMANTICS_VERSION: output identities include it, so older outputs are recomputed
rather than reused.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

SEMANTICS_VERSION = 1

STATIC_AGGS = ("static/present", "static/first")
CODE_AGGS = ("code/count",)
VALUE_AGGS = ("value/count", "value/sum", "value/sum_sqd", "value/min", "value/max")
DYNAMIC_AGGS = CODE_AGGS + VALUE_AGGS
DEFAULT_WINDOWS = ("1d", "7d", "30d", "365d", "full")
SUPPORTED_AGGS = STATIC_AGGS + DYNAMIC_AGGS
# Feature-name suffix in metadata/codes.parquet of each aggregation.
SUFFIXES = {
    "static/present": "/static/present",
    "static/first": "/static/first",
    "code/count": "/code",
    **dict.fromkeys(VALUE_AGGS, "/value"),
}

_UNITS = {
    "us": 1,
    "ms": 1_000,
    "s": 1_000_000,
    "m": 60_000_000,
    "h": 3_600_000_000,
    "d": 86_400_000_000,
    "w": 604_800_000_000,
}


def window_microseconds(name: str) -> int:
    """Return a window width in integer microseconds; ``full`` is -1 (all history).

    Accepts the pandas-Timedelta-compatible subset ``<positive int><us|ms|s|m|h|d|w>``.
    """
    if name == "full":
        return -1
    match = re.fullmatch(r"([1-9][0-9]*)(us|ms|s|m|h|d|w)", name)
    if not match:
        raise ValueError(f"Unsupported window size {name!r}; use e.g. 7d, 12h or full.")
    width = int(match[1]) * _UNITS[match[2]]
    if width >= 2**63:
        raise ValueError(f"Window {name!r} exceeds signed 64-bit microseconds.")
    return width


@dataclass(frozen=True)
class TabularConfig:
    """The tabularization settings that determine the native backend's outputs; names and defaults as in the
    tabularization config."""

    window_sizes: tuple[str, ...] = DEFAULT_WINDOWS
    aggs: tuple[str, ...] = SUPPORTED_AGGS
    allowed_codes: tuple[str, ...] | None = None
    min_code_inclusion_count: int | None = 10
    min_code_inclusion_frequency: float | None = None
    max_included_codes: int | None = None
    do_compress: bool = False
    label_column: str = "boolean_value"

    def __post_init__(self):
        object.__setattr__(self, "window_sizes", tuple(self.window_sizes))
        object.__setattr__(self, "aggs", tuple(self.aggs))
        if self.allowed_codes is not None:
            object.__setattr__(self, "allowed_codes", tuple(self.allowed_codes))
        self.validate()

    def validate(self):
        if not self.aggs or len(set(self.aggs)) != len(self.aggs):
            raise ValueError("aggs must be distinct and nonempty.")
        unknown = set(self.aggs) - set(SUPPORTED_AGGS)
        if unknown:
            raise ValueError(f"The native backend does not implement the aggregations {sorted(unknown)}.")
        if any(agg in DYNAMIC_AGGS for agg in self.aggs):
            if not self.window_sizes or len(set(self.window_sizes)) != len(self.window_sizes):
                raise ValueError("window_sizes must be distinct and nonempty.")
            for window in self.window_sizes:
                window_microseconds(window)
        frequency = self.min_code_inclusion_frequency
        if frequency is not None and not 0 <= frequency <= 1:
            raise ValueError("min_code_inclusion_frequency must be between 0 and 1.")
        for name in ("min_code_inclusion_count", "max_included_codes"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None.")

    @property
    def dynamic_aggs(self) -> tuple[str, ...]:
        return tuple(agg for agg in DYNAMIC_AGGS if agg in self.aggs)

    @property
    def static_aggs(self) -> tuple[str, ...]:
        return tuple(agg for agg in STATIC_AGGS if agg in self.aggs)

    def block_identity(self) -> dict:
        """Everything that can change tabularized blocks (the label column cannot)."""
        identity = asdict(self)
        del identity["label_column"]
        identity["semantics"] = SEMANTICS_VERSION
        return identity

    @classmethod
    def from_block_identity(cls, identity: dict, label_column: str) -> TabularConfig:
        values = {k: v for k, v in identity.items() if k != "semantics"}
        if identity.get("semantics") != SEMANTICS_VERSION or set(values) != set(cls.__dataclass_fields__) - {
            "label_column"
        }:
            raise ValueError("The recorded configuration is from another semantics version.")
        return cls(**values, label_column=label_column)
