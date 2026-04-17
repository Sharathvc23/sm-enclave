"""sm-enclave: Speculative execution sandbox with commit/discard semantics."""

from .committer import SideEffectCommitter
from .committers import (
    LoggingEffectCommitter,
    NoOpEffectCommitter,
    create_default_committer,
)
from .enclave import Enclave
from .executor import Fork, SpeculativeExecutor
from .types import (
    CommitResult,
    EffectCommitter,
    EffectStatus,
    EffectType,
    StagedEffect,
)

__version__ = "0.2.0"

__all__ = [
    "CommitResult",
    "EffectCommitter",
    "EffectStatus",
    "EffectType",
    "Enclave",
    "Fork",
    "LoggingEffectCommitter",
    "NoOpEffectCommitter",
    "SideEffectCommitter",
    "SpeculativeExecutor",
    "StagedEffect",
    "create_default_committer",
]
