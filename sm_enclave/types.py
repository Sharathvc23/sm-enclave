"""Core types for sm-enclave: enums, dataclasses, and protocols."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


class EffectType(str, Enum):
    """Types of side effects that can be staged."""

    FACT = "fact"
    MESSAGE = "message"
    COMMAND = "command"
    METRIC = "metric"
    EVENT = "event"
    EXTERNAL_API = "external_api"


class EffectStatus(str, Enum):
    """Status of a staged effect."""

    STAGED = "staged"
    COMMITTED = "committed"
    DISCARDED = "discarded"
    FAILED = "failed"


@dataclass(slots=True)
class StagedEffect:
    """A side effect staged for potential commit.

    Staged effects are held in the enclave until the branch is
    selected as the winner (committed) or loser (discarded).

    Attributes:
        effect_id: Unique identifier
        effect_type: Type of effect
        payload: Effect-specific data
        reversible: Whether this effect can be undone
        priority: Commit order priority (lower = first)
        dependencies: Effect IDs that must commit before this
        metadata: Additional context
        staged_at: When the effect was staged
        status: Current status
        commit_result: Result after commit attempt
        signature: Optional cryptographic signature
    """

    effect_id: str = field(default_factory=lambda: f"eff:{secrets.token_hex(8)}")
    effect_type: EffectType = EffectType.FACT
    payload: dict[str, Any] = field(default_factory=dict)
    reversible: bool = True
    priority: int = 100
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    staged_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: EffectStatus = EffectStatus.STAGED
    commit_result: dict[str, Any] | None = None
    signature: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "effect_id": self.effect_id,
            "effect_type": self.effect_type.value,
            "payload": dict(self.payload),
            "reversible": self.reversible,
            "priority": self.priority,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
            "staged_at": self.staged_at.isoformat(),
            "status": self.status.value,
            "commit_result": self.commit_result,
            "signature": self.signature,
        }


@dataclass(slots=True)
class CommitResult:
    """Result of committing an enclave's effects."""

    sandbox_id: str
    option_key: str
    committed_effects: list[str]  # effect_ids
    failed_effects: list[tuple[str, str]]  # (effect_id, error)
    total_time_ms: float
    success: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str | None = None

    @property
    def commit_count(self) -> int:
        return len(self.committed_effects)

    @property
    def failure_count(self) -> int:
        return len(self.failed_effects)


class EffectCommitter(Protocol):
    """Protocol for effect-specific committers."""

    async def commit(self, effect: StagedEffect) -> dict[str, Any]:
        """Commit an effect and return result."""
        ...

    async def rollback(self, effect: StagedEffect) -> bool:
        """Attempt to roll back a committed effect."""
        ...
