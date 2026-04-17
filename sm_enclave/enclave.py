"""Isolated execution context for speculative branches.

Each Enclave accumulates staged effects without applying them.
When the branch wins, effects are committed; when it loses,
effects are discarded.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime, timezone
from typing import Any

from .types import EffectType, StagedEffect

logger = logging.getLogger(__name__)


class Enclave:
    """Isolated execution context for a single speculative branch.

    Each enclave accumulates staged effects without applying them.
    When the branch wins, effects are committed; when it loses,
    effects are discarded.

    Thread-Safety:
        Uses asyncio.Lock for enclave operations.
        No cross-enclave state is shared.

    Usage:
        enclave = Enclave(
            fork_id="fork:abc123",
            option_key="option_A",
        )

        # Stage effects during speculative execution
        enclave.stage_fact("position", {"x": 1.0, "y": 2.0})
        enclave.stage_message("recipient:1", {"cmd": "execute"})

        # If this branch wins
        await committer.commit(enclave)

        # If this branch loses
        await committer.discard(enclave)
    """

    def __init__(
        self,
        fork_id: str,
        option_key: str,
        *,
        tenant_id: str = "",
        allow_irreversible: bool = False,
        max_staged_effects: int = 1000,
        max_log_entries: int = 10000,
    ) -> None:
        """Initialize enclave.

        Args:
            fork_id: Parent fork ID
            option_key: Option this enclave executes
            tenant_id: Tenant ID for multi-tenant isolation
            allow_irreversible: Allow irreversible effects (dangerous)
            max_staged_effects: Maximum effects to stage
            max_log_entries: Maximum execution log entries retained. Older
                entries are trimmed (FIFO) when the cap is exceeded.
        """
        self._sandbox_id = f"sandbox:{secrets.token_hex(8)}"
        self._fork_id = fork_id
        self._option_key = option_key
        self._tenant_id = tenant_id
        self._allow_irreversible = allow_irreversible
        self._max_effects = max_staged_effects
        self._max_log_entries = max_log_entries

        self._lock = asyncio.Lock()
        self._effects: dict[str, StagedEffect] = {}
        self._effect_order: list[str] = []
        self._sealed = False
        self._committed = False
        self._discarded = False

        self._local_state: dict[str, Any] = {}
        self._execution_log: list[dict[str, Any]] = []

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    @property
    def fork_id(self) -> str:
        return self._fork_id

    @property
    def option_key(self) -> str:
        return self._option_key

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def effect_count(self) -> int:
        return len(self._effects)

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    @property
    def is_committed(self) -> bool:
        return self._committed

    @property
    def is_discarded(self) -> bool:
        return self._discarded

    def stage_fact(
        self,
        fact_type: str,
        payload: dict[str, Any],
        *,
        fact_id: str | None = None,
        priority: int = 100,
    ) -> str:
        """Stage a fact creation/update.

        Args:
            fact_type: Type of fact
            payload: Fact payload
            fact_id: Optional fact ID (generated if not provided)
            priority: Commit priority

        Returns:
            Effect ID
        """
        return self._stage_effect(
            effect_type=EffectType.FACT,
            payload={
                "fact_type": fact_type,
                "fact_id": fact_id or f"fact:{secrets.token_hex(8)}",
                "data": payload,
            },
            reversible=True,
            priority=priority,
        )

    def stage_message(
        self,
        recipient: str,
        payload: dict[str, Any],
        *,
        message_type: str = "generic",
        priority: int = 200,
    ) -> str:
        """Stage a message.

        Args:
            recipient: Recipient identifier
            payload: Message payload
            message_type: Type of message
            priority: Commit priority

        Returns:
            Effect ID
        """
        return self._stage_effect(
            effect_type=EffectType.MESSAGE,
            payload={
                "recipient": recipient,
                "message_type": message_type,
                "data": payload,
            },
            reversible=False,
            priority=priority,
        )

    def stage_command(
        self,
        command_type: str,
        payload: dict[str, Any],
        *,
        irreversible: bool = True,
        target_id: str = "",
        priority: int = 50,
    ) -> str:
        """Stage a command.

        Args:
            command_type: Type of command
            payload: Command parameters
            irreversible: Whether command can be reversed
            target_id: Target system identifier
            priority: Commit priority

        Returns:
            Effect ID

        Raises:
            ValueError: If irreversible commands not allowed
        """
        if irreversible and not self._allow_irreversible:
            raise ValueError(
                f"Irreversible command '{command_type}' blocked. "
                "Set allow_irreversible=True to enable."
            )

        return self._stage_effect(
            effect_type=EffectType.COMMAND,
            payload={
                "command_type": command_type,
                "target_id": target_id,
                "parameters": payload,
            },
            reversible=not irreversible,
            priority=priority,
        )

    def stage_metric(
        self,
        metric_key: str,
        value: float,
        *,
        priority: int = 300,
    ) -> str:
        """Stage a metric update.

        Args:
            metric_key: Metric to update
            value: New value
            priority: Commit priority

        Returns:
            Effect ID
        """
        return self._stage_effect(
            effect_type=EffectType.METRIC,
            payload={
                "metric_key": metric_key,
                "value": value,
            },
            reversible=True,
            priority=priority,
        )

    def stage_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        priority: int = 400,
    ) -> str:
        """Stage an event emission.

        Args:
            event_type: Event type
            payload: Event data
            priority: Commit priority

        Returns:
            Effect ID
        """
        return self._stage_effect(
            effect_type=EffectType.EVENT,
            payload={
                "event_type": event_type,
                "data": payload,
            },
            reversible=False,
            priority=priority,
        )

    def _stage_effect(
        self,
        effect_type: EffectType,
        payload: dict[str, Any],
        *,
        reversible: bool,
        priority: int,
        dependencies: list[str] | None = None,
    ) -> str:
        """Internal method to stage any effect."""
        if self._sealed:
            raise RuntimeError(
                f"Enclave {self._sandbox_id} is sealed, " "cannot stage new effects"
            )

        if len(self._effects) >= self._max_effects:
            raise RuntimeError(
                f"Enclave {self._sandbox_id} at capacity "
                f"({self._max_effects} effects)"
            )

        effect = StagedEffect(
            effect_type=effect_type,
            payload=payload,
            reversible=reversible,
            priority=priority,
            dependencies=dependencies or [],
            metadata={
                "sandbox_id": self._sandbox_id,
                "fork_id": self._fork_id,
                "option_key": self._option_key,
                "tenant_id": self._tenant_id,
            },
        )

        self._effects[effect.effect_id] = effect
        self._effect_order.append(effect.effect_id)

        self._execution_log.append(
            {
                "action": "staged",
                "effect_id": effect.effect_id,
                "effect_type": effect_type.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self._execution_log) > self._max_log_entries:
            # Trim oldest entries (FIFO) — long-running branches must not
            # grow the log without bound.
            overflow = len(self._execution_log) - self._max_log_entries
            del self._execution_log[:overflow]

        return effect.effect_id

    def set_local_state(self, key: str, value: Any) -> None:
        """Set enclave-local state (not persisted on commit)."""
        self._local_state[key] = value

    def get_local_state(self, key: str, default: Any = None) -> Any:
        """Get enclave-local state."""
        return self._local_state.get(key, default)

    def seal(self) -> None:
        """Seal the enclave, preventing new effects from being staged."""
        self._sealed = True

    def mark_committed(self) -> None:
        """Mark the enclave as successfully committed.

        Called by the committer after all staged effects have been applied.
        Sets both the committed flag and clears the discarded flag so the
        state is unambiguous.
        """
        self._committed = True
        self._discarded = False

    def mark_failed(self) -> None:
        """Mark the enclave as failed (commit attempted but not successful).

        Called by the committer after a rollback. Leaves ``is_committed`` and
        ``is_discarded`` both false so callers can distinguish a failed commit
        from a successful one or a deliberate discard.
        """
        self._committed = False
        self._discarded = False

    def mark_discarded(self) -> None:
        """Mark the enclave as discarded (losing branch, never committed).

        Called by the committer on non-winning enclaves.
        """
        self._discarded = True
        self._committed = False

    def get_effects(self) -> list[StagedEffect]:
        """Get all staged effects in commit order (sorted by priority)."""
        effects = [self._effects[eid] for eid in self._effect_order]
        return sorted(
            effects,
            key=lambda e: (
                e.priority,
                self._effect_order.index(e.effect_id),
            ),
        )

    def get_effect(self, effect_id: str) -> StagedEffect | None:
        """Get a specific effect by ID."""
        return self._effects.get(effect_id)

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Get the enclave execution log."""
        return list(self._execution_log)

    def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of enclave state."""
        return {
            "sandbox_id": self._sandbox_id,
            "fork_id": self._fork_id,
            "option_key": self._option_key,
            "effect_count": len(self._effects),
            "sealed": self._sealed,
            "committed": self._committed,
            "discarded": self._discarded,
            "effects": [e.as_dict() for e in self.get_effects()],
        }
