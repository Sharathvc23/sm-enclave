"""Default effect committers: no-op, logging, and factory."""

from __future__ import annotations

import logging
from typing import Any

from .committer import SideEffectCommitter
from .types import EffectType, StagedEffect

logger = logging.getLogger(__name__)


class NoOpEffectCommitter:
    """No-op committer for testing or disabled effect types."""

    async def commit(self, effect: StagedEffect) -> dict[str, Any]:
        return {"status": "noop", "effect_id": effect.effect_id}

    async def rollback(self, effect: StagedEffect) -> bool:
        """Returns True by design -- no-op rollback always succeeds."""
        return True


class LoggingEffectCommitter:
    """Committer that logs effects without applying them."""

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    async def commit(self, effect: StagedEffect) -> dict[str, Any]:
        logger.log(
            self._log_level,
            "COMMIT: %s - %s: %s",
            effect.effect_type.value,
            effect.effect_id,
            effect.payload,
        )
        return {"logged": True, "effect_id": effect.effect_id}

    async def rollback(self, effect: StagedEffect) -> bool:
        logger.log(self._log_level, "ROLLBACK: %s", effect.effect_id)
        return True


def create_default_committer() -> SideEffectCommitter:
    """Create a committer with LoggingEffectCommitter for all types.

    Returns:
        SideEffectCommitter with logging committers for every EffectType
    """
    committer = SideEffectCommitter()
    logging_committer = LoggingEffectCommitter()
    for effect_type in EffectType:
        committer.register_committer(effect_type, logging_committer)
    return committer
