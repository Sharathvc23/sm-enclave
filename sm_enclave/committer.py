"""Side effect committer with atomic commit and rollback semantics."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from .enclave import Enclave
from .types import CommitResult, EffectCommitter, EffectStatus, EffectType, StagedEffect

logger = logging.getLogger(__name__)


class SideEffectCommitter:
    """Commits or discards enclave effects.

    Orchestrates the commit/discard process for all effects in an
    enclave, maintaining atomicity and ordering.

    Usage:
        committer = SideEffectCommitter(
            effect_committers={
                EffectType.FACT: fact_committer,
                EffectType.MESSAGE: message_committer,
            }
        )

        # Commit winning enclave
        result = await committer.commit(winning_enclave)

        # Discard losing enclaves
        for enclave in losing_enclaves:
            await committer.discard(enclave)
    """

    def __init__(
        self,
        effect_committers: dict[EffectType, EffectCommitter] | None = None,
        *,
        on_commit: Callable[[StagedEffect, dict[str, Any]], None] | None = None,
        on_discard: Callable[[StagedEffect], None] | None = None,
        commit_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize committer.

        Args:
            effect_committers: Type-specific committers
            on_commit: Callback after each effect commit
            on_discard: Callback after each effect discard
            commit_timeout_seconds: Timeout for entire commit
        """
        self._committers = effect_committers or {}
        self._on_commit = on_commit
        self._on_discard = on_discard
        self._timeout = commit_timeout_seconds

    def register_committer(
        self, effect_type: EffectType, committer: EffectCommitter
    ) -> None:
        """Register a committer for an effect type."""
        self._committers[effect_type] = committer

    async def commit(self, enclave: Enclave) -> CommitResult:
        """Commit all effects from an enclave.

        Commits effects in priority order, respecting dependencies.
        If any effect fails, rolls back previously committed effects.

        Args:
            enclave: Enclave to commit

        Returns:
            CommitResult with details
        """
        start_time = time.perf_counter()

        if enclave.is_committed:
            return CommitResult(
                sandbox_id=enclave.sandbox_id,
                option_key=enclave.option_key,
                committed_effects=[],
                failed_effects=[],
                total_time_ms=0.0,
                success=True,
            )

        if enclave.is_discarded:
            return CommitResult(
                sandbox_id=enclave.sandbox_id,
                option_key=enclave.option_key,
                committed_effects=[],
                failed_effects=[("already_discarded", "Enclave was already discarded")],
                total_time_ms=0.0,
                success=False,
            )

        # Seal enclave to prevent new effects during commit
        enclave.seal()

        effects = enclave.get_effects()
        committed, failed, _committed_effects = await self._apply_effects(
            effects, enclave
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return CommitResult(
            sandbox_id=enclave.sandbox_id,
            option_key=enclave.option_key,
            committed_effects=committed,
            failed_effects=failed,
            total_time_ms=elapsed_ms,
            success=len(failed) == 0,
        )

    async def _apply_effects(
        self,
        effects: list[StagedEffect],
        enclave: Enclave,
    ) -> tuple[list[str], list[tuple[str, str]], list[StagedEffect]]:
        """Apply all effects in order, rolling back on failure."""
        committed: list[str] = []
        failed: list[tuple[str, str]] = []
        committed_effects: list[StagedEffect] = []

        try:
            for effect in effects:
                # Check dependencies
                deps_met = True
                for dep_id in effect.dependencies:
                    if dep_id not in committed:
                        failed.append(
                            (
                                effect.effect_id,
                                f"Dependency {dep_id} not committed",
                            )
                        )
                        effect.status = EffectStatus.FAILED
                        deps_met = False
                        break
                if not deps_met:
                    continue

                # Get committer for effect type
                committer = self._committers.get(effect.effect_type)

                if committer is None:
                    # No committer - log warning but don't fail
                    logger.warning(
                        "No committer for effect type %s, skipping %s",
                        effect.effect_type,
                        effect.effect_id,
                    )
                    effect.status = EffectStatus.COMMITTED
                    effect.commit_result = {
                        "skipped": True,
                        "reason": "no_committer",
                    }
                    committed.append(effect.effect_id)
                    continue

                try:
                    timeout = self._timeout / max(len(effects), 1)
                    result = await asyncio.wait_for(
                        committer.commit(effect),
                        timeout=timeout,
                    )
                    effect.status = EffectStatus.COMMITTED
                    effect.commit_result = result
                    committed.append(effect.effect_id)
                    committed_effects.append(effect)

                    if self._on_commit:
                        try:
                            self._on_commit(effect, result)
                        except Exception:
                            logger.warning(
                                "on_commit callback failed for %s",
                                effect.effect_id,
                                exc_info=True,
                            )

                except TimeoutError:
                    failed.append((effect.effect_id, "Commit timeout"))
                    effect.status = EffectStatus.FAILED
                    break

                except Exception as e:
                    failed.append((effect.effect_id, str(e)))
                    effect.status = EffectStatus.FAILED
                    logger.warning(
                        "Effect %s commit failed: %s",
                        effect.effect_id,
                        e,
                        exc_info=True,
                    )
                    break

            # If any failures, attempt rollback
            if failed:
                await self._rollback_committed(committed_effects)
                enclave.mark_failed()
            else:
                enclave.mark_committed()

        except Exception as e:
            failed.append(("commit_error", str(e)))
            logger.error(
                "Enclave %s commit error: %s",
                enclave.sandbox_id,
                e,
                exc_info=True,
            )

        return committed, failed, committed_effects

    async def _rollback_committed(self, effects: list[StagedEffect]) -> None:
        """Attempt to rollback committed effects on failure."""
        for effect in reversed(effects):
            if not effect.reversible:
                logger.warning(
                    "Cannot rollback irreversible effect %s",
                    effect.effect_id,
                )
                continue

            committer = self._committers.get(effect.effect_type)
            if committer is None:
                continue

            try:
                await committer.rollback(effect)
                effect.status = EffectStatus.DISCARDED
            except Exception as e:
                logger.warning("Rollback failed for %s: %s", effect.effect_id, e)

    async def discard(self, enclave: Enclave) -> None:
        """Discard all effects from an enclave.

        Simply marks all effects as discarded without applying them.

        Args:
            enclave: Enclave to discard
        """
        if enclave.is_committed:
            logger.warning(
                "Cannot discard already committed enclave %s",
                enclave.sandbox_id,
            )
            return

        enclave.seal()

        for effect in enclave.get_effects():
            effect.status = EffectStatus.DISCARDED

            if self._on_discard:
                try:
                    self._on_discard(effect)
                except Exception:
                    logger.warning(
                        "on_discard callback failed for %s",
                        effect.effect_id,
                        exc_info=True,
                    )

        enclave.mark_discarded()

        enclave._discarded = True
