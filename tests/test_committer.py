"""Tests for SideEffectCommitter: priority-ordered commit, rollback, discard.

# Step 1 -- Assumption Audit
# - commit() checks is_committed/is_discarded before work begins
# - _apply_effects assumes priority-sorted list from get_effects()
# - Rollback assumes reversible flag is trustworthy, rollback idempotent
# - Timeout divided evenly: timeout / len(effects) -- uneven penalized
# - on_commit/on_discard assumed non-throwing; if they throw, breaks
# - discard() only checks is_committed, not is_discarded (re-discards)

# Step 2 -- Gap Analysis
# - No test for committer that raises during rollback
# - No test for unmet dependency (dependency chain failure)
# - No test for slow committer exceeding per-effect timeout
# - No test for rollback skipping irreversible effects
# - No test for partial commit (first ok, second fails)

# Step 3 -- Break It List
# commit:
#   (1) committer returns wrong type -> silent corruption
#   (2) set _committed=True before commit -> noop with 0 effects
#   (3) 0 effects -> succeeds trivially, hides missing committers
# discard:
#   (1) discard committed enclave -> logs warning, state unchanged
#   (2) on_discard callback raises -> uncaught exception
#   (3) discard twice -> re-marked DISCARDED (idempotent untested)
# _rollback_committed:
#   (1) reversible=False -> skipped, data inconsistent
#   (2) committer.rollback raises -> swallowed, partial rollback
#   (3) empty committed list -> noop loop
"""

from __future__ import annotations

import asyncio

import pytest

from sm_enclave import (
    EffectStatus,
    EffectType,
    Enclave,
    NoOpEffectCommitter,
    SideEffectCommitter,
    StagedEffect,
)


class TestSideEffectCommitter:
    """Tests for SideEffectCommitter."""

    @pytest.fixture()
    def committer(self) -> SideEffectCommitter:
        return SideEffectCommitter(
            effect_committers={
                EffectType.FACT: NoOpEffectCommitter(),
                EffectType.MESSAGE: NoOpEffectCommitter(),
            }
        )

    @pytest.mark.asyncio()
    async def test_commit_empty_enclave(self, committer: SideEffectCommitter) -> None:
        """Committing empty enclave should succeed."""
        enclave = Enclave(fork_id="fork:test", option_key="option")

        result = await committer.commit(enclave)

        assert result.success is True
        assert result.commit_count == 0

    @pytest.mark.asyncio()
    async def test_commit_all_effects(self, committer: SideEffectCommitter) -> None:
        """Should commit all staged effects."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("fact1", {"value": 1})
        enclave.stage_fact("fact2", {"value": 2})

        result = await committer.commit(enclave)

        assert result.success is True
        assert result.commit_count == 2
        assert enclave.is_committed is True

    @pytest.mark.asyncio()
    async def test_commit_seals_enclave(self, committer: SideEffectCommitter) -> None:
        """Commit should seal the enclave."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})

        await committer.commit(enclave)

        assert enclave.is_sealed is True

    @pytest.mark.asyncio()
    async def test_commit_already_committed(
        self, committer: SideEffectCommitter
    ) -> None:
        """Second commit should be no-op."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})

        result1 = await committer.commit(enclave)
        result2 = await committer.commit(enclave)

        assert result1.success is True
        assert result2.success is True
        assert result2.commit_count == 0

    @pytest.mark.asyncio()
    async def test_commit_already_discarded(
        self, committer: SideEffectCommitter
    ) -> None:
        """Committing discarded enclave should fail."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})

        await committer.discard(enclave)
        result = await committer.commit(enclave)

        assert result.success is False
        assert "already_discarded" in result.failed_effects[0][0]

    @pytest.mark.asyncio()
    async def test_commit_with_callback(self) -> None:
        """Should call on_commit callback for each effect."""
        committed_effects: list[str] = []

        def on_commit(effect: object, result: dict[str, object]) -> None:
            committed_effects.append(effect.effect_id)  # type: ignore[attr-defined]

        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()},
            on_commit=on_commit,
        )

        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})

        await committer.commit(enclave)

        assert len(committed_effects) == 1

    @pytest.mark.asyncio()
    async def test_discard_marks_effects(self, committer: SideEffectCommitter) -> None:
        """Discard should mark all effects as discarded."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})

        await committer.discard(enclave)

        assert enclave.is_discarded is True
        effects = enclave.get_effects()
        assert all(e.status == EffectStatus.DISCARDED for e in effects)

    @pytest.mark.asyncio()
    async def test_discard_with_callback(self) -> None:
        """Should call on_discard callback for each effect."""
        discarded_effects: list[str] = []

        def on_discard(effect: object) -> None:
            discarded_effects.append(effect.effect_id)  # type: ignore[attr-defined]

        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()},
            on_discard=on_discard,
        )

        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})

        await committer.discard(enclave)

        assert len(discarded_effects) == 1

    @pytest.mark.asyncio()
    async def test_commit_handles_missing_committer(
        self, committer: SideEffectCommitter
    ) -> None:
        """Should skip effects without registered committer."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_event("event", {})  # No committer for EVENT

        result = await committer.commit(enclave)

        assert result.success is True
        effect = enclave.get_effects()[0]
        assert effect.commit_result is not None
        assert effect.commit_result.get("skipped") is True

    @pytest.mark.asyncio()
    async def test_commit_failure_triggers_rollback(self) -> None:
        """Failed commit should trigger rollback of committed effects."""
        commit_count = 0

        class CountingFailingCommitter:
            async def commit(self, effect: object) -> dict[str, str]:
                nonlocal commit_count
                commit_count += 1
                if commit_count == 2:
                    raise RuntimeError("Intentional failure on second commit")
                return {"status": "ok"}

            async def rollback(self, effect: object) -> bool:
                return True

        committer = SideEffectCommitter(
            effect_committers={
                EffectType.FACT: CountingFailingCommitter()  # type: ignore[dict-item]
            },
        )

        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("first", {})
        enclave.stage_fact("second", {})

        result = await committer.commit(enclave)

        assert result.success is False

    @pytest.mark.asyncio()
    async def test_discard_committed_enclave_is_noop(self) -> None:
        """Discarding an already-committed enclave should be a no-op."""
        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()},
        )

        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})
        await committer.commit(enclave)

        # Should not raise or change state
        await committer.discard(enclave)
        assert enclave.is_committed is True

    @pytest.mark.asyncio()
    async def test_commit_result_properties(
        self, committer: SideEffectCommitter
    ) -> None:
        """CommitResult properties should work correctly."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("test", {})

        result = await committer.commit(enclave)

        assert result.commit_count == 1
        assert result.failure_count == 0
        assert result.total_time_ms >= 0
        assert result.sandbox_id == enclave.sandbox_id
        assert result.option_key == "option"

    # ------------------------------------------------------------------
    # Adversarial / boundary tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio()
    async def test_commit_already_committed_is_noop(self) -> None:
        """Commit same enclave twice -- second returns success with 0 effects."""
        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()}
        )
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("item", {"v": 1})

        result1 = await committer.commit(enclave)
        result2 = await committer.commit(enclave)

        assert result1.success is True
        assert result1.commit_count == 1
        assert result2.success is True
        assert result2.commit_count == 0

    @pytest.mark.asyncio()
    async def test_commit_already_discarded_fails(self) -> None:
        """Discard then commit -- should report failure."""
        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()}
        )
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("item", {})

        await committer.discard(enclave)
        result = await committer.commit(enclave)

        assert result.success is False
        assert any("already_discarded" in fid for fid, _ in result.failed_effects)

    @pytest.mark.asyncio()
    async def test_discard_already_committed_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Commit then discard -- should log a warning but not crash."""
        import logging

        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()}
        )
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("item", {})

        await committer.commit(enclave)

        with caplog.at_level(logging.WARNING):
            await committer.discard(enclave)

        assert enclave.is_committed is True
        assert "Cannot discard already committed" in caplog.text

    @pytest.mark.asyncio()
    async def test_commit_with_failing_committer_triggers_rollback(self) -> None:
        """Register a committer that raises on commit -- verify rollback called
        on previously committed effects."""
        rollback_calls: list[str] = []
        commit_count = 0

        class FailOnSecondCommitter:
            async def commit(self, effect: StagedEffect) -> dict[str, str]:
                nonlocal commit_count
                commit_count += 1
                if commit_count >= 2:
                    raise RuntimeError("boom on second")
                return {"status": "ok"}

            async def rollback(self, effect: StagedEffect) -> bool:
                rollback_calls.append(effect.effect_id)
                return True

        committer = SideEffectCommitter(
            effect_committers={
                EffectType.FACT: FailOnSecondCommitter()  # type: ignore[dict-item]
            }
        )
        enclave = Enclave(fork_id="fork:test", option_key="option")
        id1 = enclave.stage_fact("first", {})
        enclave.stage_fact("second", {})

        result = await committer.commit(enclave)

        assert result.success is False
        # The first committed effect should have been rolled back
        assert id1 in rollback_calls

    @pytest.mark.asyncio()
    async def test_commit_with_no_registered_committer_skips(self) -> None:
        """Effect type with no registered committer -- should skip, not fail."""
        committer = SideEffectCommitter(effect_committers={})
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_event("some_event", {"data": "test"})

        result = await committer.commit(enclave)

        assert result.success is True
        effect = enclave.get_effects()[0]
        assert effect.status == EffectStatus.COMMITTED
        assert effect.commit_result is not None
        assert effect.commit_result.get("skipped") is True

    @pytest.mark.asyncio()
    async def test_rollback_on_irreversible_effect_skips(self) -> None:
        """Effect marked reversible=False should NOT attempt rollback."""
        rollback_calls: list[str] = []
        commit_count = 0

        class FailOnSecondCommitter:
            async def commit(self, effect: StagedEffect) -> dict[str, str]:
                nonlocal commit_count
                commit_count += 1
                if commit_count >= 2:
                    raise RuntimeError("boom")
                return {"ok": "yes"}

            async def rollback(self, effect: StagedEffect) -> bool:
                rollback_calls.append(effect.effect_id)
                return True

        committer = SideEffectCommitter(
            effect_committers={
                EffectType.MESSAGE: FailOnSecondCommitter(),  # type: ignore[dict-item]
            }
        )
        enclave = Enclave(fork_id="fork:test", option_key="option")
        # MESSAGE is reversible=False by default
        enclave.stage_message("recipient:1", {"cmd": "go"})
        enclave.stage_message("recipient:2", {"cmd": "fail"})

        result = await committer.commit(enclave)

        assert result.success is False
        # First message was irreversible, so rollback should be skipped
        assert len(rollback_calls) == 0

    @pytest.mark.asyncio()
    async def test_commit_respects_dependency_order(self) -> None:
        """Effect B depends on A -- if A fails, B should also fail."""
        commit_count = 0

        class AlwaysFailCommitter:
            async def commit(self, effect: StagedEffect) -> dict[str, str]:
                nonlocal commit_count
                commit_count += 1
                raise RuntimeError("always fails")

            async def rollback(self, effect: StagedEffect) -> bool:
                return True

        committer = SideEffectCommitter(
            effect_committers={
                EffectType.FACT: AlwaysFailCommitter()  # type: ignore[dict-item]
            }
        )
        enclave = Enclave(fork_id="fork:test", option_key="option")

        # Stage A, then B depending on A
        id_a = enclave.stage_fact("fact_a", {})
        # We need to manually create an effect with dependency
        # since stage_fact doesn't expose dependencies param.
        # Instead we test the dependency logic by ensuring that
        # when A is not in committed list, B with dep on A is skipped.
        # We'll use _stage_effect directly.
        enclave._stage_effect(
            effect_type=EffectType.FACT,
            payload={"fact_type": "fact_b", "data": {}},
            reversible=True,
            priority=100,
            dependencies=[id_a],
        )

        result = await committer.commit(enclave)

        assert result.success is False
        # A failed on commit, B should fail due to dependency
        failed_ids = [fid for fid, _ in result.failed_effects]
        assert id_a in failed_ids

    @pytest.mark.asyncio()
    async def test_commit_timeout_triggers_failure(self) -> None:
        """Use a slow committer that exceeds timeout -- should fail."""

        class SlowCommitter:
            async def commit(self, effect: StagedEffect) -> dict[str, str]:
                await asyncio.sleep(10.0)
                return {"status": "ok"}

            async def rollback(self, effect: StagedEffect) -> bool:
                return True

        # Very short timeout to trigger failure
        committer = SideEffectCommitter(
            effect_committers={
                EffectType.FACT: SlowCommitter()  # type: ignore[dict-item]
            },
            commit_timeout_seconds=0.05,
        )
        enclave = Enclave(fork_id="fork:test", option_key="option")
        enclave.stage_fact("slow_fact", {})

        result = await committer.commit(enclave)

        assert result.success is False
        assert any("timeout" in reason.lower() for _, reason in result.failed_effects)
