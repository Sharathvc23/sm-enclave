"""Tests for SpeculativeExecutor: parallel execution, finalize winner/losers.

# Step 1 -- Assumption Audit
# - execute_speculatively assumes executors dict keys match fork.options() keys
# - finalize assumes winner_key exists in enclaves dict
# - Semaphore assumes max_concurrent_branches is a positive integer
# - asyncio.gather with return_exceptions=True silently swallows
# - execute_single delegates to execute_speculatively (same timeout)
# - Stats counters are plain ints, no lock -- not concurrency-safe
# - Context shallow-copied via set_local_state -- shared references

# Step 2 -- Gap Analysis
# - No test for finalize with a winner_key not present in enclaves
# - No test for executor that raises but others succeed (isolation)
# - No test for executor that sleeps beyond timeout (timeout handling)
# - No test that verifies all losers are actually discarded
# - No test for execute_single with auto_commit=True end-to-end
# - No test that stats track correctly after multiple operations

# Step 3 -- Break It List
# execute_speculatively:
#   (1) executor for key not in fork.options() -> silently skipped
#   (2) sync function that blocks event loop -> hangs
#   (3) mutate context dict during execution -> race condition
# finalize:
#   (1) winner_key="" -> ValueError if not in enclaves
#   (2) empty enclaves dict -> ValueError
#   (3) finalize twice -> double-commit/double-discard
# get_stats:
#   (1) call before any execution -> all zeros (trivially correct)
#   (2) stats not reset between runs -> leaks if shared instance
#   (3) concurrent increments -> race condition on counters
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sm_enclave import (
    Enclave,
    SpeculativeExecutor,
    create_default_committer,
)


class MockFork:
    """Simple fork implementing the Fork protocol."""

    def __init__(
        self,
        fork_id: str = "fork:test",
        option_keys: list[str] | None = None,
    ) -> None:
        self._id = fork_id
        self._option_keys = option_keys or [
            "option_A",
            "option_B",
            "option_C",
        ]

    @property
    def id(self) -> str:
        return self._id

    def options(self) -> dict[str, dict[str, Any]]:
        return {key: {"key": key} for key in self._option_keys}


# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
THREE_BRANCH_KEYS = ["branch_1", "branch_2", "branch_3"]
SHORT_TIMEOUT_S = 0.1
LONG_SLEEP_S = 5.0


class TestSpeculativeExecutor:
    """Tests for SpeculativeExecutor."""

    @pytest.fixture()
    def executor(self) -> SpeculativeExecutor:
        committer = create_default_committer()
        return SpeculativeExecutor(
            committer=committer,
            max_concurrent_branches=5,
        )

    @pytest.fixture()
    def fork(self) -> MockFork:
        return MockFork()

    @pytest.mark.asyncio()
    async def test_execute_speculatively_creates_enclaves(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Should create enclave for each option with executor."""

        async def executor_a(enclave: Enclave) -> None:
            enclave.stage_fact("result_a", {"value": "A"})

        async def executor_b(enclave: Enclave) -> None:
            enclave.stage_fact("result_b", {"value": "B"})

        enclaves = await executor.execute_speculatively(
            fork,
            executors={
                "option_A": executor_a,
                "option_B": executor_b,
            },
        )

        assert "option_A" in enclaves
        assert "option_B" in enclaves
        assert "option_C" not in enclaves  # No executor

        assert enclaves["option_A"].effect_count == 1
        assert enclaves["option_B"].effect_count == 1

    @pytest.mark.asyncio()
    async def test_execute_speculatively_parallel(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Executors should run in parallel."""
        execution_order: list[str] = []

        async def make_executor(option_key: str, delay: float) -> Any:
            async def executor_fn(enclave: Enclave) -> None:
                await asyncio.sleep(delay)
                execution_order.append(option_key)

            return executor_fn

        enclaves = await executor.execute_speculatively(
            fork,
            executors={
                "option_A": await make_executor("option_A", 0.05),
                "option_B": await make_executor("option_B", 0.05),
            },
        )

        assert len(execution_order) == 2
        assert len(enclaves) == 2

    @pytest.mark.asyncio()
    async def test_execute_speculatively_with_context(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Should pass context to enclaves."""
        captured_context: dict[str, Any] = {}

        async def capture_executor(enclave: Enclave) -> None:
            captured_context["value"] = enclave.get_local_state("shared_value")

        await executor.execute_speculatively(
            fork,
            executors={"option_A": capture_executor},
            context={"shared_value": 42},
        )

        assert captured_context["value"] == 42

    @pytest.mark.asyncio()
    async def test_finalize_commits_winner(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Finalize should commit winner and discard losers."""

        async def setup_enclave(enclave: Enclave) -> None:
            enclave.stage_fact("test", {"option": enclave.option_key})

        enclaves = await executor.execute_speculatively(
            fork,
            executors={
                "option_A": setup_enclave,
                "option_B": setup_enclave,
            },
        )

        result = await executor.finalize("option_A", enclaves)

        assert result.success is True
        assert result.option_key == "option_A"
        assert enclaves["option_A"].is_committed is True
        assert enclaves["option_B"].is_discarded is True

    @pytest.mark.asyncio()
    async def test_finalize_unknown_winner_raises(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Finalize with unknown winner should raise."""
        enclaves = await executor.execute_speculatively(
            fork,
            executors={"option_A": lambda s: None},
        )

        with pytest.raises(ValueError, match="not in enclaves"):
            await executor.finalize("unknown_option", enclaves)

    @pytest.mark.asyncio()
    async def test_execute_single_option(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Should execute single option directly."""

        async def single_executor(enclave: Enclave) -> None:
            enclave.stage_fact("single", {"value": 1})

        enclave, result = await executor.execute_single(
            fork,
            "option_A",
            single_executor,
            auto_commit=False,
        )

        assert enclave.effect_count == 1
        assert result is None

    @pytest.mark.asyncio()
    async def test_execute_single_with_auto_commit(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Auto-commit should commit after execution."""

        async def single_executor(enclave: Enclave) -> None:
            enclave.stage_fact("single", {"value": 1})

        enclave, result = await executor.execute_single(
            fork,
            "option_A",
            single_executor,
            auto_commit=True,
        )

        assert enclave.is_committed is True
        assert result is not None
        assert result.success is True

    def test_get_stats(self, executor: SpeculativeExecutor) -> None:
        """Should track executor statistics."""
        stats = executor.get_stats()

        assert "total_executions" in stats
        assert "total_commits" in stats
        assert "total_discards" in stats
        assert "max_concurrent" in stats

    @pytest.mark.asyncio()
    async def test_handles_executor_timeout(self) -> None:
        """Should handle executor timeout gracefully."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(
            committer=committer,
            execution_timeout_seconds=0.1,
        )

        fork = MockFork(option_keys=["slow"])

        async def slow_executor(enclave: Enclave) -> None:
            await asyncio.sleep(1.0)

        enclaves = await executor.execute_speculatively(
            fork,
            executors={"slow": slow_executor},
        )

        assert "slow" in enclaves

    @pytest.mark.asyncio()
    async def test_sync_executor_function(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Should handle synchronous executor functions."""

        def sync_executor(enclave: Enclave) -> None:
            enclave.stage_fact("sync", {"value": 1})

        enclaves = await executor.execute_speculatively(
            fork,
            executors={"option_A": sync_executor},
        )

        assert enclaves["option_A"].effect_count == 1

    @pytest.mark.asyncio()
    async def test_executor_exception_does_not_crash(
        self, executor: SpeculativeExecutor, fork: MockFork
    ) -> None:
        """Executor raising should not crash other branches."""

        async def failing_executor(enclave: Enclave) -> None:
            raise RuntimeError("boom")

        async def ok_executor(enclave: Enclave) -> None:
            enclave.stage_fact("ok", {})

        enclaves = await executor.execute_speculatively(
            fork,
            executors={
                "option_A": failing_executor,
                "option_B": ok_executor,
            },
        )

        assert enclaves["option_B"].effect_count == 1

    @pytest.mark.asyncio()
    async def test_stats_increment_after_operations(
        self,
    ) -> None:
        """Stats should increment after execute and finalize."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(committer=committer)
        fork = MockFork(option_keys=["a", "b"])

        async def noop(enclave: Enclave) -> None:
            enclave.stage_fact("x", {})

        enclaves = await executor.execute_speculatively(
            fork, executors={"a": noop, "b": noop}
        )
        await executor.finalize("a", enclaves)

        stats = executor.get_stats()
        assert stats["total_executions"] == 1
        assert stats["total_commits"] == 1
        assert stats["total_discards"] == 1

    # ------------------------------------------------------------------
    # Adversarial / boundary tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio()
    async def test_finalize_with_nonexistent_winner_raises(self) -> None:
        """winner_key not in sandboxes dict should raise ValueError."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(committer=committer)
        fork = MockFork(option_keys=["a", "b"])

        async def noop(enclave: Enclave) -> None:
            pass

        enclaves = await executor.execute_speculatively(
            fork, executors={"a": noop, "b": noop}
        )

        with pytest.raises(ValueError, match="not in enclaves"):
            await executor.finalize("nonexistent_key", enclaves)

    @pytest.mark.asyncio()
    async def test_execute_with_failing_executor_doesnt_crash(self) -> None:
        """One executor raises, others succeed -- gather handles it."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(committer=committer)
        fork = MockFork(option_keys=["good", "bad", "also_good"])

        async def good_exec(enclave: Enclave) -> None:
            enclave.stage_fact("result", {"ok": True})

        async def bad_exec(enclave: Enclave) -> None:
            raise RuntimeError("intentional explosion")

        enclaves = await executor.execute_speculatively(
            fork,
            executors={
                "good": good_exec,
                "bad": bad_exec,
                "also_good": good_exec,
            },
        )

        assert enclaves["good"].effect_count == 1
        assert enclaves["also_good"].effect_count == 1
        # bad enclave exists but has no staged effects
        assert enclaves["bad"].effect_count == 0

    @pytest.mark.asyncio()
    async def test_execute_with_timeout_executor(self) -> None:
        """Executor that sleeps beyond timeout should not block others."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(
            committer=committer,
            execution_timeout_seconds=SHORT_TIMEOUT_S,
        )
        fork = MockFork(option_keys=["fast", "slow"])

        async def fast_exec(enclave: Enclave) -> None:
            enclave.stage_fact("fast_result", {})

        async def slow_exec(enclave: Enclave) -> None:
            await asyncio.sleep(LONG_SLEEP_S)
            enclave.stage_fact("should_not_appear", {})

        enclaves = await executor.execute_speculatively(
            fork,
            executors={"fast": fast_exec, "slow": slow_exec},
        )

        assert enclaves["fast"].effect_count == 1
        # slow executor timed out, so no effects staged
        assert enclaves["slow"].effect_count == 0

    @pytest.mark.asyncio()
    async def test_finalize_discards_all_losers(self) -> None:
        """3 branches: verify the 2 losers are both discarded."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(committer=committer)
        fork = MockFork(option_keys=THREE_BRANCH_KEYS)

        async def stage_one(enclave: Enclave) -> None:
            enclave.stage_fact("data", {"branch": enclave.option_key})

        enclaves = await executor.execute_speculatively(
            fork,
            executors={k: stage_one for k in THREE_BRANCH_KEYS},
        )

        winner = THREE_BRANCH_KEYS[0]
        result = await executor.finalize(winner, enclaves)

        assert result.success is True
        assert enclaves[winner].is_committed is True
        for loser_key in THREE_BRANCH_KEYS[1:]:
            assert enclaves[loser_key].is_discarded is True

    @pytest.mark.asyncio()
    async def test_execute_single_with_auto_commit_adversarial(self) -> None:
        """execute_single with auto_commit=True should commit and return result."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(committer=committer)
        fork = MockFork(option_keys=["solo"])

        async def solo_exec(enclave: Enclave) -> None:
            enclave.stage_fact("solo_fact", {"value": 42})
            enclave.stage_metric("counter", 1.0)

        enclave, result = await executor.execute_single(
            fork, "solo", solo_exec, auto_commit=True
        )

        assert result is not None
        assert result.success is True
        assert result.commit_count == 2
        assert enclave.is_committed is True
        assert enclave.is_sealed is True

    @pytest.mark.asyncio()
    async def test_stats_tracking(self) -> None:
        """Verify get_stats() counts executions, commits, discards correctly."""
        committer = create_default_committer()
        executor = SpeculativeExecutor(committer=committer)

        # Before any operations
        stats_before = executor.get_stats()
        assert stats_before["total_executions"] == 0
        assert stats_before["total_commits"] == 0
        assert stats_before["total_discards"] == 0

        # Execute with 3 branches, finalize 1 winner (2 discards)
        fork = MockFork(option_keys=THREE_BRANCH_KEYS)

        async def noop(enclave: Enclave) -> None:
            enclave.stage_fact("x", {})

        enclaves = await executor.execute_speculatively(
            fork, executors={k: noop for k in THREE_BRANCH_KEYS}
        )
        await executor.finalize(THREE_BRANCH_KEYS[0], enclaves)

        stats_after = executor.get_stats()
        assert stats_after["total_executions"] == 1
        assert stats_after["total_commits"] == 1
        assert stats_after["total_discards"] == 2
