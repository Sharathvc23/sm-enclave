"""Speculative executor for parallel branch execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol

from .committer import SideEffectCommitter
from .enclave import Enclave
from .types import CommitResult

logger = logging.getLogger(__name__)


class Fork(Protocol):
    """Minimal fork interface for speculative execution.

    A fork is anything that exposes a stable ``id`` and a set of options to
    evaluate in parallel. Options are returned as a dict keyed by the option
    name; the value is an option-specific payload dict which may be empty.
    """

    @property
    def id(self) -> str: ...

    def options(self) -> dict[str, dict[str, Any]]: ...


class SpeculativeExecutor:
    """Orchestrates parallel enclave execution.

    Creates enclaves for each option, runs executors in parallel,
    and manages commit/discard based on the selected winner.

    Usage:
        executor = SpeculativeExecutor(
            committer=committer,
            max_concurrent_branches=5,
        )

        # Execute all options speculatively
        enclaves = await executor.execute_speculatively(
            fork=fork,
            executors={
                "option_A": execute_a,
                "option_B": execute_b,
            },
        )

        # Finalize with winner
        result = await executor.finalize("option_A", enclaves)
    """

    def __init__(
        self,
        committer: SideEffectCommitter,
        *,
        max_concurrent_branches: int = 5,
        allow_irreversible: bool = False,
        execution_timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize executor.

        Args:
            committer: Side effect committer
            max_concurrent_branches: Maximum parallel branches
            allow_irreversible: Allow irreversible effects
            execution_timeout_seconds: Execution timeout
        """
        self._committer = committer
        self._max_concurrent = max_concurrent_branches
        self._allow_irreversible = allow_irreversible
        self._timeout = execution_timeout_seconds

        self._total_executions = 0
        self._total_commits = 0
        self._total_discards = 0

    async def execute_speculatively(
        self,
        fork: Fork,
        executors: dict[str, Callable[[Enclave], Any]],
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Enclave]:
        """Execute all options speculatively in parallel enclaves.

        Args:
            fork: Fork to execute options for
            executors: Map of option_key -> async executor function
            context: Optional shared context (read-only)

        Returns:
            Dict mapping option_key to Enclave
        """
        self._total_executions += 1

        enclaves: dict[str, Enclave] = {}
        options = fork.options()

        for option_key in options:
            if option_key not in executors:
                continue

            enclaves[option_key] = Enclave(
                fork_id=fork.id,
                option_key=option_key,
                allow_irreversible=self._allow_irreversible,
            )

            if context:
                for key, value in context.items():
                    enclaves[option_key].set_local_state(key, value)

        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def run_executor(option_key: str, enclave: Enclave) -> None:
            async with semaphore:
                executor_fn = executors[option_key]
                try:
                    if asyncio.iscoroutinefunction(executor_fn):
                        await asyncio.wait_for(
                            executor_fn(enclave),
                            timeout=self._timeout,
                        )
                    else:
                        executor_fn(enclave)
                except TimeoutError:
                    logger.warning("Executor for %s timed out", option_key)
                except Exception as e:
                    logger.warning(
                        "Executor for %s failed: %s",
                        option_key,
                        e,
                        exc_info=True,
                    )

        tasks = [run_executor(key, enclave) for key, enclave in enclaves.items()]
        await asyncio.gather(*tasks, return_exceptions=True)

        return enclaves

    async def finalize(
        self,
        winner_key: str,
        enclaves: dict[str, Enclave],
    ) -> CommitResult:
        """Finalize execution by committing winner and discarding losers.

        Args:
            winner_key: Option key that won
            enclaves: All enclaves from speculative execution

        Returns:
            CommitResult for the winning enclave
        """
        if winner_key not in enclaves:
            raise ValueError(f"Winner key '{winner_key}' not in enclaves")

        # Discard all non-winners first
        discard_tasks = []
        for key, enclave in enclaves.items():
            if key != winner_key:
                discard_tasks.append(self._committer.discard(enclave))
                self._total_discards += 1

        await asyncio.gather(*discard_tasks, return_exceptions=True)

        # Commit winner
        self._total_commits += 1
        return await self._committer.commit(enclaves[winner_key])

    async def execute_single(
        self,
        fork: Fork,
        option_key: str,
        executor: Callable[[Enclave], Any],
        *,
        context: dict[str, Any] | None = None,
        auto_commit: bool = False,
    ) -> tuple[Enclave, CommitResult | None]:
        """Execute a single option (non-speculative).

        Useful for emergency decisions or single-option forks.

        Args:
            fork: Fork to execute option for
            option_key: Option to execute
            executor: Async executor function
            context: Optional context
            auto_commit: Automatically commit after execution

        Returns:
            Tuple of (enclave, commit_result or None)
        """
        enclaves = await self.execute_speculatively(
            fork=fork,
            executors={option_key: executor},
            context=context,
        )

        enclave = enclaves[option_key]
        commit_result = None

        if auto_commit:
            commit_result = await self.finalize(option_key, enclaves)

        return enclave, commit_result

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            "total_executions": self._total_executions,
            "total_commits": self._total_commits,
            "total_discards": self._total_discards,
            "max_concurrent": self._max_concurrent,
            "allow_irreversible": self._allow_irreversible,
        }
