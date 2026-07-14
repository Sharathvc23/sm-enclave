"""Tests for NoOp, Logging committers and create_default_committer factory.

# Step 1 -- Assumption Audit
# - NoOp rollback always returns True -- assumes no-op is safe
# - LoggingEffectCommitter assumes logger.log never raises
# - create_default_committer registers one LoggingEffectCommitter
#   instance for all types
# - log_level defaults to INFO -- caller must know logging levels
# - rollback on both committers always True -- no partial failure

# Step 2 -- Gap Analysis
# - No test for NoOp rollback return value (always True)
# - No test for LoggingEffectCommitter with very large payloads
# - No test that create_default covers every EffectType member
# - No test for log_level=0 or invalid log levels
# - No test for commit with None fields in the effect

# Step 3 -- Break It List
# NoOpEffectCommitter.commit:
#   (1) effect with None effect_id -> returns {"effect_id": None}
#   (2) huge payload -> still returns noop
#   (3) commit 1000 times -> stateless, always ok
# LoggingEffectCommitter.commit:
#   (1) effect_type with % in value -> format string injection
#   (2) non-serializable payload -> logger formats it
#   (3) log_level=0 (NOTSET) -> still logs
# create_default_committer:
#   (1) new EffectType without updating factory -> uncovered
#   (2) call twice -> two independent committers (no singleton)
#   (3) mutate _committers dict -> affects only that instance
"""

from __future__ import annotations

import logging

import pytest

from sm_enclave import (
    EffectType,
    LoggingEffectCommitter,
    NoOpEffectCommitter,
    StagedEffect,
    create_default_committer,
)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
LARGE_PAYLOAD_SIZE = 10_000


class TestNoOpEffectCommitter:
    """Tests for NoOpEffectCommitter."""

    @pytest.mark.asyncio()
    async def test_commit_returns_noop(self) -> None:
        """Should return noop result."""
        committer = NoOpEffectCommitter()
        effect = StagedEffect(effect_type=EffectType.FACT, payload={})

        result = await committer.commit(effect)

        assert result["status"] == "noop"
        assert result["effect_id"] == effect.effect_id

    @pytest.mark.asyncio()
    async def test_rollback_succeeds(self) -> None:
        """Rollback should always succeed."""
        committer = NoOpEffectCommitter()
        effect = StagedEffect(effect_type=EffectType.FACT, payload={})

        result = await committer.rollback(effect)

        assert result is True

    # --- Adversarial ---

    @pytest.mark.asyncio()
    async def test_noop_rollback_always_returns_true(self) -> None:
        """NoOp rollback must return True for every effect type."""
        committer = NoOpEffectCommitter()
        for effect_type in EffectType:
            effect = StagedEffect(
                effect_type=effect_type,
                payload={"data": "test"},
            )
            result = await committer.rollback(effect)
            assert result is True, f"Rollback failed for {effect_type}"


class TestLoggingEffectCommitter:
    """Tests for LoggingEffectCommitter."""

    @pytest.mark.asyncio()
    async def test_commit_returns_logged(self) -> None:
        """Should return logged result."""
        committer = LoggingEffectCommitter()
        effect = StagedEffect(
            effect_type=EffectType.MESSAGE,
            payload={"data": "test"},
        )

        result = await committer.commit(effect)

        assert result["logged"] is True
        assert result["effect_id"] == effect.effect_id

    @pytest.mark.asyncio()
    async def test_commit_logs_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log the effect details."""
        committer = LoggingEffectCommitter(log_level=logging.INFO)
        effect = StagedEffect(
            effect_type=EffectType.FACT,
            payload={"fact_type": "test"},
        )

        with caplog.at_level(logging.INFO):
            await committer.commit(effect)

        assert "COMMIT" in caplog.text
        assert "fact" in caplog.text

    @pytest.mark.asyncio()
    async def test_rollback_succeeds(self) -> None:
        """Rollback should always succeed."""
        committer = LoggingEffectCommitter()
        effect = StagedEffect(effect_type=EffectType.FACT, payload={})

        result = await committer.rollback(effect)

        assert result is True

    @pytest.mark.asyncio()
    async def test_rollback_logs_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log rollback."""
        committer = LoggingEffectCommitter(log_level=logging.INFO)
        effect = StagedEffect(effect_type=EffectType.FACT, payload={})

        with caplog.at_level(logging.INFO):
            await committer.rollback(effect)

        assert "ROLLBACK" in caplog.text

    @pytest.mark.asyncio()
    async def test_custom_log_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should use the configured log level."""
        committer = LoggingEffectCommitter(log_level=logging.DEBUG)
        effect = StagedEffect(effect_type=EffectType.FACT, payload={})

        with caplog.at_level(logging.DEBUG):
            await committer.commit(effect)

        assert len(caplog.records) >= 1

    # --- Adversarial ---

    @pytest.mark.asyncio()
    async def test_logging_committer_handles_large_payload(self) -> None:
        """Payload with many characters should not crash the logger."""
        committer = LoggingEffectCommitter(log_level=logging.DEBUG)
        large_data = "x" * LARGE_PAYLOAD_SIZE
        effect = StagedEffect(
            effect_type=EffectType.FACT,
            payload={"fact_type": "big", "data": large_data},
        )

        # Should not raise
        result = await committer.commit(effect)
        assert result["logged"] is True
        assert result["effect_id"] == effect.effect_id


class TestCreateDefaultCommitter:
    """Tests for create_default_committer factory."""

    def test_creates_with_logging_committers(self) -> None:
        """Should create committer with logging handlers for all types."""
        committer = create_default_committer()

        assert committer is not None
        # All effect types should have a committer registered
        assert len(committer._committers) == len(EffectType)

    def test_all_types_registered(self) -> None:
        """Every EffectType should have a registered committer."""
        committer = create_default_committer()

        for effect_type in EffectType:
            assert effect_type in committer._committers, (
                f"Missing committer for {effect_type}"
            )

    # --- Adversarial ---

    def test_create_default_committer_covers_all_types(self) -> None:
        """Verify that every single EffectType enum member has a registered
        committer, and that the committer is callable (has commit/rollback)."""
        committer = create_default_committer()
        all_types = set(EffectType)
        registered_types = set(committer._committers.keys())

        missing = all_types - registered_types
        assert missing == set(), f"EffectTypes without committer: {missing}"

        for effect_type in EffectType:
            c = committer._committers[effect_type]
            assert hasattr(c, "commit"), f"{effect_type} committer missing commit()"
            assert hasattr(c, "rollback"), f"{effect_type} committer missing rollback()"
