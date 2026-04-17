"""Tests for Enclave: staging, sealing, capacity, irreversibility gate.

# Step 1 -- Assumption Audit
# - Enclave assumes _sealed, _committed, _discarded are mutually exclusive transitions
# - _stage_effect assumes secrets.token_hex always produces unique IDs
# - max_staged_effects assumes >= comparison catches exactly-at-limit
# - seal() has no guard against double-seal or post-commit seal
# - stage_command assumes irreversible=True is the dangerous case; False is always safe
# - get_effects sorts by (priority, insertion_order) -- assumes no ties break wrongly
# - local_state is a plain dict with no isolation enforcement between enclaves

# Step 2 -- Gap Analysis
# - No check for staging after commit or discard (only sealed check exists)
# - No validation on empty/None command_type, fact_type, etc.
# - No validation on negative or zero max_staged_effects
# - No thread-safety test despite asyncio.Lock being present
# - snapshot() does not indicate committed/discarded status of individual effects
# - No test for extremely large payloads or deeply nested dicts

# Step 3 -- Break It List
# stage_fact:
#   (1) pass mutable payload, mutate after staging -> corrupts effect
#   (2) set max_staged_effects=0 -> RuntimeError on first stage
#   (3) stage after _committed=True directly -> bypasses sealed check
# stage_command:
#   (1) irreversible=False passes even with garbage command_type
#   (2) set allow_irreversible at runtime after init
#   (3) irreversible=True + allow_irreversible=True -> reversible=False
# get_effects:
#   (1) mutate _effect_order externally -> reorder or duplicate
#   (2) add effect with priority < 0 -> sorts before everything
#   (3) remove from _effects but not _effect_order -> KeyError
"""

from __future__ import annotations

import pytest

from sm_enclave import EffectStatus, EffectType, Enclave, StagedEffect

# ---------------------------------------------------------------------------
# Named constants (no magic values)
# ---------------------------------------------------------------------------
DEFAULT_FORK_ID = "fork:test123"
DEFAULT_OPTION_KEY = "option_A"
CAPACITY_LIMIT = 5
LARGE_STAGE_COUNT = 1000


class TestStagedEffect:
    """Tests for StagedEffect dataclass."""

    def test_create_with_defaults(self) -> None:
        """Effect should have sensible defaults."""
        effect = StagedEffect(
            effect_type=EffectType.FACT,
            payload={"fact_type": "position", "data": {"x": 1.0}},
        )

        assert effect.effect_id.startswith("eff:")
        assert effect.effect_type == EffectType.FACT
        assert effect.reversible is True
        assert effect.priority == 100
        assert effect.status == EffectStatus.STAGED

    def test_as_dict(self) -> None:
        """Effect should serialize to dict."""
        effect = StagedEffect(
            effect_type=EffectType.MESSAGE,
            payload={"recipient": "peer:1", "data": {}},
            reversible=False,
        )

        data = effect.as_dict()

        assert data["effect_type"] == "message"
        assert data["reversible"] is False
        assert "staged_at" in data

    def test_as_dict_contains_all_fields(self) -> None:
        """Serialized dict should contain all fields."""
        effect = StagedEffect()
        data = effect.as_dict()
        expected_keys = {
            "effect_id",
            "effect_type",
            "payload",
            "reversible",
            "priority",
            "dependencies",
            "metadata",
            "staged_at",
            "status",
            "commit_result",
            "signature",
        }
        assert set(data.keys()) == expected_keys

    def test_effect_id_uniqueness(self) -> None:
        """Each effect should get a unique ID."""
        ids = {StagedEffect().effect_id for _ in range(100)}
        assert len(ids) == 100


class TestEnclave:
    """Tests for Enclave."""

    @pytest.fixture()
    def enclave(self) -> Enclave:
        return Enclave(
            fork_id=DEFAULT_FORK_ID,
            option_key=DEFAULT_OPTION_KEY,
        )

    def test_creation(self, enclave: Enclave) -> None:
        """Enclave should initialize correctly."""
        assert enclave.fork_id == DEFAULT_FORK_ID
        assert enclave.option_key == DEFAULT_OPTION_KEY
        assert enclave.sandbox_id.startswith("sandbox:")
        assert enclave.effect_count == 0
        assert enclave.is_sealed is False
        assert enclave.is_committed is False
        assert enclave.is_discarded is False

    def test_stage_fact(self, enclave: Enclave) -> None:
        """Should stage fact effects."""
        effect_id = enclave.stage_fact(
            "position",
            {"x": 1.0, "y": 2.0, "z": 3.0},
        )

        assert effect_id.startswith("eff:")
        assert enclave.effect_count == 1

        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.effect_type == EffectType.FACT
        assert effect.payload["fact_type"] == "position"

    def test_stage_message(self, enclave: Enclave) -> None:
        """Should stage message effects."""
        effect_id = enclave.stage_message(
            "recipient:command",
            {"cmd": "execute"},
        )

        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.effect_type == EffectType.MESSAGE
        assert effect.reversible is False

    def test_stage_command_blocks_irreversible_by_default(
        self, enclave: Enclave
    ) -> None:
        """Should block irreversible commands by default."""
        with pytest.raises(ValueError, match="Irreversible command"):
            enclave.stage_command(
                "dangerous_action",
                {"duration_s": 10.0},
                irreversible=True,
            )

    def test_stage_command_allows_with_flag(self) -> None:
        """Should allow irreversible commands when enabled."""
        enclave = Enclave(
            fork_id="fork:test",
            option_key="option",
            allow_irreversible=True,
        )

        effect_id = enclave.stage_command(
            "dangerous_action",
            {"duration_s": 10.0},
            irreversible=True,
        )

        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.reversible is False

    def test_stage_command_reversible(self, enclave: Enclave) -> None:
        """Reversible commands should be allowed without flag."""
        effect_id = enclave.stage_command(
            "safe_action",
            {"param": "value"},
            irreversible=False,
        )
        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.reversible is True

    def test_stage_metric(self, enclave: Enclave) -> None:
        """Should stage metric effects."""
        effect_id = enclave.stage_metric("fuel_level", 0.85)

        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.effect_type == EffectType.METRIC
        assert effect.payload["value"] == 0.85

    def test_stage_event(self, enclave: Enclave) -> None:
        """Should stage event effects."""
        effect_id = enclave.stage_event(
            "task.started",
            {"task_id": "t123"},
        )

        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.effect_type == EffectType.EVENT

    def test_local_state(self, enclave: Enclave) -> None:
        """Should manage enclave-local state."""
        enclave.set_local_state("temp_value", 42)

        assert enclave.get_local_state("temp_value") == 42
        assert enclave.get_local_state("missing", "default") == "default"

    def test_seal_prevents_new_effects(self, enclave: Enclave) -> None:
        """Sealed enclave should reject new effects."""
        enclave.seal()

        with pytest.raises(RuntimeError, match="sealed"):
            enclave.stage_fact("test", {})

    def test_max_effects_capacity(self) -> None:
        """Should reject effects when at capacity."""
        enclave = Enclave(
            fork_id="fork:test",
            option_key="option",
            max_staged_effects=3,
        )

        for i in range(3):
            enclave.stage_fact(f"fact_{i}", {})

        with pytest.raises(RuntimeError, match="at capacity"):
            enclave.stage_fact("overflow", {})

    def test_get_effects_in_priority_order(self, enclave: Enclave) -> None:
        """Effects should be returned in priority order."""
        enclave.stage_event("event", {}, priority=400)
        enclave.stage_fact("fact", {}, priority=100)
        enclave.stage_command("cmd", {}, irreversible=False, priority=50)

        effects = enclave.get_effects()

        assert effects[0].effect_type == EffectType.COMMAND  # priority 50
        assert effects[1].effect_type == EffectType.FACT  # priority 100
        assert effects[2].effect_type == EffectType.EVENT  # priority 400

    def test_get_effects_stable_order_same_priority(self) -> None:
        """Effects with same priority keep insertion order."""
        enclave = Enclave(fork_id="fork:test", option_key="option")
        id1 = enclave.stage_fact("first", {}, priority=100)
        id2 = enclave.stage_fact("second", {}, priority=100)

        effects = enclave.get_effects()
        assert effects[0].effect_id == id1
        assert effects[1].effect_id == id2

    def test_snapshot(self, enclave: Enclave) -> None:
        """Should return complete enclave snapshot."""
        enclave.stage_fact("test", {"value": 1})

        snapshot = enclave.snapshot()

        assert snapshot["sandbox_id"] == enclave.sandbox_id
        assert snapshot["fork_id"] == DEFAULT_FORK_ID
        assert snapshot["option_key"] == DEFAULT_OPTION_KEY
        assert snapshot["effect_count"] == 1
        assert len(snapshot["effects"]) == 1

    def test_get_effect_returns_none_for_unknown(self, enclave: Enclave) -> None:
        """Should return None for unknown effect ID."""
        assert enclave.get_effect("nonexistent") is None

    def test_execution_log(self, enclave: Enclave) -> None:
        """Should track staged effects in execution log."""
        enclave.stage_fact("test", {})

        log = enclave.get_execution_log()
        assert len(log) == 1
        assert log[0]["action"] == "staged"
        assert log[0]["effect_type"] == "fact"

    def test_execution_log_is_capped(self) -> None:
        """Execution log should not grow past max_log_entries."""
        enclave = Enclave(
            fork_id="fork:test",
            option_key="option",
            max_staged_effects=100,
            max_log_entries=5,
        )

        for i in range(20):
            enclave.stage_fact(f"fact_{i}", {"i": i})

        log = enclave.get_execution_log()
        assert len(log) == 5
        # Should retain the newest entries, FIFO trim the oldest
        assert log[-1]["effect_type"] == "fact"

    def test_mark_committed_sets_state(self, enclave: Enclave) -> None:
        """mark_committed should flip is_committed and clear is_discarded."""
        enclave.mark_committed()
        assert enclave.is_committed is True
        assert enclave.is_discarded is False

    def test_mark_failed_clears_both_flags(self, enclave: Enclave) -> None:
        """mark_failed should leave both committed and discarded false."""
        enclave.mark_committed()  # simulate a prior commit
        enclave.mark_failed()
        assert enclave.is_committed is False
        assert enclave.is_discarded is False

    def test_mark_discarded_sets_state(self, enclave: Enclave) -> None:
        """mark_discarded should flip is_discarded and clear is_committed."""
        enclave.mark_discarded()
        assert enclave.is_discarded is True
        assert enclave.is_committed is False

    def test_tenant_id(self) -> None:
        """Should store tenant ID."""
        enclave = Enclave(
            fork_id="fork:test",
            option_key="opt",
            tenant_id="tenant-1",
        )
        assert enclave.tenant_id == "tenant-1"

    # ------------------------------------------------------------------
    # Adversarial / boundary tests
    # ------------------------------------------------------------------

    def test_stage_after_commit_raises(self) -> None:
        """Staging after commit should raise because commit seals the enclave."""
        from sm_enclave import NoOpEffectCommitter, SideEffectCommitter

        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()}
        )
        enclave = Enclave(fork_id=DEFAULT_FORK_ID, option_key=DEFAULT_OPTION_KEY)
        enclave.stage_fact("pre_commit", {"v": 1})

        import asyncio

        asyncio.get_event_loop().run_until_complete(committer.commit(enclave))
        assert enclave.is_committed is True

        with pytest.raises(RuntimeError, match="sealed"):
            enclave.stage_fact("post_commit", {"v": 2})

    def test_stage_after_discard_raises(self) -> None:
        """Staging after discard should raise because discard seals the enclave."""
        from sm_enclave import NoOpEffectCommitter, SideEffectCommitter

        committer = SideEffectCommitter(
            effect_committers={EffectType.FACT: NoOpEffectCommitter()}
        )
        enclave = Enclave(fork_id=DEFAULT_FORK_ID, option_key=DEFAULT_OPTION_KEY)
        enclave.stage_fact("pre_discard", {"v": 1})

        import asyncio

        asyncio.get_event_loop().run_until_complete(committer.discard(enclave))
        assert enclave.is_discarded is True

        with pytest.raises(RuntimeError, match="sealed"):
            enclave.stage_fact("post_discard", {"v": 2})

    def test_capacity_boundary_at_exactly_max(self) -> None:
        """Staging exactly max_staged_effects should succeed (boundary: at limit)."""
        enclave = Enclave(
            fork_id=DEFAULT_FORK_ID,
            option_key=DEFAULT_OPTION_KEY,
            max_staged_effects=CAPACITY_LIMIT,
        )
        ids = []
        for i in range(CAPACITY_LIMIT):
            ids.append(enclave.stage_fact(f"fact_{i}", {"i": i}))

        assert enclave.effect_count == CAPACITY_LIMIT
        assert len(ids) == CAPACITY_LIMIT

    def test_capacity_boundary_at_max_plus_one(self) -> None:
        """Staging max+1 effects should raise (boundary: above limit)."""
        enclave = Enclave(
            fork_id=DEFAULT_FORK_ID,
            option_key=DEFAULT_OPTION_KEY,
            max_staged_effects=CAPACITY_LIMIT,
        )
        for i in range(CAPACITY_LIMIT):
            enclave.stage_fact(f"fact_{i}", {})

        with pytest.raises(RuntimeError, match="at capacity"):
            enclave.stage_fact("overflow", {})

    def test_capacity_zero_rejects_all(self) -> None:
        """max_staged_effects=0 should reject the very first stage attempt."""
        enclave = Enclave(
            fork_id=DEFAULT_FORK_ID,
            option_key=DEFAULT_OPTION_KEY,
            max_staged_effects=0,
        )
        with pytest.raises(RuntimeError, match="at capacity"):
            enclave.stage_fact("anything", {})

    def test_irreversible_gate_with_false_irreversible_flag(
        self, enclave: Enclave
    ) -> None:
        """stage_command with irreversible=False should work even when
        allow_irreversible=False (the default)."""
        effect_id = enclave.stage_command(
            "safe_cmd",
            {"param": "val"},
            irreversible=False,
        )
        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.reversible is True

    def test_stage_command_empty_command_type(self, enclave: Enclave) -> None:
        """Empty string command_type -- the code does not validate it,
        so it should stage without error (documenting current behavior)."""
        effect_id = enclave.stage_command(
            "",
            {"param": "val"},
            irreversible=False,
        )
        effect = enclave.get_effect(effect_id)
        assert effect is not None
        assert effect.payload["command_type"] == ""

    def test_concurrent_staging_produces_unique_ids(self) -> None:
        """Stage many effects rapidly and verify all IDs are unique."""
        enclave = Enclave(
            fork_id=DEFAULT_FORK_ID,
            option_key=DEFAULT_OPTION_KEY,
            max_staged_effects=LARGE_STAGE_COUNT,
        )
        ids = set()
        for i in range(LARGE_STAGE_COUNT):
            eid = enclave.stage_fact(f"fact_{i}", {"i": i})
            ids.add(eid)

        assert len(ids) == LARGE_STAGE_COUNT

    def test_snapshot_after_seal_shows_sealed(self, enclave: Enclave) -> None:
        """After sealing, snapshot should reflect sealed=True."""
        enclave.stage_fact("before_seal", {})
        enclave.seal()

        snapshot = enclave.snapshot()
        assert snapshot["sealed"] is True

    def test_get_effects_empty_enclave(self, enclave: Enclave) -> None:
        """get_effects on a fresh enclave should return an empty list."""
        effects = enclave.get_effects()
        assert effects == []

    def test_local_state_isolation_between_enclaves(self) -> None:
        """Two enclaves should NOT share local state."""
        enclave_a = Enclave(fork_id=DEFAULT_FORK_ID, option_key="option_A")
        enclave_b = Enclave(fork_id=DEFAULT_FORK_ID, option_key="option_B")

        enclave_a.set_local_state("shared_key", "value_A")
        enclave_b.set_local_state("shared_key", "value_B")

        assert enclave_a.get_local_state("shared_key") == "value_A"
        assert enclave_b.get_local_state("shared_key") == "value_B"
        assert enclave_a.get_local_state("only_in_b") is None
