# sm-enclave: Speculative Execution with Staged Side Effects

## Problem

Autonomous agents must evaluate multiple decision branches before committing to one. During evaluation, branches may produce side effects -- facts, messages, commands -- that should only become real if the branch wins. Without isolation, speculative exploration contaminates shared state and may trigger irreversible real-world actions.

## Core Concepts

### Staged Effects

Every side effect produced during speculative execution is *staged*, not applied. The `Enclave` collects effects as `StagedEffect` objects with:

- **Type** -- one of six categories (fact, message, command, metric, event, external API), each with distinct reversibility semantics
- **Priority** -- determines commit order (lower values commit first)
- **Dependencies** -- effect IDs that must commit before this effect
- **Reversibility** -- whether the effect can be rolled back after commit

Effects remain in `STAGED` status until the enclave is committed or discarded.

### Irreversibility Gating

Commands marked `irreversible=True` are blocked by default. The `allow_irreversible` flag must be explicitly set on the enclave to permit them. This prevents speculative branches from triggering hardware actuations, financial settlements, or other actions that cannot be undone.

The gate operates at staging time, not commit time -- an irreversible effect that should not exist is never created.

### Atomic Commit with Rollback

`SideEffectCommitter` commits an enclave's effects in priority order:

1. **Seal** the enclave to prevent new effects during commit
2. **Iterate** effects in priority order, checking dependency satisfaction
3. **Commit** each effect via its type-specific `EffectCommitter`
4. **On failure**, rollback all previously committed (reversible) effects in reverse order
5. **Mark** the enclave as committed or failed

The result is atomic: either all effects succeed, or the system rolls back to a clean state. Irreversible effects that were committed before a failure are logged but cannot be undone -- this is why the irreversibility gate matters.

### Parallel Speculative Execution

`SpeculativeExecutor` orchestrates the full lifecycle:

1. **Create** an isolated `Enclave` for each decision option
2. **Execute** branch logic in parallel, bounded by a concurrency semaphore
3. **Finalize** by committing the winner and discarding all losers

Each enclave is fully isolated -- no shared mutable state between branches. Local state (`set_local_state` / `get_local_state`) is enclave-scoped and never persisted.

### Fork Protocol

The `Fork` protocol is minimal:

```python
class Fork(Protocol):
    @property
    def id(self) -> str: ...
    def options(self) -> dict[str, dict[str, Any]] | list[str]: ...
```

Any decision system can implement this protocol to drive speculative execution. The executor does not know or care how options are generated, scored, or selected -- it only needs the option keys and a way to identify the fork.

## Architecture

```
Fork
  |
  v
SpeculativeExecutor
  |
  +-- Enclave (option_A)  -->  [StagedEffect, StagedEffect, ...]
  |
  +-- Enclave (option_B)  -->  [StagedEffect, StagedEffect, ...]
  |
  v
Winner selected
  |
  +-- SideEffectCommitter.commit(winner)    --> CommitResult
  |
  +-- SideEffectCommitter.discard(losers)   --> effects marked DISCARDED
```

## Effect Types

| Type | Reversible | Default Priority | Description |
|------|-----------|-----------------|-------------|
| `COMMAND` | Configurable | 50 | Hardware or system commands |
| `FACT` | Yes | 100 | Create or update facts |
| `MESSAGE` | No | 200 | Outbound messages |
| `METRIC` | Yes | 300 | Metric updates |
| `EVENT` | No | 400 | Event emissions |
| `EXTERNAL_API` | No | 100 | External API calls |

## Design Decisions

**Staging at creation time, not commit time.** The irreversibility gate rejects effects when `stage_command()` is called, not when `commit()` runs. This fail-fast approach prevents branches from accumulating effects they can never safely commit.

**Priority-ordered commit.** Effects commit in priority order rather than insertion order. This ensures commands execute before facts are recorded and facts are recorded before events are emitted -- preserving causal ordering.

**Rollback in reverse.** On failure, committed effects roll back in reverse priority order. This mirrors database transaction semantics and ensures dependent effects are undone before their dependencies.

**No cross-enclave communication.** Enclaves are strictly isolated. There is no mechanism for one branch to read another branch's staged effects. This eliminates a class of concurrency bugs and ensures branch independence.

## Ecosystem

| Package | What it provides |
|---------|-----------------|
| **sm-enclave** | Speculative execution sandbox (this package) |
| **sm-airlock** | Capability-gated message validation |
| **sm-bridge** | Trust bridge to external systems |
| **sm-model-card** | Model card metadata |
| **sm-model-governance** | ML model governance gates |
| **sm-model-integrity-layer** | Cryptographic model integrity |
| **sm-model-provenance** | Model provenance metadata |

## Status

v0.2.0 -- zero runtime dependencies. Python 3.10+.

---

*First published: 2026-04-15 | Last modified: 2026-04-15*

*Personal research contributions aligned with [Project NANDA](https://projectnanda.org) standards. [Stellarminds.ai](https://stellarminds.ai)*
