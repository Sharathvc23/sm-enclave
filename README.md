# sm-enclave

Speculative execution sandbox with staged side effects and commit/discard semantics for autonomous agents.

## What It Does

- **Staged side effects** — Effects produced inside a speculative branch are held in an `Enclave` and never applied until the branch wins. Losing branches are discarded without touching external state.
- **Six effect types** — Facts, messages, commands, metrics, events, and external API calls, each with its own reversibility default. `stage_external_api()` is the path for work that lands on someone else's system: a charge, a booking, a provisioning call.
- **Irreversibility gate** — Any effect staged as irreversible is blocked unless the enclave is explicitly configured with `allow_irreversible=True`. The check lives in `_stage_effect`, so it covers every effect type: a hardware command, an un-retractable message, an emitted event, an outbound API call. Messages, events and external API calls are irreversible by default; pass `irreversible=False` to opt out.
- **Commit with compensation, not rewind** — Effects commit in priority order. If any effect fails, effects already committed in that run are revisited in reverse order and their committer's `rollback` is called. For anything that left the process this is a *compensating* action — a refund, a cancellation — not an undo. An effect staged as irreversible is skipped: nothing is attempted and nothing pretends to have been. The flag therefore means "a committer can compensate this," not "the world can be rewound."
- **Parallel execution** — `SpeculativeExecutor` runs multiple branches concurrently in isolated enclaves, then commits the winner and discards the losers.
- **Pluggable committers** — An `EffectCommitter` Protocol lets callers attach their own commit logic per effect type. A logging committer is provided for testing.
- **Zero dependencies** — Standard library only. Python 3.10+.

## Install

```bash
pip install git+https://github.com/Sharathvc23/sm-enclave.git
```

## Quick Start

```python
import asyncio
from sm_enclave import Enclave, SpeculativeExecutor, create_default_committer


class RoutingFork:
    id = "fork:route"

    def options(self):
        return {"route_A": {"cost": 10}, "route_B": {"cost": 20}}


async def main():
    committer = create_default_committer()
    executor = SpeculativeExecutor(committer=committer)

    async def run_route_a(enclave: Enclave):
        enclave.stage_fact("route", {"selected": "A", "cost": 10})
        enclave.stage_metric("route_cost", 10.0)

    async def run_route_b(enclave: Enclave):
        enclave.stage_fact("route", {"selected": "B", "cost": 20})
        enclave.stage_metric("route_cost", 20.0)

    enclaves = await executor.execute_speculatively(
        fork=RoutingFork(),
        executors={"route_A": run_route_a, "route_B": run_route_b},
    )

    result = await executor.finalize("route_A", enclaves)
    print(f"Committed {result.commit_count} effects, success={result.success}")


asyncio.run(main())
```

## Effect Types

| Type | Reversible by default | Default priority | Description |
|------|-----------------------|-----------------|-------------|
| `COMMAND` | Configurable (safe: no) | 50 | Hardware or system commands |
| `FACT` | Yes | 100 | Create or update facts |
| `MESSAGE` | Configurable (default: no) | 200 | Outbound messages |
| `METRIC` | Yes | 300 | Metric updates |
| `EVENT` | Configurable (default: no) | 400 | Event emissions |
| `EXTERNAL_API` | Configurable (default: no) | 150 | Calls to an external service |

`EXTERNAL_API` commits at 150 — after local facts (100) are durable, before messages (200)
announce the result. Every type whose default is "no" is refused by a default enclave; build it
with `allow_irreversible=True`, or pass `irreversible=False` at the call site when a committer is
wired to compensate that specific effect.

### Staging a call that leaves the process

```python
from sm_enclave import Enclave

# A losing branch must never dispatch the charge, so it is staged, not called.
enclave = Enclave(fork_id="fork:checkout", option_key="express")

charge = enclave.stage_external_api(
    "/v1/charges",
    {"amount_cents": 2500, "currency": "usd"},
    irreversible=False,  # the committer below can issue the refund
    idempotency_key="idem:order-4417",
)
enclave.stage_message("ops@example.com", {"order": "4417"}, irreversible=False)

# Nothing above has touched the network. The committer decides, later,
# whether this branch won -- and on a failed commit calls rollback(), where
# the refund lives.
```

Leave `irreversible` at its default when no committer can compensate the endpoint. The call is
then refused by a default enclave, which is the honest outcome: an un-compensable charge does not
belong in a speculative branch.

## Where It's Useful

- AI agents that use tools with real-world side effects (emails, payments, file writes, API calls)
- Robotics motion planning that evaluates multiple trajectories before actuating
- Autonomous trading where strategies are compared before orders are placed
- Industrial process control where control actions are evaluated before dispatch
- Any system where a probabilistic proposer generates actions destined for irreversible actuators

## Related Packages

| Package | Purpose |
|---------|---------|
| [sm-bridge](https://github.com/Sharathvc23/sm-bridge) | NANDA-compatible registry endpoints, AgentFacts, and delta sync |
| [sm-airlock](https://github.com/Sharathvc23/sm-airlock) | Attribute-level capability restriction for agent plugins |
| [sm-locp](https://github.com/Sharathvc23/sm-locp) | Open Compliance Protocol — defeasible logic and W3C Verifiable Credentials |
| [sm-model-card](https://github.com/Sharathvc23/sm-model-card) | Unified model card schema for agent registries |
| [sm-model-provenance](https://github.com/Sharathvc23/sm-model-provenance) | Model identity and provenance metadata |
| [sm-model-integrity-layer](https://github.com/Sharathvc23/sm-model-integrity-layer) | Cryptographic integrity verification for model artifacts |
| [sm-model-governance](https://github.com/Sharathvc23/sm-model-governance) | Three-plane ML governance with Ed25519 signatures |

## License

MIT

---

*First published: 2026-04-15 | Last modified: 2026-04-16*

*Personal research contributions aligned with [Project NANDA](https://projectnanda.org) standards. [Stellarminds.ai](https://stellarminds.ai)*
