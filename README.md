# sm-enclave

Speculative execution sandbox with staged side effects and commit/discard semantics for autonomous agents.

## What It Does

- **Staged side effects** — Effects produced inside a speculative branch are held in an `Enclave` and never applied until the branch wins. Losing branches are discarded without touching external state.
- **Six effect types** — Facts, messages, commands, metrics, events, and external API calls, each with its own reversibility default.
- **Irreversibility gate** — Commands marked `irreversible=True` are blocked unless the enclave is explicitly configured with `allow_irreversible=True`. Prevents speculative branches from firing hardware actions, sending un-retractable messages, or triggering other actions that cannot be undone.
- **Atomic commit with rollback** — Effects commit in priority order. If any effect fails, previously committed reversible effects roll back in reverse order.
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
| `MESSAGE` | No | 200 | Outbound messages |
| `METRIC` | Yes | 300 | Metric updates |
| `EVENT` | No | 400 | Event emissions |
| `EXTERNAL_API` | No | 100 | External API calls |

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
