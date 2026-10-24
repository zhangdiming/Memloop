# memloop

Detection → attribution → recovery for LLM-agent long-term memory, with certificates a third party can check.

## Correctness criterion

Recovering from a seed set *S* (poisoned inputs, or entries to forget) must leave memory, outputs and actions equal to a run in which *S* never happened. That is the **never-poisoned counterfactual**, and it is exact when the agent's step function is deterministic. `tests/test_recovery.py` checks it directly: each random script is run with and without attacker steps, and the recovered state must match the clean run.

Rollback and verified deletion are the same operation (`mode="quarantine"` vs `mode="delete"`).

## Components

| module | role |
|---|---|
| `ledger.py` | Append-only Merkle log with RFC 6962/9162 hashing, inclusion and consistency proofs, and HMAC-signed tree heads |
| `labels.py` | A label is bound to its origin when content enters the agent. A derived label is the join (minimum trust, union of origins), and nothing can raise it, so content cannot be laundered through summaries or tool echoes. Also holds the action policy. |
| `trace.py` | Provenance graph with steps as first-class records (trigger, query/k/hits, products). The log stores salted commitments, never plaintext. |
| `store.py` | Vector memory. Retrieval is logged. Actions are checked against IFC rules. The index digest commits to exactly the vectors that are live. |
| `recovery.py` | `recover()`, `attribute()`, and the auditor's `verify_certificate()` / `audit()` |
| `harness.py` | `run_step()`, which plugs an LLM in as `step_fn(query, obs, hits) -> [(kind, text, meta)]` |

## How recovery works

Steps are processed in their original order, starting from the earliest seed. A step is re-run when one of three things holds:

1. **Its trigger is tainted.** The step is dropped.
2. **Its context or a product's parent is tainted.** The step is re-run on clean context.
3. **Its retrieval would now return different hits** (retrieval displacement). The step is re-run.

Case 3 is not found by following parent edges. On toy random scripts, 7% of runs had at least one such step. Pruning re-searches a step only when a changed entry could enter or leave its top-k, which cut searches by 24% with an identical result.

Irreversible actions that the re-run does not reproduce are listed as `compensations`. That includes the attacker's own actions, since they do not exist in the counterfactual.

A step that produced a seed is not re-run, because re-running it would regenerate the very entry being removed. Only the products derived from the seed are taken out.

## Contain-then-repair

`contain(store, seeds)` runs before `recover()`. It immediately pulls every entry in the parent-edge closure out of the index. This over-approximates the damage, but the store is safe at once. `recover()` then re-runs the affected steps and restores the benign part.

- **Final state:** the same as `recover()` alone.
- **Safety during repair:** no contaminated entry is retrievable at any point. Plain `recover()` does not guarantee this: entries from later steps stay live until the replay reaches their step. The tests show this window occurs.
- **Audit:** `audit_containment()` checks the containment step against the log.

## Baseline B3 and what the displacement check is for

B3 is parent-edge replay only, as in dependency-guided rollback: `recover(..., displaced=lambda *a: False)`. It never re-runs a step because of retrieval displacement.

On the toy scripts it has not been observed to diverge from the counterfactual (0 of ~1500). A recomputed entry usually carries the same advice as the entry it would displace. In the MINJA pilot, displacement caused 0–1 re-runs per pair.

The check is still needed: without it the exactness guarantee does not hold. But it is a correctness condition, not a common effect, and should be presented that way.

## Determinism

Greedy decoding on vLLM is **not** deterministic by default. Batching and prefix caching change the numerics, so identical prompts decode differently. In the MINJA pilot, 2 of 9 pre-attack episodes already differed between runs.

- **Exact claims:** `serve/vllm_qwen3_14b_bi.sh` (`VLLM_BATCH_INVARIANT=1`, no prefix caching) makes outputs bitwise stable, at about 2.4× lower throughput.
- **Non-deterministic decoding:** an episode whose inputs equal the counterfactual's is already a valid sample from the counterfactual distribution. Skipping it therefore keeps *distributional* equivalence. Fidelity should then be reported against a counterfactual-vs-counterfactual noise floor.

## What the auditor checks

- The pre- and post-recovery heads are signed, and the log stayed append-only between them.
- Every resolution record has an inclusion proof.
- Given the full log, the auditor also:
  - recomputes the taint closure and checks that every node in it, and every product of a re-run step, was resolved;
  - checks that the index digest equals the live set implied by the log, which rules out ghost vectors.

## Known limitations

- **Retrieval-displacement claims cannot be checked from the log alone.** The auditor would need the vectors to re-run those searches.
- **Exactness assumes deterministic steps.** See the Determinism section above; the distributional version still needs a formal statement.
- **One retrieval per step, before any product is emitted.** Multi-retrieval agents need sub-steps.
- **Blocked actions are logged, but a blocked action does not yet start a recovery.**
- **The in-memory index rebuilds its matrix after every write.** That needs replacing before running at scale.

## Running

```bash
source ~/miniconda3/bin/activate memsec
pip install -e .
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```
