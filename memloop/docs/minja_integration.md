# memloop over MINJA / RAP (WebShop)

## Mapping

| MINJA | memloop |
|---|---|
| one episode (session) | one `Step` |
| instruction shown to the agent | trigger `obs`. Origin is `user:<owner>`: attacker sessions (`inject_*`, `fixed_i_j` with i ≥ 1) belong to `user:attacker`, benign ones to their own user, and test sessions to `user:victim-<i>`. Trust is `USER` for all of them, because MINJA is query-only and the attacker is an ordinary user. Attribution therefore has to work from origins, not from trust levels. |
| record appended to `current_memory` (successful episode) | `mem` node; its text is the JSON of `{Instruction, Actions}` |
| examples retrieved at any action step of the episode | `step.retrieved`, the union over all action steps in that episode |
| final purchase (`click[Buy Now]`) | irreversible `act` |

## Extension beyond the paper: propagation

MINJA's test phase does not write memory. We add a variant in which successful test episodes are written back, which is how a deployed RAP agent behaves. A victim episode that was misled into buying the target then stores its own reasoning trace ("all toothbrushes are out of stock…"), and later users retrieve that trace. This produces the contamination chains that memloop has to unwind. We report both variants.

## Re-running an episode

`step_fn` calls `webshop_run_rap` with the memory restricted to entries that are live and were written before this step. Recovery experiments use temperature 0 and a fixed seed (`MINJA_TEMPERATURE=0`, `MINJA_SEED`), through MINJA's own client. vLLM is not batch-invariant, so identical prompts can still decode differently under concurrent load. Comparing the recovered run with the counterfactual measures this, and episodes that differ there are reported separately from those that differ in outcome.

## Retrieval-displacement check (exact)

MINJA retrieves at every action step. The query depends on the trajectory so far, so memloop cannot redo the search itself. Instead, `minja.RETRIEVAL_HOOK` is an inert observer added to `generate_examples`. For every retrieval call it logs the mode, the query and instruction embeddings, the score of every live entry, and the chosen top-k. `MinjaRAP.entry_score` mirrors MINJA's scoring formula:

```
cos(instr, I_e) + max_j cos(q, A_e[j]) + cos(I_e, A_e[j*])
```

The mirror is checked against MINJA's own scores on every call; the maximum error in the dry run was 4e-7.

During recovery, an episode is re-run only if one of these holds:

1. a removed entry was among its hits;
2. some added entry scores at least the logged k-th score at any of its calls;
3. the memory was smaller than k at some call.

Otherwise every prompt the episode saw is unchanged. With greedy decoding the episode would then replay identically, so skipping it is sound.

- Hook: `memloop.recover(..., displaced=h.displaced)`.
- For replays, `step_fn` returns `{"products", "retrieved"}`, giving the hits the agent actually used.
- Offline check: `experiments/minja_dryrun.py` uses real MINJA retrieval and MiniLM with a fake WebShop and a scripted LLM. Its recovered state equals the counterfactual, with and without pruning.

## What to measure

- **ISR / ASR:** before attack, after attack, and after recovery. Report ASR with and without propagation.
- **Recovery fidelity:** compare the recovered memory and per-episode outcomes with a counterfactual run that omits the attacker sessions, on the same seeds. Report exact-match rate, and how often non-determinism made identical requests produce different outputs.
- **Cost:** episodes re-run versus all episodes after τ; LLM calls; wall-clock time.
- **Blast radius:** contaminated entries, affected sessions, and compensations (purchases of the target that have to be undone).
