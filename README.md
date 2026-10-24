# MemLoop

MemLoop recovers the long-term memory of an LLM agent after poisoned inputs have been found. Every step the agent takes is recorded in a tamper-evident log, together with what the step read and wrote. Recovery uses this record in two stages:

1. **Contain:** remove every record derived from the poisoned inputs from retrieval.
2. **Repair:** re-execute the affected steps, so memory, outputs and actions match a run in which those inputs never occurred.

An auditor can check the recovery against the log.

## Layout

| Path | Contents |
|---|---|
| `memloop/memloop/` | Python package: ledger, labels, provenance trace, memory store, recovery and attribution, agent adapters |
| `memloop/tests/` | Unit tests |
| `memloop/experiments/` | Experiment drivers: MINJA on RAP/WebShop, the MINJA QA agent, RAP on ALFWorld, the detector, and the deletion baselines |
| `memloop/docs/` | How `memloop` is connected to MINJA |
| `scripts/` | Shell drivers for the experiment runs and sweeps, and summary scripts. Diagnostic analyses are in `scripts/diag/`. |
| `serve/` | vLLM launch scripts for the following local models: Qwen3-8B; Qwen3-14B, in default and batch-invariant modes; Qwen3-32B-AWQ |
| `patches/` | Changes to MINJA and WebShop |
| `paper/` | Exporters that collect the result summaries, and the generators of the paper's figures and tables |

## Setup

The scripts assume the repository is at `~/Memloop` and conda is at `~/miniconda3`:

```bash
git clone https://github.com/zhangdiming/Memloop.git ~/Memloop
```

### Conda environments

| Environment | Python | Used for |
|---|---|---|
| `memsec` | 3.11 | `memloop` and vLLM serving: vLLM 0.30.0, torch 2.13.0+cu130 |
| `minja-rap` | 3.10 | The MINJA RAP agent: faiss-cpu, OpenJDK 21. Activate the environment, because it sets `JAVA_HOME`. |
| `webshop` | 3.8 | The WebShop server. It needs the following: <ul><li>OpenJDK 11</li><li>spaCy 3.4.4 with pydantic 1.10</li><li>werkzeug 2.2.3</li><li>en_core_web_lg 3.4.1</li></ul> |
| `alfworld` | 3.11 | ALFWorld 0.3.4 |

### Baselines

```bash
mkdir -p ~/Memloop/baselines && cd ~/Memloop/baselines
git clone https://github.com/dsh3n77/MINJA.git
git -C MINJA checkout a3ec8da0f7740b3fe629f6706e1c14296ee6861d
git -C MINJA apply ~/Memloop/patches/minja_rap.patch
git clone https://github.com/princeton-nlp/WebShop.git MINJA/rap/webshop
git -C MINJA/rap/webshop checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
git -C MINJA/rap/webshop apply ~/Memloop/patches/webshop.patch
git clone https://github.com/PanasonicConnect/rap.git rap-orig
git -C rap-orig checkout ebf0683d1eaeab5bca8ab1722d09b2db97adead7
```

What the patches change:
- `minja_rap.patch`:
  - lets MINJA use a local OpenAI-compatible server (`OPENAI_BASE_URL`), with a fixed temperature and seed (`MINJA_TEMPERATURE`, `MINJA_SEED`);
  - reads the WebShop address from `WEBSHOP_URL`;
  - adds `RETRIEVAL_HOOK`, an observer of every retrieval call that has no effect when unset.
- `webshop.patch`:
  - sets the server port from `WEBSHOP_PORT`;
  - switches WebShop to the full product files.

We took the WebShop product data from the Hugging Face mirror `YWZBrandon/webshop-data`, because the official download links failed.

## Running

Unit tests:

```bash
cd ~/Memloop/memloop
pip install -e .[test]
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

The result should be 48 tests passed and 1 skipped. The skipped test needs a running model server.

For the experiments, start a model server and the WebShop servers first:

```bash
serve/vllm_qwen3_14b_bi.sh              # batch-invariant Qwen3-14B on :8000
scripts/webshop_instance.sh 3000        # one WebShop server per port
```

Then run one of the following:
- **One pair:** `scripts/run_minja_recovery.sh <victim 0-8> <seed> <phase>` runs one victim/target pair through one phase. The phases are `attack`, `recover`, `counterfactual`, `baselines` and `eval`.
- **Main sweep:** `scripts/main_sweep.sh` runs all 9 pairs with 3 seeds each.
- **Other experiments:**
  - `scripts/listing_sweep.sh`: the malicious-listing attack;
  - `scripts/qa_sweep.sh`: the QA agent;
  - `scripts/scale_sweep.sh`: model scale;
  - `scripts/alfworld_*.sh`: ALFWorld.

Results are written to `results/` and logs to `logs/`.

## Data

The run outputs, and the JSON summaries that the figures and tables are built from, will be published on IEEE DataPort.

- Exporters:
  - `paper/export_data.py` and the scripts in `paper/tables/src/export/` produce the summaries from `results/`.
  - `paper/export_data.py` writes to `paper/data/`.
  - `paper/tables/src/export/export_qa_tables.py` takes its output directory as an argument; the other scripts there print JSON.
- Figures: with the summaries in `paper/figures/src/data/`, `python paper/figures/src/make_figures.py` writes `paper/figures/*.pdf`.
- Tables: with the summaries in `paper/tables/src/data/`, `python paper/tables/src/make_tables.py` writes `paper/tables/*.tex`.

## License

MIT, see `LICENSE`.
