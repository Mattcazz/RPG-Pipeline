# Story pipeline

A pipeline of LLM stages that generates a story artifact step by step. Each stage calls a local model (via Ollama), is forced to answer with a JSON schema, and feeds its output to the next stage.

```
idea ─▶ world ─▶ characters ─▶ conflict ─▶ dialog
```

## Setup

1. **Python dependencies** (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. **Ollama** running on `localhost:11434` with the model the stages use:
   ```bash
   ollama pull qwen3:4b
   ollama pull nomic-embed-text           # only for the O6/O7 scoring metrics
   curl http://localhost:11434/api/tags   # should return JSON
   ```
3. **Environment variables**: copy `.env.example` to `.env` and fill it in (Langfuse keys, and `USERNAME`, which is sent to Langfuse as the user id).

## Running

Run everything from the project root:

```bash
python pipeline.py "A heist in a floating city"               # every stage calls the LLM
python pipeline.py "A heist in a floating city" --live dialog # only dialog calls the LLM
python pipeline.py --live characters,conflict                 # several live stages
python pipeline.py --live dialog --run-id my-test             # name the run (default: a random UUID)
python pipeline.py --live dialog -n 5                         # 5 runs in a row (run ids: random UUIDs)
python pipeline.py --live dialog -n 5 --run-id exp            # 5 runs: exp-1 ... exp-5
```

### Running specific models

By default, qwen3:4b is chosen for all stages

`--llm-model` selects a different default model for all stages

You may also specify a specific model for a specific stage, these options take priority over the default set with `--llm-model`

`--llm-model-world` selects the LLM model that will be used for the world stage

`--llm-model-characters` selects the LLM model that will be used for the world stage

`--llm-model-conflict` selects the LLM model that will be used for the world stage

`--llm-model-dialog` selects the LLM model that will be used for the world stage


### Live vs baseline

Every stage runs in one of two modes:

- **Live**: the stage calls the LLM and generates its artifact. This is the stage you are testing.
- **Baseline**: the stage does **not** call the LLM. It loads a fixed, pre-written artifact from the `baseline/` folder instead, so the stages you are *not* testing give the same input every time. That makes your results comparable with everyone else's, and runs faster.

`--live` chooses which stages are live (`world`, `characters`, `conflict`, `dialog`, a comma-separated list, or `all`, the default). All the others use their baseline. For example, `--live dialog` loads the world, characters and conflict from `baseline/` and makes a single LLM call for the dialog.

The fixed artifacts are in [`baseline/`](baseline/), one file per stage: `world.json`, `characters.json`, `conflict.json` and `dialog.json`. They match the schemas in `schemas/`. They are shared by the whole group, so don't change them for a single experiment; agree on it first. To make a better baseline, copy a good file from `output/<run-id>/`.

The `idea` argument is only used by the `world` stage, so it has an effect only when `world` is live.

## Where things go

| Path | What it is |
|---|---|
| `pipeline.py` | Entry point: parses the CLI, runs the stages in order |
| `llm.py` | `run_llm(...)`: the single function that talks to the model (Ollama through the OpenAI client, traced by Langfuse) |
| `stages/` | One class per stage, on top of the `Stage` base class in `stages/stage.py` |
| `schemas/` | JSON schema for each stage's output; it constrains the model's answer |
| `scoring/` | Objective rubric metrics (O1–O7) and the script that pushes them to Langfuse (see [Scoring](#scoring)) |
| `baseline/` | Fixed artifacts used by the stages that aren't live (see above) |
| `output/<run-id>/` | Artifacts of each run, one folder per run (git-ignored). It has all four files: generated ones for live stages and copies of the baseline for the rest, plus `run.json` with the run's live stages |

## Working on a stage

A stage is a class that sets a few attributes:

```python
class Dialog(Stage):
    name = "dialog-writer"
    model = "qwen3:4b"
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT          # template filled with the stage's inputs
    filename = "dialog.json"
    schema_path = SCHEMAS_DIR / "dialog.schema.json"
    params = {"reasoning_effort": "none", "temperature": 0.7}
```

- The user prompt is filled with `str.format`, so literal `{` and `}` must be written as `{{` and `}}`.
- Override `build_prompt(**inputs)` if a stage needs custom prompt building.
- Inputs per stage: `world(idea)`, `characters(world)`, `conflict(world, characters)`, `dialog(world, characters, conflict)`.
- The `world`, `characters` and `conflict` prompts are still empty (`# TODO`), so they can't run live yet.

## Langfuse

Every call is traced. Traces carry the stage name, live stages, user id and run id (a UUID unless you pass `--run-id`), and each trace is tagged `run:<run-id>` so you can filter all the traces of a run.

## Scoring

Scoring is **not** part of `pipeline.py`: run it by hand after a run, once its traces are in Langfuse.

```bash
python -m scoring.score <run-id>                      # score one run
python -m scoring.score --all                         # every run in output/ (one score set per run per live stage)
python -m scoring.score <run-id-1> <run-id-2>         # several runs; also computes O7 (novelty) across them
python -m scoring.score <run-id> --live dialog        # runs without run.json: say which stages were live
python -m scoring.score <run-id> --no-embeddings      # skip O6/O7, no embedding model needed
python -m scoring.score <run-id> --dry-run            # print the scores, send nothing
```

- Only the **live** stages are scored, as listed in `output/<run-id>/run.json` (written by `pipeline.py`). Runs made before `run.json` existed need `--live`.
- It reads `output/<run-id>/`, computes the metrics in [`scoring/checks.py`](scoring/checks.py) and attaches them as scores to the Langfuse trace of each stage (found through the `run_id` metadata). Look for them in the trace's **Scores** tab; they can take up to ~30 seconds to appear.
- Re-running is safe: the score id is `trace id + score name`, so scores are overwritten, not duplicated.
- Without `--no-embeddings`, `O6`/`O7` need `nomic-embed-text` in Ollama. If it is missing the script fails **before** sending anything, so nothing shows up in Langfuse.
- `O7` (novelty) compares the **output of a stage** across runs: it embeds each run's stage JSON and averages the pairwise cosine similarity (lower = more varied). It works for any number of runs (k runs = k·(k−1)/2 pairs) and the same value is written on every run of the group. The `idea` is **not** compared; it only decides the group: runs are compared only if they have the same `idea` (from `run.json`), so `--all` never mixes different inputs. Runs without `run.json` are skipped by `--all` unless you pass `--live`.
  - Caveat: the group is defined by `idea` only. If a stage depends on earlier *live* stages (e.g. `characters` after a live `world`), each run gets a different input, so `O7` also reflects that, not just the model's randomness. It is cleanest when the stage under test is the only live one (e.g. `--live dialog`), where all inputs come from the baseline. Don't mix runs with different live stages in one group.
- O1 word-count targets live in `LENGTH_TARGETS` in `scoring/checks.py` (inferred from the baseline; tune them freely).
