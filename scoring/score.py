"""Score the artifacts of one or more runs with the objective rubric and push the scores to Langfuse.

    python -m scoring.score (--all | <run-id> [<run-id> ...]) [--live a,b] [--dry-run] [--no-embeddings]

Only LIVE stages are scored: the ones listed in output/<run-id>/run.json (written by pipeline.py;
the baseline stages are only copies). Scores are attached to the trace of each live stage. With several runs, O7 (novelty) is also computed per stage across the runs that share the same
input (the `idea` in run.json). Re-running is idempotent (score id = trace + name).
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client

from scoring import checks
from stages.characters import Characters
from stages.conflict import Conflict
from stages.dialog import Dialog
from stages.world import World
from llm import embed as embed_texts

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"
# CLI stage name -> (artifact file, Langfuse observation name); both come from the Stage classes.
STAGES = {
    "world": World,
    "characters": Characters,
    "conflict": Conflict,
    "dialog": Dialog,
}
STAGES = {k: (c.filename, c.name) for k, c in STAGES.items()}


def load(run_dir):
    """All four artifacts: from the run folder when the stage was live, else from baseline/."""
    out = {}
    for s, (f, _) in STAGES.items():
        path = run_dir / f if (run_dir / f).is_file() else ROOT / "baseline" / f
        out[s] = json.loads(path.read_text(encoding="utf-8"))
    return out


def live_stages_of(run_dir, override=None):
    """Live stages of a run: --live if given, else run.json written by pipeline.py."""
    if override is not None:
        return override
    meta = run_dir / "run.json"
    if not meta.is_file():
        raise SystemExit(f"{meta} not found (run made before run.json existed?): pass --live <stages>.")
    live = json.loads(meta.read_text(encoding="utf-8"))["live_stages"]
    return [s for s in STAGES if s in live]


def idea_of(run_dir):
    """Input idea of a run (from run.json); runs without one share the group ''."""
    meta = run_dir / "run.json"
    return json.loads(meta.read_text(encoding="utf-8")).get("idea", "") if meta.is_file() else ""


def schema_valid(stage, artifact):
    try:
        import jsonschema
    except ImportError:
        return True  # filter skipped when jsonschema is missing
    schema = json.loads((SCHEMAS / f"{stage}.schema.json").read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema).is_valid(artifact)


def score_run(a, stage, embed):
    if stage == "world":
        return checks.score_world(a["world"], embed)
    if stage == "characters":
        return checks.score_characters(a["characters"], embed)
    if stage == "conflict":
        return checks.score_conflict(a["conflict"], a["characters"], embed)
    return checks.score_dialog(a["dialog"], a["characters"], a["conflict"], embed)


def find_trace_id(lf, run_id, obs_name):
    """Trace id of the observation `obs_name` that carries metadata run_id."""
    cursor = None
    while True:
        res = lf.api.observations.get_many(
            name=obs_name, fields="core,metadata", expand_metadata="run_id", limit=100, cursor=cursor
        )
        for o in res.data:
            if (o.metadata or {}).get("run_id") == run_id:
                return o.trace_id
        cursor = res.meta.cursor
        if not cursor:
            return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_ids", nargs="*")
    p.add_argument("--all", action="store_true", help="Score every run folder in output/ (runs without run.json are skipped unless --live is given).")
    p.add_argument("--live", help="Comma-separated live stages; overrides run.json (for runs without one).")
    p.add_argument("--dry-run", action="store_true", help="Print scores, send nothing.")
    p.add_argument("--no-embeddings", action="store_true", help="Skip O6/O7 (no embedding model needed).")
    args = p.parse_args()
    override = None
    if args.live:
        override = [x.strip() for x in args.live.split(",")]
        bad = [x for x in override if x not in STAGES]
        if bad:
            p.error(f"unknown stage(s) in --live: {', '.join(bad)}")

    if args.all:
        run_ids = sorted(d.name for d in (ROOT / "output").iterdir() if d.is_dir() and (override or (d / "run.json").is_file()))
        skipped = sorted(d.name for d in (ROOT / "output").iterdir() if d.is_dir() and d.name not in run_ids)
        if skipped:
            print(f"skipping runs without run.json (use --live to include them): {', '.join(skipped)}")
    else:
        run_ids = args.run_ids
    if not run_ids:
        p.error("give at least one run id, or --all")

    lf = None if args.dry_run else get_client()
    embed = None if args.no_embeddings else embed_texts
    runs = {r: load(ROOT / "output" / r) for r in run_ids}
    per_run = {}  # (run_id, stage) -> scores

    for run_id, a in runs.items():
        live = live_stages_of(ROOT / "output" / run_id, override)
        print(f"[{run_id}] scoring live stages: {', '.join(live) or 'none'}")
        for stage in live:
            if not schema_valid(stage, a[stage]):
                print(f"[{run_id}] {stage}: schema-invalid, not scored")
                per_run[(run_id, stage)] = {"schema_valid": 0.0}
                continue
            per_run[(run_id, stage)] = {"schema_valid": 1.0, **score_run(a, stage, embed)}

    if len(runs) > 1 and embed:  # O7: across runs with the same input, per stage
        for stage in STAGES:
            groups = {}
            for r in runs:
                if (r, stage) in per_run and per_run[(r, stage)]["schema_valid"]:
                    groups.setdefault(idea_of(ROOT / "output" / r), []).append(r)
            for idea, rs in groups.items():
                if len(rs) < 2:
                    continue
                sim = checks.novelty([json.dumps(runs[r][stage], ensure_ascii=False) for r in rs], embed)
                print(f"O7 {stage} (idea {idea!r}): {len(rs)} runs, similarity {sim:.4g}")
                for r in rs:
                    per_run[(r, stage)]["O7_cross_run_similarity"] = sim

    for (run_id, stage), scores in per_run.items():
        trace_id = None if args.dry_run and not lf else find_trace_id(lf, run_id, STAGES[stage][1])
        print(f"\n[{run_id}] {stage} (trace {trace_id})")
        for name, value in scores.items():
            print(f"  {name} = {value:.4g}")
            if lf and trace_id:
                lf.create_score(
                    name=name, value=float(value), trace_id=trace_id,
                    data_type="BOOLEAN" if name in ("schema_valid",) or name.startswith("O1_in_range") else "NUMERIC",
                    score_id=f"{trace_id}-{name}",
                    metadata={"run_id": run_id, "stage": stage},
                )
    if lf:
        lf.flush()


if __name__ == "__main__":
    main()
