import argparse
import json
import os
import uuid
from pathlib import Path

from stages.characters import Characters
from stages.conflict import Conflict
from stages.dialog import Dialog
from stages.world import World

ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "baseline"  # fixed artifacts for the stages that are not live

# Pipeline order; the key is the name used in --live.
STAGES = {
    "world": World,
    "characters": Characters,
    "conflict": Conflict,
    "dialog": Dialog,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the story pipeline.")
    parser.add_argument(
        "idea",
        nargs="?",
        default="A short fantasy adventure",
        help="Initial prompt; only used by the world stage.",
    )
    parser.add_argument(
        "--live",
        default="all",
        help=f"Comma-separated stages that call the LLM ({', '.join(STAGES)}) or 'all'. "
        "The rest are loaded from the baseline folder.",
    )
    parser.add_argument(
        "--run-id", default=None, help="Defaults to a random UUID."
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=1,
        help="Number of runs to execute in a row. With --run-id and n > 1, run ids are <run-id>-1 ... <run-id>-n.",
    )
    parser.add_argument(
            "--llm-model", type=str, default="qwen3:4b",
            help="Which LLM model to use as the default option for each of the stages",
    )
    parser.add_argument(
            "--llm-model-world", type=str, default=None,
            help="Which LLM model to use for the world stage, this overrides the default set by --llm-model",
    )
    parser.add_argument(
            "--llm-model-characters", type=str, default=None,
            help="Which LLM model to use for the characters stage, this overrides the default set by --llm-model",
    )
    parser.add_argument(
            "--llm-model-conflict", type=str, default=None,
            help="Which LLM model to use for the conflict stage, this overrides the default set by --llm-model",
    )
    parser.add_argument(
            "--llm-model-dialog", type=str, default=None,
            help="Which LLM model to use for the dialog stage, this overrides the default set by --llm-model",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("-n must be at least 1")

    live = list(STAGES) if args.live == "all" else [s.strip() for s in args.live.split(",")]
    unknown = [s for s in live if s not in STAGES]
    if unknown:
        parser.error(f"unknown stage(s) in --live: {', '.join(unknown)}")
    args.live = live
    return args


def run(args, run_id):
    out_dir = ROOT / "output" / run_id
    if out_dir.exists():
        raise SystemExit(f"Run folder already exists: {out_dir}")

    out_dir.mkdir(parents=True)
    # Tells the scorer which stages of this run were live (the rest are baseline copies).
    (out_dir / "run.json").write_text(
        json.dumps({"run_id": run_id, "live_stages": args.live, "idea": args.idea}, indent=2) + "\n",
        encoding="utf-8",
    )

    def model_for(args, name):
        # Get specific stage llm model if available
        if getattr(args, f"llm_model_{name}", None) is None:
            # Get specified default llm model if available, otherwise None (this has a default of qwen3:4b so this should never return None but)
            return getattr(args, f"llm_model", None)
        return getattr(args, f"llm_model_{name}")

    metadata = {"live_stages": ",".join(args.live), "idea": args.idea}
    stages = {name: cls(out_dir, run_id, metadata, model=model_for(args, name)) for name, cls in STAGES.items()}

    def get(name, **inputs):
        stage = stages[name]
        if name in args.live:
            print(f"[{name}] LIVE", flush=True)
            return stage.run(**inputs)
        print(f"[{name}] baseline", flush=True)
        return stage.load_baseline(BASELINE_DIR)

    world = get("world", idea=args.idea)
    characters = get("characters", world=world)
    conflict = get("conflict", world=world, characters=characters)
    _ = get("dialog", world=world, characters=characters, conflict=conflict)

    print(f"\nRun {run_id} saved in {out_dir} (user: {os.getenv('USERNAME')})\n")


def main():
    args = parse_args()
    for i in range(1, args.runs + 1):
        if args.run_id:
            run_id = f"{args.run_id}-{i}" if args.runs > 1 else args.run_id
        else:
            run_id = str(uuid.uuid4())
        if args.runs > 1:
            print(f"=== run {i}/{args.runs} ===", flush=True)
        run(args, run_id)


if __name__ == "__main__":
    main()
