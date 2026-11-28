"""Load the curated examples in rag/seed/ into MongoDB (collections `inputs` and `dialogs`).

Safe to run again: documents are replaced by id, never duplicated.

Run from the project root, with `docker compose up -d` already running:
    python -m rag.seed

Each folder in rag/seed/ has world.json and characters.json, plus one or more
conflict<N>.json / dialog<N>.json pairs (or a single conflict.json / dialog.json).
Every pair becomes one input and one example dialog.
"""
import json
from pathlib import Path

from llm import embed
from rag.db import dialogs, inputs
from rag.serialize import serialize_input

SEED_DIR = Path(__file__).parent / "seed"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pairs():
    """Yield (input_id, input, dialogue) for every conflict/dialog pair in rag/seed/."""
    for folder in sorted(p for p in SEED_DIR.iterdir() if p.is_dir()):
        world = load(folder / "world.json")
        characters = load(folder / "characters.json")["characters"]
        for conflict_path in sorted(folder.glob("conflict*.json")):
            suffix = conflict_path.stem.removeprefix("conflict")  # "", "1", "2", ...
            dialog_path = folder / f"dialog{suffix}.json"
            input_id = f"{folder.name}-{suffix}" if suffix else folder.name
            inp = {"world": world, "characters": characters, "conflict": load(conflict_path)}
            yield input_id, inp, load(dialog_path)["dialogue"]


def main():
    items = list(pairs())
    vectors = embed([serialize_input(inp) for _, inp, _ in items])  # one call for all of them

    for (input_id, inp, dialogue), vector in zip(items, vectors):
        inputs().replace_one(
            {"_id": input_id},
            {"_id": input_id, "split": "index", "input": inp, "embedding": vector},
            upsert=True,
        )
        dialog_id = f"{input_id}-dialog"
        dialogs().replace_one(
            {"_id": dialog_id},
            {"_id": dialog_id, "input_id": input_id, "role": "example", "dialogue": dialogue},
            upsert=True,
        )
        print(f"seeded {input_id}  ({len(dialogue)} turns, {len(vector)}-dim embedding)")

    print(f"done: {inputs().count_documents({})} inputs, {dialogs().count_documents({})} dialogs in the database")


if __name__ == "__main__":
    main()
