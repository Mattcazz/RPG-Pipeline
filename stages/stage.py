import json
from pathlib import Path

from llm import run_llm

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


class Stage:
    """Base class for a pipeline step: calls a model and saves its output as an artifact.

    Subclasses set the class attributes below and can override `build_prompt`
    when they need custom behavior.
    """

    name: str
    model: str
    system_prompt: str
    user_prompt: str
    filename: str  # artifact name, used in the run folder and in the baseline folder
    schema_path: Path  # JSON schema that constrains the model output
    params: dict = {}

    def __init__(self, out_dir, run_id, metadata=None, model=None):
        self.out_dir = Path(out_dir)  # folder of this run, e.g. output/<run_id>/
        self.run_id = run_id
        self.extra_metadata = metadata or {}
        self.model = model if model is not None else self.model

    def build_prompt(self, **inputs):
        """Turn the inputs into the user prompt. Override for custom prompts."""
        return self.user_prompt.format(**inputs)

    def _save(self, text):
        path = self.out_dir / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def run(self, **inputs):
        """Call the model, save the artifact in the run folder and return it."""
        output = run_llm(
            name=self.name,
            model=self.model,
            system_prompt=self.system_prompt,
            user_prompt=self.build_prompt(**inputs),
            metadata={**self.extra_metadata, "stage": self.name},
            run_id=self.run_id,
            schema=json.loads(Path(self.schema_path).read_text(encoding="utf-8")),
            **self.params,
        )
        self._save(output)
        return output

    def load_baseline(self, baseline_dir):
        """Use the fixed artifact from `baseline_dir` instead of calling the model.

        A copy is saved in the run folder so every run folder is complete.
        """
        path = Path(baseline_dir) / self.filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline for stage '{self.name}': {path}")
        text = path.read_text(encoding="utf-8")
        json.loads(text)  # fail early if the baseline is not valid JSON
        self._save(text)
        return text
