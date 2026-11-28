from stages.stage import SCHEMAS_DIR, Stage


SYSTEM_PROMPT = ""
USER_PROMPT = ""

# TODO: fill in prompts and parameters
class Conflict(Stage):
    name = "conflict"
    model = "qwen3:4b"
    system_prompt = SYSTEM_PROMPT 
    user_prompt = USER_PROMPT 
    filename = "conflict.json"
    schema_path = SCHEMAS_DIR / "conflict.schema.json"
