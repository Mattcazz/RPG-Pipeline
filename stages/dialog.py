from stages.stage import SCHEMAS_DIR, Stage

SYSTEM_PROMPT = (
    "You are a dialog writer for an RPG. "
    "Return ONLY valid JSON (RFC 8259). "
    "Do NOT add any text, explanation or markdown before or after the JSON. "
    "Use double quotes only. "
    "Write the spoken lines in the voice of each character, "
    "never speak as the narrator or as yourself."
)

# NOTE: literal braces are doubled ({{ }}) because this string goes through str.format.
USER_PROMPT = """Write a dialog between the characters below. The dialog must happen inside the given world and explore the given conflict.

Rules:
- Use ONLY the characters provided. Do NOT invent new characters.
- "speaker" must be the exact "name" of one of the characters.
- Each character must speak according to their personality and voice (tone and speech_style).
- Respect the world's genre, tone and rules.
- The dialog must make the conflict and the relationships between the characters visible, but do NOT resolve it.
- "text" contains only the spoken words: no speaker name, no quotation marks, no stage directions.
- Write between 6 and 12 turns.
- Don't always return the same amount of characters. It's ok to have a short response if it makes sense.

Return a JSON object with this structure:
{{
  "dialogue": [
    {{ "speaker": "", "text": "" }}
  ]
}}

World:
{world}

Characters:
{characters}

Conflict:
{conflict}
"""


class Dialog(Stage):
    name = "dialog-writer"
    model = "qwen3:4b"
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT
    filename = "dialog.json"
    schema_path = SCHEMAS_DIR / "dialog.schema.json"
    params = {"temperature": 0.7}
