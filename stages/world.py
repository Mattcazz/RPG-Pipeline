from stages.stage import SCHEMAS_DIR, Stage

SYSTEM_PROMPT = """
    You are a worldbuilder writer for an RPG. "
    "Return ONLY valid JSON (RFC 8259). "
    "Do NOT add any text, explanation or markdown before or after the JSON. "
    "Use double quotes only. 
    """
USER_PROMPT = """
    Generate all the required elements for a credible RPG world based on the given idea.
    Rules:
        - The idea (the prompt given by the user, not to be confused with this variable (USER_PROMPT) has the priority. You can add more elements to the world to add credibility, to make it more interesting, etc., but the user's requirements needs to be met.
        - The world must be coherent with itself. Don't contradict something already defined.
        - Make it original and interesting, don't always use cliches and common tropes.
        - Always add useful and connected information. Try to make a cohesive world with interconected elements.


        Return a JSON object with this structure:

        Idea:
        {idea}

        {{
        "narrative": "",
        "genre": "",
        "tone": "",
        "rules": [""],
        "species": [""],
        "classes": [""],
        "places": [
            {{"name": "","desc": ""}}
        ]
        }}
    """

class World(Stage):
    name = "world"
    model = "qwen3:4b"
    system_prompt = SYSTEM_PROMPT 
    user_prompt = USER_PROMPT 
    filename = "world.json"
    schema_path = SCHEMAS_DIR / "world.schema.json"
