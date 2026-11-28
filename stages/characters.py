from stages.stage import SCHEMAS_DIR, Stage

SYSTEM_PROMPT = """
    You are a text based character creator for an RPG. "
    "Return ONLY valid JSON (RFC 8259). "
    "Do NOT add any text, explanation or markdown before or after the JSON. "
    "Use double quotes only. 
    """
USER_PROMPT = """
    Generate between 2 and 5 credible characters based on the given world.
    Rules:
        - The characters must be coherent with the world.
        - Overall, the characters need to be consistent. They can have conflicting and contradictory traits, but always as a resource to make them more interesting. This needs to be scarce and not overused.
        - Make it original and interesting, don't always use cliches and common tropes.
        - Always add useful and connected information.
        - The characters need to be part of any species existent in the world. Only override this rule if it makes the character more interesting (p.e mixed-races). This kind of overrides need to be uncommon and never overused.
        - The characters can have any kind of class/especialty, related to a profesion (wizard, ninja, pilot, etc.). This needs to be coherent with the setting of the world. It's not mandatory, don't force it if it doesn't add depth or make the character more interesting.
        - The characters need a defined personality (how are they as an individual, behaviouraly, psychologicaly, etc.) and some traits (interesting pieces of information about them, previous history, relationship with other characters of with elements of their world, etc.).
        - The characters need a general tone when talking. Even if this depends on the situation, there must be a tone coherent with their personality (cheerful, agressive, neutral, etc.)
        - The characters need a defined speech style, and they can have distinct speech traits (not mandatory).

    Return a JSON object with this structure:
    {{

    World:
    {world}

    "characters": [
        {{
        "name": "",
        "desc": "",
        "gender": "",
        "age": 0,
        "personality": "",
        "voice": {{"tone": "", "speech_style": ""}},
        "specie": "",
        "class": "",
        "traits": [""]
        }},
        {{...}}
    ]
    }}
    """
class Characters(Stage):
    name = "characters"
    model = "qwen3:4b"
    system_prompt = SYSTEM_PROMPT 
    user_prompt = USER_PROMPT 
    filename = "characters.json"
    schema_path = SCHEMAS_DIR / "characters.schema.json"
