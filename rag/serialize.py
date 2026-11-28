"""Turns a dialog input (world, characters, conflict) into the text that gets embedded.

The same function must be used when storing an input and when searching with a new one,
otherwise the similarity scores mean nothing.
"""


def serialize_input(inp):
    world = inp["world"]
    conflict = inp["conflict"]

    characters = "; ".join(
        f"{c['personality']} (voice: {c['voice']['tone']})" for c in inp["characters"]
    )
    relationships = " ".join(r["description"] for r in conflict.get("relationships", []))

    return (
        f"genre: {world['genre']}. tone: {world['tone']}. "
        f"conflict genre: {', '.join(conflict.get('conflict_genre', []))}. "
        f"characters: {characters}. "
        f"conflict: {conflict['conflict']} "
        f"relationships: {relationships}"
    )
