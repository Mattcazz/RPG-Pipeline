"""Fetch the stored example dialogs whose inputs are most similar to a new input."""
from llm import embed
from rag.db import VECTOR_INDEX, inputs


def retrieve_examples(query_text, k=2):
    """query_text: the new input, serialized with rag.serialize.serialize_input.

    Returns up to k {"similarity", "input", "dialogue"}, most similar first.
    """
    query = embed([query_text])[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": query,
                "numCandidates": 50,
                "limit": k,
                "filter": {"split": "index"},
            }
        },
        {
            "$lookup": {
                "from": "dialogs",
                "localField": "_id",
                "foreignField": "input_id",
                "pipeline": [{"$match": {"role": "example"}}],
                "as": "dialogs",
            }
        },
        {"$match": {"dialogs": {"$ne": []}}},
        {
            "$project": {
                "_id": 0,
                "similarity": {"$meta": "vectorSearchScore"},
                "input": 1,
                "dialogue": {"$first": "$dialogs.dialogue"},
            }
        },
    ]
    return list(inputs().aggregate(pipeline))
