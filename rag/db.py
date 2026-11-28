"""MongoDB access for the dialog RAG: connection and collection setup."""
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
MONGO_DB = os.getenv("MONGO_DB", "rpg_rag")

VECTOR_INDEX = "inputs_vector"  # Atlas Vector Search index on inputs.embedding
EMBEDDING_DIMS = 768  # nomic-embed-text

INPUTS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["_id", "split", "input", "embedding"],
        "properties": {
            "_id": {"bsonType": "string"},
            "split": {"enum": ["index", "test"]},
            "input": {
                "bsonType": "object",
                "required": ["world", "characters", "conflict"],
            },
            "embedding": {"bsonType": "array", "items": {"bsonType": "double"}},
        },
    }
}

DIALOGS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["_id", "input_id", "role", "dialogue"],
        "properties": {
            "_id": {"bsonType": "string"},
            "input_id": {"bsonType": "string"},
            "role": {"enum": ["example", "candidate"]},
            "dialogue": {
                "bsonType": "array",
                "minItems": 2,
                "items": {
                    "bsonType": "object",
                    "required": ["speaker", "text"],
                },
            },
        },
    }
}

COLLECTIONS = {"inputs": INPUTS_VALIDATOR, "dialogs": DIALOGS_VALIDATOR}
INDEXES = {
    "inputs": ["split"],
    "dialogs": ["input_id", "role"],
}


def ensure_schema(db):
    """Create the collections, validators and indexes. Safe to run any number of times."""
    existing = set(db.list_collection_names())
    for name, validator in COLLECTIONS.items():
        if name in existing:
            db.command("collMod", name, validator=validator)  # updates the validator in place
        else:
            db.create_collection(name, validator=validator)
        for field in INDEXES[name]:
            db[name].create_index(field)
    _ensure_vector_index(db)


def _ensure_vector_index(db):
    """Create the vector search index if it is missing, then wait until it can be queried."""
    if not any(i["name"] == VECTOR_INDEX for i in db.inputs.list_search_indexes()):
        db.inputs.create_search_index(
            SearchIndexModel(
                name=VECTOR_INDEX,
                type="vectorSearch",
                definition={
                    "fields": [
                        {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMS, "similarity": "cosine"},
                        {"type": "filter", "path": "split"},
                    ]
                },
            )
        )
    for _ in range(60):
        if any(i.get("queryable") for i in db.inputs.list_search_indexes(VECTOR_INDEX)):
            return
        time.sleep(1)
    raise RuntimeError(f"vector search index '{VECTOR_INDEX}' is not ready after 60 seconds")


@lru_cache
def get_db():
    """Connect to MongoDB (once per process) and make sure the collections exist."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB]
    ensure_schema(db)
    return db


def inputs():
    return get_db()["inputs"]


def dialogs():
    return get_db()["dialogs"]
