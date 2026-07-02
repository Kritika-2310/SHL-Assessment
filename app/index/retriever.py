import pickle
from functools import lru_cache
from pathlib import Path

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

INDEX_DIR   = Path("data/index")
EMBED_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_index():
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = None
    storage_ctx = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context=storage_ctx)


@lru_cache(maxsize=1)
def _load_entries() -> list[dict]:
    with open(INDEX_DIR / "entries.pkl", "rb") as f:
        return pickle.load(f)


def search_catalog(query: str, top_k: int = 10) -> list[dict]:
    index     = _load_index()
    entries   = _load_entries()
    entry_map = {e["name"]: e for e in entries}
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes     = retriever.retrieve(query)
    results, seen = [], set()
    for node in nodes:
        name = node.metadata.get("name", "")
        if name and name not in seen:
            seen.add(name)
            entry = entry_map.get(name)
            if entry:
                results.append(entry)
    return results
