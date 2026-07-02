"""
Build a LlamaIndex vector index over the SHL catalog.
Uses the default SimpleVectorStore (JSON-based, no FAISS dependency).

Run from repo root:
    python -m app.index.build_index
"""
import json
import pickle
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

CATALOG_PATH = Path("data/raw/catalog_detail.json")
INDEX_DIR    = Path("data/index")
EMBED_MODEL  = "all-MiniLM-L6-v2"


def entry_to_document(entry: dict) -> Document:
    keys_str   = ", ".join(entry.get("keys", []))
    levels_str = ", ".join(entry.get("job_levels", []))
    duration   = entry.get("duration", "") or "Not specified"

    text = (
        f"Assessment: {entry['name']}\n"
        f"Test Type: {keys_str}\n"
        f"Job Levels: {levels_str or 'All levels'}\n"
        f"Duration: {duration}\n"
        f"Remote Testing: {'Yes' if entry.get('remote') == 'yes' else 'No'}\n"
        f"Adaptive/IRT: {'Yes' if entry.get('adaptive') == 'yes' else 'No'}\n"
        f"Description: {(entry.get('description') or '').strip()}"
    )

    return Document(
        text=text,
        metadata={
            "name":      entry["name"],
            "url":       entry["link"],
            "test_type": keys_str,
            "entity_id": str(entry.get("entity_id", "")),
        },
        doc_id=str(entry.get("entity_id", entry["name"])),
    )


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading catalog...")
    raw     = json.load(open(CATALOG_PATH))
    entries = raw if isinstance(raw, list) else raw.get("entries", raw)
    print(f"  {len(entries)} entries loaded")

    docs = [entry_to_document(e) for e in entries]
    print(f"  {len(docs)} documents created")

    print(f"Loading embedding model: {EMBED_MODEL} ...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None

    print("Building index...")
    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    print("Saving index...")
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    # Save raw entries for fast lookup
    with open(INDEX_DIR / "entries.pkl", "wb") as f:
        pickle.dump(entries, f)

    print(f"Done! Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
