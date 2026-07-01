import json
import pickle
from pathlib import Path
 
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
 
CATALOG_PATH = Path("data/raw/catalog_detail.json")
INDEX_DIR    = Path("data/index")
EMBED_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
EMBED_DIM    = 384
 
 
def entry_to_document(entry: dict) -> Document:
    """Convert a catalog entry to a LlamaIndex Document."""
    keys_str      = ", ".join(entry.get("keys", []))
    levels_str    = ", ".join(entry.get("job_levels", []))
    langs_str     = ", ".join(entry.get("languages", [])[:5])  # first 5
    remote_str    = "Yes" if entry.get("remote") == "yes" else "No"
    adaptive_str  = "Yes" if entry.get("adaptive") == "yes" else "No"
    duration_str  = entry.get("duration", "") or "Not specified"
 
    text = (
        f"Assessment: {entry['name']}\n"
        f"Test Type: {keys_str}\n"
        f"Job Levels: {levels_str or 'All levels'}\n"
        f"Duration: {duration_str}\n"
        f"Remote Testing: {remote_str}\n"
        f"Adaptive/IRT: {adaptive_str}\n"
        f"Languages: {langs_str or 'English'}\n"
        f"Description: {entry.get('description', '').strip()}"
    )
 
    return Document(
        text=text,
        metadata={
            "name":       entry["name"],
            "url":        entry["link"],
            "test_type":  keys_str,
            "job_levels": levels_str,
            "remote":     remote_str,
            "adaptive":   adaptive_str,
            "duration":   duration_str,
            "entity_id":  str(entry.get("entity_id", "")),
        },
        doc_id=str(entry.get("entity_id", entry["name"])),
    )
 
 
def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
 
    # ── Load catalog ──────────────────────────────────────────────────────────
    print("Loading catalog...")
    raw = json.load(open(CATALOG_PATH))
    entries = raw if isinstance(raw, list) else raw.get("entries", raw)
    print(f"  {len(entries)} entries loaded")
 
    # ── Build documents ───────────────────────────────────────────────────────
    docs = [entry_to_document(e) for e in entries]
    print(f"  {len(docs)} documents created")
 
    # ── Set up embedding model ────────────────────────────────────────────────
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.embed_model = embed_model
    Settings.llm = None          # no LLM needed for indexing
 
    # ── Build FAISS index ─────────────────────────────────────────────────────
    print("Building FAISS index...")
    faiss_index   = faiss.IndexFlatIP(EMBED_DIM)   # inner-product (cosine after norm)
    vector_store  = FaissVectorStore(faiss_index=faiss_index)
    storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)
 
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_ctx,
        show_progress=True,
    )
 
    # ── Persist ───────────────────────────────────────────────────────────────
    print("Saving index...")
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
 
    # Also save raw entries for lookup by entity_id / name
    with open(INDEX_DIR / "entries.pkl", "wb") as f:
        pickle.dump(entries, f)
 
    print(f"Index saved to {INDEX_DIR}")
    print("Done!")
 
 
if __name__ == "__main__":
    main()