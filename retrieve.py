"""
Step 3b — given a question, returns the most relevant chunks from
the knowledge base built by ingest.py.

Quick manual test:
    python retrieve.py what cybersecurity courses has she completed
"""

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("index.faiss")
CHUNKS_PATH = Path("chunks.json")
MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once and reused, not on every call — loading the model is the slow part
_model = None
_index = None
_chunks = None


def _load():
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX_PATH))
        _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def retrieve(query: str, k: int = 3):
    """Returns the top-k most relevant chunks as a list of
    {"source": filename, "text": chunk text, "score": similarity 0-1}."""
    _load()
    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = _index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx]
        results.append({"source": chunk["source"], "text": chunk["text"], "score": float(score)})
    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What Cybersecurity courses has Gayathri completed?"
    for r in retrieve(query):
        print(f"[{r['score']:.3f}] {r['source']}")
        print(r["text"][:200], "...\n")
