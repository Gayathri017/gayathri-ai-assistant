"""
Step 3b (LangChain version) — given a question, returns the most relevant
chunks from the index built by ingest.py.

Quick manual test:
    python retrieve.py what cybersecurity courses has she completed

IMPORTANT DIFFERENCE from the original retrieve.py:
Your custom version used cosine similarity, where a HIGHER score (closer
to 1.0) meant a better match. LangChain's FAISS wrapper returns L2
distance by default, where a LOWER score means a better match (0 would
be an exact match; bigger numbers mean further apart). Same underlying
FAISS library, same index — just a different number to report, and you
sort/interpret it the opposite way.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = Path("faiss_index")
MODEL_NAME = "all-MiniLM-L6-v2"

_vectorstore = None


def _load():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )
        # allow_dangerous_deserialization=True is required because loading a
        # FAISS index involves unpickling data. This is safe here because
        # it's an index *you* built yourself in the step before this, not
        # one downloaded from somewhere else.
        _vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )


def retrieve(query: str, k: int = 4):
    """Returns the top-k most relevant chunks as a list of
    {"source": filename, "text": chunk text, "distance": lower = better}."""
    _load()
    results_with_scores = _vectorstore.similarity_search_with_score(query, k=k)

    results = []
    for doc, distance in results_with_scores:
        results.append({
            "source": doc.metadata.get("source", "unknown"),
            "text": doc.page_content,
            "distance": float(distance),
        })
    return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What Cybersecurity courses has Gayathri completed?"
    for r in retrieve(query):
        print(f"[distance: {r['distance']:.3f}] {r['source']}")
        print(r["text"][:200], "...\n")
