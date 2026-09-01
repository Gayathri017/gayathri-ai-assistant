"""
Step 3a — builds a searchable index out of everything in /knowledge.

Run this once, and again any time you add or edit a file in /knowledge:
    python ingest.py

What it does:
1. Reads every .md file in /knowledge
2. Splits each one into chunks along its "## " section headers
   (so retrieval can return just the relevant section, not a whole file)
3. Turns each chunk into a vector (embedding) using a small free
   open-source model that runs locally, no API calls or cost
4. Stores those vectors in a FAISS index for fast similarity search,
   plus the original chunk text in chunks.json
"""

import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

KNOWLEDGE_DIR = Path("knowledge")
INDEX_PATH = Path("index.faiss")
CHUNKS_PATH = Path("chunks.json")
MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, runs on CPU, plenty good at this scale


def load_chunks():
    chunks = []
    for filepath in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        # Split before every "## " heading, so each section becomes its own chunk
        sections = re.split(r"\n(?=## )", text)
        for section in sections:
            section = section.strip()
            if section:
                chunks.append({"source": filepath.name, "text": section})
    return chunks


def build_index():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {KNOWLEDGE_DIR}/")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Normalize so inner product = cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # brute-force search — fine at this corpus size
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved index to {INDEX_PATH} and chunk text to {CHUNKS_PATH}")


if __name__ == "__main__":
    build_index()
