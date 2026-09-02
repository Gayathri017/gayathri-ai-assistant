"""
Step 3a (LangChain version) — builds a searchable index out of everything in /knowledge.

Run this once, and again any time you add or edit a file in /knowledge:
    python ingest.py

What's different from the original ingest.py:
- Chunks become LangChain `Document` objects (page_content + metadata), the
  standard shape every LangChain component expects.
- Splitting on "## " headers is done by MarkdownHeaderTextSplitter instead
  of a hand-written regex — same idea, LangChain just has a built-in tool
  for exactly this.
- The FAISS index is built and saved through LangChain's own FAISS wrapper,
  which bundles the vectors with the Documents so retrieve.py doesn't need
  a separate chunks.json file.
"""

from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_DIR = Path("knowledge")
INDEX_DIR = Path("faiss_index")
MODEL_NAME = "all-MiniLM-L6-v2"  # same small, free, CPU-friendly model as before


def load_documents():
    # Split on level-2 markdown headers ("## "); each resulting chunk keeps
    # the header text in its metadata under the key "section".
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])

    documents = []
    for filepath in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)  # returns a list of Document objects
        for chunk in chunks:
            chunk.metadata["source"] = filepath.name  # tag each chunk with its file
            documents.append(chunk)
    return documents


def build_index():
    documents = load_documents()
    print(f"Loaded {len(documents)} chunks from {KNOWLEDGE_DIR}/")

    # device="cpu" explicitly — without this, the same ZeroGPU CUDA error
    # we already fixed once in the original version would show up again here.
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Saved index to {INDEX_DIR}/")


if __name__ == "__main__":
    build_index()
