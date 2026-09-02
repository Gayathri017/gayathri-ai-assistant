# LangChain version — a side-by-side comparison

This folder is a second implementation of the [same AI portfolio chatbot](../README.md), rebuilt with [LangChain](https://www.langchain.com/) instead of calling the embedding model, FAISS, and Gemini API directly. It exists purely as a learning exercise and a comparison — **the live, deployed chatbot uses the manual version**, not this one.

## Why keep both?

Because the honest answer to "should I use LangChain?" is "it depends on the project," and having a real working example of both sides of that trade-off is more useful than picking one and forgetting the alternative existed.

## What's different here

| | Manual version (deployed) | This version (LangChain) |
|---|---|---|
| Chunking | Hand-written regex splitting on `## ` headers | `MarkdownHeaderTextSplitter` — a built-in LangChain tool for exactly this |
| Chunk format | Plain dicts (`{"source", "text"}`) | `Document` objects (`page_content` + `metadata`), LangChain's standard shape |
| Vector store | Raw `faiss` + `sentence-transformers` | `langchain_community.vectorstores.FAISS` wrapping the same libraries |
| Similarity scoring | Cosine similarity — **higher is better** | L2 distance — **lower is better** (opposite direction) |
| LLM call | Direct `google-genai` SDK call, prompt built as an f-string | `ChatGoogleGenerativeAI` + `ChatPromptTemplate`, composed via LCEL (`retriever \| format_docs \| prompt \| llm \| StrOutputParser()`) |
| Dependencies | `sentence-transformers`, `faiss-cpu`, `google-genai` | All of the above, plus `langchain`, `langchain-community`, `langchain-huggingface`, `langchain-google-genai`, `langchain-text-splitters` |

## What building this actually taught me

- LangChain's abstractions genuinely pay for themselves once a project needs conversation memory, multi-step retrieval, or agent-style tool use. For a static 28-chunk knowledge base, they mostly just rename steps the manual version already did in fewer lines.
- More dependencies really did mean more ways to break, not hypothetically: this version hit two real bugs the manual version never had —
  1. `langchain-community`'s FAISS integration is being sunset in favor of standalone packages (deprecation warning, not yet broken, but a sign of an ecosystem still actively splitting apart).
  2. `ChatGoogleGenerativeAI` is documented to fall back from `GOOGLE_API_KEY` to `GEMINI_API_KEY` automatically — in practice that fallback didn't reliably trigger, and when it silently failed, the client fell through to searching for Google Cloud credentials instead, hanging indefinitely rather than erroring clearly. Fixed by setting `GOOGLE_API_KEY` explicitly at startup (see `app.py`).
- `check_env.py` and `test_llm.py` are the small diagnostic scripts that isolated that second bug — kept here as a record of the debugging process, not just the fix.

## My conclusion

Keep the manual version deployed. It's simpler, has fewer moving parts, and every one of those parts is something I wrote and fully understand. This version stays here as proof I evaluated the alternative deliberately rather than either avoiding LangChain out of unfamiliarity or reaching for it by default because it's the popular name.

## Running this version

```bash
python -m venv venv
venv\Scripts\activate      # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
python ingest.py           # builds faiss_index/ from ../knowledge
python app.py               # starts the chatbot locally
```

Needs its own `.env` with `GEMINI_API_KEY=your_key_here`, same as the main project.
