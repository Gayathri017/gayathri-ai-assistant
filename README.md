# Gayathri Rajeev — Portfolio + AI Assistant

A personal portfolio site with an AI chatbot built into it — trained on my actual CV, transcripts, and project history, so it can answer questions about my background directly.

**Live site:** https://gayathri017.github.io/gayathri-ai-assistant/
**Chatbot (standalone):** https://gayathri77-gayathri-ai-assistant.hf.space

## What's here

- A static portfolio site (hero, about, skills, projects, competitions, contact) with a dark terminal-inspired theme
- A retrieval-augmented chatbot embedded directly in the page — it only answers from my own knowledge base, and points to my email for anything outside that
- Everything built with free tools: GitHub Pages for hosting the site, Hugging Face Spaces for hosting the chatbot, Google Gemini's free tier for the AI, and a small open-source model for retrieval

## How the chatbot works

1. My background is broken into topic files under `knowledge/` (education, skills, projects, competitions, etc.)
2. `ingest.py` splits those into chunks and turns each into a vector using a local embedding model (`sentence-transformers`), stored in a FAISS index
3. When a question comes in, `retrieve.py` finds the most relevant chunks
4. Those chunks get passed to Gemini as context, with instructions to answer only from what's given — and to point to my email if the answer isn't in there
5. `app.py` wraps all of this in a Gradio chat interface, styled to match the site, and embedded via iframe

## Project structure

```
.
├── index.html            # Portfolio site
├── style.css
├── script.js
├── assets/
│   └── photo.jpg
├── app.py                # Chatbot: retrieval + Gemini + Gradio UI
├── retrieve.py            # Vector search over the knowledge base
├── ingest.py              # Builds the FAISS index from /knowledge
├── requirements.txt
└── knowledge/             # My background, as source-of-truth text files
    ├── profile.md
    ├── education.md
    ├── ktu_btech_transcript.md
    ├── skills.md
    ├── projects.md
    ├── experience_leadership.md
    ├── certificates_languages.md
    └── competitions.md
```

## Running it locally

```bash
pip install -r requirements.txt
python ingest.py      # builds the index from /knowledge
python app.py          # starts the chatbot at http://127.0.0.1:7860
```

You'll need a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey), set in a local `.env` file:

```
GEMINI_API_KEY=your_key_here
```

To preview the portfolio site itself, just open `index.html` directly in a browser — no build step needed.

## Updating what the bot knows

Add or edit a `.md` file under `knowledge/`, using `## ` headers to break it into sections — that's what the retrieval step chunks on. Re-run `python ingest.py` to rebuild the index locally. On Hugging Face Spaces, this happens automatically on every deploy.

## License

MIT — see [LICENSE](LICENSE).

## Contact

- [GitHub](https://github.com/Gayathri017)
- [LinkedIn](https://linkedin.com/in/gayathrirajeev017)
- [Kaggle](https://www.kaggle.com/gayathrirajeev77)
- gayathrirajeev726@gmail.com
