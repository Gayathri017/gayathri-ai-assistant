"""
Step 4/5 — the chatbot, now with a dark terminal-style UI.

Ties four things together:
1. retrieve.py       -> finds the relevant chunk(s) of Gayathri's knowledge base
2. Gemini            -> writes an answer using ONLY those chunks
3. gr.ChatInterface   -> the chat mechanics
4. Custom theme + CSS -> the dark/terminal look, wrapped in gr.Blocks
                         so we can add a portfolio-style header above it

Run locally:
    python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import gradio as gr
import spaces
from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieve import retrieve
from ingest import build_index, INDEX_PATH

load_dotenv()

if not INDEX_PATH.exists():
    print("No index found — building one now (first run on this machine)...")
    build_index()

client = genai.Client()
MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are a chatbot that answers questions about Gayathri Rajeev —
her education, technical skills, projects, and work experience.

Rules:
- Only answer using the information given to you in CONTEXT below. It comes
  straight from Gayathri's own CV, transcripts, and records.
- Never guess, assume, or make up anything not present in CONTEXT.
- If CONTEXT does not contain enough information to answer the question,
  respond with exactly this and nothing else:
  "I don't have that information on hand — feel free to email Gayathri
  directly at gayathrirajeev726@gmail.com and she can help."
- Keep answers conversational and concise, like a knowledgeable friend
  answering on her behalf, not a resume readout.
"""


@spaces.GPU
def _zerogpu_warmup():
    """Never actually called. Its only job is to exist, decorated, so
    Hugging Face's ZeroGPU startup check is satisfied. Our real chatbot
    logic below runs entirely on CPU and never needs this."""
    pass


def answer(message, history):
    chunks = retrieve(message, k=4)
    context = "\n\n---\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{message}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


# ---------------------------------------------------------------------------
# Dark terminal theme
# ---------------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0a0e14",
    body_background_fill_dark="#0a0e14",
    body_text_color="#c9d1d9",
    body_text_color_dark="#c9d1d9",
    background_fill_primary="#0d1117",
    background_fill_primary_dark="#0d1117",
    background_fill_secondary="#0d1117",
    background_fill_secondary_dark="#0d1117",
    border_color_primary="#1f2937",
    border_color_primary_dark="#1f2937",
    block_background_fill="#0d1117",
    block_background_fill_dark="#0d1117",
    block_border_color="#1f2937",
    block_border_color_dark="#1f2937",
    block_title_text_color="#39ff14",
    block_title_text_color_dark="#39ff14",
    button_primary_background_fill="#39ff14",
    button_primary_background_fill_dark="#39ff14",
    button_primary_text_color="#0a0e14",
    button_primary_text_color_dark="#0a0e14",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
    input_border_color="#1f2937",
    input_border_color_dark="#1f2937",
)

CUSTOM_CSS = """
.gradio-container { background: #0a0e14 !important; }

#header-block {
    text-align: center;
    padding: 36px 16px 24px 16px;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 16px;
}
#header-block .prompt-line {
    color: #6b7280;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}
#header-block h1 {
    color: #39ff14;
    font-size: 2.2rem;
    letter-spacing: 0.08em;
    margin: 8px 0 4px 0;
    text-shadow: 0 0 12px rgba(57, 255, 20, 0.35);
}
#header-block .tagline {
    color: #c9d1d9;
    font-size: 1rem;
    margin-bottom: 18px;
}
#header-block .tagline::after {
    content: "_";
    color: #39ff14;
    animation: blink 1s step-start infinite;
}
@keyframes blink {
    50% { opacity: 0; }
}
#header-block .links a {
    display: inline-block;
    margin: 0 6px;
    padding: 6px 14px;
    border: 1px solid #39ff14;
    border-radius: 4px;
    color: #39ff14 !important;
    text-decoration: none !important;
    font-size: 0.9rem;
    transition: all 0.15s ease-in-out;
}
#header-block .links a:hover {
    background: #39ff14;
    color: #0a0e14 !important;
    box-shadow: 0 0 10px rgba(57, 255, 20, 0.6);
}
"""

HEADER_HTML = """
<div id="header-block">
  <div class="prompt-line">gayathri@ai-assistant:~$ whoami</div>
  <h1>GAYATHRI RAJEEV</h1>
  <div class="tagline">Computer Science &amp; Cybersecurity Engineering grad student</div>
  <div class="links">
    <a href="https://github.com/Gayathri017" target="_blank">[ GitHub ]</a>
    <a href="https://linkedin.com/in/gayathrirajeev017" target="_blank">[ LinkedIn ]</a>
    <a href="mailto:gayathrirajeev726@gmail.com">[ Email ]</a>
  </div>
</div>
"""

with gr.Blocks(title="Ask Gayathri") as demo:
    gr.HTML(HEADER_HTML)
    gr.ChatInterface(
        fn=answer,
        examples=[
            "What is she currently studying?",
            "What cybersecurity courses has she completed?",
            "What programming languages does she know?",
            "Tell me about her negotiation agent project.",
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=theme, css=CUSTOM_CSS)