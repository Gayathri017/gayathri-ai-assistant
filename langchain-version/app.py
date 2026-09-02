"""
Step 4 (LangChain version) — the chatbot, built with an LCEL chain instead
of manually building the prompt string and calling Gemini directly.

New concepts here:
- ChatGoogleGenerativeAI: LangChain's wrapper around the Gemini API — same
  model, just called through LangChain's standard LLM interface.
- retriever = vectorstore.as_retriever(...): wraps the vector store so it
  behaves like any other LangChain "Runnable" step, pluggable into a chain.
- ChatPromptTemplate: a reusable template with {placeholders} filled in
  at call time, instead of an f-string built by hand.
- LCEL (LangChain Expression Language): the `|` pipe chains steps so the
  output of one becomes the input of the next — same idea as a Unix pipe.
  Reading the chain below left to right: take the question, run it through
  retrieval AND pass it through unchanged at the same time, feed both into
  the prompt template, send that to the LLM, then parse out plain text.
"""

import gradio as gr
import spaces
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ingest import build_index

load_dotenv()

# langchain-google-genai is documented to fall back from GOOGLE_API_KEY to
# GEMINI_API_KEY automatically, but a known bug means that fallback doesn't
# always trigger — when it doesn't, the library silently searches for Google
# Cloud credentials instead, which is slow and eventually fails. Setting
# GOOGLE_API_KEY explicitly here removes any need for that fallback at all.
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

INDEX_DIR = Path("faiss_index")
MODEL_NAME = "all-MiniLM-L6-v2"

if not INDEX_DIR.exists():
    print("No index found — building one now (first run on this machine)...")
    build_index()

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cpu"})
vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

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

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
])


def format_docs(docs):
    """Turns the list of retrieved Documents into one text block, same
    shape as the manual version's context string."""
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


# The chain: question in -> (retrieve+format context, pass question through)
# -> fill the prompt template -> call the LLM -> extract plain text out.
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@spaces.GPU
def _zerogpu_warmup():
    """Never actually called. Exists only so Hugging Face's ZeroGPU
    startup check is satisfied — the real chain below runs on CPU."""
    pass


def answer(message, history):
    return chain.invoke(message)


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
#header-block { text-align: center; padding: 36px 16px 24px 16px; border-bottom: 1px solid #1f2937; margin-bottom: 16px; }
#header-block .prompt-line { color: #6b7280; font-size: 0.85rem; letter-spacing: 0.05em; }
#header-block h1 { color: #39ff14; font-size: 2.2rem; letter-spacing: 0.08em; margin: 8px 0 4px 0; text-shadow: 0 0 12px rgba(57, 255, 20, 0.35); }
#header-block .tagline { color: #c9d1d9; font-size: 1rem; margin-bottom: 18px; }
#header-block .tagline::after { content: "_"; color: #39ff14; animation: blink 1s step-start infinite; }
@keyframes blink { 50% { opacity: 0; } }
#header-block .links a { display: inline-block; margin: 0 6px; padding: 6px 14px; border: 1px solid #39ff14; border-radius: 4px; color: #39ff14 !important; text-decoration: none !important; font-size: 0.9rem; transition: all 0.15s ease-in-out; }
#header-block .links a:hover { background: #39ff14; color: #0a0e14 !important; box-shadow: 0 0 10px rgba(57, 255, 20, 0.6); }
"""

HEADER_HTML = """
<div id="header-block">
  <div class="prompt-line">gayathri@ai-assistant:~$ whoami</div>
  <h1>GAYATHRI RAJEEV</h1>
  <div class="tagline">Computer Science &amp; Knowledge Engineering grad student</div>
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