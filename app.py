"""
Step 4 — the actual chatbot.

Ties three things together:
1. retrieve.py  -> finds the relevant chunk(s) of Gayathri's knowledge base
2. Gemini       -> writes an answer using ONLY those chunks
3. Gradio       -> gives it a ChatGPT-style chat window

Run locally:
    python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import os

import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieve import retrieve

load_dotenv()  # reads GEMINI_API_KEY from your local .env file

client = genai.Client()  # picks up GEMINI_API_KEY from the environment automatically
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


demo = gr.ChatInterface(
    fn=answer,
    title="Ask about Gayathri",
    description="Ask me about Gayathri Rajeev's education, skills, and projects. "
    "If I don't know something, I'll point you to her email.",
    examples=[
        "What is she currently studying?",
        "What cybersecurity courses has she completed?",
        "What programming languages does she know?",
        "Tell me about her negotiation agent project.",
    ],
)

if __name__ == "__main__":
    demo.launch()
