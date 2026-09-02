import os
from dotenv import load_dotenv

load_dotenv()
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

print("Step 1: keys set, importing ChatGoogleGenerativeAI...")
from langchain_google_genai import ChatGoogleGenerativeAI

print("Step 2: constructing the model...")
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

print("Step 3: calling invoke() — this is the part we expect might hang...")
response = llm.invoke("Say hello in one short sentence.")

print("Step 4: got a response!")
print(response.content)
