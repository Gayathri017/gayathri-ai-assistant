from dotenv import load_dotenv
import os

result = load_dotenv()
print("load_dotenv() found a .env file:", result)
print("Current working directory:", os.getcwd())
print("GEMINI_API_KEY:", repr(os.environ.get("GEMINI_API_KEY")))
print("GOOGLE_API_KEY:", repr(os.environ.get("GOOGLE_API_KEY")))
