from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1beta"}
)

r = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents="hello git"
)

print("Embedding dim:", len(r.embeddings[0].values))
print("✅ SDK working correctly!")