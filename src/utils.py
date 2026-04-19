import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def configure_gemini():
    """
    Dummy function to keep compatibility with evaluator.py.
    The new SDK handles auth via get_client(), but evaluator.py 
    still tries to import this function.
    """
    load_dotenv()

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")
    return genai.Client(api_key=api_key)

def get_gemini_model(model_name: str = "gemini-2.5-flash"):
    """Return model name string — upgraded default."""
    return model_name

def get_groq_client():
    from groq import Groq
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_content_groq(prompt: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    """Generate text using Groq API as a fallback."""
    try:
        client = get_groq_client()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_name,
        )
        return "⚡ [Groq Fallback] " + chat_completion.choices[0].message.content
    except Exception as e:
        print(f"  [Error] Groq API also failed: {e}")
        return "I'm sorry, but both Gemini and Groq APIs are currently unavailable."

def generate_content(prompt: str, model_name: str = "gemini-2.5-flash", retries: int = 1) -> str:
    """Generate text using Gemini via the new SDK, with automatic Groq Fallback."""
    client = get_client()
    clean_model_name = "gemini-2.5-flash"
    
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=clean_model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            err_msg = str(e)
            if "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                if attempt < retries:
                    wait_time = 2
                    print(f"  [API Warning] Gemini overloaded. Waiting {wait_time}s... (Attempt {attempt+1})")
                    time.sleep(wait_time)
                else:
                    print("  [API Fallback] Gemini overloaded. Switching to Groq API...")
                    return generate_content_groq(prompt)
            else:
                print("  [API Fallback] Gemini Exception. Switching to Groq API...")
                return generate_content_groq(prompt)

def get_embedding(text: str, retries: int = 3) -> list:
    client = get_client()
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return result.embeddings[0].values
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [Retry {attempt+1}] Waiting {wait}s: {e}")
                time.sleep(wait)
            else:
                raise e

def get_query_embedding(query: str) -> list:
    client = get_client()
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values