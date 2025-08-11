# main.py

from fastapi import FastAPI, Request, HTTPException
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

app = FastAPI()

# Configure the Gemini API key from environment variables
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        print("Warning: GOOGLE_API_KEY environment variable not set. Bias analysis will be disabled.")
except Exception as e:
    print(f"Error configuring Google API: {e}")
    GOOGLE_API_KEY = None

def get_article_content(url: str) -> str | None:
    """
    Fetches and extracts the main content of an article from a given URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        selectors = [
            {'tag': 'div', 'class_': 'story-element'},
            {'tag': 'div', 'class_': 'story-body'},
            {'tag': 'article', 'class_': None}
        ]
        
        article_content = []
        for selector in selectors:
            elements = soup.find_all(selector['tag'], class_=selector.get('class_'))
            if elements:
                for element in elements:
                    paragraphs = element.find_all('p')
                    for p in paragraphs:
                        article_content.append(p.get_text())
                if article_content:
                    break
        
        return "\n".join(article_content) if article_content else None
    except requests.exceptions.RequestException as e:
        print(f"Request to {url} failed: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {url}: {e}")
        return None

def analyze_political_bias(article_text: str) -> str | None:
    """
    Analyzes the political bias of an article using the Gemini API.
    """
    if not GOOGLE_API_KEY:
        print("Google API key not configured. Skipping bias analysis.")
        return None
        
    system_prompt = """
    You are an expert AI news analyst. Your task is to evaluate the political bias of the provided article. Classify it as "Left-leaning," "Neutral," or "Right-leaning".
    Analyze the article's tone, framing, and choice of sources. Return only the classification label.
    """
    
    try:
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash-latest',
            system_instruction=system_prompt
        )
        response = model.generate_content(article_text)
        return response.text.strip()
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None

@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze_bias")
async def analyze_bias(request: Request):
    """
    API endpoint to analyze the political bias of an article from a URL.
    """
    try:
        data = await request.json()
        url = data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL not provided.")
            
        article_text = get_article_content(url)
        if not article_text:
            raise HTTPException(status_code=404, detail="Failed to extract article content.")
            
        bias_result = analyze_political_bias(article_text)
        if bias_result is None:
            raise HTTPException(status_code=500, detail="Failed to analyze political bias.")
        
        return {"political_bias": bias_result}
    except Exception as e:
        print(f"An unexpected error occurred in /analyze_bias: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")