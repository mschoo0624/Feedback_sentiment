from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import requests
from collections import Counter
from typing import List, Dict, Any # Added Dict, Any for type hints
import os
from pathlib import Path # Import Path for robust path handling
import logging
# Bypass Bot Detection. 
from playwright.async_api import async_playwright
from pydantic import HttpUrl

from app.database import SessionLocal, engine
from app import models
from app.ml.sentiment import analyze_sentiment
from app.schemas import FeedbackOut, FeedbackIn
from fastapi.responses import JSONResponse
from app.crud import save_feedback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure DB directory exists
os.makedirs('./DB', exist_ok=True)

logger.info("Creating database tables...")
models.Base.metadata.create_all(bind=engine)
logger.info("Tables created successfully!")

# FastAPI app instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

STOP_WORDS = set() # Renamed from STOP_WORD to STOP_WORDS for consistency with common naming
# Construct the path to the Stop_Words.txt file
stopwords_file_path = Path(__file__).parent / "resources" / "Stop_Words.txt"
try:
    with open(stopwords_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word: # Add word only if not empty
                STOP_WORDS.add(word)
    logger.info(f"Loaded {len(STOP_WORDS)} stopwords from {stopwords_file_path}")
except FileNotFoundError:
    logger.error(f"Debugging: Stopwords file not found at {stopwords_file_path}.")
    # For now, it will proceed with an empty set if not found.
    # In a production app, you might want to raise an exception or use a default list.
except Exception as e:
    logger.error(f"Error loading stopwords from {stopwords_file_path}: {e}")

# Root route
@app.get("/")
def read_root():
    return {"message": "FeedbackSentinel is running 🚀"}

# Sentiment analysis endpoint
@app.post("/analyze", response_model=FeedbackOut)
def analyze_feedback(request: FeedbackIn, db: Session = Depends(get_db)):
    try:
        logger.info(f"Analyzing text: {request.text[:50]}...")
        result = analyze_sentiment(text=request.text)
        
        feedback_obj = save_feedback(
            db=db,
            data=request,
            sentiment=result["sentiment"],
            is_sarcastic=result["is_sarcastic"],
            was_translated=result["translated"],
            confidence=result["confidence"]
        )
        
        # Convert integers back to booleans
        is_sarcastic_bool = bool(feedback_obj.is_sarcastic) if feedback_obj.is_sarcastic is not None else None
        was_translated_bool = bool(feedback_obj.was_translated) if feedback_obj.was_translated is not None else None
        
        return FeedbackOut(
            text=feedback_obj.text,
            is_sarcastic=is_sarcastic_bool,
            was_translated=was_translated_bool,
            sentiment=feedback_obj.sentiment,
            confidence=feedback_obj.confidence
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Playright for Amazon specific website. 
async def rendered(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto(url, wait_until="networkidle")
        
        # Wait for Amazon reviews to load specifically
        if "amazon." in url:
            try:
                await page.wait_for_selector("span[data-hook='review-body']", timeout=5000)
            except:
                logger.warning("Amazon review selector not found, continuing anyway")
        
        content = await page.content()
        await browser.close()
        return content

# Web Scraping + Live Sentiment Analysis Pipeline
@app.get("/scrape-and-analyze")
async def scrape_and_analyze(url: HttpUrl = Query(..., description="URL of the page to scrape for comments")):
    """
        Accepts a URL with public comments and returns sentiment analysis + keyword insights.
    """
    try:
    ##############################################################################################################################
        # Fetching the url webpage content. (It sends only the initial HTML content. for server.)
        # html_content = requests.get(url, timeout=10)
        # For debugging. 
        # print("DEBUGGING: URL - ", html_content.content)

        # Check if the request was successful
        # if response.status_code != 200:
        #     raise HTTPException(status_code=400, detail=f"Failed to fetch the webpage. Status code: {response.status_code}")
    ##############################################################################################################################
        
        # Using Playwright to bypass bot detection
        html_content = await rendered(str(url))
        logger.info(f"Successfully fetched content from {url}")
        
        # Parse the HTML content with BeautifulSoup
        scrape = BeautifulSoup(html_content, 'html.parser')
        logger.info("HTML parsing completed successfully")
        
        # Collect comments from <p> tags and other relevant elements
        # comments = []
        comments: List[str] = []
        
        # Try Amazon-specific selectors first
        if "amazon." in str(url):
            logger.info("Applying Amazon-specific scraping logic.")
            
            # Try multiple Amazon review selectors
            review_selectors = [
                "span[data-hook='review-body']",
                ".review-text",
                ".cr-original-review-text",
                "[data-hook='review-body'] span"
            ]
            
            for selector in review_selectors:
                review_elements = scrape.select(selector)
                if review_elements:
                    comments.extend([elem.get_text(strip=True) for elem in review_elements if len(elem.get_text(strip=True)) >= 20])
                    logger.info(f"Found {len(review_elements)} reviews using selector: {selector}")
                    break
        
        # If no Amazon-specific comments found, try generic selectors
        if not comments:
            logger.info("Trying generic comment selectors")
            generic_selectors = [
                "p",
                ".comment",
                ".review",
                "[class*='comment']",
                "[class*='review']"
            ]
            
            for selector in generic_selectors:
                elements = scrape.select(selector)
                potential_comments = [elem.get_text(strip=True) for elem in elements if len(elem.get_text(strip=True)) >= 20]
                if potential_comments:
                    comments.extend(potential_comments)
                    logger.info(f"Found {len(potential_comments)} comments using selector: {selector}")
                    break
        
        # Check if we found any comments
        if not comments:
            logger.error("No comments found on the page")
            raise HTTPException(
                status_code=404, 
                detail="No comments found on the page. The page might not have reviews/comments, or they might be loaded dynamically."
            )

        logger.info(f"Found {len(comments)} comments for analysis")
        results: List[Dict[str, Any]] = []
        
        # Analyze sentiment for each comment
        for i, comment in enumerate(comments):
            try:
                res = analyze_sentiment(comment)
                results.append({
                    "text": comment,
                    "is_sarcastic": res["is_sarcastic"],
                    "was_translated": res["translated"],
                    "sentiment": res["sentiment"],
                    "confidence": res["confidence"]
                })
                logger.info(f"Analyzed comment {i+1}/{len(comments)}")
            except Exception as e:
                logger.error(f"Failed to analyze comment {i+1}: {e}")
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="Failed to analyze any comments")
        
        # Collect words from 'Dislike' comments
        dislike_comments_text = [d['text'] for d in results if d['sentiment'] == 'Dislike']
        all_dislike_words = []
        
        for comment_text in dislike_comments_text:
            words = [
                word.lower()
                for word in comment_text.split()
                if word.isalpha() and len(word) > 2 and word.lower() not in STOP_WORDS 
            ]
            all_dislike_words.extend(words)
            
        # Calculate the most mentioned words in "Dislike"
        most_mentioned_word = Counter(all_dislike_words).most_common(1)[0][0] if all_dislike_words else None
         
        # Count 'Like' and 'Dislike' sentiments
        like_count = sum(1 for r in results if r["sentiment"] == "Like")
        dislike_count = sum(1 for r in results if r["sentiment"] == "Dislike")

        logger.info(f"Analysis complete: {like_count} likes, {dislike_count} dislikes")
        
        # Return the structured results
        return {
            "summary": {
                "total_comments": len(results),
                "Like": like_count,
                "Dislike": dislike_count,
                "TopDislikeWord": most_mentioned_word
            },
            "detailed": results
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Scraping and analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")