from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import requests
from collections import Counter
from typing import List
import os

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

# Web Scraping + Live Sentiment Analysis Pipeline
@app.get("/scrape-and-analyze")
def scrape_and_analyze(url: str):
    """
        Accepts a URL with public comments and returns sentiment analysis + keyword insights.
    """
    try:
        # Fetching the url webpage content. 
        response = requests.get(url, timeout=10)
        # For the debuigging. 
        print("DEBUGGING: URL - ", response.content)
        
        # Check if the request was successful
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch the webpage. Status code: {response.status_code}")

        # using a BeautifulSoup python library for parsing HTML and XML docs. 
        # HTML parsing libraries like html5lib, lxml, html.parser, etc.
        scrape = BeautifulSoup(response.text, 'html.parser')
        
        # collect comments from <p> tags. 
        # This will need to be refined based on the actual HTML structure of comment sections.
        comments = [p.text.strip() for p in scrape.find_all("p")  if len(p.text.strip()) >= 20]
        
        # If the comments are not found. 
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found on the page. Try a different URL or adjust the scraping logic.")

        results = [] # To store detailed analysis for each comment
        dislike_words = [] # To collect words from 'Dislike' comments
        
        # storing the each comments and perfomr the sentiment functions. 
        for comment in comments:
            res = analyze_sentiment(comment)
            results.append({
                "text": comment,
                "is_sarcastic": res["is_sarcastic"],
                "was_translated": res["translated"],
                "sentiment": res["sentiment"],
                "confidence": res["confidence"]
            })
            # If the sentiment is 'Dislike', add its words to the dislike_words list
            if res["sentiment"] == "Dislike":
                dislike_words.extend(comment.lower().split())
            
        # Calculate the most mentioned words in "Dislike"
        most_mentioned_word = Counter(dislike_words).most_common(1)[0][0] if dislike_words else None
         
        # Count 'Like' and 'Dislike' sentiments
        like_count = sum(1 for r in results if r["sentiment"] == "Like")
        dislike_count = sum(1 for r in results if r["sentiment"] == "Dislike")

        # Return the structured results
        return {
            "summary": {
                "Like": like_count,
                "Dislike": dislike_count,
                "TopDislikeWord": most_mentioned_word
            },
            "detailed": results
        }

    except requests.exceptions.RequestException as req_e:
        # Handle network-related errors
        logger.error(f"Network error during scraping: {req_e}")
        raise HTTPException(status_code=500, detail=f"Could not connect to the URL: {str(req_e)}")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Scraping and analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}") 