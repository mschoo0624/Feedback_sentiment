# Other Libraries 
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional  # For optional fields
# Other files. 
from app.database import SessionLocal, engine
from app import models
from app.ml.sentiment import analyze_sentiment
from app.schemas import FeedbackOut, FeedbackIn
from app.crud import save_feedback

# Ensure DB directory exists
import os
os.makedirs('./DB', exist_ok=True)

print("Creating tables with new schema...")
models.Base.metadata.create_all(bind=engine)
print("Tables recreated successfully with correct column names!")

# FastAPI app instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific frontend origins in production
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

# Root route (health check)
@app.get("/")
def read_root():
    return {"message": "FeedbackSentinel is running 🚀"}

# Main sentiment analysis endpoint
@app.post("/analyze", response_model=FeedbackOut)
def analyze_feedback(request: FeedbackIn, db: Session = Depends(get_db)):
    try:
        result = analyze_sentiment(
            text=request.text,
            db_session=db
        )
        print("Debugging: main.py it worked here(1)!!!")
        feedback_obj = save_feedback(
            db=db,
            data=request,
            sentiment=result["sentiment"],  # Now comes before optional args
            is_sarcastic=result["is_sarcastic"],  # Changed from get("sarcasm")
            was_translated=result["translated"],  # Changed from get("translate")
            confidence=result.get("confidence")
        )
        print("Debugging: main.py before returning")
        return FeedbackOut(
            text=feedback_obj.text,
            is_sarcastic=feedback_obj.is_sarcastic,
            was_translated=feedback_obj.was_translated,
            sentiment=feedback_obj.sentiment,
            confidence=feedback_obj.confidence  # Fixed typo: was "confidenc"
        )
        
    except Exception as e:
        print("Debugging: Error!!!")
        raise HTTPException(status_code=500, detail=str(e))