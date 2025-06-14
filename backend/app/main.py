from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
import os
from app.database import SessionLocal, engine
from app import models
from app.ml.sentiment import analyze_sentiment
from app.schemas import FeedbackOut, FeedbackIn
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

# Debug endpoint for sarcasm detection
@app.post("/debug-sarcasm")
def debug_sarcasm(text: str):
    try:
        from app.ml.sarcasm_detection import AdvancedSarcasmDetector
        detector = AdvancedSarcasmDetector()
        return detector.detect_sarcasm(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))