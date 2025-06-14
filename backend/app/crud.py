from .models import Feedback
from sqlalchemy.orm import Session
from typing import Optional
from .schemas import FeedbackIn
import logging

logger = logging.getLogger(__name__)

def save_feedback(
    db: Session,
    data: FeedbackIn,
    sentiment: str,
    is_sarcastic: bool,
    was_translated: Optional[bool] = None,
    confidence: Optional[float] = None
) -> Feedback:
    
    # Convert booleans to integers
    is_sarcastic_int = 1 if is_sarcastic else 0
    was_translated_int = 1 if was_translated else 0 if was_translated is not None else None
    
    fb = Feedback(
        text=data.text,
        is_sarcastic=is_sarcastic_int,
        was_translated=was_translated_int,
        sentiment=sentiment,
        confidence=confidence
    )
    
    db.add(fb)
    try:
        db.commit()
        db.refresh(fb)
        return fb
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise