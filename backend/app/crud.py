from .models import Feedback
from sqlalchemy.orm import Session
from typing import Optional
from .schemas import FeedbackIn

def save_feedback(
    db: Session,
    data: FeedbackIn,
    sentiment: str,  # Required (no default) → Must come first!
    is_sarcastic: Optional[bool] = None,  # Optional (default=None)
    was_translated: Optional[bool] = None,  # Optional
    confidence: Optional[float] = None  # Optional
) -> Feedback:
    
    fb = Feedback(
        text=data.text,
        is_sarcastic=is_sarcastic,
        was_translated=was_translated,
        sentiment=sentiment,
        confidence=confidence
    )
    db.add(fb)
    try:
        db.commit()
        db.refresh(fb)  # Only needed if you need post-insert DB values (like `id`)
        return fb
    except Exception as e:
        db.rollback()
        raise ValueError(f"Database error: {e}") from e
