from pydantic import BaseModel
from typing import Optional

class FeedbackIn(BaseModel):
    text: str # Input model that expects only text field

class FeedbackOut(FeedbackIn):
    is_sarcastic: Optional[bool] = None  # Make optional with default None
    was_translated: Optional[bool] = None  # Make optional with default None
    sentiment: str # Sentiment analysis result (e.g., "positive", "negative")
    confidence: Optional[float] = None  # Make optional with default None