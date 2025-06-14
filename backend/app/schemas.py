from pydantic import BaseModel
from typing import Optional

class FeedbackIn(BaseModel):
    text: str

class FeedbackOut(FeedbackIn):
    is_sarcastic: Optional[bool] = None
    was_translated: Optional[bool] = None
    sentiment: str 
    confidence: Optional[float] = None