from sqlalchemy import Column, Integer, String, Float, Boolean
from .database import Base

class Feedback(Base):
    # Creating a table name as feedback. 
    __tablename__ = "feedback"
    
    # Columns:
    # Make optional fields nullable=True to match your CRUD logic
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    is_sarcastic = Column(Boolean, nullable=True)  # Renamed from 'sarcasm'
    was_translated = Column(Boolean, nullable=True)  # Renamed from 'translate'
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)  # Changed to nullable=True since it can be None