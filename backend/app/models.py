from sqlalchemy import Column, Integer, String, Float, Boolean
from .database import Base

class Feedback(Base):
    # Creating a table name as feedback. 
    __tablename__ = "feedback"
    
    # Columns:
    # Make optional fields nullable=True to match your CRUD logic
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    is_sarcastic = Column(Integer, nullable=True)  # Changed to Integer
    was_translated = Column(Integer, nullable=True)  # Changed to Integer
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)  # Changed to nullable=True since it can be None