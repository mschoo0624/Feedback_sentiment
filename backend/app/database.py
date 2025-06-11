from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

# SQLALCHEMY_DATABASE_URL = os.getenv(
#     "DATABASE_URL", "postgresql://postgres:mschoo0624@localhost:5432/feedback_db"
# )

# Use SQLite for local development
SQLALCHEMY_DATABASE_URL = "sqlite:///./DB/feedback.db"

# Create engine with SQLite-specific configuration (only create once!)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # SQLite specific
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()