from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
load_dotenv()
from urllib.parse import quote

# Database connection details
DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{quote(os.getenv('DB_PASSWORD'))}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?charset=utf8mb4"
engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_size=60, max_overflow=70)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

# Base class for declarative models
def get_db():
    db = SessionLocal()
    try:
        # Attempt a simple query to verify connection
        db.execute(text("SELECT 1"))
        yield db
        print("Connection to MySQL established successfully.")
    except Exception as e:
        # Handle connection error
        print(f"Database connection error: {e}")
        raise
    finally:
        db.close()

def CreateTables():
    from models import Base
    Base.metadata.create_all(bind=engine)