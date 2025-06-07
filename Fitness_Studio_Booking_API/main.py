"""
Fitness Studio Booking API
A comprehensive booking system for fitness classes with timezone management,
validation, and error handling.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import pytz
import logging
from typing import List, Optional
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./fitness_studio.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class FitnessClass(Base):
    __tablename__ = "fitness_classes"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    instructor = Column(String, nullable=False)
    datetime_utc = Column(DateTime, nullable=False)  # Store in UTC
    total_slots = Column(Integer, nullable=False)
    available_slots = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, nullable=False)
    client_name = Column(String, nullable=False)
    client_email = Column(String, nullable=False)
    booking_time = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class ClassResponse(BaseModel):
    id: int
    name: str
    instructor: str
    datetime_local: str
    timezone: str
    available_slots: int
    total_slots: int
    
    class Config:
        from_attributes = True

class BookingRequest(BaseModel):
    class_id: int
    client_name: str
    client_email: EmailStr
    
    @validator('client_name')
    def validate_client_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Client name must be at least 2 characters long')
        return v.strip()

class BookingResponse(BaseModel):
    id: int
    class_id: int
    client_name: str
    client_email: str
    booking_time: str
    class_name: str
    class_datetime: str
    instructor: str
    
    class Config:
        from_attributes = True

class ErrorResponse(BaseModel):
    error: str
    message: str

# FastAPI app
app = FastAPI(
    title="Fitness Studio Booking API",
    description="A booking system for fitness classes with timezone management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Timezone utilities
def get_timezone(tz_name: str = "Asia/Kolkata") -> pytz.timezone:
    """Get timezone object, defaults to IST"""
    try:
        return pytz.timezone(tz_name)
    except pytz.exceptions.UnknownTimeZoneError:
        logger.warning(f"Unknown timezone {tz_name}, falling back to IST")
        return pytz.timezone("Asia/Kolkata")

def convert_utc_to_timezone(utc_dt: datetime, target_tz: str) -> tuple:
    """Convert UTC datetime to target timezone"""
    tz = get_timezone(target_tz)
    utc_dt = utc_dt.replace(tzinfo=pytz.UTC)
    local_dt = utc_dt.astimezone(tz)
    return local_dt, tz.zone

# Initialize sample data
def init_sample_data(db: Session):
    """Initialize sample fitness classes with fresh data"""
    # Clear existing data
    db.query(FitnessClass).delete()
    db.commit()
    logger.info("Old fitness class records deleted")

    # Define sample classes in IST timezone
    ist = pytz.timezone("Asia/Kolkata")
    sample_classes = [
        {
            "name": "Morning Yoga",
            "instructor": "Priya Sharma",
            "datetime_ist": datetime(2025, 6, 9, 7, 0),  # 7:00 AM IST (future date)
            "total_slots": 15
        },
        {
            "name": "Evening Zumba",
            "instructor": "Rahul Kumar",
            "datetime_ist": datetime(2025, 6, 9, 18, 30),  # 6:30 PM IST
            "total_slots": 20
        },
        {
            "name": "HIIT Workout",
            "instructor": "Anjali Patel",
            "datetime_ist": datetime(2025, 6, 10, 6, 30),  # 6:30 AM IST
            "total_slots": 12
        },
        {
            "name": "Power Yoga",
            "instructor": "Vikram Singh",
            "datetime_ist": datetime(2025, 6, 10, 19, 0),  # 7:00 PM IST
            "total_slots": 18
        }
    ]

    for class_data in sample_classes:
        # Convert IST to UTC
        ist_dt = ist.localize(class_data["datetime_ist"])
        utc_dt = ist_dt.astimezone(pytz.UTC).replace(tzinfo=None)

        logger.info(f"Seeding class '{class_data['name']}' at {ist_dt} IST / {utc_dt} UTC")

        fitness_class = FitnessClass(
            name=class_data["name"],
            instructor=class_data["instructor"],
            datetime_utc=utc_dt,
            total_slots=class_data["total_slots"],
            available_slots=class_data["total_slots"]
        )
        db.add(fitness_class)

    db.commit()
    logger.info("Sample fitness classes seeded successfully")

# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"message": "Fitness Studio Booking API is running!", "version": "1.0.0"}

@app.get("/classes", response_model=List[ClassResponse], tags=["Classes"])
async def get_classes(
    timezone: str = Query("Asia/Kolkata", description="Timezone for displaying class times"),
    db: Session = Depends(get_db)
):
    """
    Get all upcoming fitness classes
    
    - **timezone**: Target timezone for displaying class times (default: Asia/Kolkata)
    - Returns classes with times converted to the specified timezone
    """
    try:
        logger.info(f"Getting classes for timezone: {timezone}")
        
        # Initialize sample data if needed
        init_sample_data(db)
        
        # Get all active classes (remove future time filter for debugging)
        classes = db.query(FitnessClass).filter(
            FitnessClass.is_active == True
        ).order_by(FitnessClass.datetime_utc).all()
        
        logger.info(f"Found {len(classes)} total classes in database")
        
        # Filter for future classes
        now = datetime.utcnow()
        future_classes = [cls for cls in classes if cls.datetime_utc > now]
        
        logger.info(f"Found {len(future_classes)} future classes")
        
        if not future_classes:
            logger.warning("No upcoming classes found")
            # Return empty list but log the classes that exist
            for cls in classes:
                logger.info(f"Class: {cls.name} at {cls.datetime_utc} UTC (past: {cls.datetime_utc <= now})")
            return []
        
        # Convert times to requested timezone
        response_classes = []
        for cls in future_classes:
            try:
                local_dt, tz_name = convert_utc_to_timezone(cls.datetime_utc, timezone)
                
                response_classes.append(ClassResponse(
                    id=cls.id,
                    name=cls.name,
                    instructor=cls.instructor,
                    datetime_local=local_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    timezone=tz_name,
                    available_slots=cls.available_slots,
                    total_slots=cls.total_slots
                ))
            except Exception as e:
                logger.error(f"Error processing class {cls.id}: {str(e)}")
                continue
        
        logger.info(f"Returning {len(response_classes)} classes for timezone {timezone}")
        return response_classes
        
    except Exception as e:
        logger.error(f"Error retrieving classes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/book", response_model=BookingResponse, tags=["Bookings"])
async def book_class(booking: BookingRequest, db: Session = Depends(get_db)):
    """
    Book a spot in a fitness class
    
    - **class_id**: ID of the class to book
    - **client_name**: Name of the client (minimum 2 characters)
    - **client_email**: Valid email address of the client
    """
    try:
        # Check if class exists and is active
        fitness_class = db.query(FitnessClass).filter(
            FitnessClass.id == booking.class_id,
            FitnessClass.is_active == True
        ).first()
        
        if not fitness_class:
            logger.warning(f"Class not found: {booking.class_id}")
            raise HTTPException(status_code=404, detail="Class not found")
        
        # Check if class is in the future
        if fitness_class.datetime_utc <= datetime.utcnow():
            logger.warning(f"Cannot book past class: {booking.class_id}")
            raise HTTPException(status_code=400, detail="Cannot book past or ongoing classes")
        
        # Check if slots are available
        if fitness_class.available_slots <= 0:
            logger.warning(f"No slots available for class: {booking.class_id}")
            raise HTTPException(status_code=400, detail="No available slots for this class")
        
        # Check for duplicate booking
        existing_booking = db.query(Booking).filter(
            Booking.class_id == booking.class_id,
            Booking.client_email == booking.client_email,
            Booking.is_active == True
        ).first()
        
        if existing_booking:
            logger.warning(f"Duplicate booking attempt: {booking.client_email} for class {booking.class_id}")
            raise HTTPException(status_code=400, detail="You have already booked this class")
        
        # Create booking
        new_booking = Booking(
            class_id=booking.class_id,
            client_name=booking.client_name,
            client_email=booking.client_email
        )
        
        # Reduce available slots
        fitness_class.available_slots -= 1
        
        db.add(new_booking)
        db.commit()
        db.refresh(new_booking)
        
        # Convert class time to IST for response
        local_dt, _ = convert_utc_to_timezone(fitness_class.datetime_utc, "Asia/Kolkata")
        
        response = BookingResponse(
            id=new_booking.id,
            class_id=new_booking.class_id,
            client_name=new_booking.client_name,
            client_email=new_booking.client_email,
            booking_time=new_booking.booking_time.strftime("%Y-%m-%d %H:%M:%S"),
            class_name=fitness_class.name,
            class_datetime=local_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
            instructor=fitness_class.instructor
        )
        
        logger.info(f"Booking created: {new_booking.id} for {booking.client_email}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating booking: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/bookings", response_model=List[BookingResponse], tags=["Bookings"])
async def get_bookings(
    email: EmailStr = Query(..., description="Client email address to fetch bookings for"),
    db: Session = Depends(get_db)
):
    """
    Get all bookings for a specific email address
    
    - **email**: Email address to search bookings for
    - Returns all active bookings for the specified email
    """
    try:
        # Get bookings with class details
        bookings_query = db.query(Booking, FitnessClass).join(
            FitnessClass, Booking.class_id == FitnessClass.id
        ).filter(
            Booking.client_email == email,
            Booking.is_active == True
        ).order_by(FitnessClass.datetime_utc)
        
        bookings_data = bookings_query.all()
        
        if not bookings_data:
            logger.info(f"No bookings found for email: {email}")
            return []
        
        # Build response
        response_bookings = []
        for booking, fitness_class in bookings_data:
            # Convert class time to IST
            local_dt, _ = convert_utc_to_timezone(fitness_class.datetime_utc, "Asia/Kolkata")
            
            response_bookings.append(BookingResponse(
                id=booking.id,
                class_id=booking.class_id,
                client_name=booking.client_name,
                client_email=booking.client_email,
                booking_time=booking.booking_time.strftime("%Y-%m-%d %H:%M:%S"),
                class_name=fitness_class.name,
                class_datetime=local_dt.strftime("%Y-%m-%d %H:%M:%S IST"),
                instructor=fitness_class.instructor
            ))
        
        logger.info(f"Retrieved {len(response_bookings)} bookings for {email}")
        return response_bookings
        
    except Exception as e:
        logger.error(f"Error retrieving bookings for {email}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/bookings/{booking_id}", tags=["Bookings"])
async def cancel_booking(booking_id: int, db: Session = Depends(get_db)):
    """Cancel a booking (bonus endpoint)"""
    try:
        booking = db.query(Booking).filter(
            Booking.id == booking_id,
            Booking.is_active == True
        ).first()
        
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        
        # Get the class and restore slot
        fitness_class = db.query(FitnessClass).filter(
            FitnessClass.id == booking.class_id
        ).first()
        
        if fitness_class:
            fitness_class.available_slots += 1
        
        booking.is_active = False
        db.commit()
        
        logger.info(f"Booking cancelled: {booking_id}")
        return {"message": "Booking cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error cancelling booking {booking_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)