from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers import auth, leave, manager
from app.database.connection import engine, Base
from app.config import settings
from app.utils.exceptions import AppException

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Leave Management System API",
    version="1.0.0",
    description="A production-ready API for managing employee leave requests",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(leave.router)
app.include_router(manager.router)


# Global exception handler for custom exceptions
@app.exception_handler(AppException)
async def app_exception_handler(request, exc: AppException):
    """
    Handle custom application exceptions with user-friendly messages.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "data": None
        }
    )


# Global exception handler for validation errors
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handle unexpected exceptions with a generic error message.
    """
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred. Please try again later or contact support.",
            "data": None
        }
    )


@app.get("/")
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "success": True,
        "message": "Welcome to Leave Management System API",
        "data": {
            "version": "1.0.0",
            "docs": "/api/docs",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "success": True,
        "message": "API is healthy",
        "data": {
            "status": "healthy",
            "service": "Leave Management System API"
        }
    }

