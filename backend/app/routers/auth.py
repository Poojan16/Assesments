from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import timedelta
from app.database.connection import get_db
from app.models.user import User, UserRoleEnum
from app.schemas.user import UserCreate, UserLogin, Token, UserResponse
from app.utils.security import verify_password, get_password_hash, create_access_token
from app.utils.exceptions import ErrorResponse
from app.config import settings

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user account.
    
    This endpoint creates a new user account with the provided information.
    It validates the input data, checks for duplicate email/employee_id,
    hashes the password, and returns a JWT token upon successful registration.
    
    Args:
        user_data: User registration data (email, password, name, etc.)
        db: Database session dependency
        
    Returns:
        Token object with access_token and user information
        
    Raises:
        HTTPException: If email or employee_id already exists, or validation fails
    """
    try:
        # Check if email already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise ErrorResponse.conflict(
                "An account with this email address already exists. Please use a different email or try logging in."
            )
        
        # Check if employee_id already exists
        existing_employee = db.query(User).filter(User.employee_id == user_data.employee_id).first()
        if existing_employee:
            raise ErrorResponse.conflict(
                f"An account with employee ID '{user_data.employee_id}' already exists. Please contact your administrator if you believe this is an error."
            )
        
        # Hash the password
        hashed_password = get_password_hash(user_data.password)
        
        # Create new user (default role is Employee)
        new_user = User(
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            employee_id=user_data.employee_id,
            department=user_data.department,
            role=getattr(user_data, 'role', UserRoleEnum.EMPLOYEE),  # Default to Employee if not provided
            hashed_password=hashed_password
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create access token
        access_token = create_access_token(
            data={"sub": new_user.email, "user_id": new_user.id},
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse.model_validate(new_user)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like our custom ErrorResponse)
        raise
    except IntegrityError as e:
        db.rollback()
        # Handle database integrity errors
        error_msg = str(e.orig) if hasattr(e, 'orig') else "Database integrity error"
        if "email" in error_msg.lower():
            raise ErrorResponse.conflict(
                "An account with this email address already exists. Please use a different email or try logging in."
            )
        elif "employee_id" in error_msg.lower():
            raise ErrorResponse.conflict(
                f"An account with employee ID '{user_data.employee_id}' already exists. Please contact your administrator if you believe this is an error."
            )
        else:
            raise ErrorResponse.bad_request(
                "Unable to create account. Please check your information and try again."
            )
    except Exception as e:
        db.rollback()
        # Log the error in production (you would use a proper logging library)
        print(f"Unexpected error during signup: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while creating your account. Please try again later or contact support."
        )


@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate a user and return a JWT token.
    
    This endpoint verifies the user's email and password, and returns
    a JWT token if the credentials are valid.
    
    Args:
        credentials: User login credentials (email and password)
        db: Database session dependency
        
    Returns:
        Token object with access_token and user information
        
    Raises:
        HTTPException: If credentials are invalid or user not found
    """
    try:
        # Find user by email
        user = db.query(User).filter(User.email == credentials.email).first()
        
        if not user:
            raise ErrorResponse.unauthorized(
                "Invalid email or password. Please check your credentials and try again."
            )
        
        # Verify password
        if not verify_password(credentials.password, user.hashed_password):
            raise ErrorResponse.unauthorized(
                "Invalid email or password. Please check your credentials and try again."
            )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id},
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse.model_validate(user)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error in production
        print(f"Unexpected error during login: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred during login. Please try again later or contact support."
        )

