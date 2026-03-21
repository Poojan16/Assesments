from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.models.user import User, UserRoleEnum
from app.utils.security import verify_token
from app.utils.exceptions import ErrorResponse

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Args:
        token: JWT token from Authorization header
        db: Database session
        
    Returns:
        User object if token is valid
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = ErrorResponse.unauthorized(
        "Could not validate credentials. Please log in again."
    )
    
    try:
        payload = verify_token(token)
        if payload is None:
            raise credentials_exception
        
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if email is None or user_id is None:
            raise credentials_exception
            
    except Exception:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id, User.email == email).first()
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_manager(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure the current user is a Manager.
    
    Args:
        current_user: Currently authenticated user (from get_current_user)
        
    Returns:
        User object if user is a Manager
        
    Raises:
        HTTPException: If user is not a Manager
    """
    if current_user.role != UserRoleEnum.MANAGER:
        raise ErrorResponse.forbidden(
            "Access denied. Manager role required to perform this action."
        )
    return current_user

