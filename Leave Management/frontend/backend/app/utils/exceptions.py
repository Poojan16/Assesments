from fastapi import HTTPException, status
from typing import Optional


class AppException(HTTPException):
    """Custom exception class for application errors."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[dict] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class ErrorResponse:
    """Standard error response format."""
    
    @staticmethod
    def not_found(message: str = "Resource not found"):
        return AppException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message
        )
    
    @staticmethod
    def bad_request(message: str = "Bad request"):
        return AppException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    @staticmethod
    def unauthorized(message: str = "Unauthorized access"):
        return AppException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message
        )
    
    @staticmethod
    def forbidden(message: str = "Forbidden"):
        return AppException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message
        )
    
    @staticmethod
    def conflict(message: str = "Resource already exists"):
        return AppException(
            status_code=status.HTTP_409_CONFLICT,
            detail=message
        )
    
    @staticmethod
    def internal_error(message: str = "Internal server error"):
        return AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message
        )

