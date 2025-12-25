from datetime import datetime, timedelta, timezone
from typing import Union, Any
import jwt
import os
from jose import JWTError
from fastapi import HTTPException, status

# Your JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

def create_expiring_link_token(email: str, expires_delta: timedelta) -> str:
    """
    Create a JWT token that includes the email in the payload
    """
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {
        "exp": expire,
        "sub": email,  # Store email as subject
        "type": "reset_password"  # Optional: Add token type for better validation
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_expiring_link_token(token: str) -> dict:
    """
    Decode the JWT token and return the payload containing email
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: No email found",
            )
        # token used to reset password
        
        # Optional: Check token type if you have different token types
        if token_type != "reset_password":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )
            
        return {
            "email": email,
            "type": token_type,
            "exp": payload.get("exp")
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    
    



# async def verify_reset_token(token: str):
#     """
#     Verify the reset token and extract the email
#     """
#     try:
#         decoded_data = decode_expiring_link_token(token)
        
#         return {
#             "status_code": 200,
#             "success": True,
#             "data": decoded_data,
#             "message": "Token verified successfully",
#         }
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Token verification failed: {str(e)}"
#         )
