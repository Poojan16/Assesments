from datetime import datetime, timedelta, timezone
from typing import Union, Any
import jwt
import os
from jose import JWTError
from fastapi import HTTPException, status
from redis_client import mark_token_used, get_reset_token, save_reset_token
import logging

# Your JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

async def create_expiring_link_token(email: str, expires_delta: timedelta) -> str:
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
    try:
        await save_reset_token(encoded_jwt, email, expire)
    except Exception as e:
        logging.error(f"Error saving reset token: {e}")
    return encoded_jwt

async def decode_expiring_link_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        email: str = payload.get("sub")
        token_type: str = payload.get("type")

        if not email or not token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        if token_type != "reset_password":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        # 🔍 Check Redis using jti (NOT token)
        token_data = await get_reset_token(token)

        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired link",
            )

        if token_data.get("is_used") == "true":
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="This link has already been used",
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
