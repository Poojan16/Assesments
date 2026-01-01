from models import *
from fastapi import APIRouter, BackgroundTasks,Depends,Request, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as httpException
from validation import *
from services import users
from database import *
from typing import Dict, Any, List, Optional, Set



router = APIRouter(
    prefix="/users",
    tags=["Users"],
)

@router.get("/")
async def getAll():
    try:
        allUsers = await users.getAll()
        return allUsers
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

@router.get("/id")
async def get_user(user_id: int):
    try:
        user = await users.get_user(user_id)
        return user
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

@router.post("/")
async def create_user(user: UserBase):
    # json parse data
    try:
        user_info = await users.create_user(user)
        return user_info
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    
@router.put("/reset")
async def reset(email: str = Form(...), password: str = Form(...)):
    try:
        user_info = await users.reset(email, password)
        return user_info
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

@router.get("/otp")
async def send_otp(email:str):
    try:
        otp = await users.send_otp(email)
        return otp
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    
@router.get("/resetPassword")
async def resetPassword(token:str):
    try:
        otp = await users.resetPassword(token)
        return otp
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    
# admin will pass minutes for rest password link expiry and set it to the function
@router.post("/expiry")
async def expiry_link(time: int):
    try:
        link = await users.expiry_link(time)
        return link
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

@router.get("/link")
async def link_check(email:str):
    try:
        link = await users.link_check(email)
        return link
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

@router.get("/decodelink")
async def decodelink(link):
    try:
        decode = await users.decodelink(link)
        return decode
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

@router.post("/login")
async def login(user_data: LoginUser, request: Request, db: SessionLocal = Depends(get_db)):
    try:
        user_info = await users.login(user_data, request, db)
        return user_info
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

@router.get("/logout")
async def logout(sessionId: str, db: SessionLocal = Depends(get_db)):
    try:
        user_info = await users.logout(sessionId, db)
        return user_info
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    
@router.get("/logins")
async def get_user_logins(email: str):
    try:
        user_info = await users.get_user_logins(email)
        return user_info
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))