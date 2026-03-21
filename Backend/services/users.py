from models import *
from fastapi import BackgroundTasks,Depends,Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as httpException
from validation import UserUpdate, UserBase, LoginUser
from database import SessionLocal
import random
from emailLogic import *
import tracemalloc
from utility import *
from fastapi.encoders import jsonable_encoder
from user_agents import parse
from database import *
from password import *
from dotenv import load_dotenv, find_dotenv,set_key
from redis_client import save_reset_token,mark_token_used
from routers.audit import post_user_audit
import logging
import uuid
from sqlalchemy import delete

load_dotenv()

dotenv_path = find_dotenv()


async def getAll():
    try:
        db = SessionLocal()
        users = db.query(User).all()
        
        return {
            "status_code": 200,
            "success": True,
            "data": users,
            "message": "Users fetched successfully",
        }
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    

async def get_user_logins(email: str):
    try:
        db = SessionLocal()
        user = db.query(Login).filter(Login.email == email).first()
        return {
            "status_code": 200,
            "success": True,
            "data": user,
            "message": "User fetched successfully",
        }
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

async def get_user(user_id: int):
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.userId == user_id).first()
        return {
            "status_code": 200,
            "success": True,
            "data": user,
            "message": "User fetched successfully",
        }
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

async def create_user(user: UserBase):
    # json parse data
    try:
        db = SessionLocal()
        password = hash_password(user.password)
        user_info = User(
            userName=user.teacher.teacherName,
            userEmail=user.teacher.teacherEmail,
            password=password,
            role=user.teacher.role
        )
        db.add(user_info)
        db.commit()
        db.refresh(user_info)
        return {
            "status_code": 200,
            "success": True,
            "data": user_info,
            "message": "User created successfully",
        }
    except Exception as e:
        db.rollback()
        raise httpException(status_code=400, detail=str(e))
    
async def reset(email:str = Form(...),password:str = Form(...),token:str = Form(...)):
    if(token):
        try:
            db = SessionLocal()
            user_info = db.query(User).filter(User.userEmail == email).first()
            user_info.password = hash_password(password)            
            db.commit()
            db.refresh(user_info)
            return {
                "status_code": 200,
                "success": True,
                "data": user_info,
                "message": "Password reset successfully",
            }
        except Exception as e:
            raise httpException(status_code=400, detail=str(e))
    elif(token == ""):
        try:
            db = SessionLocal()
            teacher = (db.query(Teacher).filter(Teacher.teacherEmail == email).first())
            user_info = User(
                userName=teacher.teacherName,
                userEmail=teacher.teacherEmail,
                password=hash_password(password),
                role=teacher.role   
            )
            db.add(user_info)
            db.commit()
            db.refresh(user_info)
            return {
                "status_code": 200,
                "success": True,
                "data": user_info,
                "message": "Password reset successfully",
            }
        except Exception as e:
            raise httpException(status_code=400, detail=str(e))
        

async def send_otp(email:str):
    try:
        otp = random.randint(1000000, 9999999)
        subject = "OTP Verification"
        body = f"Your OTP is: {otp}. Please use this OTP to verify your account. OTP is valid for 5 minutes."
        try:
            send_email_sync(email, subject, body)
            return {
                "status_code": 200,
                "success": True,
                "data": otp,
                "message": "Email sending initiated in the background",
            }
        except Exception as e:
            raise httpException(status_code=400, detail=str(e))
        
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    
async def resetPassword(token:str):
    try:
        db = SessionLocal()
        decode = await decode_expiring_link_token(token)
        user_info = db.query(User).filter(User.userEmail == decode['email']).first()
        if(user_info == None):
            raise httpException(status_code=400, detail="User not found")
        print(decode)
        if(decode['type'] == "reset_password"):
            subject = "Reset Password"
            body = f"Click on this link: <a href='{os.getenv('FRONTEND_URL')}/set-password/{quote(token)}'>Reset Password</a>, once you click on link a otp will generate and sent to your email address"
            try:
                send_email_sync('pujansoni.jcasp@gmail.com', subject, body)
                db = SessionLocal()
                await post_user_audit(user_info.userId, "Reset Password Link Sent")
                return {
                    "status_code": 200,
                    "success": True,
                    "message": "Email sending initiated in the background",
                }
            except Exception as e:
                raise httpException(status_code=400, detail=str(e))
        else:
            raise httpException(status_code=400, detail="Invalid token")
        
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    
# admin will pass minutes for rest password link expiry and set it to the function
async def expiry_link(time: int):
    try:
        expiry_time = timedelta(minutes=time)
        os.environ["EXPIRY_TIME"] = str(time)  # Convert to string as env vars are strings
        return {
            "status_code": 200,
            "success": True,
            "data": expiry_time,
            "message": "Expiry time set successfully",
        }
        
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

async def link_check(email: str):
    """
    Generate a password reset link with token containing email
    """
    try:
        # Get expiry time from environment (in minutes)
        expiry_minutes = int(os.getenv("EXPIRY_TIME", "15"))
        expiry_time = timedelta(minutes=expiry_minutes)
        
        # Create token with email embedded
        token = await create_expiring_link_token(email, expiry_time)
        
        return {
            "status_code": 200,
            "success": True,
            "data": {
                "token": token,
                "email": email,  # Return email for verification purposes
                "expires_in_minutes": expiry_minutes
            },
            "message": "Reset link created successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

async def decodelink(link):
    try:
        decode = await decode_expiring_link_token(link)
        await mark_token_used(link)
        return {
            "status_code": 200,
            "success": True,
            "data": decode,
            "message": "Link decoded successfully",
        }
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

async def login(user_data: LoginUser, request: Request, db: SessionLocal = Depends(get_db)):
    if not user_data.email:
        return {"status_code": 400, "detail": "Email is required"}
    if not user_data.password:
        return {"status_code": 400, "detail": "Password is required"}

    user_agent_string = request.headers.get("User-Agent")
    # print(user_agent_string)
    if not user_agent_string:
        raise HTTPException(status_code=400, detail="User-Agent header missing")

    ua = parse(user_agent_string)
    current_device_type = ua.device.family if ua.device.family != "Other" else "Desktop"
    current_os = ua.os.family or "Unknown"
    current_browser = ua.browser.family or "Unknown"

    user_info = (
        db.query(User)
        .filter(User.userEmail == user_data.email)
        .first()
    )
    
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")

    if not check_password(user_data.password, user_info.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    existing_info = (
        db.query(SessionLog)
        .filter(SessionLog.userId == user_info.userId, SessionLog.isActive == True)
        .first()
    )

    client_ip = request.client.host
    
    mark = db.query(Role).filter(Role.roleId == user_info.role).first().mark
    

    if existing_info:
        existing_info.expiresAt = datetime.now()
        existing_info.isActive = False
        db.commit()
        new_login = SessionLog(
            userId=user_info.userId,
            sessionId=uuid.uuid4().hex,
            deviceInfo={
                "device": current_device_type,
                "os": current_os,
                "browser": current_browser,
                "ip": client_ip
            },
            loginTime=datetime.now(),
            lastActivity=datetime.now(),
            expiresAt=datetime.now()+timedelta(days=1),
            isActive=True
        )
        db.add(new_login)  
        db.commit()
        user_info.token = new_login.sessionId
        user_info.expiresAt = new_login.isActive
        await post_user_audit(user_info.userId, "user logged in", new_login.id)
        return {
            "statusCode": 200,
            "message": "All set! You've logged out from your other device and can continue here.",
            "data": user_info,
            "mark": mark
        }

    try:
        new_login = SessionLog(
            userId=user_info.userId,
            sessionId=uuid.uuid4().hex,
            deviceInfo={
                "device": current_device_type,
                "os": current_os,
                "browser": current_browser,
                "ip": client_ip
            },
            loginTime=datetime.now(),
            lastActivity=datetime.now(),
            expiresAt=datetime.now()+timedelta(days=1),
            isActive=True
        )
        db.add(new_login)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    
    user_info.token = new_login.sessionId
    user_info.expiresAt = new_login.isActive
    await post_user_audit(user_info.userId, "user logged in", new_login.id)

    return {
        "statusCode": 200,
        "message": "Login successful.",
        "data": user_info,
        "mark": mark
    }
    
async def logout(sessionId: str, db: SessionLocal = Depends(get_db)):
    try:
        db.query(SessionLog).filter(SessionLog.sessionId == sessionId).update({"isActive": False})
        db.commit()
        return {
            "status_code": 200,
            "success": True,
            "message": "Logout successful.",
        }
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))