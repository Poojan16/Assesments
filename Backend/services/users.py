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

from routers.audit import post_user_audit

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
    elif(token == ""):
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
        decode = decode_expiring_link_token(token)
        print(decode)
        db = SessionLocal()
        if(decode['type'] == "reset_password"):
            subject = "Reset Password"
            body = f"Click on this link: <a href='{os.getenv('FRONTEND_URL')}/set-password/{quote(token)}'>Reset Password</a>, once you click on link a otp will generate and sent to your email address"
            try:
                send_email_sync('pujansoni.jcasp@gmail.com', subject, body)
                db = SessionLocal()
                user_info = (db.query(User).filter(User.userEmail == decode['email']).first())
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
        token = create_expiring_link_token(email, expiry_time)
        
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
        decode = decode_expiring_link_token(link)
        return {
            "status_code": 200,
            "success": True,
            "data": decode,
            "message": "Link decoded successfully",
        }
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))

async def login(user_data: LoginUser, request: Request, db: SessionLocal = Depends(get_db)):
    # Validate inputs
    if not user_data.email:
        return {"status_code": 400, "detail": "Email is required"}
    if not user_data.password:
        return {"status_code": 400, "detail": "Password is required"}

    # Get User-Agent
    user_agent_string = request.headers.get("User-Agent")
    if not user_agent_string:
        raise HTTPException(status_code=400, detail="User-Agent header missing")

    ua = parse(user_agent_string)
    current_device_type = "mobile" if ua.is_mobile else ("tablet" if ua.is_tablet else "desktop")
    current_os = ua.os.family or "Unknown"
    current_browser = ua.browser.family or "Unknown"

    # Fetch user
    user_info = (
        db.query(User)
        .filter(User.userEmail == user_data.email)
        .first()
    )

    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")

    if not check_password(user_data.password, user_info.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Fetch login info
    existing_info = (
        db.query(Login)
        .filter(Login.email == user_data.email)
        .first()
    )

    client_ip = request.client.host
    
    mark = db.query(Role).filter(Role.roleId == user_info.role).first().mark

    # If login matched previous device
    if existing_info:
        try:
            different_device = (
                existing_info.ip != client_ip or
                existing_info.device != current_device_type or
                existing_info.os != current_os or
                existing_info.browser != current_browser
            )

            existing_info.loginTime = datetime.now()
            db.add(existing_info)
            
            print(jsonable_encoder(user_info))

            if different_device:
                await post_user_audit(user_info.userId, "user logged in with a different device")
                return {
                    "statusCode": 200,
                    "message": "You have logged in from a different device.",
                    "data": user_info,
                    "mark": mark
                }
            
            await post_user_audit(user_info.userId, "user logged in")


            return {
                "statusCode": 200,
                "message": "Login successful.",
                "data": user_info,
                "mark": mark
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # No previous login, create new entry
    new_login = Login(
        email=user_data.email,
        ip=client_ip,
        device=current_device_type,
        os=current_os,
        browser=current_browser,
        loginTime=datetime.now()
    )
    db.add(new_login)
    db.commit()
    
    print(user_info)
    
    await post_user_audit(user_info.userId, "user logged in")

    return {
        "statusCode": 200,
        "message": "Login successful.",
        "data": user_info,
        "mark": mark
    }

    



