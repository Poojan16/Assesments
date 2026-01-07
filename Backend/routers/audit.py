from fastapi import APIRouter, Depends, HTTPException
from database import SessionLocal
from database import get_db
from models import Audit, UserAudit
from services import audit
from typing import Optional

router = APIRouter(
    prefix="/audit",
    tags=["Audit"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def get_audits(limit: int = 10, offset: int = 0):
    try:
        audits = await audit.get_audits(limit, offset)
        return audits
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching audits")

@router.get("/id")
async def get_audit(auditId: int):
    try:
        audit = await audit.get_audit(auditId)
        return audit
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching audit")


# --------------------------------------------------------------
#  -------------- USER AUDIT APIs ---------------- 
# --------------------------------------------------------------

@router.get("/user_audits")
async def get_user_audits(limit: int = 10, offset: int = 0):
    try:
        user_audits = await audit.get_user_audits(limit, offset)
        return user_audits
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching user audits")

@router.post("/user_audits")
async def post_user_audit(
    user_id: int, activity: str,
    sessionId: Optional[int] = None
):
    try:
        user_audit = await audit.post_user_audit(user_id, activity, sessionId)
        return user_audit
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while creating user audit")