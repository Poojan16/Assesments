from fastapi import APIRouter, Depends, HTTPException
from database import SessionLocal
from database import get_db
from models import Audit, UserAudit
from services import audit

router = APIRouter(
    prefix="/audit",
    tags=["Audit"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def get_audits():
    try:
        audits = await audit.get_audits()
        return audits
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/id")
async def get_audit(auditId: int):
    try:
        audit = await audit.get_audit(auditId)
        return audit
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------
#  -------------- USER AUDIT APIs ---------------- 
# --------------------------------------------------------------

@router.get("/user_audits")
async def get_user_audits():
    try:
        user_audits = await audit.get_user_audits()
        return user_audits
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/user_audits")
async def post_user_audit(
    user_id: int, activity: str,
    sessionId: int
):
    try:
        user_audit = await audit.post_user_audit(user_id, activity, sessionId)
        return user_audit
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))