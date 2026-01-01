from fastapi import  Depends, HTTPException
from database import SessionLocal
from database import get_db
from models import Audit, UserAudit


async def get_audits():
    try:
        db = SessionLocal()
        audits = db.query(Audit).all()
        return {
            "status_code": 200,
            "success": True,
            "data": audits,
            "message": "Audits fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while fetching audits'))
    finally:
        db.close()

async def get_audit(auditId: int):
    try:
        db = SessionLocal()
        audit = db.query(Audit).filter(Audit.auditId == auditId).first()
        return {
            "status_code": 200,
            "success": True,
            "data": audit,
            "message": "Audit fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while fetching audit'))
    finally:
        db.close()


# --------------------------------------------------------------
#  -------------- USER AUDIT APIs ---------------- 
# --------------------------------------------------------------

async def get_user_audits():
    try:
        db = SessionLocal()
        user_audits = db.query(UserAudit).all()
        return {
            "status_code": 200,
            "success": True,
            "data": user_audits,
            "message": "User audits fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while fetching user audits'))
    finally:
        db.close()

async def post_user_audit(
    user_id: int, activity: str, sessionId
):
    try:
        db = SessionLocal()
        user_audit = UserAudit(user_id=user_id, activity=activity, sessionId=sessionId)
        db.add(user_audit)
        db.commit()
        db.refresh(user_audit)
        return {
            "status_code": 200,
            "success": True,
            "data": user_audit,
            "message": "User audit created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while creating user audit'))
    finally:
        db.close()