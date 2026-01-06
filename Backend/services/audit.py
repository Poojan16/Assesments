from fastapi import  Depends, HTTPException
from database import SessionLocal
from database import get_db
from models import Audit, UserAudit
from sqlalchemy import select
import math


async def get_audits(limit: int = 10, offset: int = 0):
    try:
        db = SessionLocal()
        total_records = db.query(Audit).count()
        query = select(Audit).limit(limit).offset(offset)
        audits = db.scalars(query).all()
        total_pages = math.ceil(total_records / limit) if total_records > 0 else 0
        return {
            "status_code": 200,
            "success": True,
            "data": audits,
            "message": "Schools fetched successfully",
            "filteredRecords": 0,
            "total_records": total_records,
            "pagination": {
                "page": offset,
                "page_size": limit,
                "total_records": total_records,
                "total_pages": total_pages
            }
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

async def get_user_audits(limit: int = 10, offset: int = 0):
    try:
        db = SessionLocal()
        total_records = db.query(UserAudit).count()
        query = select(UserAudit).limit(limit).offset(offset)
        user_audits = db.scalars(query).all()
        total_pages = math.ceil(total_records / limit) if total_records > 0 else 0
        return {
            "status_code": 200,
            "success": True,
            "data": user_audits,
            "message": "User audits fetched successfully",
            "pagination": {
                "page": offset,
                "page_size": limit,
                "total_records": total_records,
                "total_pages": total_pages
            }
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