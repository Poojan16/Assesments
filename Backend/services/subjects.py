from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Subject
from validation import SubjectBase




async def create_subject(data: SubjectBase):
    try:
        db = SessionLocal()
        subject = Subject(subjectName=data.subjectName, subjectCode=data.subjectCode, classId=data.classId)
        db.add(subject)
        db.commit()
        db.refresh(subject)
        return {
            "status_code": 200,
            "success": True,
            "data": subject,
            "message": "Subject created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


async def get_subjects():
    try:
        db = SessionLocal()
        subjects = db.query(Subject).all()
        return {
            "status_code": 200,
            "success": True,
            "data": subjects,
            "message": "Subjects fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


async def get_subject(subjectId: int):
    try:
        db = SessionLocal()
        subject = db.query(Subject).filter(Subject.subjectId == subjectId).first()
        return {
            "status_code": 200,
            "success": True,
            "data": subject,
            "message": "Subject fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()