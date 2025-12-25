from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Subject
from validation import SubjectBase
from services import subjects

router = APIRouter(
    prefix="/subjects",
    tags=["subjects"],
    responses={404: {"description": "Not found"}},
)


@router.post("/")
async def create_subject(data: SubjectBase):
    try:
        cls = await subjects.create_subject(data)
        return cls
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_subjects():
    try:
        allSubjects = await subjects.get_subjects()
        return allSubjects
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/id")
async def get_subject(subjectId: int):
    try:
        subject = await subjects.get_subject(subjectId)
        return subject
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))