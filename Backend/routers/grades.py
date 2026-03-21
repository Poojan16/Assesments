from database import SessionLocal
from models import Grade
from validation import GradeBase, GradeUpdate, GradeResponse
from fastapi import APIRouter,Response, HTTPException
from typing import List, Union
from services import grades

router = APIRouter(
    prefix="/grades",
    tags=["Grades"],
)


# get all the grades
@router.get("/")
async def getAll():
    try:
        allGrades = await grades.getAll()
        return allGrades
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching grades")
@router.get("/id")
async def getById(gradeId: int):
    try:
        grade = await grades.getById(gradeId)
        return grade
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching grade")

@router.post("/")
async def create_grade(data: GradeBase):
    try:
        grade = await grades.create_grade(data)
        return grade
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while creating grade")

@router.put("/")
async def update_grade(gradeId: int, data: GradeUpdate):
    try:
        grade = await grades.update_grade(gradeId, data)
        return grade
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while updating grade")
    

@router.delete("/")
async def delete_grade(gradeId: int):
    try:
        grade = await grades.delete_grade(gradeId)
        return grade
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while deleting grade")