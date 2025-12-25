from database import SessionLocal
from models import Grade
from validation import GradeBase, GradeUpdate, GradeResponse
from fastapi import APIRouter,Response, HTTPException
from typing import List, Union

# get all the grades
async def getAll():
    try:
        db = SessionLocal()
        grades = db.query(Grade).all()
        return {
            "status_code": 200,
            "success": True,
            "data": grades,
            "message": "Grades fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while fetching grades'))
    finally:
        db.close()

async def getById(gradeId: int):
    try:
        db = SessionLocal()
        grade = db.query(Grade).filter(Grade.gradeId == gradeId).first()
        return {
            "status_code": 200,
            "success": True,
            "data": grade,
            "message": "Grade fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while fetching grade'))
    finally:
        db.close()

async def create_grade(data: GradeBase):
    try:
        db = SessionLocal()
        grade = Grade(gradeLetter=data.gradeLetter, upperLimit=data.upperLimit, lowerLimit=data.lowerLimit)
        db.add(grade)
        db.commit()
        db.refresh(grade)
        return {
            "status_code": 200,
            "success": True,
            "data": grade,
            "message": "Grade created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while creating grade'))
    finally:
        db.close()

async def update_grade(gradeId: int, data: GradeUpdate):
    try:
        db = SessionLocal()
        grade = db.query(Grade).filter(Grade.gradeId == gradeId).first()
        grade.gradeLetter = data.gradeLetter
        grade.upperLimit = data.upperLimit
        grade.lowerLimit = data.lowerLimit
        db.commit()
        db.refresh(grade)
        return {
            "status_code": 200,
            "success": True,
            "data": grade,
            "message": "Grade updated successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while updating grade'))
    finally:
        db.close()
    

async def delete_grade(gradeId: int):
    try:
        db = SessionLocal()
        grade = db.query(Grade).filter(Grade.gradeId == gradeId).first()
        db.delete(grade)
        db.commit()
        return {
            "status_code": 200,
            "success": True,
            "data": grade,
            "message": "Grade deleted successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while deleting grade'))
    finally:
        db.close()