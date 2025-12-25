from fastapi import APIRouter
from models import *
from database import *
from fastapi import HTTPException
from fastapi import Form, File, UploadFile
from typing import List, Union
from uploadFile import *
from services import admin


router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
)

# Admin APIs:
# - avg performace of all schools month wise


@router.get("/schools")
async def get_schools():
    try:
        schools = await admin.get_schools()
        return schools
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@router.get("/schools/{schoolId}")       
async def get_school(schoolId: int):
    try:
        school = await admin.schooolById(schoolId)
        return school
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teachers")
async def get_teachers():
    try:
        teachers = await admin.get_teachers()
        return teachers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students")
async def get_students():
    try:
        students = await admin.get_students()
        return students
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schools")
async def schools(
    schoolName: str = Form(...),
    schoolEmail: str = Form(...),
    primaryContactNo: str = Form(...),
    secondaryContactNo: str = Form(...),
    additionalContactNo: str = Form(...),
    address: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    country: str = Form(...),
    pin: int = Form(...),
    board: str = Form(...),
    established_year: int = Form(...),
    studentsPerClass: int = Form(...),
    maxClassLimit: int = Form(...),
    attachments: Union[UploadFile, None] = File(None),
):
    try:
        schools = await admin.schools(
            schoolName=schoolName,
            schoolEmail=schoolEmail,
            primaryContactNo=primaryContactNo,
            secondaryContactNo=secondaryContactNo,
            additionalContactNo=additionalContactNo,
            address=address,
            city=city,
            state=state,
            country=country,
            pin=pin,
            board=board,
            established_year=established_year,
            studentsPerClass=studentsPerClass,
            maxClassLimit=maxClassLimit,
            attachments=attachments,
        )
        return schools
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))