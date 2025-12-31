# student apis

from fastapi import APIRouter
from models import *
from database import *
from fastapi import HTTPException
from uploadFile import upload_and_encrypt_file, decrypt_file
from fastapi import Form, File, UploadFile
from typing import Optional, Union
from pydantic import EmailStr
from datetime import date
from validation import StudentBase
from services import student

router = APIRouter(
    prefix="/students",
    tags=["Students"],
)

# get all the students

@router.get("/")
async def get_students():
    try:
        students = await student.get_students()
        return students
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# get student by id

@router.get("/id")
async def get_student(studentId: int):
    try:
        studentById = await student.get_student(studentId)
        return studentById
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# create student

@router.post("/")
async def create_student(
    studentName:  str = Form(...),
    DOB: date = Form(...),
    gender: GenderEnum = Form(...),
    classId: int = Form(...),
    schoolId: int = Form(...),
    address: str =  Form(...),
    iStatus: Optional[bool] = Form(False),
    adhaar: Union[UploadFile, None] = File(None),
    birthCertificate: Union[UploadFile, None] = File(None),
    country: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    pin: int = Form(...),
    photo: Union[UploadFile, None] = File(None),
    parentName: Optional[str] = Form(None),
    parentEmail: EmailStr = Form(...),
    parentContact: Optional[str] = Form(None),
    parentRelation: Optional[str] = Form(None),
    parentAadhar: Optional[Union[UploadFile, None]] = None,
):
    # Fix: front-end may send "null" as string
    try:
        if isinstance(parentAadhar, str):
            parentAadhar = None

        return await student.create_student(
            studentName,
            DOB,
            gender,
            classId,
            schoolId,
            address,
            iStatus,
            adhaar,
            birthCertificate,
            country,
            city,
            state,
            pin,
            photo,
            parentName,
            parentEmail,
            parentContact,
            parentRelation,
            parentAadhar,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#  student update API

@router.put("/id")
async def update_student(
    studentId: int = Form(...),
    studentName:  str = Form(...),
    DOB: date = Form(...),
    gender: GenderEnum = Form(...),
    classId: int = Form(...),
    schoolId: int = Form(...),
    address: str =  Form(...),
    iStatus: Optional[bool] = Form(False),
    adhaar: Optional[UploadFile] = File(None),
    birthCertificate: Optional[UploadFile] = File(None),
    country: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    pin: int = Form(...),
    photo: Optional[UploadFile] = File(None),
):
    try:
        return await student.update_student(
            studentId,
            studentName,
            DOB,
            gender,
            classId,
            schoolId,
            address,
            iStatus,
            adhaar,
            birthCertificate,
            country,
            city,
            state,
            pin,
            photo
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/importExcel")
async def add_bulk_studentData(data: List[StudentBase]):
    try:
        return await student.add_bulk_studentData(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  ---------------------------------------------------
#  ------------------Student Scores-------------------
#  ---------------------------------------------------


@router.get("/score")
async def getAllScores():
    try:
        studentScores = await student.getAllScores()
        return studentScores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/score/{studentId}")
async def get_student_scores(studentId: int):
    try:
        studentScore = await student.get_student_scores(studentId)
        return studentScore
    except Exception as e:  
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/score")
async def create_student_score(studentId: int = Form(...), subjectId: int = Form(...), score: int = Form(...)):
    try:
        addScore = await student.create_student_score(studentId, subjectId, score)
        return addScore
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/score/{id}")
async def update_student_score(id: int, score: int = Form(...)):
    try:
        updateScore = await student.update_student_score(id,score)
        return updateScore
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

