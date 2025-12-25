from models import *
from database import *
from fastapi import HTTPException
from fastapi import Form, File, UploadFile
from typing import List, Union
from uploadFile import *
from services import teachers, student

async def get_schools():
    try:
        db = SessionLocal()
        schools = db.query(School).all()
        return {
            "status_code": 200,
            "success": True,
            "data": schools,
            "message": "Schools fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Schools not found')) 
    finally:
        db.close()
 
async def schooolById(schoolId: int):
    try:
        db = SessionLocal()
        school = db.query(School).filter(School.schoolId == schoolId).first()
        return {
            "status_code": 200,
            "success": True,
            "data": school,
            "message": "School fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('School not found'))
    finally:
        db.close()


async def get_teachers():
    try:
        teachers = await teachers.getAll()
        return teachers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Teachers not found'))


async def get_students():
    try:
        db = SessionLocal()
        students = await student.get_students()
        return students
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Students not found')) 

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
    db = SessionLocal()
    try:
        if(attachments):
            try:
                file = await upload_and_encrypt_file(attachments, 'schools/')
            except Exception as e:
                raise HTTPException(status_code=500, detail='File upload failed')
            
        schools = School(
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
            attachments=file["file_url"]
        )
        db.add(schools)
        db.commit()
        db.refresh(schools)
        return {
            "status_code": 200,
            "success": True,
            "data": schools,
            "message": "School created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('School not created'))