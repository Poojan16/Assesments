from models import *
from database import *
from fastapi import HTTPException
from fastapi import Form, File, UploadFile, Query
from typing import List, Union, Optional
from sqlalchemy import select
from uploadFile import *
import math
from services import teachers, student

async def get_schools():
    try:
        db = SessionLocal()
        schools = db.query(School).all()
        return schools
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Schools not found'))
    
async def filtered_schools(
    searchTerm: Optional[str] = Form(...),
    filterState: Optional[str] = Form(...),
    filterBoard: Optional[str] = Form(...),
    limit: int = Query(8, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    try:
        db = SessionLocal()
        
        # Get all distinct states
        states = db.query(School.state).distinct().all()
        states = [state[0] for state in states if state[0]]
        
        boards = db.query(School.board).distinct().all()
        boards = [board[0] for board in boards if board[0]]
        
        total_records = db.query(School).count()
        
        if(searchTerm == ''  and filterState == 'all' and filterBoard == 'all'):
            query = select(School).limit(limit).offset(offset)
            schools = db.scalars(query).all()
            total_pages = math.ceil(total_records / limit) if total_records > 0 else 0
            return {
                "status_code": 200,
                "success": True,
                "data": schools,
                "states": states,
                "boards": boards,
                "message": "Schools fetched successfully",
                "filteredRecords": 0,
                "pagination": {
                    "page": offset,
                    "page_size": limit,
                    "total_records": total_records,
                    "total_pages": total_pages
                }
            }
        
        query = select(School)
        if(searchTerm):
            query = query.where(School.schoolName.contains(searchTerm))
        if(filterState != 'all'):
            query = query.where(School.state == filterState)
        if(filterBoard != 'all'):
            query = query.where(School.board == filterBoard)
                
        schools = query.limit(limit).offset(offset)
        pageWiseSchools = db.scalars(schools).all()
        
        total_filtered_records = len(pageWiseSchools)
        
        total_pages = math.ceil(total_filtered_records / limit) if total_filtered_records > 0 else 0
        
        return {
            "status_code": 200,
            "success": True,
            "data": pageWiseSchools,
            "states": states,
            "boards": boards,
            "message": "Schools fetched successfully",
            "filteredRecords": total_filtered_records,
            "pagination": {
                "page": offset,
                "page_size": limit,
                "total_records": total_records,
                "total_pages": total_pages
            }
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if db:
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