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
import uuid
from validation import StudentBase
import base64
import routers.teachers
from dateutil import parser
import pandas as pd
import logging


async def get_students():
    try:
        db = SessionLocal()
        students = db.query(Student).all()
        return {
            "status_code": 200,
            "success": True,
            "data": students,
            "message": "Students fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# get student by id

async def get_student(studentId: int):
    try:
        db = SessionLocal()

        student = db.query(Student).filter(Student.studentId == studentId).first()
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        async def to_base64(file_path):
            try:
                if not file_path:
                    return None
                decrypted = await decrypt_file(file_path)  # ensure await works
                return base64.b64encode(decrypted).decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        photo_b64 = await to_base64(student.photo)
        aadhar_b64 = await to_base64(student.adhaar)
        birth_b64 = await to_base64(student.birthCertificate)
        
        student.photo = photo_b64
        student.adhaar = aadhar_b64
        student.birthCertificate = birth_b64
        
        return {
            "status_code": 200,
            "data": student,
            "message": "Student found",
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# create student

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

        db = SessionLocal()

        db_parent = db.query(Parent).filter(Parent.parentEmail == parentEmail).first()

        if not db_parent and parentName and parentEmail and parentContact:
            try:
                parentAadhar = (await upload_and_encrypt_file(parentAadhar, 'parents/'))["file_url"]
            except:
                raise HTTPException(status_code=500, detail="Parent Aadhar not uploaded")

            db_parent = Parent(
                parentName=parentName,
                parentEmail=parentEmail,
                parentContact=parentContact,
                parentRelation=parentRelation,
                parentAdhaar=parentAadhar
            )
            db.add(db_parent)
            db.commit()
            db.refresh(db_parent)

        # Student documents (optional check)
        if adhaar:
            try:
                adhaar = (await upload_and_encrypt_file(adhaar, 'students/'))["file_url"]
            except:
                raise HTTPException(status_code=500, detail="Aadhar not uploaded")

        if birthCertificate:
            try:
                birthCertificate = (await upload_and_encrypt_file(birthCertificate, 'students/'))["file_url"]
            except:
                raise HTTPException(status_code=500, detail="Birth Certificate not uploaded")

        if photo:
            try:
                photo = (await upload_and_encrypt_file(photo, 'students/'))["file_url"]
            except:
                raise HTTPException(status_code=500, detail="Photo not uploaded")

        i = db.query(Student).order_by(Student.studentId.desc()).first().studentId + 1
        rollId = f"SCH{schoolId}CLS0{classId}000{i}"

        student = Student(
            studentName=studentName,
            DOB=DOB,
            gender=gender,
            classId=classId,
            schoolId=schoolId,
            address=address,
            iStatus=iStatus,
            adhaar=adhaar,
            birthCertificate=birthCertificate,
            city=city,
            state=state,
            pin=pin,
            photo=photo,
            parentId=db_parent.parentId,
            grade=6,
            rollId=rollId,
            country=country
            
        )

        db.add(student)
        db.commit()
        db.refresh(student)
        return {
            "status_code": 200,
            "success": True,
            "data": student,
            "message": "Student created successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

#  student update API

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
    db = SessionLocal()
    print("Updating student with ID:", studentId)
    print("Received data:", studentName, DOB, gender, classId, schoolId, address, adhaar, birthCertificate, city, state, pin, photo, iStatus, country)
    try:
        db_student = (
            db.query(Student)
            .filter(Student.studentId == studentId)
            .first()
        )

        if not db_student:
            raise HTTPException(status_code=404, detail="Student not found")

        # Update basic fields
        if(studentName):
            db_student.studentName = studentName
        
        if(DOB):
            db_student.DOB = DOB

        if(gender):
            db_student.gender = gender

        if(classId):
            db_student.classId = classId

        if(schoolId):
            db_student.schoolId = schoolId

        if(address):
            db_student.address = address

        if(city):
            db_student.city = city

        if(state):
            db_student.state = state

        if(pin):
            db_student.pin = pin

        if(country):
            db_student.country = country

        # Aadhar upload
        if adhaar and type(adhaar) != str:
            aadhar_url = (
                await upload_and_encrypt_file(adhaar, "students/")
            )["file_url"]

            if db_student.adhaar:
                os.unlink(db_student.adhaar)

            db_student.adhaar = aadhar_url

        # Birth certificate upload
        if birthCertificate and type(birthCertificate) != str:
            bc_url = (
                await upload_and_encrypt_file(birthCertificate, "students/")
            )["file_url"]

            if db_student.birthCertificate:
                os.unlink(db_student.birthCertificate)

            db_student.birthCertificate = bc_url

        # Photo upload
        if photo and type(photo) != str:
            photo_url = (
                await upload_and_encrypt_file(photo, "students/")
            )["file_url"]

            if db_student.photo:
                os.unlink(db_student.photo)

            db_student.photo = photo_url

        db.commit()
        db.refresh(db_student)

        return {
            "status_code": 200,
            "success": True,
            "data": db_student,
            "message": "Student updated successfully",
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()


async def add_bulk_studentData(data: List[StudentBase]):
    db = SessionLocal()
    try:
        # Cache parents to avoid repeated DB calls
        parent_cache = {}

        for idx, item in enumerate(data):

            # --- Handle Parent ---
            if item.parentEmail not in parent_cache:
                parent = db.query(Parent).filter(Parent.parentEmail == item.parentEmail).first()

                if not parent:
                    parent = Parent(
                        parentName=item.parentName,
                        parentEmail=item.parentEmail,
                        parentContact="None",
                        parentRelation="None",
                    )
                    db.add(parent)
                    db.flush()  # get parentId without full commit

                parent_cache[item.parentEmail] = parent

            parent = parent_cache[item.parentEmail]


            # --- Roll ID (with zero-padding) ---
            rollId = f"SCH{item.schoolId}CLS{str(item.classId).zfill(2)}{str(idx).zfill(4)}"

            # --- Create Student ---
            student = Student(
                studentName=item.studentName,
                DOB=data[idx].DOB,
                gender=item.gender,
                classId=item.classId,
                schoolId=item.schoolId,
                address=item.address,
                iStatus=item.iStatus,
                country=item.country,
                city=item.city,
                state=item.state,
                pin=item.pin,
                parentId=parent.parentId,
                grade=item.grade,
                rollId=rollId,
            )

            db.add(student)

        db.commit()
        return {
            "status_code": 200,
            "success": True,
            "message": "Students added successfully",
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()


#  ---------------------------------------------------
#  ------------------Student Scores-------------------
#  ---------------------------------------------------


async def getAllScores():
    try:
        db = SessionLocal()
        studentScores = db.query(StudentScore).all()
        return {
            "status_code": 200,
            "success": True,
            "data": studentScores,
            "message": "Scores fetched successfully",
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def get_student_scores(studentId: int):
    try:
        db = SessionLocal()
        studentScores = db.query(StudentScore).filter(StudentScore.studentId == studentId).all()
        return {
            "status_code": 200,
            "success": True,
            "data": studentScores,
            "message": "Scores fetched successfully",
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def create_student_score(studentId: int = Form(...), subjectId: int = Form(...), score: int = Form(...)):
    db = SessionLocal()
    try:
        grade = db.query(Grade).filter(score <= Grade.upperLimit, score >= Grade.lowerLimit ).first()
        studentScore = StudentScore(studentId=studentId, subjectId=subjectId, score=score, grade=grade.gradeId)
        db.add(studentScore)
        db.commit()
        db.refresh(studentScore)
        return {
            "status_code": 200,
            "success": True,
            "data": studentScore,
            "message": "Score created successfully",
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def update_student_score(id:int, score: int = Form(...)):
    db = SessionLocal()
    student = db.query(StudentScore).filter(StudentScore.studentScoreId == id).first()
    try:
        grade = db.query(Grade).filter(score <= Grade.upperLimit, score >= Grade.lowerLimit ).first()
        student.score = score
        student.grade = grade.gradeId
        db.commit()
        db.refresh(student)
        return {
            "status_code": 200,
            "success": True,
            "data": student,
            "message": "Score updated successfully",
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))