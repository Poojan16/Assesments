from database import SessionLocal
from models import *
from uploadFile import upload_and_encrypt_file, decrypt_file, FileScan
from fastapi import APIRouter, HTTPException
from fastapi import UploadFile, File, Form, Body
from pydantic import EmailStr
from datetime import date, datetime
from validation import GenderEnum, MapTeacher, ProvisionRequest, ReportReviewAuditBase
from typing import Dict, Any, List, Optional
import os
import json
from emailLogic import send_email_sync
from fastapi.encoders import jsonable_encoder
import base64
from validation import TeacherBase
import pandas as pd
from dateutil import parser
from io import BytesIO
from utility import *
from datetime import *
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()


async def create_teacher(
    teacherName: str = Form(...),
    teacherEmail: EmailStr = Form(...),
    teacherContact: int = Form(...),
    emergencyContact: int = Form(...),
    onboardingDate: date = Form(...),
    address: str = Form(...),
    country: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    pin: int = Form(...),
    qualification: str = Form(...),
    role: str = Form(...),
    schoolId: int = Form(...),
    DOB: date = Form(...),
    gender: GenderEnum = Form(...),
    iStatus: Optional[bool] = Form(False),

    photo: Optional[UploadFile] = File(None),
    PAN: Optional[UploadFile] = File(None),
    aadhar: Optional[UploadFile] = File(None),
    addressProof: Optional[UploadFile] = File(None),
    DL: Optional[UploadFile] = File(None),
):  
    
    try:
        db = SessionLocal()
        if photo:
            check_file = await FileScan(photo)
            if(check_file):
                photo_url = await upload_and_encrypt_file(photo, "teachers/")
                photo_url = photo_url["file_url"]
            else:
                photo_url = None
        else:
            photo_url = None
        if PAN:
            check_file = await FileScan(PAN)
            if(check_file):
                PAN_url = await upload_and_encrypt_file(PAN, "teachers/")
                PAN_url = PAN_url["file_url"]
            else:
                PAN_url = None
        else:
            PAN_url = None
        if aadhar:
            check_file = await FileScan(aadhar)
            if(check_file):
                aadhar_url = await upload_and_encrypt_file(aadhar, "teachers/")
                aadhar_url = aadhar_url["file_url"]
            else:
                aadhar_url = None
        else:
            aadhar_url = None
        if addressProof:
            check_file = await FileScan(addressProof)
            if(check_file):
                addressProof_url = await upload_and_encrypt_file(addressProof, "teachers/")
                addressProof_url = addressProof_url["file_url"]
            else:
                addressProof_url = None
        else:
            addressProof_url = None
        if DL:
            check_file = await FileScan(DL)
            if(check_file):
                DL_url = await upload_and_encrypt_file(DL, "teachers/")
                DL_url = DL_url["file_url"]
            else:
                DL_url = None
        else:
            DL_url = None
        teacher = Teacher(
            teacherName=teacherName,
            teacherEmail=teacherEmail,
            teacherContact=teacherContact,
            emergencyContact=emergencyContact,
            onboardingDate=onboardingDate,
            address=address,
            country=country,
            city=city,
            state=state,
            pin=pin,
            photo=photo_url,
            PAN=PAN_url,
            aadhar=aadhar_url,
            addressProof=addressProof_url,
            DL=DL_url,
            qualification=qualification,
            role=db.query(Role).filter(Role.roleName == role).first().roleId,
            schoolId=schoolId,
            DOB=DOB,
            gender=gender,
            iStatus=iStatus,
        )
        db.add(teacher)
        db.commit()
        db.refresh(teacher)
        
        expiry_time = timedelta(minutes=int(os.getenv("EXPIRY_TIME")))
        try:
            link = create_expiring_link_token(teacherEmail, expiry_time)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        email_subject = "Welcome to Student Grading System"
        email_body = f"Hello {teacherName},\n\nWelcome to Student Grading System. We are glad to have you on board.\n\nBest,\nStudent Grading System \n Click on this link: <a href='{os.getenv('FRONTEND_URL')}/set-password/{link}'>Set New Password</a>, once you click on link it will take you to set password page"
        send_email_sync('pujansoni.jcasp@gmail.com', email_subject, email_body)
        return {
            "message": "Teacher created successfully",
            "data": teacher,
            "status": 200,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def getAll():
    db = SessionLocal()
    teachers = db.query(Teacher).all()
    return {
        "message": "Teachers fetched successfully",
        "data": teachers,
        "status": 200,
        "success": True
    }

async def search_teacher(query: str, schoolId: int):
    try:
        db = SessionLocal()
        teachers = db.query(Teacher).filter(Teacher.teacherName.contains(query), Teacher.schoolId == schoolId).all()
        return {
            "message": "Teachers fetched successfully",
            "data": teachers,
            "status": 200,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def getById(teacherId: int):
    try:
        db = SessionLocal()
        teacher = db.query(Teacher).filter(Teacher.teacherId == teacherId).first()
        return {
            "message": "Teacher fetched successfully",
            "data": teacher,
            "status": 200,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_teacher_by_email(email: EmailStr):
    try:
        db = SessionLocal()
        teacher = db.query(Teacher).filter(Teacher.teacherEmail == email).first()
        return {
            "message": "Teacher fetched successfully",
            "data": teacher,
            "status": 200,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# map teacher with class and subject and data can be in any form. so we'll check and store accordingly

async def get_class_and_subjects(teacherId: int):
    try:
        db = SessionLocal()
        teacher = db.query(TeacherClassSubject).filter(TeacherClassSubject.teacherId == teacherId).all()
        return {
            "message": "Class and subjects fetched successfully",
            "data": teacher,
            "status": 200,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def map_teacher(data: MapTeacher):
    db = SessionLocal()
    try:
        for teacher_data in data.teachers:
            
            # Fetch teacher by email
            teacher = db.query(Teacher).filter(
                Teacher.teacherEmail == teacher_data.email
            ).first()

            if not teacher:
                raise HTTPException(status_code=404, detail=f"Teacher not found: {teacher_data.email}")

            # Loop classes
            for cls in teacher_data.classes:
                
                class_ = db.query(Class).filter(Class.classId == cls.classId).first()
                if not class_:
                    raise HTTPException(status_code=404, detail=f"Class not found: {cls.classId}")

                # Loop subjects
                for sub in cls.subjects:
                    subject = db.query(Subject).filter(Subject.subjectId == sub).first()
                    
                    if not subject:
                        raise HTTPException(status_code=404, detail=f"Subject not found: {sub}")

                    teacherClassSubject = TeacherClassSubject(
                        teacherId=teacher.teacherId,
                        classId=class_.classId,
                        subjectId=subject.subjectId,
                    )

                    db.add(teacherClassSubject)

        db.commit()
        return {
            "message": "Teacher mapped successfully",
            "data": data,
            "status": 200,
            "success": True
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    
# provision api for teacher and student
# student fields = studentName(studentId), email, class, provision
#  teacher fields = teacherName, email, subject, class, provision

async def Provision(
    data: ProvisionRequest
):
    db = SessionLocal()
    try:
        # student model has active and remark field. so if provision is for student then we'll update active and remark
        if(data.studentId):
            db_student = db.query(Student).filter(Student.studentId == data.studentId).first()
            if(db_student):
                activeAudit = Audit(
                    reason=data.provision,
                    changedBy=data.changedBy,
                    tableName="student",
                    fieldId=db_student.studentId,
                    fieldName="active",
                    oldValue=db_student.active,
                    newValue= False if db_student.active else True
                )
                remarkAudit = Audit(
                    reason=data.provision,
                    changedBy=data.changedBy,
                    tableName="student",
                    fieldId=db_student.studentId,
                    fieldName="remark",
                    oldValue=db_student.remark if db_student.remark != None else "N/A",
                    newValue=data.provision
                )
                db.add(activeAudit)
                db.add(remarkAudit)
                db_student.active = False if db_student.active else True
                db_student.remark = data.provision                
                db.commit()
                db.refresh(db_student)
                
                subject = f"Student provison to {'continue' if db_student.active else 'discontinue'}!"
                body = f"Student {db_student.studentName} has been provisioned to {'continue' if db_student.active else 'discontinue'}. Provision: {data.provision}"
                
                
                send_email_sync('pujansoni.jcasp@gmail.com', subject, body)
                
                return {
                    "status_code": 200,
                    "success": True,
                    "data": db_student,
                    "message": "Student provisioned to discontinue",
                }
            else:
                raise HTTPException(status_code=404, detail="Student not found")
        # teacher model has active and remark field. so if provision is for teacher then we'll update active and remark
        elif(data.teacherId):
            db_teacher = db.query(Teacher).filter(Teacher.teacherId == data.teacherId).first()
            changed_by = db.query(Teacher).filter(Teacher.teacherId == data.changedBy).first()
            if(db_teacher):
                if(data.offBoardingDate):
                    offBoardAudit = Audit(
                        reason=data.provision,
                        changedBy=data.changedBy,
                        tableName="teacher",
                        
                        fieldId=db_teacher.teacherId,
                        fieldName="offBoardingDate",
                        oldValue= 'N/A' if db_teacher.offboardingDate == None else db_teacher.offboardingDate,
                        newValue=data.offBoardingDate
                    )
                    db.add(offBoardAudit)
                    db_teacher.offBoardingDate = data.offBoardingDate
                activeAudit = Audit(
                    reason=data.provision,
                    changedBy=data.changedBy,
                    tableName="teacher",
                    fieldId=db_teacher.teacherId,
                    fieldName="active",
                    oldValue=db_teacher.active,
                    newValue=False if db_teacher.active else True
                )
                remarkAudit = Audit(
                    reason=data.provision,
                    changedBy=data.changedBy,
                    tableName="teacher",
                    fieldId=db_teacher.teacherId,
                    fieldName="remark",
                    oldValue=db_teacher.remark if db_teacher.remark != None else "N/A",
                    newValue=data.provision
                )
                db.add(activeAudit)
                db.add(remarkAudit)
                db_teacher.active = False if db_teacher.active else True
                db_teacher.remark = data.provision
                db.commit()
                db.refresh(db_teacher)
                
                subject = f"Your account has been {'activated' if db_teacher.active else 'deactivated'}"
                body = f"Hi {db_teacher.teacherName}, your account has been {'activated' if db_teacher.active else 'deactivated'} by {changed_by.teacherName}. Reason: {data.provision}"
                
                send_email_sync('pujansoni.jcasp@gmail.com', subject, body)
                
                return {
                    "status_code": 200,
                    "success": True,
                    "data": db_teacher,
                    "message": "Teacher deactivated successfully",
                }
            else:
                raise HTTPException(status_code=404, detail="Teacher not found")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# subject teacher upload report
async def upload_report(comments: str = Form(...),teacherId: int = Form(...), attachment: UploadFile = File(...), studentId: int = Form(...)):
    try:
        report = await upload_and_encrypt_file(attachment, "work-flow-audit/")
        if(report['file_url']):
            try:
                reportdata = StudentReports(
                    studentId=studentId,
                    report=report['file_url'],
                    teacherId=teacherId,
                    comments=comments
                )
                db = SessionLocal()
                db.add(reportdata)
                db.commit()
                db.refresh(reportdata)
                
                try:
                    db = SessionLocal()
                    teacher = db.query(Teacher).filter(Teacher.teacherId == teacherId).first()
                    db_report = ReportReviewAudit(
                        comments=comments,
                        teacherId=teacherId,
                        attachment=report['file_url'],
                        role = teacher.role,
                        school = teacher.schoolId,
                        reportId=reportdata.reportId
                    )
                    db.add(db_report)
                    db.commit()
                    db.refresh(db_report)
                    return {
                        "status_code": 200,
                        "success": True,
                        "message": "Report successfully uploaded"
                    }
                except Exception as e:
                    db.rollback()
                    raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def create_workFlow(reportId : int = Form(...), status: bool = Form(...), teacherId: int = Form(...),comments: str = Form(...)):
    print(comments)
    db = SessionLocal()
    report = db.query(StudentReports).filter(StudentReports.reportId == reportId).first()
    teacher = db.query(Teacher).filter(Teacher.teacherId == teacherId).first()
    try:
        if(status == True):
            report.status = status
            db.add(report)
            db.commit()
            db.refresh(report)
        
        db_report = ReportReviewAudit(
            comments=comments,
            teacherId=teacherId,
            attachment=report.report,
            role = teacher.role,
            school = teacher.schoolId,
            reportId = reportId
        )
        db.add(db_report)
        db.commit()
        db.refresh(db_report)
        return {
            "status_code": 200,
            "success": True,
            "message": "Report successfully uploaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_workFlow_by_id(teacherId: int):
    try:
        db = SessionLocal()
        teacher = db.query(StudentReports).filter(StudentReports.teacherId == teacherId).all()
        for i in range(len(teacher)):
            try:
                pdf_bytes = await decrypt_file(teacher[i].report)
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                teacher[i].report = pdf_base64
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        return {
            "status_code": 200,
            "success": True,
            "data": teacher,  
            "message": "Workflow fetched successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_workflow_report():
    try:
        db = SessionLocal()
        teacher = db.query(StudentReports).all()
        for i in range(len(teacher)):
            try:
                pdf_bytes = await decrypt_file(teacher[i].report)
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                teacher[i].report = pdf_base64
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        return {
            "status_code": 200,
            "success": True,
            "data": teacher,  
            "message": "Workflow fetched successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_workflow():
    try:
        db = SessionLocal()
        teacher = db.query(ReportReviewAudit).all()
        for i in range(len(teacher)):
            try:
                pdf_bytes = await decrypt_file(teacher[i].attachment)
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                teacher[i].attachment = pdf_base64
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        return {
            "status_code": 200,
            "success": True,
            "data": teacher,  
            "message": "Workflow fetched successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def get_workflow_id(reportIds: List[int]):
    try:
        db = SessionLocal()
        reportReviews = db.query(ReportReviewAudit).filter(ReportReviewAudit.reportId.in_(reportIds), ReportReviewAudit.iStatus == False).all()
        for reportReview in reportReviews:
            try:
                pdf_bytes = await decrypt_file(reportReview.attachment)
                pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
                reportReview.attachment = pdf_base64
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        return {
            "status_code": 200,
            "success": True,
            "data": reportReviews,
            "message": "Workflow fetched successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def add_bulk_teacherData(file: UploadFile = File(...)):
    # 1. Validate file type
    if not (file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an Excel file.")

    # 2. Read Excel file into pandas DataFrame
    contents = await file.read()
    try:
        df = pd.read_excel(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")

    # 3. Convert DataFrame rows to list of dicts
    records = df.to_dict(orient="records")

    db = SessionLocal()
    try:
        teachers_to_add = []
        print(records[0])
        for record in records:
            # Parse onboardingDate
            raw_onboarding = record.get("onboardingDate")
            try:
                if isinstance(raw_onboarding, (datetime, pd.Timestamp)):
                    onboarding_date = raw_onboarding.date()
                elif isinstance(raw_onboarding, date):
                    onboarding_date = raw_onboarding
                else:
                    onboarding_date = parser.parse(str(raw_onboarding)).date()
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid onboardingDate format: {raw_onboarding}")

            # Parse DOB
            raw_dob = record.get("DOB")
            try:
                if isinstance(raw_dob, (datetime, pd.Timestamp)):
                    dob = raw_dob.date()
                elif isinstance(raw_dob, date):
                    dob = raw_dob
                else:
                    dob = parser.parse(str(raw_dob)).date()
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid DOB format: {raw_dob}")

            teacher = Teacher(
                teacherName      = record.get("teacherName"),
                teacherEmail     = record.get("teacherEmail"),
                teacherContact   = record.get("teacherContact"),
                emergencyContact = record.get("emergencyContact"),
                onboardingDate   = onboarding_date,
                address          = record.get("address"),
                city             = record.get("city"),
                state            = record.get("state"),
                country          = record.get("country"),
                pin              = record.get("pin"),
                qualification    = record.get("qualification"),
                role             = record.get("role"),
                schoolId         = record.get("schoolId"),
                DOB              = dob,
                gender           = record.get("gender")
            )
            teachers_to_add.append(teacher)

        # Bulk add
        db.add_all(teachers_to_add)
        db.commit()

        return {
            "message": "Teachers added successfully",
            "status": 200,
            "success": True
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


async def teacherSign(
    signature: Union[UploadFile, None] = File(None),
    studentId: str = Form(...),
    teacherId: int = Form(...),
    subjectId: int = Form(...),
    score : int = Form(...)
):
    db = SessionLocal()
    try:
        signatureData = db.query(TeacherSignature).filter(TeacherSignature.teacherId == teacherId, TeacherSignature.subjectId == subjectId, TeacherSignature.studentId == studentId).first()
        if signatureData:
            if(signature):
                signature = await upload_and_encrypt_file(signature,'teacherSigns/')
            signatureData.signature = signature['file_url']
            signatureData.score = score
            db.commit()
            db.refresh(signatureData)
            return {
                "message": "Signature updated successfully",
                "data": signatureData,
                "status": 200,
                "success": True
            }
        teacher = db.query(Teacher).filter(Teacher.teacherId == teacherId).first()
        if teacher:
            if(signature):
                signature = await upload_and_encrypt_file(signature,'teacherSigns/')
            
            for student in studentId.split(","):
                sign = TeacherSignature(
                    signature = signature['file_url'],
                    studentId = int(student),
                    teacherId = teacherId,
                    subjectId = subjectId,
                    score = score
                )
                db.add(sign)
            db.commit()
            db.refresh(sign)
        
            try:
                # handle audit
                audit = Audit(
                    changedBy = teacherId,
                    tableName = "TeacherSignature",
                    fieldId = sign.signatureId,
                    fieldName = "signature",
                    oldValue = "",
                    newValue = signature['file_url'],
                    reason = "Signature added"
                )
                db.add(audit)
                db.commit()
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=str(e))
            return {
                "message": "Signature added successfully",
                "data": sign,
                "status_code": 200,
                "success": True
            }
        else:
            raise HTTPException(status_code=404, detail="Teacher not found")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def get_teacherSign():
    try:
        db = SessionLocal()
        teacher = db.query(TeacherSignature).all()
        return {
            "message": "Signature fetched successfully",
            "data": teacher,
            "status": 200,
            "success": True
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
