from models import *
from fastapi import APIRouter, HTTPException
from fastapi import UploadFile, File, Form, Body, Query
from pydantic import EmailStr
from datetime import date, datetime
from validation import GenderEnum, MapTeacher, ProvisionRequest, ReportReviewAuditBase
from typing import Dict, Any, List, Optional, Set
import base64
from utility import *
from datetime import *
from dotenv import load_dotenv
from services import teachers
from urllib.parse import quote

load_dotenv()

# SET UP ROUTER

router = APIRouter(
    prefix="/teachers",
    tags=["Teachers"],
)

@router.post("/")
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
        return await teachers.create_teacher(
            teacherName,
            teacherEmail,
            teacherContact,
            emergencyContact,
            onboardingDate,
            address,
            country,
            city,
            state,
            pin,
            qualification,
            role,
            schoolId,
            DOB,
            gender,
            iStatus,
            photo,
            PAN,
            aadhar,
            addressProof,
            DL,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while creating teacher")

@router.get("/")
async def getAll():
    try:
        allTeachers = await teachers.getAll()
        return allTeachers
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching teachers")

@router.get("/id")
async def getById(teacherId: int):
    try:
        return await teachers.getById(teacherId)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching teacher")
  
@router.get("/search")  
async def search_teacher(query: str, schoolId: int):
    try:
        query = quote(query.lower())
        return await teachers.search_teacher(query, schoolId)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while searching teacher")

@router.get("/email")
async def get_teacher_by_email(email: EmailStr):
    try:
        teacher = await teachers.get_teacher_by_email(email)
        return teacher
    except Exception as e:    
        raise HTTPException(status_code=500, detail="Something went wrong while fetching teacher")


@router.get("/class_and_subjects")
async def get_class_and_subjects(teacherId: int):
    try:
        return await teachers.get_class_and_subjects(teacherId)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching class and subjects")

@router.post("/map_teacher")
async def map_teacher(data: MapTeacher):
    try:
        return await teachers.map_teacher(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while mapping teacher")

@router.post("/provision")
async def Provision(
    data: ProvisionRequest
):
    try:
        return await teachers.Provision(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while provisioning teacher")

# subject teacher upload report
@router.post("/uploadReport")
async def upload_report(comments: str = Form(...),teacherId: int = Form(...), attachment: UploadFile = File(...), studentId: int = Form(...)):
    try:
        return await teachers.upload_report(comments,teacherId, attachment, studentId)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while uploading report")

@router.post("/workFlow")
async def create_workFlow(reportId : int = Form(...), status: bool = Form(...), teacherId: int = Form(...),comments: str = Form(...)):
    try:
        return await teachers.create_workFlow(reportId, status, teacherId,comments)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while creating workflow")
    
@router.get("/getWorkFlow")
async def get_workFlow_by_id(teacherId: int):
    try:
        workflowAudits = await teachers.get_workFlow_by_id(teacherId)
        return workflowAudits
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching workflow")
    
@router.get("/getWorkFlow/ids")
async def get_workFlow_id(reportIds: List[int] = Query(...)):
    try:
        workflowAudits = await teachers.get_workflow_id(reportIds)
        return workflowAudits
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching workflow")

@router.get("/getWorkflow_all")
async def get_workflow():
    try:
        workflowAudits = await teachers.get_workflow_report()
        return workflowAudits
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching workflow")

@router.get("/workflowAudit")
async def get_workflow():
    try:
        workflowAudits = await teachers.get_workflow()
        return workflowAudits
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching workflow")

@router.post("/importExcel")
async def add_bulk_teacherData(file: UploadFile = File(...)):
    try:
        return await teachers.add_bulk_teacherData(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while adding bulk teacher data") 


@router.post("/teacherSign")
async def teacherSign(
    signature: Union[UploadFile, None] = File(None),
    studentId: str = Form(...),
    teacherId: int = Form(...),
    subjectId: int = Form(...),
    score : int = Form(...)
):
    try:
        addSignature = await teachers.teacherSign(signature, studentId, teacherId, subjectId, score)
        return addSignature
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while adding teacher signature")

@router.get("/teacherSign/")
async def get_teacherSign():
    try:
        signature = await teachers.get_teacherSign()
        return signature
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching teacher signature")
