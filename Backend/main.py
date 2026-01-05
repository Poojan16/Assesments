from fastapi import FastAPI, BackgroundTasks, HTTPException,  UploadFile, File, Form, Query, Body
import math
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import ssl
from database import *
from dotenv import load_dotenv
from validation import *
from emailLogic import *
from routers import roles, teachers, users,grades, admin, classes, subjects, student, audit, paymentConfig
load_dotenv()
from emailLogic import  send_email_endpoint, EmailSchema1
from fastapi.exceptions import HTTPException as httpException
from user_agents import parse
from database import SessionLocal
from models import *
from datetime import *
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from uploadFile import decrypt_file, upload_and_encrypt_file
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import random
from redis_client import redis_client,get_reset_token       
import logging
from services.paymentConfig import *
import pika
import json
from rabbitMQ import Consumer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # RabbitMQ Connection establishment
    # check_rabbitmq_connection()
    
    scheduler = AsyncIOScheduler()
    # scheduler.add_job(func=schedule_annual_notifications, trigger="interval", seconds=20)
    # scheduler.add_job(func=Consumer, trigger="interval", seconds=10)
    scheduler.start()
    yield
    scheduler.shutdown()
    
    
    
# FastAPI app    
app = FastAPI(
    lifespan=lifespan,
    title="Student Grading System",
    description="This is the main API for managing all the student grading system.",
    version="1.0.0",
)

def publish_message(message: str):
    # Connect to RabbitMQ using the service name defined in docker-compose
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq', port=5672))
    channel = connection.channel()
    channel.queue_declare(queue='fastapi_queue')
    channel.basic_publish(exchange='', routing_key='fastapi_queue', body=message.encode())
    connection.close()

@app.post("/send")
def send_message_to_rabbitmq(message: dict):
    try:
        publish_message(json.dumps(message))
        return {"status": "Message sent to RabbitMQ", "message_body": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

        
@app.on_event("shutdown")
async def shutdown():
    await redis_client.close()

origins = [
    "http://localhost:3000",  # Example for a React frontend on port 3000
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(roles.router)
app.include_router(users.router)
app.include_router(grades.router)
app.include_router(admin.router)
app.include_router(teachers.router)
app.include_router(classes.router)
app.include_router(subjects.router)
app.include_router(student.router)
app.include_router(audit.router)


@app.get("/tokens")
async def get_reset_token():
    return get_reset_token()
@app.get("/")
def read_root():
    conn = get_db()
    CreateTables()
    return {"Connection:": conn}

@app.get("/health")
def health_check():
    return {"status": "healthy", "scheduler_running": app.state.scheduler.running}


@app.post("/send-email/")
async def send_email(email_data: EmailSchema1, background_tasks: BackgroundTasks):
    # background_tasks.add_task(send_email_sync, email_data.recipient_email, email_data.subject, email_data.body)
    await send_email_endpoint(email_data, background_tasks)
    return {"message": "Email sending initiated in the background"}

@app.post("/importExcel")
async def add_records_in_school_duplicate_table(data: List[SchoolBase]):
    
    print(data[0])
    
    db = SessionLocal()
    for record in data:
        db_school = School(
            schoolName=record.schoolName,
            schoolEmail=record.schoolEmail,
            primaryContactNo=record.primaryContactNo,
            secondaryContactNo=record.secondaryContactNo,
            additionalContactNo=record.additionalContactNo,
            address=record.address,
            city=record.city,
            state=record.state,
            country=record.country,
            pin=record.pin,
            board=record.board,
            studentsPerClass=record.studentsPerClass,
            maxClassLimit=record.maxClassLimit,
            established_year=record.established_year,
        )
        db.add(db_school)
        db.commit()
        db.refresh(db_school)
    return "Successfully added records in school table"

@app.post("/send-final-report")
async def send_final_report(student_id: int = Form(...), background_tasks: BackgroundTasks = BackgroundTasks(), file: UploadFile = File(...)):
    await final_report_mail(background_tasks, student_id=student_id, file=file)
    return {"message": "Email sending initiated in the background"}

@app.get("/addFees")
async def add_fees():
    try:
        db = SessionLocal()
        students = db.query(Student).all()
        for student in students:
            fees = Fees(
                studentId=student.studentId,
                amount= 10000 + (1000*student.classId),
                isPaid=False,
                classId=student.classId
            )
            db.add(fees)
        db.commit()
        db.close()
        return {"message": "Fees added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transfer-student-next-class")
async def transfer_student_next_class(classId: int, schoolId: int, studentId: Optional[int] = None):
    db = SessionLocal()
    school = db.query(School).filter(School.schoolId == schoolId).first()
    if(studentId):
        try:            
            student = db.query(Student).filter(Student.studentId == studentId).first()
            if(not student):
                return {"message": "Student not found"}
            if((student.grade != 5 or student.grade != 6) and school.maxClassLimit != student.classId):
                student.classId = student.classId + 1
                student.grade = db.query(Grade).filter(Grade.gradeLetter == 'N').first().gradeId
                db.add(student)
                fees = Fees(
                    studentId=student.studentId,
                    amount= 10000 + (1000*student.classId),
                    isPaid=False,
                    classId=student.classId
                )
                db.add(fees)
                db.commit()
                db.close()
                return {"message": "Student transferred to next class successfully"}
            else:
                return {"message": "Student already in last class"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    try:
        students = db.query(Student).filter(Student.classId == classId, Student.schoolId == schoolId).all()
        for student in students:
            if(student.grade != 5 and student.grade != 6 and school.maxClassLimit != student.classId):
                student.classId = student.classId + 1
                student.grade = db.query(Grade).filter(Grade.gradeLetter == 'N').first().gradeId
                db.add(student)
                fees = Fees(
                    studentId=student.studentId,
                    amount= 10000 + (1000*student.classId),
                    isPaid=False,
                    classId=student.classId
                )
                db.add(fees)
        db.commit()
        db.close()
        return {"message": "Students transferred to next class successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  
@app.post("/send-central-email")
async def Central_email_testing(
    email: EmailStr = 'pujansoni.jcasp@gmail.com',
    subject: str = "Email Subject",
    body: str = "Email Body",
    attachment: Optional[UploadFile] = File(None),
    metadatas: Any = None,
    max_retries: int = 3,
    retry_interval: int = 300,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    email_data = EmailRequest(
        emails=[email],
        subject=subject,
        body=body,
        attachment=attachment,  # UploadFile or None
        metadatas=metadatas,
        max_retries=max_retries,
        retry_interval=retry_interval
    )

    try:
        response = await send_email_with_retry(email_data,background_tasks)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-bulk-student-score")  
async def add_bulk_student_score():
    db = SessionLocal()
    try:
        students = db.query(Student).all()
        subjects = db.query(Subject).all()
        
        existing_scores = db.query(StudentScore).all()
        existing_score_set = {(score.studentId, score.subjectId) for score in existing_scores}
        
        grades = db.query(Grade).all()
        grade_map = {}
        for grade in grades:
            for score in range(grade.lowerLimit, grade.upperLimit + 1):
                grade_map[score] = grade.gradeId
        
        subjectwise_marks_given = 0
        new_scores = []
        
        for student in students:
            for subject in subjects:
                if (student.studentId, subject.subjectId) in existing_score_set:
                    subjectwise_marks_given += 1
                    continue
                
                try:
                    score = random.randint(32, 99)
                    grade_id = grade_map.get(score)
                    
                    if not grade_id:
                        grade = db.query(Grade).filter(
                            Grade.upperLimit >= score,
                            Grade.lowerLimit <= score
                        ).first()
                        grade_id = grade.gradeId if grade else None
                    
                    if grade_id:
                        new_scores.append({
                            'studentId': student.studentId,
                            'subjectId': subject.subjectId,
                            'score': score,  
                            'grade': grade_id
                        })
                        
                except Exception as e:
                    print(f"Error processing student {student.studentId}, subject {subject.subjectId}: {e}")
        
        if new_scores:
            try:
                db.bulk_insert_mappings(StudentScore, new_scores)
                db.commit()
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=f"Bulk insert failed: {str(e)}")
        
        total_students = len(students)
        total_subjects = len(subjects)
        total_possible = total_students * total_subjects
        new_records_added = len(new_scores)
        
        return {
            "message": f"Processed {total_possible} student-subject combinations",
            "details": {
                "already_existing": subjectwise_marks_given,
                "newly_added": new_records_added,
                "failed": total_possible - (subjectwise_marks_given + new_records_added)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        db.close()
  
@app.get("/schedule-annual-notifications")      
async def schedule_notitfication():
    result = await schedule_annual_notifications()
    return result

@app.post("/background-test")
async def background_testing(background_tasks: BackgroundTasks, metadatas: Optional[dict] = None):
    db = SessionLocal()
    email = CentralEmailService(db, EmailNotificationSystem(db))
    emailRequest = EmailSchemaForRetry(
        emails=['pujansoni.jcasp@gmail.com'],
        subject="Testing background task",
        body="Testing the background tasks",
        attachment=None,
        metadatas=metadatas
    )
    background_tasks.add_task(email._send_message_with_retry_and_rate_limit, emailRequest, metadatas)
    return {"message": "Email sending initiated in the background"}

@app.get("/sessions")
async def sessions(sessionId:str):
    try:
        db = SessionLocal()
        sessions = db.query(SessionLog).filter(SessionLog.sessionId == sessionId).first()
        return {
            "status_code": 200,
            "success": True,
            "data": {
                "expired": False if sessions.isActive else True,
                "valid": sessions.isActive,
                "data":sessions,
                "expiresAt": sessions.expiresAt
            },
            "message": "Sessions fetched successfully",
        }
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))
    
# set cron job for scheduling the task again and again
# @app.get("/consumer")
# async def consumer():
#     try:
#         await Consumer()
#     except Exception as e:
#         raise httpException(status_code=400, detail=str(e))
 
@app.post("/mail")   
async def mail(email_schema: EmailSchema1, background_tasks: BackgroundTasks):
    try:
        await send_otp_mail(email_schema, background_tasks)
    except Exception as e:
        raise httpException(status_code=400, detail=str(e))