from datetime import datetime, timedelta, time as dt_time
from pytz import timezone
import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager
from rabbitMQ import Producer
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import BackgroundTasks, FastAPI, HTTPException, Depends, APIRouter
from fastapi_mail import FastMail, MessageSchema, MessageType, ConnectionConfig
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func
from pydantic import BaseModel, EmailStr, Field
import redis.asyncio as redis
from emailLogic import *
from models import School, Student, Parent, Fees, Grade
from database import get_db, SessionLocal



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================ Configuration & Constants ================
class NotificationConfig:
    """Centralized configuration for notification system"""
    
    NOTIFICATION_INTERVALS = {
        '10_days_before': 24 * 3600, 
        '5_days_before': 12 * 3600,  
        '3_days_before': 6 * 3600,    
        'due_date': 3600,             
        

    }
    
    # Retry Configuration
    MAX_RETRY_COUNT = 3
    RETRY_BACKOFF_BASE = 2
    
    # Tracking
    NOTIFICATION_EXPIRY_DAYS = 30
    
    @classmethod
    def get_batch_size(cls, notification_type: str) -> int:
        """Get appropriate batch size based on notification type"""
        if 'due_date' in notification_type:
            return cls.MIN_BATCH_SIZE  # Smaller batches for urgent notifications
        return cls.DEFAULT_BATCH_SIZE

# ================ Models ================
# class EmailSchema1(BaseModel):
#     email: List[EmailStr]
#     subject: str
#     body: str

# class BatchEmailRequest(BaseModel):
#     student_ids: List[int]
#     notification_type: str = '2_days_before'
#     scheduled_datetime: Optional[datetime] = None
#     batch_size: Optional[int] = Field(default=None, ge=1, le=NotificationConfig.MAX_BATCH_SIZE)
    
# class BatchEmailResponse(BaseModel):
#     total_students: int = 0
#     emails_sent: int = 0
#     failed_students: List[int] = []
#     failed_details: List[Dict[str, Any]] = []
#     batch_id: Optional[str] = None
#     message: str = "Batch email sending initiated for students"
    
# class PaymentResponse(BaseModel):
#     status_code: int = 200
#     success: bool = True
#     data: BatchEmailResponse
#     message: str = "Emails have been sent successfully"

class StudentEmailData(BaseModel):
    student_id: str
    student_name: str
    parent_name: str
    parent_email: str
    class_name: str
    grade: str
    total_fees: float
    due_date: datetime
    fee_id: int
    fee_status: str
    school_info: Dict[str, Any]

async def prepare_student_email_data(student_id: int, db: SessionLocal) -> Optional[StudentEmailData]:
    """Prepare student email data with payment status re-validation"""
    logger.info(f"Preparing student email data for student {student_id}")
    try:
        stmt = (
            select(
                Student,
                Parent,
                School,
                Fees
            )
            .join(Parent, Student.parentId == Parent.parentId)
            .join(School, Student.schoolId == School.schoolId)
            .outerjoin(Fees, and_(
                Student.studentId == Fees.studentId,
                Fees.isPaid == False  # Only unpaid fees
            ))
            .where(and_(
                Student.studentId == student_id,
                Student.active == True,
                Parent.parentEmail.isnot(None),
                Parent.parentEmail != ""
            ))
        )
        
        result = db.execute(stmt).first()
        if not result:
            return "No student found"
        
        logger.info(f"Found student {result} with unpaid fees")
        student, parent, school, fees = result
        
        if fees:
            db.refresh(fees)
            if fees.isPaid:
                logger.info(f"Fee {fees.feeId} for student {student_id} was paid recently, skipping")
                return None
        
        total_fees_due = fees.amount if fees and not fees.isPaid else 0
        
        grade_info = db.query(Grade).filter(Grade.gradeId == student.grade).first()
        grade_letter = grade_info.gradeLetter if grade_info else "N"
        
        return StudentEmailData(
            student_id=student.rollId,
            student_name=student.studentName,
            parent_name=parent.parentName,
            parent_email=parent.parentEmail,
            class_name=f"Class {student.classId}",
            grade="Not graded" if grade_letter == "N" else grade_letter,
            total_fees=total_fees_due,
            due_date=datetime(datetime.now().year, 7, 7),
            fee_id=fees.feesId if fees else 0,
            fee_status="Paid" if fees and fees.isPaid else "Unpaid",
            school_info={
                "name": school.schoolName,
                "email": school.schoolEmail,
                "phone": school.primaryContactNo,
                "address": school.address,
                "city": school.city,
                "state": school.state,
                "country": school.country,
                "pin": school.pin,
                "established_year": school.established_year
            }
        )
    except Exception as e:
        logger.error(f"Error fetching data for student {student_id}: {e}")
        return None
    
def generate_email_body(data: StudentEmailData, notification_type: str) -> str:
    """
    Generates an HTML email body for fee payment reminders
    
    Args:
        data (StudentEmailData): The student data
        notification_type (str): The type of notification
    
    Returns:
        str: The HTML body of the email
    """
    styles = """
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
            .content { padding: 20px; }
            .details { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; }
            .fee-amount { color: #d32f2f; font-size: 24px; font-weight: bold; }
            .due-date { color: #ff9800; font-weight: bold; }
            .school-info { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 15px 0; }
            .footer { background-color: #f1f1f1; padding: 15px; text-align: center; font-size: 12px; color: #666; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th { background-color: #4CAF50; color: white; padding: 10px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #ddd; }
            .highlight { background-color: #fffacd; padding: 5px; border-radius: 3px; }
        </style>
    """
    
    notification_messages = {
        "10_days_before": f"Reminder:fees due in 10 days",
        "5_days_before": f"Reminder:fees due in 5 days",
        "3_days_before": f"Reminder:fees due in 3 days",
        "due_date_morning": f"Important:fees are due today",
        "due_date_noon": f"Final Reminder:fees are due today",
        "immediate": f"Fees Payment Reminder"
    }
    
    notification_msg = notification_messages.get(
        notification_type, 
        "Annual fees Reminder"
    )
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        {styles}
        <meta charset="UTF-8">
        <title>Fee Payment Reminder</title>
    </head>
    <body>
        <div class="header">
            <h1>{data.school_info['name']}</h1>
            <h2>Fee Payment Reminder</h2>
        </div>
        
        <div class="content">
            <p>Dear <strong>{data.parent_name}</strong>,</p>
            
            <div class="details">
                <h3>Student Information:</h3>
                <table>
                    <tr>
                        <th>Field</th>
                        <th>Details</th>
                    </tr>
                    <tr>
                        <td>Student Name:</td>
                        <td><strong>{data.student_name}</strong></td>
                    </tr>
                    <tr>
                        <td>Class:</td>
                        <td>{data.class_name}</td>
                    </tr>
                    <tr>
                        <td>Grade:</td>
                        <td>{data.grade}</td>
                    </tr>
                    <tr>
                        <td>Roll Number:</td>
                        <td>#{data.student_id}</td>
                    </tr>
                </table>
            </div>
            
            <div class="details">
                <h3>Fee Details:</h3>
                <table>
                    <tr>
                        <th>Description</th>
                        <th>Amount</th>
                    </tr>
                    <tr>
                        <td>Total fees Due:</td>
                        <td class="fee-amount">₹{data.total_fees:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Due Date:</td>
                        <td class="due-date">{data.due_date.strftime('%B %d, %Y')}</td>
                    </tr>
                    <tr>
                        <td>Notification:</td>
                        <td class="highlight">{notification_msg}</td>
                    </tr>
                </table>
                
                <p style="margin-top: 15px;">
                    <strong>Important:</strong> Please ensure payment is made before the due date to avoid late fees.
                </p>
            </div>
            
            <div class="school-info">
                <h3>School Information:</h3>
                <p><strong>School:</strong> {data.school_info['name']}</p>
                <p><strong>Address:</strong> {data.school_info['address']}, {data.school_info['city']}, 
                   {data.school_info['state']} - {data.school_info['pin']}</p>
                <p><strong>Contact:</strong> {data.school_info['phone']} | Email: {data.school_info['email']}</p>
                <p><strong>Established:</strong> {data.school_info['established_year']}</p>
            </div>
            
            <div style="margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-radius: 5px;">
                <h4>Payment Instructions:</h4>
                <ol>
                    <li>Log in to the parent portal at [Portal URL]</li>
                    <li>Navigate to the fees section</li>
                    <li>Select the outstanding amount and proceed to payment</li>
                    <li>You can pay via UPI, Net Banking, or Credit/Debit Card</li>
                    <li>Save the payment receipt for future reference</li>
                </ol>
                
                <p style="margin-top: 10px;">
                    <strong>Need Help?</strong> Contact our accounts department at 
                    {data.school_info['phone']} during school hours (9 AM - 4 PM).
                </p>
            </div>
        </div>
        
        <div class="footer">
            <p>This is an automated message. Please do not reply to this email.</p>
            <p>© {datetime.now().year} {data.school_info['name']}. All rights reserved.</p>
            <p>If you have already made the payment, please ignore this reminder.</p>
        </div>
    </body>
    </html>
    """
    logging.info(f"Email HTML body generated:{len(html_body)}")
    return html_body

async def schedule_annual_notifications():
    """Schedule annual notification batches"""
    try:
        logger.info("Scheduling annual notifications...")
        db = SessionLocal()
        scheduler = AsyncIOScheduler()
        fees = db.query(Fees).filter(Fees.isPaid == False).all()
        student_ids = []
        
        for fee in fees:
            student_info = db.query(Student).filter(
                Student.studentId == fee.studentId,
                Student.active == True
            ).first()
            if student_info:
                student_ids.append(student_info.studentId)
        
        if not student_ids:
            logger.info("No students with unpaid fees found for scheduling")
            db.close()
            return
        
        current_date = datetime.now()
        fee_due_date = datetime(current_date.year, 7, 7)
        
        if current_date > fee_due_date:
            fee_due_date = datetime(current_date.year + 1, 7, 7)
        
        notification_schedule = [
            # ("10_days_before", fee_due_date - timedelta(days=10), dt_time(9, 0)),  
            # ("5_days_before", fee_due_date - timedelta(days=5), dt_time(10, 0)),   
            # ("3_days_before", fee_due_date - timedelta(days=3), dt_time(11, 0)),   
            # ("due_date_morning", fee_due_date, dt_time(9, 0)),                     
            # ("due_date_noon", fee_due_date, dt_time(13, 0)),
            ("in_1_minute", datetime.now(), dt_time(datetime.now().hour, datetime.now().minute + 1))                       
        ]
        
        for notification_type, date_obj, time_obj in notification_schedule:
            scheduled_datetime = datetime.combine(
                date_obj.date(), 
                time_obj
            )
            
            logger.info(f"Scheduling {scheduled_datetime} for {datetime.now()}")
            
            logger.info(datetime.now()+timedelta(minutes=1))
            
            if scheduled_datetime > datetime.now():
                async def send_scheduled_emails(
                    s_ids=list(student_ids),  
                    n_type=notification_type  
                ):
                    """Task to send scheduled emails"""
                    task_db = SessionLocal()
                    for student_id in s_ids:
                        try:
                            logger.info(f"Executing scheduled email task: {n_type} for {student_id} students")
                                                        
                            result = await prepare_student_email_data(student_id, task_db)
                            
                            body = generate_email_body(result, n_type)
                            
                            email_schema = EmailSchema1(
                                subject="Payment Reminder",
                                emails=['pujansoni.jcasp@gmail.com'],
                                body=body
                            )
                            await Producer(email_schema)
                            
                            logger.info(f"Scheduled task {n_type} completed successfully")
                            
                        except Exception as e:
                            logger.error(f"Error in scheduled task {n_type}: {e}")
                            raise
                        finally:
                            task_db.close()
                
                job_id = f"batch-email-{notification_type}-{date_obj.strftime('%Y%m%d')}"
                
                result = await send_scheduled_emails(
                    student_ids, notification_type
                )
                logger.info(result)
                scheduler.add_job(
                    send_scheduled_emails,
                    trigger=DateTrigger(run_date=scheduled_datetime),
                    id=job_id,
                    replace_existing=True,
                    misfire_grace_time=300,  
                    coalesce=True,
                )
                
                logger.info(f"Scheduled {notification_type} for {len(student_ids)} "
                           f"students at {scheduled_datetime}")
        
        db.close()
        
        logger.info(f"Scheduled {len(notification_schedule)} notification batches "
                   f"for {len(student_ids)} students")
    
    except Exception as e:
        logger.error(f"Error scheduling annual notifications: {e}")
        if 'db' in locals():
            db.close()
        raise