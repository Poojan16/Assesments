import datetime
import asyncio
import logging
from typing import List, Optional, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from fastapi import BackgroundTasks, APIRouter, HTTPException, Depends
from fastapi_mail import FastMail, MessageSchema, MessageType
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from contextlib import asynccontextmanager
import json
from fastapi_mail import ConnectionConfig


# Import your database models
from models import School, Student, Parent, Fees
from database import get_db

router = APIRouter(
    prefix="/paymentConfig",
    tags=["Payment Config"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Email configuration (replace with your actual config)
conf = ConnectionConfig(
    MAIL_USERNAME="your-email@gmail.com",
    MAIL_PASSWORD="your-password",
    MAIL_FROM="your-email@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERDS=True
)

# Pydantic models
class EmailSchema1(BaseModel):
    email: List[EmailStr]
    subject: str
    body: str

class BatchEmailRequest(BaseModel):
    student_ids: List[int]
    notification_type: str
    scheduled_datetime: Optional[datetime.datetime] = None

class StudentEmailData(BaseModel):
    student_id: int
    student_name: str
    parent_name: str
    parent_email: str
    class_name: str
    grade: str
    total_fees: float
    due_date: datetime.date
    school_info: Dict[str, Any]

# Email Sender Class
class BatchEmailSender:
    """Optimized email sender for batch processing"""
    
    def __init__(self):
        self.fm = FastMail(conf)
        self.batch_size = 50  # Send 50 emails per batch
        self.delay_between_batches = 1  # 1 second between batches
        
    async def prepare_student_email_data(self, student_id: int, db: Session) -> Optional[StudentEmailData]:
        """Fetch all required data for a student's email"""
        try:
            # Query student with parent and school details
            stmt = (
                select(
                    Student,
                    Parent,
                    School,
                    Fees
                )
                .join(Parent, Student.parentId == Parent.parentId)
                .join(School, Student.schoolId == School.schoolId)
                .outerjoin(Fees, Student.studentId == Fees.studentId)
                .where(Student.studentId == student_id)
            )
            
            result = db.execute(stmt).first()
            if not result:
                return None
            
            student, parent, school, fees = result
            
            # Calculate total fees (assuming you have a way to get current due)
            # Modify this based on your actual fees structure
            total_fees_due = fees.amount if fees and not fees.isPaid else 0
            
            return StudentEmailData(
                student_id=student.studentId,
                student_name=student.studentName,
                parent_name=parent.parentName,
                parent_email=parent.parentEmail,
                class_name=f"Class {student.classId}",  # Modify based on your class model
                grade=f"Grade {student.grade}",  # Modify based on your grade model
                total_fees=total_fees_due,
                due_date=datetime.date(datetime.datetime.now().year, 7, 7),  # Default July 7
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
    
    def generate_email_body(self, data: StudentEmailData, notification_type: str) -> str:
        """Generate personalized HTML email body"""
        
        # Color scheme for better readability
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
        
        # Determine notification message
        notification_messages = {
            "10_days_before": f"Reminder: Fees due in 10 days",
            "5_days_before": f"Reminder: Fees due in 5 days",
            "3_days_before": f"Reminder: Fees due in 3 days",
            "due_date_morning": f"Important: Fees are due today",
            "due_date_noon": f"Final Reminder: Fees are due today",
            "immediate": f"Fees Payment Reminder"
        }
        
        notification_msg = notification_messages.get(
            notification_type, 
            "Annual Fees Reminder"
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
                <h1>📚 {data.school_info['name']}</h1>
                <h2>Fee Payment Reminder</h2>
            </div>
            
            <div class="content">
                <p>Dear <strong>{data.parent_name}</strong>,</p>
                
                <div class="details">
                    <h3>📋 Student Information:</h3>
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
                    <h3>💰 Fee Details:</h3>
                    <table>
                        <tr>
                            <th>Description</th>
                            <th>Amount</th>
                        </tr>
                        <tr>
                            <td>Total Fees Due:</td>
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
                    <h3>🏫 School Information:</h3>
                    <p><strong>School:</strong> {data.school_info['name']}</p>
                    <p><strong>Address:</strong> {data.school_info['address']}, {data.school_info['city']}, 
                       {data.school_info['state']} - {data.school_info['pin']}</p>
                    <p><strong>Contact:</strong> {data.school_info['phone']} | Email: {data.school_info['email']}</p>
                    <p><strong>Established:</strong> {data.school_info['established_year']}</p>
                </div>
                
                <div style="margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-radius: 5px;">
                    <h4>📝 Payment Instructions:</h4>
                    <ol>
                        <li>Log in to the parent portal at [Portal URL]</li>
                        <li>Navigate to the Fees section</li>
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
                <p>© {datetime.datetime.now().year} {data.school_info['name']}. All rights reserved.</p>
                <p>If you have already made the payment, please ignore this reminder.</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    async def send_batch_emails(
        self, 
        student_ids: List[int], 
        notification_type: str,
        db: Session,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Send emails to multiple students in optimized batches"""
        
        total_students = len(student_ids)
        logger.info(f"Starting batch email sending for {total_students} students")
        
        # Group emails by batch
        batches = [
            student_ids[i:i + self.batch_size] 
            for i in range(0, total_students, self.batch_size)
        ]
        
        total_sent = 0
        failed_students = []
        
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch)} students")
            
            # Prepare email data for this batch
            email_data_list = []
            recipients = []
            
            for student_id in batch:
                data = await self.prepare_student_email_data(student_id, db)
                if data:
                    email_data_list.append((student_id, data))
                    recipients.append(data.parent_email)
            
            if not email_data_list:
                continue
            
            # Generate unified subject
            subject = f"Fee Payment Reminder - {notification_type.replace('_', ' ').title()}"
            
            # Send emails in background task
            background_tasks.add_task(
                self._send_single_batch_emails,
                email_data_list,
                recipients,
                subject,
                notification_type
            )
            
            total_sent += len(email_data_list)
            
            # Add delay between batches if not last batch
            if batch_num < len(batches):
                await asyncio.sleep(self.delay_between_batches)
        
        return {
            "total_students": total_students,
            "emails_sent": total_sent,
            "failed_students": failed_students,
            "message": f"Batch email sending initiated for {total_sent} students"
        }
    
    async def _send_single_batch_emails(
        self, 
        email_data_list: List[tuple], 
        recipients: List[str],
        subject: str,
        notification_type: str
    ):
        """Send a single batch of emails"""
        try:
            # Create individual message bodies for each student
            messages = []
            for student_id, data in email_data_list:
                body = self.generate_email_body(data, notification_type)
                
                message = MessageSchema(
                    subject=f"{subject} - {data.student_name}",
                    recipients=[data.parent_email],
                    body=body,
                    subtype=MessageType.html
                )
                messages.append(message)
            
            # Send all emails (FastMail handles multiple recipients)
            # Note: You might need to adjust based on your FastMail configuration
            for message in messages:
                await self.fm.send_message(message)
            
            logger.info(f"Successfully sent {len(messages)} emails in batch")
            
        except Exception as e:
            logger.error(f"Error sending batch emails: {e}")

# Initialize email sender
email_sender = BatchEmailSender()

# Scheduler function
def schedule_annual_notifications_optimized(
    scheduler: AsyncIOScheduler,
    current_date: datetime.date,
    student_ids: List[int],
    db: Session,
    notification_times: Optional[List[str]] = None
):
    """Optimized scheduler for batch email notifications"""
    
    fee_due_date = datetime.date(current_date.year, 7, 7)
    
    if current_date > fee_due_date:
        fee_due_date = datetime.date(current_date.year + 1, 7, 7)
    
    # Define notification schedule with types
    notification_schedule = [
        ("10_days_before", fee_due_date - datetime.timedelta(days=10), datetime.time(12, 0)),
        ("5_days_before", fee_due_date - datetime.timedelta(days=5), datetime.time(12, 0)),
        ("3_days_before", fee_due_date - datetime.timedelta(days=3), datetime.time(12, 0)),
        ("due_date_morning", fee_due_date, datetime.time(9, 0)),
        ("due_date_noon", fee_due_date, datetime.time(12, 0)),
        # Test: send in 2 minutes
        ("test_immediate", datetime.datetime.now() + datetime.timedelta(minutes=2),
         (datetime.datetime.now() + datetime.timedelta(minutes=2)).time()),
    ]
    
    # Add custom times
    if notification_times:
        # Parse custom times (implementation depends on your format)
        pass
    
    # Schedule each notification
    for notification_type, date_obj, time_obj in notification_schedule:
        scheduled_datetime = datetime.datetime.combine(date_obj, time_obj)
        
        if scheduled_datetime > datetime.datetime.now():
            
            # Create scheduled task
            async def scheduled_task(
                student_ids=student_ids,
                notification_type=notification_type,
                db=db
            ):
                from fastapi import BackgroundTasks
                
                logger.info(f"Executing scheduled email task: {notification_type} for {len(student_ids)} students")
                
                # Use BackgroundTasks for async processing
                background_tasks = BackgroundTasks()
                
                result = await email_sender.send_batch_emails(
                    student_ids=student_ids,
                    notification_type=notification_type,
                    db=db,
                    background_tasks=background_tasks
                )
                
                logger.info(f"Scheduled task {notification_type} completed: {result}")
            
            # Add to scheduler
            scheduler.add_job(
                scheduled_task,
                trigger=DateTrigger(run_date=scheduled_datetime),
                id=f"batch-email-{notification_type}-{int(scheduled_datetime.timestamp())}",
                replace_existing=True,
                misfire_grace_time=300,
                coalesce=True
            )
            
            logger.info(f"Scheduled {notification_type} for {len(student_ids)} "
                       f"students at {scheduled_datetime}")

# FastAPI Endpoints
@router.post("/send-immediate-batch")
async def send_immediate_batch(
    request: BatchEmailRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Send immediate batch emails for testing"""
    try:
        result = await email_sender.send_batch_emails(
            student_ids=request.student_ids,
            notification_type=request.notification_type,
            db=db,
            background_tasks=background_tasks
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Batch email sending initiated"
        }
        
    except Exception as e:
        logger.error(f"Error in immediate batch sending: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schedule-batch-notifications")
async def schedule_batch_notifications(
    request: BatchEmailRequest,
    db: Session = Depends(get_db)
):
    """Schedule batch email notifications"""
    try:
        # Initialize scheduler
        scheduler = AsyncIOScheduler()
        scheduler.start()
        
        # Schedule notifications
        schedule_annual_notifications_optimized(
            scheduler=scheduler,
            current_date=datetime.date.today(),
            student_ids=request.student_ids,
            db=db,
            notification_times=None  # Add custom times if needed
        )
        
        return {
            "success": True,
            "message": f"Scheduled batch notifications for {len(request.student_ids)} students",
            "scheduled_datetime": request.scheduled_datetime
        }
        
    except Exception as e:
        logger.error(f"Error scheduling batch notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper endpoint to get student data for preview
@router.get("/student-email-preview/{student_id}")
async def get_student_email_preview(
    student_id: int,
    notification_type: str = "10_days_before",
    db: Session = Depends(get_db)
):
    """Preview email content for a student"""
    try:
        data = await email_sender.prepare_student_email_data(student_id, db)
        
        if not data:
            raise HTTPException(status_code=404, detail="Student not found")
        
        email_body = email_sender.generate_email_body(data, notification_type)
        
        return {
            "student_data": data.dict(),
            "email_body": email_body,
            "parent_email": data.parent_email
        }
        
    except Exception as e:
        logger.error(f"Error generating email preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

