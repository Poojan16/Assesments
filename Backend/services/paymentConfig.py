from datetime import datetime, timedelta, time as dt_time
from pytz import timezone
import asyncio
import logging
import time
import uuid
from typing import List, Optional, Dict, Any, Tuple
from contextlib import asynccontextmanager

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
    
    # SMTP Configuration
    SMTP_MAX_RETRIES = 3
    SMTP_RETRY_DELAYS = [5, 30, 60]  # seconds for each retry
    SMTP_RATE_LIMIT_DELAY = 60  # seconds when rate-limited
    
    # Batch Configuration
    DEFAULT_BATCH_SIZE = 50  # Conservative batch size to avoid SMTP throttling
    MAX_BATCH_SIZE = 100
    MIN_BATCH_SIZE = 10
    
    # Rate Limiting
    MAX_EMAILS_PER_MINUTE = 100  # Adjust based on your SMTP provider
    MAX_EMAILS_PER_HOUR = 500
    
    # Notification Intervals
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
class EmailSchema1(BaseModel):
    email: List[EmailStr]
    subject: str
    body: str

class BatchEmailRequest(BaseModel):
    student_ids: List[int]
    notification_type: str = '2_days_before'
    scheduled_datetime: Optional[datetime] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=NotificationConfig.MAX_BATCH_SIZE)
    
class BatchEmailResponse(BaseModel):
    total_students: int = 0
    emails_sent: int = 0
    failed_students: List[int] = []
    failed_details: List[Dict[str, Any]] = []
    batch_id: Optional[str] = None
    message: str = "Batch email sending initiated for students"
    
class PaymentResponse(BaseModel):
    status_code: int = 200
    success: bool = True
    data: BatchEmailResponse
    message: str = "Emails have been sent successfully"

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

#  Redis & Tracking 
class RedisManager:
    """Manages Redis connections and operations"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.redis_client = None
        return cls._instance
    
    async def connect(self, redis_url: str = "redis://localhost:6379"):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                redis_url, 
                encoding="utf-8", 
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Redis connected successfully")
            return self.redis_client
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            # Fallback to in-memory storage
            self.redis_client = None
            return None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔻 Redis connection closed")

class NotificationTracker:
    """Tracks notifications to prevent duplicates and manage retries"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def mark_notification_sent(self, student_id: int, fee_id: int, notification_type: str, batch_id: str = None):
        """Mark a notification as sent"""
        if not self.redis:
            return
        
        try:
            # Store notification record
            notification_key = f"notif:{student_id}:{fee_id}:{notification_type}"
            notification_data = {
                "sent_at": datetime.now().isoformat(),
                "batch_id": batch_id or "",
                "retry_count": 0
            }
            
            # Store for 30 days
            await self.redis.hset(notification_key, mapping=notification_data)
            await self.redis.expire(
                notification_key, 
                NotificationConfig.NOTIFICATION_EXPIRY_DAYS * 86400
            )
            
            # Store in sent notifications set for quick lookup
            sent_key = f"sent_notifs:{student_id}"
            await self.redis.sadd(sent_key, f"{fee_id}:{notification_type}")
            await self.redis.expire(sent_key, 86400 * 7)  # Keep for 7 days
            
        except Exception as e:
            logger.error(f"Error marking notification sent: {e}")
    
    async def was_notification_sent(self, student_id: int, fee_id: int, notification_type: str) -> bool:
        """Check if notification was already sent"""
        if not self.redis:
            return False
        
        try:
            # Check specific notification
            notification_key = f"notif:{student_id}:{fee_id}:{notification_type}"
            if await self.redis.exists(notification_key):
                return True
            
            # Check in sent notifications set
            sent_key = f"sent_notifs:{student_id}"
            return await self.redis.sismember(sent_key, f"{fee_id}:{notification_type}")
        except Exception as e:
            logger.error(f"Error checking notification sent: {e}")
            return False
    
    async def increment_retry_count(self, student_id: int, fee_id: int, notification_type: str) -> int:
        """Increment retry count for a notification"""
        if not self.redis:
            return 0
        
        try:
            key = f"retry:{student_id}:{fee_id}:{notification_type}"
            retry_count = await self.redis.incr(key)
            await self.redis.expire(key, 3600)  # Expire after 1 hour
            return retry_count
        except Exception as e:
            logger.error(f"Error incrementing retry count: {e}")
            return 1
    
    async def get_retry_count(self, student_id: int, fee_id: int, notification_type: str) -> int:
        """Get retry count for a notification"""
        if not self.redis:
            return 0
        
        try:
            key = f"retry:{student_id}:{fee_id}:{notification_type}"
            count = await self.redis.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Error getting retry count: {e}")
            return 0
    
    async def store_failed_notification(self, student_id: int, fee_id: int, 
                                       notification_type: str, error: str, batch_id: str):
        """Store failed notification for retry"""
        if not self.redis:
            return
        
        try:
            failed_key = f"failed:{batch_id}:{student_id}"
            failed_data = {
                "student_id": str(student_id),
                "fee_id": str(fee_id),
                "notification_type": notification_type,
                "error": error,
                "failed_at": datetime.now().isoformat(),
                "retry_count": 0
            }
            await self.redis.hset(failed_key, mapping=failed_data)
            await self.redis.expire(failed_key, 86400)  # Keep for 24 hours
        except Exception as e:
            logger.error(f"Error storing failed notification: {e}")

# Email Sender with Rate Limiting 
class EnhancedBatchEmailSender:
    """Enhanced email sender with rate limiting, retries, and failure handling"""
    
    def __init__(self, redis_manager: RedisManager):
        # Email configuration
        self.conf = ConnectionConfig(
            MAIL_USERNAME="pujansoni.jcasp@gmail.com",
            MAIL_PASSWORD="dyas iwyo rtjx jpcf",
            MAIL_PORT=465,
            MAIL_SERVER="smtp.gmail.com",
            MAIL_STARTTLS=False,
            MAIL_SSL_TLS=True,
            USE_CREDENTIALS=True,
            VALIDATE_CERTS=True
        )
        self.fm = FastMail(self.conf)
        
        self.rate_limit_key = "email_rate_limit"
        self.rate_limit_window = 60
        self.max_emails_per_window = NotificationConfig.MAX_EMAILS_PER_MINUTE
        
        self.redis_manager = redis_manager
        self.redis_client = None
        self.tracker = None
        
        self.task_queue = asyncio.Queue()
        self.max_concurrent_tasks = 5
        self.is_processing = False
        
    async def initialize(self):
        """Initialize Redis connection and tracker"""
        self.redis_client = await self.redis_manager.connect()
        if self.redis_client:
            self.tracker = NotificationTracker(self.redis_client)
        
        self.is_processing = True
        asyncio.create_task(self._process_task_queue())
    
    async def shutdown(self):
        """Shutdown the email sender"""
        self.is_processing = False
        await self.redis_manager.close()
    
    async def check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        if not self.redis_client:
            return True
        
        try:
            current_time = time.time()
            window_start = current_time - self.rate_limit_window
            
            await self.redis_client.zremrangebyscore(
                self.rate_limit_key, 0, window_start
            )
            
            count = await self.redis_client.zcount(
                self.rate_limit_key, window_start, current_time
            )
            
            return count < self.max_emails_per_window
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True
    
    async def update_rate_limit(self):
        """Update rate limit counter"""
        if not self.redis_client:
            return
        
        try:
            current_time = time.time()
            member = f"email:{current_time}:{uuid.uuid4()}"
            await self.redis_client.zadd(
                self.rate_limit_key, 
                {member: current_time}
            )
            await self.redis_client.expire(self.rate_limit_key, 3600)
        except Exception as e:
            logger.error(f"Error updating rate limit: {e}")
    
    async def prepare_student_email_data(self, student_id: int, db: SessionLocal) -> Optional[StudentEmailData]:
        """Prepare student email data with payment status re-validation"""
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
                return None
            
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
    
    def generate_email_body(self, data: StudentEmailData, notification_type: str) -> str:
        """Generate HTML email body"""
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
                <p>© {datetime.now().year} {data.school_info['name']}. All rights reserved.</p>
                <p>If you have already made the payment, please ignore this reminder.</p>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    async def send_single_email_with_retry(
        self,
        student_id: int,
        data: StudentEmailData,
        notification_type: str,
        batch_id: str,
        retry_count: int = 0
    ) -> Tuple[bool, str]:
        """Send single email with retry logic and rate limiting"""
        
        if self.tracker and data.fee_id:
            was_sent = await self.tracker.was_notification_sent(
                student_id, data.fee_id, notification_type
            )
            if was_sent:
                logger.info(f"Notification already sent for student {student_id}, fee {data.fee_id}")
                return True, "Already sent"
        
        if not await self.check_rate_limit():
            logger.warning(f"Rate limit exceeded, delaying email for student {student_id}")
            await asyncio.sleep(NotificationConfig.SMTP_RATE_LIMIT_DELAY)
        
        max_retries = NotificationConfig.SMTP_MAX_RETRIES
        for attempt in range(retry_count, max_retries):
            try:
                if attempt > 0:
                    delay = NotificationConfig.SMTP_RETRY_DELAYS[min(
                        attempt - 1, 
                        len(NotificationConfig.SMTP_RETRY_DELAYS) - 1
                    )]
                    logger.info(f"Retry attempt {attempt} for student {student_id}, waiting {delay}s")
                    await asyncio.sleep(delay)
                
                await self.update_rate_limit()
                
                body = self.generate_email_body(data, notification_type)
                
                recipient_email = "pujansoni.jcasp@gmail.com"  # Replace with data.parent_email for production
                
                message = MessageSchema(
                    subject=f"Fee Payment Reminder - {notification_type.replace('_', ' ').title()} - {data.student_name}",
                    recipients=[recipient_email],
                    body=body,
                    subtype=MessageType.html
                )
                
                logger.info(f"Sending email to {recipient_email} (attempt {attempt + 1})")
                await self.fm.send_message(message)
                
                if self.tracker and data.fee_id:
                    await self.tracker.mark_notification_sent(
                        student_id, data.fee_id, notification_type, batch_id
                    )
                
                logger.info(f"Email sent successfully to student {student_id}")
                return True, "Success"
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Attempt {attempt + 1} failed for student {student_id}: {error_msg}")
                
                if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                    logger.warning(f"SMTP rate limit hit, waiting {NotificationConfig.SMTP_RATE_LIMIT_DELAY}s")
                    await asyncio.sleep(NotificationConfig.SMTP_RATE_LIMIT_DELAY)
                    continue
                
                if self.tracker and data.fee_id:
                    await self.tracker.store_failed_notification(
                        student_id, data.fee_id, notification_type, error_msg, batch_id
                    )
                
                if attempt == max_retries - 1:
                    return False, error_msg
        
        return False, "Max retries exceeded"
    
    async def send_batch_emails(
        self,
        student_ids: List[int],
        notification_type: str,
        db: SessionLocal,
        background_tasks: BackgroundTasks,
        batch_size: Optional[int] = None,
        batch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send emails to multiple students with proper batching and rate limiting"""
        if not batch_id:
            batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        batch_size = batch_size or NotificationConfig.get_batch_size(notification_type)
        batch_size = min(batch_size, NotificationConfig.MAX_BATCH_SIZE)
        
        total_students = len(student_ids)
        logger.info(f"Starting batch email sending for {total_students} students, batch size: {batch_size}")
        
        batches = [
            student_ids[i:i + batch_size]
            for i in range(0, total_students, batch_size)
        ]
        
        total_sent = 0
        failed_students = []
        failed_details = []
        
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch)} students")
            
            email_data_list = []
            for student_id in batch:
                data = await self.prepare_student_email_data(student_id, db)
                if data and data.total_fees > 0:  # Only if there's actually a fee due
                    email_data_list.append((student_id, data))
                else:
                    logger.info(f"Skipping student {student_id} - no unpaid fees")
            
            if not email_data_list:
                logger.info(f"Batch {batch_num} has no students with unpaid fees, skipping")
                continue
            
            batch_task = asyncio.create_task(
                self._process_email_batch(
                    email_data_list, notification_type, batch_id, batch_num
                )
            )
            
            batch_result = await batch_task
            
            total_sent += batch_result['sent']
            failed_students.extend(batch_result['failed_students'])
            failed_details.extend(batch_result['failed_details'])
            
            if batch_num < len(batches):
                await asyncio.sleep(NotificationConfig.SMTP_RATE_LIMIT_DELAY)
        
        return {
            "total_students": total_students,
            "emails_sent": total_sent,
            "failed_students": failed_students,
            "failed_details": failed_details,
            "batch_id": batch_id,
            "message": f"Batch email sending completed. Sent: {total_sent}, Failed: {len(failed_students)}"
        }
    
    async def _process_email_batch(
        self,
        email_data_list: List[Tuple[int, StudentEmailData]],
        notification_type: str,
        batch_id: str,
        batch_num: int
    ) -> Dict[str, Any]:
        """Process a single batch of emails"""
        sent = 0
        failed_students = []
        failed_details = []
        
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        async def process_single(student_id: int, data: StudentEmailData):
            async with semaphore:
                success, error = await self.send_single_email_with_retry(
                    student_id, data, notification_type, batch_id
                )
                return student_id, success, error
        
        tasks = [
            process_single(student_id, data)
            for student_id, data in email_data_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            
            student_id, success, error = result
            if success:
                sent += 1
            else:
                failed_students.append(student_id)
                failed_details.append({
                    "student_id": student_id,
                    "error": error,
                    "notification_type": notification_type,
                    "batch_num": batch_num
                })
        
        return {
            "sent": sent,
            "failed_students": failed_students,
            "failed_details": failed_details
        }
    
    async def _process_task_queue(self):
        """Process tasks from the queue with rate limiting"""
        while self.is_processing:
            try:
                task = await self.task_queue.get()
                await task()
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing task: {e}")

class NotificationScheduler:
    """Manages scheduled notification tasks"""
    
    def __init__(self, email_sender: EnhancedBatchEmailSender, db: SessionLocal):
        self.scheduler = AsyncIOScheduler()
        self.email_sender = email_sender
        self.db = db
        self.is_running = False
    
    def start(self):
        """Start the scheduler"""
        if not self.is_running:
            self.scheduler.add_job(
                self.check_and_send_due_notifications,
                IntervalTrigger(hours=1),  
                id="hourly_notification_check",
                replace_existing=True
            )
            
            self.scheduler.add_job(
                self.retry_failed_notifications,
                IntervalTrigger(minutes=30), 
                id="failed_notification_retry",
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Notification scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("🔻 Notification scheduler stopped")
    
    async def check_and_send_due_notifications(self):
        """Check for due notifications and send them"""
        logger.info("Checking for due notifications...")
        
        try:
            current_date = datetime.now().date()
            fee_due_date = datetime.date(current_date.year, 7, 7)
            
            if current_date > fee_due_date:
                fee_due_date = datetime.date(current_date.year + 1, 7, 7)
            
            days_until_due = (fee_due_date - current_date).days
            
            notification_type = None
            if days_until_due == 10:
                notification_type = "10_days_before"
            elif days_until_due == 5:
                notification_type = "5_days_before"
            elif days_until_due == 3:
                notification_type = "3_days_before"
            elif days_until_due == 0:
                current_hour = datetime.now().hour
                if 9 <= current_hour < 12:
                    notification_type = "due_date_morning"
                elif 12 <= current_hour < 15:
                    notification_type = "due_date_noon"
            
            if notification_type:
                student_ids = self.get_students_with_unpaid_fees()
                
                if student_ids:
                    background_tasks = BackgroundTasks()
                    
                    result = await self.email_sender.send_batch_emails(
                        student_ids=student_ids,
                        notification_type=notification_type,
                        db=self.db,
                        background_tasks=background_tasks
                    )
                    
                    logger.info(f"Sent {notification_type} notifications: {result}")
            
        except Exception as e:
            logger.error(f"Error in notification check: {e}")
    
    async def retry_failed_notifications(self):
        """Retry failed notifications"""
        logger.info("Checking for failed notifications to retry...")
        
        try:
            email_logs = self.db.query(EmailLog).filter(
                EmailLog.status == EmailStatus.FAILED,
                EmailLog.retry_count < 3
            ).all()
            
            for email_log in email_logs:
                await self.email_sender.retry_failed_email(email_log.emailId)
        except Exception as e:
            logger.error(f"Error retrying failed notifications: {e}")
            
    def get_students_with_unpaid_fees(self) -> List[int]:
        """Get list of student IDs with unpaid fees"""
        try:
            fees = self.db.query(Fees).filter(
                Fees.isPaid == False,
                Fees.dueDate >= datetime.now().date()  
            ).all()
            
            student_ids = []
            for fee in fees:
                student = self.db.query(Student).filter(
                    Student.studentId == fee.studentId,
                    Student.active == True
                ).first()
                
                if student:
                    student_ids.append(student.studentId)
            
            return student_ids
        except Exception as e:
            logger.error(f"Error getting students with unpaid fees: {e}")
            return []


# Global instances
redis_manager = RedisManager()
email_sender = EnhancedBatchEmailSender(redis_manager)
notification_scheduler = None


async def send_immediate_batch(
    request: BatchEmailRequest,
    background_tasks: BackgroundTasks,
):
    db = SessionLocal()
    try:
        result = await email_sender.send_batch_emails(
            student_ids=request.student_ids,
            notification_type=request.notification_type,
            db=db,
            background_tasks=background_tasks,
            batch_size=request.batch_size
        )
        return {
            "status_code": 200,
            "success": True,
            "data": result,
            "message": "Batch email sending initiated successfully"
        }

    except Exception as e:
        logger.exception("Error in immediate batch sending")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

    finally:
        db.close()


# async def get_notification_status(batch_id: str):
#     """Get status of a notification batch"""
#     # This would query Redis for batch status
#     # Implementation depends on your tracking needs
#     return {"batch_id": batch_id, "status": "processing"}


async def schedule_annual_notifications(scheduler: AsyncIOScheduler):
    """Schedule annual notification batches"""
    try:
        # Create a new session for this function
        db = SessionLocal()
        
        # Get students with unpaid fees
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
        
        # Define notification schedule with specific times
        current_date = datetime.now()
        fee_due_date = datetime(current_date.year, 7, 7)
        
        if current_date > fee_due_date:
            fee_due_date = datetime(current_date.year + 1, 7, 7)
        
        # Define schedule with specific times
        notification_schedule = [
            ("10_days_before", fee_due_date - timedelta(days=10), dt_time(9, 0)),  
            ("5_days_before", fee_due_date - timedelta(days=5), dt_time(10, 0)),   
            ("3_days_before", fee_due_date - timedelta(days=3), dt_time(11, 0)),   
            ("due_date_morning", fee_due_date, dt_time(9, 0)),                     
            ("due_date_noon", fee_due_date, dt_time(13, 0)),
            ("in 1 minute", datetime.now() + timedelta(minutes=1), dt_time(0, 0))                       
        ]
        
        for notification_type, date_obj, time_obj in notification_schedule:
            # Combine date and time
            scheduled_datetime = datetime.combine(
                date_obj.date(), 
                time_obj
            )
            
            if scheduled_datetime > datetime.now():
                # Create task with proper parameter binding
                async def send_scheduled_emails(
                    s_ids=list(student_ids),  # Create copy to avoid closure issues
                    n_type=notification_type  # Capture current value
                ):
                    """Task to send scheduled emails"""
                    # Create new database session for the task
                    task_db = SessionLocal()
                    try:
                        logger.info(f"Executing scheduled email task: {n_type} for {len(s_ids)} students")
                        
                        # Create background tasks instance
                        background_tasks = BackgroundTasks()
                        
                        # Send emails
                        result = await email_sender.send_batch_emails(
                            student_ids=s_ids,
                            notification_type=n_type,
                            db=task_db,
                            background_tasks=background_tasks
                        )
                        
                        # Execute background tasks
                        await background_tasks()
                        
                        logger.info(f"Scheduled task {n_type} completed successfully")
                        return result
                        
                    except Exception as e:
                        logger.error(f"Error in scheduled task {n_type}: {e}")
                        raise
                    finally:
                        task_db.close()
                
                # Add job to scheduler
                job_id = f"batch-email-{notification_type}-{date_obj.strftime('%Y%m%d')}"
                
                scheduler.add_job(
                    send_scheduled_emails,
                    trigger=DateTrigger(run_date=scheduled_datetime),
                    id=job_id,
                    replace_existing=True,
                    misfire_grace_time=300,  # 5 minutes grace period
                    coalesce=True
                )
                
                logger.info(f"Scheduled {notification_type} for {len(student_ids)} "
                           f"students at {scheduled_datetime}")
        
        # Close the main session
        db.close()
        
        logger.info(f"Scheduled {len(notification_schedule)} notification batches "
                   f"for {len(student_ids)} students")
    
    except Exception as e:
        logger.error(f"Error scheduling annual notifications: {e}")
        if 'db' in locals():
            db.close()
        raise