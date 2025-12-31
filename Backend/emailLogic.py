from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import asyncio
import logging
import time
from pydantic import BaseModel, EmailStr
from fastapi_mail import FastMail, MessageSchema, MessageType
from contextlib import contextmanager
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi_mail import ConnectionConfig
from database import SessionLocal
from models import *
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)

class EmailConfig:
    DEFAULT_BATCH_SIZE = 50  
    MAX_BATCH_SIZE = 100     
    MIN_BATCH_SIZE = 10      
    
    SMTP_RATE_LIMIT_REQUESTS_PER_MINUTE = 100  
    SMTP_RATE_LIMIT_REQUESTS_PER_DAY = 2000    
    SMTP_RATE_LIMIT_RESET_WINDOW = 86400      
    
    INITIAL_RETRY_DELAY = 30  
    MAX_RETRY_DELAY = 3600    
    BACKOFF_MULTIPLIER = 2
    
    BATCH_PROCESSING_INTERVAL = 60 
    
    MAX_CONCURRENT_SMTP_CONNECTIONS = 3

class EmailSchemaForRetry(BaseModel):
    emails: List[EmailStr]
    subject: str
    body: str
    attachment: Optional[str] = None
    metadatas: Optional[dict] = None

class EmailRequest(BaseModel):
    emails: List[EmailStr]
    subject: str
    body: str
    attachment: Optional[str] = None
    priority: Optional[int] = 0
    metadatas: Optional[dict] = None
    max_retries: Optional[int] = 3
    retry_interval: Optional[int] = 300
    batch_size: Optional[int] = None 
    
class EmailSchema1(BaseModel):
    emails: List[EmailStr]
    subject: str
    body: str
    
conf = ConnectionConfig(
    MAIL_USERNAME="pujansoni.jcasp@gmail.com",
    MAIL_PASSWORD="dyas iwyo rtjx jpcf",
    MAIL_PORT=465,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=False,
    MAIL_SSL_TLS=True, # Use True for SSL
    USE_CREDENTIALS=True,   
    VALIDATE_CERTS=True
)

SMTP_SERVER = "smtp.gmail.com"  
SMTP_PORT = 465  
EMAIL_ADDRESS = os.getenv("MAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
    
class RateLimitTracker:
    """Track and manage SMTP rate limiting"""
    def __init__(self):
        self.request_timestamps: List[float] = []
        self.daily_request_count = 0
        self.daily_reset_time = time.time() + EmailConfig.SMTP_RATE_LIMIT_RESET_WINDOW
        self.last_rate_limit_hit: Optional[float] = None
        self.consecutive_rate_limits = 0
        
    def can_send_request(self) -> bool:
        """Check if we can send another request based on rate limits"""
        current_time = time.time()
        
        if current_time >= self.daily_reset_time:
            self.daily_request_count = 0
            self.daily_reset_time = current_time + EmailConfig.SMTP_RATE_LIMIT_RESET_WINDOW
        
        if self.daily_request_count >= EmailConfig.SMTP_RATE_LIMIT_REQUESTS_PER_DAY:
            logging.warning("Daily SMTP rate limit reached")
            return False
        
        one_minute_ago = current_time - 60
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > one_minute_ago]
        
        if len(self.request_timestamps) >= EmailConfig.SMTP_RATE_LIMIT_REQUESTS_PER_MINUTE:
            logging.warning("Per-minute SMTP rate limit reached")
            return False
        
        return True
    
    def record_request(self):
        """Record a successful request"""
        current_time = time.time()
        self.request_timestamps.append(current_time)
        self.daily_request_count += 1
        self.consecutive_rate_limits = 0  
    
    def record_rate_limit_hit(self):
        """Record when we hit a rate limit"""
        self.last_rate_limit_hit = time.time()
        self.consecutive_rate_limits += 1
    
    def get_backoff_delay(self) -> float:
        """Calculate exponential backoff delay"""
        if self.consecutive_rate_limits == 0:
            return EmailConfig.INITIAL_RETRY_DELAY
        
        delay = EmailConfig.INITIAL_RETRY_DELAY * (
            EmailConfig.BACKOFF_MULTIPLIER ** (self.consecutive_rate_limits - 1)
        )
        return min(delay, EmailConfig.MAX_RETRY_DELAY)

class EmailNotificationSystem:
    def __init__(self, db_session_factory):
        logging.info("EmailNotificationSystem initialized")
        self.db_session_factory = db_session_factory
        self.fm = FastMail(conf)  
        
    async def send_email_notification(self, email_log: EmailLog, success: bool):
        """Notify user about email delivery status"""
        try:
            if success:
                logging.info(f"Email {email_log.email_id} sent successfully")
                notification_subject = "Email Delivered Successfully"
                notification_body = f"Your email '{email_log.subject}' was sent successfully."
            else:
                logging.info(f"Email {email_log.email_id} failed after {email_log.max_retries} attempts")
                notification_subject = "Email Delivery Failed"
                notification_body = f"Your email '{email_log.subject}' failed after {email_log.max_retries} attempts."
                
            logging.info(f"Notification: {notification_subject} - {notification_body}")
            
            notification_message = MessageSchema(
                subject=notification_subject,
                recipients=['pujansoni.jcasp@gmail.com'],
                body=notification_body,
                subtype=MessageType.plain
            )
            await self.fm.send_message(notification_message)
            logging.info("Notification sent successfully")
            
        except Exception as e:
            if("550, '5.4.5 Daily user sending limit exceeded" in str(e)):
                logging.warning("Daily SMTP rate limit exceeded, waiting for reset")
                await asyncio.sleep(EmailConfig.SMTP_RATE_LIMIT_RESET_WINDOW)

class CentralEmailService:
    def __init__(self, db_session_factory, notification_system: EmailNotificationSystem):
        self.db_session_factory = db_session_factory
        self.notification_system = notification_system
        self.fm = FastMail(conf)
        self.rate_limit_tracker = RateLimitTracker()
        self.smtp_semaphore = asyncio.Semaphore(EmailConfig.MAX_CONCURRENT_SMTP_CONNECTIONS)
        self.current_batch_size = EmailConfig.DEFAULT_BATCH_SIZE
        
    async def _handle_smtp_rate_limit_error(self, e, email_log: EmailLog, modify_by: Optional[int] = None):
        if "550, '5.4.5 Daily user sending limit exceeded" in str(e):
            logging.warning("Daily SMTP rate limit exceeded, waiting for reset")
            await asyncio.sleep(30)
        else:
            email_log.status = EmailStatus.FAILED
            email_log.modified_at = datetime.now()
            email_log.modified_by = modify_by
            with self.db_session_factory as db:
                db.commit()
    
    async def log_email(self, email_request: EmailRequest, modify_by: Optional[int] = None) -> int:
        """Log email attempt in database"""
        with self.db_session_factory as db:
            email_log = EmailLog(
                to=",".join(email_request.emails),
                subject=email_request.subject,
                body=email_request.body,
                attachment=email_request.attachment,
                status=EmailStatus.PENDING,
                max_retries=email_request.max_retries or 3,
                retry_interval=email_request.retry_interval or 300,
                modify_by=modify_by,
                metadatas=email_request.metadatas,
                created_at=datetime.now(),
                modified_at=datetime.now()
            )
            db.add(email_log)
            db.commit()
            db.refresh(email_log)
            return email_log.email_id
    
    async def send_email(self, email_schema: EmailSchemaForRetry, background_tasks: BackgroundTasks) -> bool:
        """Send email with background task and batch consideration"""
        try:
            logging.info(email_schema)
            
            emails = email_schema.emails
            batch_size = self._validate_and_get_batch_size()
            
            if len(emails) > batch_size:
                for i in range(0, len(emails), batch_size):
                    batch_emails = emails[i:i + batch_size]
                    batch_schema = EmailSchemaForRetry(
                        emails=batch_emails,
                        subject=email_schema.subject,
                        body=email_schema.body,
                        attachment=email_schema.attachment,
                        metadatas=email_schema.metadatas
                    )
                    
                    # Add each batch to background tasks
                    background_tasks.add_task(
                        self._send_message_with_retry_and_rate_limit, 
                        batch_schema, 
                        email_schema.metadatas
                    )
            else:
                # Single batch
                background_tasks.add_task(
                    self._send_message_with_retry_and_rate_limit, 
                    email_schema, 
                    email_schema.metadatas
                )
            
            logging.info(f"Email queued for sending in {max(1, len(emails) // batch_size)} batches")
            return True
            
        except Exception as e:
            logging.error(f"Failed to queue email: {str(e)}")
            return False
    
    async def _send_message_with_retry_and_rate_limit(self, email_schema: EmailSchemaForRetry, metadatas: Optional[dict] = None):
        """Internal method to send email with retry logic and rate limit handling"""
        logging.info("Sending email with retry and rate limit logic")
        email_id = metadatas.get('email_id') if metadatas else None
        
        if email_id:
            with self.db_session_factory as db:
                email_log = db.query(EmailLog).filter(EmailLog.email_id == email_id).first()
                if email_log:
                    await self._attempt_send_with_retry_and_rate_limit(email_log, email_schema, db)
        else:
            try:
                if not self.rate_limit_tracker.can_send_request():
                    await asyncio.sleep(self.rate_limit_tracker.get_backoff_delay())
                
                async with self.smtp_semaphore:
                    message = MessageSchema(
                        subject=email_schema.subject,
                        recipients=email_schema.emails,
                        body=email_schema.body,
                        subtype=MessageType.html
                    )
                    
                    
                    if email_schema.attachment:
                        message.attachments = [(email_schema.attachment, None)]
                    
                    await self.fm.send_message(message)
                    email_log.status = EmailStatus.SENT
                    email_log.modified_at = datetime.now()
                    db.commit()
                    self.rate_limit_tracker.record_request()
                    
            except Exception as e:
                logging.error(f"Email send failed: {str(e)}")
                
                backoff_delay = await self._handle_smtp_rate_limit_error(e)
                if backoff_delay:
                    await asyncio.sleep(backoff_delay)
                    asyncio.create_task(
                        self._send_message_with_retry_and_rate_limit(email_schema, metadatas)
                    )
    
    async def _attempt_send_with_retry_and_rate_limit(self, email_log: EmailLog, email_schema: EmailSchemaForRetry, db: SessionLocal):
        """Attempt to send email with retry logic and rate limit handling"""
        try:
            email_log.status = EmailStatus.RETRYING
            email_log.retry_count += 1
            email_log.last_attempt = datetime.now()
            db.commit()
            
            if not self.rate_limit_tracker.can_send_request():
                backoff_delay = self.rate_limit_tracker.get_backoff_delay()
                logging.warning(f"Rate limit prevented sending email {email_log.email_id}, delaying {backoff_delay}s")
                await self._schedule_retry_with_backoff(email_log, backoff_delay)
                return
            
            try:
                async with self.smtp_semaphore:
                    message = MessageSchema(
                        subject=email_log.subject,
                        recipients=email_log.to.split(','),
                        body=email_log.body,
                        subtype=MessageType.html,
                        attachments=[(email_log.attachment, None)] if email_log.attachment else []
                    )
                    
                    await self.fm.send_message(message)
                    self.rate_limit_tracker.record_request()
                    
                logging.info(f"Email {email_log.email_id} sent successfully")
                email_log.status = EmailStatus.SENT
                email_log.modified_at = datetime.now()
                db.commit()
                
                await self.notification_system.send_email_notification(email_log, success=True)
                
            except Exception as e:
                logging.error(f"Email {email_log.email_id} sending failed: {str(e)}")
                
                backoff_delay = await self._handle_smtp_rate_limit_error(e, email_log)
                if backoff_delay:
                    return  
                
                if email_log.can_retry:
                    await self._schedule_retry(email_log)
                else:
                    email_log.status = EmailStatus.FAILED
                    email_log.error_details = str(e)[:500]
                    email_log.modified_at = datetime.now()
                    db.commit()
                    await self.notification_system.send_email_notification(email_log, success=False)
                    
        except Exception as e:
            logging.error(f"Error in attempt_send for email {email_log.email_id}: {str(e)}")
        
    
    async def process_pending_emails_in_batches(self, batch_size: Optional[int] = None):
        """Process all pending/failed emails in configurable batches"""
        pending_emails = self.db_session_factory.query(EmailLog).filter(
            EmailLog.status.in_([EmailStatus.PENDING, EmailStatus.FAILED, EmailStatus.RETRYING]),
            EmailLog.retry_count < EmailLog.max_retries
        ).limit(EmailConfig.MIN_BATCH_SIZE).all()
        
        logging.info(f"Found {len(pending_emails)} pending/failed emails")
                
        for i in range(0, len(pending_emails), EmailConfig.MIN_BATCH_SIZE):
            batch = pending_emails[i:i + EmailConfig.MIN_BATCH_SIZE]
            logging.info(f"Processing batch {i//EmailConfig.MIN_BATCH_SIZE + 1} with {len(batch)} emails")
            
            for email_log in batch:
                if (email_log.last_attempt is None or 
                    datetime.now() - email_log.last_attempt > timedelta(seconds=email_log.retry_interval)):
                    
                    if not self.rate_limit_tracker.can_send_request():
                        backoff_delay = self.rate_limit_tracker.get_backoff_delay()
                        logging.warning(f"Rate limit reached, delaying batch processing for {backoff_delay}s")
                        await asyncio.sleep(backoff_delay)
                    
                    try:
                        email_schema = EmailSchemaForRetry(
                            emails=[f"{email_log.to}"],
                            subject=email_log.subject,
                            body=email_log.body,
                            attachment=email_log.attachment,
                            metadatas={'email_id': email_log.email_id}
                        )
                        await self._send_message_with_retry_and_rate_limit(email_schema, {'email_id': email_log.email_id}) 
                    except Exception as e:
                        logging.error(f"Failed to process email {email_log.email_id}: {str(e)}")
                                
            if i + EmailConfig.MIN_BATCH_SIZE < len(pending_emails):
                await asyncio.sleep(1)  
        
        logging.info(f"Processed {len(pending_emails)} emails in batches of {EmailConfig.MIN_BATCH_SIZE}")
            
    
    async def central_email_system(self, interval_seconds: int = 60, batch_size: Optional[int] = None):
        """Continuous email processing system with batch support"""
        logging.info("Starting central email processing system with batch support")
        
        while True:
            try:
                await self.process_pending_emails_in_batches(batch_size)
                logging.info(f"Batch processing completed. Next run in {interval_seconds} seconds")
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logging.error(f"Error in central email system: {str(e)}")
                await asyncio.sleep(interval_seconds)


async def send_email_with_retry(email_request: EmailRequest, background_tasks: BackgroundTasks, modify_by: Optional[int] = None):
    db_session_factory = SessionLocal()
    notification_system = EmailNotificationSystem(db_session_factory)
    email_service = CentralEmailService(db_session_factory, notification_system)
    
    email_id = await email_service.log_email(email_request, modify_by)
    logging.info(f"Email logged with ID: {email_id}")
    
    email_schema = EmailSchemaForRetry(
        emails=email_request.emails,
        subject=email_request.subject,
        body=email_request.body,
        attachment=email_request.attachment,
        metadatas={'email_id': email_id}
    )
    
    try:
        success = await email_service.send_email(email_schema, background_tasks)
    except Exception as e:
        logging.error(f"Email send failed: {str(e)}")
        success = False
    
    email_log = db_session_factory.query(EmailLog).filter(EmailLog.email_id == email_id).first()
    if not success:
        email_log.status = EmailStatus.FAILED
        email_log.modified_at = datetime.now()
        db_session_factory.commit()

    return {"message": "Email queued for sending", "email_id": email_id}

def send_email_sync(recipient_email: str, subject: str, body: str):
    """
    Send an email using the SMTP server details provided in the environment variables.

    Args:
        recipient_email (str): The email address of the recipient.
        subject (str): The subject of the email.
        body (str): The body of the email.

    Raises:
        Exception: If there is an error sending the email.

    Returns:
        None
    """
    try:
        recipient_email = "pujansoni.jcasp@gmail.com"
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        msg.attach()
    

        # Establish a secure connection using SMTP_SSL for port 465
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
        print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Error sending email: {e}")


async def send_email_endpoint(email_schema: EmailSchema1, background_tasks: BackgroundTasks):
    
    async def send_email_task(fm: FastMail, message: MessageSchema):
        try:
            await fm.send_message(message)
            print(f"Email sent to {len(message.recipients)} recipients")
        except Exception as e:
            print(f"Email sending failed: {e}")
            
    fm = FastMail(conf)
    message = MessageSchema(
        subject=email_schema.subject,
        recipients=email_schema.email[5:10],
        body=email_schema.body,
        subtype=MessageType.html
    )
    background_tasks.add_task(send_email_task, fm, message)
    print("Email sending initiated in background.")
    return {"message": "Email sending initiated in background."}

async def final_report_mail(background_tasks: BackgroundTasks, file:UploadFile = File(...), student_id: int = Form(...)):
    try:
        fm = FastMail(conf)
        db = SessionLocal()
        student = db.query(Student).filter(Student.studentId == student_id).first()
        if(student is None):
            raise HTTPException(status_code=404, detail="Student not found")
        parent = db.query(Parent).filter(Parent.parentId == student.parentId).all()
        if(parent is None):
            raise HTTPException(status_code=404, detail="Parent not found")
        email = []
        for i in parent:
            email.append('pujansoni.jcasp@gmail.com')
            
        subject = f"Final Report for {student.studentName if student else 'Student'}"
        body = f"""
            <html>
                <body>
                    <h1>Final Report for {student.studentName if student else 'Student'}</h1>
                    <p>Attached is the Final Report for {student.studentName if student else 'Student'}</p>
                    <p>Kindly connect with us for any further queries</p>
                    <p>Best regards,</p>
                    <p>Student Grading System Team</p>

                </body>
            </html>
            """
        
        message = MessageSchema(
            subject=subject,
            recipients=email,
            body=body,
            subtype=MessageType.html
        )
        message.attachments = [(file, None)]
        background_tasks.add_task(fm.send_message, message)
        return {"message": "Email sending initiated in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def send_otp_mail(email_schema: EmailSchema1, background_tasks: BackgroundTasks):
    fm = FastMail(conf)
    message = MessageSchema(
        subject=email_schema.subject,
        recipients=email_schema.email,
        body=email_schema.body,
        subtype=MessageType.plain
    )
    background_tasks.add_task(fm.send_message, message)
    return {"message": "Email sending initiated in background."}