from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum
import asyncio
import logging
from pydantic import BaseModel, EmailStr
from fastapi_mail import FastMail, MessageSchema, MessageType
from contextlib import contextmanager
import os
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi_mail import ConnectionConfig
from database import SessionLocal
from models import *
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)

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
    metadatas: Optional[dict] = None
    max_retries: Optional[int] = 3
    retry_interval: Optional[int] = 300
    
    
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

class EmailSchema1(BaseModel):
    email: List[EmailStr]
    subject: str
    body: str


SMTP_SERVER = "smtp.gmail.com"  
SMTP_PORT = 465  
EMAIL_ADDRESS = os.getenv("MAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("MAIL_PASSWORD")

class EmailSchema(BaseModel):
    email: EmailStr
    subject: str
    body: str


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
            logging.error(f"Failed to send notification: {str(e)}")

class CentralEmailService:
    def __init__(self, db_session_factory, notification_system: EmailNotificationSystem):
        self.db_session_factory = db_session_factory
        self.notification_system = notification_system
        self.fm = FastMail(conf)
        
    @contextmanager
    def get_db(self):
        """Context manager for database sessions"""
        db = self.db_session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    
    async def log_email(self, email_request: EmailRequest, modify_by: Optional[int] = None) -> EmailLog:
        """Log email attempt in database"""
        with self.get_db() as db:
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
            db.flush()
            db.commit()
            return email_log.email_id
    
    async def send_email(self, email_schema: EmailSchemaForRetry, background_tasks: BackgroundTasks) -> bool:
        """Send email with background task"""
        try:
            logging.info(email_schema)
            message = MessageSchema(
                subject=email_schema.subject,
                recipients=email_schema.emails,
                body=email_schema.body,
                subtype=MessageType.plain
            )
            
            if email_schema.attachment:
                message.attachments = [(email_schema.attachment, None)]
            
            # Add to background tasks for async sending
            background_tasks.add_task(self._send_message_with_retry, message, email_schema.metadatas)
            # await self._send_message_with_retry( message, email_schema.metadatas)
            logging.info("Email queued for sending in the background")
            return True
            
        except Exception as e:
            logging.error(f"Failed to queue email: {str(e)}")
            return False
    
    async def _send_message_with_retry(self, message: MessageSchema, metadatas: Optional[dict] = None):
        """Internal method to send email with retry logic"""
        logging.info("Sending email with retry logic")
        email_id = metadatas.get('email_id') if metadatas else None
        logging.info(f"Sending email with email_id: {email_id}")
        
        if email_id:
            with self.get_db() as db:
                email_log = db.query(EmailLog).filter(EmailLog.email_id == email_id).first()
                if email_log:
                    await self._attempt_send_with_retry(email_log, message, db)
                    logging.info(f"Email with email_id: {email_id} sent successfully")
        else:
            try:
                await self.fm.send_message(message)
            except Exception as e:
                logging.error(f"Email send failed: {str(e)}")
    
    async def _attempt_send_with_retry(self, email_log: EmailLog, message: MessageSchema, db: SessionLocal):
        """Attempt to send email with retry logic"""
        try:
            email_log.status = EmailStatus.RETRYING
            email_log.retry_count += 1
            email_log.last_attempt = datetime.now()
            db.commit()
            logging.info(f"Email {email_log.email_id} is being retried")
            
            try:
                await self.fm.send_message(message)
            except Exception as e:
                logging.error(f"Email {email_log.email_id} sending failed: {str(e)}")
                
            logging.info(f"Email {email_log.email_id} sent successfully")
            email_log.status = EmailStatus.SENT
            email_log.modified_at = datetime.now()
            db.commit()
            
            logging.info(f"Email {email_log.email_id} sent successfully")
            
            await self.notification_system.send_email_notification(email_log, success=True)
            
        except Exception as e:
            email_log.status = EmailStatus.FAILED
            email_log.error_details = str(e)[:500]  # Truncate if too long
            email_log.modified_at = datetime.now()
            db.commit()
            
            logging.error(f"Email {email_log.email_id} failed: {str(e)}")
            
            if email_log.can_retry:
                await self._schedule_retry(email_log)
            else:
                await self.notification_system.send_email_notification(email_log, success=False)
    
    async def _schedule_retry(self, email_log: EmailLog):
        """Schedule email for retry after interval"""
        retry_delay = email_log.retry_interval
        
        async def retry_task():
            await asyncio.sleep(retry_delay)
            await self.retry_failed_email(email_log.email_id)
            logging.info(f"Retry for email {email_log.email_id} completed")
        
        asyncio.create_task(retry_task())
        logging.info(f"Scheduled retry for email {email_log.email_id} in {retry_delay} seconds")
    
    async def retry_failed_email(self, email_id: int):
        """Retry sending a specific failed email"""
        with self.get_db() as db:
            email_log = db.query(EmailLog).filter(EmailLog.email_id == email_id).first()
            
            if not email_log or not email_log.can_retry:
                return
            
            try:
                email_schema = EmailSchemaForRetry(
                    emails=email_log.to.split(','),
                    subject=email_log.subject,
                    body=email_log.body,
                    attachment=email_log.attachment,
                    metadatas={'email_id': email_id}
                )
                
                await self._send_message_with_retry(
                    MessageSchema(
                        subject=email_log.subject,
                        recipients=email_log.to.split(','),
                        body=email_log.body,
                        subtype=MessageType.plain,
                        attachments=[(email_log.attachment, None)] if email_log.attachment else []
                    ),
                    {'email_id': email_id}
                )
                
            except Exception as e:
                logging.error(f"Failed to retry email {email_id}: {str(e)}")
    
    async def process_pending_emails(self):
        """Process all pending/failed emails that need retrying"""
        with self.get_db() as db:
            pending_emails = db.query(EmailLog).filter(
                EmailLog.status.in_([EmailStatus.PENDING, EmailStatus.FAILED]),
                EmailLog.retry_count < EmailLog.max_retries
            ).all()
            logging.info(f"Processing {len(pending_emails)} pending/failed emails")
            
            for email_log in pending_emails:
                if (email_log.last_attempt is None or 
                    datetime.now() - email_log.last_attempt > timedelta(seconds=email_log.retry_interval)):
                    
                    await self.retry_failed_email(email_log.email_id)
            
            logging.info("Pending/failed emails processed")
    
    async def central_email_system(self, interval_seconds: int = 60):
        """Continuous email processing system"""
        logging.info("Starting central email processing system")
        
        while True:
            try:
                await self.process_pending_emails()
                logging.info("Pending/failed emails processed")
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logging.error(f"Error in central email system: {str(e)}")
                await asyncio.sleep(interval_seconds)


async def send_email_with_retry(email_request: EmailRequest, modify_by: Optional[int] = None):
    db_session_factory = SessionLocal  
    notification_system = EmailNotificationSystem(db_session_factory)
    email_service = CentralEmailService(db_session_factory, notification_system)
    
    email_id = await email_service.log_email(email_request, modify_by)
    
    email_schema = EmailSchemaForRetry(
        emails=email_request.emails,
        subject=email_request.subject,
        body=email_request.body,
        attachment=email_request.attachment,
        metadatas={'email_id': email_id}
    )
    
    background_tasks = BackgroundTasks()
    try:
        success = await email_service.send_email(email_schema, background_tasks)
    except Exception as e:
        logging.error(f"Email send failed: {str(e)}")
        success = False
    
    email_log = db_session_factory().query(EmailLog).filter(EmailLog.email_id == email_id).first()
    if not success:
        email_log.status = EmailStatus.FAILED
        email_log.modified_at = datetime.now()
        db_session_factory().commit()

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
        
        EmailLog(to=email, subject=subject, body=body, status=False, attachment=file.filename).save()
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