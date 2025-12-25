import os
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import BackgroundTasks, UploadFile, File, Form, HTTPException
from fastapi_mail import FastMail
from fastapi_mail import MessageSchema, MessageType
from pydantic import BaseModel, EmailStr
from typing import List
from fastapi_mail import ConnectionConfig
from database import SessionLocal
from models import *

# env = Environment(
#     loader=FileSystemLoader("templates"),
#     autoescape=select_autoescape(["html", "xml"])
# )


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


SMTP_SERVER = "smtp.gmail.com"  # e.g., "smtp.gmail.com"
SMTP_PORT = 465  # Use 465 for SSL/TLS
EMAIL_ADDRESS = os.getenv("MAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("MAIL_PASSWORD")

class EmailSchema(BaseModel):
    email: EmailStr
    subject: str
    body: str

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
        try:
            background_tasks.add_task(fm.send_message, message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
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