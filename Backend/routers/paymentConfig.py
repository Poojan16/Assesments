# import datetime
# import asyncio
# import logging
# from typing import List, Optional, Dict, Any
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.triggers.date import DateTrigger
# from fastapi import BackgroundTasks, APIRouter, HTTPException, Depends
# from fastapi_mail import FastMail, MessageSchema, MessageType
# from sqlalchemy.orm import Session
# from sqlalchemy import select
# from pydantic import BaseModel, EmailStr
# from contextlib import asynccontextmanager
# import json
# from fastapi_mail import ConnectionConfig
# from models import School, Student, Parent, Fees, Grade
# from database import get_db, SessionLocal


# router = APIRouter(
#     prefix="/paymentConfig",
#     tags=["Payment Config"],
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# conf = ConnectionConfig(
#     MAIL_USERNAME="pujansoni.jcasp@gmail.com",
#     MAIL_PASSWORD="dyas iwyo rtjx jpcf",
#     MAIL_PORT=465,
#     MAIL_SERVER="smtp.gmail.com",
#     MAIL_STARTTLS=False,
#     MAIL_SSL_TLS=True, # Use True for SSL
#     USE_CREDENTIALS=True,   
#     VALIDATE_CERTS=True
# )

# class EmailSchema1(BaseModel):
#     email: List[EmailStr]
#     subject: str
#     body: str

# class BatchEmailRequest(BaseModel):
#     student_ids: List[int]
#     notification_type: str = '2 days before'
#     scheduled_datetime: Optional[datetime.datetime] = None
    
# class BatchEmailResponse(BaseModel):
#     total_students: int = 0
#     emails_sent: int = 0
#     failed_students: List[int] = 0
#     message: str = "Batch email sending initiated for students"
    
# class PaymentResponse(BaseModel):
#     status_code: int = 200
#     success: bool = True
#     data: BatchEmailResponse
#     message: str = "Emails have sent successfully"

# class StudentEmailData(BaseModel):
#     student_id: str
#     student_name: str
#     parent_name: str
#     parent_email: str
#     class_name: str
#     grade: str
#     total_fees: float
#     due_date: datetime.date
#     school_info: Dict[str, Any]

# class BatchEmailSender:    
#     def __init__(self):
#         self.fm = FastMail(conf)
#         self.batch_size = 500  
#         self.delay_between_batches = 1  
        
#     async def prepare_student_email_data(self, student_id: int, db: SessionLocal) -> Optional[StudentEmailData]:
#         try:
#             stmt = (
#                 select(
#                     Student,
#                     Parent,
#                     School,
#                     Fees
#                 )
#                 .join(Parent, Student.parentId == Parent.parentId)
#                 .join(School, Student.schoolId == School.schoolId)
#                 .outerjoin(Fees, Student.studentId == Fees.studentId)
#                 .where(Student.studentId == student_id)
#             )
            
#             result = db.execute(stmt).first()
#             if not result:
#                 return None
            
#             student, parent, school, fees = result
            
#             total_fees_due = fees.amount if fees and not fees.isPaid else 0
            
#             return StudentEmailData(
#                 student_id=student.rollId,
#                 student_name=student.studentName,
#                 parent_name=parent.parentName,
#                 parent_email=parent.parentEmail,
#                 class_name=f"Class {student.classId}", 
#                 grade=f"{'Not graded' if db.query(Grade).filter(Grade.gradeId == student.grade).first().gradeLetter == 'N' else db.query(Grade).filter(Grade.gradeId == student.grade).first().gradeLetter}",  # Modify based on your grade model
#                 total_fees=total_fees_due,
#                 due_date=datetime.date(datetime.datetime.now().year, 7, 7), 
#                 school_info={
#                     "name": school.schoolName,
#                     "email": school.schoolEmail,
#                     "phone": school.primaryContactNo,
#                     "address": school.address,
#                     "city": school.city,
#                     "state": school.state,
#                     "country": school.country,
#                     "pin": school.pin,
#                     "established_year": school.established_year
#                 }
#             )
#         except Exception as e:
#             logger.error(f"Error fetching data for student {student_id}: {e}")
#             return None
    
#     def generate_email_body(self, data: StudentEmailData, notification_type: str) -> str:        
#         styles = """
#             <style>
#                 body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
#                 .header { background-color: #4CAF50; color: white; padding: 20px; text-align: center; }
#                 .content { padding: 20px; }
#                 .details { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 15px 0; }
#                 .fee-amount { color: #d32f2f; font-size: 24px; font-weight: bold; }
#                 .due-date { color: #ff9800; font-weight: bold; }
#                 .school-info { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 15px 0; }
#                 .footer { background-color: #f1f1f1; padding: 15px; text-align: center; font-size: 12px; color: #666; }
#                 table { width: 100%; border-collapse: collapse; margin: 10px 0; }
#                 th { background-color: #4CAF50; color: white; padding: 10px; text-align: left; }
#                 td { padding: 10px; border-bottom: 1px solid #ddd; }
#                 .highlight { background-color: #fffacd; padding: 5px; border-radius: 3px; }
#             </style>
#         """
        
#         notification_messages = {
#             "10_days_before": f"Reminder: Fees due in 10 days",
#             "5_days_before": f"Reminder: Fees due in 5 days",
#             "3_days_before": f"Reminder: Fees due in 3 days",
#             "due_date_morning": f"Important: Fees are due today",
#             "due_date_noon": f"Final Reminder: Fees are due today",
#             "immediate": f"Fees Payment Reminder"
#         }
        
#         notification_msg = notification_messages.get(
#             notification_type, 
#             "Annual Fees Reminder"
#         )
        
#         html_body = f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             {styles}
#             <meta charset="UTF-8">
#             <title>Fee Payment Reminder</title>
#         </head>
#         <body>
#             <div class="header">
#                 <h1>📚 {data.school_info['name']}</h1>
#                 <h2>Fee Payment Reminder</h2>
#             </div>
            
#             <div class="content">
#                 <p>Dear <strong>{data.parent_name}</strong>,</p>
                
#                 <div class="details">
#                     <h3>📋 Student Information:</h3>
#                     <table>
#                         <tr>
#                             <th>Field</th>
#                             <th>Details</th>
#                         </tr>
#                         <tr>
#                             <td>Student Name:</td>
#                             <td><strong>{data.student_name}</strong></td>
#                         </tr>
#                         <tr>
#                             <td>Class:</td>
#                             <td>{data.class_name}</td>
#                         </tr>
#                         <tr>
#                             <td>Grade:</td>
#                             <td>{data.grade}</td>
#                         </tr>
#                         <tr>
#                             <td>Roll Number:</td>
#                             <td>#{data.student_id}</td>
#                         </tr>
#                     </table>
#                 </div>
                
#                 <div class="details">
#                     <h3>💰 Fee Details:</h3>
#                     <table>
#                         <tr>
#                             <th>Description</th>
#                             <th>Amount</th>
#                         </tr>
#                         <tr>
#                             <td>Total Fees Due:</td>
#                             <td class="fee-amount">₹{data.total_fees:,.2f}</td>
#                         </tr>
#                         <tr>
#                             <td>Due Date:</td>
#                             <td class="due-date">{data.due_date.strftime('%B %d, %Y')}</td>
#                         </tr>
#                         <tr>
#                             <td>Notification:</td>
#                             <td class="highlight">{notification_msg}</td>
#                         </tr>
#                     </table>
                    
#                     <p style="margin-top: 15px;">
#                         <strong>Important:</strong> Please ensure payment is made before the due date to avoid late fees.
#                     </p>
#                 </div>
                
#                 <div class="school-info">
#                     <h3>🏫 School Information:</h3>
#                     <p><strong>School:</strong> {data.school_info['name']}</p>
#                     <p><strong>Address:</strong> {data.school_info['address']}, {data.school_info['city']}, 
#                        {data.school_info['state']} - {data.school_info['pin']}</p>
#                     <p><strong>Contact:</strong> {data.school_info['phone']} | Email: {data.school_info['email']}</p>
#                     <p><strong>Established:</strong> {data.school_info['established_year']}</p>
#                 </div>
                
#                 <div style="margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-radius: 5px;">
#                     <h4>📝 Payment Instructions:</h4>
#                     <ol>
#                         <li>Log in to the parent portal at [Portal URL]</li>
#                         <li>Navigate to the Fees section</li>
#                         <li>Select the outstanding amount and proceed to payment</li>
#                         <li>You can pay via UPI, Net Banking, or Credit/Debit Card</li>
#                         <li>Save the payment receipt for future reference</li>
#                     </ol>
                    
#                     <p style="margin-top: 10px;">
#                         <strong>Need Help?</strong> Contact our accounts department at 
#                         {data.school_info['phone']} during school hours (9 AM - 4 PM).
#                     </p>
#                 </div>
#             </div>
            
#             <div class="footer">
#                 <p>This is an automated message. Please do not reply to this email.</p>
#                 <p>© {datetime.datetime.now().year} {data.school_info['name']}. All rights reserved.</p>
#                 <p>If you have already made the payment, please ignore this reminder.</p>
#             </div>
#         </body>
#         </html>
#         """
        
#         return html_body
    
#     async def send_batch_emails(
#         self, 
#         student_ids: List[int], 
#         notification_type: str,
#         db: SessionLocal,
#         background_tasks: BackgroundTasks
#     ) -> Dict[str, Any]:
#         """Send emails to multiple students in optimized batches"""
        
#         total_students = len(student_ids)
#         logger.info(f"Starting batch email sending for {total_students} students")
        
#         batches = [
#             student_ids[i:i + self.batch_size] 
#             for i in range(0, total_students, self.batch_size)
#         ]
        
#         total_sent = 0
#         failed_students = []
        
#         for batch_num, batch in enumerate(batches, 1):
#             logger.info(f"Processing batch {batch_num}/{len(batches)} with {len(batch)} students")
            
#             email_data_list = []
#             recipients = []
            
#             for student_id in batch:
#                 data = await self.prepare_student_email_data(student_id, db)
#                 if data:
#                     email_data_list.append((student_id, data))
#                     recipients.append(data.parent_email)
            
#             if not email_data_list:
#                 continue
            
#             subject = f"Fee Payment Reminder - {notification_type.replace('_', ' ').title()}"
            
#             background_tasks.add_task(
#                 self._send_single_batch_emails,
#                 email_data_list,
#                 subject,
#                 notification_type
#             )
            
#             total_sent += len(email_data_list)
            
#             if batch_num < len(batches):
#                 await asyncio.sleep(self.delay_between_batches)
        
#         return {
#             "total_students": total_students,
#             "emails_sent": total_sent,
#             "failed_students": failed_students,
#             "message": f"Batch email sending initiated for {total_sent} students"
#         }
    
#     async def _send_single_batch_emails(
#         self, 
#         email_data_list: List[tuple], 
#         subject: str,
#         notification_type: str
#     ):
#         try:
#             messages = []
#             logger.info(f"Sending batch emails for {len(email_data_list)} students")
            
#             for student_id, data in email_data_list:
#                 # change parent email to pujansoni.jcasp@gmail.com
#                 data.parent_email = "pujansoni.jcasp@gmail.com"
#                 body = self.generate_email_body(data, notification_type)
                
#                 message = MessageSchema(
#                     subject=f"{subject} - {data.student_name}",
#                     recipients=[data.parent_email],
#                     body=body,
#                     subtype=MessageType.html
#                 )
#                 messages.append(message)
            
#             for message in messages:
#                 print(f"Sending email to {message.recipients}")
#                 await self.fm.send_message(message)
            
#             logger.info(f"Successfully sent {len(messages)} emails in batch")
            
#         except Exception as e:
#             logger.error(f"Error sending batch emails: {e}")

# email_sender = BatchEmailSender()

# def schedule_annual_notifications(
#     scheduler: AsyncIOScheduler,
#     current_date: datetime.date,
#     student_ids: List[int],
#     db: SessionLocal,
#     notification_times: Optional[List[str]] = None
# ):
    
#     fee_due_date = datetime.date(current_date.year, 7, 7)
    
#     if current_date > fee_due_date:
#         fee_due_date = datetime.date(current_date.year + 1, 7, 7)
    
#     notification_schedule = [
#         ("10_days_before", fee_due_date - datetime.timedelta(days=10), datetime.time(12, 0)),
#         ("5_days_before", fee_due_date - datetime.timedelta(days=5), datetime.time(12, 0)),
#         ("3_days_before", fee_due_date - datetime.timedelta(days=3), datetime.time(12, 0)),
#         ("due_date_morning", fee_due_date, datetime.time(9, 0)),
#         ("due_date_noon", fee_due_date, datetime.time(12, 0)),
#         # Test: send in 2 minutes
#         ("test_immediate", datetime.datetime.now() + datetime.timedelta(minutes=1),
#          (datetime.datetime.now() + datetime.timedelta(minutes=1)).time()),
#     ]
    
#     # if notification_times:
#     #     pass
    
#     for notification_type, date_obj, time_obj in notification_schedule:
#         scheduled_datetime = datetime.datetime.combine(date_obj, time_obj)
        
#         if scheduled_datetime > datetime.datetime.now():
            
#             async def scheduled_task(
#                 student_ids=student_ids,
#                 notification_type=notification_type,
#                 db=db
#             ):
                
#                 logger.info(f"Executing scheduled email task: {notification_type} for {len(student_ids)} students")
                
                
#                 background_tasks = BackgroundTasks()
                
#                 result = await email_sender.send_batch_emails(
#                     student_ids=student_ids,
#                     notification_type=notification_type,
#                     db=db,
#                     background_tasks=background_tasks
#                 )
                
#                 logger.info(f"Scheduled task {notification_type} completed: {result}")
            
           
#             scheduler.add_job(
#                 scheduled_task,
#                 trigger=DateTrigger(run_date=scheduled_datetime),
#                 id=f"batch-email-{notification_type}-{int(scheduled_datetime.timestamp())}",
#                 replace_existing=True,
#                 misfire_grace_time=300,
#                 coalesce=True
#             )
            
#             logger.info(f"Scheduled {notification_type} for {len(student_ids)} "
#                        f"students at {scheduled_datetime}")

# @router.post("/send-immediate-batch", response_model=PaymentResponse)
# async def send_immediate_batch(
#     request: BatchEmailRequest,
#     background_tasks: BackgroundTasks,
#     db: SessionLocal = Depends(get_db)
# ):
#     try:
#         result = await email_sender.send_batch_emails(
#             student_ids=request.student_ids,
#             notification_type=request.notification_type,
#             db=db,
#             background_tasks=background_tasks
#         )
        
#         return {
#             "status_code": 200,
#             "success": True,
#             "data": result,
#             "message": "Batch email sending initiated"
#         }
        
#     except Exception as e:
#         logger.error(f"Error in immediate batch sending: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# router.on_event("shutdown")
# async def shutdown_event():
#     logger.info("Shutting down email scheduler")
#     await email_sender.shutdown()


from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from services.paymentConfig import send_immediate_batch,BatchEmailRequest
from database import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Router
router = APIRouter(
    prefix="/paymentConfig",
    tags=["Payment Config"],
)

@router.post("/send-immediate-batch")
async def send_immediate(
    request: BatchEmailRequest,
    background_tasks: BackgroundTasks,
):
    try:
        result = await send_immediate_batch(
            request=request,
            background_tasks=background_tasks,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in immediate batch sending: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )