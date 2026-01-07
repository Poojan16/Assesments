# models.py
from sqlalchemy import (
    Column, Integer, String, Boolean, ForeignKey, Date, TIMESTAMP,
    Enum, Text, Enum as SQLEnum, JSON
)
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
import enum
from typing import List
from datetime import datetime


Base = declarative_base()

# ENUMS
class GenderEnum(str, enum.Enum):
    male = "male"
    female = "female"
    
class RoleEnum(str, enum.Enum):
    Admin = "A"
    Head_Teacher = "HT",
    Subject_Teacher = "ST",

class EmailStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"

# ROLES
class Role(Base):
    __tablename__ = "roles"

    roleId = Column(Integer, primary_key=True)
    roleName = Column(String(50), nullable=False)
    mark = Column(Enum(RoleEnum), nullable=False)
    roleDescription = Column(String(255))
    iModifyBy = Column(Integer, ForeignKey("users.userId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())


# GRADES
class Grade(Base):
    __tablename__ = "grades"

    gradeId = Column(Integer, primary_key=True)
    gradeLetter = Column(String(1), nullable=False)
    upperLimit = Column(Integer, nullable=False)
    lowerLimit = Column(Integer, nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())


# SCHOOLS
class School(Base):
    __tablename__ = "schools"

    schoolId = Column(Integer, primary_key=True)
    schoolName = Column(String(255), nullable=False)
    schoolEmail = Column(String(150), nullable=False, unique=True)
    primaryContactNo = Column(String(10), nullable=False)
    secondaryContactNo = Column(String(10), nullable=False)
    additionalContactNo = Column(String(10), nullable=False)
    address = Column(String(255), nullable=False)
    city = Column(String(50), nullable=False)
    state = Column(String(50), nullable=False)
    country = Column(String(50), nullable=False)
    pin = Column(Integer, nullable=False)
    board = Column(String(10), nullable=False)
    studentsPerClass = Column(Integer)
    maxClassLimit = Column(Integer)
    attachments = Column(String(255), nullable=True, default=None)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    cStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())
    established_year = Column(Integer, nullable=False)


class Class(Base):
    __tablename__ = "classes"

    classId = Column(Integer, primary_key=True)
    className = Column(String(50), nullable=False)
    schoolId = Column(Integer, ForeignKey("schools.schoolId"), nullable=False)
    limit = Column(Integer, nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())
    



# ---------------------------
# SUBJECTS
# ---------------------------
class Subject(Base):
    __tablename__ = "subjects"

    subjectId = Column(Integer, primary_key=True)
    subjectName = Column(String(50), nullable=False)
    subjectCode = Column(String(10), unique=True)
    classId = Column(Integer, ForeignKey("classes.classId"), nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())


# TEACHERS
class Teacher(Base):
    __tablename__ = "teachers"

    teacherId = Column(Integer, primary_key=True)
    teacherName = Column(String(50), nullable=False)
    teacherEmail = Column(String(150), nullable=False, unique=True)
    teacherContact = Column(String(10), nullable=False)
    emergencyContact = Column(String(10), nullable=False)
    onboardingDate = Column(Date, nullable=False)
    offboardingDate = Column(Date)
    active = Column(Boolean, default=True)
    address = Column(String(255), nullable=False)
    country = Column(String(50), nullable=False)
    city = Column(String(50), nullable=False)
    state = Column(String(50), nullable=False)
    pin = Column(Integer, nullable=False)
    photo = Column(String(255), nullable=True, default=None)
    PAN = Column(String(255), nullable=True, default=None)
    aadhar = Column(String(255), nullable=True, default=None)
    addressProof = Column(String(255), nullable=True, default=None)
    DL = Column(String(255), nullable=True, default=None)
    qualification = Column(String(50), nullable=False)
    role = Column(Integer, ForeignKey("roles.roleId"), nullable=False)
    schoolId = Column(Integer, ForeignKey("schools.schoolId"), nullable=False)
    remark = Column(String(255))
    DOB = Column(Date, nullable=False)
    gender = Column(Enum(GenderEnum))
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())

class TeacherSignature(Base):
    __tablename__ = "teachersignature"

    signatureId = Column(Integer, primary_key=True)
    teacherId = Column(Integer, ForeignKey("teachers.teacherId"), nullable=False)
    studentId = Column(Integer, ForeignKey("students.studentId"), nullable=False)
    score = Column(Integer)
    subjectId = Column(Integer, ForeignKey("subjects.subjectId"), nullable=False)
    signature = Column(String(255), nullable=True, default=None)
    signed_at = Column(TIMESTAMP, default=datetime.now())

# STUDENTS
class Student(Base):
    __tablename__ = "students"

    studentId = Column(Integer, primary_key=True)
    studentName = Column(String(50), nullable=False)
    DOB = Column(Date, nullable=False)
    gender = Column(Enum(GenderEnum))
    classId = Column(Integer, ForeignKey("classes.classId"), nullable=False)
    schoolId = Column(Integer, ForeignKey("schools.schoolId"), nullable=False)
    photo = Column(String(255), nullable=True, default=None)
    rollId = Column(String(50), nullable=False)
    parentId = Column(Integer, ForeignKey("parents.parentId"), nullable=False)
    address = Column(String(255), nullable=False)
    grade = Column(Integer, ForeignKey("grades.gradeId"), nullable=False)
    remark = Column(String(255))
    active = Column(Boolean, default=True)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())
    adhaar = Column(String(255), nullable=True, default=None)
    birthCertificate = Column(String(255), nullable=True, default=None)
    city = Column(String(50), nullable=False)
    state = Column(String(50), nullable=False)
    country = Column(String(50), nullable=False)    
    pin = Column(Integer, nullable=False)


# REPORTS
class StudentReports(Base):
    __tablename__ = "student_reports"

    reportId = Column(Integer, primary_key=True)
    studentId = Column(Integer, ForeignKey("students.studentId"), nullable=False)
    status = Column(Boolean, default=False)
    comments = Column(String(255))
    teacherId = Column(Integer, ForeignKey("teachers.teacherId"), nullable=False)
    report = Column(String(255), nullable=True, default=None)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(Date)
    created_at = Column(TIMESTAMP, default=datetime.now())


# STUDENT SCORES
class StudentScore(Base):
    __tablename__ = "student_scores"

    studentScoreId = Column(Integer, primary_key=True)
    studentId = Column(Integer, ForeignKey("students.studentId"), nullable=False)
    subjectId = Column(Integer, ForeignKey("subjects.subjectId"), nullable=False)
    score = Column(Integer)
    grade = Column(Integer, ForeignKey("grades.gradeId"), nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, default=datetime.now())


# REPORT REVIEW AUDIT
class ReportReviewAudit(Base):
    __tablename__ = "review_report_audit"

    reviewId = Column(Integer, primary_key=True)
    comments = Column(String(255))
    teacherId = Column(Integer, ForeignKey("teachers.teacherId"), nullable=False)
    attachment = Column(String(255), nullable=True, default=None)
    role = Column(Integer, ForeignKey("roles.roleId"), nullable=False)
    school = Column(Integer, ForeignKey("schools.schoolId"), nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())
    created_at = Column(TIMESTAMP, default=datetime.now())
    reportId = Column(Integer, ForeignKey("student_reports.reportId"), nullable=False)


# USERS
class User(Base):
    __tablename__ = "users"

    userId = Column(Integer, primary_key=True)
    userName = Column(String(50), nullable=False)
    userEmail = Column(String(150), nullable=False, unique=True)
    role = Column(Integer, ForeignKey("roles.roleId"), nullable=False)
    password = Column(String(255))
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())
    

class Login(Base):
    __tablename__ = "login"

    loginId = Column(Integer, primary_key=True)
    email = Column(String(150), nullable=False)
    password = Column(String(255))
    ip = Column(String(255))
    device = Column(String(255))
    os = Column(String(255))
    browser = Column(String(255))
    loginTime = Column(TIMESTAMP, default=datetime.now())

# PARENTS
class Parent(Base):
    __tablename__ = "parents"

    parentId = Column(Integer, primary_key=True)
    parentName = Column(String(50), nullable=False)
    parentEmail = Column(String(150), nullable=False, unique=True)
    parentContact = Column(String(10), nullable=False)
    parentAdhaar = Column(String(255), nullable=True, default=None)
    parentRelation = Column(String(50), nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())


# AUDIT
class Audit(Base):
    __tablename__ = "audit"

    auditId = Column(Integer, primary_key=True)
    reason = Column(String(255))
    changedBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=False)
    tableName = Column(String(50), nullable=False)
    fieldId = Column(Integer, nullable=False)
    fieldName = Column(String(50), nullable=False)
    oldValue = Column(String(255), nullable=False)
    newValue = Column(String(255), nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())


# Mapping Teacher by class and subject, multiple classes, multiple subjects
class TeacherClassSubject(Base):
    __tablename__ = "teacher_class_subject"

    mapId = Column(Integer, primary_key=True)
    teacherId = Column(Integer, ForeignKey("teachers.teacherId"), nullable=False)
    classId = Column(Integer, ForeignKey("classes.classId"), nullable=False)
    subjectId = Column(Integer, ForeignKey("subjects.subjectId"), nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())
    
class UserAudit(Base):
    __tablename__ = "useraudit"

    ua_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.userId"), nullable=False)
    activity = Column(String(255))
    sessionId = Column(Integer, ForeignKey("sessions.id"), nullable=True, default=None)
    created_at = Column(TIMESTAMP, default=datetime.now())
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    DModify = Column(TIMESTAMP, default=datetime.now())
    
class Fees(Base):
    __tablename__ = "fees"

    feesId = Column(Integer, primary_key=True)
    studentId = Column(Integer, ForeignKey("students.studentId"), nullable=False)
    amount = Column(Integer, nullable=False)
    isPaid = Column(Boolean, default=False)
    classId = Column(Integer, ForeignKey("classes.classId"), nullable=False)
    iModifyBy = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    iStatus = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, default=datetime.now())
    DModify = Column(TIMESTAMP, default=datetime.now())
    
class EmailPriority(enum.Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    
class EmailLog(Base):
    __tablename__ = "email_logs"

    email_id = Column(Integer, primary_key=True)
    to = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    body = Column(String(7000), nullable=False)
    attachment = Column(String(255), nullable=True, default=None)
    status = Column(SQLEnum(EmailStatus), default=EmailStatus.PENDING, index=True)
    priority = Column(SQLEnum(EmailPriority), default=EmailPriority.LOW)
    retry_count = Column(Integer, default=0, index=True)
    max_retries = Column(Integer, default=3)
    retry_interval = Column(Integer, default=300) 
    last_attempt = Column(TIMESTAMP, nullable=True)
    error_details = Column(String(500), nullable=True)
    modify_by = Column(Integer, ForeignKey("teachers.teacherId"), nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.now, index=True)
    modified_at = Column(TIMESTAMP, default=datetime.now, onupdate=datetime.now)
    metadatas = Column(JSON, nullable=True) 
    
    @property
    def can_retry(self):
        return (
            self.status in [EmailStatus.FAILED, EmailStatus.PENDING] and 
            self.retry_count < self.max_retries
        )
        
class SessionLog(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    userId = Column(Integer, ForeignKey("users.userId"))
    sessionId = Column(String(255), unique=True, index=True)
    deviceInfo = Column(JSON) 
    loginTime = Column(TIMESTAMP, default=datetime.now)
    lastActivity = Column(TIMESTAMP, default=datetime.now)
    expiresAt = Column(TIMESTAMP, default=datetime.now)
    isActive = Column(Boolean, default=True)
    
class APIENUM(enum.Enum):
    GET = 0
    POST = 1
    PUT = 2
    DELETE = 3

class RESPONSEENUM(enum.Enum):
    SUCCESS = 0
    FAILED = 1
    
class Logger(Base):
    __tablename__ = "logger"
    
    id = Column(Integer, primary_key=True, index=True)
    api_endpoint = Column(String(255), nullable=False)
    api_type = Column(SQLEnum(APIENUM), nullable=False)
    request = Column(String(255), nullable=False)
    statusCode = Column(Integer, nullable=False)
    status = Column(SQLEnum(RESPONSEENUM), nullable=False)
    response = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.now())