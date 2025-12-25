from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import Optional, Annotated
from enum import Enum
from typing import List, Dict, Any
from fastapi import Form, File, UploadFile


# ----------------------------------------------------------
# ENUMS
# ----------------------------------------------------------
class GenderEnum(str, Enum):
    male = "male"
    female = "female"


# ----------------------------------------------------------
# ROLE SCHEMAS
# ----------------------------------------------------------
class RoleBase(BaseModel):
    roleName: str = 'Admin'
    mark: str = 'Admin'
    roleDescription: Optional[str] = 'Role Description'
    iStatus: Optional[bool] = False


class RoleUpdate(RoleBase):
    iModifyBy: int
    

class RoleResponse(BaseModel):
    status_code: int = 200
    success: bool = True
    data: List[RoleBase] | RoleBase
    message: str = "Roles fetched successfully"


# ----------------------------------------------------------
# GRADE SCHEMAS
# ----------------------------------------------------------
class GradeBase(BaseModel):
    gradeLetter: str
    upperLimit: int
    lowerLimit: int
    iStatus: Optional[bool] = False


class GradeUpdate(GradeBase):
    gradeLetter: Optional[str] = None
    upperLimit: Optional[int] = None
    lowerLimit: Optional[int] = None
    iStatus: Optional[bool] = False
    iModifyBy: int
    DModify: Optional[datetime]

class GradeResponse(BaseModel):
    grades: List[GradeBase]


# ----------------------------------------------------------
# SCHOOL SCHEMAS
# ----------------------------------------------------------

class SchoolBase(BaseModel):
    schoolName: str 
    schoolEmail: EmailStr 
    primaryContactNo: int 
    secondaryContactNo: int 
    additionalContactNo: int 
    address: str 
    city: str 
    state: str 
    country: str
    pin: int 
    board: str 
    studentsPerClass: Optional[int] 
    maxClassLimit: Optional[int]  
    attachments: Optional[str] = None
    iStatus: Optional[bool] = False
    established_year: int 


    
class SchoolCreate(SchoolBase):
    iModifyBy: int


class SchoolRead(SchoolBase):
    schoolId: int
    DModify: Optional[datetime]

    class Config:
        orm_mode = True


# ----------------------------------------------------------
# CLASS SCHEMAS
# ----------------------------------------------------------
class ClassBase(BaseModel):
    className: str
    schoolId: int
    limit: int
    iStatus: Optional[bool] = False


class ClassCreate(ClassBase):
    iModifyBy: int


class ClassRead(ClassBase):
    classId: int
    DModify: Optional[datetime]

    class Config:
        orm_mode = True


# ----------------------------------------------------------
# SUBJECT SCHEMAS
# ----------------------------------------------------------

class SubjectBase(BaseModel):
    subjectName: str
    subjectCode: str
    classId: int
    iStatus: Optional[bool] = False


class SubjectUpdate(SubjectBase):
    iModifyBy: int


# ----------------------------------------------------------
# REPORT SCHEMAS
# ----------------------------------------------------------
class ReportBase(BaseModel):
    studentId: int
    status: Optional[bool] = False
    comments: Optional[str]
    teacherId: int
    iStatus: Optional[bool] = False


class ReportCreate(ReportBase):
    iModifyBy: int


class ReportRead(ReportBase):
    reportId: int
    DModify: Optional[date]

    class Config:
        orm_mode = True
        
class TeacherBase(BaseModel):
    teacherName: str
    teacherEmail: EmailStr
    teacherContact: int
    emergencyContact: int
    onboardingDate: date
    address: str
    country: str
    city: str
    state: str
    pin: int
    qualification: str
    role: int
    schoolId: int
    DOB: date
    gender: GenderEnum
    iStatus: Optional[bool] = False 

class TeacherUpdate(TeacherBase):
    iModifyBy: int


class StudentBase(BaseModel):
    studentName: str = Form(...)
    address: str = Form(...)
    country: str = Form(...)
    city: str = Form(...)
    state: str = Form(...)
    pin: int = Form(...)
    DOB: str = Form(...)
    gender: GenderEnum = Form(...)
    iStatus: Optional[bool] = False
    classId: int = Form(...)
    schoolId: int = Form(...)
    grade: int = Form(...)
    parentEmail: EmailStr = Form(...)
    parentName: str = Form(...)

# ----------------------------------------------------------
# STUDENT SCORE SCHEMAS
# ----------------------------------------------------------
class StudentScoreBase(BaseModel):
    studentId: int
    subjectId: int
    score: Optional[int]
    iStatus: Optional[bool] = False


class StudentScoreUpdate(StudentScoreBase):
    iModifyBy: int



# ----------------------------------------------------------
# REPORT REVIEW AUDIT SCHEMAS
# ----------------------------------------------------------


class ReportReviewAuditBase(BaseModel):
    reviewId: int
    comments: Optional[str]
    teacherId: int
    attachment: UploadFile = File(...)
    iStatus: Optional[bool] = False


# ----------------------------------------------------------
# USER SCHEMAS
# ----------------------------------------------------------
class UserBase(BaseModel):
    teacher: TeacherBase
    password: str
    
class UserList(BaseModel):
    userName: str = 'John'
    userEmail: EmailStr = 'jYBwQ@example.com'
    role: int = 1
    iStatus: Optional[bool] = False
    password: str = '12345678'
    
class UserResponse(BaseModel):
    status_code: int = 200
    success: bool = True
    data: List[UserList] | UserList | None
    message: str = "Users fetched successfully"


class UserUpdate(UserBase):
    iModifyBy: int
    
class LoginUser(BaseModel):
    email: EmailStr
    password: str
        
class ClassMap(BaseModel):
    classId: str
    subjects: List[int]

class TeacherMap(BaseModel):
    email: str
    classes: List[ClassMap]

class MapTeacher(BaseModel):
    teachers: List[TeacherMap]
    
class ProvisionRequest(BaseModel):
    studentId: Optional[str] = None
    teacherId: Optional[str] = None
    provision: Optional[str] = None
    offBoardingDate: date | None = None
    changedBy: Optional[int] = None
