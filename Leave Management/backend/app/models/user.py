from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func
from app.database.connection import Base
import enum


class DepartmentEnum(str, enum.Enum):
    HR = "HR"
    IT = "IT"
    FINANCE = "Finance"
    MARKETING = "Marketing"
    SALES = "Sales"
    OPERATIONS = "Operations"
    ENGINEERING = "Engineering"
    OTHER = "Other"


class UserRoleEnum(str, enum.Enum):
    EMPLOYEE = "Employee"
    MANAGER = "Manager"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    first_name = Column(String(100), nullable=False, index=True)
    last_name = Column(String(100), nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    employee_id = Column(String(50), unique=True, nullable=False, index=True)
    department = Column(Enum(DepartmentEnum), nullable=False)
    role = Column(Enum(UserRoleEnum), default=UserRoleEnum.EMPLOYEE, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, employee_id={self.employee_id}, role={self.role})>"

