from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey, Text, Date
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database.connection import Base
import enum


class LeaveTypeEnum(str, enum.Enum):
    CASUAL = "Casual"
    SICK = "Sick"


class LeaveStatusEnum(str, enum.Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"


class Leave(Base):
    __tablename__ = "leaves"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    leave_type = Column(Enum(LeaveTypeEnum), nullable=False)
    start_date = Column(Date, nullable=False, index=True)
    end_date = Column(Date, nullable=False, index=True)
    reason = Column(Text, nullable=False)
    status = Column(Enum(LeaveStatusEnum), default=LeaveStatusEnum.PENDING, nullable=False, index=True)
    remarks = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship with User
    user = relationship("User", backref="leaves")

    def __repr__(self):
        return f"<Leave(id={self.id}, user_id={self.user_id}, leave_type={self.leave_type}, status={self.status})>"

