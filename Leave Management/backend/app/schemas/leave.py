from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from datetime import date, datetime
from app.models.leave import LeaveTypeEnum, LeaveStatusEnum


class LeaveBase(BaseModel):
    leave_type: LeaveTypeEnum
    start_date: date
    end_date: date
    reason: str = Field(..., min_length=10, max_length=1000)

    @field_validator('start_date')
    @classmethod
    def validate_start_date(cls, v):
        if v < date.today():
            raise ValueError('Start date cannot be in the past')
        return v

    @model_validator(mode='after')
    def validate_dates(self):
        if self.end_date < self.start_date:
            raise ValueError('End date must be after or equal to start date')
        return self

    @field_validator('reason')
    @classmethod
    def validate_reason(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Reason must be at least 10 characters long')
        return v


class LeaveCreate(LeaveBase):
    pass


class LeaveUpdate(BaseModel):
    status: LeaveStatusEnum
    remarks: Optional[str] = Field(None, max_length=1000)


class LeaveResponse(LeaveBase):
    id: int
    user_id: int
    status: LeaveStatusEnum
    remarks: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LeaveResponseWithUser(LeaveResponse):
    user: Optional[dict] = None

    class Config:
        from_attributes = True


class LeaveListResponse(BaseModel):
    success: bool = True
    message: str = "Leaves retrieved successfully"
    data: list[dict]  # Allow dict to include user info
    total: int

