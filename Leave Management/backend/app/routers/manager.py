from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Optional
from datetime import date
from app.database.connection import get_db
from app.models.leave import Leave, LeaveTypeEnum, LeaveStatusEnum
from app.models.user import User
from app.schemas.leave import LeaveResponse, LeaveListResponse, LeaveUpdate
from app.utils.dependencies import get_current_user, get_current_manager
from app.utils.exceptions import ErrorResponse

router = APIRouter(prefix="/api/manager", tags=["Manager"])


@router.get("/leaves", response_model=LeaveListResponse, status_code=status.HTTP_200_OK)
async def get_all_leave_requests(
    status_filter: Optional[LeaveStatusEnum] = Query(None, description="Filter by leave status"),
    leave_type: Optional[LeaveTypeEnum] = Query(None, description="Filter by leave type"),
    start_date: Optional[date] = Query(None, description="Filter by start date (from)"),
    end_date: Optional[date] = Query(None, description="Filter by end date (to)"),
    employee_id: Optional[str] = Query(None, description="Filter by employee ID"),
    department: Optional[str] = Query(None, description="Filter by department"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of records to return"),
    current_manager: User = Depends(get_current_manager),
    db: Session = Depends(get_db)
):
    """
    Get all leave requests (Manager only).
    
    This endpoint allows managers to view all leave requests submitted by employees
    with optional filtering by status, leave type, date range, employee, and department.
    
    Args:
        status_filter: Optional filter by leave status
        leave_type: Optional filter by leave type
        start_date: Optional filter by start date (from)
        end_date: Optional filter by end date (to)
        employee_id: Optional filter by employee ID
        department: Optional filter by department
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return (1-100)
        current_manager: Currently authenticated manager (from JWT token)
        db: Database session
        
    Returns:
        LeaveListResponse with list of leave requests and total count
        
    Raises:
        HTTPException: If user is not a manager or database error occurs
    """
    try:
        # Build query
        query = db.query(Leave)
        
        # Join with User to filter by employee/department
        query = query.join(User, Leave.user_id == User.id)
        
        # Apply filters
        if status_filter:
            query = query.filter(Leave.status == status_filter)
        
        if leave_type:
            query = query.filter(Leave.leave_type == leave_type)
        
        if start_date:
            query = query.filter(Leave.start_date >= start_date)
        
        if end_date:
            query = query.filter(Leave.end_date <= end_date)
        
        if employee_id:
            query = query.filter(User.employee_id.ilike(f"%{employee_id}%"))
        
        if department:
            query = query.filter(User.department == department)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination and ordering (newest first)
        leaves = query.order_by(Leave.created_at.desc()).offset(skip).limit(limit).all()
        
        # Create response with user information
        leave_responses = []
        for leave in leaves:
            leave_dict = LeaveResponse.model_validate(leave).model_dump()
            if leave.user:
                leave_dict['user'] = {
                    'id': leave.user.id,
                    'first_name': leave.user.first_name,
                    'last_name': leave.user.last_name,
                    'email': leave.user.email,
                    'employee_id': leave.user.employee_id,
                    'department': leave.user.department.value if hasattr(leave.user.department, 'value') else str(leave.user.department),
                }
            leave_responses.append(leave_dict)
        
        return LeaveListResponse(
            success=True,
            message="Leave requests retrieved successfully",
            data=leave_responses,
            total=total
        )
        
    except Exception as e:
        # Log the error in production
        print(f"Unexpected error while retrieving leave requests: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while retrieving leave requests. Please try again later or contact support."
        )


@router.get("/leaves/{leave_id}", response_model=LeaveResponse, status_code=status.HTTP_200_OK)
async def get_leave_request_by_id(
    leave_id: int,
    current_manager: User = Depends(get_current_manager),
    db: Session = Depends(get_db)
):
    """
    Get a specific leave request by ID (Manager only).
    
    Args:
        leave_id: ID of the leave request
        current_manager: Currently authenticated manager
        db: Database session
        
    Returns:
        LeaveResponse object with leave request details
        
    Raises:
        HTTPException: If leave not found or user is not a manager
    """
    try:
        leave = db.query(Leave).filter(Leave.id == leave_id).first()
        
        if not leave:
            raise ErrorResponse.not_found(
                "Leave request not found. Please check the leave ID."
            )
        
        return LeaveResponse.model_validate(leave)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error while retrieving leave request: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while retrieving the leave request. Please try again later or contact support."
        )


@router.patch("/leaves/{leave_id}/action", response_model=LeaveResponse, status_code=status.HTTP_200_OK)
async def action_leave_request(
    leave_id: int,
    action_data: LeaveUpdate,
    current_manager: User = Depends(get_current_manager),
    db: Session = Depends(get_db)
):
    """
    Approve or reject a leave request (Manager only).
    
    This endpoint allows managers to approve or reject leave requests
    with optional remarks. Employees cannot approve their own requests.
    
    Args:
        leave_id: ID of the leave request
        action_data: Action data (status and optional remarks)
        current_manager: Currently authenticated manager
        db: Database session
        
    Returns:
        LeaveResponse object with updated leave request
        
    Raises:
        HTTPException: If validation fails, leave not found, or user tries to approve own request
    """
    try:
        # Get the leave request
        leave = db.query(Leave).filter(Leave.id == leave_id).first()
        
        if not leave:
            raise ErrorResponse.not_found(
                "Leave request not found. Please check the leave ID."
            )
        
        # Check if manager is trying to approve their own leave
        if leave.user_id == current_manager.id:
            raise ErrorResponse.forbidden(
                "You cannot approve or reject your own leave request. Please contact another manager."
            )
        
        # Validate status change
        if leave.status == LeaveStatusEnum.APPROVED and action_data.status == LeaveStatusEnum.REJECTED:
            # Allow changing from approved to rejected
            pass
        elif leave.status == LeaveStatusEnum.REJECTED and action_data.status == LeaveStatusEnum.APPROVED:
            # Allow changing from rejected to approved
            pass
        elif leave.status != LeaveStatusEnum.PENDING:
            raise ErrorResponse.bad_request(
                f"This leave request has already been {leave.status.value.lower()}. "
                "You can only modify pending requests or change between approved/rejected."
            )
        
        # Update leave request
        leave.status = action_data.status
        if action_data.remarks:
            leave.remarks = action_data.remarks.strip()
        
        db.commit()
        db.refresh(leave)
        
        return LeaveResponse.model_validate(leave)
        
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as e:
        db.rollback()
        error_msg = str(e.orig) if hasattr(e, 'orig') else "Database integrity error"
        raise ErrorResponse.bad_request(
            "Unable to update leave request. Please check your information and try again."
        )
    except Exception as e:
        db.rollback()
        print(f"Unexpected error while updating leave request: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while processing the leave request. Please try again later or contact support."
        )


@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_manager_stats(
    current_manager: User = Depends(get_current_manager),
    db: Session = Depends(get_db)
):
    """
    Get statistics for all leave requests (Manager only).
    
    Returns:
        Dictionary with statistics about leave requests
    """
    try:
        total = db.query(Leave).count()
        pending = db.query(Leave).filter(Leave.status == LeaveStatusEnum.PENDING).count()
        approved = db.query(Leave).filter(Leave.status == LeaveStatusEnum.APPROVED).count()
        rejected = db.query(Leave).filter(Leave.status == LeaveStatusEnum.REJECTED).count()
        
        return {
            "success": True,
            "message": "Statistics retrieved successfully",
            "data": {
                "total": total,
                "pending": pending,
                "approved": approved,
                "rejected": rejected,
            }
        }
    except Exception as e:
        print(f"Unexpected error while retrieving statistics: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while retrieving statistics. Please try again later."
        )

