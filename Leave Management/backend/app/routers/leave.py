from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_
from typing import Optional
from datetime import date
from app.database.connection import get_db
from app.models.leave import Leave, LeaveTypeEnum, LeaveStatusEnum
from app.models.user import User
from app.schemas.leave import LeaveCreate, LeaveResponse, LeaveListResponse
from app.utils.dependencies import get_current_user
from app.utils.exceptions import ErrorResponse

router = APIRouter(prefix="/api/leaves", tags=["Leaves"])


@router.post("/apply", response_model=LeaveResponse, status_code=status.HTTP_201_CREATED)
async def apply_for_leave(
    leave_data: LeaveCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Apply for a leave request.
    
    This endpoint allows an authenticated employee to submit a leave request
    with leave type (Casual/Sick), start date, end date, and reason.
    
    Args:
        leave_data: Leave request data
        current_user: Currently authenticated user (from JWT token)
        db: Database session
        
    Returns:
        LeaveResponse object with the created leave request
        
    Raises:
        HTTPException: If validation fails or database error occurs
    """
    try:
        # Validate date range (already validated in schema, but double-check)
        if leave_data.end_date < leave_data.start_date:
            raise ErrorResponse.bad_request(
                "End date must be after or equal to start date. Please check your dates and try again."
            )
        
        # Check for overlapping leave requests
        overlapping_leave = db.query(Leave).filter(
            Leave.user_id == current_user.id,
            Leave.status.in_([LeaveStatusEnum.PENDING, LeaveStatusEnum.APPROVED]),
            (
                (and_(Leave.start_date <= leave_data.start_date, leave_data.start_date <= Leave.end_date)) |
                (and_(Leave.start_date <= leave_data.end_date, leave_data.end_date <= Leave.end_date)) |
                (and_(leave_data.start_date <= Leave.start_date, Leave.end_date <= leave_data.end_date))
            )
        ).first()
        
        if overlapping_leave:
            raise ErrorResponse.conflict(
                f"You already have a {overlapping_leave.status.value.lower()} leave request "
                f"from {overlapping_leave.start_date} to {overlapping_leave.end_date}. "
                "Please choose different dates or wait for the current request to be processed."
            )
        
        # Create leave request
        new_leave = Leave(
            user_id=current_user.id,
            leave_type=leave_data.leave_type,
            start_date=leave_data.start_date,
            end_date=leave_data.end_date,
            reason=leave_data.reason.strip(),
            status=LeaveStatusEnum.PENDING
        )
        
        db.add(new_leave)
        db.commit()
        db.refresh(new_leave)
        
        return LeaveResponse.model_validate(new_leave)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except IntegrityError as e:
        db.rollback()
        error_msg = str(e.orig) if hasattr(e, 'orig') else "Database integrity error"
        raise ErrorResponse.bad_request(
            "Unable to create leave request. Please check your information and try again."
        )
    except Exception as e:
        db.rollback()
        # Log the error in production (you would use a proper logging library)
        print(f"Unexpected error during leave application: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while processing your leave request. Please try again later or contact support."
        )


@router.get("/my-leaves", response_model=LeaveListResponse, status_code=status.HTTP_200_OK)
async def get_my_leaves(
    status_filter: Optional[LeaveStatusEnum] = Query(None, description="Filter by leave status"),
    leave_type: Optional[LeaveTypeEnum] = Query(None, description="Filter by leave type"),
    start_date: Optional[date] = Query(None, description="Filter by start date (from)"),
    end_date: Optional[date] = Query(None, description="Filter by end date (to)"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of records to return"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all leave requests for the current authenticated user.
    
    This endpoint allows employees to view their leave requests with optional
    filtering by status, leave type, and date range. Results are paginated.
    
    Args:
        status_filter: Optional filter by leave status (Pending/Approved/Rejected)
        leave_type: Optional filter by leave type (Casual/Sick)
        start_date: Optional filter by start date (from)
        end_date: Optional filter by end date (to)
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return (1-100)
        current_user: Currently authenticated user (from JWT token)
        db: Database session
        
    Returns:
        LeaveListResponse with list of leave requests and total count
        
    Raises:
        HTTPException: If database error occurs
    """
    try:
        # Build query
        query = db.query(Leave).filter(Leave.user_id == current_user.id)
        
        # Apply filters
        if status_filter:
            query = query.filter(Leave.status == status_filter)
        
        if leave_type:
            query = query.filter(Leave.leave_type == leave_type)
        
        if start_date:
            query = query.filter(Leave.start_date >= start_date)
        
        if end_date:
            query = query.filter(Leave.end_date <= end_date)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination and ordering (newest first)
        leaves = query.order_by(Leave.created_at.desc()).offset(skip).limit(limit).all()
        
        return LeaveListResponse(
            success=True,
            message="Leaves retrieved successfully",
            data=[LeaveResponse.model_validate(leave).model_dump() for leave in leaves],
            total=total
        )
        
    except Exception as e:
        # Log the error in production
        print(f"Unexpected error while retrieving leaves: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while retrieving your leave requests. Please try again later or contact support."
        )


@router.get("/{leave_id}", response_model=LeaveResponse, status_code=status.HTTP_200_OK)
async def get_leave_by_id(
    leave_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific leave request by ID.
    
    This endpoint allows employees to view details of a specific leave request.
    Users can only view their own leave requests.
    
    Args:
        leave_id: ID of the leave request
        current_user: Currently authenticated user (from JWT token)
        db: Database session
        
    Returns:
        LeaveResponse object with leave request details
        
    Raises:
        HTTPException: If leave not found or user doesn't have access
    """
    try:
        leave = db.query(Leave).filter(
            Leave.id == leave_id,
            Leave.user_id == current_user.id
        ).first()
        
        if not leave:
            raise ErrorResponse.not_found(
                "Leave request not found. Please check the leave ID or ensure you have access to this request."
            )
        
        return LeaveResponse.model_validate(leave)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error while retrieving leave: {str(e)}")
        raise ErrorResponse.internal_error(
            "An unexpected error occurred while retrieving the leave request. Please try again later or contact support."
        )

