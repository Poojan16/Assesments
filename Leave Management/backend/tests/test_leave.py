import pytest
from datetime import date, timedelta
from app.models.leave import LeaveTypeEnum, LeaveStatusEnum
from app.models.user import DepartmentEnum


class TestApplyLeave:
    """Test cases for leave application endpoint."""
    
    def test_apply_leave_success(self, client, auth_headers_employee):
        """Test successful leave application."""
        from datetime import date, timedelta
        
        leave_data = {
            "leave_type": "Casual",
            "start_date": (date.today() + timedelta(days=1)).isoformat(),
            "end_date": (date.today() + timedelta(days=3)).isoformat(),
            "reason": "Family vacation for testing purposes"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["leave_type"] == "Casual"
        assert data["status"] == "Pending"
        assert data["user_id"] is not None
    
    def test_apply_leave_without_auth(self, client):
        """Test leave application without authentication."""
        leave_data = {
            "leave_type": "Casual",
            "start_date": (date.today() + timedelta(days=1)).isoformat(),
            "end_date": (date.today() + timedelta(days=3)).isoformat(),
            "reason": "Testing without auth"
        }
        
        response = client.post("/api/leaves/apply", json=leave_data)
        
        assert response.status_code == 401
    
    def test_apply_leave_overlapping_dates(self, client, auth_headers_employee, test_leave):
        """Test overlapping leave dates are rejected."""
        leave_data = {
            "leave_type": "Sick",
            "start_date": test_leave.start_date.isoformat(),  # Same as existing
            "end_date": (test_leave.start_date + timedelta(days=2)).isoformat(),
            "reason": "Overlapping leave request"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 409
        assert "already have" in response.json()["detail"].lower() or "overlapping" in response.json()["detail"].lower()
    
    def test_apply_leave_end_before_start(self, client, auth_headers_employee):
        """Test that end date before start date is rejected."""
        leave_data = {
            "leave_type": "Casual",
            "start_date": (date.today() + timedelta(days=5)).isoformat(),
            "end_date": (date.today() + timedelta(days=1)).isoformat(),  # Before start
            "reason": "Invalid date range test"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_apply_leave_short_reason(self, client, auth_headers_employee):
        """Test that short reason is rejected."""
        leave_data = {
            "leave_type": "Casual",
            "start_date": (date.today() + timedelta(days=1)).isoformat(),
            "end_date": (date.today() + timedelta(days=2)).isoformat(),
            "reason": "Short"  # Less than 10 characters
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 422
    
    def test_apply_leave_past_start_date(self, client, auth_headers_employee):
        """Test that past start date is rejected."""
        leave_data = {
            "leave_type": "Casual",
            "start_date": (date.today() - timedelta(days=1)).isoformat(),  # Past date
            "end_date": date.today().isoformat(),
            "reason": "Past date leave request"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 422
    
    def test_apply_leave_sick_type(self, client, auth_headers_employee):
        """Test leave application with Sick type."""
        leave_data = {
            "leave_type": "Sick",
            "start_date": (date.today() + timedelta(days=1)).isoformat(),
            "end_date": (date.today() + timedelta(days=1)).isoformat(),
            "reason": "Sick leave for testing purposes"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 201
        assert response.json()["leave_type"] == "Sick"
    
    def test_apply_leave_same_day(self, client, auth_headers_employee):
        """Test leave application for single day."""
        future_date = date.today() + timedelta(days=1)
        leave_data = {
            "leave_type": "Casual",
            "start_date": future_date.isoformat(),
            "end_date": future_date.isoformat(),  # Same day
            "reason": "Single day leave for testing purposes"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 201


class TestGetMyLeaves:
    """Test cases for getting user's leaves."""
    
    def test_get_my_leaves_success(self, client, auth_headers_employee, test_leave):
        """Test getting user's leaves."""
        response = client.get(
            "/api/leaves/my-leaves",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "total" in data
        assert isinstance(data["data"], list)
    
    def test_get_my_leaves_without_auth(self, client):
        """Test getting leaves without authentication."""
        response = client.get("/api/leaves/my-leaves")
        
        assert response.status_code == 401
    
    def test_get_my_leaves_filter_by_status(self, client, auth_headers_employee, test_leave):
        """Test filtering leaves by status."""
        response = client.get(
            "/api/leaves/my-leaves?status=Pending",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 200
        data = response.json()
        for leave in data["data"]:
            assert leave["status"] == "Pending"
    
    def test_get_my_leaves_filter_by_type(self, client, auth_headers_employee, test_leave):
        """Test filtering leaves by type."""
        response = client.get(
            "/api/leaves/my-leaves?leave_type=Casual",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 200
        data = response.json()
        for leave in data["data"]:
            assert leave["leave_type"] == "Casual"
    
    def test_get_my_leaves_pagination(self, client, auth_headers_employee):
        """Test pagination of leaves."""
        response = client.get(
            "/api/leaves/my-leaves?skip=0&limit=5",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) <= 5
    
    def test_get_my_leaves_empty_result(self, client, auth_headers_manager):
        """Test getting leaves when none exist."""
        response = client.get(
            "/api/leaves/my-leaves",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 0


class TestGetLeaveById:
    """Test cases for getting specific leave by ID."""
    
    def test_get_leave_by_id_success(self, client, auth_headers_employee, test_leave):
        """Test getting a specific leave by ID."""
        response = client.get(
            f"/api/leaves/{test_leave.id}",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_leave.id
        assert data["user_id"] == test_leave.user_id
    
    def test_get_leave_by_id_not_found(self, client, auth_headers_employee):
        """Test getting non-existent leave."""
        response = client.get(
            "/api/leaves/99999",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_leave_by_id_unauthorized(self, client, auth_headers_employee, test_leave, db_session, test_another_employee):
        """Test that user cannot access another user's leave."""
        # Create a leave for another employee
        from app.models.leave import Leave
        other_leave = Leave(
            user_id=test_another_employee.id,
            leave_type=LeaveTypeEnum.CASUAL,
            start_date=date.today() + timedelta(days=1),
            end_date=date.today() + timedelta(days=2),
            reason="Other employee's leave",
            status=LeaveStatusEnum.PENDING
        )
        db_session.add(other_leave)
        db_session.commit()
        
        response = client.get(
            f"/api/leaves/{other_leave.id}",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 404
    
    def test_get_leave_by_id_without_auth(self, client, test_leave):
        """Test getting leave without authentication."""
        response = client.get(f"/api/leaves/{test_leave.id}")
        
        assert response.status_code == 401


class TestLeaveEdgeCases:
    """Edge case tests for leave functionality."""
    
    def test_apply_leave_future_date_range(self, client, auth_headers_employee):
        """Test applying leave for a long future period."""
        leave_data = {
            "leave_type": "Casual",
            "start_date": (date.today() + timedelta(days=30)).isoformat(),
            "end_date": (date.today() + timedelta(days=45)).isoformat(),
            "reason": "Extended vacation for testing purposes to meet minimum character requirement"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 201
    
    def test_apply_leave_missing_fields(self, client, auth_headers_employee):
        """Test leave application with missing required fields."""
        leave_data = {
            "leave_type": "Casual",
            "start_date": (date.today() + timedelta(days=1)).isoformat(),
            # Missing end_date and reason
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 422
    
    def test_apply_leave_invalid_type(self, client, auth_headers_employee):
        """Test leave application with invalid leave type."""
        leave_data = {
            "leave_type": "InvalidType",
            "start_date": (date.today() + timedelta(days=1)).isoformat(),
            "end_date": (date.today() + timedelta(days=2)).isoformat(),
            "reason": "Invalid type test for validation"
        }
        
        response = client.post(
            "/api/leaves/apply",
            json=leave_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 422

