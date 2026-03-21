import pytest
from datetime import date, timedelta
from app.models.leave import LeaveTypeEnum, LeaveStatusEnum
from app.models.user import UserRoleEnum


class TestManagerLeaves:
    """Test cases for manager leave management endpoints."""
    
    def test_get_all_leaves_as_manager(self, client, auth_headers_manager, test_leave):
        """Test manager can get all leave requests."""
        response = client.get(
            "/api/manager/leaves",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "total" in data
        assert isinstance(data["data"], list)
    
    def test_get_all_leaves_as_employee_forbidden(self, client, auth_headers_employee):
        """Test employee cannot access manager leave endpoint."""
        response = client.get(
            "/api/manager/leaves",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 403
        assert "manager" in response.json()["detail"].lower()
    
    def test_get_all_leaves_without_auth(self, client):
        """Test manager endpoint without authentication."""
        response = client.get("/api/manager/leaves")
        
        assert response.status_code == 401
    
    def test_get_all_leaves_filter_by_status(self, client, auth_headers_manager, test_leave):
        """Test filtering manager leaves by status."""
        response = client.get(
            "/api/manager/leaves?status=Pending",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        for leave in data["data"]:
            assert leave["status"] == "Pending"
    
    def test_get_all_leaves_filter_by_type(self, client, auth_headers_manager, test_leave):
        """Test filtering manager leaves by type."""
        response = client.get(
            "/api/manager/leaves?leave_type=Casual",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        for leave in data["data"]:
            assert leave["leave_type"] == "Casual"
    
    def test_get_all_leaves_pagination(self, client, auth_headers_manager):
        """Test pagination of manager leaves."""
        response = client.get(
            "/api/manager/leaves?skip=0&limit=5",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) <= 5


class TestManagerGetLeaveById:
    """Test cases for manager getting specific leave."""
    
    def test_get_leave_by_id_as_manager(self, client, auth_headers_manager, test_leave):
        """Test manager can get specific leave by ID."""
        response = client.get(
            f"/api/manager/leaves/{test_leave.id}",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_leave.id
    
    def test_get_leave_by_id_not_found(self, client, auth_headers_manager):
        """Test getting non-existent leave as manager."""
        response = client.get(
            "/api/manager/leaves/99999",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_leave_by_id_as_employee_forbidden(self, client, auth_headers_employee, test_leave):
        """Test employee cannot get manager-specific leave endpoint."""
        response = client.get(
            f"/api/manager/leaves/{test_leave.id}",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 403


class TestManagerApproveLeave:
    """Test cases for approving leave requests."""
    
    def test_approve_leave_success(self, client, auth_headers_manager, test_leave):
        """Test manager can approve a leave request."""
        action_data = {
            "status": "Approved",
            "remarks": "Approved by manager for testing"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{test_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Approved"
        assert data["remarks"] == "Approved by manager for testing"
    
    def test_approve_leave_without_remarks(self, client, auth_headers_manager, test_leave):
        """Test manager can approve without remarks."""
        action_data = {
            "status": "Approved"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{test_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "Approved"
    
    def test_approve_leave_not_found(self, client, auth_headers_manager):
        """Test approving non-existent leave."""
        action_data = {
            "status": "Approved",
            "remarks": "Test"
        }
        
        response = client.patch(
            "/api/manager/leaves/99999/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        assert response.status_code == 404
    
    def test_approve_leave_as_employee_forbidden(self, client, auth_headers_employee, test_leave):
        """Test employee cannot approve leave."""
        action_data = {
            "status": "Approved",
            "remarks": "Not allowed"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{test_leave.id}/action",
            json=action_data,
            headers=auth_headers_employee
        )
        
        assert response.status_code == 403


class TestManagerRejectLeave:
    """Test cases for rejecting leave requests."""
    
    def test_reject_leave_success(self, client, auth_headers_manager, test_leave):
        """Test manager can reject a leave request."""
        action_data = {
            "status": "Rejected",
            "remarks": "Rejected due to business needs"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{test_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Rejected"
        assert data["remarks"] == "Rejected due to business needs"
    
    def test_reject_leave_without_remarks(self, client, auth_headers_manager, test_leave):
        """Test manager can reject without remarks."""
        action_data = {
            "status": "Rejected"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{test_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "Rejected"
    
    def test_reject_already_approved_leave(self, client, auth_headers_manager, approved_leave):
        """Test rejecting an already approved leave."""
        action_data = {
            "status": "Rejected",
            "remarks": "Changed to rejected"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{approved_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        # Should work - allow changing from approved to rejected
        assert response.status_code == 200
        assert response.json()["status"] == "Rejected"


class TestManagerCannotApproveOwnLeave:
    """Test cases for manager cannot approve their own leave."""
    
    def test_manager_cannot_approve_own_leave(
        self, client, db_session, auth_headers_manager, test_manager
    ):
        """Test that manager cannot approve their own leave request."""
        from app.models.leave import Leave
        
        # Create leave for manager
        manager_leave = Leave(
            user_id=test_manager.id,
            leave_type=LeaveTypeEnum.CASUAL,
            start_date=date.today() + timedelta(days=1),
            end_date=date.today() + timedelta(days=2),
            reason="Manager's own leave request for testing purposes",
            status=LeaveStatusEnum.PENDING
        )
        db_session.add(manager_leave)
        db_session.commit()
        db_session.refresh(manager_leave)
        
        # Try to approve own leave
        action_data = {
            "status": "Approved",
            "remarks": "Trying to approve own leave"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{manager_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        # Should be forbidden
        assert response.status_code == 403
        assert "own" in response.json()["detail"].lower()


class TestManagerStats:
    """Test cases for manager statistics endpoint."""
    
    def test_get_stats_success(self, client, auth_headers_manager, test_leave):
        """Test manager can get leave statistics."""
        response = client.get(
            "/api/manager/stats",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "total" in data["data"]
        assert "pending" in data["data"]
        assert "approved" in data["data"]
        assert "rejected" in data["data"]
    
    def test_get_stats_as_employee_forbidden(self, client, auth_headers_employee):
        """Test employee cannot access stats endpoint."""
        response = client.get(
            "/api/manager/stats",
            headers=auth_headers_employee
        )
        
        assert response.status_code == 403
    
    def test_get_stats_without_auth(self, client):
        """Test stats endpoint without authentication."""
        response = client.get("/api/manager/stats")
        
        assert response.status_code == 401


class TestManagerActionEdgeCases:
    """Edge case tests for manager actions."""
    
    def test_action_with_invalid_status(self, client, auth_headers_manager, test_leave):
        """Test action with invalid status value."""
        action_data = {
            "status": "InvalidStatus",
            "remarks": "Test"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{test_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        assert response.status_code == 422
    
    def test_action_missing_status(self, client, auth_headers_manager, test_leave):
        """Test action without status field."""
        action_data = {
            "remarks": "Missing status"
        }
        
        response = client.patch(
            f"/api/manager/leaves/{test_leave.id}/action",
            json=action_data,
            headers=auth_headers_manager
        )
        
        assert response.status_code == 422
    
    def test_manager_view_all_employee_leaves(
        self, client, auth_headers_manager, db_session, test_employee, test_another_employee
    ):
        """Test manager can see leaves from different departments."""
        from app.models.leave import Leave
        
        # Create leaves for different employees
        leave1 = Leave(
            user_id=test_employee.id,
            leave_type=LeaveTypeEnum.CASUAL,
            start_date=date.today() + timedelta(days=1),
            end_date=date.today() + timedelta(days=2),
            reason="First employee leave for testing",
            status=LeaveStatusEnum.PENDING
        )
        leave2 = Leave(
            user_id=test_another_employee.id,
            leave_type=LeaveTypeEnum.SICK,
            start_date=date.today() + timedelta(days=3),
            end_date=date.today() + timedelta(days=4),
            reason="Second employee leave for testing",
            status=LeaveStatusEnum.PENDING
        )
        db_session.add_all([leave1, leave2])
        db_session.commit()
        
        response = client.get(
            "/api/manager/leaves",
            headers=auth_headers_manager
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 2

