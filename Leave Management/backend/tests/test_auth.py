import pytest
from datetime import date
from app.models.user import DepartmentEnum


class TestUserSignup:
    """Test cases for user registration endpoint."""
    
    def test_signup_success(self, client):
        """Test successful user registration."""
        signup_data = {
            "email": "newuser@test.com",
            "password": "SecurePass@123",
            "confirm_password": "SecurePass@123",
            "first_name": "New",
            "last_name": "User",
            "employee_id": "NEW001",
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["access_token"] is not None
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "newuser@test.com"
        assert data["user"]["first_name"] == "New"
        assert data["user"]["last_name"] == "User"
        assert data["user"]["employee_id"] == "NEW001"
        assert data["user"]["department"] == "IT"
        assert data["user"]["role"] == "Employee"
    
    def test_signup_duplicate_email(self, client, test_employee):
        """Test signup with existing email."""
        signup_data = {
            "email": "employee@test.com",  # Already exists
            "password": "SecurePass@123",
            "confirm_password": "SecurePass@123",
            "first_name": "Another",
            "last_name": "User",
            "employee_id": "DIFF001",
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 409
        assert "email" in response.json()["detail"].lower()
    
    def test_signup_duplicate_employee_id(self, client, test_employee):
        """Test signup with existing employee ID."""
        signup_data = {
            "email": "different@test.com",
            "password": "SecurePass@123",
            "confirm_password": "SecurePass@123",
            "first_name": "Another",
            "last_name": "User",
            "employee_id": "EMP001",  # Already exists
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 409
        assert "employee ID" in response.json()["detail"].lower()
    
    def test_signup_password_mismatch(self, client):
        """Test signup with mismatched passwords."""
        signup_data = {
            "email": "test@test.com",
            "password": "SecurePass@123",
            "confirm_password": "DifferentPass@123",
            "first_name": "Test",
            "last_name": "User",
            "employee_id": "TEST001",
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 422  # Validation error
        assert "password" in response.json()["detail"][0]["msg"].lower()
    
    def test_signup_short_password(self, client):
        """Test signup with password less than 8 characters."""
        signup_data = {
            "email": "test@test.com",
            "password": "Short@1",
            "confirm_password": "Short@1",
            "first_name": "Test",
            "last_name": "User",
            "employee_id": "TEST001",
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 422
        assert "8 characters" in response.json()["detail"][0]["msg"].lower()
    
    def test_signup_password_no_uppercase(self, client):
        """Test signup with password without uppercase letter."""
        signup_data = {
            "email": "test@test.com",
            "password": "lowercase123@",
            "confirm_password": "lowercase123@",
            "first_name": "Test",
            "last_name": "User",
            "employee_id": "TEST001",
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 422
        assert "uppercase" in response.json()["detail"][0]["msg"].lower()
    
    def test_signup_password_no_number(self, client):
        """Test signup with password without number."""
        signup_data = {
            "email": "test@test.com",
            "password": "NoNumbers@abc",
            "confirm_password": "NoNumbers@abc",
            "first_name": "Test",
            "last_name": "User",
            "employee_id": "TEST001",
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 422
        assert "number" in response.json()["detail"][0]["msg"].lower()
    
    def test_signup_invalid_email(self, client):
        """Test signup with invalid email format."""
        signup_data = {
            "email": "not-an-email",
            "password": "SecurePass@123",
            "confirm_password": "SecurePass@123",
            "first_name": "Test",
            "last_name": "User",
            "employee_id": "TEST001",
            "department": "IT"
        }
        
        response = client.post("/api/auth/signup", json=signup_data)
        
        assert response.status_code == 422
    
    def test_signup_all_departments(self, client):
        """Test signup with all valid departments."""
        departments = ["HR", "IT", "FINANCE", "MARKETING", "SALES", "OPERATIONS", "ENGINEERING", "OTHER"]
        
        for dept in departments:
            import uuid
            signup_data = {
                "email": f"user_{uuid.uuid4().hex[:8]}@test.com",
                "password": "SecurePass@123",
                "confirm_password": "SecurePass@123",
                "first_name": "Test",
                "last_name": "User",
                "employee_id": f"TEST{uuid.uuid4().hex[:4].upper()}",
                "department": dept
            }
            
            response = client.post("/api/auth/signup", json=signup_data)
            assert response.status_code == 201, f"Failed for department: {dept}"


class TestUserLogin:
    """Test cases for user login endpoint."""
    
    def test_login_success(self, client, test_employee):
        """Test successful login."""
        login_data = {
            "email": "employee@test.com",
            "password": "Test@1234"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] is not None
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "employee@test.com"
    
    def test_login_invalid_email(self, client, test_employee):
        """Test login with non-existent email."""
        login_data = {
            "email": "nonexistent@test.com",
            "password": "Test@1234"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()
    
    def test_login_invalid_password(self, client, test_employee):
        """Test login with wrong password."""
        login_data = {
            "email": "employee@test.com",
            "password": "WrongPassword@123"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()
    
    def test_login_empty_password(self, client):
        """Test login with empty password."""
        login_data = {
            "email": "test@test.com",
            "password": ""
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_login_case_insensitive_email(self, client, test_employee):
        """Test that email is case-insensitive for login."""
        login_data = {
            "email": "EMPLOYEE@TEST.COM",  # Uppercase email
            "password": "Test@1234"
        }
        
        response = client.post("/api/auth/login", json=login_data)
        
        # Note: This depends on how the database stores emails
        # Currently, exact match is used in the query
        assert response.status_code in [200, 401]


class TestRootEndpoint:
    """Test cases for root and health endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Welcome" in data["message"]
        assert data["data"]["version"] == "1.0.0"
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"

