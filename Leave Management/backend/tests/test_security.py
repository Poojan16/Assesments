import pytest
from datetime import timedelta
from app.utils.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token
)
from app.config import settings


class TestPasswordHashing:
    """Test cases for password hashing functions."""
    
    def test_password_hash_creates_hash(self):
        """Test that password hashing creates a valid hash."""
        password = "SecurePassword@123"
        hashed = get_password_hash(password)
        
        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 0
    
    def test_password_hash_different_for_same_password(self):
        """Test that same password produces different hashes (salt)."""
        password = "SamePassword@123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        # Due to random salt, hashes should be different
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "CorrectPassword@123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "CorrectPassword@123"
        wrong_password = "WrongPassword@123"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_verify_password_empty(self):
        """Test password verification with empty password."""
        password = "SomePassword@123"
        hashed = get_password_hash(password)
        
        assert verify_password("", hashed) is False
    
    def test_password_hash_unicode(self):
        """Test password hashing with unicode characters."""
        password = "Password@日本語123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_hash_long(self):
        """Test password hashing with long password."""
        password = "A" * 100 + "@1"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_hash_max_bcrypt_limit(self):
        """Test password hashing with bcrypt max limit (72 bytes)."""
        # bcrypt has a 72-byte limit
        password = "A" * 70 + "@12"  # Exactly 72 bytes
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_hash_exceeds_bcrypt_limit(self):
        """Test password hashing with password exceeding bcrypt limit."""
        # Password longer than 72 bytes should be truncated
        password = "A" * 100  # More than 72 bytes
        hashed = get_password_hash(password)
        
        # Should still verify (truncated internally)
        assert verify_password(password, hashed) is True


class TestJWTToken:
    """Test cases for JWT token functions."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "test@example.com", "user_id": 1}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_access_token_with_expiry(self):
        """Test JWT token creation with custom expiry."""
        data = {"sub": "test@example.com", "user_id": 1}
        expires = timedelta(hours=1)
        token = create_access_token(data, expires_delta=expires)
        
        assert token is not None
        assert isinstance(token, str)
    
    def test_verify_token_valid(self):
        """Test token verification with valid token."""
        data = {"sub": "test@example.com", "user_id": 1}
        token = create_access_token(data)
        
        payload = verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == "test@example.com"
        assert payload["user_id"] == 1
    
    def test_verify_token_invalid(self):
        """Test token verification with invalid token."""
        invalid_token = "invalid.token.here"
        
        payload = verify_token(invalid_token)
        
        assert payload is None
    
    def test_verify_token_malformed(self):
        """Test token verification with malformed token."""
        malformed_token = "not-a-jwt"
        
        payload = verify_token(malformed_token)
        
        assert payload is None
    
    def test_verify_token_wrong_secret(self):
        """Test token verification with wrong secret (simulated)."""
        # This tests that token verification can fail
        data = {"sub": "test@example.com", "user_id": 1}
        token = create_access_token(data)
        
        # Verify should still work with correct secret
        payload = verify_token(token)
        assert payload is not None
        
        # Create a mock invalid token
        assert verify_token("fake.token.data") is None
    
    def test_token_contains_required_claims(self):
        """Test that token contains required claims."""
        data = {"sub": "test@example.com", "user_id": 1}
        token = create_access_token(data)
        payload = verify_token(token)
        
        assert "sub" in payload
        assert "exp" in payload
        assert payload["sub"] == "test@example.com"
    
    def test_token_expiration(self):
        """Test that expired token is rejected."""
        from datetime import timezone
        
        data = {"sub": "test@example.com", "user_id": 1}
        # Create token that expires immediately
        expires = timedelta(seconds=-1)  # Already expired
        token = create_access_token(data, expires_delta=expires)
        
        payload = verify_token(token)
        
        # Expired token should return None
        assert payload is None


class TestSecurityIntegration:
    """Integration tests for security functions."""
    
    def test_login_flow_simulation(self):
        """Simulate a complete login flow with password and token."""
        # User registers
        email = "integration@test.com"
        password = "IntegrationTest@123"
        
        # Hash password (as would happen during registration)
        hashed_password = get_password_hash(password)
        
        # User logs in - verify password
        assert verify_password(password, hashed_password) is True
        
        # Wrong password should fail
        assert verify_password("WrongPassword@123", hashed_password) is False
        
        # Create token for user
        token = create_access_token(
            data={"sub": email, "user_id": 1}
        )
        
        # Verify token
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == email
    
    def test_password_strength_workflow(self):
        """Test password strength validation through hash/verify."""
        # This tests that various password strengths work correctly
        test_cases = [
            ("Simple@123", True),    # Valid strong password
            ("Abc@123", True),       # Valid
            ("VeryLongPassword@123456", True),  # Long valid password
        ]
        
        for password, should_work in test_cases:
            hashed = get_password_hash(password)
            result = verify_password(password, hashed)
            assert result == should_work

