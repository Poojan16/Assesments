import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database.connection import Base, get_db
from app.main import app
from app.models.user import User, UserRoleEnum, DepartmentEnum
from app.models.leave import Leave, LeaveTypeEnum, LeaveStatusEnum
from app.utils.security import get_password_hash, create_access_token
from datetime import timedelta
from app.config import settings


# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override the dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database session override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Reset dependency override
    app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def test_employee(db_session):
    """Create a test employee user."""
    user = User(
        email="employee@test.com",
        first_name="Test",
        last_name="Employee",
        employee_id="EMP001",
        department=DepartmentEnum.IT,
        role=UserRoleEnum.EMPLOYEE,
        hashed_password=get_password_hash("Test@1234")
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_manager(db_session):
    """Create a test manager user."""
    user = User(
        email="manager@test.com",
        first_name="Test",
        last_name="Manager",
        employee_id="MGR001",
        department=DepartmentEnum.IT,
        role=UserRoleEnum.MANAGER,
        hashed_password=get_password_hash("Manager@1234")
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_another_employee(db_session):
    """Create another test employee user."""
    user = User(
        email="another@test.com",
        first_name="Another",
        last_name="Employee",
        employee_id="EMP002",
        department=DepartmentEnum.HR,
        role=UserRoleEnum.EMPLOYEE,
        hashed_password=get_password_hash("Another@1234")
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def employee_token(test_employee):
    """Create a valid JWT token for employee."""
    return create_access_token(
        data={"sub": test_employee.email, "user_id": test_employee.id},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )


@pytest.fixture
def manager_token(test_manager):
    """Create a valid JWT token for manager."""
    return create_access_token(
        data={"sub": test_manager.email, "user_id": test_manager.id},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )


@pytest.fixture
def auth_headers_employee(employee_token):
    """Get authorization headers for employee."""
    return {"Authorization": f"Bearer {employee_token}"}


@pytest.fixture
def auth_headers_manager(manager_token):
    """Get authorization headers for manager."""
    return {"Authorization": f"Bearer {manager_token}"}


@pytest.fixture
def test_leave(db_session, test_employee):
    """Create a test leave request."""
    from datetime import date, timedelta
    
    leave = Leave(
        user_id=test_employee.id,
        leave_type=LeaveTypeEnum.CASUAL,
        start_date=date.today() + timedelta(days=1),
        end_date=date.today() + timedelta(days=3),
        reason="Test leave reason for testing",
        status=LeaveStatusEnum.PENDING
    )
    db_session.add(leave)
    db_session.commit()
    db_session.refresh(leave)
    return leave


@pytest.fixture
def approved_leave(db_session, test_employee):
    """Create an approved leave request."""
    from datetime import date, timedelta
    
    leave = Leave(
        user_id=test_employee.id,
        leave_type=LeaveTypeEnum.SICK,
        start_date=date.today() + timedelta(days=5),
        end_date=date.today() + timedelta(days=6),
        reason="Sick leave for testing purposes",
        status=LeaveStatusEnum.APPROVED
    )
    db_session.add(leave)
    db_session.commit()
    db_session.refresh(leave)
    return leave

