# Leave Management System - Backend API

A production-ready FastAPI backend for managing employee leave requests. Supports both SQLite (for development) and MySQL (for production) databases.

## Features

- ✅ User authentication (Sign up & Login)
- ✅ JWT token-based authentication
- ✅ Password hashing with bcrypt
- ✅ Employee Leave Management
  - Apply for leave (Casual/Sick)
  - View leave status (Pending/Approved/Rejected)
  - Filter and pagination support
- ✅ Manager Features
  - View all team leave requests
  - Approve/Reject leave requests
  - Team performance tracking
- ✅ Comprehensive error handling
- ✅ User-friendly error messages
- ✅ Database connection pooling
- ✅ Input validation with Pydantic
- ✅ CORS configuration
- ✅ API documentation (Swagger/ReDoc)
- ✅ Ready-to-use demo data via seed script

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Seed Database with Demo Data (Recommended)

The `seed_data.py` script is a database seeding tool that automatically creates demo users and leave requests:

```bash
python seed_data.py
```

**What this script does:**
- Deletes all existing data from the database
- Creates demo users (managers and employees)
- Generates sample leave requests for each employee
- Populates the database with realistic test data

**Demo Users Created:**

| Role | Email | Password | Department |
|------|-------|----------|------------|
| Manager | manager@company.com | Manager123 | HR |
| Manager | tech.lead@company.com | Manager123 | Engineering |
| Employee | john.doe@company.com | Employee123 | IT |
| Employee | jane.smith@company.com | Employee123 | Engineering |
| Employee | bob.wilson@company.com | Employee123 | Marketing |
| Employee | alice.brown@company.com | Employee123 | Finance |
| Employee | charlie.davis@company.com | Employee123 | Sales |
| Employee | emma.taylor@company.com | Employee123 | HR |
| Employee | david.miller@company.com | Employee123 | Operations |
| Employee | sophia.garcia@company.com | Employee123 | IT |

### 3. Run the Application

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base URL:** `http://localhost:8000`
- **Swagger Docs:** `http://localhost:8000/api/docs`
- **ReDoc:** `http://localhost:8000/api/redoc`

## Database Setup

### Default: SQLite (Development)

The application uses SQLite by default for development. The database file (`leave_management.db`) is created automatically when you run the application or seed the database.

### Production: MySQL

To use MySQL instead of SQLite:

1. Create a `.env` file in the backend directory:

```bash
cp .env.example .env
```

2. Update `.env` with your MySQL credentials:

```env
# MySQL Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_USER=your_username
DATABASE_PASSWORD=your_password
DATABASE_NAME=leave_management

# Or use a full DATABASE_URL
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/leave_management

# JWT Configuration
SECRET_KEY=your-secure-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Configuration
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

3. Create the database in MySQL:

```sql
CREATE DATABASE leave_management CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

4. Run the application - tables will be created automatically.

## API Endpoints

### Authentication

#### Sign Up
```
POST /api/auth/signup
Content-Type: application/json

{
  "email": "user@company.com",
  "first_name": "John",
  "last_name": "Doe",
  "employee_id": "EMP001",
  "department": "IT",
  "password": "SecurePass123",
  "confirm_password": "SecurePass123"
}
```

**Response:**
```json
{
  "success": true,
  "message": "User registered successfully",
  "data": {
    "id": 1,
    "email": "user@company.com",
    "first_name": "John",
    "last_name": "Doe",
    "employee_id": "EMP001",
    "department": "IT",
    "role": "EMPLOYEE",
    "created_at": "2024-01-01T00:00:00"
  }
}
```

#### Login
```
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@company.com",
  "password": "SecurePass123"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "user": {
      "id": 1,
      "email": "user@company.com",
      "first_name": "John",
      "last_name": "Doe",
      "employee_id": "EMP001",
      "department": "IT",
      "role": "MANAGER"
    }
  }
}
```

---

### Leave Management (Employees)

**Note:** All leave endpoints require authentication. Include the JWT token in the Authorization header:
```
Authorization: Bearer <your_access_token>
```

#### Apply for Leave
```
POST /api/leaves/apply
Authorization: Bearer <token>
Content-Type: application/json

{
  "leave_type": "Casual",
  "start_date": "2024-02-01",
  "end_date": "2024-02-03",
  "reason": "Family vacation and personal time off"
}
```

**Leave Types:** `Casual` or `Sick`

**Response:**
```json
{
  "success": true,
  "message": "Leave application submitted successfully",
  "data": {
    "id": 1,
    "user_id": 1,
    "leave_type": "Casual",
    "start_date": "2024-02-01",
    "end_date": "2024-02-03",
    "reason": "Family vacation and personal time off",
    "status": "Pending",
    "created_at": "2024-01-15T10:30:00"
  }
}
```

#### Get My Leaves
```
GET /api/leaves/my-leaves?status=Pending&skip=0&limit=10
Authorization: Bearer <token>
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | All | Filter by status - `Pending`, `Approved`, or `Rejected` |
| `leave_type` | string | All | Filter by type - `Casual` or `Sick` |
| `start_date` | date | None | Filter by start date (from) |
| `end_date` | date | None | Filter by end date (to) |
| `skip` | int | 0 | Number of records to skip (pagination) |
| `limit` | int | 100 | Maximum records to return (1-100) |

#### Get Leave by ID
```
GET /api/leaves/{leave_id}
Authorization: Bearer <token>
```

---

### Manager Features

**Note:** Manager endpoints require authentication with a manager role.

#### Get All Team Leave Requests
```
GET /api/manager/leaves?status=Pending&skip=0&limit=10
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "message": "Leaves retrieved successfully",
  "data": {
    "leaves": [
      {
        "id": 1,
        "user": {
          "id": 2,
          "first_name": "John",
          "last_name": "Doe",
          "employee_id": "EMP001",
          "department": "IT"
        },
        "leave_type": "Casual",
        "start_date": "2024-02-01",
        "end_date": "2024-02-03",
        "reason": "Family vacation",
        "status": "Pending",
        "created_at": "2024-01-15T10:30:00"
      }
    ],
    "total": 10,
    "pending": 5,
    "approved": 3,
    "rejected": 2
  }
}
```

#### Approve Leave Request
```
PUT /api/manager/leaves/{leave_id}/approve
Authorization: Bearer <token>
Content-Type: application/json

{
  "remarks": "Approved. Have a great vacation!"
}
```

#### Reject Leave Request
```
PUT /api/manager/leaves/{leave_id}/reject
Authorization: Bearer <token>
Content-Type: application/json

{
  "remarks": "Rejected due to critical project deadline"
}
```

#### Get Team Performance Stats
```
GET /api/manager/stats
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "message": "Stats retrieved successfully",
  "data": {
    "total_employees": 8,
    "total_leaves": 45,
    "pending_leaves": 12,
    "approved_leaves": 28,
    "rejected_leaves": 5,
    "department_wise": {
      "IT": 15,
      "Engineering": 12,
      "Marketing": 8,
      "Finance": 5,
      "Sales": 5
    },
    "type_wise": {
      "Casual": 30,
      "Sick": 15
    }
  }
}
```

---

### Health Check
```
GET /api/health
```

**Response:**
```json
{
  "success": true,
  "message": "API is healthy",
  "data": {
    "status": "healthy",
    "service": "Leave Management System API",
    "version": "1.0.0"
  }
}
```

---

## Response Format

All responses follow this structure:

**Success Response:**
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": { ... }
}
```

**Error Response:**
```json
{
  "success": false,
  "message": "User-friendly error message",
  "data": null
}
```

---

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings and environment variables
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py      # Database connection setup (SQLAlchemy)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py            # User database model with roles and departments
│   │   └── leave.py           # Leave database model with types and status
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py            # Pydantic schemas for user validation
│   │   └── leave.py           # Pydantic schemas for leave validation
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py            # Authentication routes (signup, login)
│   │   ├── leave.py           # Leave management routes (employees)
│   │   └── manager.py         # Manager routes (approve, reject, stats)
│   └── utils/
│       ├── __init__.py
│       ├── security.py        # Password hashing & JWT token handling
│       ├── exceptions.py      # Custom exception handlers
│       ├── dependencies.py    # Authentication dependencies
│       └── ...
├── main.py                     # FastAPI application entry point
├── seed_data.py                # Database seeding script (creates demo data)
├── setup_database.sql          # Database setup script (MySQL)
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_HOST` | localhost | Database server host |
| `DATABASE_PORT` | 3306 | Database server port |
| `DATABASE_USER` | root | Database username |
| `DATABASE_PASSWORD` | - | Database password |
| `DATABASE_NAME` | leave_management | Database name |
| `DATABASE_URL` | - | Full database URL (overrides individual settings) |
| `SECRET_KEY` | - | JWT secret key for token signing |
| `ALGORITHM` | HS256 | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 30 | Token expiration time |
| `CORS_ORIGINS` | - | Comma-separated list of allowed CORS origins |

---

## Security Features

- ✅ Passwords are hashed using bcrypt
- ✅ JWT tokens for authentication
- ✅ Input validation and sanitization
- ✅ SQL injection protection (SQLAlchemy ORM)
- ✅ CORS configuration
- ✅ Environment variables for sensitive data
- ✅ Role-based access control (Employee vs Manager)

---

## Error Handling

The API provides user-friendly error messages for:
- Invalid credentials
- Duplicate email/employee ID
- Leave validation errors (date ranges, overlapping leaves)
- Unauthorized access
- Validation errors
- Database errors
- Unexpected errors

---

## Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test API with curl

**Signup:**
```bash
curl -X POST "http://localhost:8000/api/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@company.com",
    "first_name": "Test",
    "last_name": "User",
    "employee_id": "EMP001",
    "department": "IT",
    "password": "TestPass123",
    "confirm_password": "TestPass123"
  }'
```

**Login:**
```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "manager@company.com",
    "password": "Manager123"
  }'
```

**Apply for Leave (with token):**
```bash
curl -X POST "http://localhost:8000/api/leaves/apply" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your_access_token>" \
  -d '{
    "leave_type": "Casual",
    "start_date": "2024-02-01",
    "end_date": "2024-02-03",
    "reason": "Family vacation"
  }'
```

---

## Production Considerations

1. **Change SECRET_KEY**: Use a strong, random secret key in production
2. **Database**: Use MySQL or PostgreSQL with connection pooling
3. **Logging**: Implement proper logging (e.g., Python's logging module)
4. **Rate Limiting**: Add rate limiting to prevent abuse
5. **HTTPS**: Always use HTTPS in production
6. **Environment Variables**: Never commit `.env` file to version control
7. **CORS**: Configure CORS origins for your frontend domain

---

## License

MIT License

