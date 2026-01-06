# Leave Management System - Backend API

A production-ready FastAPI backend with MySQL database integration for managing employee leave requests.

## Features

- ✅ User authentication (Sign up & Login)
- ✅ JWT token-based authentication
- ✅ Password hashing with bcrypt
- ✅ Employee Leave Management
  - Apply for leave (Casual/Sick)
  - View leave status (Pending/Approved/Rejected)
  - Filter and pagination support
- ✅ Comprehensive error handling
- ✅ User-friendly error messages
- ✅ Database connection pooling
- ✅ Input validation with Pydantic
- ✅ CORS configuration
- ✅ API documentation (Swagger/ReDoc)

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Database Setup

#### Install MySQL
Make sure MySQL is installed and running on your system.

#### Create Database
```sql
CREATE DATABASE leave_management;
```

#### Configure Database Connection
Copy the example environment file and update with your database credentials:

```bash
cp .env.example .env
```

Edit `.env` file with your MySQL credentials:
```
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_USER=root
DATABASE_PASSWORD=your_password
DATABASE_NAME=leave_management
SECRET_KEY=your-secret-key-here
```

### 3. Run the Application

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

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

#### Login
```
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@company.com",
  "password": "SecurePass123"
}
```

### Leave Management

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

#### Get My Leaves
```
GET /api/leaves/my-leaves?status=Pending&skip=0&limit=10
Authorization: Bearer <token>
```

**Query Parameters:**
- `status` (optional): Filter by status - `Pending`, `Approved`, or `Rejected`
- `leave_type` (optional): Filter by type - `Casual` or `Sick`
- `start_date` (optional): Filter by start date (from)
- `end_date` (optional): Filter by end date (to)
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum records to return, 1-100 (default: 100)

#### Get Leave by ID
```
GET /api/leaves/{leave_id}
Authorization: Bearer <token>
```

### Response Format

All responses follow this structure:

**Success Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@company.com",
    "first_name": "John",
    "last_name": "Doe",
    "employee_id": "EMP001",
    "department": "IT",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00"
  }
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

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py      # Database connection setup
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py            # User database model
│   │   └── leave.py           # Leave database model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py            # Pydantic schemas for user validation
│   │   └── leave.py           # Pydantic schemas for leave validation
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py            # Authentication routes
│   │   └── leave.py           # Leave management routes
│   └── utils/
│       ├── __init__.py
│       ├── security.py        # Password hashing & JWT
│       ├── exceptions.py       # Custom exception handlers
│       └── dependencies.py     # Authentication dependencies
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── setup_database.sql          # Database setup script
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Security Features

- ✅ Passwords are hashed using bcrypt
- ✅ JWT tokens for authentication
- ✅ Input validation and sanitization
- ✅ SQL injection protection (SQLAlchemy ORM)
- ✅ CORS configuration
- ✅ Environment variables for sensitive data

## Error Handling

The API provides user-friendly error messages for:
- Invalid credentials
- Duplicate email/employee ID
- Leave validation errors (date ranges, overlapping leaves)
- Validation errors
- Database errors
- Unexpected errors

## Production Considerations

1. **Change SECRET_KEY**: Use a strong, random secret key in production
2. **Database**: Use connection pooling and proper indexing
3. **Logging**: Implement proper logging (e.g., with Python's logging module)
4. **Rate Limiting**: Add rate limiting to prevent abuse
5. **HTTPS**: Always use HTTPS in production
6. **Environment Variables**: Never commit `.env` file to version control

## Testing

You can test the API using:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
- Postman or any HTTP client
- curl commands

Example curl for signup:
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

