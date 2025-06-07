# Fitness Studio Booking API

A comprehensive booking system for fitness classes built with FastAPI, featuring timezone management, validation, error handling, and comprehensive testing.

## 🚀 Features

- **Complete REST API** with all required endpoints
- **Timezone Management** - Classes stored in UTC, displayed in any timezone
- **Input Validation** with Pydantic models
- **Error Handling** for edge cases and business logic
- **SQLite Database** with SQLAlchemy ORM
- **Comprehensive Logging** for debugging and monitoring
- **Unit Tests** with pytest covering all scenarios
- **API Documentation** with interactive Swagger UI

## 📋 Requirements

- Python 3.8+
- FastAPI
- SQLAlchemy
- Pydantic
- PyTZ for timezone handling

## 🛠️ Installation

1. **Clone or create the project files:**
   ```bash
   mkdir fitness_booking_api
   cd fitness_booking_api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API:**
   - API Base URL: `http://localhost:8000`
   - Interactive API Documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

## 📚 API Endpoints

### 🏥 Health Check
- **GET** `/` - Health check endpoint

### 🧘‍♀️ Classes
- **GET** `/classes` - Get all upcoming fitness classes
  - Optional query parameter: `timezone` (default: "Asia/Kolkata")
  - Returns classes with times converted to specified timezone

### 📝 Bookings
- **POST** `/book` - Book a spot in a fitness class
  - Request body: `{"class_id": int, "client_name": string, "client_email": email}`
  - Validates availability and prevents double booking

- **GET** `/bookings` - Get all book