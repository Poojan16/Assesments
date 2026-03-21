#!/usr/bin/env python3
"""
Seed data script for Leave Management System.
Deletes existing data and recreates demo users and leave requests.
"""

from datetime import date, timedelta
import random
from sqlalchemy import text

from app.database.connection import SessionLocal, engine, Base
from app.models.user import User, DepartmentEnum, UserRoleEnum
from app.models.leave import Leave, LeaveTypeEnum, LeaveStatusEnum
from app.utils.security import get_password_hash

# Create tables if not exists
Base.metadata.create_all(bind=engine)


def seed_database():
    db = SessionLocal()

    try:
        print("🧹 Cleaning existing data...")

        # IMPORTANT: delete child table first
        db.execute(text("DELETE FROM leaves"))
        db.execute(text("DELETE FROM users"))
        db.commit()

        print("🌱 Seeding database with fresh demo data...")

        # --------------------
        # Create demo users
        # --------------------
        users_data = [
            # Managers
            {
                "email": "manager@company.com",
                "first_name": "Sarah",
                "last_name": "Johnson",
                "employee_id": "MGR001",
                "department": DepartmentEnum.HR,
                "role": UserRoleEnum.MANAGER,
                "password": "Manager123",
            },
            {
                "email": "tech.lead@company.com",
                "first_name": "Michael",
                "last_name": "Chen",
                "employee_id": "MGR002",
                "department": DepartmentEnum.ENGINEERING,
                "role": UserRoleEnum.MANAGER,
                "password": "Manager123",
            },

            # Employees
            {
                "email": "john.doe@company.com",
                "first_name": "John",
                "last_name": "Doe",
                "employee_id": "EMP001",
                "department": DepartmentEnum.IT,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
            {
                "email": "jane.smith@company.com",
                "first_name": "Jane",
                "last_name": "Smith",
                "employee_id": "EMP002",
                "department": DepartmentEnum.ENGINEERING,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
            {
                "email": "bob.wilson@company.com",
                "first_name": "Bob",
                "last_name": "Wilson",
                "employee_id": "EMP003",
                "department": DepartmentEnum.MARKETING,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
            {
                "email": "alice.brown@company.com",
                "first_name": "Alice",
                "last_name": "Brown",
                "employee_id": "EMP004",
                "department": DepartmentEnum.FINANCE,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
            {
                "email": "charlie.davis@company.com",
                "first_name": "Charlie",
                "last_name": "Davis",
                "employee_id": "EMP005",
                "department": DepartmentEnum.SALES,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
            {
                "email": "emma.taylor@company.com",
                "first_name": "Emma",
                "last_name": "Taylor",
                "employee_id": "EMP006",
                "department": DepartmentEnum.HR,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
            {
                "email": "david.miller@company.com",
                "first_name": "David",
                "last_name": "Miller",
                "employee_id": "EMP007",
                "department": DepartmentEnum.OPERATIONS,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
            {
                "email": "sophia.garcia@company.com",
                "first_name": "Sophia",
                "last_name": "Garcia",
                "employee_id": "EMP008",
                "department": DepartmentEnum.IT,
                "role": UserRoleEnum.EMPLOYEE,
                "password": "Employee123",
            },
        ]

        created_users = []
        for u in users_data:
            user = User(
                email=u["email"],
                first_name=u["first_name"],
                last_name=u["last_name"],
                employee_id=u["employee_id"],
                department=u["department"],
                role=u["role"],
                hashed_password=get_password_hash(u["password"]),
            )
            db.add(user)
            created_users.append(user)

        db.commit()

        for user in created_users:
            db.refresh(user)

        print(f"✅ Created {len(created_users)} users")

        # --------------------
        # Leave data
        # --------------------
        leave_reasons = {
            LeaveTypeEnum.CASUAL: [
                "Family vacation planned.",
                "Personal work to attend.",
                "Attending a wedding.",
                "Mental health break.",
                "Visiting parents.",
            ],
            LeaveTypeEnum.SICK: [
                "Fever and flu.",
                "Doctor appointment.",
                "Migraine issue.",
                "Food poisoning.",
                "Back pain.",
            ],
        }

        manager_remarks = {
            LeaveStatusEnum.APPROVED: [
                "Approved.",
                "Approved. Take care.",
                None,
            ],
            LeaveStatusEnum.REJECTED: [
                "Rejected due to workload.",
                "Team availability issue.",
            ],
        }

        employees = [u for u in created_users if u.role == UserRoleEnum.EMPLOYEE]
        today = date.today()

        leaves_created = 0

        for employee in employees:
            for _ in range(random.randint(3, 6)):
                leave_type = random.choice(list(leave_reasons.keys()))

                # ✅ START DATE IS NEVER IN THE PAST
                start_date = today + timedelta(days=random.randint(0, 30))
                duration = random.randint(1, 5)
                end_date = start_date + timedelta(days=duration - 1)

                status = random.choices(
                    [
                        LeaveStatusEnum.PENDING,
                        LeaveStatusEnum.APPROVED,
                        LeaveStatusEnum.REJECTED,
                    ],
                    weights=[50, 35, 15],
                )[0]

                remarks = (
                    random.choice(manager_remarks[status])
                    if status != LeaveStatusEnum.PENDING
                    else None
                )

                leave = Leave(
                    user_id=employee.id,
                    leave_type=leave_type,
                    start_date=start_date,
                    end_date=end_date,
                    reason=random.choice(leave_reasons[leave_type]),
                    status=status,
                    remarks=remarks,
                )

                db.add(leave)
                leaves_created += 1

        db.commit()

        print(f"✅ Created {leaves_created} leave requests")
        print("✨ Database seeded successfully!")

    except Exception as e:
        db.rollback()
        print(f"❌ Error seeding database: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
