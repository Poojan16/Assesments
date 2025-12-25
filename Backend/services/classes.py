# classes apis
from fastapi import APIRouter
from models import *
from database import *
from fastapi import HTTPException
from validation import ClassBase

# get all the classes
async def get_classes():
    try:
        db = SessionLocal()
        classes = db.query(Class).all()
        return {
            "status_code": 200,
            "success": True,
            "data": classes,
            "message": "Classes fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while fetching classes'))
    finally:
        db.close()

# get class by id
async def get_class(classId: int):
    try:
        db = SessionLocal()
        cls = db.query(Class).filter(Class.classId == classId).first()
        return {
            "status_code": 200,
            "success": True,
            "data": cls,
            "message": "Class fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while fetching class'))
    finally:
        db.close()

# create a class
async def create_class(data: ClassBase):
    try:
        db = SessionLocal()
        cls = Class(className=data.className, schoolId=data.schoolId, limit=data.limit)
        db.add(cls)
        db.commit()
        db.refresh(cls)
        return {
            "status_code": 200,
            "success": True,
            "data": cls,
            "message": "Class created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while creating class'))
    finally:
        db.close()

# update a class
async def update_class(classId: int, data: ClassBase):
    try:
        db = SessionLocal()
        cls = db.query(Class).filter(Class.classId == classId).first()
        cls.className = data.className
        cls.schoolId = data.schoolId
        cls.limit = data.limit
        db.commit()
        db.refresh(cls)
        return {
            "status_code": 200,
            "success": True,
            "data": cls,
            "message": "Class updated successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while updating class'))
    finally:
        db.close()

# delete a class
async def delete_class(classId: int):
    try:
        db = SessionLocal()
        cls = db.query(Class).filter(Class.classId == classId).first()
        db.delete(cls)
        db.commit()
        return {
            "status_code": 200,
            "success": True,
            "message": "Class deleted successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str('Something went wrong while deleting class'))
    finally:
        db.close()