# classes apis
from fastapi import APIRouter
from models import *
from database import *
from fastapi import HTTPException
from validation import ClassBase
from services import classes

router = APIRouter(
    prefix="/classes",
    tags=["Classes"],
)

# get all the classes
@router.get("/")
async def get_classes():
    try:
        allClass = await classes.get_classes()
        return allClass
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching classes")

# get class by id
@router.get("/id")
async def get_class(classId: int):
    try:
        cls = await classes.get_class(classId)
        return cls
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while fetching class")
    
# create a class
@router.post("/")
async def create_class(data: ClassBase):
    try:
        cls = await classes.create_class(data)
        return cls
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while creating class")

# update a class
@router.put("/")
async def update_class(classId: int, data: ClassBase):
    try:
        cls = await classes.update_class(classId, data)
        return cls
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while updating class")

# delete a class
@router.delete("/")
async def delete_class(classId: int):
    try:
        cls = await classes.delete_class(classId)
        return cls
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong while deleting class")