# View File for the roles table
from models import Role
from validation import *
from fastapi import APIRouter, HTTPException
from database import SessionLocal
from services import roles as role


router = APIRouter(
    prefix="/roles",
    tags=["Roles"],
)

@router.get("/", response_model=RoleResponse)
async def getAll():
    try:
        roles = await role.getAll()
        return roles
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/id", response_model=RoleResponse)
async def getById(roleId: int):
    try:
        roleById = await role.getById(roleId)
        return roleById
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=RoleResponse)
async def create_role(data: RoleBase):
    try:
        createRole = await role.create_role(data)
        return createRole
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/", response_model=RoleResponse)
async def update_role(roleId: int, data: RoleUpdate):
    try:
        updateRole = await role.update_role(roleId, data)
        return updateRole
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/", response_model=RoleResponse)
async def delete_role(roleId: int):
    try:
        deleteRole = await role.delete_role(roleId)
        return deleteRole
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
