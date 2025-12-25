# View File for the roles table
from models import Role
from validation import RoleUpdate, RoleBase
from fastapi import APIRouter, HTTPException
from database import SessionLocal


async def getAll():
    try:
        db = SessionLocal()
        roles = db.query(Role).all()
        return {
            "status_code": 200,
            "success": True,
            "data": roles,
            "message": "Roles fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def getById(roleId: int):
    try:
        db = SessionLocal()
        role = db.query(Role).filter(Role.roleId == roleId).first()
        return {
            "status_code": 200,
            "success": True,
            "data": role,
            "message": "Role fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def create_role(data: RoleBase):
    try:
        db = SessionLocal()
        role = Role(roleName=data.roleName, roleDescription=data.roleDescription, mark=data.mark)
        db.add(role)
        db.commit()
        db.refresh(role)
        return {
            "status_code": 200,
            "success": True,
            "data": role,
            "message": "Role created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def update_role(roleId: int, data: RoleUpdate):
    try:
        db = SessionLocal()
        role = db.query(Role).filter(Role.roleId == roleId).first()
        if(data.roleName):
            role.roleName = data.roleName
        if(data.roleDescription):
            role.roleDescription = data.roleDescription
        if(data.mark):
            role.mark = data.mark
        db.commit()
        db.refresh(role)
        return {
            "status_code": 200,
            "success": True,
            "data": role,
            "message": "Role updated successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def delete_role(roleId: int):
    try:
        db = SessionLocal()
        role = db.query(Role).filter(Role.roleId == roleId).first()
        db.delete(role)
        db.commit()
        return {
            "status_code": 200,
            "success": True,
            "message": "Role deleted successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
