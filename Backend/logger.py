# Maintain the log of all apis and their responses 
from fastapi import HTTPException
from models import *
from validation import * 
from database import *


async def get_logger():
    try:
        db = SessionLocal()
        loggers = db.query(Logger).all()
        return {
            "status_code": 200,
            "success": True,
            "data": loggers,
            "message": "Loggers fetched successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unable to fetch logger data")
    finally:
        db.close()
        
async def add_log(logger: Logger):
    try:
        db = SessionLocal()
        log = Logger(
            api=logger.api,
            response=logger.response,
            apiEnum=logger.apiEnum,
            responseEnum=logger.responseEnum,
            created_at=datetime.now(),
        )
        db.add(log)
        db.commit()
        return {
            "status_code": 200,
            "success": True,
            "data": logger,
            "message": "Logger added successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unable to add logger data")
    finally:
        db.close()