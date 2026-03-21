# Maintain the log of all apis and their responses 
from fastapi import HTTPException
from models import *
from validation import * 
from database import *
import math


async def get_logger(limit: int = 10, offset: int = 0):
    try:
        db = SessionLocal()
        total_records = db.query(Logger).count()
        logs = db.query(Logger).limit(limit).offset(offset).all()
        total_pages = math.ceil(total_records / limit) if total_records > 0 else 0
        return {
            "status_code": 200,
            "success": True,
            "data": logs,
            "message": "Logger fetched successfully",
            "pagination": {
                "page": offset,
                "page_size": limit,
                "total_records": total_records,
                "total_pages": total_pages
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unable to fetch logger data")
    finally:
        db.close()
        
async def add_log(logger: LoggerBase):
    try:
        db = SessionLocal()
        log = Logger(
            api_endpoint=logger.api_endpoint,
            api_type=logger.api_type,
            request=logger.request,
            statusCode=logger.statusCode,
            status=logger.status,
            response=logger.response,
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
        raise HTTPException(status_code=500, detail="Unable to add logger data" + str(e))
    finally:
        db.close()