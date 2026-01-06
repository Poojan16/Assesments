from fastapi import FastAPI, UploadFile, File, HTTPException
from cryptography.fernet import Fernet, InvalidToken
import os
import secrets
from starlette.responses import StreamingResponse
import io
import tempfile
from dotenv import load_dotenv
import logging
import requests
load_dotenv()


cipher_suite = Fernet(os.getenv("ENCRYPTION_KEY"))


async def upload_and_encrypt_file(file: UploadFile = File(...), folder_name: str = None):
    """
    Uploads a file, encrypts its content, saves it to a folder, and returns the relative path (URL) for database storage.
    """
    try:
        # Create the folder if it doesn't exist folder_name  = '/files/student/aadhar'
        UPLOAD_FOLDER = os.path.join(os.getcwd(), os.path.dirname(folder_name))
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        # Read the file content
        try:
            file_content = await file.read()
        except Exception as e:
            return f"File read error: {e}"

        # Encrypt the file content
        encrypted_content = cipher_suite.encrypt(file_content)
        logging.info(encrypted_content)

        # Generate a secure filename to prevent path traversal and collisions
        file_extension = os.path.splitext(file.filename)[1]
        secure_filename = f"{secrets.token_hex(16)}{file_extension}"
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename)

        # Save the encrypted file
        with open(file_path, "wb") as f:
            f.write(encrypted_content)

        file_url_for_db = f"{folder_name}{secure_filename}"
        return {"filename": secure_filename, "file_url": file_url_for_db}
    except Exception as e:
        print(e)
        return f"File processing error: {e}"

async def decrypt_file(filepath: str) -> bytes:
    """
    Decrypts and returns the raw bytes of an encrypted file.
    """
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(filepath, "rb") as f:
            encrypted_content = f.read()

        try:
            decrypted_bytes = cipher_suite.decrypt(encrypted_content)
        except InvalidToken:
            raise HTTPException(status_code=400, detail="Invalid encryption key or corrupted file")

        # RETURN RAW BYTES — DO NOT DECODE
        return decrypted_bytes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption error: {str(e)}")

async def FileScan(file: UploadFile = File(...)):
    try:
        url = os.getenv("VIRUSTOTAL_URL")
        apikey = os.getenv("VIRUSTOTAL_APIKEY")
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(await file.read())
            temp_file.flush()
            files = {'file': open(temp_file.name, 'rb')}
            headers = {'x-apikey': apikey}
            try:
                filesRes = requests.post(url, files=files, headers=headers)
                filesResult = filesRes.json()
                dataId = filesResult['data']['id']
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File scan error: {str(e)}")
            
            url = f"{os.getenv('VIRUSTOTAL_ANALYSES_URL')}{dataId}"
            headers = {'x-apikey': apikey}
            try:
                analysesRes = requests.get(url, headers=headers)
                analysesResult = analysesRes.json()
                status = analysesResult['data']['attributes']['status']
                malicious = analysesResult['data']['attributes']['stats']['malicious']
                suspicious = analysesResult['data']['attributes']['stats']['suspicious']
                harmless = analysesResult['data']['attributes']['stats']['harmless']
                if status == 'completed' and malicious == 0 and suspicious == 0 and harmless == 0:
                    return True
                else:
                    return {"status": status, "malicious": malicious, "suspicious": suspicious, "harmless": harmless}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"File scan error: {str(e)}")
        
    except Exception as e:
        raise HTTPException(stacompletedtus_code=500, detail=f"File read error: {str(e)}")




