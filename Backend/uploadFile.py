from fastapi import FastAPI, UploadFile, File, HTTPException
from cryptography.fernet import Fernet, InvalidToken
import os
import secrets
from starlette.responses import StreamingResponse
import io
import tempfile
from dotenv import load_dotenv
import logging
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

# async def decrypt_file(filepath: str):
#     """
#     Decrypts and returns the content of an encrypted file.
#     """
#     # file path like this filepath = "/{folder_name}/encrypted.txt"
#     UPLOAD_FOLDER = os.path.join(os.path.dirname(filepath))
#     print(UPLOAD_FOLDER)
#     filename = os.path.basename(filepath)
#     print(filename)
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     print(file_path)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")
#     try:
#         with open(file_path, "rb") as f:
#             encrypted_content = f.read()
        
#         decrypted_content = cipher_suite.decrypt(encrypted_content)

#         file_extension = os.path.splitext(filename)[1].lower()
#         print(file_extension)
#         media_type = "application/octet-stream"  # Default for binary data
#         if file_extension == ".txt":
#             media_type = "text/plain"
#         elif file_extension == ".jpg" or file_extension == ".jpeg":
#             media_type = "image/jpeg"
#         elif file_extension == ".png":
#             media_type = "image/png"
#         elif file_extension == ".pdf":
#             media_type = "application/pdf"
            
#         print(media_type)
#         with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
#             decrypted_stream = io.BytesIO(decrypted_content)
#             async for chunk in decrypted_stream:
#                 temp_file.write(chunk)
#             temp_file.flush()
#             return StreamingResponse(temp_file.name, media_type=media_type)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Decryption error: {e}")


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







