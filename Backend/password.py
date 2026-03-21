# two functions to hash and check password
import hashlib
import random

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(password, hashed_password):
    return hashlib.sha256(password.encode()).hexdigest() == hashed_password

def generate_password_reset_token(user_id):
    token = hashlib.sha256(str(user_id).encode()).hexdigest()
    return token

async def generate_forgot_password_link(user_id):
    token = generate_password_reset_token(user_id)
    return f"http://localhost:3000/reset-password/{token}"

