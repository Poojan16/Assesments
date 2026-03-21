import redis.asyncio as redis

REDIS_URL = "redis://localhost:6379"

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True  # returns str instead of bytes
)



RESET_TOKEN_PREFIX = "reset_token:"

# Save token as unused
async def save_reset_token(jti: str, email: str, expiry_seconds: int):
    try:
        key = f"{RESET_TOKEN_PREFIX}{jti}"
        await redis_client.hmset(key, {"email": email, "is_used": "false"})
        await redis_client.expire(key, expiry_seconds)
        return True
    except Exception as e:
        print(e)
        return False


# Get token details
async def get_reset_token(jti: str):
    try:
        key = f"{RESET_TOKEN_PREFIX}{jti}"
        return await redis_client.hgetall(key)
    except Exception as e:
        print(e)
        return None


# Mark token as used
async def mark_token_used(jti: str):
    try:
        key = f"{RESET_TOKEN_PREFIX}{jti}"
        await redis_client.hset(key, "is_used", "true")
        return True
    except Exception as e:
        print(e)
        return False
