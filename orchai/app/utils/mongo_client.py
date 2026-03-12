from motor.motor_asyncio import AsyncIOMotorClient
from app.utils.config import settings
import logging

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient = None


def get_client() -> AsyncIOMotorClient:
    return _client


def get_db():
    return _client[settings.MONGO_DB_NAME]


async def connect():
    global _client
    _client = AsyncIOMotorClient(settings.MONGO_URI)
    # Bağlantıyı test et
    await _client.admin.command("ping")
    logger.info(f"MongoDB bağlandı: {settings.MONGO_DB_NAME}")


async def disconnect():
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB bağlantısı kapatıldı.")
