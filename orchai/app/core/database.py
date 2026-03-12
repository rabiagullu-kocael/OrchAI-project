from app.utils.mongo_client import connect, disconnect, get_db
import logging

logger = logging.getLogger(__name__)


async def connect_db():
    await connect()
    await _ensure_indexes()


async def disconnect_db():
    await disconnect()


async def _ensure_indexes():
    """Koleksiyonlar için gerekli indexleri oluşturur."""
    db = get_db()
    try:
        # sessions: session_id unique index
        await db.sessions.create_index("session_id", unique=True)
        await db.sessions.create_index("created_at")

        # messages: session bazlı sorgular için
        await db.messages.create_index("session_id")
        await db.messages.create_index("created_at")
        await db.messages.create_index([("session_id", 1), ("created_at", 1)])

        # memory_store: session + type bazlı
        await db.memory_store.create_index("session_id")
        await db.memory_store.create_index([("session_id", 1), ("memory_type", 1)])
        await db.memory_store.create_index("updated_at")

        # documents: text search için
        await db.documents.create_index([("content", "text"), ("title", "text")])
        await db.documents.create_index("createdAt")

        # trace_logs: analiz sorguları için
        await db.trace_logs.create_index("session_id")
        await db.trace_logs.create_index("created_at")
        await db.trace_logs.create_index("route_type")

        logger.info("MongoDB indexleri oluşturuldu/doğrulandı.")
    except Exception as e:
        logger.warning(f"Index oluşturma hatası (devam ediliyor): {e}")
