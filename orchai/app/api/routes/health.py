from fastapi import APIRouter
from datetime import datetime, timezone
from app.utils.mongo_client import get_db
from app.utils.config import settings

router = APIRouter()


@router.get("/health", summary="Sistem sağlık durumu")
async def health():
    """MongoDB bağlantısı ve servis durumunu kontrol eder."""
    db_ok = False
    db_error = None
    try:
        db = get_db()
        await db.command("ping")
        db_ok = True
    except Exception as e:
        db_error = str(e)

    return {
        "status": "healthy" if db_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "services": {
            "mongodb": {"status": "ok" if db_ok else "error", "error": db_error},
            "openai_model": settings.OPENAI_MODEL,
            "claude_model": settings.CLAUDE_MODEL,
        },
        "config": {
            "db_name": settings.MONGO_DB_NAME,
            "max_short_term_messages": settings.MAX_SHORT_TERM_MESSAGES,
            "long_term_summary_threshold": settings.LONG_TERM_SUMMARY_THRESHOLD,
        }
    }
