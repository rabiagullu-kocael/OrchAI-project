from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import chat, health
from app.core.database import connect_db, disconnect_db
from app.utils.config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OrchAI başlatılıyor...")
    await connect_db()
    logger.info("MongoDB bağlantısı kuruldu.")
    yield
    await disconnect_db()
    logger.info("MongoDB bağlantısı kapatıldı.")


app = FastAPI(
    title="OrchAI - LLM Orchestration API",
    description="""
## OrchAI - Akıllı LLM Orkestrasyon Sistemi

Bu API, gelen kullanıcı isteklerini analiz ederek üç farklı akışa yönlendirir:

- **Direct**: Basit sorular için doğrudan OpenAI GPT-4o-mini
- **RAG**: Bilgi tabanı gerektiren sorular için MongoDB doküman arama + LLM
- **Web**: Güncel bilgi gerektiren sorular için web arama + LLM (Claude Sonnet)

### Özellikler
- 🧠 Kısa + uzun vadeli memory (MongoDB)
- 📊 Her istek için token/cost loglama
- 🔀 Otomatik routing (3 akış)
- 📝 Session yönetimi
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "OrchAI",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }
