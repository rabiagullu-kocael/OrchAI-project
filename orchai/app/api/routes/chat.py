from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime, timezone
import uuid

from app.models.chat_models import (
    ChatRequest, ChatResponse,
    SessionCreateRequest, SessionResponse, SessionHistoryResponse,
    ScenarioTestRequest, RouteType
)
from app.services.chat_service import chat_service
from app.services.memory_service import memory_service
from app.services.trace_service import trace_service
from app.utils.mongo_client import get_db

router = APIRouter()

# ── Sohbet ────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Mesaj gönder")
async def chat(request: ChatRequest):
    """
    Kullanıcı mesajını işler. Otomatik olarak **DIRECT / RAG / WEB** akışlarından birine yönlendirir.

    - `session_id` boş bırakılırsa yeni oturum oluşturulur
    - `force_route` ile akış zorla seçilebilir (test için)
    """
    try:
        return await chat_service.process(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Senaryo Testi ─────────────────────────────────────────────────────────────

DEFAULT_SCENARIOS = [
    ("Merhaba! Nasılsın?", None),
    ("Chatbot nedir ve nasıl çalışır?", None),
    ("Yapay zeka ile makine öğrenmesi arasındaki fark nedir?", None),
    ("Transformer modeli ne demek?", None),
    ("2 + 2 kaç eder?", None),
    ("RAG sistemi nedir?", None),
    ("Bugünkü hava durumu nasıl?", None),
    ("NLP ne işe yarar?", None),
    ("Derin öğrenme nasıl çalışır?", None),
    ("Teşekkürler, görüşürüz!", None),
]


@router.post("/chat/test-scenarios", summary="10 senaryo testi çalıştır")
async def run_scenarios(request: ScenarioTestRequest):
    """
    Varsayılan veya özel 10 senaryoyu sırayla çalıştırır.
    Her senaryonun hangi akışa gittiğini ve sonucunu gösterir.
    """
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    results = []

    scenarios = request.scenarios if request.scenarios else [s[0] for s in DEFAULT_SCENARIOS]

    for i, message in enumerate(scenarios, 1):
        try:
            req = ChatRequest(message=message, session_id=session_id)
            resp = await chat_service.process(req)
            results.append({
                "scenario": i,
                "input": message,
                "route": resp.route_used.value,
                "model": resp.model_used,
                "answer_preview": resp.answer[:150] + ("..." if len(resp.answer) > 150 else ""),
                "tokens": resp.token_usage.total_tokens,
                "cost_usd": resp.token_usage.estimated_cost_usd,
                "processing_ms": resp.processing_time_ms,
                "rag_docs": len(resp.rag_contexts) if resp.rag_contexts else 0,
                "web_used": resp.web_results_used,
                "status": "ok"
            })
        except Exception as e:
            results.append({
                "scenario": i,
                "input": message,
                "status": "error",
                "error": str(e)
            })

    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_cost = sum(r.get("cost_usd", 0) for r in results)

    return {
        "session_id": session_id,
        "total_scenarios": len(results),
        "successful": sum(1 for r in results if r["status"] == "ok"),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "results": results
    }


# ── Session Yönetimi ──────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionResponse, summary="Yeni oturum oluştur")
async def create_session(request: SessionCreateRequest):
    db = get_db()
    session_id = request.session_id or f"session-{uuid.uuid4().hex[:12]}"

    existing = await db.sessions.find_one({"session_id": session_id})
    if existing:
        raise HTTPException(status_code=409, detail=f"Session zaten mevcut: {session_id}")

    now = datetime.now(timezone.utc)
    await db.sessions.insert_one({
        "session_id": session_id,
        "created_at": now,
        "message_count": 0,
        "metadata": request.metadata or {}
    })

    return SessionResponse(
        session_id=session_id,
        created_at=now,
        message_count=0,
        metadata=request.metadata or {}
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse, summary="Oturum bilgisi")
async def get_session(session_id: str):
    db = get_db()
    session = await db.sessions.find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session bulunamadı")

    count = await db.messages.count_documents({"session_id": session_id})
    return SessionResponse(
        session_id=session_id,
        created_at=session.get("created_at", datetime.now(timezone.utc)),
        message_count=count,
        metadata=session.get("metadata", {})
    )


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse, summary="Konuşma geçmişi")
async def get_history(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    db = get_db()
    cursor = db.messages.find(
        {"session_id": session_id}
    ).sort("created_at", 1).limit(limit)

    messages = []
    async for msg in cursor:
        msg["_id"] = str(msg["_id"])
        if "created_at" in msg and isinstance(msg["created_at"], datetime):
            msg["created_at"] = msg["created_at"].isoformat()
        messages.append(msg)

    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages)
    )


@router.delete("/sessions/{session_id}", summary="Oturum memory'sini temizle")
async def clear_session(session_id: str):
    await memory_service.clear_session_memory(session_id)
    return {"message": f"Session temizlendi: {session_id}"}


# ── Memory & Trace ────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/memory", summary="Memory istatistikleri")
async def get_memory_stats(session_id: str):
    stats = await memory_service.get_memory_stats(session_id)
    return stats


@router.get("/sessions/{session_id}/traces", summary="Token/cost logları")
async def get_traces(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    traces = await trace_service.get_session_traces(session_id, limit)
    return {"session_id": session_id, "traces": traces, "total": len(traces)}


@router.get("/sessions/{session_id}/cost-summary", summary="Maliyet özeti")
async def get_cost_summary(session_id: str):
    return await trace_service.get_cost_summary(session_id)
