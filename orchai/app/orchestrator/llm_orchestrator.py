"""
LLM Orchestrator - 3 Akışı Koordine Eden Ana Motor

Akışlar:
--------
DIRECT  → memory context + OpenAI GPT-4o-mini
RAG     → MongoDB doküman arama + context + OpenAI GPT-4o-mini
WEB     → DuckDuckGo web arama + context + Claude Sonnet
"""

import logging
import time
import uuid
from datetime import datetime, timezone

from app.models.chat_models import (
    ChatRequest, ChatResponse, RouteType, TokenUsage
)
from app.routing.router_engine import router_engine
from app.services.llm_service import llm_service
from app.services.memory_service import memory_service
from app.services.rag_service import rag_service
from app.services.web_service import web_service
from app.services.trace_service import trace_service
from app.utils.config import settings
from app.utils.mongo_client import get_db

logger = logging.getLogger(__name__)


class LLMOrchestrator:

    async def handle(self, request: ChatRequest) -> ChatResponse:
        start_time = time.time()

        # 1. Session yönetimi
        session_id = request.session_id or f"session-{uuid.uuid4().hex[:12]}"
        await self._ensure_session(session_id)

        # 2. Routing kararı
        route, routing_reason = await router_engine.determine_route(
            request.message, request.force_route
        )
        logger.info(f"[{session_id}] Route: {route.value} | {routing_reason}")

        # 3. Memory context al
        memory_ctx = await memory_service.get_memory_context(session_id)

        # 4. Akışa göre yanıt üret
        answer, token_usage, rag_contexts, web_used, model_used = \
            await self._execute_route(route, request.message, memory_ctx)

        # 5. Mesajları memory'e kaydet
        processing_ms = int((time.time() - start_time) * 1000)
        message_id = f"msg-{uuid.uuid4().hex[:10]}"

        await memory_service.add_message(session_id, "user", request.message)
        await memory_service.add_message(session_id, "assistant", answer, {
            "route": route.value,
            "model": model_used,
            "tokens": token_usage.total_tokens
        })

        # 6. Trace log kaydet
        await trace_service.log_request(
            session_id=session_id,
            message_id=message_id,
            user_message=request.message,
            assistant_response=answer,
            route_type=route,
            model_used=model_used,
            token_usage=token_usage,
            processing_time_ms=processing_ms,
            rag_doc_count=len(rag_contexts) if rag_contexts else 0,
            web_results_used=web_used,
            routing_reason=routing_reason
        )

        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            answer=answer,
            route_used=route,
            model_used=model_used,
            token_usage=token_usage,
            rag_contexts=rag_contexts if rag_contexts else None,
            web_results_used=web_used,
            processing_time_ms=processing_ms,
            created_at=datetime.now(timezone.utc)
        )

    async def _execute_route(self, route, message, memory_ctx):
        """Seçilen akışı çalıştırır."""

        if route == RouteType.DIRECT:
            return await self._direct_flow(message, memory_ctx)

        elif route == RouteType.RAG:
            return await self._rag_flow(message, memory_ctx)

        elif route == RouteType.WEB:
            return await self._web_flow(message, memory_ctx)

        return await self._direct_flow(message, memory_ctx)

    # ── DIRECT Akışı ──────────────────────────────────────────────────────────
    async def _direct_flow(self, message, memory_ctx):
        messages, system_prompt = llm_service.build_context_messages(
            user_message=message,
            short_term=memory_ctx.short_term,
            long_term_summary=memory_ctx.long_term
        )

        answer, token_usage = await llm_service.complete_openai(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=settings.MAX_TOKENS_DIRECT
        )

        return answer, token_usage, None, False, settings.OPENAI_MODEL

    # ── RAG Akışı ─────────────────────────────────────────────────────────────
    async def _rag_flow(self, message, memory_ctx):
        # Doküman arama
        rag_contexts, context_text = await rag_service.get_context(message)

        if not rag_contexts:
            logger.info("RAG: Doküman bulunamadı, DIRECT akışa geçiliyor")
            return await self._direct_flow(message, memory_ctx)

        messages, system_prompt = llm_service.build_context_messages(
            user_message=message,
            short_term=memory_ctx.short_term,
            long_term_summary=memory_ctx.long_term,
            extra_context=context_text
        )

        answer, token_usage = await llm_service.complete_openai(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=settings.MAX_TOKENS_RAG
        )

        return answer, token_usage, rag_contexts, False, settings.OPENAI_MODEL

    # ── WEB Akışı ─────────────────────────────────────────────────────────────
    async def _web_flow(self, message, memory_ctx):
        # Web araması
        web_context, web_results = await web_service.search(message)

        messages, system_prompt = llm_service.build_context_messages(
            user_message=message,
            short_term=memory_ctx.short_term,
            long_term_summary=memory_ctx.long_term,
            extra_context=web_context if web_results else None
        )

        # WEB akışı Claude ile çalışır
        answer, token_usage = await llm_service.complete_claude(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=settings.MAX_TOKENS_WEB
        )

        return answer, token_usage, None, bool(web_results), settings.CLAUDE_MODEL

    async def _ensure_session(self, session_id: str):
        """Session yoksa oluşturur."""
        db = get_db()
        existing = await db.sessions.find_one({"session_id": session_id})
        if not existing:
            await db.sessions.insert_one({
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc),
                "message_count": 0,
                "metadata": {}
            })


orchestrator = LLMOrchestrator()
