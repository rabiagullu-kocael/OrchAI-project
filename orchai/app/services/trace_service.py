"""
Trace Service - Her istek için token/cost log kaydı

Her LLM çağrısında trace_logs koleksiyonuna yazılır.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from app.utils.mongo_client import get_db
from app.models.chat_models import RouteType, TokenUsage

logger = logging.getLogger(__name__)


class TraceService:

    async def log_request(
        self,
        session_id: str,
        message_id: str,
        user_message: str,
        assistant_response: str,
        route_type: RouteType,
        model_used: str,
        token_usage: TokenUsage,
        processing_time_ms: int,
        rag_doc_count: int = 0,
        web_results_used: bool = False,
        routing_reason: str = "",
        extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """Her istek sonunda trace_logs koleksiyonuna yazar."""
        db = get_db()

        doc = {
            "session_id": session_id,
            "message_id": message_id,
            "route_type": route_type.value,
            "model_used": model_used,
            "routing_reason": routing_reason,
            "user_message_preview": user_message[:200],
            "response_preview": assistant_response[:200],
            "token_usage": {
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "total_tokens": token_usage.total_tokens,
                "estimated_cost_usd": token_usage.estimated_cost_usd,
            },
            "processing_time_ms": processing_time_ms,
            "rag_doc_count": rag_doc_count,
            "web_results_used": web_results_used,
            "created_at": datetime.now(timezone.utc),
            "extra": extra or {}
        }

        result = await db.trace_logs.insert_one(doc)
        logger.info(
            f"Trace kaydedildi | route={route_type.value} | "
            f"model={model_used} | tokens={token_usage.total_tokens} | "
            f"cost=${token_usage.estimated_cost_usd:.6f} | {processing_time_ms}ms"
        )
        return str(result.inserted_id)

    async def get_session_traces(self, session_id: str, limit: int = 20) -> list:
        db = get_db()
        cursor = db.trace_logs.find(
            {"session_id": session_id}
        ).sort("created_at", -1).limit(limit)

        traces = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            traces.append(doc)
        return traces

    async def get_cost_summary(self, session_id: str) -> Dict[str, Any]:
        """Session bazlı toplam maliyet özeti."""
        db = get_db()
        pipeline = [
            {"$match": {"session_id": session_id}},
            {"$group": {
                "_id": "$route_type",
                "total_tokens": {"$sum": "$token_usage.total_tokens"},
                "total_cost": {"$sum": "$token_usage.estimated_cost_usd"},
                "request_count": {"$sum": 1},
                "avg_processing_ms": {"$avg": "$processing_time_ms"}
            }}
        ]
        cursor = db.trace_logs.aggregate(pipeline)
        breakdown = {}
        grand_total_tokens = 0
        grand_total_cost = 0.0

        async for row in cursor:
            route = row["_id"]
            breakdown[route] = {
                "request_count": row["request_count"],
                "total_tokens": row["total_tokens"],
                "total_cost_usd": round(row["total_cost"], 6),
                "avg_processing_ms": round(row["avg_processing_ms"], 0)
            }
            grand_total_tokens += row["total_tokens"]
            grand_total_cost += row["total_cost"]

        return {
            "session_id": session_id,
            "grand_total_tokens": grand_total_tokens,
            "grand_total_cost_usd": round(grand_total_cost, 6),
            "breakdown_by_route": breakdown
        }


trace_service = TraceService()
