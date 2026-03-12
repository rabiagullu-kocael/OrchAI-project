from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RouteType(str, Enum):
    DIRECT = "direct"
    RAG = "rag"
    WEB = "web"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ── Request / Response Modelleri ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="Kullanıcının mesajı", min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None, description="Oturum ID (boş bırakılırsa yeni oturum)")
    force_route: Optional[RouteType] = Field(None, description="Zorla yönlendirme (test için)")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Chatbot nedir ve nasıl çalışır?",
                "session_id": "user-123-session-456"
            }
        }


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RAGContext(BaseModel):
    document_id: str
    title: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    route_used: RouteType
    model_used: str
    token_usage: TokenUsage
    rag_contexts: Optional[List[RAGContext]] = None
    web_results_used: bool = False
    processing_time_ms: int
    created_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-123-session-456",
                "message_id": "msg-789",
                "answer": "Chatbot, yapay zeka destekli bir konuşma programıdır...",
                "route_used": "rag",
                "model_used": "gpt-4o-mini",
                "token_usage": {
                    "prompt_tokens": 350,
                    "completion_tokens": 180,
                    "total_tokens": 530,
                    "estimated_cost_usd": 0.000106
                },
                "rag_contexts": [
                    {
                        "document_id": "69b0b7b6d5167409c6cf5827",
                        "title": "Chatbot Nedir",
                        "score": 0.89,
                        "snippet": "Chatbot, kullanıcılarla doğal dil ile iletişim kurabilen..."
                    }
                ],
                "web_results_used": False,
                "processing_time_ms": 1240,
                "created_at": "2026-03-12T10:00:00Z"
            }
        }


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Özel session ID (boş = otomatik)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int
    metadata: Dict[str, Any]


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    total: int


class ScenarioTestRequest(BaseModel):
    scenarios: Optional[List[str]] = Field(
        None,
        description="Test edilecek senaryolar (boş = varsayılan 10 senaryo)"
    )
