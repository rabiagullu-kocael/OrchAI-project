from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"   # Son N mesaj - ham halde saklanır
    LONG_TERM = "long_term"     # Özet/önemli bilgiler - LLM ile özetlenir


class MemoryEntry(BaseModel):
    session_id: str
    memory_type: MemoryType
    content: str                        # Mesaj içeriği veya özet
    role: Optional[str] = None          # "user" / "assistant" (short_term için)
    importance_score: float = 0.5       # 0-1 arası önem skoru (long_term için)
    keywords: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRetrievalResult(BaseModel):
    short_term: List[Dict[str, Any]] = Field(default_factory=list)
    long_term: Optional[str] = None     # Özet metin
    total_tokens_estimated: int = 0


class MemoryStats(BaseModel):
    session_id: str
    short_term_count: int
    long_term_summary_exists: bool
    last_updated: Optional[datetime]
