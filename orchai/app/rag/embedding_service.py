"""
Embedding Service - Metin vektörleştirme

Dokümanlar ve sorgu metinleri OpenAI text-embedding-3-small
ile vektörleştirilir. Cosine similarity ile benzerlik hesaplanır.
"""

import logging
import math
from typing import List
from openai import AsyncOpenAI
from app.utils.config import settings

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """İki vektör arasındaki cosine similarity hesaplar."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingService:

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL

    async def embed_text(self, text: str) -> List[float]:
        """Tek metin için embedding oluşturur."""
        try:
            # Metni temizle ve kısalt (max 8000 token)
            text = text.strip()[:6000]

            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding hatası: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Birden fazla metin için batch embedding."""
        try:
            cleaned = [t.strip()[:6000] for t in texts]
            response = await self.client.embeddings.create(
                model=self.model,
                input=cleaned
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding hatası: {e}")
            raise

    def compute_similarity(self, query_embedding: List[float], doc_embedding: List[float]) -> float:
        """İki embedding arasındaki benzerliği hesaplar (0-1)."""
        return cosine_similarity(query_embedding, doc_embedding)


embedding_service = EmbeddingService()
