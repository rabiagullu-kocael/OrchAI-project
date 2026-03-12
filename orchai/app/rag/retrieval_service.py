"""
Retrieval Service - RAG için doküman arama

Strateji:
1. Önce MongoDB text search (hızlı, ücretsiz)
2. Bulunan dokümanları embedding ile re-rank (daha doğru)
3. Threshold altındaki sonuçları filtrele
"""

import logging
from typing import List, Dict, Any, Tuple
from app.utils.mongo_client import get_db
from app.utils.config import settings
from app.rag.embedding_service import embedding_service
from app.models.chat_models import RAGContext

logger = logging.getLogger(__name__)


class RetrievalService:

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = None
    ) -> Tuple[List[RAGContext], str]:
        """
        Sorgu için en alakalı dokümanları getirir.

        Returns: (rag_contexts, formatted_context_text)
        """
        threshold = threshold or settings.RAG_SIMILARITY_THRESHOLD
        db = get_db()

        # 1. MongoDB text search ile aday dokümanları bul
        candidates = await self._text_search(db, query, limit=10)

        if not candidates:
            logger.info(f"RAG: Metin aramasında doküman bulunamadı: '{query[:50]}'")
            return [], ""

        # 2. Embedding ile sorgu vektörü oluştur
        try:
            query_embedding = await embedding_service.embed_text(query)
        except Exception as e:
            logger.warning(f"Embedding hatası, text search sonuçları kullanılıyor: {e}")
            # Embedding başarısız olursa text search sonuçlarını direkt kullan
            return await self._fallback_results(candidates, top_k)

        # 3. Doküman embedding'leri ile benzerlik hesapla
        scored_docs = []
        for doc in candidates:
            doc_text = f"{doc.get('title', '')} {doc.get('content', '')}"

            try:
                # Dokümanın önceden embedding'i varsa kullan, yoksa hesapla
                if "embedding" in doc and doc["embedding"]:
                    doc_embedding = doc["embedding"]
                else:
                    doc_embedding = await embedding_service.embed_text(doc_text[:2000])

                score = embedding_service.compute_similarity(query_embedding, doc_embedding)
                scored_docs.append((doc, score))
            except Exception as e:
                logger.warning(f"Doküman embedding hatası: {e}")
                scored_docs.append((doc, 0.5))  # Varsayılan skor

        # 4. Skora göre sırala ve threshold filtrele
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        filtered = [(doc, score) for doc, score in scored_docs if score >= threshold]

        if not filtered:
            # Threshold altında ama en iyiyi yine de dön (minimum 1 sonuç)
            if scored_docs:
                filtered = [scored_docs[0]]
                logger.info(f"RAG threshold altında, en iyi sonuç kullanılıyor: score={scored_docs[0][1]:.3f}")

        # 5. RAGContext listesi oluştur
        results = []
        for doc, score in filtered[:top_k]:
            content = doc.get("content", "")
            snippet = content[:300] + "..." if len(content) > 300 else content

            results.append(RAGContext(
                document_id=str(doc.get("_id", "")),
                title=doc.get("title", "Başlıksız"),
                score=round(score, 4),
                snippet=snippet
            ))

        # 6. LLM için context metni oluştur
        context_text = self._format_context(results, filtered)

        logger.info(f"RAG: {len(results)} doküman bulundu (threshold={threshold})")
        return results, context_text

    async def _text_search(self, db, query: str, limit: int = 10) -> List[Dict]:
        """MongoDB text search ile aday dokümanları bulur."""
        try:
            cursor = db.documents.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "embedding": 0}  # embedding alanını getirme (büyük)
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)

            docs = []
            async for doc in cursor:
                docs.append(doc)
            return docs
        except Exception as e:
            logger.warning(f"Text search hatası, tüm dokümanlar taranıyor: {e}")
            # Text index yoksa tüm dokümanları al (küçük koleksiyonlar için)
            cursor = db.documents.find({}, {"embedding": 0}).limit(limit)
            docs = []
            async for doc in cursor:
                docs.append(doc)
            return docs

    async def _fallback_results(
        self, candidates: List[Dict], top_k: int
    ) -> Tuple[List[RAGContext], str]:
        """Embedding başarısız olduğunda text search sonuçlarını kullan."""
        results = []
        for doc in candidates[:top_k]:
            content = doc.get("content", "")
            snippet = content[:300] + "..." if len(content) > 300 else content
            results.append(RAGContext(
                document_id=str(doc.get("_id", "")),
                title=doc.get("title", "Başlıksız"),
                score=0.7,
                snippet=snippet
            ))

        context_text = self._format_context(results, [(d, 0.7) for d in candidates[:top_k]])
        return results, context_text

    def _format_context(
        self, rag_contexts: List[RAGContext], scored_docs: List[Tuple]
    ) -> str:
        """RAG sonuçlarını LLM context metni olarak formatlar."""
        if not scored_docs:
            return ""

        parts = []
        for i, (doc, score) in enumerate(scored_docs[:3], 1):
            title = doc.get("title", "Başlıksız")
            content = doc.get("content", "")[:800]
            parts.append(f"[Kaynak {i}: {title} (benzerlik: {score:.2f})]\n{content}")

        return "\n\n".join(parts)


retrieval_service = RetrievalService()
