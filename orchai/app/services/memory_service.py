"""
Memory Service - Kısa ve Uzun Vadeli Bellek Yönetimi

Kısa Vadeli Memory (short_term):
---------------------------------
- Son MAX_SHORT_TERM_MESSAGES mesajı ham halde saklar
- Her sohbet turunda context olarak LLM'e verilir
- MongoDB messages koleksiyonundan çekilir

Uzun Vadeli Memory (long_term):
--------------------------------
- Mesaj sayısı LONG_TERM_SUMMARY_THRESHOLD'u geçtiğinde tetiklenir
- Eski mesajlar LLM ile özetlenerek memory_store'a yazılır
- Özet, yeni konuşmalarda "geçmiş bağlam" olarak kullanılır

Geri Çağırma Kuralları:
------------------------
1. Her istekte önce uzun vadeli özet alınır (varsa)
2. Kısa vadeli son N mesaj alınır
3. İkisi birleştirilerek LLM'e system/context olarak verilir
4. Uzun vadeye geçiş eşiği aşılırsa arka planda özetleme tetiklenir
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.utils.mongo_client import get_db
from app.utils.config import settings
from app.models.memory_models import MemoryType, MemoryRetrievalResult, MemoryStats

logger = logging.getLogger(__name__)


class MemoryService:

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Mesajı messages koleksiyonuna ekler."""
        db = get_db()
        doc = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "memory_type": MemoryType.SHORT_TERM,
            "created_at": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
        result = await db.messages.insert_one(doc)
        logger.debug(f"Mesaj eklendi: {session_id} [{role}]")

        # Eşik kontrolü - arka planda özetleme tetikle
        await self._check_and_summarize(session_id)

        return str(result.inserted_id)

    async def get_memory_context(self, session_id: str) -> MemoryRetrievalResult:
        """
        Kısa + uzun vadeli memory'yi birleştirerek context döndürür.

        Geri Çağırma Kuralları:
        1. Uzun vadeli özet (varsa) ilk alınır
        2. Kısa vadeli son N mesaj alınır
        3. Toplam token tahmini hesaplanır
        """
        db = get_db()
        result = MemoryRetrievalResult()

        # 1. Uzun vadeli özet
        long_term_doc = await db.memory_store.find_one(
            {"session_id": session_id, "memory_type": MemoryType.LONG_TERM},
            sort=[("updated_at", -1)]
        )
        if long_term_doc:
            result.long_term = long_term_doc.get("content", "")
            logger.debug(f"Uzun vadeli memory bulundu: {session_id}")

        # 2. Kısa vadeli son mesajlar
        short_term_cursor = db.messages.find(
            {"session_id": session_id}
        ).sort("created_at", -1).limit(settings.MAX_SHORT_TERM_MESSAGES)

        messages = []
        async for msg in short_term_cursor:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "created_at": msg["created_at"].isoformat() if isinstance(msg.get("created_at"), datetime) else str(msg.get("created_at", ""))
            })

        # Kronolojik sıraya çevir
        result.short_term = list(reversed(messages))

        # Token tahmini (yaklaşık: 4 karakter = 1 token)
        total_chars = sum(len(m["content"]) for m in result.short_term)
        if result.long_term:
            total_chars += len(result.long_term)
        result.total_tokens_estimated = total_chars // 4

        logger.info(f"Memory context: {session_id} | short={len(result.short_term)} | long={'var' if result.long_term else 'yok'}")
        return result

    async def _check_and_summarize(self, session_id: str):
        """Mesaj sayısı eşiği aşarsa özetleme başlatır."""
        db = get_db()
        count = await db.messages.count_documents({"session_id": session_id})

        if count > 0 and count % settings.LONG_TERM_SUMMARY_THRESHOLD == 0:
            logger.info(f"Özetleme tetiklendi: {session_id} ({count} mesaj)")
            await self._summarize_to_long_term(session_id)

    async def _summarize_to_long_term(self, session_id: str):
        """
        Eski mesajları alıp LLM ile özetler, memory_store'a yazar.
        Uzun vadeli hafıza burada oluşturulur.
        """
        try:
            db = get_db()

            # Özetlenecek eski mesajları al
            all_messages = []
            cursor = db.messages.find(
                {"session_id": session_id}
            ).sort("created_at", 1).limit(settings.LONG_TERM_SUMMARY_THRESHOLD)

            async for msg in cursor:
                all_messages.append(f"{msg['role'].upper()}: {msg['content']}")

            if not all_messages:
                return

            conversation_text = "\n".join(all_messages)

            # Önceki özet varsa ekle
            prev_summary_doc = await db.memory_store.find_one(
                {"session_id": session_id, "memory_type": MemoryType.LONG_TERM}
            )
            prev_summary = prev_summary_doc.get("content", "") if prev_summary_doc else ""

            # LLM ile özet oluştur (lazy import - circular import önlemi)
            from app.services.llm_service import llm_service

            # ✅ DÜZELTME: f-string içinde backslash kullanılamaz (Python 3.11)
            # Koşullu kısmı önceden değişkene atıyoruz
            prev_section = f"Önceki Özet:\n{prev_summary}\n\n" if prev_summary else ""

            summary_prompt = (
                "Aşağıdaki konuşmayı özetle. Önemli bilgileri, kullanıcı tercihlerini "
                "ve konuşulan ana konuları koru.\n\n"
                f"{prev_section}"
                f"Yeni Konuşma:\n{conversation_text}\n\n"
                "Özet (maksimum 300 kelime):"
            )

            summary = await llm_service.complete_simple(summary_prompt, max_tokens=400)

            # memory_store'a yaz (upsert)
            await db.memory_store.update_one(
                {"session_id": session_id, "memory_type": MemoryType.LONG_TERM},
                {
                    "$set": {
                        "content": summary,
                        "memory_type": MemoryType.LONG_TERM,
                        "updated_at": datetime.now(timezone.utc),
                        "message_count_at_summary": await db.messages.count_documents({"session_id": session_id})
                    },
                    "$setOnInsert": {
                        "session_id": session_id,
                        "created_at": datetime.now(timezone.utc)
                    }
                },
                upsert=True
            )
            logger.info(f"Uzun vadeli özet oluşturuldu: {session_id}")

        except Exception as e:
            logger.error(f"Özetleme hatası: {e}")

    async def get_memory_stats(self, session_id: str) -> MemoryStats:
        """Memory istatistiklerini döndürür."""
        db = get_db()
        short_count = await db.messages.count_documents({"session_id": session_id})
        long_term_doc = await db.memory_store.find_one(
            {"session_id": session_id, "memory_type": MemoryType.LONG_TERM}
        )

        last_updated = None
        if long_term_doc and "updated_at" in long_term_doc:
            last_updated = long_term_doc["updated_at"]

        return MemoryStats(
            session_id=session_id,
            short_term_count=short_count,
            long_term_summary_exists=long_term_doc is not None,
            last_updated=last_updated
        )

    async def clear_session_memory(self, session_id: str):
        """Session memory'sini temizler."""
        db = get_db()
        await db.messages.delete_many({"session_id": session_id})
        await db.memory_store.delete_many({"session_id": session_id})
        logger.info(f"Session memory temizlendi: {session_id}")


memory_service = MemoryService()
