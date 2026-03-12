"""
Router Engine - Gelen isteği analiz ederek doğru akışa yönlendirir.

Routing Kuralları:
-----------------
1. DIRECT  → Basit soru/cevap, genel bilgi, selamlama, matematik
2. RAG     → Bilgi tabanında olabilecek konular (chatbot, AI, teknik terimler)
3. WEB     → Güncel haber, bugünkü tarih/kur/hava, son gelişmeler

Karar Sırası: force_route → keyword analizi → LLM classifier
"""

import re
import logging
from app.models.chat_models import RouteType
from app.utils.config import settings

logger = logging.getLogger(__name__)

# ── Keyword Tabanlı Hızlı Routing ─────────────────────────────────────────────

WEB_KEYWORDS = [
    "bugün", "şu an", "şu anda", "güncel", "son dakika", "haber",
    "today", "current", "latest", "news", "right now", "2024", "2025", "2026",
    "döviz", "kur", "borsa", "hisse", "bitcoin", "kripto",
    "hava durumu", "weather", "deprem", "earthquake",
    "kim kazandı", "maç sonucu", "score"
]

RAG_KEYWORDS = [
    "chatbot", "yapay zeka", "artificial intelligence", "machine learning",
    "derin öğrenme", "deep learning", "nlp", "doğal dil işleme",
    "transformer", "gpt", "llm", "büyük dil modeli",
    "rag", "retrieval", "embedding", "vektör",
    "nedir", "ne demek", "nasıl çalışır", "açıkla", "anlat",
    "what is", "how does", "explain", "describe",
    "neural network", "sinir ağı", "algoritma", "algorithm"
]

DIRECT_PATTERNS = [
    r"^(merhaba|selam|hey|hi|hello|nasılsın|naber)",
    r"^(\d+[\+\-\*\/]\d+)",  # Matematik işlemleri
    r"^(teşekkür|sağ ol|tamam|ok|anladım|tamamdır)",
]


class RouterEngine:
    """
    Üç aşamalı routing kararı:
    1. Force route varsa direkt kullan
    2. Keyword/pattern analizi
    3. Belirsizse LLM ile classify et
    """

    def __init__(self):
        self.web_keywords = [k.lower() for k in WEB_KEYWORDS]
        self.rag_keywords = [k.lower() for k in RAG_KEYWORDS]
        self.direct_patterns = [re.compile(p, re.IGNORECASE) for p in DIRECT_PATTERNS]

    async def determine_route(
        self,
        message: str,
        force_route: RouteType = None
    ) -> tuple[RouteType, str]:
        """
        Returns: (route_type, reason)
        """
        # 1. Force route
        if force_route:
            logger.info(f"Force route: {force_route}")
            return force_route, f"Zorunlu yönlendirme: {force_route}"

        message_lower = message.lower().strip()

        # 2. Pattern bazlı DIRECT kontrolü
        for pattern in self.direct_patterns:
            if pattern.match(message_lower):
                return RouteType.DIRECT, "Basit selamlama/işlem pattern eşleşmesi"

        # 3. WEB keyword kontrolü (öncelikli - güncel bilgi kritik)
        web_matches = [kw for kw in self.web_keywords if kw in message_lower]
        if web_matches:
            reason = f"Web keyword eşleşmesi: {', '.join(web_matches[:3])}"
            logger.info(f"WEB route → {reason}")
            return RouteType.WEB, reason

        # 4. RAG keyword kontrolü
        rag_matches = [kw for kw in self.rag_keywords if kw in message_lower]
        if rag_matches:
            reason = f"RAG keyword eşleşmesi: {', '.join(rag_matches[:3])}"
            logger.info(f"RAG route → {reason}")
            return RouteType.RAG, reason

        # 5. Soru işareti + uzun mesaj → RAG dene
        if "?" in message and len(message.split()) > 5:
            return RouteType.RAG, "Uzun soru - RAG ile denenecek"

        # 6. Default: DIRECT
        return RouteType.DIRECT, "Varsayılan akış - direkt yanıt"

    async def classify_with_llm(self, message: str, llm_service) -> RouteType:
        """
        Belirsiz durumlarda LLM ile classify et.
        Sadece gerektiğinde çağrılır (keyword match olmadığında).
        """
        prompt = f"""Aşağıdaki kullanıcı mesajını analiz et ve en uygun işlem akışını belirle.

Mesaj: "{message}"

Seçenekler:
- DIRECT: Genel bilgi, selamlama, basit soru/cevap
- RAG: Teknik/kavramsal sorular - bilgi tabanından yanıt gerekiyor
- WEB: Güncel bilgi, haber, fiyat, tarihsel olmayan canlı veri

Sadece tek kelime yaz: DIRECT, RAG veya WEB"""

        try:
            result = await llm_service.complete_simple(prompt, max_tokens=10)
            result = result.strip().upper()
            if "WEB" in result:
                return RouteType.WEB
            elif "RAG" in result:
                return RouteType.RAG
            return RouteType.DIRECT
        except Exception:
            return RouteType.DIRECT


router_engine = RouterEngine()
