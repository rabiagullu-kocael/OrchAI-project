"""
Web Service - Güncel bilgi için web araması

DuckDuckGo Instant Answer API kullanır (ücretsiz, API key gerekmez).
Birden fazla kaynak kombine edilerek LLM'e context sağlanır.
"""

import logging
import aiohttp
import json
from typing import List, Dict, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DDGO_SEARCH_URL = "https://api.duckduckgo.com/"
HEADERS = {"User-Agent": "OrchAI/1.0 (AI Assistant Research Tool)"}


class WebService:

    async def search(self, query: str, max_results: int = 3) -> Tuple[str, List[Dict]]:
        """
        Web araması yapar ve context metni döndürür.

        Returns: (formatted_context, raw_results)
        """
        logger.info(f"Web araması: '{query[:60]}'")

        results = []

        try:
            results = await self._duckduckgo_search(query)
        except Exception as e:
            logger.warning(f"DDG arama hatası: {e}")

        if not results:
            # Arama başarısız olursa tarih/zaman bilgisi ver
            now = datetime.now(timezone.utc)
            fallback_context = (
                f"Web araması şu an kullanılamıyor. "
                f"Mevcut UTC tarihi/saati: {now.strftime('%Y-%m-%d %H:%M:%S')}. "
                f"Genel bilgimle yanıt vermeye çalışacağım."
            )
            return fallback_context, []

        context = self._format_web_context(results[:max_results])
        logger.info(f"Web: {len(results)} sonuç bulundu")
        return context, results[:max_results]

    async def _duckduckgo_search(self, query: str) -> List[Dict]:
        """DuckDuckGo Instant Answer API ile arama."""
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
            "skip_disambig": "1"
        }

        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(
                DDGO_SEARCH_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json(content_type=None)

        results = []

        # Abstract (ana özet)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "DDG Abstract"),
                "snippet": data["Abstract"],
                "url": data.get("AbstractURL", ""),
                "source": "DuckDuckGo Abstract"
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "snippet": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                    "source": "DuckDuckGo Related"
                })

        # Answer (hesap makinesi, dönüşüm gibi anlık cevaplar)
        if data.get("Answer"):
            results.insert(0, {
                "title": "Anlık Cevap",
                "snippet": data["Answer"],
                "url": "",
                "source": "DuckDuckGo Answer"
            })

        return results

    def _format_web_context(self, results: List[Dict]) -> str:
        """Web sonuçlarını LLM context metni olarak formatlar."""
        if not results:
            return ""

        now = datetime.now(timezone.utc)
        parts = [f"[Web Arama Sonuçları - {now.strftime('%Y-%m-%d %H:%M UTC')}]"]

        for i, result in enumerate(results, 1):
            title = result.get("title", "")[:100]
            snippet = result.get("snippet", "")[:500]
            url = result.get("url", "")

            part = f"\n[Sonuç {i}] {title}"
            if url:
                part += f"\nKaynak: {url}"
            part += f"\n{snippet}"
            parts.append(part)

        return "\n".join(parts)


web_service = WebService()
