"""
LLM Service - OpenAI ve Claude API yönetimi

Model Seçim Stratejisi:
-----------------------
- OpenAI GPT-4o-mini: Hızlı, ucuz - DIRECT ve RAG akışları
- Claude Sonnet: Güçlü, uzun context - WEB akışı ve karmaşık sorular
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
import anthropic
from app.utils.config import settings
from app.models.chat_models import TokenUsage

logger = logging.getLogger(__name__)

# Token başına maliyet (USD) - Mart 2026 yaklaşık değerler
PRICING = {
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "claude-sonnet-4-20250514": {"input": 0.003 / 1000, "output": 0.015 / 1000},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING.get(model, {"input": 0.001 / 1000, "output": 0.002 / 1000})
    return (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])


class LLMService:

    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.claude_client = anthropic.AsyncAnthropic(api_key=settings.CLAUDE_API_KEY)

    async def complete_openai(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: str = None
    ) -> Tuple[str, TokenUsage]:
        """
        OpenAI GPT-4o-mini ile tamamlama.
        DIRECT ve RAG akışları için kullanılır.
        """
        start = time.time()

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            content = response.choices[0].message.content
            usage = response.usage

            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=calculate_cost(
                    settings.OPENAI_MODEL,
                    usage.prompt_tokens,
                    usage.completion_tokens
                )
            )

            elapsed = int((time.time() - start) * 1000)
            logger.info(
                f"OpenAI tamamlandı | tokens={usage.total_tokens} | "
                f"cost=${token_usage.estimated_cost_usd:.6f} | {elapsed}ms"
            )
            return content, token_usage

        except Exception as e:
            logger.error(f"OpenAI hatası: {e}")
            raise

    async def complete_claude(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system_prompt: str = None
    ) -> Tuple[str, TokenUsage]:
        """
        Claude Sonnet ile tamamlama.
        WEB akışı ve uzun context gerektiren işlemler için kullanılır.
        """
        start = time.time()

        # Claude formatı: system ayrı parametre
        claude_system = system_prompt or "Sen yardımcı bir AI asistanısın. Türkçe yanıt ver."

        # Claude messages formatına çevir
        claude_messages = []
        for msg in messages:
            role = msg["role"] if msg["role"] in ["user", "assistant"] else "user"
            claude_messages.append({"role": role, "content": msg["content"]})

        try:
            response = await self.claude_client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=claude_system,
                messages=claude_messages,
                temperature=temperature
            )

            content = response.content[0].text
            usage = response.usage

            token_usage = TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                estimated_cost_usd=calculate_cost(
                    settings.CLAUDE_MODEL,
                    usage.input_tokens,
                    usage.output_tokens
                )
            )

            elapsed = int((time.time() - start) * 1000)
            logger.info(
                f"Claude tamamlandı | tokens={token_usage.total_tokens} | "
                f"cost=${token_usage.estimated_cost_usd:.6f} | {elapsed}ms"
            )
            return content, token_usage

        except Exception as e:
            logger.error(f"Claude hatası: {e}")
            raise

    async def complete_simple(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Basit tek prompt tamamlama (routing classifier, özetleme için).
        Maliyet düşük tutmak için OpenAI kullanır.
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Simple complete hatası: {e}")
            return ""

    def build_context_messages(
        self,
        user_message: str,
        short_term: List[Dict],
        long_term_summary: Optional[str] = None,
        extra_context: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        Memory ve context'i birleştirerek LLM mesaj listesi oluşturur.

        Returns: (messages, system_prompt)
        """
        system_parts = [
            "Sen OrchAI adlı yardımcı bir yapay zeka asistanısın.",
            "Türkçe sorulara Türkçe, İngilizce sorulara İngilizce yanıt ver.",
            "Yanıtların doğru, özlü ve yardımcı olsun."
        ]

        # Uzun vadeli memory özeti
        if long_term_summary:
            system_parts.append(
                f"\n## Geçmiş Konuşma Özeti:\n{long_term_summary}"
            )

        # Ek context (RAG veya web)
        if extra_context:
            system_parts.append(
                f"\n## İlgili Bilgi:\n{extra_context}\n\nYukarıdaki bilgileri kullanarak yanıt ver."
            )

        system_prompt = "\n".join(system_parts)

        # Mesaj listesi: kısa vadeli geçmiş + yeni mesaj
        messages = []
        for msg in short_term:
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        return messages, system_prompt


llm_service = LLMService()
