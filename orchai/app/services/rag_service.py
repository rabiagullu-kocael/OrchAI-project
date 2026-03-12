from app.rag.retrieval_service import retrieval_service
from app.models.chat_models import RAGContext
from typing import List, Tuple


class RAGService:
    async def get_context(self, query: str) -> Tuple[List[RAGContext], str]:
        return await retrieval_service.retrieve(query)


rag_service = RAGService()
