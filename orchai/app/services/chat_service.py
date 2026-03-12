from app.orchestrator.llm_orchestrator import orchestrator
from app.models.chat_models import ChatRequest, ChatResponse


class ChatService:
    async def process(self, request: ChatRequest) -> ChatResponse:
        return await orchestrator.handle(request)


chat_service = ChatService()
