import re
from typing import Any, Dict, List

from backend.chat.enums import StreamEvent
from backend.schemas.chat import ChatMessage
from backend.schemas.cohere_chat import CohereChatRequest
from backend.schemas.context import Context
from community.model_deployments import BaseDeployment
from community.graph_rag import graph_utils


class GraphRagDeployment(BaseDeployment):
    def __init__(self):
        pass

    @property
    def rerank_enabled(self) -> bool:
        return False

    @classmethod
    def list_models(cls) -> List[str]:
        return []

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def invoke_chat_stream(
        self, chat_request: CohereChatRequest, **kwargs: Any
    ) -> Any:
        """
        Cohere request (max_tokens, message, chat_history, documents, temperature)
        """

        # if chat_request.max_tokens is None:
        #     chat_request.max_tokens = 200

        # if len(chat_request.documents) == 0:
        #     prompt = self.prompt_template.dummy_chat_template(
        #         chat_request.message, chat_request.chat_history
        #     )
        # else:
        #     prompt = self.prompt_template.dummy_rag_template(
        #         chat_request.message, chat_request.chat_history, chat_request.documents
        #     )

        # stream = model(
        #     prompt,
        #     stream=True,
        #     max_tokens=chat_request.max_tokens,
        #     temperature=chat_request.temperature,
        # )

        stream = await aget_response_stream(chat_request.message)
        report = await stream.__anext__()

        yield {
            "event_type": "stream-start",
            "generation_id": "",
        }

        async for item in stream:
            yield {
                "event_type": "text-generation",
                "text": item,
            }

        yield {
            "event_type": "stream-end",
            "finish_reason": "COMPLETE",
        }

    async def invoke_chat(
        self, chat_request: CohereChatRequest, ctx: Context, **kwargs: Any
    ) -> Any:
        model = self._get_model()

        # if chat_request.max_tokens is None:
        #     chat_request.max_tokens = 200

        # response = model(
        #     chat_request.message,
        #     stream=False,
        #     max_tokens=chat_request.max_tokens,
        #     temperature=chat_request.temperature,
        # )

        response, context = get_result_graphrag(chat_request.message)

        return {"text": response}

    async def invoke_rerank(
        self, query: str, documents: List[Dict[str, Any]], ctx: Context, **kwargs: Any
    ) -> Any:
        return None