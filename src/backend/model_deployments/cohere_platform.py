from typing import Any, Dict, List

import cohere
import requests
import os

from backend.chat.collate import to_dict
from backend.config.settings import Settings
from backend.model_deployments.base import BaseDeployment
from backend.model_deployments.utils import get_model_config_var
from backend.schemas.cohere_chat import CohereChatRequest
from backend.schemas.context import Context
from backend.services.logger.utils import LoggerFactory
from community.graph_rag import graph_utils

import logging

COHERE_API_KEY_ENV_VAR = "COHERE_API_KEY"
COHERE_ENV_VARS = [COHERE_API_KEY_ENV_VAR]
DEFAULT_RERANK_MODEL = "rerank-english-v2.0"

os.environ['GRAPHRAG_API_KEY'] = '...'


class CohereDeployment(BaseDeployment):
    """Cohere Platform Deployment."""

    # client_name = "cohere-toolkit"
    # api_key = Settings().deployments.cohere_platform.api_key

    def __init__(self, **kwargs: Any):
        # Override the environment variable from the request
        # api_key = get_model_config_var(
        #     COHERE_API_KEY_ENV_VAR, CohereDeployment.api_key, **kwargs
        # )
        # self.client = cohere.Client(api_key, client_name=self.client_name)
        pass

    @property
    def rerank_enabled(self) -> bool:
        return True

    @classmethod
    def list_models(cls) -> List[str]:
        # logger = LoggerFactory().get_logger()
        # if not CohereDeployment.is_available():
        #     return []

        # url = "https://api.cohere.ai/v1/models"
        # headers = {
        #     "accept": "application/json",
        #     "authorization": f"Bearer {cls.api_key}",
        # }

        # response = requests.get(url, headers=headers)

        # if not response.ok:
        #     logger.warning(
        #         event=f"[Cohere Deployment] Error retrieving models: Invalid HTTP {response.status_code} response",
        #     )
        #     return []

        # models = response.json()["models"]
        # return [model["name"] for model in models if model.get("endpoints") and "chat" in model["endpoints"]]
        return ['graph_rag']

    @classmethod
    def is_available(cls) -> bool:
        # return CohereDeployment.api_key is not None
        return True

    async def invoke_chat(
        self, chat_request: CohereChatRequest, ctx: Context, **kwargs: Any
    ) -> Any:
        # response = self.client.chat(
        #     **chat_request.model_dump(exclude={"stream", "file_ids", "agent_id"}),
        # )
        # yield to_dict(response)
        response, context = graph_utils.get_result_graphrag(chat_request.message)

        return {"text": response}

    async def invoke_chat_stream(
        self, chat_request: CohereChatRequest, ctx: Context, **kwargs: Any
    ) -> Any:
        # logger = ctx.get_logger()

        # stream = self.client.chat_stream(
        #     **chat_request.model_dump(exclude={"stream", "file_ids", "agent_id"}),
        # )

        # for event in stream:
        #     event_dict = to_dict(event)

        #     event_dict_log = event_dict.copy()
        #     event_dict_log.pop("conversation_id", None)
        #     logger.debug(
        #         event="Chat event",
        #         **event_dict_log,
        #         conversation_id=ctx.get_conversation_id(),
        #     )

        #     logging.error(str(event_dict))

        #     yield event_dict

        stream = await graph_utils.aget_response_stream(chat_request.message)
        context = await stream.__anext__()






        yield {'event_type': 'stream-start', 'generation_id': '123'}

        # async for i in stream:
        #     yield {'event_type': 'text-generation', 'text': i}
        async for chunk in stream:
            logging.warning('CHUNKKKKKK')
            logging.warning(chunk)
            yield {'event_type': 'text-generation', 'text': chunk}

        yield {'event_type': 'stream-end', 'finish_reason': 'COMPLETE', 'response': {'text': 'abc'}}

        # yield {
        #     'event_type': 'stream-end',
        #     'finish_reason': 'COMPLETE',
        #     'response': {
        #         'text': response,
        #         'generation_id':
        #         '123',
        #         'citations': context,
        #         'documents': None,
        #         'is_search_required': None,
        #         'search_queries': None,
        #         'search_results': None,
        #         'finish_reason':
        #         'COMPLETE',
        #         'tool_calls': None,
        #         # 'chat_history': [{'role': 'USER', 'message': 'hi', 'tool_calls': None}, {'role': 'CHATBOT', 'message': 'Hi! How can I help you today?', 'tool_calls': None}, {'role': 'USER', 'message': 'hi', 'tool_calls': None}, {'role': 'CHATBOT', 'message': "Hello! How's it going?", 'tool_calls': None}],
        #         'chat_history': [],
        #         'prompt': None,
        #         'meta': {'api_version': {'version': '1', 'is_deprecated': None, 'is_experimental': None}, 'billed_units': {'input_tokens': 91.0, 'output_tokens': 11.0, 'search_units': None, 'classifications': None}, 'tokens': {'input_tokens': 880.0, 'output_tokens': 74.0}, 'warnings': None}
        #     }
        # }

    async def invoke_rerank(
        self, query: str, documents: List[Dict[str, Any]], ctx: Context, **kwargs: Any
    ) -> Any:
        # response = self.client.rerank(
        #     query=query, documents=documents, model=DEFAULT_RERANK_MODEL
        # )
        # return to_dict(response)
        return None