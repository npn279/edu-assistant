from typing import Any, Dict, List

from community.tools import BaseTool
from community.tools.graph_rag.graph_utils import *


class GraphRagTool(BaseTool):
    NAME = "graph_rag"

    def __init__(self):
        pass

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def call(self, parameters: dict, **kwargs: Any) -> List[Dict[str, Any]]:
        query = parameters.get("query", "")

        stream = await aget_response_stream(query)
        return [{'stream_output': stream}]

        # report = await stream.__anext__()
        # print('Report:', report)


        # response = []
        # async for result in stream:
        #     response.append(result)
        #     print(response[-1], end='', flush=True)



