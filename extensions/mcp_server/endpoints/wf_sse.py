import json
import time
from typing import Mapping
from werkzeug import Request, Response
from dify_plugin import Endpoint
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from mcp import types
from starlette.applications import Starlette
from starlette.routing import Route


class WorkflowEndpoint(Endpoint):
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        Invokes the endpoint with the given request.
        """
        print("request", r)
        print("values", values)
        print("settings", settings)

        def generator():
            for i in range(10):
                time.sleep(1)
                test_data = {
                    "text": f"{i}",
                    "type": "text",
                }
                print(test_data)
                yield f"data: {json.dumps(test_data)}\n\n"

            yield "data: [DONE]\n\n"

        return Response(
            generator(),
            status=200,
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )
        # server = Server('dify-workflow')


        
        # @server.list_tools()
        # async def list_tools() -> list[types.Tool]:
        #     return [
        #         types.Tool(
        #             id="1",
        #             name="Tool 1",
        #         )
        #     ]
