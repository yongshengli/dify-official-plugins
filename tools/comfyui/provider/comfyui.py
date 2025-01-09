from typing import Any
import websocket
from yarl import URL
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from dify_plugin import ToolProvider


class ComfyUIProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        ws = websocket.WebSocket()
        base_url = URL(credentials.get("base_url"))
        ws_address = f"ws://{base_url.authority}/ws?clientId=test123"
        try:
            ws.connect(ws_address)
        except Exception:
            raise ToolProviderCredentialValidationError(
                f"can not connect to {ws_address}"
            )
        finally:
            ws.close()