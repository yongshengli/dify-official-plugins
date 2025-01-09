from typing import Any
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from dify_plugin import ToolProvider
from tools.fishaudio import FishAudio

class FishaudioProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        api_key = credentials.get("api_key")
        api_base = credentials.get("api_base")
        client = FishAudio(api_key=api_key, url_base=api_base)
        try:
            client.model_list(10)
        except Exception:
            raise ToolProviderCredentialValidationError("Fish audio API key is invalid")
