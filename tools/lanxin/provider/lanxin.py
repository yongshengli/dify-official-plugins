from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from tools.lanxin_group_bot import LanxinGroupBotTool

class LanxinProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        LanxinGroupBotTool()
        pass
