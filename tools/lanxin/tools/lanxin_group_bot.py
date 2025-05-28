from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
import httpx
from . import lanxin_api_utils as lanxin_utils

class LanxinGroupBotTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        invoke tools
        LanXin custom group robot API docs:
        https://developer.lanxin.cn/official/article?module=back-end-api&article_id=646eda463d4e4adb7039c150
        """
        content = tool_parameters.get("content")
        if not content:
            yield self.create_text_message("Invalid parameter content")
        api_url = tool_parameters.get("api_url")
        if not api_url:
            yield self.create_text_message(
                "Invalid parameter api_url. Regarding information about security details,please refer to the LanXin docs:https://developer.lanxin.cn/official/article?module=back-end-api&article_id=646eda463d4e4adb7039c150"
            )
        hook_token = tool_parameters.get("hook_token", "")
        if not hook_token:
            yield self.create_text_message(
                f"Invalid parameter hook_token, Regarding information about security details,please refer to the DingTalk docs:https://developer.lanxin.cn/official/article?module=back-end-api&article_id=646eda463d4e4adb7039c150"
            )
            return
        sign_secret = tool_parameters.get("sign_secret")
        if not sign_secret:
            yield self.create_text_message(
                "Invalid parameter sign_secret. Regarding information about security details,please refer to the DingTalk docs:https://developer.lanxin.cn/official/article?module=back-end-api&article_id=646eda463d4e4adb7039c150"
            )
        headers = {"Content-Type": "application/json"}
        sign, timestamp = lanxin_utils.gen_sign(sign_secret)
        params = {
            'hook_token':hook_token
        }
        payload = {
            "sign": sign,
            "timestamp": timestamp,
            "msgType": "text",
            "msgData": {"text": {"content": content}}
        }
        print(payload)
        try:
            res = httpx.post(api_url, headers=headers,params=params, json=payload)
            if res.is_success:
                result = res.json()
                if result.get("errCode") == 0:
                    yield self.create_text_message("Text message sent successfully")
                else:
                    yield self.create_text_message(
                        f"Failed to send the text message, status code: {result.get("errCode")}, response: {result.get("errMsg")}"
                    ) 
            else:
                yield self.create_text_message(
                    f"Failed to send the text message, status code: {res.status_code}, response: {res.text}"
                )
        except Exception as e:
            yield self.create_text_message(
                "Failed to send message to group chat bot. {}".format(e)
            )
