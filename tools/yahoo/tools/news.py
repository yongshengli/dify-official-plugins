from collections.abc import Generator
from typing import Any
import yfinance
from requests.exceptions import HTTPError, ReadTimeout
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool


class YahooFinanceSearchTickerTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        invoke tools
        """
        query = tool_parameters.get("symbol", "")
        if not query:
            yield self.create_text_message("Please input symbol")
            return
        try:
            yield from self.run(ticker=query)
        except (HTTPError, ReadTimeout):
            yield self.create_text_message("There is a internet connection problem. Please try again later.")

    def run(self, ticker: str) -> Generator[ToolInvokeMessage, None, None]:
        company = yfinance.Ticker(ticker)
        try:
            if company.isin is None:
                yield self.create_text_message(f"Company ticker {ticker} not found.")
        except (HTTPError, ReadTimeout, ConnectionError):
            yield self.create_text_message(f"Company ticker {ticker} not found.")
        links = []
        try:
            links = [n["link"] for n in company.news if n["type"] == "STORY"]
        except (HTTPError, ReadTimeout, ConnectionError):
            if not links:
                yield self.create_text_message(f"There is nothing about {ticker} ticker")
                return
        if not links:
            yield self.create_text_message(f"No news found for company that searched with {ticker} ticker.")
            return
        yield self.create_json_message({"links": links})
