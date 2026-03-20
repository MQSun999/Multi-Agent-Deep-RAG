"""Financial Datasets MCP module with LangChain integration."""

######## MCP SETUP ###############
# MCP GITHUB
# https://github.com/langchain-ai/langchain-mcp-adapters
# https://github.com/alphavantage/alpha_vantage_mcp

import warnings

warnings.filterwarnings("ignore")

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")


system_prompt = """
                You are a financial research assistant. Always call tools first before answering.

                ## Tools
                - `get_current_stock_price(ticker)` — latest stock price
                - `get_historical_stock_prices(ticker, start_date, end_date)` — price history (YYYY-MM-DD)
                - `get_income_statements(ticker, period, limit)` — revenue, net income, EPS (period: annual/quarterly)
                - `get_balance_sheets(ticker, period, limit)` — assets, liabilities, equity
                - `get_cash_flow_statements(ticker, period, limit)` — operating/free cash flow
                - `get_company_news(ticker, limit)` — latest news
                - `get_current_crypto_price(symbol)` — latest crypto price
                - `get_historical_crypto_prices(symbol, start_date, end_date)` — crypto price history
                - `get_available_crypto_tickers()` — list supported crypto symbols

                ## Rules
                1. **Always fetch data first**, never answer from memory.
                2. For comprehensive analysis, call: current price → historical prices → income statements → news.
                3. If unsure of a crypto symbol, call `get_available_crypto_tickers` first.
                4. Default historical window: last 3 months unless specified.
                5. If a tool returns an error, tell the user clearly — never fabricate data.
                6. Present results with % changes, trends, and a brief 2-3 sentence insight summary.
                """

from pathlib import Path
UV_PATH = Path(r"C:\Users\18911\.local\bin\uv.exe")
FINANCIAL_DATASETS_MCP_SERVER_DIR = Path(
    r"E:\Udemy\mcp_server\financial_datasets_mcp_server\mcp-server"
)
FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
async def get_tools():
    client = MultiServerMCPClient(
        {
            "financial-datasets": {
                "command": str(UV_PATH),
                "args": [
                    "--directory",
                    str(FINANCIAL_DATASETS_MCP_SERVER_DIR),
                    "run",
                    "server.py",
                ],
                "env": {"FINANCIAL_DATASETS_API_KEY": FINANCIAL_DATASETS_API_KEY},
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    print(f"Loaded {len(tools)} tools")
    print(f"Tools available: {[tool.name for tool in tools]}")

    return tools


async def finance_research(query):
    tools = await get_tools()

    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    result = await agent.ainvoke({"messages": [HumanMessage(query)]})

    response = result["messages"][-1].text

    print(response)

    return response


if __name__ == "__main__":
    #query = "Show me the current price of Nvidia stock."
    query = "What is the current stock price and recent performance of Apple (AAPL)? Also show me the latest news."

    asyncio.run(finance_research(query))
