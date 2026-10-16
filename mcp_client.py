import os
import asyncio
from langchain.mcp import MCPAdapter


async def main():
    client = MCPAdapter(
        {
            "finsight": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_server.py"],
            }
        }
    )

    tools= await client.list_tools()
    print("/nAvailable Tools.....")
    for tool in tools:
        print(f"Name: {tool.name}")
        print(f"Description: {tool.description}")
        print(f"Arguments: {tool.args}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())