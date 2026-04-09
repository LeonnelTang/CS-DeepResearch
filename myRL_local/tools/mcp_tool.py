import asyncio
import json
from typing import Any, Dict
from mcp import ClientSession
from mcp.client.sse import sse_client
from .base import BaseTool  # 引入你定义的基类


class MCPToolAdapter(BaseTool):
    """MCP 工具的适配器，将远程 MCP 工具包装为本地的 BaseTool"""

    def __init__(self, name: str, description: str, server_url: str, mcp_tool_name: str, api_key: str):
        super().__init__(name, description)
        self.server_url = server_url
        self.mcp_tool_name = mcp_tool_name
        self.api_key = api_key

    async def run(self, params: Any) -> str:
        """同步执行入口，桥接异步的 MCP 调用"""
        # ReAct 循环传进来的 params 可能是 JSON 字符串，也可能是字典，这里做一下兼容
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return f"Error: 无法解析工具参数格式，期望 JSON，实际收到: {params}"

        # 使用 await 执行异步的 MCP 调用
        return await self._async_call_mcp(params)

    async def _async_call_mcp(self, arguments: Dict[str, Any]) -> str:
        """核心异步调用逻辑，通过 SSE 连接 DashScope MCP"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # 如果 DashScope 需要其他的特定的 Header，可以在这里添加
        }

        try:
            # 建立 SSE 连接
            async with sse_client(url=self.server_url, headers=headers) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # 初始化 MCP 会话
                    await session.initialize()

                    # 调用远程工具
                    result = await session.call_tool(self.mcp_tool_name, arguments=arguments)

                    # 解析 MCP 返回的内容
                    output = ""
                    for content in result.content:
                        if content.type == "text":
                            output += content.text

                    if result.isError:
                        return f"MCP 执行报错: {output}"
                    return output

        except Exception as e:
            return f"MCP 连接或执行过程中发生异常: {str(e)}"