import json
from .tools_example import FlightSearchTool, FlightBookTool
import os
from .mcp_tool import MCPToolAdapter, BaseTool
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

load_dotenv()

async def load_mcp_tools(server_url: str, api_key: str) -> list[BaseTool]:
    """连接 MCP 服务器并拉取所有可用工具，转换为 BaseTool 列表"""
    headers = {"Authorization": f"Bearer {api_key}"}
    mcp_base_tools = []

    print(f"正在连接 MCP 服务器加载工具: {server_url} ...")
    async with sse_client(url=server_url, headers=headers) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 获取远程 MCP 服务器提供的所有工具
            mcp_tools = await session.list_tools()

            for t in mcp_tools.tools:
                schema_str = json.dumps(t.inputSchema, ensure_ascii=False)
                tool_description = f"{t.description}\n参数格式: {schema_str}"

                # 将每个远程工具包装成你的 BaseTool
                mcp_base_tools.append(
                    MCPToolAdapter(
                        name=t.name,
                        description=tool_description,
                        server_url=server_url,
                        mcp_tool_name=t.name,
                        api_key=api_key
                    )
                )
    print(f"成功加载 {len(mcp_base_tools)} 个 MCP 工具。")
    return mcp_base_tools


import os
import asyncio


# 假设你的 load_mcp_tools 依然保持原样:
# async def load_mcp_tools(server_url: str, api_key: str) -> list[BaseTool]:
#     ...

async def fetch_all_mcp_tools(mcp_configs: list[dict]) -> list:
    """并发请求多个 MCP URL，支持每个 URL 使用独立的 API Key"""
    tasks = []
    valid_urls = []

    # 1. 遍历配置，构建任务列表
    for config in mcp_configs:
        url = config.get("url")
        api_key = config.get("api_key")

        # 校验：如果没有拿到 API Key，直接跳过并警告
        if not api_key:
            print(f"⚠️ 警告: 未找到 API Key，跳过加载 MCP 节点: {url}")
            continue

        # 把 load_mcp_tools 作为一个异步任务加进去
        tasks.append(load_mcp_tools(url, api_key))
        valid_urls.append(url)  # 记录一下有效 URL，方便后面报错时对应

    # 如果所有配置都没 Key，直接返回空列表
    if not tasks:
        return []

    # 2. 并发执行所有任务 (return_exceptions=True 防止单个失败导致整体崩溃)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. 汇总所有成功加载的工具
    all_mcp_tools = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"❌ 加载 MCP 工具失败 [{valid_urls[i]}]: {res}")
        else:
            all_mcp_tools.extend(res)

    return all_mcp_tools


def get_tool_list():
    # 1. 把 URL 和对应的 API Key 组合成字典列表
    # 从环境变量中分别读取不同的 Key
    mcp_configs = [
        {
            "url": os.getenv("Bailian_WebSearch_MCP_URL"),
            "api_key": os.getenv("DASHSCOPE_API_KEY")
        },
        {
            "url": os.getenv("Bailian_WebParser_MCP_URL"),
            "api_key": os.getenv("DASHSCOPE_API_KEY")
        },
        # {
        #     "url": os.getenv("Bailian_ArxivPaper_MCP_URL"),
        #     "api_key": os.getenv("DASHSCOPE_API_KEY")
        # },
        {
            "url": os.getenv("Bailian_YepCode_MCP_URL"),
            "api_key": os.getenv("DASHSCOPE_API_KEY")
        }
    ]

    # 2. 启动并发加载
    print(f"准备并发加载 {len(mcp_configs)} 个 MCP 配置...")
    mcp_tools = asyncio.run(fetch_all_mcp_tools(mcp_configs))

    # 3. 将 MCP 工具与你原有的本地工具合并
    local_tools = [
        FlightSearchTool(),
        FlightBookTool()
    ]

    all_tools = local_tools + mcp_tools
    print(f"✅ 工具总数: {len(all_tools)} (本地: {len(local_tools)}, MCP: {len(mcp_tools)})")

    return all_tools

if __name__ == "__main__":
    tools = get_tool_list()
    print([t.name for t in tools])