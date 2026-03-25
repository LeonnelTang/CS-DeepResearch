import json
from .local_tool import ArxivSearchEngine, GetRecentAIPapers, ListSkillsTool, GetSkillPromptTool
import os
from .mcp_tool import MCPToolAdapter, BaseTool
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

# 5. 将MCP工具与本地工具合并
local_tools = [
    # ArxivSearchEngine(),
    # GetRecentAIPapers(),
    ListSkillsTool(),
    GetSkillPromptTool()
]

class ToolConfig:
    """工具配置类"""
    def __init__(self, mcp_config_path, cache_path="./mcp_tools_cache.json"):
        self.mcp_config_path = mcp_config_path
        self.all_mcp_configs = []
        self.cache_path = cache_path
        self.all_tool_options = []

        """获取所有MCP配置信息"""
        with open(self.mcp_config_path, "r", encoding="utf-8") as f:
            self.all_mcp_configs = json.load(f)
            self.all_tool_options = [
                {
                    "name": mcp["name"],
                    "description": mcp["description"]
                }
                for mcp in self.all_mcp_configs
            ]
            self.all_tool_options.extend(
                [
                    {
                        "name": tool.name,
                        "description": tool.description
                    }
                    for tool in local_tools
                ]
            )

    async def load_mcp_tools(self, server_url: str, api_key: str) -> list[BaseTool]:
        """连接 MCP 服务器并拉取所有可用工具，转换为 BaseTool 列表"""

        # 1. 尝试读取本地缓存
        cache_data = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
            except Exception as e:
                print(f"⚠️ 读取缓存失败，将重新拉取: {e}")

        mcp_base_tools = []

        # 2. 如果缓存命中，直接使用本地数据构建工具 (跳过网络请求)
        if server_url in cache_data:
            # print(f"⚡ 从本地缓存极速加载 MCP 工具: {server_url}")
            for t_data in cache_data[server_url]:
                mcp_base_tools.append(
                    MCPToolAdapter(
                        name=t_data["name"],
                        description=t_data["description"],
                        server_url=server_url,
                        mcp_tool_name=t_data["mcp_tool_name"],
                        api_key=api_key  # 注意：API Key 从环境变量动态获取，不写入缓存保证安全
                    )
                )
            return mcp_base_tools

        # 3. 如果没有缓存，建立连接拉取远程工具
        headers = {"Authorization": f"Bearer {api_key}"}
        # print(f"正在连接 MCP 服务器加载工具: {server_url} ...")
        server_cache_entry = []  # 用于保存当前服务器的工具元数据

        async with sse_client(url=server_url, headers=headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # 获取远程 MCP 服务器提供的所有工具
                mcp_tools = await session.list_tools()

                for t in mcp_tools.tools:
                    schema_str = json.dumps(t.inputSchema, ensure_ascii=False)
                    tool_description = f"{t.description}\n参数格式: {schema_str}"

                    # 将构建好的工具信息记录下来，准备写入缓存
                    server_cache_entry.append({
                        "name": t.name,
                        "description": tool_description,
                        "mcp_tool_name": t.name
                    })

                    # 将每个远程工具包装成 BaseTool
                    mcp_base_tools.append(
                        MCPToolAdapter(
                            name=t.name,
                            description=tool_description,
                            server_url=server_url,
                            mcp_tool_name=t.name,
                            api_key=api_key
                        )
                    )

        # 4. 更新缓存文件
        cache_data[server_url] = server_cache_entry
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)

        print(f"✅ 成功加载并缓存 {len(mcp_base_tools)} 个 MCP 工具。")
        return mcp_base_tools

    async def fetch_all_mcp_tools(self, mcp_configs: list[dict]) -> list:
        """并发请求多个 MCP URL，支持每个 URL 使用独立的 API Key"""
        tasks = []
        valid_urls = []

        # 1. 遍历配置，构建任务列表
        for config in mcp_configs:
            url = os.getenv(config.get("env_url"))
            api_key = os.getenv(config.get("env_api_key"))

            # 校验：如果没有拿到 API Key，直接跳过并警告
            if not api_key:
                print(f"⚠️ 警告: 未找到 API Key，跳过加载 MCP 节点: {url}")
                continue

            # 把 load_mcp_tools 作为一个异步任务加进去
            tasks.append(self.load_mcp_tools(url, api_key))
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

    def config_tools(self, selected_mcp_names: list[str] = None):
        """根据选择的MCP名称加载工具配置"""

        # 如果没有指定选择，则加载所有可用的MCP
        if selected_mcp_names is None:
            mcp_configs = self.all_mcp_configs
        else:
            # 3. 根据用户选择过滤MCP配置
            mcp_configs = [config for config in self.all_mcp_configs
                          if config["name"] in selected_mcp_names]

            # 检查是否有未找到的MCP名称
            found_names = [config["name"] for config in mcp_configs]
            missing_names = set(selected_mcp_names) - set(found_names)
            if missing_names:
                print(f"⚠️ 以下MCP未找到配置: {missing_names}")

        # 4. 启动并发加载
        if mcp_configs:
            # print(f"准备并发加载 {len(mcp_configs)} 个 MCP 配置...")
            mcp_tools = asyncio.run(self.fetch_all_mcp_tools(mcp_configs))
        else:
            print("没有有效的MCP配置需要加载")
            mcp_tools = []

        all_tools = local_tools + mcp_tools

        # print(f"✅ 工具总数: {len(all_tools)} (本地: {len(local_tools)}, MCP: {len(mcp_tools)})")

        return all_tools

    def get_tools_by_indices(self, indices: list[int]):
        """
        根据传入的索引列表获取对应的MCP工具配置 (用于自动化批量测试)

        Args:
            indices (list[int]): 对应 all_mcp_configs 列表的索引

        Returns:
            list[BaseTool]: 实例化的工具列表
        """
        selected_names = []
        for idx in indices:
            # 确保索引在有效范围内
            if 0 <= idx < len(self.all_mcp_configs):
                selected_names.append(self.all_mcp_configs[idx]["name"])
            else:
                print(f"⚠️ 警告: 工具配置索引 {idx} 超出范围 (最大索引为 {len(self.all_mcp_configs) - 1})，已忽略。")

        if not selected_names:
            print("⚠️ 警告: 未匹配到任何有效的 MCP 配置，将返回空工具列表。")
            return []

        # print(f"🔄 准备加载配置: {selected_names}")
        # 复用现有的 config_tools 方法加载并初始化工具
        tools = self.config_tools(selected_names)
        return tools

    async def interactive_select_tools(self):
        """交互式选择要加载的MCP工具"""
        # 打印所有可用的MCP工具信息供用户选择
        print("可用的MCP工具:")
        for i, config_info in enumerate(self.all_mcp_configs):
            print(f"{i + 1}. {config_info['name']}: {config_info['description']}")

        # 获取用户选择
        selected_tool_ids = input("请输入要加载的工具编号(用逗号分隔，留空则加载全部，输入'none'则不加载任何工具): ").strip()

        if selected_tool_ids.lower() == 'none':
            # 用户明确表示不加载任何MCP工具
            selected_names = []
        elif selected_tool_ids:
            # 解析用户输入的编号
            try:
                indices = [int(tool_id.strip()) - 1 for tool_id in selected_tool_ids.split(",")]
                selected_names = [self.all_mcp_configs[i]["name"] for i in indices if
                                  0 <= i < len(self.all_mcp_configs)]
            except ValueError:
                print("输入格式错误，将加载所有工具")
                selected_names = None
        else:
            selected_names = None

        # 根据用户选择加载工具
        tools = await self.config_tools(selected_names)
        print([t.name for t in tools])
        return tools

if __name__ == "__main__":
    tool_config = ToolConfig("react_agent\\react_tools\\mcp_configs.json", cache_path="D:\DissertationProject\DissertationProject\\backend\mcp_tools_cache.json")
    _ = asyncio.run(tool_config.load_mcp_tools("https://mcp.api-inference.modelscope.net/afba08f1dd5e47/sse", "your_api_key"))
    

