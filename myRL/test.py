import sys
import os


from myRL.tools.base import BaseTool
from myRL.tools.tool_config import ToolConfig

base_dir = "/root"
mcp_config_path = os.path.join(base_dir, "myRL", "tools", "mcp_configs.json")
cache_path = os.path.join(base_dir, "mcp_tools_cache.json")

try:
    tool_config = ToolConfig(mcp_config_path=mcp_config_path, cache_path=cache_path)
    selected_tool_indices = [0, 1, 2, 3]
    tools = tool_config.get_tools_by_indices(selected_tool_indices)
except Exception as e:
    # 建议将 debug_logger 替换为标准的 logging，或确保其已在全局初始化
    print(f"工具加载失败: {e}")
    tools = []

tool_map = {tool.name: tool for tool in tools}

print([tool.name for tool in tools])