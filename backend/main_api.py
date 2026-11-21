import json
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入你的核心引擎和工具
from react_agent import LLMExecutor
from react_agent import ResearchLoop
from react_agent.tools import ToolConfig

from openai import OpenAI, AsyncOpenAI

client = OpenAI(
    api_key=os.getenv('DMX_API_KEY'),
    base_url=os.getenv('DMX_API_BASE'),
)

app = FastAPI(title="IterResearch API")

# ==========================================
# 1. 配置跨域 (CORS) - 允许前端连接
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境下允许所有端口访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. 初始化工具配置 (全局挂载)
# ==========================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"项目根目录: {base_dir}")
mcp_config_path = os.path.join(base_dir, "react_agent", "tools", "mcp_configs.json")
print(f"MCP 配置文件: {mcp_config_path}")
tool_config = ToolConfig(mcp_config_path, cache_path=os.path.join(base_dir, "backend", "mcp_tools_cache.json"))
print(f"缓存文件: {tool_config.cache_path}")

# ==========================================
# 2.5 获取工具列表接口
# ==========================================
@app.get("/api/tools")
async def get_tools_endpoint():
    """向前端提供可用的工具列表"""
    tools_list = []
    for i, opt in enumerate(tool_config.all_mcp_configs):
        tools_list.append({
            "index": i,
            "name": opt["name"],
            "description": opt["description"]
        })
    return {"tools": tools_list}


# ==========================================
# 3. 核心流式接口 (SSE)
# ==========================================
@app.get("/api/research")
async def research_endpoint(query: str, tools_idx: str = ""):
    """
    接收前端研究请求，返回 SSE 数据流
    参数 query: 研究问题
    参数 tools_idx: 选中的工具索引 (逗号分隔，如 "0,1")
    """

    # 1. 解析前端传来的工具索引
    indices = []
    if tools_idx:
        try:
            indices = [int(i) for i in tools_idx.split(",") if i.strip().isdigit()]
        except ValueError:
            pass

    # 如果没传，默认加载所有工具 (你可以根据需要调整)
    if not indices:
        indices = list(range(len(tool_config.all_mcp_configs)))

    tools = await tool_config.get_tools_by_indices(indices)

    # 2. 构造 SSE 生成器
    def event_stream():
        reactor = ResearchLoop(
            tools=tools,
            llm_executor=LLMExecutor(client=client, model_name=os.getenv("RESEARCH_MODEL", "gpt-3.5-turbo")),
            max_steps=30
        )

        # 遍历 engine 中的流式生成器
        for update_dict in reactor.run_stream(query):
            # SSE 的标准格式是: "data: {json_string}\n\n"
            yield f"data: {json.dumps(update_dict, ensure_ascii=False)}\n\n"

    # 3. 返回 StreamingResponse，指定 media_type 为 event-stream
    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    # 启动命令: python main_api.py
    print("🚀 正在启动 IterResearch FastAPI 后端...")
    uvicorn.run(app, host="127.0.0.1", port=8000)