# 导入核心引擎 (从顶层包直接导入)
import asyncio
from openai import OpenAI
import os

from react_agent import ResearchLoop, LLMExecutor

# 导入工具 (从工具包导入)
from react_agent.tools import ToolConfig
import datetime

my_client = OpenAI(
    api_key=None,
    base_url="http://127.0.0.1:6009/v1"
)

dmx_client = OpenAI(
    api_key=os.getenv('MY_DMX_API_KEY'),
    base_url=os.getenv('DMX_API_BASE')
)

def main():
    """主函数：演示ReAct循环的使用"""
    # 初始化工具
    mcp_config_path = "D:\\DissertationProject\\DissertationProject\\react_agent\\tools\\mcp_configs.json"
    tool_config = ToolConfig(mcp_config_path)
    tools = asyncio.run(tool_config.interactive_select_tools())

    print("可用工具列表：")
    for tool in tools:
        print(f"- {tool.name}")

    # 定义任务
    task = """BrowseComp-ZH最新的评测榜单"""

    # 轨迹文件名 时间+数据集名字
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "research_test"
    trajectory_filename = f"D:\\DissertationProject\\DissertationProject\\react_agent\\trajectories\\{timestamp}_{dataset_name}.jsonl"

    reactor = ResearchLoop(
        tools=tools,
        llm_executor=LLMExecutor(client=dmx_client, model_name="qwen3.5-plus"),
        max_steps=30
    )

    print("\n" + "=" * 50)
    print(f"开始执行 IterResearch 深度研究任务...\n目标: {task}")
    print("=" * 50 + "\n")

    final_result, trajectory, chatml_builder = reactor.run(task)

    # 输出结果
    print("\n" + "=" * 50)
    print("最终结果：", final_result)
    # print("\n完整执行轨迹：")
    # print(trajectory)
    chatml_builder.save_to_jsonl(trajectory_filename)
    print("=" * 50)


if __name__ == "__main__":
    main()