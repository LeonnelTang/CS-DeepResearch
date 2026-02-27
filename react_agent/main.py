# 导入核心引擎 (从顶层包直接导入)
from react_agent import ReactCoreLoop, llm_executor

# 导入工具 (从工具包导入)
from react_agent.tools import ToolConfig
import datetime

def main():
    """主函数：演示ReAct循环的使用"""
    # 初始化工具
    mcp_config_path = "D:\\DissertationProject\\DissertationProject\\react_agent\\tools\\mcp_configs.json"
    tool_config = ToolConfig(mcp_config_path)
    tools = tool_config.interactive_select_tools()

    print("可用工具列表：")
    for tool in tools:
        print(f"- {tool.name}")

    # 定义任务
    task = "有一种多功能的Transformer架构，它通过一种独特的训练策略来同时适应多种图像修复任务。这种训练策略涉及同步更新模型的核心部分和针对特定任务的调整部分。此外，这种方法还基于另一种技术路线（一种为生成模型设计的去噪技术）得到了扩展和应用。请问，这种采用了上述同步训练策略，并且其方法被扩展到另一种去噪技术上的、能处理多种图像修复任务的核心Transformer架构具体是什么？"

    # 轨迹文件名 时间+数据集名字
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "qa_test"
    trajectory_filename = f"D:\\DissertationProject\\DissertationProject\\react_agent\\trajectories\\{timestamp}_{dataset_name}.jsonl"


    # 创建并运行ReAct循环
    reactor = ReactCoreLoop(
        tools=tools,
        llm_executor=llm_executor,
        max_steps=30,
        output_file=trajectory_filename,
    )

    final_result, trajectory = reactor.run(task)

    # 输出结果
    print("\n" + "=" * 50)
    print("最终结果：", final_result)
    print("\n完整执行轨迹：")
    print(trajectory)
    print("=" * 50)


if __name__ == "__main__":
    main()