# 导入核心引擎 (从顶层包直接导入)
from react_agent import ResearchLoop, llm_executor

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
    task = """目前的智能体架构：以 IterResearch 为核心的马尔可夫决策过程（MDP）框架。该框架通过“演进式报告”取代线性上下文积累，理论上可实现无限的研究深度。
IterResearch 范式参考 WebResearcher 论文，将深度研究过程重构为三个结构化元信息循环：
Think：认知草稿纸，用于内部推理。
Report：演进式报告，作为智能体的“核心记忆”进行迭代合成。
Action：工具调用或生成最终答案。
这种方法确保了无论研究轮次如何增加，工作空间（Workspace）的大小始终恒定，维持了稳定的推理能力。

你觉得在强化学习时，应该使用答案正确性的二元奖励，还是过程奖励，请给出详细的讲解"""

    # 轨迹文件名 时间+数据集名字
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "research_test"
    trajectory_filename = f"D:\\DissertationProject\\DissertationProject\\react_agent\\trajectories\\{timestamp}_{dataset_name}.jsonl"

    reactor = ResearchLoop(
        tools=tools,
        llm_executor=llm_executor,
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