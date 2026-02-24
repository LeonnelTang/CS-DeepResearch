# 导入核心引擎 (从顶层包直接导入)
from react_agent import ReactCoreLoop, openai_executor

# 导入工具 (从工具包导入)
from react_agent.tools import get_tool_list
import datetime

def main():
    """主函数：演示ReAct循环的使用"""
    # 初始化工具
    tools = get_tool_list()

    # 定义任务
    task = "有一种技术被设计用于处理极长的文本输入窗口，但其性能并非仅由长度决定，上下文中的信息密度对其影响更大。为了全面评估这种技术，研究人员开发了一个将数据库推理操作引入文本上下文的新框架。在该框架的测试结论中，出现了一个有趣的对比：涉及查找最大值或最小值的查询任务受上下文长度影响较小，而需要聚合多条信息的任务准确率则随长度增加显著下降。请问这个评估框架的名称是什么？"

    # 轨迹文件名 时间+数据集名字
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "search_test"
    trajectory_filename = f"trajectories/{timestamp}_{dataset_name}.jsonl"


    # 创建并运行ReAct循环
    reactor = ReactCoreLoop(
        tools=tools,
        llm_executor=openai_executor,
        max_steps=10,
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