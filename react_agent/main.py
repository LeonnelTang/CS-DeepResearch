# 导入核心引擎 (从顶层包直接导入)
from react_agent import ReactCoreLoop, openai_executor

# 导入工具 (从工具包导入)
from react_agent.tools import FlightSearchTool, FlightBookTool


def main():
    """主函数：演示ReAct循环的使用"""
    # 初始化工具
    tools = [FlightSearchTool(), FlightBookTool()]

    # 定义任务
    task = "查询明天从深圳到海南的航班，选最便宜、航班时间在晚上的那班并预订"

    # 创建并运行ReAct循环
    reactor = ReactCoreLoop(
        tools=tools,
        llm_executor=openai_executor,
        max_steps=10
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