import json
import datetime
import os
from typing import List

# 导入核心引擎 (从顶层包直接导入)
from react_agent import ResearchLoop, llm_executor
# 导入工具 (从工具包导入)
from react_agent.tools import ToolConfig


def load_qa_dataset(filepath: str) -> List[dict]:
    """读取外部QA JSON文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据集文件: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数：批量运行评估数据集"""

    # --- 1. 路径配置 ---
    base_dir = "D:\\DissertationProject\\DissertationProject"
    mcp_config_path = os.path.join(base_dir, "react_agent", "tools", "mcp_configs.json")
    dataset_path = os.path.join(base_dir, "dataset", "qa_dataset.json")  # 替换为你的实际JSON路径
    trajectory_dir = os.path.join(base_dir, "SFT", "trajectories")

    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    # --- 2. 初始化工具配置 ---
    tool_config = ToolConfig(mcp_config_path)

    # --- 3. 加载数据集 ---
    try:
        qa_data = load_qa_dataset(dataset_path)
        print(f"成功加载数据集，共 {len(qa_data)} 条测试用例。\n")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    dataset_name = "sft_traj"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 4. 批量执行 ---
    for index, item in enumerate(qa_data):
        question = item.get("question")
        selected_tool_indices = item.get("selected_tools", [])

        if not question:
            print(f"跳过索引 {index}: 缺少 'question' 字段")
            continue

        print("=" * 60)
        print(f"🚀 开始执行任务 [{index + 1}/{len(qa_data)}]")
        print(f"📝 问题: {question}")
        print(f"🔧 指定工具索引: {selected_tool_indices}")

        # 根据索引获取工具实例 (需要在 ToolConfig 中实现该方法，见下方说明)
        # 如果获取工具的逻辑比较复杂，可以在这里添加 try-except 捕获工具加载错误
        try:
            tools = tool_config.get_tools_by_indices(selected_tool_indices)
            print(f"🛠️ 实际挂载工具: {[t.name for t in tools]}")
        except Exception as e:
            print(f"工具加载失败跳过此任务: {e}")
            continue

        # 为当前任务生成独立的轨迹文件 (方便后续进行数据清洗和 SFT 训练集构建)
        trajectory_filename = os.path.join(
            trajectory_dir,
            f"{timestamp}_{dataset_name}_task{index + 1}.jsonl"
        )

        # 创建并运行 ResearchLoop
        reactor = ResearchLoop(
            tools=tools,
            llm_executor=llm_executor,
            max_steps=30,
            output_file=trajectory_filename,
        )

        # 执行任务
        final_result, trajectory = reactor.run(question)

        # 输出结果
        print(f"\n✅ 任务 [{index + 1}] 完成！")
        print(f"💡 最终结果：\n{final_result}")
        print(f"💾 轨迹已保存至: {trajectory_filename}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()