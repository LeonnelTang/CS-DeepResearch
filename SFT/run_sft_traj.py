import json
import datetime
import os
from typing import List
import logging

# 导入核心引擎 (从顶层包直接导入)
from react_agent import ResearchLoop, llm_executor
# 导入工具 (从工具包导入)
from react_agent.tools import ToolConfig

#导入评估逻辑
from eval.qa_judge import extract_answer

# 配置日志格式和级别
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)  # 获取当前模块的 logger



def load_qa_dataset(filepath: str) -> List[dict]:
    """读取外部QA JSONL文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据集文件: {filepath}")

    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行，防止 json.loads 报错
            if not line:
                continue

            # 将每一行的 JSON 字符串解析为字典并存入列表
            dataset.append(json.loads(line))

    return dataset[100:]

def main():
    """主函数：批量运行评估数据集"""

    # --- 1. 路径配置 ---
    base_dir = "D:\\DissertationProject\\DissertationProject"
    mcp_config_path = os.path.join(base_dir, "react_agent", "tools", "mcp_configs.json")
    dataset_path = os.path.join(base_dir, "dataset", "general_dataset", "sft_candidates.jsonl")  # 替换为你的实际JSON路径
    trajectory_dir = os.path.join(base_dir, "SFT", "general_traj")

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
        correct_answer = item.get("final answer")
        task = question + "\tYour final response should be in the following format: Explanation: {{your explanation for your final answer}} Exact Answer: {{your succinct, final answer}} Confidence: {{your confidence score between 0% and 100% for your answer}}"

        selected_tool_indices = [0, 1, 2, 3]

        if not question:
            logger.warning(f"跳过索引 {index}: 缺少 'question' 字段")
            continue

        logger.info("=" * 60)
        logger.info(f"🚀 开始执行任务 [{index + 1}/{len(qa_data)}]")
        logger.info(f"📝 问题: {question}")
        logger.info(f"🔧 指定工具索引: {selected_tool_indices}")

        # 根据索引获取工具实例 (需要在 ToolConfig 中实现该方法，见下方说明)
        # 如果获取工具的逻辑比较复杂，可以在这里添加 try-except 捕获工具加载错误
        try:
            tools = tool_config.get_tools_by_indices(selected_tool_indices)
            logger.info(f"🛠️ 实际挂载工具: {[t.name for t in tools]}")
        except Exception as e:
            logger.error(f"工具加载失败跳过此任务: {e}")
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
            max_steps=30
        )

        # 执行任务
        final_result, trajectory, chatml_builder = reactor.run(task)

        # 验证结果
        judge_result = extract_answer(question, correct_answer, final_result, "en")
        logger.info(f"QA 验证结果: {judge_result.get('correct', '未知')}")
        if judge_result.get("correct", "") == "yes":
            chatml_builder.save_to_jsonl(trajectory_filename)

        # 输出结果
        logger.info(f"\n✅ 任务 [{index + 1}] 完成！")
        logger.info(f"💡 最终结果：\n{final_result}")
        logger.info(f"💾 轨迹已保存至: {trajectory_filename}")
        logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()