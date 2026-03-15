import json
import datetime
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 导入核心引擎 (从顶层包直接导入)
from react_agent import ResearchLoop, llm_executor
# 导入工具 (从工具包导入)
from react_agent.tools import ToolConfig

# 导入评估逻辑
from eval.qa_judge import extract_answer

# 配置日志格式和级别
log_filename = f"sft_traj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

# 屏蔽 httpx 库的 HTTP 请求日志
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)  # 获取当前模块的 logger


def load_qa_dataset(filepath: str) -> List[dict]:
    """读取外部QA JSONL文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据集文件: {filepath}")

    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dataset.append(json.loads(line))

    return dataset


def process_single_task(index: int, item: dict, tool_config: ToolConfig, trajectory_dir: str, dataset_name: str, timestamp: str):
    """处理单个任务的独立工作函数"""
    question = item.get("question")
    correct_answer = item.get("answer")
    task = question + "\tYour final response should be in the following format: Explanation: {{your explanation for your final answer}} Exact Answer: {{your succinct, final answer}} Confidence: {{your confidence score between 0% and 100% for your answer}}"

    selected_tool_indices = [0, 1, 2, 3]

    if not question:
        return f"⚠️ 跳过索引 {index}: 缺少 'question' 字段"

    logger.info(f"🚀 开始执行任务 [{index + 1}]...")

    try:
        tools = tool_config.get_tools_by_indices(selected_tool_indices)
    except Exception as e:
        return f"❌ 任务 [{index + 1}] 工具加载失败跳过此任务: {e}"

    trajectory_filename = os.path.join(
        trajectory_dir,
        f"{timestamp}_{dataset_name}_task{index + 1}.jsonl"
    )

    # 创建并运行 ResearchLoop
    reactor = ResearchLoop(
        tools=tools,
        llm_executor=llm_executor,
        max_steps=100
    )

    try:
        # 执行任务
        final_result, trajectory, chatml_builder = reactor.run(task)

        # 验证结果
        judge_result = extract_answer(question, correct_answer, final_result, "en")
        is_correct = judge_result.get("correct", "") == "yes"

        if is_correct:
            chatml_builder.save_to_jsonl(trajectory_filename)

        # 格式化返回结果
        result_msg = (
            f"✅ 任务 [{index + 1}] 完成！ (QA 验证: {judge_result.get('correct', '未知')})\n"
            f"💡 最终结果：\n{final_result}\n"
        )
        if is_correct:
            result_msg += f"💾 轨迹已保存至: {trajectory_filename}"
        else:
            result_msg += "⚠️ 答案不匹配，轨迹未保存。"

        return result_msg

    except Exception as e:
        return f"❌ 任务 [{index + 1}] 运行过程中发生异常: {str(e)}"


def main():
    """主函数：批量运行评估数据集 (并发版本)"""

    # --- 1. 路径配置 ---
    base_dir = "D:\\DissertationProject\\DissertationProject"
    mcp_config_path = os.path.join(base_dir, "react_agent", "tools", "mcp_configs.json")
    dataset_path = os.path.join(base_dir, "dataset", "CS_dataset", "splits", "sft_seeds_100.jsonl")
    trajectory_dir = os.path.join(base_dir, "SFT", "trajectories", "CS_traj")

    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    # --- 2. 初始化工具配置 ---
    tool_config = ToolConfig(mcp_config_path)

    # --- 3. 加载数据集 ---
    try:
        qa_data = load_qa_dataset(dataset_path)
        logger.info(f"成功加载数据集，共 {len(qa_data)} 条测试用例。\n")
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return

    dataset_name = "CS_sft_traj"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 4. 并发执行 ---
    # 注意：max_workers 决定了并发数量。请根据你的 LLM API 的并发限制 (Rate Limit) 进行调整。
    # 如果设置过高，可能会导致触发 HTTP 429 Too Many Requests 错误。
    max_workers = 20
    logger.info(f"⚡ 启动线程池，最大并发数: {max_workers}")
    logger.info("=" * 60)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        futures = {
            executor.submit(
                process_single_task,
                index, item, tool_config, trajectory_dir, dataset_name, timestamp
            ): index
            for index, item in enumerate(qa_data)
        }

        # 获取并打印结果 (as_completed 会在任务完成时立即 yield)
        for future in as_completed(futures):
            try:
                result = future.result()
                logger.info(result)
                logger.info("-" * 60)
            except Exception as e:
                logger.error(f"任务执行出现未捕获的异常: {e}")
                logger.error("-" * 60)

    logger.info("\n🎉 所有并发任务执行完毕！")


if __name__ == "__main__":
    main()