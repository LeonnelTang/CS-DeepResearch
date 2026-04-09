import asyncio
import json
import datetime
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from openai import OpenAI

# 导入核心引擎 (从顶层包直接导入)
from react_agent import ResearchLoop, LLMExecutor
# 导入工具 (从工具包导入)
from react_agent.tools import ToolConfig

# 导入评估逻辑
from eval.qa_judge import extract_answer

# 配置日志格式和级别
log_filename = f"eval_traj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    """读取外部QA数据集文件，支持JSONL和JSON格式"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据集文件: {filepath}")

    dataset = []

    # 根据文件扩展名判断格式
    if filepath.endswith('.json'):
        # JSON 格式
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                dataset = data
            else:
                raise ValueError("JSON文件格式错误：根节点应为数组")
    elif filepath.endswith('.jsonl'):
        # JSONL 格式
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dataset.append(json.loads(line))
    else:
        raise ValueError(f"不支持的文件格式: {filepath}，仅支持 .json 和 .jsonl")

    return dataset


DMX_client = OpenAI(
    base_url=os.getenv("DMX_API_BASE"),
    api_key=os.getenv("DMX_API_KEY")
)

Ali_client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

my_client = OpenAI(
    api_key=None,
    base_url="http://127.0.0.1:6009/v1"
)

async def process_single_task(index: int, item: dict, tool_config: ToolConfig, trajectory_dir: str, dataset_name: str, timestamp: str):
    """处理单个任务的独立工作函数"""
    question = item.get("question")
    correct_answer = item.get("answer")
    task = question + "\tYour final response should be in the following format: Explanation: {{your explanation for your final answer}} Exact Answer: {{your succinct, final answer}} Confidence: {{your confidence score between 0% and 100% for your answer}}"

    selected_tool_indices = [-1]

    if not question:
        return f"⚠️ 跳过索引 {index}: 缺少 'question' 字段"

    logger.info(f"🚀 开始执行任务 [{index + 1}]...")

    try:
        tools = await tool_config.get_tools_by_indices(selected_tool_indices)
    except Exception as e:
        return f"❌ 任务 [{index + 1}] 工具加载失败跳过此任务: {e}"

    trajectory_filename = os.path.join(
        trajectory_dir,
        f"{timestamp}_{dataset_name}_task{index + 1}.jsonl"
    )

    # 创建并运行 ResearchLoop
    reactor = ResearchLoop(
        tools=tools,
        llm_executor=LLMExecutor(client=my_client, model_name="IterScholar-CS-35B-A3B-SFT"),
        max_steps=30
    )

    try:
        # 执行任务
        final_result, trajectory, chatml_builder = await asyncio.to_thread(reactor.run, task)

        # 全部保存
        chatml_builder.save_to_jsonl(trajectory_filename)


        return_result = {
            "question": question,
            "prediction": final_result
        }

        return return_result

    except Exception as e:
        return_result = {
            "question": question,
            "prediction": None
        }

        return return_result


async def main():
    """主函数：批量运行评估数据集 (并发版本)"""

    # --- 1. 路径配置 ---
    base_dir = "D:\\DissertationProject\\DissertationProject"
    mcp_config_path = os.path.join(base_dir, "react_agent", "tools", "mcp_configs.json")
    dataset_path = os.path.join(base_dir, "dataset", "BrowseComp-ZH", "browsecomp-zh-sample-50.json")
    trajectory_dir = os.path.join(base_dir, "eval", "eval_traj", "BrowseComp-ZH_traj", "SFT_IterScholar")


    dataset_name = "BrowseComp-ZH_traj"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 动态生成文件名，加入时间戳和数据集名称，防止结果被覆盖
    result_filename = f"{timestamp}_{dataset_name}_eval_results.json"
    result_filepath = os.path.join(trajectory_dir, result_filename)

    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)

    # --- 2. 初始化工具配置 ---
    tool_config = ToolConfig(mcp_config_path, cache_path="mcp_tools_cache.json")

    # --- 3. 加载数据集 ---
    try:
        qa_data = load_qa_dataset(dataset_path)
        logger.info(f"成功加载数据集，共 {len(qa_data)} 条测试用例。\n")
        q_to_id = {item['question'].strip(): item['id'] for item in qa_data}
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return

    # load only questions without responses
    if os.path.exists(r'D:\DissertationProject\DissertationProject\eval\BrowseComp-ZH\prediction_results\formatted_eval_results.json'):
        with open(r'D:\DissertationProject\DissertationProject\eval\BrowseComp-ZH\prediction_results\formatted_eval_results.json', "r", encoding='utf-8') as f:
            predictions = json.load(f)
        qa_data = [q for q in qa_data if str(q["id"]) not in predictions]
        logger.info(f"当前还需处理 {len(qa_data)} 条测试用例。\n")
    else:
        predictions = {}


    # --- 4. 并发执行 ---
    # 注意：max_workers 决定了并发数量。请根据你的 LLM API 的并发限制 (Rate Limit) 进行调整。
    # 如果设置过高，可能会导致触发 HTTP 429 Too Many Requests 错误。
    max_workers = 1
    logger.info(f"⚡ 启动异步任务，最大并发数: {max_workers}")
    logger.info("=" * 60)

    save_results = []

    # 使用 Semaphore 控制异步并发数量
    semaphore = asyncio.Semaphore(max_workers)

    async def bounded_task(index, item):
        """用信号量包装单个任务，以控制并发"""
        async with semaphore:
            return await process_single_task(
                item["id"], item, tool_config, trajectory_dir, dataset_name, timestamp
            )

    # 创建所有协程任务
    tasks = [
        bounded_task(index, item)
        for index, item in enumerate(qa_data)
    ]

    # 使用 asyncio.as_completed 获取并打印结果 (任务完成时立即 yield)
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            logger.info(result.get("message"))
            logger.info("-" * 60)

            save_results.append(
                {
                    "question": result.get("question"),
                    "prediction": result.get("prediction")
                }
            )

        except Exception as e:
            logger.error(f"任务执行出现未捕获的异常: {e}")
            logger.error("-" * 60)

    logger.info("\n🎉 所有并发任务执行完毕！")

    # --- 保存最终评估结果 ---
    try:
        missing_count = 0

        for entry in save_results:
            # 这里的 question 字段可能在评估结果的字典里
            q_text = entry.get('question', '').strip()
            if q_text in q_to_id:
                task_id = q_to_id[q_text]
                predictions[task_id] = entry
            else:
                missing_count += 1

        with open(result_filepath, "w", encoding="utf-8") as f:
            # 直接使用 json.dump 序列化整个列表，确保生成标准的 JSON 格式
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 所有评估结果已成功保存至: {result_filepath}")
        print(f"\n处理完成！")
        print(f"- 成功匹配并转换后总共: {len(predictions)} 条")
        print(f"- 未能匹配条数: {missing_count}")
        print(f"- 结果已保存至: {result_filepath}")
    except Exception as e:
        logger.error(f"❌ 保存评估结果时发生错误: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())