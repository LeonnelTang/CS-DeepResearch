import json
import datetime
import os
from typing import List

# 导入核心引擎 (从顶层包直接导入)
from react_agent import ResearchLoop, llm_executor
# 导入工具 (从工具包导入)
from react_agent.tools import ToolConfig

#导入评估逻辑
from eval.qa_judge import extract_answer


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

    return dataset

def main():
    """主函数：批量运行评估数据集"""

    # --- 1. 路径配置 ---
    base_dir = "D:\\DissertationProject\\DissertationProject"
    mcp_config_path = os.path.join(base_dir, "react_agent", "tools", "mcp_configs.json")
    dataset_path = os.path.join(base_dir, "dataset", "general_dataset", "sft_candidates.jsonl")  # 替换为你的实际JSON路径
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
        correct_answer = item.get("final answer")
        task = question + "\tYour final response should be in the following format: Explanation: {{your explanation for your final answer}} Exact Answer: {{your succinct, final answer}} Confidence: {{your confidence score between 0% and 100% for your answer}}"

        selected_tool_indices = [0, 1, 2, 3]

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
            max_steps=30
        )

        # 执行任务
        final_result, trajectory, chatml_builder = reactor.run(task)

        # 验证结果
        judge_result = extract_answer(question, correct_answer, final_result, "en")
        print(f"QA 验证结果: {judge_result.get('correct', '未知')}")
        if judge_result.get("correct", "") == "yes":
            chatml_builder.save_to_jsonl(trajectory_filename)

        # 输出结果
        print(f"\n✅ 任务 [{index + 1}] 完成！")
        print(f"💡 最终结果：\n{final_result}")
        print(f"💾 轨迹已保存至: {trajectory_filename}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # main()

    import time

    start = time.time()
    judge_result = extract_answer(
        question="A mystery begins in a state whose flag bears an Osage shield, where a humorist known for his rope tricks and folksy wisdom was born, who shared a tragic fate with a pioneering aviatrix that vanished over the Pacific, whose academic patronage came from an institution famed for its engineering programs and a astronaut alumnus who took humanity's first steps on another world, ultimately leading to the vessel that carried him there.",
        correct_answer="Apollo 11 Command Module (Columbia)",
        response="Explanation: 谜题从描述一个州旗带有奥萨格盾牌的州开始，即俄克拉荷马州。该州出生的以绳索技巧和民间智慧闻名的幽默家是威尔·罗杰斯（Will Rogers），他于1935年因飞机失事身亡。他与在太平洋上空失踪的先驱女飞行员阿梅莉亚·埃尔哈特（Amelia Earhart）共享悲剧性的航空事故命运。罗杰斯的“学术资助”指向麻省理工学院（MIT），一所以工程闻名的机构，其仪器实验室开发了阿波罗制导计算机，对阿波罗计划至关重要。MIT的宇航员校友巴兹·奥尔德林（Buzz Aldrin）是阿波罗11号登月舱飞行员，第二个在月球上行走的人，MIT的贡献使人类在月球上的第一步成为可能。最终，线索指向将宇航员带到月球的飞船：阿波罗11号登月舱“鹰”（Apollo 11 Lunar Module \"Eagle\"）。Exact Answer: Apollo 11 Lunar Module \"Eagle\" Confidence: 95%",
        language="en"
    )
    end = time.time()
    print(f"QA 验证耗时: {end - start}s")

    print(judge_result)

    print(f"QA 验证结果: {judge_result.get('correct', '未知')}")