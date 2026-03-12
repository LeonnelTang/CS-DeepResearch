from typing import List, Tuple, Callable
from react_agent.tools import BaseTool
from .parser import LLMOutputParser
from .memory import StateManager
from .chatml import ChatMLBuilder
from datetime import datetime


class ResearchLoop:
    """Research 核心循环控制器"""

    def __init__(self,
                 tools: List[BaseTool],
                 llm_executor: Callable[[str], str],
                 max_steps: int = 10
                 ):

        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.llm_executor = llm_executor
        self.parser = LLMOutputParser()

        # 基于 IterResearch 的 Prompt 模板
        self.xml_prompt = """你是一个深度研究智能体。你需要通过“思考→报告→行动”(Think-Report-Action) 的迭代循环来解决复杂问题。
你没有历史对话记录！你唯一的知识载体和上下文是“全局演进报告 (report)”。你必须在每一次迭代中，将新线索与旧报告进行深度语义融合、去重和提炼，而非简单堆砌。

**输出结构要求 (必须且只能包含以下三个 XML 标签，禁止输出任何标签外的文本)**：
1. <think> 认知暂存区：分析当前工作区状态，评估上次行动的结果(observation)，准备更新报告 (report)并推理下一步工具调用策略(action)。此部分属于你的内部推理，不会带入下一轮。</think>
2. <report> 演进报告 (中心记忆)：这是你唯一的上下文记忆！请将本次新发现与上一轮的旧报告进行综合，输出一份结构化、高信息密度的最新进展摘要。禁止简单追加，必须进行语义融合。</report>
3. <action> 行动：仅使用提供的工具，格式为"工具名[参数]"。如果认为已有足够证据回答初始问题，请输出"finish[最终回复]"，这是你向用户展示结果的唯一渠道。</action>

支持工具：
{tool_descriptions}

原始研究任务 (Question)：{task}

【当前工作区状态】
当前时间：{current_datetime}
上一轮报告 (Report_{{i-1}})：
{report}

上一轮行动 (Action_{{i-1}})：{prev_action}
上一轮结果 (Observation_{{i-1}})：{prev_observation}

请输出当前步骤的 <think>, <report> 和 <action> (仅输出XML标签)：
"""

    def _build_prompt(self, task: str, workspace_state: dict) -> str:
        """构建 IterResearch 的 Prompt"""
        tool_descriptions = "\n".join([
            f"- {name}：{tool.description}"
            for name, tool in self.tool_map.items()
        ])

        return self.xml_prompt.format(
            tool_descriptions=tool_descriptions,
            task=task,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report=workspace_state["report"],
            prev_action=workspace_state["prev_action"],
            prev_observation=workspace_state["prev_observation"]
        ).strip()

    def run(self, task: str) -> Tuple[str, List[dict], ChatMLBuilder]:
        workspace = StateManager()
        chatml_builder = ChatMLBuilder()

        for step in range(self.max_steps):
            # 1. 获取当前 MDP 状态并构建 prompt
            current_state = workspace.get_current_state()
            prompt = self._build_prompt(task, current_state)

            chatml_builder.add_message("user", prompt)

            # 2. 调用 LLM
            llm_output = self.llm_executor(prompt)
            chatml_builder.add_message("assistant", llm_output)

            try:
                # 3. 解析 Think, Report, Action
                think, report, action = self.parser.parse_tra(llm_output)

                # 如果模型没有生成报告，维持记忆
                if not report:
                    report = current_state["report"]

                # 4. 执行行动并获取观察
                observation = self._execute_action(action)

                # 5. 重构状态并更新工作区
                workspace.update_state(think, report, action, observation)

                # print(f"\n=== Step {step + 1} ===")
                # print(f"[Think]: {think}")
                # print(f"[Report Updated]: {report}")
                # print(f"[Action]: {action}")
                # print(f"[Observation]: {str(observation)[:50]}...")

                chatml_builder.add_to_jsonl()
                chatml_builder.clear()

                # 6. 检查是否完成任务
                if action.startswith("finish["):
                    result = self.parser.parse_finish_action(action)
                    return result, workspace.full_trajectory, chatml_builder

            except ValueError as e:
                error_observation = f"解析错误或格式不合规：{e}"
                chatml_builder.add_message("user", error_observation)
                # 遇到错误时，维持之前的报告，只更新失败的 action 和 observation
                workspace.update_state("解析失败", current_state["report"], action if 'action' in locals() else "未知",
                                       error_observation)
                print(f"step {step + 1} 错误：{error_observation}")

        return f"任务未完成（已达最大步数{self.max_steps}）", workspace.full_trajectory, chatml_builder

    def _execute_action(self, action: str) -> str:
        """
        执行具体行动

        Args:
            action: 行动字符串

        Returns:
            观察结果
        """
        if action.startswith("finish["):
            return "任务完成"
        try:
            tool_name, params = self.parser.parse_tool_action(action, list(self.tool_map.keys()))
            if tool_name in self.tool_map:
                return self.tool_map[tool_name].run(params)
            else:
                return f"工具不存在：{tool_name}"
        except ValueError:
            return f"无效行动：{action}，支持的工具为{list(self.tool_map.keys())}"