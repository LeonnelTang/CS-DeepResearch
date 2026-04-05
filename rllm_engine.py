import re
from typing import List, Tuple, Callable
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import Workflow
from datetime import datetime

from myRL.tools import BaseTool
from myRL.core.memory import StateManager
from myRL.core.parser import LLMOutputParser

class IterScholarAgent:
    """深度研究多步执行体"""

    def __init__(self,
                 rollout_engine: RolloutEngine,
                 tools: List[BaseTool],
                 max_steps: int = 10
                 ):
        self.rollout_engine = rollout_engine
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.parser = LLMOutputParser()

        # 你的 Prompt 模板
        self.citation_prompt = """你是一个深度研究智能体。你需要通过“思考→报告→行动”(Think-Report-Action) 的迭代循环来解决复杂问题。
        你没有历史对话记录！你唯一的知识载体和上下文是“全局演进报告 (report)”。你必须在每一次迭代中，将新线索与旧报告进行深度语义融合、去重和提炼，而非简单堆砌。

        **输出结构要求 (必须且只能包含以下三个 XML 标签，禁止输出任何标签外的文本)**：
        1. <think> 认知暂存区：分析当前工作区状态，评估上次行动的结果(observation)，准备更新报告 (report)并推理下一步工具调用策略(action)。此部分属于你的内部推理，不会带入下一轮。</think>
        2. <report> 演进报告 (中心记忆)：这是你唯一的上下文记忆！请将本次新发现与上一轮的旧报告进行综合，输出一份结构化、高信息密度的最新进展摘要。禁止简单追加，必须进行语义融合。</report>
        3. <action> 行动：仅使用提供的工具，格式为"工具名[参数]"。如果认为已有足够证据回答初始问题，请输出"finish[最终回复]"，这是你向用户展示结果的唯一渠道。</action>

        **信息溯源与引用规范**
        1. 你的所有事实陈述必须有据可查。
        2. 在更新 <report> 时，必须在报告的最下方维护一个 [References] 列表。格式为：\\n[1] 标题 (URL或来源ID)。
        3. 在最终通过 finish[] 提交答案时，必须使用内联引用格式（如“该算法在多步推理上表现优异 [1][3]”），并在finish[]中附上完整的参考文献列表。

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

        return self.citation_prompt.format(
            tool_descriptions=tool_descriptions,
            task=task,
            current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report=workspace_state["report"],
            prev_action=workspace_state["prev_action"],
            prev_observation=workspace_state["prev_observation"]
        ).strip()

    async def execute_research(self, problem: str) -> Trajectory:
        """执行迭代循环，收集并返回完整的多步轨迹"""
        workspace = StateManager()
        steps = []

        messages = [{"role": "system", "content": "你是一个深度研究助手。"}]

        for step_idx in range(self.max_steps):
            # 1. 构建当前步的上下文 Prompt
            current_state = workspace.get_current_state()
            prompt = self._build_prompt(problem, current_state)
            messages.append({"role": "user", "content": prompt})

            # 2. 调用底层的 RolloutEngine
            output: ModelOutput = await self.rollout_engine.get_model_response(messages)
            response_text = output.content

            try:
                # 3. 解析模型输出 (Think/Report/Action)
                think, report, action = self.parser.parse_tra(response_text)
                # 如果模型没有生成报告，维持记忆
                if not report:
                    report = current_state["report"]

                # 4. 执行工具调用
                observation = self._execute_action(action)

                # 5. 重构状态并更新工作区
                workspace.update_state(think, report, action, observation)

                # 组装为 rLLM 标准的 Step 对象
                current_step = Step(
                    chat_completions=messages + [{"role": "assistant", "content": response_text}],
                    thought=think,  # 提取出的思考过程
                    report=report,
                    action=action,  # 提取出的具体行动
                    observation=observation,
                    model_output=output  # 原始的底层输出对象
                )
                steps.append(current_step)

                # 6. 判定结束
                if action.startswith("finish["):
                    result = self.parser.parse_finish_action(action)
                    break

            except ValueError as e:
                error_think = "解析失败"
                error_report = current_state["report"]
                error_action = action if 'action' in locals() else "未知"
                error_observation = f"解析错误或格式不合规：{e}"

                # 遇到错误时，维持之前的报告，只更新失败的 action 和 observation
                workspace.update_state(error_think, error_report, error_action, error_observation)

                # 组装为 rLLM 标准的 Step 对象
                current_step = Step(
                    chat_completions=messages + [{"role": "assistant", "content": response_text}],
                    thought=error_think,  # 提取出的思考过程
                    report=error_report,
                    action=error_action,  # 提取出的具体行动
                    observation=error_observation,
                    model_output=output  # 原始的底层输出对象
                )
                steps.append(current_step)

        # 将所有的步骤打包为一条完整的 Trajectory
        return Trajectory(name="iter_scholar_research", steps=steps)


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