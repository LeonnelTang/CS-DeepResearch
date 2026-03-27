from typing import List, Dict, Union, Any, Tuple
from abc import ABC
import logging
from datetime import datetime

# 假设这些是你原本环境中的依赖包和类型
from tools import BaseTool
from core.parser import LLMOutputParser
from core.memory import StateManager

from swift.infer_engine import GRPOVllmEngine
from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, RequestConfig, RolloutInferRequest, RolloutOutput)

from swift.rollout.multi_turn import MultiTurnScheduler

logger = logging.getLogger(__name__)


class IterResearchScheduler(MultiTurnScheduler):
    """
    专为 IterResearch (Think-Report-Action) 架构设计的强化学习多轮调度器。

    核心逻辑：
    1. 抛弃传统的对话历史拼接。
    2. 每一轮基于 StateManager 的最新状态 (Report, 观察) 动态生成全新的 Prompt。
    3. 将每一轮的交互作为独立的 RolloutOutput 返回，以支持细粒度的 GRPO / SFT 训练。
    """

    def __init__(self, tools: List[BaseTool], max_steps: int = 10, *args, **kwargs):
        # 初始化父类
        super().__init__(*args, **kwargs)

        # IterResearch 专属配置
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = self.max_turns
        self.parser = LLMOutputParser()

        # 核心 Prompt 模板 (包含引用规范)
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
        """构建 IterResearch 每一轮的独立 Prompt"""
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

    def _execute_action(self, action: str) -> str:
        """执行具体行动并返回环境观察"""
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

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> List['RolloutOutput']:
        """
        重写 run 方法，接管完整的强化学习多轮 Rollout 流程。
        返回 List[RolloutOutput]，让 Trainer 可以将每一步当作独立的轨迹计算 Reward 和 Loss。
        """
        # 提取初始任务。假设 infer_request 传进来的第一个 user message 就是用户的原始提问
        task = infer_request.messages[0]['content']

        current_request = infer_request
        current_turn = 1
        rollout_outputs = []
        workspace = StateManager()

        final_result = None

        while True:
            # 1. 获取当前 MDP 状态并构建全新的无记忆 prompt
            current_state = workspace.get_current_state()
            prompt_text = self._build_prompt(task, current_state)

            # 由于 IterResearch 依赖 Report 而非历史记录，我们直接覆盖 messages
            current_messages = [{'role': 'user', 'content': prompt_text}]
            current_request.messages = current_messages

            # 2. 调用 VLLM / ms-swift 异步推理引擎获取模型输出
            response = await self.infer_engine.infer_async(current_request, request_config, **kwargs)
            response_choice = response.choices[0]
            completion = response_choice.message.content

            current_messages.append({'role': 'assistant', 'content': completion})

            # 3. 封装当前轮次的独立轨迹数据
            round_output = RolloutOutput(
                response=response,
                messages=current_messages,  # 仅包含当前轮次
                response_token_ids=response_choice.token_ids,
                rollout_infos={'num_turns': current_turn, 'task': task}
            )
            rollout_outputs.append(round_output)

            # 4. 状态机流转：解析输出 -> 执行工具 -> 更新环境状态
            try:
                think, report, action = self.parser.parse_tra(completion)
                if not report:
                    report = current_state["report"]

                # 执行环境动作
                observation = self._execute_action(action)

                # 更新工作区记忆，供下一轮使用
                workspace.update_state(think, report, action, observation)

                # 检查是否完成任务
                if action.startswith("finish["):
                    final_result = self.parser.parse_finish_action(action)
                    break

            except Exception as e:
                # 解析异常时的容错处理，将错误信息作为 observation 传给下一轮
                action = "parse_error, unknown action" if 'action' not in locals() else action
                error_observation = f"解析错误或格式不合规：{e}"
                workspace.update_state("解析失败", current_state["report"], action, error_observation)
                logger.warning(f"Step {current_turn} 解析错误：{error_observation}")

            # 5. 检查是否满足终止条件
            if final_result or current_turn >= self.max_steps:
                break

            current_turn += 1

        return rollout_outputs

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        """
        由于我们完全重写了 run 方法来自定义多轮流转逻辑，
        这个供默认 run() 调用的 step() 方法在此处不再被使用，保持空实现即可。
        """
        pass