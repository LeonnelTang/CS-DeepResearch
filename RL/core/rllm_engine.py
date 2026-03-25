import asyncio
from typing import List
from datetime import datetime
from rllm.sdk import trajectory, get_chat_client_async
from react_agent.tools import BaseTool
from .parser import LLMOutputParser
from .memory import StateManager


class IterResearchTrajectoryLoop:
    """基于 rLLM 重构的 IterResearch 核心循环 (专为 RL 训练设计)"""

    def __init__(self, tools: List[BaseTool], max_steps: int = 10, use_proxy: bool = True):
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.parser = LLMOutputParser()

        # 1. 替换为 rLLM 的异步 Client
        self.client = get_chat_client_async(
            base_url="http://localhost:4000/v1",
            api_key="EMPTY",
            use_proxy=use_proxy  # 训练收集数据时设为 True，纯推理时可设为 False
        )

        # 挂载你正在进行 SFT/RL 训练的基础模型
        self.model = "IterScholar-CS-35B-A3B-SFT"

        # 保留原有的 Prompt 模板
        self.citation_prompt = """... (保留你原有的 citation_prompt 内容，此处省略以保持整洁) ..."""

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

    # 2. @trajectory 装饰器
    @trajectory(name="iter_scholar_solver")
    async def run(self, task: str):
        """
        核心 Rollout 循环。
        该函数执行完毕后，rLLM 会自动返回一个包含所有 LLM 交互步骤的 TrajectoryView。
        """
        workspace = StateManager()

        for step in range(self.max_steps):
            # 获取当前 MDP 状态并构建 prompt
            current_state = workspace.get_current_state()
            prompt = self._build_prompt(task, current_state)

            # 异步调用 LLM (rLLM 会在这里自动拦截并记录 Step)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # 强化学习 Rollout 时需要一定的随机性来探索多条轨迹
            )
            llm_output = response.choices[0].message.content

            try:
                # 解析输出
                think, report, action = self.parser.parse_tra(llm_output)

                # 如果模型没有生成报告，维持记忆
                if not report:
                    report = current_state["report"]

                # 执行行动并获取观察结果
                observation = self._execute_action(action)

                # 5. 重构状态并更新工作区
                workspace.update_state(think, report, action, observation)

                # 4. 检查是否完成：返回最终结果，自动结束轨迹记录
                if action.startswith("finish["):
                    final_result = self.parser.parse_finish_action(action)
                    return final_result

            except Exception as e:
                # 异常处理：将错误作为 Observation 反馈给模型，要求其自我纠正
                error_observation = f"解析错误或格式不合规：{e}"
                # 遇到错误时，维持之前的报告，只更新失败的 action 和 observation
                workspace.update_state("解析失败", current_state["report"], action if 'action' in locals() else "未知",
                                       error_observation)
                print(f"Step {step + 1} 错误：{error_observation}")

        # 达到最大步数仍未完成
        return f"任务未完成（已达最大步数 {self.max_steps}）"

    def _execute_action(self, action: str) -> str:
        """执行具体行动 (与原版逻辑保持一致)"""
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