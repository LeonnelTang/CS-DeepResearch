from typing import List
from myRL.tools import ToolConfig
from rllm_engine import IterResearchTrajectoryLoop
from rllm.sdk import TrajectoryView

class IterScholarWorkflow:
    def __init__(self, **kwargs):
        # 初始化我们重构好的多步研究 Agent
        # 传入系统中实际定义的 tools 列表
        mcp_config_path = r"myRL/tools/mcp_configs.json"
        tool_config = ToolConfig(mcp_config_path)

        selected_tool_indices = [0, 1, 2, 3]
        tools = tool_config.get_tools_by_indices(selected_tool_indices)

        self.agent_loop = IterResearchTrajectoryLoop(
            tools=tools,
            max_steps=150,
            use_proxy=True
        )

    async def run(self, task: dict, **kwargs) -> List[TrajectoryView]:
        """运行核心循环并返回带有 Reward 的 Trajectory"""
        problem = task["question"]

        # 执行 IterResearch 核心循环 (rLLM 会自动拦截并生成 TrajectoryView)
        # 这里的 traj_view 包含了 Agent 从 Think 到 Report 到最终 Action 的所有步骤
        traj_view = await self.agent_loop.run(problem)

        # 计算整条轨迹的奖励
        total_reward = cs_literature_reward_fn(task, traj_view)

        # 为多步 Agent 分配奖励
        # 在 GRPO 中，通常将总奖励分配给整条轨迹，算法会自动计算 advantage
        traj_view.reward = total_reward

        # 也可以选择性地为每一步分配密集的 Process Reward
        for step in traj_view.steps:
            step.reward = total_reward / len(traj_view.steps)

        # Trainer 需要一个列表格式
        return [traj_view]