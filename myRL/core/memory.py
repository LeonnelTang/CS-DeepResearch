from typing import List, Dict


class StateManager:
    """上下文管理器：基于马尔可夫决策过程 (MDP) 的状态维护"""

    def __init__(self) -> None:
        """初始化迭代工作区"""
        # 核心 MDP 状态变量
        self.current_report: str = ""          # Report_{i-1}: 演进报告（核心记忆）
        self.prev_action: str = ""             # Action_{i-1}: 上一步动作
        self.prev_observation: str = ""        # Observation_{i-1}: 上一步的工具反馈

        # 仅用于本地日志记录/存储，不参与 LLM 的 prompt 构建
        self.full_trajectory: List[Dict[str, str]] = []

    def update_state(self, think: str, report: str, action: str, observation: str) -> None:
        """
        更新当前轮次的 MDP 状态

        Args:
            think: 当前轮次的思考 (不带入下一轮 prompt)
            report: 智能体新合成的演进报告 (覆盖旧报告)
            action: 当前执行的动作
            observation: 工具返回的观察结果
        """
        # 1. 状态重构 (State Reconstruction)
        # 丢弃 <think>，不作为下一轮的状态，避免上下文污染
        self.current_report = report
        self.prev_action = action
        self.prev_observation = observation

        # 2. 轨迹存档
        # self.full_trajectory.append({
        #     "think": think,
        #     "report": report,
        #     "action": action,
        #     "observation": observation
        # })

    def get_current_state(self) -> Dict[str, str]:
        """
        获取当前精简的 MDP 状态，用于构建 Prompt
        """
        return {
            "report": self.current_report if self.current_report else "无（第一轮，尚未生成报告）",
            "prev_action": self.prev_action if self.prev_action else "无",
            "prev_observation": self.prev_observation if self.prev_observation else "无"
        }