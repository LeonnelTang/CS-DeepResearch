from typing import List
from react_agent.schema import TAO


class ContextManager:
    """上下文管理器：存储、裁剪与提取历史TAO轨迹"""

    def __init__(self, max_length: int = 4000) -> None:
        """
        初始化上下文管理器

        Args:
            max_length: 上下文最大字符数
        """
        self.max_length = max_length
        self.tao_trajectory: List[TAO] = []  # 存储TAO三元组：[{"thought": "", "action": "", "observation": ""}]

    def add_tao(self, thought: str, action: str, observation: str) -> None:
        """
        添加TAO三元组并裁剪上下文

        Args:
            thought: 思维内容
            action: 执行的动作
            observation: 观察结果
        """
        self.tao_trajectory.append(TAO(thought, action, observation))
        self._prune_trajectory()

    def _prune_trajectory(self) -> None:
        """裁剪超长轨迹：保留近期5轮+早期摘要"""
        trajectory_str = str(self.tao_trajectory)
        if len(trajectory_str) <= self.max_length:
            return

        # 保留近期3轮完整轨迹
        recent_count = min(5, len(self.tao_trajectory))
        recent_trajectory = self.tao_trajectory[-recent_count:]

        # 生成早期轨迹摘要（如果有早期轨迹）
        if len(self.tao_trajectory) > recent_count:
            early_items = self.tao_trajectory[:-recent_count]
            early_actions = [item.action for item in early_items if item.action]
            early_summary = self._generate_early_summary(early_items, early_actions)

            # 重构上下文：摘要 + 近期轨迹
            self.tao_trajectory = [
                TAO("【早期轨迹摘要】", "", early_summary),
                *recent_trajectory
            ]

    def _generate_early_summary(self, early_items: List[TAO], early_actions: List[str]) -> str:
        """
        生成早期轨迹摘要

        Args:
            early_items: 早期TAO项列表
            early_actions: 早期动作列表

        Returns:
            摘要字符串
        """
        actions_sample = ', '.join(early_actions[:2])

        # 提取成功观察的关键信息
        success_observations = [
            item.observation[:30]
            for item in early_items
            if '成功' in item.observation or 'success' in item.observation.lower()
        ]
        success_info = success_observations[:1] if success_observations else []

        return f"早期行动：{actions_sample}... 关键结果：{success_info}"

    def get_context_str(self) -> str:
        """
        生成模型可理解的上下文字符串

        Returns:
            格式化的上下文字符串
        """
        if not self.tao_trajectory:
            return "无历史执行轨迹"

        return "\n".join([
            f"step {idx + 1}: <thought>{item.thought}</thought> | <action>{item.action}</action> | <observation>{item.observation}</observation>"
            for idx, item in enumerate(self.tao_trajectory)
        ])