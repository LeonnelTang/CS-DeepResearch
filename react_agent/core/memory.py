from typing import List
from react_agent.schema import TAO
from react_agent.llm_client import client


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

        # 保留近期5轮完整轨迹
        recent_count = min(5, len(self.tao_trajectory))
        recent_trajectory = self.tao_trajectory[-recent_count:]

        # 生成早期轨迹摘要（如果有早期轨迹）
        if len(self.tao_trajectory) > recent_count:
            early_items = self.tao_trajectory[:-recent_count]
            early_actions = [item.action for item in early_items if item.action]
            # early_summary = self._generate_early_summary(early_items, early_actions)
            early_summary = self._generate_semantic_summary(early_items)

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

    def _generate_semantic_summary(self, early_items: List[TAO]) -> str:
        """
        调用 LLM 将早期的 TAO 轨迹压缩为语义化摘要。

        Args:
            early_items: 需要被压缩的早期轨迹列表

        Returns:
            XML 格式的精简状态摘要
        """
        # 1. 准备 System Prompt
        system_prompt = """
        # Role
        你是一个专为自主智能体（Autonomous Agent）提供“记忆压缩与状态总结”的后台分析模块。
        
        # Objective
        你的任务是将智能体早期的执行轨迹（包含思考、动作和观察结果）进行高度语义化压缩，提取核心上下文。目的是让智能体在有限的上下文窗口内，能够瞬间理解“我之前做了什么”、“已经得到了什么信息”、“哪些路走不通”以及“当前处于什么阶段”。
        
        # Input Format
        你将接收到一段历史轨迹数据，包含多个轮次的 [Thought, Action, Observation]。
        
        # Guidelines
        1. 极简主义：剔除冗长的输出细节（如大段的日志报错、无关的网页乱码），仅提取具有后续指导意义的元信息。
        2. 聚焦失败边界：必须明确列出“已经尝试过但失败或无效的 Action”，这是防止智能体在后续推理中陷入死循环的核心。
        3. 建立知识快照：必须提取具体的关键数据、核心逻辑、工具调用返回的重要参数或论文/代码的核心结论，禁止诸如“成功获取了某某信息”等等没有信息量的废话。
        4. 状态对齐：基于历史轨迹，用一句话总结智能体当前面临的卡点或进度。
        5. 增量更新原则：只能在已有早期轨迹摘要基础上追加新信息，严禁删改历史内容
        
        # Output Format
        请严格按照以下 XML 标签结构输出你的摘要结果，要求语言高度精炼，不要输出任何解释性废话：
        
        <memory_summary>
          <milestones>
            - [成功] 动作A：获取了关键数据 X
            - [成功] 动作B：成功验证了配置 Y
          </milestones>
          <failed_attempts>
            - [无效] 尝试使用方法C，报错“...” -> 结论：该 API 不可用 / 路径错误
            - [无效] 搜索关键词D，未找到相关结果
          </failed_attempts>
          <current_state>
            已完成前置数据准备，当前卡在[问题Z]，亟待寻找替代方案或解析[已知条件W]。
          </current_state>
        </memory_summary>
        """

        # 2. 将早期轨迹格式化为易于 LLM 阅读的纯文本格式
        # 避免直接传入带有各种转义字符的 raw object，减少 Token 消耗
        trajectory_text = "\n".join([
            f"[Step {i + 1}] 思考: {item.thought} | 动作: {item.action} | 观察: {item.observation}..."
            for i, item in enumerate(early_items)
        ])

        user_prompt = f"请压缩以下早期执行轨迹：\n{trajectory_text}"

        # 3. 调用大模型

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # 保持极低的温度，确保输出格式稳定和事实准确
                max_tokens=8192  # 限制摘要的最大长度
            )
            summary = response.choices[0].message.content.strip()

            print(f"[Memory Info] 语义摘要生成成功: {summary}")

            return summary

        except Exception as e:
            # 4. 异常降级兜底：如果 API 调用失败，退回到最基础的字符串截取
            print(f"[Memory Warning] 语义摘要生成失败: {e}. 触发降级处理。")
            fallback_actions = [item.action for item in early_items if item.action][:2]
            return f"<memory_summary>早期尝试了动作 {', '.join(fallback_actions)}... (语义摘要失败)</memory_summary>"