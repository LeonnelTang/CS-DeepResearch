from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from tool import *

@dataclass
class TAO:
    """思维-行动-观察三元组数据结构"""
    thought: str
    action: str
    observation: str


class ContextManager:
    """上下文管理器：存储、裁剪与提取历史TAO轨迹"""

    def __init__(self, max_length: int = 4000) -> None:
        """
        初始化上下文管理器

        Args:
            max_length: 上下文最大字符数
        """
        self.max_length = max_length
        self.tao_trajectory: List[TAO] = [] # 存储TAO三元组：[{"thought": "", "action": "", "observation": ""}]

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
        """裁剪超长轨迹：保留近期3轮+早期摘要"""
        trajectory_str = str(self.tao_trajectory)
        if len(trajectory_str) <= self.max_length:
            return

        # 保留近期3轮完整轨迹
        recent_count = min(3, len(self.tao_trajectory))
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
            if '成功' in item.observation
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
            f"步骤{idx + 1}：思维：{item.thought} | 行动：{item.action} | 观察：{item.observation}"
            for idx, item in enumerate(self.tao_trajectory)
        ])

class LLMOutputParser:
    """LLM输出解析器"""

    @staticmethod
    def parse_thought_action(llm_output: str) -> Tuple[str, str]:
        """
        解析LLM输出中的思维和行动

        Args:
            llm_output: LLM原始输出

        Returns:
            (思维内容, 行动内容) 元组

        Raises:
            ValueError: 解析失败时抛出异常
        """
        try:
            # 分割思维部分
            thought_part = llm_output.split("思维：")[1]
            thought, action_part = thought_part.split("行动：", 1)

            thought = thought.strip()
            action = action_part.strip()

            return thought, action
        except (IndexError, ValueError) as e:
            raise ValueError(f"解析LLM输出失败：{e}，原始输出：{llm_output}")

    @staticmethod
    def parse_finish_action(action: str) -> str:
        """
        解析完成行动的结果

        Args:
            action: 行动字符串

        Returns:
            提取的结果

        Raises:
            ValueError: 解析失败时抛出异常
        """
        if not (action.startswith("finish[") and action.endswith("]")):
            raise ValueError(f"无效的finish格式：{action}")

        return action[len("finish["):-1].strip()

    @staticmethod
    def parse_tool_action(action: str, tool_names: List[str]) -> Tuple[str, str]:
        """
        解析工具行动

        Args:
            action: 行动字符串
            tool_names: 支持的工具名称列表

        Returns:
            (工具名称, 参数字符串) 元组

        Raises:
            ValueError: 解析失败时抛出异常
        """
        for tool_name in tool_names:
            if action.startswith(tool_name):
                param_str = action[len(tool_name) + 1:-1].strip()
                return tool_name, param_str

        raise ValueError(f"未识别的工具调用：{action}")


class ReactCoreLoop:
    """ReAct核心循环控制器"""

    def __init__(self, tools: List[BaseTool], max_steps: int = 6):
        """
        初始化ReAct循环

        Args:
            tools: 可用工具列表
            max_steps: 最大迭代步数
        """
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.parser = LLMOutputParser()

        # 提示词模板（含Few-shot示例）
        self.prompt_template = """你是ReAct智能体，需通过"思维→行动→观察"循环完成任务，严格遵循以下规则：
1. 思维：分析任务目标与历史轨迹，说明下一步行动的逻辑依据；
2. 行动：仅使用提供的工具，格式为"工具名[参数]"，支持工具：{tool_descriptions}；
3. 观察：根据工具反馈调整后续策略，不可仅凭记忆回答。

示例：
任务：查询昨天从深圳到广州最便宜上午的航班
历史轨迹：无历史执行轨迹
思维：需获取昨天深圳到广州上午的航班信息，调用航班查询工具，参数为"深圳,广州,昨天,上午"
行动：flight_search[深圳,广州,昨天,上午]
观察：符合条件航班列表：1. CZ3201（深圳宝安→广州白云，08:30-09:10，票价230元）；2. HU7125（深圳宝安→广州白云，09:40-10:20，票价280元）
思维：已获取航班列表，需筛选最便宜的航班（CZ3201，230元），调用航班预订工具完成预订
行动：flight_book[CZ3201,张三,123456199001011234]
观察：航班预订成功：航班号CZ3201，乘客张三（身份证号：1234），请携带有效证件提前2小时到机场办理登机手续
思维：已完成航班查询与预订任务，提交结果
行动：finish[昨天深圳到广州最便宜上午航班为CZ3201（08:30-09:10，票价230元），已完成预订，乘客张三]

当前任务：{task}
历史轨迹：{context}
请输出当前步骤的思维和行动（仅输出思维和行动，无其他内容）：
思维：
行动："""

    def _build_prompt(self, task: str, context: str) -> str:
        """
        构建提示词

        Args:
            task: 任务描述
            context: 历史上下文

        Returns:
            完整的提示词
        """
        tool_descriptions = "\n".join([
            f"- {name}：{tool.description}"
            for name, tool in self.tool_map.items()
        ])

        return self.prompt_template.format(
            tool_descriptions=tool_descriptions,
            task=task,
            context=context
        ).strip()

    def _simulate_llm_call(self, prompt: str, step: int) -> str:
        """
        模拟LLM调用（实际应用应替换为真实LLM API）

        Args:
            prompt: 提示词
            step: 当前步骤

        Returns:
            模拟的LLM输出
        """
        # 这里仅为演示目的，实际应用中应调用真实的LLM API
        simulation_outputs = [
            """思维：当前任务是查询明天从深圳到海南的航班，选最便宜、航班时间在晚上的那班并预订，无历史数据。需先调用航班查询工具，参数为"深圳,海南,明天,晚上"，获取符合条件的航班列表
行动：flight_search[深圳,海南,明天,晚上]""",

            """思维：已获取明天深圳到海南晚上的航班列表，从观察结果可知最便宜的是HU7089（票价480元）。下一步需调用航班预订工具，参数包含航班号HU7089、乘客信息（假设乘客为李四，身份证号440301199505056789）
行动：flight_book[HU7089,李四,440301199505056789]""",

            """思维：已完成最便宜晚上航班的查询与预订，观察结果显示预订成功，所有任务目标均达成，可提交最终结果
行动：finish[明天从深圳到海南最便宜的晚上航班为HU7089（深圳宝安→海口美兰，20:15-21:45，票价480元），已完成预订，乘客李四（身份证号：6789）]"""
        ]

        if step < len(simulation_outputs):
            return simulation_outputs[step]
        return """思维：任务已完成，无需进一步行动
行动：finish[任务已完成]"""

    def run(self, task: str) -> Tuple[str, str]:
        """
        运行ReAct核心循环

        Args:
            task: 需要完成的任务描述

        Returns:
            (最终结果, 完整执行轨迹) 元组
        """
        context_manager = ContextManager()

        for step in range(self.max_steps):
            # 1. 构建提示词并调用LLM
            prompt = self._build_prompt(task, context_manager.get_context_str())

            # 实际应用应替换为真实LLM调用
            llm_output = self._simulate_llm_call(prompt, step)

            try:
                # 2. 解析思维与行动
                thought, action = self.parser.parse_thought_action(llm_output)

                # 3. 执行行动并获取观察
                observation = self._execute_action(action)

                # 4. 更新上下文
                context_manager.add_tao(thought, action, observation)

                # 打印执行过程
                print(f"步骤{step + 1}：思维：{thought} | 行动：{action} | 观察：{observation}")

                # 5. 检查是否完成任务
                if action.startswith("finish["):
                    result = self.parser.parse_finish_action(action)
                    return result, context_manager.get_context_str()

            except ValueError as e:
                # 处理解析错误
                error_observation = f"解析错误：{e}"
                context_manager.add_tao("解析错误", action if 'action' in locals() else "未知", error_observation)
                print(f"步骤{step + 1}错误：{error_observation}")

        # 超时终止
        return f"任务未完成（已达最大步数{self.max_steps}）", context_manager.get_context_str()

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
            # 解析工具调用
            tool_name, params = self.parser.parse_tool_action(action, list(self.tool_map.keys()))

            # 调用工具
            if tool_name in self.tool_map:
                return self.tool_map[tool_name].run(params)
            else:
                return f"工具不存在：{tool_name}"

        except ValueError:
            # 无效行动
            return f"无效行动：{action}，支持的工具为{list(self.tool_map.keys())}"


def main():
    """主函数：演示ReAct循环的使用"""
    # 初始化工具
    tools = [FlightSearchTool(), FlightBookTool()]

    # 定义任务
    task = "查询明天从深圳到海南的航班，选最便宜、航班时间在晚上的那班并预订"

    # 创建并运行ReAct循环
    reactor = ReactCoreLoop(tools, max_steps=6)
    final_result, trajectory = reactor.run(task)

    # 输出结果
    print("\n" + "=" * 50)
    print("最终结果：", final_result)
    print("\n完整执行轨迹：")
    print(trajectory)
    print("=" * 50)


if __name__ == "__main__":
    main()