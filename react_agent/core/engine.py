from typing import List, Dict, Any, Optional, Callable, Tuple
from react_agent.tools import BaseTool
from .parser import LLMOutputParser
from .memory import ContextManager
from react_agent import ChatMLBuilder


class ReactCoreLoop:
    """ReAct核心循环控制器"""

    def __init__(self,
                 tools: List[BaseTool],
                 llm_executor: Callable[[str], str],
                 max_steps: int = 6
                 ):
        """
        初始化ReAct循环

        Args:
            tools: 可用工具列表
            max_steps: 最大迭代步数
        """
        self.tool_map = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.llm_executor = llm_executor
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

        self.xml_prompt = """你是ReAct智能体，需通过"思维→行动→观察"循环完成任务，严格遵循以下规则：
1. 思维：分析任务目标与历史轨迹，说明下一步行动的逻辑依据；
2. 行动：仅使用提供的工具，格式为"工具名[参数]"，支持工具：{tool_descriptions}；
3. 观察：根据工具反馈调整后续策略，不可仅凭记忆回答。

**输出格式要求**：
请严格按照以下XML格式输出当前步骤的思维和行动（仅输出XML标签，不要添加任何额外内容）：
<thought>你的思维内容</thought>
<action>你的行动内容</action>

示例1（初始步骤）：
任务：查询昨天从深圳到广州最便宜上午的航班
历史轨迹：无历史执行轨迹
输出：
<thought>需获取昨天深圳到广州上午的航班信息，调用航班查询工具，参数为"深圳,广州,昨天,上午"</thought>
<action>flight_search[深圳,广州,昨天,上午]</action>

示例2（后续步骤）：
任务：查询昨天从深圳到广州最便宜上午的航班
历史轨迹：步骤1：思维：... | 行动：flight_search... | 观察：符合条件航班列表...
输出：
<thought>已获取航班列表，需筛选最便宜的航班（CZ3201，230元），调用航班预订工具完成预订</thought>
<action>flight_book[CZ3201,张三,123456199001011234]</action>

示例3（任务完成）：
任务：查询昨天从深圳到广州最便宜上午的航班
历史轨迹：步骤1：... 步骤2：...
输出：
<thought>已完成航班查询与预订任务，提交结果</thought>
<action>finish[昨天深圳到广州最便宜上午航班为CZ3201（08:30-09:10，票价230元），已完成预订，乘客张三]</action>

当前任务：{task}
历史轨迹：{context}
请输出当前步骤的思维和行动（仅输出XML标签）：
"""

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

        return self.xml_prompt.format(
            tool_descriptions=tool_descriptions,
            task=task,
            context=context
        ).strip()

    def _call_llm(self, prompt: str) -> str:
        """
        调用真实LLM
        """
        try:
            # 直接调用传入的执行函数
            response = self.llm_executor(prompt)
            return response
        except Exception as e:
            # 简单的错误处理，实际可加重试机制
            return f"<thought>LLM调用失败，错误信息：{e}</thought>\n<action>finish[系统错误]</action>"

    def run(self, task: str) -> Tuple[str, str]:
        """
        运行ReAct核心循环

        Args:
            task: 需要完成的任务描述

        Returns:
            (最终结果, 完整执行轨迹) 元组
        """
        context_manager = ContextManager()

        chatml_builder = ChatMLBuilder()
        chatml_builder.add_message("user", task)

        for step in range(self.max_steps):
            # 1. 构建提示词并调用LLM
            prompt = self._build_prompt(task, context_manager.get_context_str())

            # 调用LLM
            llm_output = self._call_llm(prompt)

            try:
                # 2. 解析思维与行动
                thought, action = self.parser.parse_thought_action(llm_output)

                # 3. 执行行动并获取观察
                observation = self._execute_action(action)

                # 4. 更新上下文
                context_manager.add_tao(thought, action, observation)

                # 打印执行过程
                print(f"step {step + 1}: <thought>{thought}</thought> | <action>{action}</action> | <observation>{observation}</observation>")

                # 5. 检查是否完成任务
                if action.startswith("finish["):
                    result = self.parser.parse_finish_action(action)
                    return result, context_manager.get_context_str()

            except ValueError as e:
                # 处理解析错误
                error_observation = f"解析错误：{e}"
                context_manager.add_tao("解析错误", action if 'action' in locals() else "未知", error_observation)
                print(f"step {step + 1} 错误：{error_observation}")

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