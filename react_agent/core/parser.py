from typing import List, Dict, Any, Optional, Tuple
import re


class LLMOutputParser:
    """LLM输出解析器"""

    @staticmethod
    def parse_tra(llm_output: str) -> Tuple[str, str, str]:
        """
        解析 XML 格式的 LLM 输出，提取思维和行动
        格式：<thought>...</thought>\n<report>...</report>\n<action>...</action>

        Args:
            llm_output: LLM原始输出

        Returns:
            (思维内容, 演进报告，行动内容) 元组

        Raises:
            ValueError: 解析失败时抛出异常
        """
        try:
            # 使用正则提取标签内容
            think_match = re.search(r'<think>(.*?)</think>', llm_output, re.DOTALL)
            report_match = re.search(r'<report>(.*?)</report>', llm_output, re.DOTALL)
            action_match = re.search(r'.*<action>(.*?)</action>', llm_output, re.DOTALL)

            think = think_match.group(1).strip() if think_match else ""
            report = report_match.group(1).strip() if report_match else ""
            action = action_match.group(1).strip() if action_match else ""

            if not action:
                raise ValueError("未在输出中找到 <action> 标签")

            return think, report, action
        except (IndexError, ValueError) as e:
            raise ValueError(f"解析LLM输出失败：{e}")

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
            raise ValueError(f"无效的finish格式，正确格式：finish[...]")

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

        # 找到第一个左括号的位置，这是工具名和参数的分界线
        bracket_idx = action.find('[')

        if bracket_idx == -1:
            raise ValueError(f"工具调用格式错误，缺少参数括号：{action}")

        tool_name = action[:bracket_idx].strip()

        if tool_name not in tool_names:
            raise ValueError(f"未识别的工具调用：{tool_name}")

        # 提取参数字符串并确保尾部闭合
        if not action.strip().endswith(']'):
            raise ValueError(f"工具调用格式错误，未找到闭合括号：{action}")

        param_str = action[bracket_idx + 1:action.rfind(']')].strip()

        return tool_name, param_str