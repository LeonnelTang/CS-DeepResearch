from typing import List, Dict, Any, Optional, Tuple
import re


class LLMOutputParser:
    """LLM输出解析器"""

    @staticmethod
    def parse_thought_action(llm_output: str) -> Tuple[str, str]:
        """
        解析 XML 格式的 LLM 输出，提取思维和行动
        格式：<thought>...</thought>\n<action>...</action>

        Args:
            llm_output: LLM原始输出

        Returns:
            (思维内容, 行动内容) 元组

        Raises:
            ValueError: 解析失败时抛出异常
        """
        try:
            # 使用正则提取 thought 标签内容
            thought_match = re.search(r'<thought>(.*?)</thought>', llm_output, re.DOTALL)
            action_match = re.search(r'<action>(.*?)</action>', llm_output, re.DOTALL)

            if not thought_match or not action_match:
                # 如果解析失败，尝试向后兼容旧格式（可选）
                # 或者直接抛出异常
                raise ValueError(f"无法从输出中解析 thought/action XML 标签：{llm_output}")

            thought = thought_match.group(1).strip()
            action = action_match.group(1).strip()

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