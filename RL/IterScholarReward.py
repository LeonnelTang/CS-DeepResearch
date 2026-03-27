import asyncio
import json
import os
import random
import re
import textwrap
import torch
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union, Tuple

from swift.infer_engine import RequestConfig, TransformersEngine
from swift.infer_engine.protocol import ChatCompletionResponse, ChatCompletionResponseChoice, RolloutInferRequest
from swift.rewards import ORM, AsyncORM, orms, rm_plugins
from swift.rewards.rm_plugin import DefaultRMPlugin
# register context manager(used in gym training)
from swift.rollout.gym_env import ContextManager, Env, context_managers, envs
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.template import Template
from swift.utils import get_logger, to_device

from openai import OpenAI
import sys


logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""

class IterScholarReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.client = OpenAI(
            api_key="sk-e3abc9639ad2478185640081eea65afd",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def __call__(self, completions, **kwargs) -> List[float]:

        max_steps = 100

        trajectory_ids: List[str] = kwargs.get('request_id')

        global_trajectorys: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs')

        rewards = []
        for local_tra_id in trajectory_ids:
            total_trajectory_inputs = global_trajectorys[local_tra_id]  # 当前轨迹
            last_turn_messages = total_trajectory_inputs[-1]['messages']
            last_turn_completion = last_turn_messages[-1]['content']    # 最后一轮 LLM 输出
            last_turn_answer = total_trajectory_inputs[-1]['answer']
            last_turn_question = total_trajectory_inputs[-1]['question']

            if not last_turn_answer:
                print("===============================\n================================\nReward No Dataset Answer")
                print("===============================\n================================\n")

            total_steps = len(total_trajectory_inputs)

            try:
                think, report, action = self.parse_tra(last_turn_completion)
            except Exception as e:
                logger.error(f"Error parsing LLM output: {e}")
                think, report, action = "", "", ""

            try:
                final_answer = self.parse_finish_action(action)
            except Exception as e:
                logger.error(f"Error parsing finish action: {e}")
                final_answer = ""

            judge_result = self.evaluate_answer(last_turn_question, last_turn_answer, final_answer)
            is_correct = judge_result.get("correct", "") == "yes"

            if is_correct:
                # 正确的话，步数越少（效率越高），额外奖励越多
                efficiency_bonus = 0.5 * ((max_steps - total_steps) / max_steps)
                rewards.append(2.0 + efficiency_bonus)
            else:
                rewards.append(0.0)  # 答错了

        return rewards

    def evaluate_answer(self, question, correct_answer: str, model_answer: str):
        prompt_template = "请根据下方精确且明确的 [correct_answer]（正确答案），判断以下对 [question]（问题）的 [response]（模型回答）是否正确。\n\n[question]: {question}\n\n[response]: {response}\n\n[correct_answer]: {correct_answer}\n\n您的判断结果必须严格以合法的 JSON 格式输出，并且必须包含以下键名（keys），请遵循各字段对应的数据类型与动态判定标准：\n\n{{\n    \"extracted_final_answer\": (字符串) 从 [response] 中提取出的最终确切答案。如果无法从回答中提取出确切的最终答案，请填入 None。\n    \"reasoning\": (字符串) 根据 [correct_answer] 解释 extracted_final_answer 是正确还是错误，只需重点关注 [correct_answer] 和 extracted_final_answer 之间是否存在实质性差异。不要对题目的背景发表评论，不要尝试去解答问题，也不要为任何不同于 [correct_answer] 的答案进行辩护，只需专注于两个答案是否匹配。\n    \"correct\": (字符串) 如果 extracted_final_answer 与上方给出的 [correct_answer] 匹配（对于数值类问题，允许存在微小的误差），请填入 yes。否则填入 no（即：如果存在任何不一致、歧义、不等价，或者提取出的答案本身是错误的）。\n    \"confidence\": (整数) 从 [response] 中提取出的置信度评分数值（介于 0 到 100 之间）。如果回答中没有提供置信度评分，请填入 100。\n}}"
        JUDGE_PROMPT = prompt_template.format(question=question, correct_answer=correct_answer, response=model_answer)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                max_completion_tokens=4096,  # overkill for judge
                messages=[
                    {"role": "user", "content": JUDGE_PROMPT}
                ],
                response_format={
                    'type': 'json_object'
                }
            )
            content = response.choices[0].message.content
            judge_json = self.safe_json_loads(content)
            return {
                "correct_answer": correct_answer,
                "model_answer": judge_json["extracted_final_answer"],
                "reasoning": judge_json["reasoning"],
                "correct": judge_json["correct"],
                "confidence": judge_json["confidence"]
            }
        except Exception as e:  # very, very rare
            print("\nError:", e)

            # 根据错误类型决定是否终止程序
            error_code = getattr(e, 'status_code', None)

            # 致命错误：认证失败、模型不存在等
            if error_code == 401:
                print("致命错误：API 认证失败，请检查 API Key")
                sys.exit(1)
            elif error_code == 403:
                print("致命错误：无权访问该模型，请检查权限")
                sys.exit(1)
            elif error_code == 404:
                print(f"致命错误：模型不存在")
                sys.exit(1)
            elif error_code == 503:
                print(f"致命错误：模型不存在")
                sys.exit(1)
            elif error_code == 500:
                print("服务器内部错误，跳过此问题")
                return None  # 非致命，继续处理其他问题
            elif error_code == 429:
                print("速率限制，建议降低并发数 (--num_workers)")
                return None  # 可以选择继续或退出
            else:
                # 网络错误等其他问题，继续处理
                return None

    def safe_json_loads(self, json_text: str):
        """
        安全解析LLM的JSON响应。
        处理包含Markdown格式、前后带有无关文本等常见LLM输出问题。
        """
        if not json_text or not isinstance(json_text, str):
            return None

        json_text = json_text.strip()

        # 1. 最理想情况：直接就是完美的 JSON
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass  # 继续尝试下面的提取逻辑

        # 2. 尝试提取 Markdown 代码块中的 JSON (如 ```json ... ```)
        # 使用 re.DOTALL 使 . 能够匹配换行符
        match = re.search(r'```(?:json)?(.*?)```', json_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_text = match.group(1).strip()
            try:
                return json.loads(extracted_text)
            except json.JSONDecodeError:
                pass

        # 3. 尝试暴力提取大括号 {...} 或中括号 [...] 内的内容
        # 寻找第一个左括号和最后一个对应的右括号
        dict_start, dict_end = json_text.find('{'), json_text.rfind('}')
        list_start, list_end = json_text.find('['), json_text.rfind(']')

        # 判断是最外层是字典还是列表
        if dict_start != -1 and dict_end != -1 and dict_end > dict_start:
            dict_text = json_text[dict_start:dict_end + 1]
            try:
                return json.loads(dict_text)
            except json.JSONDecodeError:
                pass

        if list_start != -1 and list_end != -1 and list_end > list_start:
            list_text = json_text[list_start:list_end + 1]
            try:
                return json.loads(list_text)
            except json.JSONDecodeError:
                pass

        # 4. 如果所有尝试都失败了，抛出异常或返回 None（这里选择记录日志并返回 None）
        print(f"[Warning] Failed to safely parse JSON. Raw text snippet: {json_text[:100]}...")
        return None

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


orms['iter_scholar'] = IterScholarReward