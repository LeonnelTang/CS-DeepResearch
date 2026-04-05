import base64
import os
import json
import argparse
import asyncio
import sys

import requests
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

from eval.utils import load_local_dataset
# from eval.DR_clients.QwenDR import TongyiDeepResearch
# from eval.DR_clients.GeminiDR import GeminiDeepResearch
# from eval.DR_clients.OpenaiDR import OpenaiDeepResearch

load_dotenv()

# client = AsyncOpenAI(
#     api_key=os.getenv('ALLMHUB_API_KEY'),
#     base_url='https://api.allmhub.com/v1',
#     timeout=600.0,
#     max_retries=1
# )

# client = AsyncOpenAI(
#     api_key=os.getenv('UIUI_API_KEY'),
#     base_url='https://sg.uiuiapi.com/v1',
#     timeout=600.0,
#     max_retries=1
# )

# client = AsyncOpenAI(
#     api_key=os.getenv('DEEPSEEK_API_KEY'),
#     base_url='https://api.deepseek.com/v1',
#     timeout=600.0,
#     max_retries=1
# )

client = AsyncOpenAI(
    api_key=os.getenv('MY_DMX_API_KEY'),
    base_url="https://www.dmxapi.cn/v1",
    timeout=600.0,
    max_retries=1
)

with open("eval/prompts.json", "r", encoding='utf-8') as f:
    prompt_templates = json.load(f)

def load_image_as_base64(image_path):
    # local image or online image
    if os.path.isfile(image_path):
        # local image
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print("Failed to read local image:", e)
            return ""
    else:
        # imitate an explorer
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(image_path, headers=headers)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print("Failed to download image:", e)
            return ""

def format_message(question):
    language = "zh"
    prompt_template = prompt_templates["answer"][language]
    SYSTEM_PROMPT = prompt_template

    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    content = [text_content]

    system_role = "user" if "o1" in args.model else "system"  # o1 no sys prompt
    messages = [
        {"role": system_role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return messages


async def attempt_question(question):
    messages = format_message(question)

    DR_models = {
        # "gemini-deep-research": GeminiDeepResearch(),
        # "tongyi-deepresearch-30b-a3b": TongyiDeepResearch(),
        # "tavily-research": TavilyDeepResearch(),
        # "o4-mini-deep-research": OpenaiDeepResearch(model_choose=args.model),
        # "o3-deep-research": OpenaiDeepResearch(model_choose=args.model)
    }

    if args.model in DR_models:
        dr_client = DR_models[args.model]
        response = dr_client.run_research(system_prompt=messages[0]["content"], dr_task=messages[1]["content"])
        content = response.get("content")
        tokens = response.get("usage", None)
    else:
        try:
            response = await client.chat.completions.create(
                model=args.model,
                temperature=args.temperature if "o1" not in args.model else None,
                # max_completion_tokens=args.max_completion_tokens,
                max_tokens=args.max_completion_tokens,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            tokens = response.usage.model_dump()
        except Exception as e:
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
                print(f"致命错误：模型 '{args.model}' 不存在")
                sys.exit(1)
            elif error_code == 503:
                print(f"致命错误：模型 '{args.model}' 不存在")
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

    if content is None:  # failed
        print(f"\n警告：问题 {question['id']} 返回空内容")
        return None

    return question["id"], content, tokens

async def attempt_all(questions):
    async def bound_func(question):
        async with semaphore:
            content = await attempt_question(question)
            return content

    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)

    return results

def main(args):

    assert args.num_workers > 1, "num_workers must be 2 or greater"

    if args.local_dataset:  # load from local dataset
        dataset = load_local_dataset(args.local_dataset)
    else:   # load from HuggingFace
        dataset = load_dataset(args.dataset, split="test").to_dict()
        # dataset = {}

    # convert to list of json for async parallelism
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        questions = questions[:args.max_samples]

    output_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_traj/other_models_results"))
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = f"{output_dir}/{os.path.basename(args.model)}.json"

    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding='utf-8') as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    results = asyncio.run(attempt_all(questions))

    # It is allowed to rerun this script multiple times if there are failed API calls
    for result in results:
        if result is None:  # API call failed
            continue
        unique_id, response, usage = result
        predictions[unique_id] = {
            "model": args.model,
            "response": response,
            "usage": usage
        }

    # cache responses
    with open(output_filepath, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HLE HF Dataset")
    parser.add_argument("--local_dataset", type=str, default=None, help="Local JSON dataset file path")
    parser.add_argument("--model", type=str, help="Model Endpoint Name")
    parser.add_argument("--max_completion_tokens", type=int, default=None,
                        help="Optional. Limit completion tokens. It can be used to avoid model collapse.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--num_workers", type=int, default=5,
                        help="Async semaphore size. This depends on your rate limit. Only work in the async version.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional. Limit evaluation to first N samples")
    args = parser.parse_args()
    main(args)