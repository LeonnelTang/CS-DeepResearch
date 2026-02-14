import pandas as pd
from collections import Counter
import json
from tqdm import tqdm
import csv
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    max_retries=3,
    timeout=60
)

class CSEntity(BaseModel):
    """Represents a single entity extracted from the abstract."""
    name: str = Field(...,
                      description="The precise name of the entity (e.g., 'ResNet-50'). Keep original capitalization.")
    category: str = Field(..., description="Category: 'Method', 'Task', 'Dataset', 'Metric', or 'Tool'.")
    specificity_score: int = Field(...,
                                   description="1-5 score. 1=Generic (e.g., 'Network'), 5=Highly Specific (e.g., 'GPT-4o-mini').")


def safe_json_loads(json_str):
    """
    尝试解析 JSON 字符串。
    如果遇到 JSONDecodeError (通常是因为 LaTeX 或 Windows 路径中的非法转义字符)，
    会自动修复非标准的反斜杠并重试。

    Args:
        json_str (str): 可能包含非法转义字符的 JSON 字符串

    Returns:
        list | dict: 解析后的 Python 对象
    """
    try:
        # 1. 尝试正常解析
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 2. 如果报错，使用正则修复非法转义
        # 正则解释：查找所有反斜杠 '\'，但排除后面跟着合法转义符 (", \, /, b, f, n, r, t, u) 的情况
        # (?!...) 是负向先行断言
        pattern = r'\\(?!["\\/bfnrtu])'

        # 将非法的单反斜杠替换为双反斜杠
        fixed_str = re.sub(pattern, r'\\\\', json_str)

        # 3. 解析修复后的字符串 (如果还报错，说明不是转义问题，而是格式真错了，直接抛出异常)
        return json.loads(fixed_str)

def extract_entity_from_corpus(corpus_path, output_file):
    """
    Extract entities from a corpus of scientific papers.

    Parameters:
    - corpus_path: Path to the CSV file containing the corpus.

    Returns:
    - A list of json, where each dictionary represents an extracted entity.
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [json.loads(line) for line in f]

    extracted_results = []

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            extracted_results = [json.loads(line) for line in f]
            extracted_paper_id = [paper['paper_id'] for paper in extracted_results if 'entities' in paper]
    else:
        extracted_paper_id = []

    print(f"Already extracted {len(extracted_results)} papers from '{output_file}'\n")

    for paper in tqdm(corpus, desc="Extracting entities", total=len(corpus)):
        if paper['id'] in extracted_paper_id:
            continue

        abstract_text = paper['abstract']
        try :
            entity_list = extract_entity_from_single_text(abstract_text)
            result = {"paper_id": paper['id'], "entities": entity_list}
        except Exception as e:
            print(f"Error: {e}")
            result = {"paper_id": paper['id']}
            continue

        extracted_results.append(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        for extracted_result in extracted_results:
            f.write(json.dumps(extracted_result, ensure_ascii=False) + '\n')

    print(f"Entity extraction completed. Processed {corpus_path} saved to '{output_file}'\n")

def extract_entity_from_single_text(abstract_text: str):

    system_prompt = """
    You are an expert Computer Science researcher building a Knowledge Graph.
    Extract technical entities from the provided abstract and output them in JSON format.

    CRITICAL RULES:
    1. **Be Specific:** Do NOT extract generic terms like "model", "algorithm", "data", "accuracy", or "deep learning". 
       - BAD: "neural network"
       - GOOD: "Convolutional Neural Network", "ResNet-101"
    2. **Version/Variant:** Always include version numbers or variants if present (e.g., "YOLOv3" instead of "YOLO").
    3. **Completeness:** If a dataset or metric is mentioned specifically, extract it.
    
    JSON OUTPUT FORMAT:
    [
        {
            "name": "The precise name of the entity (e.g., 'ResNet-50'). Keep original capitalization.",
            "category": 'Method', 'Task', 'Dataset', 'Metric', or 'Tool' etc.",
            "specificity_score": "1-5 score. 1=Generic (e.g., 'Network'), 5=Highly Specific (e.g., 'GPT-4o-mini')."
        },
        ...
    ]
    """

    # 3. 运行提取 (模拟单条)
    # abstract_text = """
    # We present MedResearcher-R1, a novel agentic framework for medical deep research.
    # Evaluated on the MedBrowseComp benchmark, it achieves a score of 27.5, surpassing Qwen2.5-32B.
    # The system utilizes a specialized Masked Trajectory Guidance (MTG) technique.
    # """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Abstract: {abstract_text}"}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=8192,
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content
        entities = safe_json_loads(response_content)


        # # 打印结果
        # print(f"--- Extracted {len(entities)} Entities ---")
        # for entity in entities:
        #     # 这里我们可以直接过滤掉泛化词 (Specificity < 3)
        #     if entity['specificity_score'] >= 3:
        #         print(f"[{entity['category']}] {entity['name']} (Score: {entity['specificity_score']})")

    except Exception as e:
        print(f"Error: {e}")

    # 输出示例预期：
    # [Method] MedResearcher-R1 (Score: 5)
    # [Dataset] MedBrowseComp (Score: 5)
    # [Method] Qwen2.5-32B (Score: 5)
    # [Method] Masked Trajectory Guidance (Score: 4)
    # [Method] MTG (Score: 4)

    return entities


def entity_counter(input_jsonl, output_csv):
    entity_stats = {}

    try:
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                data = json.loads(line)
                entities = data.get("entities", [])

                for entity in entities:
                    name = entity.get("name")
                    if not name: continue

                    # 如果是第一次遇到该实体，初始化信息
                    if name not in entity_stats:
                        entity_stats[name] = {
                            "count": 0,
                            "category": entity.get("category"),
                            "specificity_score": entity.get("specificity_score"),
                            "paper_id": data.get("paper_id")
                        }

                    # 计数加 1
                    entity_stats[name]["count"] += 1

        # 转换为列表，按出现次数从低到高排序
        sorted_results = sorted(
            [{"name": k, **v} for k, v in entity_stats.items()],
            key=lambda x: x["count"],
            reverse=False
        )

        # 打印前 5 个查看结果
        print(f"总共发现 {len(sorted_results)} 个唯一实体。前 5 个稀有实体：")
        for item in sorted_results[:5]:
            print(item)

        # 保存为 CSV 文件
        keys = ["name", "count", "category", "specificity_score", "paper_id"]
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(sorted_results)
        print(f"\n统计结果已保存至: {output_csv}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_jsonl}")
    except json.JSONDecodeError:
        print("错误：文件格式不是有效的 JSONL")


def get_rare_entities(input_csv, output_file):

    df = pd.read_csv(input_csv)

    total_entities_count = df['count'].sum()

    # 1. 筛选 count = 1, specificity_score = 5 的实体
    df_filtered = df[(df['count'] == 1) & (df['specificity_score'] == 5)]

    # 2. 过滤出 Method, Dataset
    df_filtered = df_filtered[(df_filtered['category'].isin(['Method']))]

    # 3. 过滤掉长度小于 2 的
    df_filtered = df_filtered[df_filtered['name'].str.len() > 2]

    # 2. 过滤掉单词数过多的
    df_filtered = df_filtered[df_filtered['name'].str.split().str.len() < 5]

    # 3. 过滤掉不包含大写字母的
    df_filtered = df_filtered[df_filtered['name'].str.contains('[A-Z]')]

    # 4. 过滤掉包含 $ 的（数学公式）
    df_filtered = df_filtered[~df_filtered['name'].str.contains('\$')]

    # 5. 过滤掉包含非英文字母的
    df_filtered = df_filtered[df_filtered['name'].str.contains('[a-zA-Z]')]


    rare_entities = df_filtered.to_dict('records')

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for rare_entity in rare_entities:
            f.write(json.dumps(rare_entity, ensure_ascii=False) + '\n')
        print("Rare entities saved to 'rare_entities.jsonl'")

    print(f"Found {len(rare_entities)} rare entities.")
    print("前十个稀有实体:")
    for i, entity_info in enumerate(rare_entities[:10]):
        print(f"{i+1}. 实体: {entity_info['name']}")
    # Example output: [{'entity': 'quicksort-x12', 'count': 2, 'frequency': 5.0e-7}]

if __name__ == "__main__":
    # extract_entity_from_corpus("corpus_arxiv_2050.jsonl", "extracted_entities.jsonl")
    entity_counter("extracted_entities.jsonl", "entity_stats.csv")
    get_rare_entities("entity_stats.csv", "rare_entities.jsonl")

    # with open("rare_entities.jsonl", "r", encoding="utf-8") as f:
    #     rare_entities = [json.loads(line) for line in f]
    #
    # # 考察 rare_entities 中的 category 键有哪些值，每个展示几条示例
    # category_examples = {}
    # for entity in rare_entities:
    #     category = entity.get("category")
    #     if category not in category_examples:
    #         category_examples[category] = []
    #     if len(category_examples[category]) < 3:  # 每个类别最多展示3条示例
    #         category_examples[category].append(entity["name"])
    #
    # print("Rare entities by category:")
    # for category, examples in category_examples.items():
    #     print(f"Category: {category}")
    #     for example in examples:
    #         print(f"  - {example}")
    #
    # print()
