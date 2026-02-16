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
    # df_filtered = df_filtered[(df_filtered['category'].isin(['Method']))]

    # 3. 过滤掉长度小于 2 的
    df_filtered = df_filtered[df_filtered['name'].str.len() > 2]

    # 2. 过滤掉单词数过多的
    df_filtered = df_filtered[df_filtered['name'].str.split().str.len() < 5]

    # 3. 过滤掉不包含大写字母的
    df_filtered = df_filtered[df_filtered['name'].str.contains('[A-Z]')]

    # 4. 过滤掉包含 $ 的（数学公式）
    df_filtered = df_filtered[~df_filtered['name'].str.contains('\$')]

    df_filtered = filter_entities_by_llm(df_filtered)

    df_filtered = df_filtered[df_filtered['keep'] == True]

    rare_entities = df_filtered.to_dict('records')

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for rare_entity in rare_entities:
            f.write(json.dumps(rare_entity, ensure_ascii=False) + '\n')
        print("Rare entities saved to 'rare_entities.jsonl'")

    print(f"Found {len(rare_entities)} rare entities.")
    # print("前十个稀有实体:")
    # for i, entity_info in enumerate(rare_entities[:10]):
    #     print(f"{i+1}. 实体: {entity_info['name']}")
    # Example output: [{'entity': 'quicksort-x12', 'count': 2, 'frequency': 5.0e-7}]

def filter_entities_by_llm(entity_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用LLM批量评估实体是否适合作为种子实体，每次处理10个实体。
    
    参数:
    - entity_df: 包含实体信息的DataFrame，必须有'name'列
    
    返回:
    - 带有'keep'列的DataFrame，标记每个实体是否保留
    """
    # 获取实体列表
    entity_list = entity_df['name'].to_list()
    
    # 分批处理，每次10个实体
    batch_size = 10
    all_results = []
    
    system_prompt = """
    You are a strict Editor for a Computer Science Knowledge Graph Benchmark. 
    Your task is to evaluate a list of extracted entities (Methods, Datasets, Tools) and decide if they are suitable as "Seed Entities" for generating hard QA questions.

    Criteria for "YES" (Keep):
    1. **Specific Proper Noun**: It must be a specific name (e.g., "Faster R-CNN", "ImageNet-1k", "AdamW").
    2. **Complete**: It should not be a fragment (e.g., "based on", "the proposed method").
    3. **Standalone**: It allows for a meaningful question without needing the specific context of a single obscure sentence.

    Criteria for "NO" (Discard):
    1. **Generic Terms**: Common nouns (e.g., "deep learning", "ablation study", "linear regression").
    2. **Fragments**: Half-sentences or verb phrases (e.g., "using the algorithm", "data processing").
    3. **Ambiguous**: Abbreviations that are too short or unclear without context (e.g., "M", "S1", "Model A").

    Output ONLY a valid JSON object with a list of results.
    Format:
    {
        "results": [
            {"entity": "Entity Name", "keep": true, "reason": "Specific method name"},
            {"entity": "Entity Name", "keep": false, "reason": "Too generic"}
        ]
    }
    """

    user_prompt_template = """
    Here is the batch of entities to evaluate:
    {entity_list}
    """

    # 分批处理实体
    for i in tqdm(range(0, len(entity_list), batch_size), desc="Processing batches", total=(len(entity_list)-1)//batch_size + 1):
        batch = entity_list[i:i + batch_size]
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_template.format(entity_list=json.dumps(batch, ensure_ascii=False))},
                ],
                max_tokens=1024,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # 解析LLM返回的结果
            llm_response = safe_json_loads(response.choices[0].message.content)
            batch_results = llm_response.get("results", [])
            all_results.extend(batch_results)
            
            # print(f"Processed batch {i//batch_size + 1}/{(len(entity_list)-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # 为失败的批次添加默认结果（全部标记为False）
            for entity in batch:
                all_results.append({"entity": entity, "keep": False, "reason": "Processing error"})
    
    # 创建映射字典
    keep_map = {item["entity"]: item["keep"] for item in all_results}
    
    # 应用到原DataFrame
    entity_df["keep"] = entity_df["name"].map(keep_map).fillna(False)
    
    return entity_df

def convert_rare_entities_to_csv(input_jsonl, output_csv):
    """
    将稀有实体的 JSONL 文件转换为 CSV 格式。

    参数:
    - input_jsonl: 输入的 JSONL 文件路径，每行是一个包含稀有实体信息的 JSON 对象。
    - output_csv: 输出的 CSV 文件路径。

    示例输入行:
    {"name": "LM-CartSeg", "count": 1, "category": "Method", "specificity_score": 5, "paper_id": 2512.03449}

    输出 CSV 包含列: name, count, category, specificity_score, paper_id
    """
    try:
        # 读取 JSONL 文件
        rare_entities = []
        with open(input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rare_entities.append(json.loads(line))

        # 定义 CSV 列名
        keys = ["name", "count", "category", "specificity_score", "paper_id", "keep"]

        # 写入 CSV 文件
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rare_entities)

        print(f"稀有实体已成功转换并保存至: {output_csv}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_jsonl}")
    except json.JSONDecodeError:
        print("错误：文件格式不是有效的 JSONL")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # extract_entity_from_corpus("corpus_arxiv_2050.jsonl", "extracted_entities.jsonl")
    # entity_counter("extracted_entities.jsonl", "entity_stats.csv")
    get_rare_entities("rare_entities.csv", "rare_entities.jsonl")

    convert_rare_entities_to_csv("rare_entities.jsonl", "rare_entities.csv")



