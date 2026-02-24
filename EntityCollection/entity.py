import asyncio

import pandas as pd
from collections import Counter
import json
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import csv
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
import re
from collections import defaultdict

load_dotenv()

client = OpenAI(
    base_url=os.environ.get("DEEPSEEK_API_BASE"),
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    max_retries=3,
    timeout=60
)

# 新增异步 aclient 供并行抽取使用
aclient = AsyncOpenAI(
    base_url=os.environ.get("DEEPSEEK_API_BASE"),
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
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

async def extract_entity_from_corpus(corpus_path, output_file, max_concurrent_tasks=20):
    """
    Extract entities from a corpus of scientific papers.

    Parameters:
    - corpus_path: Path to the CSV file containing the corpus.

    Returns:
    - A list of json, where each dictionary represents an extracted entity.
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [json.loads(line) for line in f]

    extracted_paper_ids = set()

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "entities" in data:
                        extracted_paper_ids.add(data['paper_id'])
                except json.JSONDecodeError:
                    continue

    print(f"Already extracted {len(extracted_paper_ids)} papers from '{output_file}'\n")

    # 过滤出尚未处理的文献
    pending_papers = [paper for paper in corpus if paper['id'] not in extracted_paper_ids]

    if not pending_papers:
        print("All papers have been processed.")
        return

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    file_write_lock = asyncio.Lock()

    async def process_and_save(paper):
        abstract_text = paper['abstract']
        try:
            entity_list = await extract_entity_from_single_text(abstract_text, semaphore)
            result = {"paper_id": paper['id'], "entities": entity_list}
        except Exception as e:
            print(f"Error processing paper {paper['id']}: {e}")
            result = {"paper_id": paper['id']}

        # 拿到结果后，加锁写入文件，避免并发写入时的行错乱
        async with file_write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 创建所有任务
    tasks = [process_and_save(paper) for paper in pending_papers]

    # 运行并显示进度条
    for task in async_tqdm.as_completed(tasks, desc="Extracting entities", total=len(tasks)):
        await task

    print(f"Entity extraction completed. Processed {corpus_path} saved to '{output_file}'\n")

async def extract_entity_from_single_text(abstract_text: str, semaphore: asyncio.Semaphore):

    system_prompt = """
    You are an expert Computer Science researcher building a Knowledge Graph focused on rare, niche, and long-tail technologies.
    Extract highly specialized, obscure, or novel technical entities from the provided abstract and output them in JSON format.
    
    CRITICAL RULES:
    1. **Focus on Rarity, Not Just Specificity:** Do NOT extract entities that are specific but universally known in the CS community (e.g., "ResNet-50", "BERT", "Adam optimizer", "ImageNet"). Target obscure, deeply specialized, or long-tail entities.
       - BAD (Specific but Common): "Transformer", "YOLOv3", "BLEU score", "CIFAR-10"
       - GOOD (Rare/Niche): "Performer architecture", "MADRAD optimizer", "Sinkhorn-Knopp algorithm", "Gromov-Wasserstein distance"
    2. **Novel & Bespoke Terms:** Pay special attention to newly proposed algorithms, custom niche datasets, or highly specific variants/ablations introduced in a very narrow subfield.
    3. **Mainstream Exclusion:** Completely ignore standard foundational models, ubiquitous metrics (e.g., Accuracy, F1, RMSE), and benchmark datasets, regardless of whether they have exact version numbers attached.
    
    JSON OUTPUT FORMAT:
    [
        {
            "name": "The precise name of the rare entity. Keep original capitalization.",
            "category": "'Method', 'Task', 'Dataset', 'Metric', or 'Tool'",
            "rarity_score": "1-5 score. 1=Familiar to most subfield specialists (e.g., 'Capsule Networks'), 5=Extremely Obscure/Novel (e.g., a newly minted algorithm or bespoke dataset)."
        }
    ]
    """

    # # 3. 运行提取 (模拟单条)
    # abstract_text = """
    # We present MedResearcher-R1, a novel agentic framework for medical deep research.
    # Evaluated on the MedBrowseComp benchmark, it achieves a score of 27.5, surpassing Qwen2.5-32B.
    # The system utilizes a specialized Masked Trajectory Guidance (MTG) technique.
    # """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Abstract: {abstract_text}"}
    ]

    # 使用信号量控制并发请求数
    async with semaphore:
        try:
            response = await aclient.chat.completions.create(
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
            #     # 这里我们可以直接过滤掉泛化词 (Rarity < 4)
            #     if entity['rarity_score'] >= 4:
            #         print(f"[{entity['category']}] {entity['name']} (Score: {entity['rarity_score']})")

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
                            "rarity_score": entity.get("rarity_score"),
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
        print(f"总共发现 {len(sorted_results)} 个实体。前 5 个稀有实体：")
        for item in sorted_results[:5]:
            print(item)

        # 保存为 CSV 文件
        keys = ["name", "count", "category", "rarity_score", "paper_id"]
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(sorted_results)
        print(f"\n统计结果已保存至: {output_csv}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_jsonl}")
    except json.JSONDecodeError:
        print("错误：文件格式不是有效的 JSONL")

def get_rare_entities(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    print(f"原始数据量: {len(df)}")

    df_filtered = df.copy()

    # 第一步：规则剔除
    # 剔除长度过长（>30字符）、包含怪异符号的实体
    df_filtered = df_filtered[
        (df_filtered['name'].str.len() <= 30) & 
        (~df_filtered['name'].str.contains(r'[^\w\s\-]', regex=True))
    ]
    
    # 第二步：保留高价值类别
    # 把少数类别（Framework, Model等）拉出来，人工 review
    core_entities = df[
        (df['category'].isin(['Framework', 'Model', 'Environment', 'Phenomenon', 'Concept']))
    ].copy()
    
    # 第三步：候选长尾实体
    # 过滤出 count <= 2 且 rarity_score == 5 的实体
    df_filtered = df_filtered[
        (df_filtered['count'] <= 2) & 
        (df_filtered['rarity_score'] == 5)
    ]

    # 第四步：图谱连通性 保留在同一paper_id下存在另外两个以上不同category的实体
    # 1. 构建论文ID到实体列表的映射
    paper_entities = defaultdict(list)
    for p_id, name, cat in zip(df['paper_id'], df['name'], df['category']):
        paper_entities[p_id].append((name, cat))

    # 2. 定义连通性检查函数
    def check_connection(p_id, current_name, current_category) -> bool:
        # 直接在集合推导式中过滤：名字不同，且类别也不同
        other_cats = {
            cat for name, cat in paper_entities.get(p_id, [])
            if name != current_name and cat != current_category
        }
        return len(other_cats) >= 2

    # 3. 使用列表推导式生成布尔掩码 (Boolean Mask)
    keep_mask = [
        check_connection(p_id, name, cat)
        for p_id, name, cat in zip(df_filtered['paper_id'], df_filtered['name'], df_filtered['category'])
    ]

    # 4. 过滤 DataFrame
    df_filtered = df_filtered[keep_mask]

    # 合并核心实体和过滤后的实体
    rare_entities = pd.concat([core_entities, df_filtered]).drop_duplicates(subset=['name'])


    print(f"过滤出 {len(rare_entities)} 个稀有实体。")

    # 保存结果为 CSV 格式
    rare_entities.to_csv(output_csv, index=False, encoding='utf-8')

    print(f"Rare entities saved to '{output_csv}'")
    print(f"Found {len(rare_entities)} rare entities.")

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


async def extract_entites():
    # max_concurrent_tasks 根据 API 的并发额度调整。
    await extract_entity_from_corpus("data/corpus_ICLR_2020-2025.jsonl", "data/extracted_entities.jsonl", max_concurrent_tasks=1)

    entity_counter("data/extracted_entities.jsonl", "data/entity_stats.csv")

    df = pd.read_csv("data/entity_stats.csv")
    # 输出count列和rarity_score列的数据分布情况
    print("Count列数据分布:")
    print(df['count'].value_counts().sort_index())
    print("\nRarity_score列数据分布:")
    print(df['rarity_score'].value_counts().sort_index())
    print("Category列数据分布:")
    print(df['category'].value_counts().sort_index())

    total_entities_count = df['count'].sum()
    print(f"实体出现次数共 {total_entities_count} 次。")

def main():
    # asyncio.run(extract_entites())

    get_rare_entities("data/entity_stats.csv", "data/rare_entities.csv")


if __name__ == "__main__":
    main()


