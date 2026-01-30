import torch
from keybert import KeyBERT
from transformers import BertModel, BertTokenizer
from collections import Counter
from tqdm import tqdm
import json

# ==========================================
# 1. 配置模型
# ==========================================
# 使用 AllenAI 原版 SciBERT
MODEL_NAME = "allenai/scibert_scivocab_uncased"

print(f"Loading SciBERT model: {MODEL_NAME}...")
# KeyBERT 允许自定义底层模型，这里我们要强制使用 SciBERT 而不是默认的 MiniLM
hf_model = BertModel.from_pretrained(MODEL_NAME)
hf_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 初始化提取器
kw_model = KeyBERT(model=hf_model, tokenizer=hf_tokenizer)

# ==========================================
# 2. 模拟数据加载 (替换为你的2048条摘要)
# ==========================================
# 假设这是你清洗后的论文摘要列表
abstracts = [
    "We propose a novel framework using Graph Neural Networks for molecular generation.",
    "The transformer architecture has revolutionized Natural Language Processing tasks.",
    "We introduce QuickSort-X, a variation of the classic sorting algorithm with O(n) complexity.",
    # ... 你的其他2000多条数据 ...
]

# ==========================================
# 3. 批量提取实体
# ==========================================
entity_counter = Counter()
total_entities = 0

print("Extracting entities from abstracts...")

# 针对 CS 领域的参数调整：
# keyphrase_ngram_range=(1, 3): 提取 1-3 个词的短语 (如 "transformer", "neural networks")
# stop_words='english': 去除常见停用词
# use_maxsum=True: 增加结果的多样性，避免提取出含义重复的词
for text in tqdm(abstracts):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=5,  # 每篇摘要提取前5个最重要的实体
        use_mmr=True,  # 使用最大边界相关算法 (MMR) 减少冗余
        diversity=0.7  # 多样性越高，提取的实体越不相似
    )

    # keywords 返回格式为 [('entity', score), ...]
    # 我们只需要实体名称
    entities = [kw[0] for kw in keywords]
    entity_counter.update(entities)
    total_entities += len(entities)

# 保存实体抽取结果到文件
print("Saving entity extraction results...")
with open('entity_test.jsonl', 'w', encoding='utf-8') as f:
    f.write(json.dumps({
        'entity_counts': dict(entity_counter),
        'total_entities': total_entities
    }, ensure_ascii=False) + '\n')

print("Entity extraction results saved to 'entity_extraction_results.json'")
