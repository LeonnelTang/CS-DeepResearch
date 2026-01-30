import pandas as pd
import re

# 1. 读取数据
df = pd.read_csv('metadata.csv')

print(f"原始数据量: {len(df)}")


# 2. 清洗函数
def clean_abstract(text):
    if not isinstance(text, str):
        return ""

    # 替换换行符为空格
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 将多个连续空格合并为一个
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# 应用清洗
df['abstract'] = df['abstract'].apply(clean_abstract)

# 3. 过滤
# 移除过短的 abstract (少于 50 个字符不作为有效摘要)
df_corpus = df[df['abstract'].str.len() > 50].copy()

df_corpus.rename(columns={'clean_id': 'id'}, inplace=True)

print(f"清洗后数据量: {len(df_corpus)}")

# 4. 导出语料库
output_jsonl = 'corpus_arxiv_2050.jsonl'
df_corpus[['id', 'title', 'abstract']].to_json(
    output_jsonl,
    orient='records',
    lines=True,
    force_ascii=False
)
print(f"已保存 JSONL: {output_jsonl}")