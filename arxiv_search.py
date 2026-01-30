from langchain_community.utilities import ArxivAPIWrapper

# 1. 初始化封装器
# doc_content_chars_max: 限制每个文档返回的字符数，防止上下文窗口溢出
arxiv = ArxivAPIWrapper(
    top_k_results=3,
    ARXIV_MAX_QUERY_LENGTH=300,
    load_all_available_meta=True
)

# 2. 执行查询（输入可以是 论文ID、作者名或关键词）
# 查询关于 "LLM Reasoning" 的论文
docs = arxiv.run("Reasoning in Large Language Models")

print(docs)