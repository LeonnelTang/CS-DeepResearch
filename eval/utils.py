import json
import re


def safe_json_loads(text: str):
    """
    安全解析LLM的JSON响应。
    处理包含Markdown格式、前后带有无关文本等常见LLM输出问题。
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # 1. 最理想情况：直接就是完美的 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # 继续尝试下面的提取逻辑

    # 2. 尝试提取 Markdown 代码块中的 JSON (如 ```json ... ```)
    # 使用 re.DOTALL 使 . 能够匹配换行符
    match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        extracted_text = match.group(1).strip()
        try:
            return json.loads(extracted_text)
        except json.JSONDecodeError:
            pass

    # 3. 尝试暴力提取大括号 {...} 或中括号 [...] 内的内容
    # 寻找第一个左括号和最后一个对应的右括号
    dict_start, dict_end = text.find('{'), text.rfind('}')
    list_start, list_end = text.find('['), text.rfind(']')

    # 判断是最外层是字典还是列表
    if dict_start != -1 and dict_end != -1 and dict_end > dict_start:
        dict_text = text[dict_start:dict_end + 1]
        try:
            return json.loads(dict_text)
        except json.JSONDecodeError:
            pass

    if list_start != -1 and list_end != -1 and list_end > list_start:
        list_text = text[list_start:list_end + 1]
        try:
            return json.loads(list_text)
        except json.JSONDecodeError:
            pass

    # 4. 如果所有尝试都失败了，抛出异常或返回 None（这里选择记录日志并返回 None）
    print(f"[Warning] Failed to safely parse JSON. Raw text snippet: {text[:100]}...")
    return None


import json


def load_local_dataset(file_path: str):
    """
    加载本地 json 数据集，同时支持标准 JSON 和 JSONL (逐行 JSON) 格式
    """
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            # 尝试作为标准 JSON 读取（如果是完整列表或字典，这里会成功）
            local_data = json.load(f)
        except json.JSONDecodeError:
            # 如果报 JSONDecodeError (例如 Extra data 错误)
            # 说明可能是 JSONL 格式，将文件指针移回开头，开始逐行读取
            f.seek(0)
            local_data = []
            for line in f:
                line = line.strip()
                if line:  # 忽略空行
                    local_data.append(json.loads(line))

    # 转换为与 HuggingFace dataset 相同的格式
    dataset = {}

    # 假设 JSON 文件是一个包含所有字段的字典列表
    if isinstance(local_data, list):
        if not local_data:
            return {}  # 防御性编程：如果数据为空，直接返回空字典

        # 如果是列表格式 [{}, {}, ...]
        # 获取所有键并确保每个项目都包含这些键
        all_keys = set()
        for item in local_data:
            all_keys.update(item.keys())

        # 为每个项目添加缺失的键并设置默认值None
        for item in local_data:
            for key in all_keys:
                if key not in item:
                    item[key] = None

        # 因为前面已经统一了所有 item 的 keys，这里使用 all_keys 生成列
        dataset = {
            key: [item[key] for item in local_data]
            for key in all_keys
        }
    else:
        # 如果是字典格式 {"field1": [...], "field2": [...]}
        dataset = local_data

    return dataset