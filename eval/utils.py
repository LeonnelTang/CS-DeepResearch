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