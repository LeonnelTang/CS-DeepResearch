import json
import os

class ChatMLBuilder:
    """
        ChatML格式构建器
    """
    def __init__(self):
        self.trajectory = {"messages": []}

    def add_message(self, role: str, content: str):
        """
        添加消息
        """
        self.trajectory["messages"].append({"role": role, "content": content})

    def to_str(self) -> str:
        """
        返回ChatML格式字符串
        """
        return json.dumps(self.trajectory, ensure_ascii=False)

    def save_to_jsonl(self, file_path: str):
        """
        将ChatML格式数据保存到文件
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(self.to_str() + '\n')

