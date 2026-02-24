import json


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
        return json.dumps(self.trajectory, ensure_ascii=False, indent=4)