import json
import os

class ChatMLBuilder:
    """
        ChatML格式构建器
    """
    def __init__(self):
        self.step_trajectory = {"messages": []}
        self.traj_jsonl = []

    def add_message(self, role: str, content: str):
        """
        添加消息
        """
        self.step_trajectory["messages"].append({"role": role, "content": content})

    def step_to_str(self) -> str:
        """
        返回ChatML格式字符串
        """
        return json.dumps(self.step_trajectory, ensure_ascii=False)

    def add_to_jsonl(self):
        """
        将一轮 ChatML 格式数据添加到到jsonl
        """
        self.traj_jsonl.append(self.step_trajectory)

    def save_to_jsonl(self, file_path: str):
        """
        将ChatML格式数据保存到文件
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 保存 self.traj_jsonl
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in self.traj_jsonl:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + '\n')

        print(f"✅ 成功保存 {len(self.traj_jsonl)} 条轨迹数据至 {file_path}")


    # 清空轨迹
    def clear(self):
        self.step_trajectory = {"messages": []}

