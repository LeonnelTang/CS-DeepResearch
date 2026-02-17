from typing import Any, List

class BaseTool:
    """工具基类，定义标准化接口"""

    def __init__(self, name:str, description:str):
        self.name = name   # 工具名称（用于行动解析）
        self.description = description   # 工具功能描述（用于模型理解）


    def run(self, params: Any) ->str:
        """核心执行方法，子类必须实现，返回结构化观察结果"""
        raise NotImplementedError("所有工具子类必须实现run方法")