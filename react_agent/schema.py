from dataclasses import dataclass, field

@dataclass
class TAO:
    """思维-行动-观察三元组数据结构"""
    thought: str
    action: str
    observation: str