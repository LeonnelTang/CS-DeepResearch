import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 配置 OpenAI 客户端
deepseek_client = OpenAI(
    base_url=os.getenv("DEEPSEEK_API_BASE"),
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

client = OpenAI(
    base_url=os.getenv("DMX_API_BASE"),
    api_key=os.getenv("DMX_API_KEY")
)

def llm_executor(prompt: str) -> str:
    """
    这是一个符合 ReactCoreLoop 要求的 llm_executor 函数
    """
    try:
        response = client.chat.completions.create(
            model="qwen-flash",  # 或者 gpt-3.5-turbo
            messages=[
                {"role": "user", "content": prompt}
            ],
            # 关键设置：ReAct 需要逻辑严谨，温度设为 0 可以减少幻觉和随机性
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        raise e