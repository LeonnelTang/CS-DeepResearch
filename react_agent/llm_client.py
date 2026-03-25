import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# 配置 OpenAI 客户端

# client = OpenAI(
#     base_url=os.getenv("DEEPSEEK_API_BASE"),
#     api_key=os.getenv("DEEPSEEK_API_KEY")
# )

DMX_client = OpenAI(
    base_url=os.getenv("DMX_API_BASE"),
    api_key=os.getenv("DMX_API_KEY")
)

# qwen3.5-35b-a3b
# deepseek-chat

class LLMExecutor:
    """
    LLM 执行器
    """
    def __init__(self,
                 client: OpenAI,
                 async_client: AsyncOpenAI = None,
                 model_name="qwen3.5-35b-a3b",
                 ):
        self.client = client
        self.async_client = async_client
        self.model_name = model_name
        pass

    def execute(self, prompt: str) -> str:
        """
        符合 ReactCoreLoop 要求的 llm_executor 函数
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=8192
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e

if __name__ == "__main__":
    llm_executor = LLMExecutor(client=DMX_client)
    print(llm_executor.execute("请用中文回答：1+1是多少"))