import os
from openai import OpenAI
from typing import List, Dict, Any, Generator

class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b:free")
        
        if not self.api_key:
            raise ValueError("API key is required for LLMClient.")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(self, messages: List[Dict[str, str]], stream=True) -> Generator[str, None, None]:
        """
        向 LLM 发送聊天消息并产生响应块。
        """
        try:
            # 移除了 reasoning 参数 - 并非所有模型都支持它
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                extra_body={"reasoning": {"enabled": True}}
            )
            
            if stream:
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            else:
                yield response.choices[0].message.content
                
        except Exception as e:
            error_msg = f"调用 LLM 时出错: {str(e)}"
            # 避免在来些 Windows 系统上可能导致 UnicodeEncodeError 输出中文到控制台
            # print(error_msg) 
            yield error_msg
