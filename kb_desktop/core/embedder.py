import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

# 从 .env 文件加载环境变量（如果存在）
load_dotenv()

class Embedder:
    """
    支持 OpenAI 兼容 API 的嵌入适配器 (OpenAI v1.x)。
    """
    
    def __init__(self, api_key=None, base_url=None, model=None):
        """
        初始化嵌入器。
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY environment variable in .env file."
            )
        
        # 配置 OpenAI 客户端 (v1.x 风格)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本列表的嵌入。
        """
        if not texts:
            return []
        
        # 确保替换换行符（Ada 的常规做法）
        texts = [t.replace("\n", " ") for t in texts]
        
        try:
            # 调用 OpenAI API
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # 从响应中提取嵌入
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            # 回退或错误
            raise Exception(f"Failed to get embeddings: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]
    
    def get_dimension(self) -> int:
        """
        获取此模型的嵌入维度。
        """
        # 已知模型的字典
        known_dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return known_dims.get(self.model, 1536)
