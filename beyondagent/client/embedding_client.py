import requests
import json
from typing import List, Sequence, Union, Optional, Dict, Any


class OpenAIEmbeddingClient:
    """
    OpenAI Embedding API客户端类
    支持调用符合OpenAI格式的embedding接口
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 model_name: str = "text-embedding-ada-002"):
        """
        初始化客户端
        
        Args:
            api_key (str): API密钥
            base_url (str): API基础URL，默认为OpenAI官方地址
            model_name (str): 模型名称，默认为text-embedding-ada-002
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
        # 设置请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def get_embeddings(self, texts: Union[str, Sequence[str]], 
                      model: Optional[str] = None,
                      encoding_format: str = "float",
                      dimensions: Optional[int] = None,
                      user: Optional[str] = None) -> Dict[str, Any]:
        """
        获取文本的嵌入向量
        
        Args:
            texts (Union[str, Sequence[str]]): 要获取嵌入向量的文本，可以是单个字符串或字符串列表
            model (Optional[str]): 模型名称，如果不指定则使用初始化时的模型
            encoding_format (str): 编码格式，默认为"float"
            dimensions (Optional[int]): 输出维度（某些模型支持）
            user (Optional[str]): 用户标识符
            
        Returns:
            Dict[str, Any]: API响应结果
            
        Raises:
            requests.RequestException: 请求异常
            ValueError: 参数错误
        """
        # 参数验证
        if not texts:
            raise ValueError("texts不能为空")
        
        # 构建请求数据
        payload = {
            "input": texts,
            "model": model or self.model_name,
            "encoding_format": encoding_format
        }
        
        # 添加可选参数
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if user is not None:
            payload["user"] = user
        
        # 发送请求
        url = f"{self.base_url}/embeddings"
        
        try:
            response = requests.post(
                url, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.RequestException as e:
            raise requests.RequestException(f"请求失败: {e}")
    
    def get_single_embedding(self, text: str, **kwargs) -> List[float]:
        """
        获取单个文本的嵌入向量（简化方法）
        
        Args:
            text (str): 要获取嵌入向量的文本
            **kwargs: 其他参数传递给get_embeddings方法
            
        Returns:
            List[float]: 嵌入向量
        """
        result = self.get_embeddings(text, **kwargs)
        return result['data'][0]['embedding']
    
    def get_multiple_embeddings(self, texts: Sequence[str], **kwargs) -> List[List[float]]:
        """
        获取多个文本的嵌入向量（简化方法）
        
        Args:
            texts (List[str]): 要获取嵌入向量的文本列表
            **kwargs: 其他参数传递给get_embeddings方法
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        result = self.get_embeddings(texts, **kwargs)
        return [item['embedding'] for item in result['data']]
    
    def set_model(self, model_name: str):
        """设置默认模型名称"""
        self.model_name = model_name
    
    def set_base_url(self, base_url: str):
        """设置base URL"""
        self.base_url = base_url.rstrip('/')
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self.api_key = api_key
        self.headers["Authorization"] = f"Bearer {self.api_key}"


# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    client = OpenAIEmbeddingClient(
        api_key="your-api-key-here",
        base_url="https://api.openai.com/v1",  # 或其他兼容的API地址
        model_name="text-embedding-ada-002"
    )
    
    try:
        # 获取单个文本的嵌入向量
        text = "Hello, world!"
        embedding = client.get_single_embedding(text)
        print(f"文本 '{text}' 的嵌入向量维度: {len(embedding)}")
        print(f"前5个值: {embedding[:5]}")
        
        # 获取多个文本的嵌入向量
        texts = ["Hello, world!", "How are you?", "I'm fine, thank you."]
        embeddings = client.get_multiple_embeddings(texts)
        print(f"\n处理了 {len(embeddings)} 个文本")
        for i, emb in enumerate(embeddings):
            print(f"文本 {i+1} 嵌入向量维度: {len(emb)}")
        
        # 获取完整的API响应
        full_response = client.get_embeddings("Test text")
        print(f"\n完整响应结构: {list(full_response.keys())}")
        
    except Exception as e:
        print(f"错误: {e}")