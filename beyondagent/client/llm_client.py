from abc import ABC, abstractmethod
import json
import os
import time
from typing import Any, Optional, Protocol

from loguru import logger
import requests


class DashScopeClient:
    """阿里云百炼API客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-plus", 
                 temperature: float = 0.7, max_tokens: int = 2048):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please set DASHSCOPE_API_KEY environment variable or pass it directly.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, messages: list[dict[str, str]], sampling_params: dict[str, Any]) -> str:
        return self.chat_with_retry(messages, **sampling_params)
    
    
    def chat_completion(self, messages: list[dict[str, str]], **kwargs) -> str:
        """发起聊天完成请求"""
        url = f"{self.base_url}/chat/completions"
        
        # 合并参数
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=params, timeout=600)
            if not response.ok:
                logger.error(f"API request failed: {response.text}")
                response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            return ""
    
    def chat_with_retry(self, messages: list[dict[str, str]], max_retries: int = 3, 
                       retry_delay: float = 1.0, **kwargs) -> str:
        """带重试机制的聊天完成"""
        for attempt in range(max_retries):
            try:
                result = self.chat_completion(messages, **kwargs)
                if result:  # 如果获得了有效响应
                    return result
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:  # 不是最后一次尝试
                time.sleep(retry_delay * (2 ** attempt))  # 指数退避
        
        logger.error(f"All {max_retries} attempts failed")
        return ""

