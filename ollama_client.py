"""
Ollama 客户端：负责与 PC 端的 Ollama 服务通信
"""
import requests
import json
import time
from typing import Dict, Optional, Generator
from config import OLLAMA_HOST, OLLAMA_MODEL, REQUEST_TIMEOUT, MAX_RETRIES,OLLAMA_KEEP_ALIVE


class OllamaClient:
    """Ollama API 客户端封装"""
    
    def __init__(self, host: str = None, model: str = None):
        self.host = host or OLLAMA_HOST
        self.model = model or OLLAMA_MODEL
        self.base_url = f"{self.host}/api"
        # 复用 HTTP 连接，降低 TCP/TLS/握手开销
        self.session = requests.Session()

    def warm_connection(self, timeout: int = 5):
        """
        轻量预热：建立连接并保活，降低首请求延迟
        """
        try:
            resp = self.session.get(f"{self.base_url}/tags", timeout=timeout)
            resp.raise_for_status()
            return True
        except Exception:
            return False
    
    def _make_request(self, endpoint: str, data: Dict, retry_count: int = 0) -> Dict:
        """发送请求，带重试机制"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.post(
                url,
                json=data,
                timeout=REQUEST_TIMEOUT,
                stream=False
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if retry_count < MAX_RETRIES:
                wait_time = 2 ** retry_count  # 指数退避
                print(f"请求失败，{wait_time}秒后重试... (尝试 {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                return self._make_request(endpoint, data, retry_count + 1)
            else:
                raise Exception(f"请求失败，已重试 {MAX_RETRIES} 次: {str(e)}")
    
    def generate(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            system: 系统提示词
            temperature: 温度参数
        
        Returns:
            生成的文本内容
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
        }
        
        if system:
            data["system"] = system
        
        response = self._make_request("generate", data)
        return response.get("response", "")

    def stream_generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        流式生成文本，逐步返回 token
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                }
        }

        if system:
            data["system"] = system

        with self.session.post(
            f"{self.base_url}/generate",
            json=data,
            timeout=REQUEST_TIMEOUT,
            stream=True
        ) as r:
            r.raise_for_status()
            # 逐行读取，chunk_size=1 可减少等待缓冲
            for line in r.iter_lines(chunk_size=1, decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Ollama 流式返回中，done=true 表示结束
                if obj.get("done"):
                    break
                token = obj.get("response", "")
                if token:
                    yield token
    
    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """
        对话模式生成
        
        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数
        
        Returns:
            生成的文本内容
        """
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        response = self._make_request("chat", data)
        return response.get("message", {}).get("content", "")
    
    def test_connection(self) -> bool:
        """测试与 Ollama 服务的连接"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            print(f"✓ 连接成功！可用模型: {[m.get('name') for m in models]}")
            return True
        except Exception as e:
            print(f"✗ 连接失败: {str(e)}")
            return False


if __name__ == "__main__":
    # 测试连接
    client = OllamaClient()
    client.test_connection()

