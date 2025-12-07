"""
Gemini 客户端：负责与 Google Gemini API 通信
"""
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
import time
from typing import Generator

class GeminiClient:
    """Google Gemini API 客户端封装"""
    
    def __init__(self, api_key=None, model_name=None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_MODEL
        self.is_configured = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.is_configured = True
            except Exception as e:
                print(f"Gemini 配置失败: {e}")
                self.model = None
        else:
            self.model = None

    def test_connection(self) -> bool:
        """测试 API 连接"""
        if not self.is_configured:
            return False
        try:
            # 简单生成测试
            response = self.model.generate_content("Hello", generation_config={'max_output_tokens': 5})
            return True if response else False
        except Exception as e:
            print(f"Gemini 连接错误: {e}")
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"提示: 模型 {self.model_name} 可能不可用，尝试列出可用模型...")
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            print(f"- {m.name}")
                except:
                    pass
            return False

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """生成文本"""
        if not self.is_configured:
             return "错误: 未配置 Gemini API Key"
        
        full_prompt = prompt
        if system:
             full_prompt = f"System Instruction:\n{system}\n\nUser Request:\n{prompt}"
             
        generation_config = genai.types.GenerationConfig(
            temperature=temperature
        )
        
        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            return f"Gemini API Error: {str(e)}"

    def stream_generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """流式生成文本"""
        if not self.is_configured:
             yield "错误: 请先在 .env 中配置 GEMINI_API_KEY"
             return

        full_prompt = prompt
        if system:
             full_prompt = f"System Instruction:\n{system}\n\nUser Request:\n{prompt}"
             
        generation_config = genai.types.GenerationConfig(
            temperature=temperature
        )
        
        try:
            response = self.model.generate_content(
                full_prompt,
                stream=True,
                generation_config=generation_config
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                 yield f"错误: 模型 '{self.model_name}' 未找到 (404)。请在 .env 中设置正确的 GEMINI_MODEL (例如 gemini-1.5-flash)。"
            else:
                 yield f"Gemini Stream Error: {error_msg}"

    def warm_connection(self):
        """兼容接口，Gemini 不需要显式预热连接"""
        pass

    def set_model(self, model_name: str):
        """动态切换模型"""
        self.model_name = model_name
        if self.is_configured:
            try:
                self.model = genai.GenerativeModel(self.model_name)
                return True
            except Exception as e:
                print(f"切换模型失败: {e}")
                return False
        return False

if __name__ == "__main__":
    client = GeminiClient()
    if client.test_connection():
        print("Gemini 连接成功")
    else:
        print("Gemini 连接失败")
