"""
RAG 生成模块：结合检索结果和用户意图，生成最终 Prompt
"""
from typing import List, Dict, Any
from ollama_client import OllamaClient
from vector_store import VectorStore
from config import TOP_K
try:
    from prompt_templates import STYLES
except ImportError:
    STYLES = {}


class RAGGenerator:
    """RAG 检索增强生成器"""
    
    def __init__(self, vector_store: VectorStore, client: Any = None):
        self.vector_store = vector_store
        # 默认使用 Ollama，但也支持传入 GeminiClient
        self.client = client or OllamaClient()
        self.current_style = "generic"
        self.system_prompt = self._get_system_prompt()
    
    def set_client(self, client: Any):
        """更换生成模型客户端"""
        self.client = client
    
    def set_style(self, style_key: str):
        """设置生成风格"""
        if style_key in STYLES:
            self.current_style = style_key
            self.system_prompt = self._get_system_prompt()
            return True
        return False

    def _get_system_prompt(self) -> str:
        """获取用于生成最终 Prompt 的系统提示词"""
        style = STYLES.get(self.current_style, STYLES.get("generic"))
        return style.get("system_prompt", "")
    
    def _build_context(self, user_intent: str, retrieved_items: List[Dict]) -> str:
        """构建上下文提示词"""
        context_parts = [f"用户意图: {user_intent}\n\n参考素材（共{len(retrieved_items)}条）:\n"]
        
        for i, item in enumerate(retrieved_items, 1):
            parts = []
            if item.get('subject'):
                parts.append(f"主体:{item['subject']}")
            if item.get('art_style'):
                parts.append(f"风格:{item['art_style']}")
            if item.get('visual_elements'):
                elements = item['visual_elements'][:3]
                parts.append(f"元素:{','.join(elements)}")
            if item.get('mood'):
                parts.append(f"氛围:{item['mood']}")
            if item.get('technical'):
                tech = item['technical'][:3]
                parts.append(f"技术:{','.join(tech)}")
            
            if parts:
                context_parts.append(f"{i}. {', '.join(parts)}")
        
        return "\n".join(context_parts)
    
    def generate(self, user_intent: str, top_k: int = None) -> Dict:
        """生成最终 Prompt"""
        top_k = top_k or TOP_K
        
        # 1. 向量检索
        retrieved = self.vector_store.search(user_intent, top_k=top_k)
        retrieved_items = [item for item, _ in retrieved]
        
        # 2. 构建上下文
        context = self._build_context(user_intent, retrieved_items)
        
        # 3. 生成最终 Prompt
        user_prompt = f"{context}\n\n请根据以上信息，生成一段高质量的中文绘图提示词："
        
        final_prompt = self.client.generate(
            prompt=user_prompt,
            system=self.system_prompt,
            temperature=0.7
        )
        
        return {
            "final_prompt": final_prompt.strip(),
            "references": retrieved_items,
            "user_intent": user_intent
        }

    def stream_generate(self, user_intent: str, top_k: int = None):
        """流式生成 Prompt"""
        top_k = top_k or TOP_K

        # 1. 向量检索
        retrieved = self.vector_store.search(user_intent, top_k=top_k)
        retrieved_items = [item for item, _ in retrieved]

        # 2. 构建上下文
        context = self._build_context(user_intent, retrieved_items)
        user_prompt = f"{context}\n\n请根据以上信息，生成一段高质量的中文绘图提示词："

        # 3. 调用流式接口
        token_generator = self.client.stream_generate(
            prompt=user_prompt,
            system=self.system_prompt,
            temperature=0.7
        )

        return token_generator, retrieved_items
