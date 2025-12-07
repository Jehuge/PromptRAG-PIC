"""
RAG 生成模块：结合检索结果和用户意图，生成最终 Prompt
"""
from typing import List, Dict
from ollama_client import OllamaClient
from vector_store import VectorStore
from config import TOP_K
try:
    from prompt_templates import STYLES
except ImportError:
    # Fallback if prompt_templates.py is missing or error
    STYLES = {
        "generic": {
            "system_prompt": """你是一位专业的 AI 绘图提示词工程师。你的任务是根据用户意图和参考素材，生成一段高质量、可直接用于 ComfyUI 的中文提示词。
要求：
1. 提示词必须是中文
2. 逻辑通顺，结构清晰
3. 包含主体、风格、视觉元素、氛围和技术参数
4. 直接输出提示词，不要包含任何解释性文字
"""
        }
    }


class RAGGenerator:
    """RAG 检索增强生成器"""
    
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient = None):
        self.vector_store = vector_store
        self.client = ollama_client or OllamaClient()
        self.current_style = "generic"
        self.system_prompt = self._get_system_prompt()
    
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
        """构建上下文提示词（优化版：更简洁）"""
        # 简化上下文，只保留关键信息
        context_parts = [f"用户意图: {user_intent}\n\n参考素材（共{len(retrieved_items)}条）:\n"]
        
        for i, item in enumerate(retrieved_items, 1):
            # 只保留最重要的信息，减少 token 数量
            parts = []
            if item.get('subject'):
                parts.append(f"主体:{item['subject']}")
            if item.get('art_style'):
                parts.append(f"风格:{item['art_style']}")
            if item.get('visual_elements'):
                # 只取前3个元素
                elements = item['visual_elements'][:3]
                parts.append(f"元素:{','.join(elements)}")
            if item.get('mood'):
                parts.append(f"氛围:{item['mood']}")
            if item.get('technical'):
                # 只取前3个技术参数
                tech = item['technical'][:3]
                parts.append(f"技术:{','.join(tech)}")
            
            if parts:
                context_parts.append(f"{i}. {', '.join(parts)}")
        
        return "\n".join(context_parts)
    
    def generate(self, user_intent: str, top_k: int = None) -> Dict:
        """
        生成最终 Prompt
        
        Args:
            user_intent: 用户意图（中文或英文）
            top_k: 检索数量（默认使用配置值）
        
        Returns:
            包含生成结果和参考素材的字典
        """
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
        
        # 清理输出
        final_prompt = final_prompt.strip()
        
        return {
            "final_prompt": final_prompt,
            "references": retrieved_items,
            "user_intent": user_intent
        }

    def stream_generate(self, user_intent: str, top_k: int = None):
        """
        流式生成 Prompt，返回 (token_generator, references)
        """
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


if __name__ == "__main__":
    # 测试示例
    from vector_store import VectorStore
    
    # 加载向量库
    store = VectorStore()
    if store.exists():
        store.load_index()
        
        # 创建生成器
        generator = RAGGenerator(store)
        
        # 测试生成
        result = generator.generate("赛博朋克风格的雨夜猫咪")
        
        print("\n" + "="*60)
        print("用户意图:", result["user_intent"])
        print("\n生成的 Prompt:")
        print(result["final_prompt"])
        print("\n参考素材数量:", len(result["references"]))
    else:
        print("向量库不存在，请先构建索引")
