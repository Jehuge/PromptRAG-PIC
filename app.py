"""
Streamlit 用户界面：Prompt 助手 (Professional Clean Design)
"""
import streamlit as st
import time
from ollama_client import OllamaClient
from gemini_client import GeminiClient
from vector_store import VectorStore
from rag_generator import RAGGenerator
from config import TOP_K, GEMINI_MODEL
try:
    from prompt_templates import STYLES
except ImportError:
    STYLES = {}

# 页面配置
st.set_page_config(
    page_title="PromptRAG",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 样式定义 ---
st.markdown("""
<style>
    /* 引入字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    :root {
        --primary-blue: #2563eb;
        --accent-orange: #f97316;
        --accent-green: #059669;
        --accent-dark: #1f2937;
        --bg-color: #f8fafc;
    }

    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: #334155;
    }

    /* 隐藏 Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 顶部导航条 */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .brand {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--primary-blue);
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .brand-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 6px;
    }

    /* 风格选择区 */
    .style-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 风格卡片说明文字 */
    .style-desc-active {
        background: #eff6ff;
        border-left: 4px solid var(--primary-blue);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        color: #1e40af;
        margin: 1rem 0 2rem 0;
        animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* 输入框优化 */
    .stTextArea textarea {
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.2s;
        background: white !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1) !important;
    }

    /* 按钮样式重置 */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        height: auto !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s !important;
    }
    
    /* 风格选择按钮的特殊处理 (通过 Python 逻辑控制 Type) */
    
    /* 生成按钮 */
    .stButton button[kind="primary"] {
        background-color: var(--primary-blue) !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        transform: translateY(-1px);
    }

    /* 结果卡片 */
    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    
    .result-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    /* 标签 */
    .meta-tag {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 6px;
        background: #f1f5f9;
        color: #475569;
    }
    
    .meta-tag.highlight {
        background: #dbeafe;
        color: #1e40af;
    }

    /* 生成结果框 */
    .output-box {
        background: #1e293b;
        color: #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
        line-height: 1.6;
        position: relative;
        overflow: hidden;
    }
    
    .output-label {
        position: absolute;
        top: 0;
        right: 0;
        background: #334155;
        color: #94a3b8;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 0 0 0 8px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_generator' not in st.session_state:
    st.session_state.rag_generator = None
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = None
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None
if 'current_style' not in st.session_state:
    st.session_state.current_style = "generic" # 默认风格

def init_components():
    """初始化组件"""
    if st.session_state.ollama_client is None:
        st.session_state.ollama_client = OllamaClient()
        st.session_state.ollama_client.warm_connection()

    if st.session_state.gemini_client is None:
        st.session_state.gemini_client = GeminiClient()
    
    if st.session_state.vector_store is None:
        with st.spinner("系统启动中..."):
            st.session_state.vector_store = VectorStore()
            if st.session_state.vector_store.exists():
                st.session_state.vector_store.load_index()
                try:
                    st.session_state.vector_store.encoder.encode(["init"])
                except:
                    pass
            else:
                st.error("索引文件缺失")
                return False
    
    if st.session_state.rag_generator is None:
        st.session_state.rag_generator = RAGGenerator(
            st.session_state.vector_store,
            st.session_state.ollama_client
        )
    return True

def display_result(item, index, distance=None):
    """显示结果卡片"""
    similarity = 1 / (1 + distance) if distance is not None else 0
    sim_percent = f"{similarity:.0%}"
    
    subject = item.get('subject', 'N/A')
    style = item.get('art_style', 'N/A')
    
    # 视觉元素标签
    tags = item.get('visual_elements', [])[:3]
    tags_html = "".join([f'<span class="meta-tag highlight">{t}</span>' for t in tags])
    
    st.markdown(f"""
    <div class="result-card">
        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
            <span style="font-weight:700; color:#1e293b;">{subject}</span>
            <span style="color:#059669; font-weight:600; font-size:0.9rem;">{sim_percent} 匹配</span>
        </div>
        <div style="margin-bottom:0.8rem; font-size:0.9rem; color:#64748b;">
            {style}
        </div>
        <div>
            {tags_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("查看详情"):
        st.code(item.get('raw', ''), language='text')

def main():
    # 顶部导航
    st.markdown("""
        <div class="navbar">
            <div class="brand">
                <span>PromptRAG</span>
                <span class="brand-subtitle">| Professional Edition</span>
            </div>
            <div style="color:#64748b; font-size:0.9rem;">
                智能提示词构建系统
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not init_components():
        st.stop()

    # --- 1. 风格选择 (四大金刚) ---
    st.markdown('<div class="style-header"><span>Step 1.</span> 选择创作风格</div>', unsafe_allow_html=True)
    
    # 定义风格按钮配置
    style_buttons = [
        {"key": "generic", "label": "通用优化", "icon": "🌐", "desc": "适用于大多数场景，结构清晰，平衡细节"},
        {"key": "creative", "label": "艺术幻梦", "icon": "✨", "desc": "强调想象力与视觉冲击，适合奇幻/超现实创作"},
        {"key": "photography", "label": "摄影质感", "icon": "📷", "desc": "专注于光影、镜头参数与真实感还原"},
        {"key": "chinese_aesthetics", "label": "东方美学", "icon": "🎋", "desc": "融入中国传统文化元素与意境表达"}
    ]
    
    cols = st.columns(4)
    for i, btn in enumerate(style_buttons):
        with cols[i]:
            # 判断是否选中，设置按钮类型
            is_active = st.session_state.current_style == btn['key']
            btn_type = "primary" if is_active else "secondary"
            
            # 按钮本身
            if st.button(f"{btn['icon']} {btn['label']}", key=f"style_btn_{btn['key']}", type=btn_type, use_container_width=True):
                st.session_state.current_style = btn['key']
                st.rerun()

    # 显示当前选中风格的描述
    current_desc = next((b['desc'] for b in style_buttons if b['key'] == st.session_state.current_style), "")
    st.markdown(f'<div class="style-desc-active">{current_desc}</div>', unsafe_allow_html=True)

    # --- 2. 输入与配置 ---
    c_input, c_config = st.columns([2, 1])
    
    with c_input:
        st.markdown('<div class="style-header"><span>Step 2.</span> 描述画面</div>', unsafe_allow_html=True)
        user_input = st.text_area(
            "Input",
            height=180,
            placeholder="例如：赛博朋克风格的街道，雨夜，霓虹灯倒影...",
            label_visibility="collapsed"
        )
    
    with c_config:
        st.markdown('<div class="style-header"><span>Step 3.</span> 模型配置</div>', unsafe_allow_html=True)
        with st.container(border=True):
            # 后端选择
            backend = st.radio("生成后端", ["Ollama (Local)", "Gemini (Cloud)"], horizontal=True)
            
            if backend == "Gemini (Cloud)":
                gemini_model = st.text_input(
                    "模型名称", 
                    value=st.session_state.gemini_client.model_name or GEMINI_MODEL,
                    help="输入 Gemini 模型名称，如 gemini-1.5-flash"
                )
                if gemini_model != st.session_state.gemini_client.model_name:
                    st.session_state.gemini_client.set_model(gemini_model)
                
                # 设置 Client
                if st.session_state.gemini_client.is_configured:
                    st.session_state.rag_generator.set_client(st.session_state.gemini_client)
                    st.caption("✅ Gemini 连接正常")
                else:
                    st.error("❌ API Key 未配置")
            else:
                # Ollama
                st.session_state.rag_generator.set_client(st.session_state.ollama_client)
                st.caption(f"✅ Local Ollama")
            
            st.divider()
            top_k = st.slider("参考样本 (Top K)", 0, 10, TOP_K, help="设置为 0 则跳过检索直接生成")
            if top_k == 0:
                st.caption("🚀 极速模式：将直接基于您的描述生成")

    # --- 3. 操作与结果 ---
    st.write("")
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        search_only = st.button("🔍 仅检索灵感", use_container_width=True)
    with c_btn2:
        do_generate = st.button("✨ 生成 Prompt", type="primary", use_container_width=True)
    
    st.divider()

    # 逻辑处理
    if search_only and user_input:
        if top_k == 0:
            st.warning("Top K 设置为 0，已跳过检索。请增加 Top K 值以查看参考。")
        else:
            st.subheader("🔍 检索结果")
            with st.spinner("检索知识库..."):
                results = st.session_state.vector_store.search(user_input, top_k=top_k)
                if not results:
                    st.info("无相关结果")
                else:
                    grid = st.columns(2)
                    for i, (item, dist) in enumerate(results):
                        with grid[i % 2]:
                            display_result(item, i+1, dist)

    elif do_generate and user_input:
        st.session_state.rag_generator.set_style(st.session_state.current_style)
        
        col_res, col_ref = st.columns([2, 1])
        
        results = []
        
        # 1. 检索阶段 (仅当 top_k > 0)
        if top_k > 0:
            with col_ref:
                st.markdown("**📚 参考来源**")
                with st.spinner("检索中..."):
                    results_with_dist = st.session_state.vector_store.search(user_input, top_k=top_k)
                    results = [item for item, _ in results_with_dist]
                    
                    if not results:
                        st.caption("无参考数据")
                    else:
                        for i, item in enumerate(results, 1):
                            display_result(item, i)
        else:
            with col_ref:
                 st.info("💡 已跳过检索 (Top K=0)")

        # 2. 生成阶段
        with col_res:
            st.markdown("**✨ AI 生成结果**")
            res_box = st.empty()
            full_text = ""
            
            if top_k == 0 or not results:
                # 无参考模式
                context = f"用户意图: {user_input}"
            else:
                # RAG 模式
                context = st.session_state.rag_generator._build_context(user_input, results)
            
            prompt = f"{context}\n\n请根据以上信息，生成一段高质量的中文绘图提示词："
            
            try:
                # 初始显示
                res_box.markdown('<div class="output-box"><div class="output-label">GENERATING</div>▋</div>', unsafe_allow_html=True)
                
                for token in st.session_state.rag_generator.client.stream_generate(
                    prompt=prompt,
                    system=st.session_state.rag_generator.system_prompt
                ):
                    full_text += token
                    res_box.markdown(f'<div class="output-box"><div class="output-label">STREAMING</div>{full_text}▋</div>', unsafe_allow_html=True)
                
                # 完成显示
                res_box.markdown(f'<div class="output-box"><div class="output-label">DONE</div>{full_text}</div>', unsafe_allow_html=True)
                
                # 复制工具
                st.caption("Prompt 文本:")
                st.code(full_text, language="text")
                
            except Exception as e:
                st.error(f"生成错误: {str(e)}")

if __name__ == "__main__":
    main()
