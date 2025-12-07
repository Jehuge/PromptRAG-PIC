"""
Streamlit 用户界面：Prompt 助手 (Redesigned)
"""
import streamlit as st
import json
import time
from ollama_client import OllamaClient
from vector_store import VectorStore
from rag_generator import RAGGenerator
from config import TOP_K
try:
    from prompt_templates import STYLES
except ImportError:
    STYLES = {}

# 页面配置
st.set_page_config(
    page_title="PromptRAG - AI 绘图提示词助手",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
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

def init_components():
    """初始化组件"""
    if st.session_state.ollama_client is None:
        st.session_state.ollama_client = OllamaClient()
        st.session_state.ollama_client.warm_connection()
    
    if st.session_state.vector_store is None:
        with st.spinner("正在初始化知识库..."):
            st.session_state.vector_store = VectorStore()
            if st.session_state.vector_store.exists():
                st.session_state.vector_store.load_index()
                try:
                    st.session_state.vector_store.encoder.encode(["预热"])
                except:
                    pass
            else:
                st.error("知识库未构建，请联系管理员")
                return False
    
    if st.session_state.rag_generator is None:
        st.session_state.rag_generator = RAGGenerator(
            st.session_state.vector_store,
            st.session_state.ollama_client
        )
    return True

def display_result_card(item, index, distance=None):
    """显示单个结果卡片"""
    similarity = 1 / (1 + distance) if distance is not None else 0
    
    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <h4>🎨 参考案例 {index} <span style="font-size:0.8em;color:#888;font-weight:normal">(相似度: {similarity:.1%})</span></h4>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**主体:** {item.get('subject', 'N/A')}")
            st.markdown(f"**风格:** {item.get('art_style', 'N/A')}")
            st.markdown(f"**氛围:** {item.get('mood', 'N/A')}")
        with c2:
            elements = item.get('visual_elements', [])
            tech = item.get('technical', [])
            st.markdown("**视觉元素:**")
            st.markdown(" ".join([f"`{e}`" for e in elements[:6]]) if elements else "N/A")
            st.markdown("**技术参数:**")
            st.markdown(" ".join([f"`{t}`" for t in tech[:6]]) if tech else "N/A")
        
        with st.expander("查看原始 Prompt"):
            st.code(item.get('raw', ''), language='text')
        
        st.markdown("---")

def main():
    # 侧边栏设置
    with st.sidebar:
        st.title("🛠️ 设置")
        
        # 风格选择
        st.subheader("🎨 生成风格")
        style_options = list(STYLES.keys())
        selected_style = st.radio(
            "选择优化风格",
            options=style_options,
            format_func=lambda x: f"{STYLES[x]['icon']} {STYLES[x]['name']}",
            help="选择不同的提示词优化专家角色"
        )
        # 显示当前风格描述
        if selected_style:
            st.info(STYLES[selected_style]['description'])
            
        st.markdown("---")
        
        top_k = st.slider("参考数量 (Top K)", 1, 10, TOP_K)
        fast_mode = st.toggle("⚡ 极速模式", value=False, help="跳过检索，直接生成")
        
        st.markdown("---")
        st.caption("系统状态")
        if st.button("🔄 重连 Ollama"):
            st.session_state.ollama_client = OllamaClient()
            if st.session_state.ollama_client.test_connection():
                st.toast("Ollama 连接成功!", icon="✅")
            else:
                st.toast("Ollama 连接失败", icon="❌")
        
        if st.session_state.vector_store and st.session_state.vector_store.index:
            st.caption(f"📚 知识库: {st.session_state.vector_store.index.ntotal} 条记录")

    # 主界面
    st.markdown('<div class="main-header"><h1>🎨 AI 绘图提示词助手</h1><p>输入你的创意，生成高质量 Prompt</p></div>', unsafe_allow_html=True)

    if not init_components():
        st.stop()

    # 输入区域
    with st.container():
        user_input = st.text_area("在此输入描述 (中文/英文)", height=120, placeholder="例如：一只穿着宇航服的猫，在太空中漂浮，背景是地球，超高清，电影质感...")
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            search_btn = st.button("🔍 搜索参考", use_container_width=True)
        with col2:
            generate_btn = st.button("✨ 生成 Prompt", type="primary", use_container_width=True)
    
    # 逻辑处理
    if search_btn and user_input:
        with st.spinner("🔍 正在检索相似灵感..."):
            try:
                results = st.session_state.vector_store.search(user_input, top_k=top_k)
                if not results:
                    st.warning("未找到相关结果")
                else:
                    st.success(f"找到 {len(results)} 个高质量参考案例")
                    for i, (item, dist) in enumerate(results, 1):
                        display_result_card(item, i, dist)
            except Exception as e:
                st.error(f"检索出错: {str(e)}")

    elif generate_btn and user_input:
        # 设置选中的风格
        if st.session_state.rag_generator:
            st.session_state.rag_generator.set_style(selected_style)
            
        results = []
        if not fast_mode:
            with st.status("🚀 正在启动生成流程...", expanded=True) as status:
                st.write("🔍 正在检索参考案例...")
                results_with_dist = st.session_state.vector_store.search(user_input, top_k=top_k)
                results = [item for item, _ in results_with_dist]
                st.write(f"✅ 检索完成，找到 {len(results)} 个参考")
                
                st.write(f"✍️ 正在使用【{STYLES[selected_style]['name']}】风格构建提示词...")
                status.update(label="✨ 正在生成中...", state="running")
        
        # 生成区域
        result_container = st.container()
        with result_container:
            st.subheader("✨ 生成结果")
            
            # 流式生成
            placeholder = st.empty()
            full_response = ""
            
            try:
                if fast_mode:
                    context = f"用户意图: {user_input}"
                else:
                    context = st.session_state.rag_generator._build_context(user_input, results)
                
                # 根据不同风格，用户提示词可能稍有不同，但 currently RAGGenerator uses a fixed user prompt format.
                # The system prompt does the heavy lifting.
                prompt = f"{context}\n\n请根据以上信息，生成一段高质量的中文绘图提示词："
                
                start_time = time.time()
                for token in st.session_state.rag_generator.client.stream_generate(
                    prompt=prompt,
                    system=st.session_state.rag_generator.system_prompt
                ):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                st.caption(f"耗时: {time.time() - start_time:.2f}s")
                
                # 复制区域
                st.markdown("### 📋 复制下方内容")
                st.code(full_response, language="text")
                
            except Exception as e:
                st.error(f"生成失败: {str(e)}")

        # 显示参考资料 (如果不是极速模式)
        if not fast_mode and results:
            with st.expander("📚 查看使用的参考案例", expanded=False):
                for i, item in enumerate(results, 1):
                    display_result_card(item, i)

if __name__ == "__main__":
    main()
