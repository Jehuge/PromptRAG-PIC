"""
配置文件：管理 Ollama 服务端连接信息和其他系统参数
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Ollama 服务端配置（PC 端）
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")  # 根据实际模型名称调整

# Embedding 模型配置（运行在 Mac 端）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 多语言支持好
# 备选：EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32"  # 图像语义对齐
MODEL_CACHE_DIR = os.path.join(os.getcwd(), "models")  # 模型下载缓存目录

# Ollama 保活配置（降低 TTFT）
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")  # 示例：30m、2h；设置为 "0" 关闭保活

# Gemini API 配置
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# 默认使用 1.5-flash (速度快/免费额度高)，也可改为 gemini-1.5-pro
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

# 数据路径配置
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 存放原始 Excel/CSV
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")  # 存放处理后的 JSONL

# 向量索引配置
DB_DIR = "db"  # 向量索引数据库目录
INDEX_PATH = os.path.join(DB_DIR, "knowledge.index")
METADATA_PATH = os.path.join(DB_DIR, "metadata.jsonl")
VECTOR_DIM = 1024  # bge-m3 的维度，如果使用其他模型需要调整

# RAG 检索配置
TOP_K = 5  # 检索 Top-K 个相似结果

# 请求配置
REQUEST_TIMEOUT = 300  # Ollama 请求超时时间（秒）
MAX_RETRIES = 3  # 最大重试次数
