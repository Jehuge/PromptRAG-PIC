# PromptRAG - 分布式本地 AI 绘图知识库系统

基于 Qwen 3 和 RAG 技术的智能绘图提示词生成系统，支持 MacBook M3 (控制端) 与 PC RTX 5070 (算力端) 分布式部署。

## 🎯 核心特性

- **高智商解析**: 使用 Qwen 3 将原始提示词智能解析为结构化数据
- **分布式架构**: Mac 端处理业务逻辑，PC 端提供 AI 算力
- **本地化部署**: 数据隐私安全，无订阅成本
- **向量检索**: 基于 FAISS 的高效相似度检索
- **智能生成**: RAG 增强的提示词生成，直接用于 ComfyUI

## 📋 系统要求

### Mac 端 (控制端)
- macOS (M3 芯片)
- Python 3.8+
- 16GB+ RAM

### PC 端 (算力端)
- Windows/Linux
- RTX 5070 (或同等算力)
- Ollama 已安装并运行
- Qwen 3 模型已下载

## 🚀 快速开始

### 1. 创建虚拟环境（推荐）

```bash
# 进入项目目录
cd PromptRAG-TEXTtoPIC

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

激活后，终端提示符前会显示 `(venv)`。

### 2. 安装依赖

```bash
# 确保虚拟环境已激活
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

**注意**: 首次安装可能需要较长时间，因为需要下载 Embedding 模型和大型依赖包（如 PyTorch、FAISS 等）。

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置 PC 端的 Ollama 地址：

```env
OLLAMA_HOST=http://192.168.1.100:11434  # 替换为你的 PC 端 IP
OLLAMA_MODEL=qwen2.5:32b                 # 根据实际模型名称调整
EMBEDDING_MODEL=BAAI/bge-m3             # Embedding 模型
```

### 4. 测试连接

```bash
# 确保虚拟环境已激活
python test_connection.py
```

或者直接测试 Ollama 连接：

```bash
python ollama_client.py
```

如果看到 "✓ 连接成功！"，说明配置正确。

### 5. 处理数据（第二阶段）

将原始 Excel/CSV 文件放入 `data/raw/` 目录，然后运行：

```bash
python process_data.py
```

脚本会：
- 读取 Excel/CSV 文件
- 逐条发送给 PC 端的 Qwen 3 进行结构化解析
- 生成 `data/processed/structured_data.jsonl` 文件

**注意**: 这个过程可能需要数小时，取决于数据量。建议先用小批量数据测试。

### 6. 构建向量索引（第三阶段）

```bash
python build_index.py
```

脚本会：
- 读取结构化 JSONL 文件
- 使用 Embedding 模型生成向量（首次运行会自动下载模型）
- 构建 FAISS 索引
- 保存到 `db/` 目录：`db/knowledge.index` 和 `db/metadata.jsonl`

### 7. 启动应用（第四阶段）

```bash
# 方式 1: 使用启动脚本（macOS/Linux）
./run.sh

# 方式 2: 直接启动
streamlit run app.py
```

浏览器会自动打开，访问 `http://localhost:8501`

## 📁 项目结构

```
PromptRAG-TEXTtoPIC/
├── app.py                 # Streamlit 主界面
├── config.py              # 配置文件
├── ollama_client.py       # Ollama 客户端
├── etl_pipeline.py       # ETL 数据处理管道
├── vector_store.py       # 向量存储与检索
├── rag_generator.py      # RAG 生成器
├── process_data.py       # 数据处理脚本
├── build_index.py        # 索引构建脚本
├── test_connection.py    # 系统测试脚本
├── test_ollama_only.py   # Ollama 连接测试脚本
├── requirements.txt      # 依赖列表
├── .env.example          # 环境变量示例
├── .env                  # 环境变量（需自行创建）
├── run.sh                # 启动脚本（macOS/Linux）
├── README.md             # 项目文档
├── README_CN.md          # 中文说明文档
├── QUICKSTART.md         # 快速开始指南
├── venv/                 # Python 虚拟环境（需自行创建）
├── data/
│   ├── raw/              # 原始数据（Excel/CSV）
│   └── processed/        # 处理后的 JSONL
└── db/                   # 向量索引数据库目录
    ├── knowledge.index   # FAISS 向量索引（构建后生成）
    └── metadata.jsonl    # 元数据文件（构建后生成）
```

## 🔧 使用说明

### 虚拟环境管理

**激活虚拟环境**（每次使用前）:
```bash
source venv/bin/activate  # macOS/Linux
```

**退出虚拟环境**:
```bash
deactivate
```

**重新创建虚拟环境**（如果出现问题）:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 数据处理流程

1. **准备数据**: 将 Excel/CSV 文件放入 `data/raw/` 目录
2. **运行 ETL**: `python process_data.py` - 生成结构化 JSONL
3. **构建索引**: `python build_index.py` - 生成向量索引

### 应用使用

1. **启动应用**: `streamlit run app.py`（确保虚拟环境已激活）
2. **输入创作想法**（支持中文）
3. **点击"生成 Prompt"**
4. **复制生成的英文 Prompt** 到 ComfyUI

## ⚙️ 配置说明

### Ollama 服务端配置（PC 端）

确保 Ollama 允许远程访问：

**Linux/Mac**:
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**Windows**:
修改 Ollama 配置文件或使用环境变量设置监听地址。

### 模型选择

- **Qwen 3**: 推荐 32B 或更高版本，用于复杂理解任务
  - 模型名称示例: `qwen2.5:32b`, `qwen2.5:72b`
- **Embedding**: 
  - `BAAI/bge-m3` (推荐，多语言支持好，1024 维)
  - `sentence-transformers/clip-ViT-B-32` (图像语义对齐，512 维)

### 环境变量说明

在 `.env` 文件中可以配置：

- `OLLAMA_HOST`: Ollama 服务地址（默认: `http://localhost:11434`）
- `OLLAMA_MODEL`: 使用的模型名称（默认: `qwen2.5:32b`）
- `EMBEDDING_MODEL`: Embedding 模型（默认: `BAAI/bge-m3`）

## 🐛 故障排除

### 虚拟环境问题

**问题**: 找不到 `venv` 命令
- **解决**: 使用 `python3 -m venv venv` 而不是 `venv venv`

**问题**: 激活后仍使用系统 Python
- **解决**: 检查激活是否成功，终端应显示 `(venv)` 前缀

### 连接失败

1. 检查 PC 端 Ollama 是否运行
2. 检查防火墙设置（确保 11434 端口开放）
3. 验证 `.env` 中的 `OLLAMA_HOST` 配置是否正确
4. 尝试在 PC 端测试: `curl http://localhost:11434/api/tags`

### 依赖安装失败

**问题**: pip 安装超时或失败
- **解决**: 使用国内镜像源
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**问题**: FAISS 安装失败（M1/M2/M3 Mac）
- **解决**: 使用 `faiss-cpu` 版本（已在 requirements.txt 中）

### 索引构建失败

1. 确保已运行 `process_data.py` 生成 JSONL
2. 检查磁盘空间（向量索引可能较大）
3. 验证 Embedding 模型是否正确下载
4. 检查内存是否充足（建议 16GB+）

### 生成质量不佳

1. 调整 `rag_generator.py` 中的系统提示词
2. 增加检索数量（修改 `config.py` 中的 `TOP_K`）
3. 优化 ETL Pipeline 中的解析提示词（`etl_pipeline.py`）
4. 确保 Qwen 3 模型版本足够大（推荐 32B+）

### Embedding 模型下载慢

首次运行会自动下载模型，可能需要较长时间。可以提前下载：

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
```

## 📝 数据格式

### 输入格式 (Excel/CSV)

单列或多列，包含原始提示词文本。脚本会自动识别第一列。

示例 Excel 结构：
| Prompt |
|--------|
| A cyberpunk cat in rainy night, neon lights, 8k masterpiece |
| Beautiful sunset over mountains, oil painting style |

### 输出格式 (JSONL)

每行一个 JSON 对象：

```json
{
  "subject": "A cyberpunk cat in rainy night",
  "art_style": "Cyberpunk",
  "visual_elements": ["Neon lights", "Rain", "Cat", "Urban street"],
  "mood": "Gloomy",
  "technical": ["8k", "Masterpiece", "Highly detailed", "Ray tracing"],
  "raw": "A cyberpunk cat in rainy night, neon lights, 8k masterpiece"
}
```

## 🔒 隐私与安全

- ✅ 所有数据在本地处理
- ✅ 不依赖任何云服务
- ✅ 向量索引可离线使用
- ✅ 虚拟环境隔离依赖
- ✅ 支持内网部署

## 📚 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 5 分钟快速上手指南
- [config.py](config.py) - 详细配置说明

## 🛠️ 开发说明

### 添加新功能

1. 在虚拟环境中开发
2. 添加新依赖到 `requirements.txt`
3. 更新文档

### 代码结构

- `ollama_client.py`: 封装 Ollama API 调用
- `etl_pipeline.py`: 数据清洗和结构化处理
- `vector_store.py`: 向量化与检索核心逻辑
- `rag_generator.py`: RAG 生成逻辑
- `app.py`: Streamlit UI 界面

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**提示**: 首次使用建议按照步骤 1-4 完成环境配置，然后使用 `test_connection.py` 验证系统是否正常工作。
