# 快速开始指南

## 🚀 5 分钟快速上手

### 步骤 1: 创建虚拟环境（推荐）

```bash
# 进入项目目录
cd PromptRAG-TEXTtoPIC

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# 或: venv\Scripts\activate  # Windows
```

### 步骤 2: 安装依赖

```bash
# 确保虚拟环境已激活（终端应显示 (venv)）
pip install --upgrade pip
pip install -r requirements.txt
```

### 步骤 3: 配置环境

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，设置你的 PC 端 Ollama 地址：
```env
OLLAMA_HOST=http://192.168.1.100:11434  # 替换为实际 IP
OLLAMA_MODEL=qwen2.5:32b
EMBEDDING_MODEL=BAAI/bge-m3
```

### 步骤 4: 测试连接

```bash
# 确保虚拟环境已激活
python test_connection.py
```

如果看到 "✓ 所有核心组件正常！"，说明环境配置成功。

### 步骤 5: 处理数据（首次使用）

1. 将你的 Excel/CSV 文件放入 `data/raw/` 目录

2. 运行数据处理（确保虚拟环境已激活）：
```bash
python process_data.py
```

3. 构建向量索引：
```bash
python build_index.py
```

### 步骤 6: 启动应用

```bash
# 确保虚拟环境已激活
# 方式 1: 使用启动脚本（macOS/Linux）
./run.sh

# 方式 2: 直接启动
streamlit run app.py
```

浏览器会自动打开，开始使用！

## 📝 使用流程

### 日常使用（数据已处理）

1. 激活虚拟环境：`source venv/bin/activate`
2. 启动应用：`streamlit run app.py`
2. 输入创作想法（支持中文）
3. 点击"生成 Prompt"
4. 复制生成的英文 Prompt 到 ComfyUI

### 添加新数据

1. 激活虚拟环境：`source venv/bin/activate`
2. 将新 Excel/CSV 放入 `data/raw/`
3. 运行 `python process_data.py`
4. 运行 `python build_index.py` 重建索引

## ⚠️ 常见问题

### Q: Ollama 连接失败？

A: 检查以下几点：
- PC 端 Ollama 是否运行
- 防火墙是否允许 11434 端口
- `.env` 中的 `OLLAMA_HOST` 是否正确

### Q: Embedding 模型下载慢？

A: 首次运行会自动下载模型，可能需要几分钟。可以提前下载：
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
```

### Q: 处理数据很慢？

A: 这是正常的，每条数据都需要调用 Qwen 3 解析。建议：
- 先用小批量数据测试
- 确保网络连接稳定
- 可以在 PC 端监控 Ollama 日志

## 🎯 下一步

- 查看 [README.md](README.md) 了解详细文档
- 自定义提示词模板（修改 `etl_pipeline.py` 和 `rag_generator.py`）
- 调整检索参数（修改 `config.py` 中的 `TOP_K`）

