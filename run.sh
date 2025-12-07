#!/bin/bash
# 启动脚本 - 一键启动 Streamlit 应用

echo "🎨 PromptRAG - AI 绘图提示词助手"
echo "================================"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "⚠️  虚拟环境不存在，正在创建..."
    python3 -m venv venv
    echo "✓ 虚拟环境创建完成"
    echo "正在安装依赖..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # 激活虚拟环境
    source venv/bin/activate
fi

# 检查依赖
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit 未安装，正在安装依赖..."
    pip install -r requirements.txt
fi

# 检查环境变量
if [ ! -f .env ]; then
    echo "⚠️  未找到 .env 文件，使用默认配置"
    echo "   如需自定义，请复制 .env.example 为 .env 并修改"
fi

# 检查向量库
if [ ! -f db/knowledge.index ]; then
    echo "⚠️  向量库不存在，请先运行:"
    echo "   1. python process_data.py  (处理数据)"
    echo "   2. python build_index.py   (构建索引)"
    echo ""
    read -p "是否继续启动应用？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 启动应用
echo "🚀 启动应用..."
echo ""
streamlit run app.py

