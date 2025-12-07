"""
连接测试脚本：快速测试系统各组件是否正常工作
"""
import sys
import warnings
# 忽略 urllib3 的 OpenSSL 警告（不影响功能）
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

from ollama_client import OllamaClient
from vector_store import VectorStore


def test_ollama():
    """测试 Ollama 连接"""
    print("="*60)
    print("测试 1: Ollama 连接")
    print("="*60)
    
    try:
        client = OllamaClient()
        if client.test_connection():
            print("✓ Ollama 连接测试通过\n")
            return True
        else:
            print("✗ Ollama 连接测试失败\n")
            return False
    except Exception as e:
        print(f"✗ Ollama 连接测试出错: {e}\n")
        return False


def test_embedding():
    """测试 Embedding 模型"""
    print("="*60)
    print("测试 2: Embedding 模型")
    print("="*60)
    print("正在加载 Embedding 模型（首次运行可能需要下载，请耐心等待）...")
    
    try:
        store = VectorStore()
        print("模型加载完成，正在测试编码...")
        test_text = "A cyberpunk cat"
        embedding = store.encoder.encode([test_text])
        print(f"✓ Embedding 模型加载成功")
        print(f"  向量维度: {embedding.shape[1]}")
        print(f"  测试文本: {test_text}\n")
        return True
    except Exception as e:
        print(f"✗ Embedding 模型测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """测试向量库"""
    print("="*60)
    print("测试 3: 向量库")
    print("="*60)
    
    try:
        store = VectorStore()
        if store.exists():
            store.load_index()
            print(f"✓ 向量库加载成功")
            print(f"  索引大小: {store.index.ntotal} 条")
            print()
            return True
        else:
            print("⚠️  向量库不存在（这是正常的，如果还没有构建索引）")
            print("   运行 python build_index.py 构建索引\n")
            return None  # 不是错误，只是未构建
    except Exception as e:
        print(f"✗ 向量库测试失败: {e}\n")
        return False


def main():
    """运行所有测试"""
    print("\n🔍 PromptRAG 系统测试")
    print("="*60)
    print()
    
    results = []
    
    # 测试 Ollama
    print("开始测试 Ollama 连接...\n")
    results.append(("Ollama 连接", test_ollama()))
    
    # 测试 Embedding（可能需要较长时间）
    print("开始测试 Embedding 模型...\n")
    results.append(("Embedding 模型", test_embedding()))
    
    # 测试向量库
    print("开始测试向量库...\n")
    results.append(("向量库", test_vector_store()))
    
    # 汇总
    print("="*60)
    print("测试汇总")
    print("="*60)
    
    for name, result in results:
        if result is True:
            status = "✓ 通过"
        elif result is False:
            status = "✗ 失败"
        else:
            status = "⚠️  跳过"
        print(f"{name}: {status}")
    
    print()
    
    # 给出建议
    all_passed = all(r is True or r is None for r in [r[1] for r in results])
    if all_passed:
        print("✓ 所有核心组件正常！可以开始使用系统了。")
        print("\n下一步:")
        if results[2][1] is None:
            print("  1. 运行 python process_data.py 处理数据")
            print("  2. 运行 python build_index.py 构建索引")
        print("  3. 运行 streamlit run app.py 启动应用")
    else:
        print("✗ 部分组件测试失败，请检查配置和依赖")


if __name__ == "__main__":
    main()

