"""
快速测试：仅测试 Ollama 连接（跳过 Embedding 模型测试）
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

from ollama_client import OllamaClient


def main():
    print("\n🔍 Ollama 连接测试")
    print("="*60)
    print()
    
    try:
        client = OllamaClient()
        print(f"Ollama 地址: {client.host}")
        print(f"模型名称: {client.model}")
        print("\n正在测试连接...\n")
        
        if client.test_connection():
            print("\n✓ 连接成功！可以开始使用系统了。")
            return True
        else:
            print("\n✗ 连接失败，请检查:")
            print("  1. PC 端 Ollama 是否运行")
            print("  2. .env 文件中的 OLLAMA_HOST 是否正确")
            print("  3. 防火墙是否允许连接")
            return False
    except Exception as e:
        print(f"\n✗ 连接出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()

