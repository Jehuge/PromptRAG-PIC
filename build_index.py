"""
索引构建脚本：从 JSONL 文件构建向量索引
"""
import sys
import os
from vector_store import VectorStore
from config import PROCESSED_DATA_DIR, INDEX_PATH, METADATA_PATH


def main():
    """主函数"""
    print("="*60)
    print("向量索引构建工具")
    print("="*60)
    
    # 查找 JSONL 文件
    jsonl_files = []
    if os.path.exists(PROCESSED_DATA_DIR):
        for file in os.listdir(PROCESSED_DATA_DIR):
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(PROCESSED_DATA_DIR, file))
    
    if not jsonl_files:
        print(f"\n✗ 未找到 JSONL 文件，请先运行 ETL Pipeline 处理数据")
        print(f"  数据目录: {PROCESSED_DATA_DIR}")
        return
    
    # 选择文件
    if len(jsonl_files) == 1:
        selected_file = jsonl_files[0]
        print(f"\n找到文件: {selected_file}")
    else:
        print("\n找到多个 JSONL 文件:")
        for i, f in enumerate(jsonl_files, 1):
            print(f"  {i}. {f}")
        
        choice = input("\n请选择文件编号 (直接回车使用第一个): ").strip()
        if choice:
            try:
                selected_file = jsonl_files[int(choice) - 1]
            except (ValueError, IndexError):
                print("无效选择，使用第一个文件")
                selected_file = jsonl_files[0]
        else:
            selected_file = jsonl_files[0]
    
    # 检查现有索引
    store = VectorStore()
    has_existing = store.exists()
    
    if has_existing:
        try:
            store.load_index()
            existing_count = store.index.ntotal
            print(f"\n检测到现有索引: {existing_count} 条记录")
        except:
            existing_count = 0
    else:
        existing_count = 0
    
    # 构建索引
    print(f"\n使用文件: {selected_file}")
    print(f"输出索引: {INDEX_PATH}")
    print(f"输出元数据: {METADATA_PATH}")
    
    if has_existing:
        print("\n构建模式:")
        print("  1. 增量模式 (推荐) - 只处理新增数据，快速更新")
        print("  2. 全量重建 - 删除旧索引，重新构建全部数据")
        
        mode_choice = input("\n请选择模式 (1/2，默认1): ").strip()
        incremental = mode_choice != '2'
        
        if incremental:
            print("✓ 使用增量模式")
        else:
            print("⚠️  使用全量重建模式")
    else:
        incremental = True
        print("\n首次构建，将创建新索引")
    
    confirm = input("\n确认开始构建？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    try:
        store.build_index(selected_file, incremental=incremental)
        print("\n✓ 构建完成！")
    except Exception as e:
        print(f"\n✗ 构建失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

