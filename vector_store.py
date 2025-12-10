"""
向量化与索引模块：使用 Embedding 模型生成向量，构建 FAISS 索引
"""
import json
import jsonlines
import os
import numpy as np
import faiss
import difflib
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from config import (
    EMBEDDING_MODEL,
    INDEX_PATH,
    METADATA_PATH,
    MODEL_CACHE_DIR,
    LOCAL_FILES_ONLY,
)


class VectorStore:
    """向量存储与检索"""
    
    # 类级别的缓存，所有实例共享同一个 encoder
    _encoder_cache = {}
    _dimension_cache = {}
    
    def __init__(self, model_name: str = None, index_path: str = None, metadata_path: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.index_path = index_path or INDEX_PATH
        self.metadata_path = metadata_path or METADATA_PATH
        
        # 使用缓存的 encoder，避免重复加载
        if self.model_name not in VectorStore._encoder_cache:
            print(f"正在加载 Embedding 模型: {self.model_name}...")
            # 检查模型是否已下载
            try:
                import os
                # 使用配置的本地缓存目录
                model_cache_path = os.path.join(MODEL_CACHE_DIR, f"models--{self.model_name.replace('/', '--')}")
                
                if os.path.exists(model_cache_path):
                    print(f"提示: 模型已在本地缓存 ({MODEL_CACHE_DIR})，正在加载...")
                else:
                    print(f"提示: 首次运行需要下载模型到本地目录 {MODEL_CACHE_DIR}（约 1-2GB），可能需要几分钟...")
            except:
                print("提示: 正在加载模型...")
            
            try:
                # 指定 cache_folder 为项目目录下的 models，必要时开启本地离线加载
                encoder = SentenceTransformer(
                    self.model_name,
                    cache_folder=MODEL_CACHE_DIR,
                    local_files_only=LOCAL_FILES_ONLY,
                )
                # 获取实际向量维度
                test_embedding = encoder.encode(["test"])
                dimension = test_embedding.shape[1]
                print(f"✓ 模型加载完成，向量维度: {dimension}")
                # 缓存 encoder 和维度
                VectorStore._encoder_cache[self.model_name] = encoder
                VectorStore._dimension_cache[self.model_name] = dimension
            except Exception as e:
                print(f"✗ 模型加载失败: {e}")
                raise
        else:
            print(f"✓ 使用内存中缓存的 Embedding 模型: {self.model_name}")
        
        # 使用缓存的 encoder
        self.encoder = VectorStore._encoder_cache[self.model_name]
        self.dimension = VectorStore._dimension_cache[self.model_name]
        
        self.index = None
        self.metadata = []
    
    def build_index(self, jsonl_path: str, incremental: bool = True):
        """
        从 JSONL 文件构建向量索引（支持增量更新）
        
        Args:
            jsonl_path: 结构化数据 JSONL 文件路径
            incremental: 是否使用增量模式（只处理新增数据）
        """
        print(f"正在读取数据: {jsonl_path}...")
        
        # 读取 JSONL 文件中的所有数据
        all_items = []
        with jsonlines.open(jsonl_path) as reader:
            for item in reader:
                all_items.append(item)
        
        print(f"✓ 读取了 {len(all_items)} 条记录")
        
        # 检查是否使用增量模式
        existing_metadata = []
        existing_raws = set()
        
        if incremental and self.exists():
            print("\n检测到现有索引，使用增量模式...")
            try:
                # 读取现有元数据
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            existing_metadata.append(item)
                            existing_raws.add(item.get('raw', ''))
                
                print(f"  现有索引: {len(existing_metadata)} 条记录")
                
                # 找出新增的记录（基于 raw 字段去重）
                new_items = [item for item in all_items if item.get('raw', '') not in existing_raws]
                
                if not new_items:
                    print("✓ 没有新数据，索引已是最新状态")
                    return
                
                print(f"  新增记录: {len(new_items)} 条")
                
                # 加载现有索引
                self.index = faiss.read_index(self.index_path)
                print(f"  已加载现有索引: {self.index.ntotal} 条")
                
                # 只处理新增数据
                texts = []
                metadata_list = []
                for item in new_items:
                    search_text = self._build_search_text(item)
                    texts.append(search_text)
                    metadata_list.append(item)
                
                # 生成新数据的向量
                if texts:
                    print(f"\n正在为 {len(texts)} 条新记录生成向量...")
                    # 使用更大的批量大小加快处理速度
                    batch_size = min(64, len(texts))
                    embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=batch_size)
                    embeddings = np.array(embeddings).astype('float32')
                    
                    # 添加到现有索引
                    print("正在将新向量添加到索引...")
                    self.index.add(embeddings)
                    
                    # 合并元数据
                    existing_metadata.extend(metadata_list)
                    metadata_list = existing_metadata
                    
                    print(f"✓ 已添加 {len(new_items)} 条新记录到索引")
                
            except Exception as e:
                print(f"⚠️  增量更新失败: {e}")
                print("   将使用全量重建模式...")
                incremental = False
        
        # 全量重建模式
        if not incremental or not self.exists():
            print("\n使用全量重建模式...")
            texts = []
            metadata_list = []
            
            for item in all_items:
                search_text = self._build_search_text(item)
                texts.append(search_text)
                metadata_list.append(item)
            
            # 生成向量
            print(f"正在为 {len(texts)} 条记录生成向量...")
            # 使用更大的批量大小加快处理速度
            batch_size = min(64, len(texts))
            embeddings = self.encoder.encode(texts, show_progress_bar=True, batch_size=batch_size)
            embeddings = np.array(embeddings).astype('float32')
            
            # 构建 FAISS 索引
            print("正在构建 FAISS 索引...")
            self.index = faiss.IndexFlatL2(self.dimension)  # L2 距离
            self.index.add(embeddings)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # 保存索引
        faiss.write_index(self.index, self.index_path)
        print(f"✓ 索引已保存: {self.index_path}")
        
        # 保存元数据
        self.metadata = metadata_list
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            for item in metadata_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ 元数据已保存: {self.metadata_path}")
        
        print(f"\n✓ 向量库构建完成！")
        print(f"  索引大小: {self.index.ntotal} 条")
    
    def _build_search_text(self, item: Dict) -> str:
        """构建用于检索的文本（组合多个字段）"""
        parts = []
        
        if item.get("subject"):
            parts.append(item["subject"])
        if item.get("art_style"):
            parts.append(item["art_style"])
        if item.get("visual_elements"):
            parts.extend(item["visual_elements"])
        if item.get("mood"):
            parts.append(item["mood"])
        if item.get("technical"):
            parts.extend(item["technical"])
        
        # 如果所有字段都为空，使用原始文本
        if not parts:
            parts.append(item.get("raw", ""))
        
        return " ".join(parts)
    
    def load_index(self):
        """加载已保存的索引"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {self.metadata_path}")
        
        print(f"正在加载索引: {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        print(f"✓ 索引加载完成，包含 {self.index.ntotal} 条记录")
        
        print(f"正在加载元数据: {self.metadata_path}...")
        self.metadata = []
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        print(f"✓ 元数据加载完成，包含 {len(self.metadata)} 条记录")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        向量检索（包含去重逻辑）
        
        Args:
            query: 查询文本
            top_k: 返回 Top-K 个结果
        
        Returns:
            (元数据, 距离) 元组列表
        """
        if self.index is None:
            raise ValueError("索引未加载，请先调用 load_index() 或 build_index()")
        
        # 生成查询向量
        query_vector = self.encoder.encode([query], show_progress_bar=False, batch_size=1)
        query_vector = np.array(query_vector).astype('float32')
        
        # 检索更多候选结果以进行去重（取 3 倍数量）
        candidate_k = top_k * 3
        distances, indices = self.index.search(query_vector, candidate_k)
        
        # 组装结果并去重
        results = []
        # 存储已采纳结果的 raw 文本，用于模糊去重
        accepted_raws = []
        
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                item = self.metadata[idx]
                raw_text = item.get('raw', '').strip()
                
                # 如果没有 raw 文本，暂时放行（或者使用其他字段构建指纹）
                if not raw_text:
                    results.append((item, float(dist)))
                    if len(results) >= top_k:
                        break
                    continue
                
                # 模糊去重检查
                is_duplicate = False
                for existing_raw in accepted_raws:
                    # 计算相似度 ratio
                    # 如果 raw 文本非常短，相似度阈值可能需要调整，这里假设 prompt 都有一定长度
                    similarity = difflib.SequenceMatcher(None, raw_text, existing_raw).ratio()
                    if similarity > 0.6:  # 相似度超过 60% 视为重复（用户反馈例子非常相似）
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                # 通过检查，加入结果集
                results.append((item, float(dist)))
                accepted_raws.append(raw_text)
                
                # 收集够了 top_k 个就停止
                if len(results) >= top_k:
                    break
        
        return results
    
    def exists(self) -> bool:
        """检查索引文件是否存在"""
        return os.path.exists(self.index_path) and os.path.exists(self.metadata_path)


if __name__ == "__main__":
    # 测试示例
    store = VectorStore()
    
    # 如果索引不存在，需要先构建
    if not store.exists():
        print("索引不存在，请先运行 ETL Pipeline 生成数据，然后构建索引")
    else:
        store.load_index()
        
        # 测试检索
        query = "赛博朋克风格的雨夜猫咪"
        results = store.search(query, top_k=3)
        
        print(f"\n查询: {query}")
        print(f"找到 {len(results)} 个结果:\n")
        for i, (metadata, distance) in enumerate(results, 1):
            print(f"{i}. 距离: {distance:.4f}")
            print(f"   主体: {metadata.get('subject', 'N/A')}")
            print(f"   风格: {metadata.get('art_style', 'N/A')}")
            print(f"   元素: {', '.join(metadata.get('visual_elements', []))}")
            print()
