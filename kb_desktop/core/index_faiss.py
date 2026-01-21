import os
import json
import numpy as np
import faiss
import shutil
import tempfile
from typing import List, Tuple

class FaissIndex:
    """
    用于向量存储和相似性搜索的 FAISS 索引管理器。
    """
    
    def __init__(self, index_path=None, meta_path=None):
        """
        初始化 FAISS 索引管理器。
        
        Args:
            index_path: 保存/加载 FAISS 索引文件的路径
            meta_path: 保存/加载元数据 (chunk_id 映射) 的路径
        """
        # 如果没有指定路径，使用 kb_desktop/data/ 目录
        if index_path is None or meta_path is None:
            # 获取 index_faiss.py 的目录（kb_desktop/core）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 往上一级到 kb_desktop 目录
            kb_desktop_dir = os.path.dirname(current_dir)
            if index_path is None:
                index_path = os.path.join(kb_desktop_dir, "data", "faiss.index")
            if meta_path is None:
                meta_path = os.path.join(kb_desktop_dir, "data", "meta.json")
        
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.chunk_ids = []  # 映射 vector_id -> chunk_id
        self.dimension = None
        
        # 确保 data 目录存在
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    def build_index(self, vectors: np.ndarray, chunk_ids: List[int], dimension: int):
        """
        从向量构建新的 FAISS 索引。
        
        Args:
            vectors: 形状为 (n_vectors, dimension) 的 numpy 数组
            chunk_ids: 与每个向量对应的 chunk ID 列表
            dimension: 向量维度
        """
        if len(vectors) != len(chunk_ids):
            raise ValueError("Number of vectors must match number of chunk_ids")
        
        self.dimension = dimension
        self.chunk_ids = chunk_ids
        
        # 创建 FAISS 索引 (L2 距离)
        # 对于 MVP，我们使用 IndexFlatL2（精确搜索）
        # 对于更大的数据集，请考虑 IndexIVFFlat 或 IndexHNSW
        self.index = faiss.IndexFlatL2(dimension)
        
        # 将向量添加到索引
        self.index.add(vectors.astype('float32'))
        
        print(f"Built FAISS index with {self.index.ntotal} vectors, dimension={dimension}")
    
    def add_to_index(self, vectors: np.ndarray, chunk_ids: List[int]):
        """
        将新向量添加到现有索引（增量索引）。
        
        Args:
            vectors: 形状为 (n_vectors, dimension) 的 numpy 数组
            chunk_ids: 与每个向量对应的 chunk ID 列表
        """
        if self.index is None:
            raise ValueError("No existing index. Build an index first.")
        
        if len(vectors) != len(chunk_ids):
            raise ValueError("Number of vectors must match number of chunk_ids")
        
        # 检查维度兼容性
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # 添加向量
        self.index.add(vectors.astype('float32'))
        self.chunk_ids.extend(chunk_ids)
        
        print(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")

    
    def save(self):
        """
        将索引和元数据保存到磁盘。
        """
        if self.index is None:
            raise ValueError("No index to save. Build or load an index first.")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        
        # 保存 FAISS 索引
        # 注意：Faiss 在 Windows 上不支持非 ASCII 路径
        # 解决方案：先保存到临时文件（ASCII路径），然后移动到目标位置
        try:
            # 创建临时文件
            fd, temp_path = tempfile.mkstemp(suffix=".index")
            os.close(fd)
            
            # 写入索引到临时文件
            faiss.write_index(self.index, temp_path)
            
            # 如果目标文件存在，先删除（Windows上shutil.move覆盖有时会失败）
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
                
            # 移动到目标位置
            shutil.move(temp_path, self.index_path)
        except Exception as e:
            # 如果失败，清理临时文件
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
        
        # 保存元数据
        meta = {
            "dimension": self.dimension,
            "chunk_ids": self.chunk_ids,
            "total": self.index.ntotal
        }
        
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"Saved index to {self.index_path}")
    
    def load(self):
        """
        从磁盘加载索引和元数据。
        成功返回 True，如果文件不存在返回 False。
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            return False
        
        try:
            # 加载 FAISS 索引
            # 由于 FAISS 在 Windows 上不支持非 ASCII 路径，我们需要先复制到临时文件
            fd, temp_path = tempfile.mkstemp(suffix=".index")
            os.close(fd)
            
            # 复制到临时文件
            shutil.copy2(self.index_path, temp_path)
            
            try:
                self.index = faiss.read_index(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # 加载元数据
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            self.dimension = meta['dimension']
            self.chunk_ids = meta['chunk_ids']
            
            print(f"Loaded index with {self.index.ntotal} vectors, dimension={self.dimension}")
            return True
            
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
        """
        搜索 k 个最近邻。
        
        Args:
            query_vector: 查询向量（长度为 dimension 的 1D 数组）
            k: 返回的最近邻数量
            
        Returns:
            (distances, chunk_ids) 的元组
        """
        if self.index is None:
            raise ValueError("No index loaded. Build or load an index first.")
        
        # 如果需要，重新整形为 2D
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # 搜索
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        # 将索引映射到 chunk_ids
        result_chunk_ids = [self.chunk_ids[idx] for idx in indices[0]]
        result_distances = distances[0].tolist()
        
        return result_distances, result_chunk_ids
    
    def get_stats(self) -> dict:
        """
        获取索引统计信息。
        """
        if self.index is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_chunks": len(self.chunk_ids)
        }
