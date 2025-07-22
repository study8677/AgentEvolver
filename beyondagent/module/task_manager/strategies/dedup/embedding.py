import os
import uuid
from typing import Any, Optional, Sequence
import chromadb
from chromadb.config import Settings

from beyondagent.client.embedding_client import OpenAIEmbeddingClient
from beyondagent.schema.trajectory import Trajectory


class EmbeddingClient:
    def __init__(self, similarity_threshold: float, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", 
                 api_key: Optional[str] = None, model: str = "text-embedding-v4",
                 chroma_db_path: str = "./chroma_db", collection_name: str = "trajectories"):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        assert api_key is not None, "DASHSCOPE_API_KEY is required"
        
        self._client = OpenAIEmbeddingClient(api_key=api_key, base_url=base_url, model_name=model)
        self.similarity_threshold = similarity_threshold
        
        # 初始化ChromaDB
        self._chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        # ID映射：ChromaDB使用string ID，我们需要维护int ID的映射
        self._id_mapping: dict[int, str] = {}
        self._reverse_id_mapping: dict[str, int] = {}
    
    def add(self, text: str, id: int):
        """
        添加文本和对应的ID，生成并存储嵌入向量到ChromaDB
        
        Args:
            text (str): 要添加的文本
            id (int): 文本对应的ID
        """
        # 获取文本的嵌入向量
        embedding = self._client.get_single_embedding(text)
        
        # 生成ChromaDB使用的string ID
        chroma_id = f"doc_{id}_{uuid.uuid4().hex[:8]}"
        
        # 更新ID映射
        self._id_mapping[id] = chroma_id
        self._reverse_id_mapping[chroma_id] = id
        
        # 添加到ChromaDB
        self._collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[chroma_id],
            metadatas=[{"original_id": id, "text_length": len(text)}]
        )
    
    def find_by_text(self, text: str) -> Optional[int]:
        """
        根据文本查找相似的已存储文本，返回对应的ID
        
        Args:
            text (str): 查询文本
            
        Returns:
            Optional[int]: 如果找到相似度超过阈值的文本则返回其ID，否则返回None
        """
        # 检查集合是否为空
        if self._collection.count() == 0:
            return None
        
        # 获取查询文本的嵌入向量
        query_embedding = self._client.get_single_embedding(text)
        
        # 在ChromaDB中查询最相似的文档
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,  # 只返回最相似的一个结果
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return None
        
        # 获取距离（ChromaDB返回的是距离，需要转换为相似度）
        distance = results["distances"][0][0] # type: ignore
        similarity = 1 - distance  # 余弦距离转换为余弦相似度
        
        # 检查相似度是否超过阈值
        if similarity >= self.similarity_threshold:
            chroma_id = results["ids"][0][0]
            return self._reverse_id_mapping.get(chroma_id)
        else:
            return None
    
    def find_top_k_by_text(self, text: str, k: int = 5) -> list[tuple[int, float, str]]:
        """
        根据文本查找最相似的k个文档
        
        Args:
            text (str): 查询文本
            k (int): 返回结果数量
            
        Returns:
            list[tuple[int, float, str]]: (ID, 相似度, 文档内容) 的列表
        """
        if self._collection.count() == 0:
            return []
        
        query_embedding = self._client.get_single_embedding(text)
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        result_list = []
        for i, chroma_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] # type: ignore
            similarity = 1 - distance
            document = results["documents"][0][i] # type: ignore
            original_id = self._reverse_id_mapping.get(chroma_id)
            
            if original_id is not None:
                result_list.append((original_id, similarity, document))
        
        return result_list
    
    def _embedding(self, texts: Sequence[str], bs=10) -> list[list[float]]:
        """
        批量获取文本的嵌入向量
        
        Args:
            texts (Sequence[str]): 文本序列
            bs (int): 批处理大小
            
        Returns:
            list[list[float]]: 嵌入向量列表
        """
        res: list[list[float]] = []
        for i in range(0, len(texts), bs):
            res.extend(self._client.get_multiple_embeddings(texts[i:i+bs]))
        
        return res
    
    def get_all_stored_texts(self) -> dict[int, str]:
        """
        获取所有已存储的文本
        
        Returns:
            dict[int, str]: ID到文本的映射
        """
        all_data = self._collection.get(include=["documents", "metadatas"])
        result = {}
        
        if all_data["ids"]:
            for i, chroma_id in enumerate(all_data["ids"]):
                original_id = self._reverse_id_mapping.get(chroma_id)
                if original_id is not None:
                    result[original_id] = all_data["documents"][i] # type: ignore
        
        return result
    
    def remove(self, id: int) -> bool:
        """
        删除指定ID的文本和嵌入向量
        
        Args:
            id (int): 要删除的ID
            
        Returns:
            bool: 删除成功返回True，ID不存在返回False
        """
        chroma_id = self._id_mapping.get(id)
        if chroma_id is None:
            return False
        
        try:
            self._collection.delete(ids=[chroma_id])
            
            # 清理ID映射
            del self._id_mapping[id]
            del self._reverse_id_mapping[chroma_id]
            
            return True
        except Exception:
            return False
    
    def clear(self):
        """清空所有存储的文本和嵌入向量"""
        # 删除集合中的所有数据
        try:
            self._chroma_client.delete_collection(self._collection.name)
            # 重新创建集合
            self._collection = self._chroma_client.get_or_create_collection(
                name=self._collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 清空ID映射
            self._id_mapping.clear()
            self._reverse_id_mapping.clear()
        except Exception as e:
            print(f"清空集合时出错: {e}")
    
    def size(self) -> int:
        """返回已存储的文本数量"""
        return self._collection.count()
    
    def get_collection_info(self) -> dict:
        """获取ChromaDB集合信息"""
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata
        }


def pack_trajectory(trajectory: Trajectory) -> str:
    """
    将轨迹打包成字符串
    
    Args:
        trajectory (Trajectory): 轨迹对象
        
    Returns:
        str: 打包后的字符串
    """
    res = ""
    for message in trajectory.steps:
        res += f"{message['role']}\n{message['content']}\n\n"
    
    return res


class StateRecorder:
    def __init__(self, similarity_threshold: float, chroma_db_path: str = "./chroma_db", collection_name: str = "trajectories"):
        self._client = EmbeddingClient(
            similarity_threshold=similarity_threshold,
            chroma_db_path=chroma_db_path,
            collection_name=collection_name
        )
        
        self._mp: dict[int, list[tuple[str, str]]] = {}
        self._idx = 0
    
    def add_state(self, trajectory: Trajectory, action: str, observation: str):
        """
        添加状态记录
        
        Args:
            trajectory (Trajectory): 轨迹
            action (str): 动作
            observation (str): 观察结果
        """
        key = pack_trajectory(trajectory)
        id = self._client.find_by_text(key)
        if id is None:
            id = self._idx
            self._mp[id] = []
            self._client.add(key, id)
            self._idx += 1
        
        self._mp[id].append((action, observation))
    
    def get_state(self, trajectory: Trajectory) -> list[tuple[str, str]]:
        """
        获取状态记录
        
        Args:
            trajectory (Trajectory): 轨迹
            
        Returns:
            list[tuple[str, str]]: 动作和观察结果的列表
        """
        key = pack_trajectory(trajectory)
        id = self._client.find_by_text(key)
        if id is None:
            return []
        else:
            return self._mp[id]
    
    def get_similar_states(self, trajectory: Trajectory, k: int = 5) -> list[tuple[int, float, list[tuple[str, str]]]]:
        """
        获取相似的状态记录
        
        Args:
            trajectory (Trajectory): 轨迹
            k (int): 返回数量
            
        Returns:
            list[tuple[int, float, list[tuple[str, str]]]]: (ID, 相似度, 动作观察列表) 的列表
        """
        key = pack_trajectory(trajectory)
        similar_results = self._client.find_top_k_by_text(key, k)
        
        result = []
        for original_id, similarity, _ in similar_results:
            if original_id in self._mp:
                result.append((original_id, similarity, self._mp[original_id]))
        
        return result
    
    def get_stats(self) -> dict[str, int]:
        """
        获取统计信息
        
        Returns:
            dict[str, int]: 统计信息字典
        """
        total_records = sum(len(records) for records in self._mp.values())
        return {
            "unique_trajectories": len(self._mp),
            "total_records": total_records,
            "stored_embeddings": self._client.size(),
            "chroma_collection_info": self._client.get_collection_info()
        }
    
    def clear(self):
        """清空所有记录"""
        self._mp.clear()
        self._client.clear()
        self._idx = 0


# 使用示例
if __name__ == "__main__":
    # 需要先安装chromadb: pip install chromadb
    
    # 初始化StateRecorder
    recorder = StateRecorder(
        similarity_threshold=0.8,
        chroma_db_path="./my_chroma_db",
        collection_name="trajectory_states"
    )
    
    print("ChromaDB向量数据库已初始化")
    print(f"当前统计信息: {recorder.get_stats()}")
    
    