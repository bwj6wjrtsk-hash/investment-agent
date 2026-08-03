"""
RAG 知识库 - 存储和检索你的投资笔记、研报等
"""
import os
import threading

import chromadb


# 知识库存储路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chromadb")
_CLIENT = None
_COLLECTION = None
_COLLECTION_LOCK = threading.Lock()


def get_collection():
    """线程安全地复用持久化客户端和集合，避免每次调用重复初始化。"""
    global _CLIENT, _COLLECTION
    if _COLLECTION is None:
        with _COLLECTION_LOCK:
            if _COLLECTION is None:
                _CLIENT = chromadb.PersistentClient(path=DB_PATH)
                _COLLECTION = _CLIENT.get_or_create_collection(
                    name="investment_knowledge",
                    metadata={"description": "个人投资知识库"},
                )
    return _COLLECTION


def add_document(content: str, metadata: dict = None, doc_id: str = None):
    """
    添加文档到知识库
    
    参数:
    - content: 文档内容
    - metadata: 元数据（如来源、日期、类型等）
    - doc_id: 文档ID（不传则自动生成）
    """
    collection = get_collection()
    if doc_id is None:
        doc_id = f"doc_{collection.count()}"
    if metadata is None:
        metadata = {}

    collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[doc_id]
    )
    return doc_id


def search_knowledge(query: str, top_k: int = 5) -> list:
    """
    在知识库中搜索相关内容
    
    参数:
    - query: 搜索查询
    - top_k: 返回最相关的前k条结果
    
    返回:
    - 相关文档列表
    """
    collection = get_collection()
    document_count = collection.count()
    if document_count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, document_count)
    )
    
    documents = []
    for i, doc in enumerate(results["documents"][0]):
        documents.append({
            "content": doc,
            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            "distance": results["distances"][0][i] if results["distances"] else None
        })
    return documents


def add_investment_note(note: str, stock_code: str = "", category: str = "笔记"):
    """添加投资笔记"""
    from datetime import datetime
    metadata = {
        "type": "note",
        "category": category,
        "stock_code": stock_code,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return add_document(note, metadata)


def add_research_report(content: str, title: str, source: str = ""):
    """添加研报内容"""
    from datetime import datetime
    metadata = {
        "type": "research_report",
        "title": title,
        "source": source,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return add_document(content, metadata)


def get_knowledge_stats() -> dict:
    """获取知识库统计信息"""
    collection = get_collection()
    return {
        "total_documents": collection.count(),
    }
