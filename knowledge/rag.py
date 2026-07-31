"""
RAG 知识库 - 存储和检索你的投资笔记、研报等
"""
import os
import chromadb
from chromadb.config import Settings


# 知识库存储路径
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chromadb")


def get_collection():
    """获取或创建知识库集合"""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="investment_knowledge",
        metadata={"description": "个人投资知识库"}
    )
    return collection


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
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count())
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
