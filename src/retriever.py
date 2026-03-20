#!/usr/bin/env python3
"""
护理教材检索引擎
- 语义检索 + Rerank 重排序
- 完整的引用溯源
- 支持列出原文来源
- 医学词典支持
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from medical_terms import (
    find_diseases_in_text,
    find_diagnoses_in_text,
    find_symptoms_in_text,
    classify_content_type,
)


# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "chroma_db"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_FILE = INDEX_DIR / "chunks_index.json"


class NursingRetriever:
    """护理教材检索器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.vectorstore = None
        self.embeddings = None
        self.reranker = None
        self._initialized = False
        self._index_data = None
    
    def _load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """加载配置"""
        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"
        
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        
        return {
            "embedding_model": "text2vec-base-chinese",
            "top_k": 5,
            "rerank": True,
            "rerank_top_n": 20,
        }
    
    def _load_index(self):
        """加载索引文件"""
        if INDEX_FILE.exists():
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                self._index_data = json.load(f)
    
    def initialize(self):
        """初始化检索器"""
        if self._initialized:
            return
        
        print("🔄 加载嵌入模型...")
        model_name = self.config.get("embedding_model", "text2vec-base-chinese")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        print("📂 加载向量数据库...")
        self.vectorstore = Chroma(
            persist_directory=str(DATA_DIR),
            embedding_function=self.embeddings,
            collection_name="nursing_textbooks",
        )
        
        # 加载索引
        self._load_index()
        
        # 可选：加载 Rerank 模型
        if self.config.get("rerank", True):
            try:
                from rerankers import Reranker
                print("🔄 加载 Rerank 模型...")
                self.reranker = Reranker("bge-reranker-base", model_type="flashrank")
            except Exception as e:
                print(f"⚠️ Rerank 模型加载失败：{e}，将使用纯向量检索")
                self.reranker = None
        
        self._initialized = True
        print("✅ 检索器初始化完成")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        textbook_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回数量
            textbook_filter: 教材过滤 (内科护理学/外科护理学/新编护理学基础)
        
        Returns:
            (文档，相似度分数) 列表
        """
        if not self._initialized:
            self.initialize()
        
        k = top_k or self.config.get("top_k", 5)
        rerank_top_n = self.config.get("rerank_top_n", 20)
        
        # 构建过滤器
        filter_dict = None
        if textbook_filter:
            filter_dict = {"textbook": textbook_filter}
        
        # 向量检索
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=rerank_top_n if self.reranker else k,
            filter=filter_dict,
        )
        
        if not docs_with_scores:
            return []
        
        # Rerank 重排序
        if self.reranker and len(docs_with_scores) > 1:
            try:
                docs = [d for d, _ in docs_with_scores]
                reranked = self.reranker.rank(query=query, docs=docs, doc_ids=list(range(len(docs))))
                
                # 重新映射分数
                reranked_docs = []
                for item in reranked:
                    original_idx = item.doc_id
                    doc, _ = docs_with_scores[original_idx]
                    reranked_docs.append((doc, item.score))
                
                docs_with_scores = reranked_docs
            except Exception as e:
                print(f"⚠️ Rerank 失败：{e}")
        
        # 排序并截取
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return docs_with_scores[:k]
    
    def format_citation(self, doc: Document, index: int = 0) -> str:
        """
        格式化引用信息
        
        格式：[教材/章节/小节]
        """
        meta = doc.metadata
        
        parts = []
        if meta.get("textbook"):
            parts.append(meta["textbook"])
        if meta.get("chapter_title"):
            parts.append(meta["chapter_title"])
        if meta.get("section_header"):
            parts.append(meta["section_header"])
        if meta.get("subsection_header"):
            parts.append(meta["subsection_header"])
        
        source = "/".join(parts) if parts else meta.get("filename", "未知来源")
        return f"[{source}]" if index == 0 else f"[{index}]{source}"
    
    def get_full_source(self, doc: Document) -> Dict[str, Any]:
        """
        获取完整的原文来源信息
        
        Returns:
            包含完整来源信息的字典
        """
        meta = doc.metadata
        
        return {
            "textbook": meta.get("textbook", ""),
            "filename": meta.get("filename", ""),
            "filepath": meta.get("filepath", ""),
            "chapter_num": meta.get("chapter_num", ""),
            "chapter_title": meta.get("chapter_title", ""),
            "section_header": meta.get("section_header", ""),
            "subsection_header": meta.get("subsection_header", ""),
            "title": meta.get("title", ""),
            "line_start": meta.get("line_start", 0),
            "line_end": meta.get("line_end", 0),
            "chunk_id": meta.get("chunk_id", ""),
        }
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: Optional[int] = None,
        textbook_filter: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        为 LLM 生成上下文和引用信息
        
        Returns:
            (上下文字符串，引用列表)
        """
        results = self.search(query, top_k, textbook_filter)
        
        if not results:
            return "未找到相关参考资料。", []
        
        context_parts = []
        citations = []
        
        for i, (doc, score) in enumerate(results, 1):
            citation_mark = f"[{i}]"
            citation_info = self.format_citation(doc)
            full_source = self.get_full_source(doc)
            
            # 上下文中包含引用标记和内容
            context_parts.append(f"{citation_mark} {doc.page_content}")
            
            citations.append({
                "index": i,
                "source": citation_info,
                "score": round(float(score), 4),
                "full_source": full_source,
                "metadata": doc.metadata,
                "content": doc.page_content,
            })
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def list_sources(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        列出检索结果的原文来源（不包含具体内容）
        
        Args:
            query: 查询文本
            top_k: 返回数量
        
        Returns:
            来源信息列表
        """
        results = self.search(query, top_k)
        
        sources = []
        for i, (doc, score) in enumerate(results, 1):
            full_source = self.get_full_source(doc)
            sources.append({
                "index": i,
                "citation": self.format_citation(doc, i),
                "score": round(float(score), 4),
                "textbook": full_source.get("textbook", ""),
                "filename": full_source.get("filename", ""),
                "chapter": full_source.get("chapter_title", ""),
                "section": full_source.get("section_header", ""),
                "subsection": full_source.get("subsection_header", ""),
                "line_range": f"{full_source.get('line_start', 0)}-{full_source.get('line_end', 0)}",
            })
        
        return sources
    
    def get_source_content(self, textbook: str, filename: str, 
                           line_start: int, line_end: int) -> Optional[str]:
        """
        根据来源信息读取原文内容
        
        Args:
            textbook: 教材名称
            filename: 文件名
            line_start: 起始行号
            line_end: 结束行号
        
        Returns:
            原文内容，如果读取失败返回 None
        """
        filepath = PROJECT_ROOT / "data" / "textbooks" / textbook / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # 提取指定行范围的内容
            content_lines = lines[line_start:line_end]
            return "".join(content_lines)
        except Exception as e:
            print(f"读取原文失败：{e}")
            return None


def create_prompt(query: str, context: str) -> str:
    """创建 LLM 提示词"""
    return f"""你是一名专业的护理学 AI 助手，基于提供的参考资料回答用户问题。

## 参考资料
{context}

## 回答要求
1. 仅基于上述资料回答，不要编造信息
2. 在引用资料处标注 [1], [2] 等编号
3. 如果资料不足，请说明"根据现有资料无法完整回答"
4. 使用专业但易懂的中文
5. 结构清晰，重点突出
6. 回答末尾列出参考来源

## 用户问题
{query}

## 回答
"""


def search_nursing(query: str, top_k: int = 5, textbook: Optional[str] = None) -> List[Tuple[Document, float]]:
    """快速检索"""
    retriever = NursingRetriever()
    return retriever.search(query, top_k, textbook)


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("护理教材检索引擎测试")
    print("=" * 60)
    
    retriever = NursingRetriever()
    retriever.initialize()
    
    test_queries = [
        "呼吸系统疾病病人的护理评估要点",
        "护理程序的基本步骤",
        "慢性阻塞性肺疾病的护理措施",
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询：{query}")
        print("\n📑 来源列表:")
        sources = retriever.list_sources(query, top_k=3)
        for src in sources:
            print(f"  [{src['index']}] {src['citation']}")
            print(f"      教材：{src['textbook']}")
            print(f"      文件：{src['filename']}")
            print(f"      章节：{src['chapter']} / {src['section']}")
            print(f"      相似度：{src['score']:.4f}")
            print()
