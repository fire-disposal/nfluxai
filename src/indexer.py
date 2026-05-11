#!/usr/bin/env python3
"""倒排索引模块 - 构建与检索"""

import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from medical_terms import tokenize, extract_keywords

PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"


def build_index(chunks: List[Dict[str, Any]]) -> None:
    """构建倒排索引并持久化"""

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    invoices: Dict[str, Set[int]] = {}
    chunks_store: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        if not content:
            continue

        # 分词建索引
        tokens = tokenize(content)
        for token in tokens:
            if token not in invoices:
                invoices[token] = set()
            invoices[token].add(i)

        # 按医学关键词额外建索引（确保这些词 100% 可命中）
        kw = extract_keywords(content)
        for word in kw.get("diseases", []) + kw.get("diagnoses", []) + kw.get("symptoms", []) + kw.get("measures", []):
            if word not in invoices:
                invoices[word] = set()
            invoices[word].add(i)

        chunks_store.append({
            "id": i,
            "content": content,
            **{k: v for k, v in chunk.items() if k != "content"},
        })

    # 转为 list 以便 pickle 序列化
    invoices_serialized: Dict[str, List[int]] = {
        k: sorted(v) for k, v in invoices.items()
    }

    with open(INDEX_DIR / "invoices.pkl", "wb") as f:
        pickle.dump(invoices_serialized, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks_store, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  索引词数: {len(invoices_serialized):,}")
    print(f"  文档块数: {len(chunks_store):,}")


def load_index() -> tuple:
    """加载索引"""
    with open(INDEX_DIR / "invoices.pkl", "rb") as f:
        invoices = pickle.load(f)
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return invoices, chunks


def search_invoices(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """倒排索引检索"""
    if not INDEX_DIR.exists() or not (INDEX_DIR / "invoices.pkl").exists():
        return []

    invoices, chunks = load_index()

    # 提取查询中的关键词
    tokens = tokenize(query)

    # 收集每个 chunk 匹配到的 token 数
    chunk_scores: Counter = Counter()

    for token in tokens:
        if token in invoices:
            for chunk_id in invoices[token]:
                chunk_scores[chunk_id] += 1

    if not chunk_scores:
        return []

    # 按匹配 token 数降序，取 top_k
    ranked = chunk_scores.most_common(top_k)
    results = []
    for chunk_id, score in ranked:
        if chunk_id < len(chunks):
            chunk = dict(chunks[chunk_id])
            chunk["score"] = score
            results.append(chunk)

    return results
