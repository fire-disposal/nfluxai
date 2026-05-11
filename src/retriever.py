#!/usr/bin/env python3
"""检索引擎 - 纯倒排索引"""

from typing import Any, Dict, List, Optional, Tuple

from indexer import search_invoices


class Retriever:
    def search(
        self,
        query: str,
        top_k: int = 5,
        textbook: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = search_invoices(query, top_k=top_k * 2)
        if textbook:
            results = [r for r in results if r.get("textbook") == textbook]
        return results[:top_k]

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        textbook: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        results = self.search(query, top_k, textbook)
        if not results:
            return "未找到相关参考资料。", []

        parts = []
        cites = []
        for i, c in enumerate(results, 1):
            parts.append(f"[{i}] {c.get('content', '')}")
            cites.append({
                "index": i,
                "textbook": c.get("textbook", ""),
                "chapter": c.get("chapter", ""),
                "title": c.get("title", ""),
                "score": c.get("score", 0),
                "content": c.get("content", ""),
            })
        return "\n\n".join(parts), cites
