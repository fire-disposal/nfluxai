#!/usr/bin/env python3
"""数据导入 - 纯倒排索引，无外部依赖"""

import sys
from pathlib import Path
from typing import Any, Dict, List

from medical_terms import extract_keywords
from indexer import build_index

PROJECT_ROOT = Path(__file__).parent.parent
TEXTBOOKS_ROOT = PROJECT_ROOT / "data" / "textbooks"

TEXTBOOKS = {
    "内科护理学": TEXTBOOKS_ROOT / "内科护理学",
    "外科护理学": TEXTBOOKS_ROOT / "外科护理学",
    "新编护理学基础": TEXTBOOKS_ROOT / "新编护理学基础",
}


def parse_markdown(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    cur = {"title": "", "content": [], "start": 0}

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#"):
            if cur["content"]:
                text = "".join(cur["content"]).strip()
                if len(text) >= 50:
                    chunks.append({"title": cur["title"], "content": text})
            cur = {"title": s.lstrip("#").strip(), "content": [], "start": i}
        else:
            cur["content"].append(line)

    if cur["content"]:
        text = "".join(cur["content"]).strip()
        if len(text) >= 50:
            chunks.append({"title": cur["title"], "content": text})
    return chunks


def ingest_all() -> List[Dict[str, Any]]:
    documents = []
    for name, path in TEXTBOOKS.items():
        if not path.exists():
            continue
        for md_file in sorted(path.glob("*.md")):
            stem = md_file.stem
            parts = stem.split("_", 3)
            chapter = parts[3].replace("_", " ") if len(parts) >= 4 else stem
            for chunk in parse_markdown(md_file):
                kw = extract_keywords(chunk["content"])
                documents.append({
                    "textbook": name,
                    "chapter": chapter,
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "diseases": kw["diseases"],
                    "diagnoses": kw["diagnoses"],
                    "symptoms": kw["symptoms"],
                    "measures": kw["measures"],
                })
    return documents


def main():
    docs = ingest_all()
    if not docs:
        print("无文档可处理")
        return
    print(f"解析 {len(docs)} 个文档块")
    build_index(docs)


if __name__ == "__main__":
    main()
