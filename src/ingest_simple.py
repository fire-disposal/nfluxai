#!/usr/bin/env python3
"""
简化版数据导入 - 使用 sentence-transformers 直接嵌入
"""

import os
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NFLUX_ROOT = Path("/home/firedisposal/nflux")

TEXTBOOKS = {
    "内科护理学": NFLUX_ROOT / "内科护理学",
    "外科护理学": NFLUX_ROOT / "外科护理学",
    "新编护理学基础": NFLUX_ROOT / "新编护理学基础",
}


def extract_metadata(filepath: Path, content: str, textbook: str) -> Dict[str, Any]:
    """提取元数据"""
    metadata = {
        "source": str(filepath),
        "filename": filepath.name,
        "textbook": textbook,
    }
    
    # 从文件名提取章节
    filename = filepath.stem
    match = re.match(r".+?_(\d+)_(第 [^ 章]+ 章)_(.+)", filename)
    if match:
        metadata["chapter_num"] = match.group(1)
        metadata["chapter"] = match.group(2)
        metadata["section"] = match.group(3).replace("_", " ")
    
    # 提取标题
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        metadata["title"] = title_match.group(1)
    
    return metadata


def split_by_headers(content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按标题分块"""
    chunks = []
    
    # 按二级标题分割
    pattern = r"(^##\s+.+?$)"
    sections = re.split(pattern, content, flags=re.MULTILINE)
    
    current_header = ""
    for i, section in enumerate(sections):
        if re.match(r"^##\s+", section):
            current_header = section.strip()
        elif current_header and section.strip():
            chunk_content = f"{current_header}\n\n{section.strip()}"
            chunk_id = hashlib.md5(chunk_content[:100].encode()).hexdigest()[:16]
            
            chunk_meta = metadata.copy()
            chunk_meta["header"] = current_header.replace("## ", "")
            chunk_meta["chunk_id"] = chunk_id
            
            chunks.append({
                "content": chunk_content,
                "metadata": chunk_meta,
            })
    
    # 如果没有二级标题，按一级标题分
    if len(chunks) == 0:
        sections = re.split(r"(^#\s+.+?$)", content, flags=re.MULTILINE)
        for i, section in enumerate(sections):
            if re.match(r"^#\s+", section):
                if i + 1 < len(sections) and sections[i + 1].strip():
                    chunk_content = f"{section}\n\n{sections[i + 1].strip()}"
                    chunk_meta = metadata.copy()
                    chunk_meta["header"] = section.replace("# ", "")
                    chunk_meta["chunk_id"] = hashlib.md5(chunk_content[:100].encode()).hexdigest()[:16]
                    chunks.append({
                        "content": chunk_content,
                        "metadata": chunk_meta,
                    })
    
    return chunks


def scan_textbooks() -> List[Dict[str, Any]]:
    """扫描教材"""
    all_chunks = []
    
    for textbook_name, textbook_path in TEXTBOOKS.items():
        print(f"📚 处理：{textbook_name}")
        if not textbook_path.exists():
            print(f"  ⚠️ 不存在：{textbook_path}")
            continue
        
        md_files = list(textbook_path.glob("*.md"))
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                metadata = extract_metadata(md_file, content, textbook_name)
                chunks = split_by_headers(content, metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"  ❌ {md_file.name}: {e}")
        
        print(f"  完成：{textbook_name}")
    
    print(f"\n✅ 共 {len(all_chunks)} 个文本块")
    return all_chunks


def save_chunks(chunks: List[Dict[str, Any]]):
    """保存分块数据"""
    DATA_DIR.mkdir(exist_ok=True)
    
    # 保存为 JSON
    output_path = DATA_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"💾 已保存：{output_path}")
    
    # 保存纯文本用于嵌入
    texts_path = DATA_DIR / "texts.txt"
    with open(texts_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk["content"] + "\n---END---\n")
    
    print(f"📝 文本：{texts_path}")


def main():
    print("=" * 60)
    print("护理教材数据导入 (简化版)")
    print("=" * 60)
    
    chunks = scan_textbooks()
    if chunks:
        save_chunks(chunks)
        print("\n✅ 完成!")
    else:
        print("\n❌ 无数据")


if __name__ == "__main__":
    main()
