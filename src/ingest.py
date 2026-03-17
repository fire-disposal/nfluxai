#!/usr/bin/env python3
"""
护理教材数据导入脚本
- 扫描三本教材的 MD 文件
- 按标题层级分块 (适配实际教材格式)
- 生成嵌入向量并存入 ChromaDB
- 保留完整的原文来源信息
"""

import os
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "chroma_db"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
NFLUX_ROOT = Path("/home/firedisposal/nflux")

# 仅处理三本核心教材
TEXTBOOKS = {
    "内科护理学": NFLUX_ROOT / "内科护理学",
    "外科护理学": NFLUX_ROOT / "外科护理学",
    "新编护理学基础": NFLUX_ROOT / "新编护理学基础",
}

# Markdown 标题层级 - 适配教材实际格式
# 教材使用 # 作为主标题 (章/节), ## 作为子标题
HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),      # 章/节标题，如 "# 第二章", "# 第一节"
    ("##", "header_2"),     # 小节标题，如 "## 【概述】"
    ("###", "header_3"),    # 子小节
    ("####", "header_4"),   # 更细粒度
]


def load_config() -> Dict[str, Any]:
    """加载或创建配置"""
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    default_config = {
        "embedding_model": "text2vec-base-chinese",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "rerank": True,
        "min_chunk_length": 50,  # 最小分块长度
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, allow_unicode=True)
    return default_config


def parse_filename(filepath: Path, textbook_name: str) -> Dict[str, str]:
    """
    从文件名解析章节信息
    
    格式：内科护理学_02_第二章_呼吸系统疾病病人的护理.md
    """
    filename = filepath.stem
    result = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "textbook": textbook_name,
        "chapter_num": "",
        "chapter_title": "",
        "section_title": "",
    }
    
    # 匹配模式：前缀_数字_第 X 章_章节标题
    match = re.match(r"(.+?)_(\d+)_(第 [^ 章]+ 章)_(.+)", filename)
    if match:
        result["chapter_num"] = match.group(2).zfill(2)  # 01, 02, ...
        result["chapter_title"] = match.group(3)  # 第二章
        result["section_title"] = match.group(4).replace("_", " ")
    else:
        # 简单文件名
        result["section_title"] = filename
    
    return result


def extract_section_headers(content: str) -> List[Tuple[str, str, int]]:
    """
    提取文档中的所有标题及其位置
    
    Returns:
        [(标题文本，标题级别，起始行号), ...]
    """
    headers = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#### '):
            headers.append((line[5:].strip(), 'header_4', i))
        elif line.startswith('### '):
            headers.append((line[4:].strip(), 'header_3', i))
        elif line.startswith('## '):
            headers.append((line[3:].strip(), 'header_2', i))
        elif line.startswith('# '):
            # 跳过纯标题如 "# 第一章"，保留有意义的如 "# 第一节 概述"
            title = line[2:].strip()
            if title and not re.match(r'^第 [一二三四五六七八九十]+章$', title):
                headers.append((title, 'header_1', i))
    
    return headers


def split_by_sections(content: str, base_metadata: Dict[str, Any]) -> List[Document]:
    """
    按章节/小节分割文档，保留完整的层级结构
    
    策略:
    1. 按一级标题 (#) 分割成大的章节
    2. 每个章节内按二级标题 (##) 分割成小节
    3. 过大的小节进一步按固定大小分块
    """
    documents = []
    
    # 移除 Frontmatter
    content_clean = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
    lines = content_clean.split('\n')
    
    # 找到所有一级标题位置
    section_starts = []
    for i, line in enumerate(lines):
        if re.match(r'^#\s+.+', line.strip()):
            section_starts.append(i)
    
    # 如果没有一级标题，整个文档作为一个章节
    if not section_starts:
        section_starts = [0]
    
    # 分割章节
    for idx, start in enumerate(section_starts):
        end = section_starts[idx + 1] if idx + 1 < len(section_starts) else len(lines)
        section_lines = lines[start:end]
        
        # 提取章节标题
        section_title = ""
        section_content_lines = []
        for line in section_lines:
            if re.match(r'^#\s+.+', line.strip()):
                section_title = re.sub(r'^#\s+', '', line.strip())
            else:
                section_content_lines.append(line)
        
        section_content = '\n'.join(section_content_lines).strip()
        if len(section_content) < 50:  # 跳过太短的章节
            continue
        
        # 在章节内按二级标题分割
        subsections = split_by_subheaders(section_content, section_title)
        
        for sub_title, sub_content in subsections:
            if len(sub_content.strip()) < 50:
                continue
            
            # 构建元数据
            meta = base_metadata.copy()
            meta["section_header"] = section_title
            meta["subsection_header"] = sub_title
            meta["title"] = f"{section_title} - {sub_title}" if sub_title else section_title
            
            # 生成唯一 ID
            chunk_id = hashlib.md5(
                f"{base_metadata['filepath']}:{sub_title}:{sub_content[:100]}".encode()
            ).hexdigest()[:16]
            meta["chunk_id"] = chunk_id
            
            # 记录行号范围 (用于原文定位)
            meta["line_start"] = start
            meta["line_end"] = end
            
            documents.append(Document(
                page_content=sub_content,
                metadata=meta
            ))
    
    return documents


def split_by_subheaders(content: str, parent_title: str) -> List[Tuple[str, str]]:
    """
    按二级标题分割内容
    
    Returns:
        [(小节标题，小节内容), ...]
    """
    subsections = []
    lines = content.split('\n')
    
    # 找到所有二级标题
    sub_starts = []
    for i, line in enumerate(lines):
        if re.match(r'^##\s+.+', line.strip()):
            sub_starts.append(i)
    
    if not sub_starts:
        # 没有二级标题，整个内容作为一个块
        return [(parent_title, content)]
    
    # 分割小节
    for idx, start in enumerate(sub_starts):
        end = sub_starts[idx + 1] if idx + 1 < len(sub_starts) else len(lines)
        sub_lines = lines[start:end]
        
        # 提取小节标题
        sub_title = ""
        sub_content_lines = []
        for line in sub_lines:
            if re.match(r'^##\s+.+', line.strip()):
                sub_title = re.sub(r'^##\s+', '', line.strip())
            else:
                sub_content_lines.append(line)
        
        sub_content = '\n'.join(sub_content_lines).strip()
        if sub_content:
            subsections.append((sub_title, sub_content))
    
    return subsections if subsections else [(parent_title, content)]


def extract_metadata(filepath: Path, content: str, textbook_name: str) -> Dict[str, Any]:
    """从文件和 Frontmatter 提取元数据"""
    # 基础元数据
    metadata = parse_filename(filepath, textbook_name)
    
    # 解析 Frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if frontmatter_match:
        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))
            for k, v in frontmatter.items():
                if v and isinstance(v, (str, int, float, bool)):
                    metadata[f"fm_{k}"] = v
        except:
            pass
    
    # 从内容提取主标题
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        metadata["main_title"] = title_match.group(1)
    
    return metadata


def scan_textbooks() -> List[Document]:
    """扫描所有教材文件并分块"""
    all_documents = []
    file_stats = []
    
    for textbook_name, textbook_path in TEXTBOOKS.items():
        print(f"\n📚 处理教材：{textbook_name}")
        if not textbook_path.exists():
            print(f"  ⚠️ 路径不存在：{textbook_path}")
            continue
        
        md_files = sorted(textbook_path.glob("*.md"))
        print(f"  找到 {len(md_files)} 个 MD 文件")
        
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                metadata = extract_metadata(md_file, content, textbook_name)
                chunks = split_by_sections(content, metadata)
                all_documents.extend(chunks)
                
                file_stats.append({
                    "file": md_file.name,
                    "chunks": len(chunks),
                    "textbook": textbook_name,
                })
                
                if len(chunks) > 0:
                    print(f"    ✓ {md_file.name} -> {len(chunks)} 块")
                
            except Exception as e:
                print(f"  ❌ 处理失败 {md_file.name}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n✅ 共生成 {len(all_documents)} 个文本块")
    
    # 打印文件统计
    print("\n📊 文件统计:")
    for stat in file_stats:
        if stat["chunks"] > 0:
            print(f"  {stat['textbook']}/{stat['file']}: {stat['chunks']} 块")
    
    return all_documents


def create_vectorstore(documents: List[Document], config: Dict[str, Any]):
    """创建向量数据库"""
    print("\n🔄 加载嵌入模型...")
    model_name = config.get("embedding_model", "text2vec-base-chinese")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print("📦 创建 ChromaDB 向量存储...")
    
    # 删除旧数据
    if DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DATA_DIR),
        collection_name="nursing_textbooks",
    )
    
    print(f"💾 已保存到：{DATA_DIR}")
    return vectorstore


def create_index_file(documents: List[Document]):
    """
    创建索引文件，用于快速查找原文来源
    保存为 JSON 格式，包含所有分块的元数据
    """
    INDEX_DIR.mkdir(exist_ok=True)
    index_path = INDEX_DIR / "chunks_index.json"
    
    index_data = []
    for doc in documents:
        index_data.append({
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "textbook": doc.metadata.get("textbook", ""),
            "filename": doc.metadata.get("filename", ""),
            "filepath": doc.metadata.get("filepath", ""),
            "chapter_num": doc.metadata.get("chapter_num", ""),
            "chapter_title": doc.metadata.get("chapter_title", ""),
            "section_header": doc.metadata.get("section_header", ""),
            "subsection_header": doc.metadata.get("subsection_header", ""),
            "title": doc.metadata.get("title", ""),
            "line_start": doc.metadata.get("line_start", 0),
            "line_end": doc.metadata.get("line_end", 0),
            "content_preview": doc.page_content[:200].replace('\n', ' '),
        })
    
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    print(f"📑 索引已保存到：{index_path}")
    return index_path


def main():
    print("=" * 60)
    print("护理教材 AI 系统 - 数据导入")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    print(f"配置：嵌入模型={config.get('embedding_model')}")
    print(f"      最小分块长度={config.get('min_chunk_length', 50)}\n")
    
    # 扫描并分块
    documents = scan_textbooks()
    
    if not documents:
        print("❌ 未找到任何文档")
        return
    
    # 创建向量库
    create_vectorstore(documents, config)
    
    # 创建索引文件
    create_index_file(documents)
    
    print("\n" + "=" * 60)
    print("✅ 数据导入完成!")
    print("=" * 60)
    
    # 打印统计
    textbook_counts = {}
    for doc in documents:
        tb = doc.metadata.get("textbook", "未知")
        textbook_counts[tb] = textbook_counts.get(tb, 0) + 1
    
    print("\n📊 教材统计:")
    for tb, count in sorted(textbook_counts.items()):
        print(f"  {tb}: {count} 块")


if __name__ == "__main__":
    main()
