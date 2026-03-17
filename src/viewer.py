#!/usr/bin/env python3
"""
护理教材索引查看工具
- 列出所有分块的来源信息
- 支持搜索和过滤
- 可查看原文内容
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).parent.parent
INDEX_FILE = PROJECT_ROOT / "data" / "index" / "chunks_index.json"
NFLUX_ROOT = Path("/home/firedisposal/nflux")


def load_index() -> List[Dict]:
    """加载索引文件"""
    if not INDEX_FILE.exists():
        print(f"❌ 索引文件不存在：{INDEX_FILE}")
        print("请先运行：uv run python src/ingest.py")
        sys.exit(1)
    
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def list_by_textbook(index_data: List[Dict], textbook: Optional[str] = None):
    """按教材列出分块统计"""
    stats = {}
    
    for chunk in index_data:
        tb = chunk.get("textbook", "未知")
        if textbook and tb != textbook:
            continue
        
        if tb not in stats:
            stats[tb] = {"count": 0, "files": {}}
        
        stats[tb]["count"] += 1
        filename = chunk.get("filename", "未知")
        if filename not in stats[tb]["files"]:
            stats[tb]["files"][filename] = 0
        stats[tb]["files"][filename] += 1
    
    print("\n" + "=" * 60)
    print("📚 教材分块统计")
    print("=" * 60)
    
    for tb, data in sorted(stats.items()):
        print(f"\n{tb}: 共 {data['count']} 块")
        print("-" * 40)
        for filename, count in sorted(data["files"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {filename}: {count} 块")
        if len(data["files"]) > 10:
            print(f"  ... 还有 {len(data['files']) - 10} 个文件")


def search_sources(index_data: List[Dict], keyword: str, limit: int = 20):
    """搜索来源"""
    results = []
    
    for chunk in index_data:
        # 在标题和章节中搜索
        search_text = " ".join([
            chunk.get("title", ""),
            chunk.get("section_header", ""),
            chunk.get("subsection_header", ""),
            chunk.get("textbook", ""),
        ]).lower()
        
        if keyword.lower() in search_text:
            results.append(chunk)
        
        if len(results) >= limit:
            break
    
    return results


def show_source_detail(chunk: Dict, show_content: bool = False):
    """显示单个来源的详细信息"""
    print("\n" + "=" * 60)
    print(f"📑 来源：{chunk.get('title', 'N/A')}")
    print("=" * 60)
    
    print(f"教材：{chunk.get('textbook', 'N/A')}")
    print(f"文件：{chunk.get('filename', 'N/A')}")
    print(f"章节：{chunk.get('chapter_title', 'N/A')}")
    print(f"小节：{chunk.get('section_header', 'N/A')} / {chunk.get('subsection_header', 'N/A')}")
    print(f"行号：{chunk.get('line_start', 0)} - {chunk.get('line_end', 0)}")
    print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
    
    if show_content:
        print("\n📄 内容预览:")
        print("-" * 40)
        content = chunk.get("content_preview", "")
        print(content[:500] if content else "无内容预览")


def read_original_content(chunk: Dict) -> Optional[str]:
    """读取原文内容"""
    filepath = Path(chunk.get("filepath", ""))
    
    if not filepath.exists():
        return None
    
    line_start = chunk.get("line_start", 0)
    line_end = chunk.get("line_end", 0)
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        content_lines = lines[line_start:line_end]
        return "".join(content_lines)
    except Exception as e:
        print(f"读取失败：{e}")
        return None


def interactive_mode():
    """交互模式"""
    index_data = load_index()
    print(f"✅ 已加载 {len(index_data)} 个分块索引")
    
    while True:
        print("\n" + "=" * 60)
        print("命令：[list] 列表 [search] 搜索 [detail] 详情 [quit] 退出")
        print("=" * 60)
        
        cmd = input("\n请输入命令：").strip().lower()
        
        if cmd == "quit" or cmd == "q":
            break
        
        elif cmd == "list" or cmd == "l":
            textbook = input("教材名称 (留空显示全部): ").strip()
            list_by_textbook(index_data, textbook if textbook else None)
        
        elif cmd == "search" or cmd == "s":
            keyword = input("搜索关键词：").strip()
            if not keyword:
                print("❌ 请输入关键词")
                continue
            
            results = search_sources(index_data, keyword)
            print(f"\n找到 {len(results)} 个结果:")
            
            for i, chunk in enumerate(results, 1):
                print(f"  [{i}] {chunk.get('textbook')} / {chunk.get('chapter_title')} / {chunk.get('section_header', 'N/A')}")
            
            # 选择查看详情
            if results:
                select = input("\n查看详情 (输入编号，留空跳过): ").strip()
                if select.isdigit() and 1 <= int(select) <= len(results):
                    show_source_detail(results[int(select) - 1])
                    
                    # 询问是否读取原文
                    read = input("\n读取原文内容？(y/n): ").strip().lower()
                    if read == "y":
                        content = read_original_content(results[int(select) - 1])
                        if content:
                            print("\n📄 原文内容:")
                            print("-" * 40)
                            print(content[:1000] + "..." if len(content) > 1000 else content)
        
        elif cmd == "detail" or cmd == "d":
            chunk_id = input("输入 Chunk ID: ").strip()
            for chunk in index_data:
                if chunk.get("chunk_id") == chunk_id:
                    show_source_detail(chunk, show_content=True)
                    break
            else:
                print("❌ 未找到该 Chunk ID")
        
        else:
            print("❌ 未知命令")


def main():
    print("=" * 60)
    print("护理教材索引查看工具")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # 命令行模式
        cmd = sys.argv[1]
        index_data = load_index()
        
        if cmd == "--list" or cmd == "-l":
            textbook = sys.argv[2] if len(sys.argv) > 2 else None
            list_by_textbook(index_data, textbook)
        
        elif cmd == "--search" or cmd == "-s":
            keyword = sys.argv[2] if len(sys.argv) > 2 else ""
            if not keyword:
                print("❌ 请提供搜索关键词")
                return
            results = search_sources(index_data, keyword)
            for chunk in results:
                show_source_detail(chunk)
        
        elif cmd == "--help" or cmd == "-h":
            print("""
用法：uv run python src/viewer.py [命令] [参数]

命令:
  -l, --list [教材]     列出分块统计
  -s, --search <关键词>  搜索来源
  -i, --interactive     交互模式
  -h, --help            显示帮助

示例:
  uv run python src/viewer.py -l 内科护理学
  uv run python src/viewer.py -s 呼吸系统
  uv run python src/viewer.py -i
""")
        else:
            interactive_mode()
    else:
        # 默认交互模式
        interactive_mode()


if __name__ == "__main__":
    main()
