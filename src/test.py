#!/usr/bin/env python3
"""
测试分块和检索功能
验证索引分块逻辑和原文来源列出
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import NursingRetriever


def test_retriever():
    """测试检索器功能"""
    print("=" * 60)
    print("🧪 测试检索器")
    print("=" * 60)
    
    retriever = NursingRetriever()
    retriever.initialize()
    
    test_queries = [
        "呼吸系统疾病病人的护理评估",
        "护理程序的基本步骤",
        "静脉输液的护理",
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询：{query}")
        print("-" * 60)
        
        # 获取来源列表
        sources = retriever.list_sources(query, top_k=3)
        
        if not sources:
            print("  ⚠️ 未找到相关来源")
            continue
        
        for src in sources:
            print(f"\n  [{src['index']}] {src['citation']}")
            print(f"      教材：{src['textbook']}")
            print(f"      文件：{src['filename']}")
            print(f"      章节：{src['chapter']}")
            print(f"      小节：{src['section']} / {src['subsection']}")
            print(f"      行号：{src['line_range']}")
            print(f"      相似度：{src['score']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


def test_source_content():
    """测试原文内容读取"""
    print("\n" + "=" * 60)
    print("🧪 测试原文内容读取")
    print("=" * 60)
    
    retriever = NursingRetriever()
    retriever.initialize()
    
    # 测试查询
    query = "呼吸系统疾病"
    sources = retriever.list_sources(query, top_k=1)
    
    if sources:
        src = sources[0]
        print(f"\n📑 来源：{src['citation']}")
        
        # 读取原文
        content = retriever.get_source_content(
            textbook=src['textbook'],
            filename=src['filename'],
            line_start=int(src['line_range'].split('-')[0]),
            line_end=int(src['line_range'].split('-')[1]),
        )
        
        if content:
            print("\n📄 原文内容预览:")
            print("-" * 40)
            print(content[:500] + "..." if len(content) > 500 else content)
        else:
            print("❌ 无法读取原文内容")


def test_metadata_structure():
    """测试元数据结构"""
    print("\n" + "=" * 60)
    print("🧪 测试元数据结构")
    print("=" * 60)
    
    retriever = NursingRetriever()
    retriever.initialize()
    
    query = "护理"
    results = retriever.search(query, top_k=1)
    
    if results:
        doc, score = results[0]
        print("\n📑 元数据字段:")
        for key, value in sorted(doc.metadata.items()):
            print(f"  {key}: {value}")


def main():
    """运行所有测试"""
    try:
        test_retriever()
        test_source_content()
        test_metadata_structure()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
