#!/usr/bin/env python3
"""
系统功能测试脚本
验证各模块的正确性
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_medical_terms():
    """测试医学词典模块"""
    print("=" * 60)
    print("测试 1: 医学词典模块")
    print("=" * 60)
    
    from medical_terms import (
        find_diseases_in_text,
        find_diagnoses_in_text,
        find_symptoms_in_text,
        get_all_diseases,
        get_all_diagnoses,
    )
    
    # 测试疾病识别
    test_queries = [
        "肺炎病人的护理措施",
        "糖尿病酮症酸中毒的处理",
        "高血压病人的健康教育",
        "清理呼吸道无效的护理",
    ]
    
    print("\n疾病识别测试:")
    for query in test_queries:
        diseases = find_diseases_in_text(query)
        print(f"  '{query}' -> {diseases}")
    
    # 测试护理诊断识别
    print("\n护理诊断识别测试:")
    diagnosis_queries = [
        "清理呼吸道无效的护理措施",
        "气体交换受损怎么办",
        "焦虑病人的护理",
        "营养失调低于机体需要量",
    ]
    for query in diagnosis_queries:
        diagnoses = find_diagnoses_in_text(query)
        print(f"  '{query}' -> {diagnoses}")
    
    # 统计
    all_diseases = get_all_diseases()
    all_diagnoses = get_all_diagnoses()
    print(f"\n词典统计:")
    print(f"  疾病名称: {len(all_diseases)} 个")
    print(f"  护理诊断: {len(all_diagnoses)} 个")
    
    return True


def test_textbook_files():
    """测试教材文件"""
    print("\n" + "=" * 60)
    print("测试 2: 教材文件")
    print("=" * 60)
    
    textbooks_dir = PROJECT_ROOT / "data" / "textbooks"
    
    if not textbooks_dir.exists():
        print(f"❌ 教材目录不存在: {textbooks_dir}")
        return False
    
    textbooks = ["内科护理学", "外科护理学", "新编护理学基础"]
    total_files = 0
    
    for textbook in textbooks:
        textbook_path = textbooks_dir / textbook
        if textbook_path.exists():
            md_files = list(textbook_path.glob("*.md"))
            total_files += len(md_files)
            print(f"  ✅ {textbook}: {len(md_files)} 个文件")
        else:
            print(f"  ❌ {textbook}: 目录不存在")
    
    print(f"\n总文件数: {total_files}")
    return total_files > 0


def test_index_files():
    """测试索引文件"""
    print("\n" + "=" * 60)
    print("测试 3: 索引文件")
    print("=" * 60)
    
    import json
    
    index_dir = PROJECT_ROOT / "data" / "index"
    
    # 检查分块文件
    chunks_file = index_dir / "chunks_index.json"
    legacy_chunks_file = index_dir / "chunks.json"
    selected_chunks_file = chunks_file if chunks_file.exists() else legacy_chunks_file

    if selected_chunks_file.exists():
        with open(selected_chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"  ✅ {selected_chunks_file.name}: {len(chunks)} 个分块")
    else:
        print(f"  ❌ chunks_index.json / chunks.json 不存在")
        return False
    
    # 检查索引文件
    index_file = index_dir / "index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        disease_count = len(index.get("disease_index", {}))
        diagnosis_count = len(index.get("diagnosis_index", {}))
        
        print(f"  ✅ index.json:")
        print(f"     - 疾病索引: {disease_count} 条")
        print(f"     - 护理诊断索引: {diagnosis_count} 条")
    else:
        print(f"  ❌ index.json 不存在")
        return False
    
    return True


def test_config():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试 4: 配置文件")
    print("=" * 60)
    
    import yaml
    
    config_file = PROJECT_ROOT / "config.yaml"
    if not config_file.exists():
        print(f"  ❌ 配置文件不存在")
        return False
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"  嵌入模型: {config.get('embedding_model')}")
    print(f"  Top-K: {config.get('top_k')}")
    print(f"  Rerank: {config.get('rerank')}")
    print(f"  LLM Provider: {config.get('llm', {}).get('provider')}")
    print(f"  LLM Model: {config.get('llm', {}).get('model')}")
    
    return True


def test_llm_module():
    """测试 LLM 模块"""
    print("\n" + "=" * 60)
    print("测试 5: LLM 模块")
    print("=" * 60)
    
    try:
        from llm import get_llm_config, generate_response
        
        config = get_llm_config()
        print(f"  Provider: {config.get('provider')}")
        print(f"  Model: {config.get('model')}")
        
        # 测试生成（不实际调用 LLM）
        print(f"  ✅ LLM 模块加载成功")
        return True
    except Exception as e:
        print(f"  ❌ LLM 模块加载失败: {e}")
        return False


def test_app_structure():
    """测试应用结构"""
    print("\n" + "=" * 60)
    print("测试 6: 应用结构")
    print("=" * 60)
    
    required_files = [
        "src/app.py",
        "src/retriever.py",
        "src/llm.py",
        "src/ingest.py",
        "src/medical_terms.py",
        "config.yaml",
        "main.py",
        "start.sh",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist


def test_user_flow():
    """测试用户交互流程（模拟）"""
    print("\n" + "=" * 60)
    print("测试 7: 用户交互流程（模拟）")
    print("=" * 60)
    
    from medical_terms import find_diseases_in_text, find_diagnoses_in_text
    
    # 模拟用户查询
    user_query = "肺炎病人的护理措施有哪些？"
    print(f"\n用户查询: {user_query}")
    
    # 1. 实体识别
    diseases = find_diseases_in_text(user_query)
    diagnoses = find_diagnoses_in_text(user_query)
    print(f"  识别疾病: {diseases}")
    print(f"  识别诊断: {diagnoses}")
    
    # 2. 模拟检索结果
    print(f"\n  模拟检索流程:")
    print(f"    1. 向量检索 -> 获取候选文档")
    print(f"    2. Rerank 重排序 -> 提高相关性")
    print(f"    3. 返回 Top-K 结果")
    
    # 3. 模拟 LLM 调用
    print(f"\n  模拟 LLM 调用:")
    print(f"    1. 构建提示词（包含上下文）")
    print(f"    2. 调用 LLM 生成回答")
    print(f"    3. 返回带引用的回答")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("护理教材 AI 系统 - 功能测试")
    print("=" * 60)
    
    results = {
        "医学词典模块": test_medical_terms(),
        "教材文件": test_textbook_files(),
        "索引文件": test_index_files(),
        "配置文件": test_config(),
        "LLM 模块": test_llm_module(),
        "应用结构": test_app_structure(),
        "用户流程": test_user_flow(),
    }
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
