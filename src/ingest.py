#!/usr/bin/env python3
"""
护理教材数据导入
- 基于语义的智能分块
- 多维度索引构建
- 护理程序完整性保持
- 医学词典支持
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import yaml
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from medical_terms import (
    find_diseases_in_text,
    find_diagnoses_in_text,
    find_symptoms_in_text,
    NURSING_PROCEDURES,
    NURSING_PROCESS_MARKERS,
)


# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "chroma_db"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
TEXTBOOKS_ROOT = PROJECT_ROOT / "data" / "textbooks"

TEXTBOOKS = {
    "内科护理学": TEXTBOOKS_ROOT / "内科护理学",
    "外科护理学": TEXTBOOKS_ROOT / "外科护理学",
    "新编护理学基础": TEXTBOOKS_ROOT / "新编护理学基础",
}


class ChunkType(Enum):
    """分块类型"""
    CHAPTER_OVERVIEW = "chapter_overview"      # 章节概述
    DISEASE_INFO = "disease_info"              # 疾病信息
    SYMPTOM_CARE = "symptom_care"              # 症状护理
    NURSING_ASSESSMENT = "nursing_assessment"  # 护理评估
    NURSING_DIAGNOSIS = "nursing_diagnosis"    # 护理诊断
    NURSING_INTERVENTION = "nursing_intervention"  # 护理措施
    NURSING_EVALUATION = "nursing_evaluation"  # 护理评价
    HEALTH_EDUCATION = "health_education"      # 健康指导
    THEORY = "theory"                          # 理论知识
    GENERAL = "general"                        # 一般内容


@dataclass
class SemanticChunk:
    """语义分块"""
    chunk_id: str
    content: str
    chunk_type: str
    
    # 层级路径
    textbook: str
    chapter_num: str
    chapter_title: str
    section: str
    subsection: str
    
    # 标题
    title: str
    
    # 位置信息
    source_file: str
    line_start: int
    line_end: int
    
    # 索引标签
    diseases: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    nursing_diagnoses: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    
    # 内容预览
    content_preview: str = ""


class SemanticChunker:
    """语义分块器"""
    
    # 护理程序标记
    SECTION_MARKERS = {
        ChunkType.NURSING_ASSESSMENT: [
            "【护理评估】", "护理评估", "健康史", "身体状况", 
            "心理-社会状况", "实验室及其他检查"
        ],
        ChunkType.NURSING_DIAGNOSIS: [
            "【护理诊断", "【常用护理诊断", "【护理问题",
            "护理诊断", "护理问题"
        ],
        ChunkType.NURSING_INTERVENTION: [
            "【护理措施", "【治疗要点", "护理措施", "治疗要点",
            "【护理要点"
        ],
        ChunkType.NURSING_EVALUATION: [
            "【护理评价】", "护理评价", "【评价】"
        ],
        ChunkType.HEALTH_EDUCATION: [
            "【健康指导】", "【健康教育】", "健康指导", "健康教育"
        ],
    }
    
    # 疾病信息标记
    DISEASE_MARKERS = [
        "【病因与发病机制】", "【临床表现】", "【诊断要点】",
        "【治疗要点】", "【预后】", "病因", "发病机制", "临床表现"
    ]
    
    # 症状标记
    SYMPTOM_MARKERS = [
        "咳嗽与咳痰", "呼吸困难", "咯血", "胸痛", "发热",
        "恶心与呕吐", "腹泻", "便秘", "疼痛", "水肿"
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.max_chunk_size = self.config.get("max_chunk_size", 2000)
    
    def chunk_file(self, filepath: Path, textbook: str) -> List[SemanticChunk]:
        """对单个文件进行语义分块"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.split('\n')
        chunks = []
        
        # 解析文件名
        file_info = self._parse_filename(filepath.name, textbook)
        
        # 识别文档结构
        structure = self._analyze_structure(lines)
        
        # 按结构分块
        for section in structure:
            chunk = self._create_semantic_chunk(
                section, lines, file_info, filepath.name
            )
            if chunk and len(chunk.content) >= self.min_chunk_size:
                chunks.append(chunk)
        
        return chunks
    
    def _parse_filename(self, filename: str, textbook: str) -> Dict[str, str]:
        """解析文件名"""
        stem = Path(filename).stem
        info = {
            "textbook": textbook,
            "chapter_num": "",
            "chapter_title": "",
            "filename": filename,
        }
        
        # 匹配：教材名_数字_第X章_标题
        match = re.match(r"(.+?)_(\d+)_(第[^章]+章)_(.+)", stem)
        if match:
            info["chapter_num"] = match.group(2).zfill(2)
            info["chapter_title"] = match.group(3) + " " + match.group(4).replace("_", " ")
        else:
            info["chapter_title"] = stem.replace("_", " ")
        
        return info
    
    def _analyze_structure(self, lines: List[str]) -> List[Dict]:
        """分析文档结构，识别语义单元"""
        sections = []
        current = None
        
        for i, line in enumerate(lines):
            # 检测标题
            header_match = re.match(r'^(#{1,4})\s+(.+)$', line)
            
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # 保存当前段落
                if current:
                    current["line_end"] = i
                    sections.append(current)
                
                # 开始新段落
                current = {
                    "level": level,
                    "title": title,
                    "line_start": i,
                    "line_end": len(lines),
                    "chunk_type": self._classify_section(title),
                }
        
        # 保存最后一个段落
        if current:
            current["line_end"] = len(lines)
            sections.append(current)
        
        return sections
    
    def _classify_section(self, title: str) -> ChunkType:
        """分类段落类型"""
        # 检查护理程序标记
        for chunk_type, markers in self.SECTION_MARKERS.items():
            for marker in markers:
                if marker in title:
                    return chunk_type
        
        # 检查疾病标记
        for marker in self.DISEASE_MARKERS:
            if marker in title:
                return ChunkType.DISEASE_INFO
        
        # 检查症状标记
        for marker in self.SYMPTOM_MARKERS:
            if marker in title:
                return ChunkType.SYMPTOM_CARE
        
        return ChunkType.GENERAL
    
    def _create_semantic_chunk(
        self, 
        section: Dict, 
        lines: List[str],
        file_info: Dict,
        filename: str
    ) -> Optional[SemanticChunk]:
        """创建语义分块"""
        content_lines = lines[section["line_start"]:section["line_end"]]
        content = '\n'.join(content_lines).strip()
        
        if len(content) < self.min_chunk_size:
            return None
        
        # 生成 ID
        chunk_id = hashlib.md5(
            f"{filename}:{section['title']}:{content[:100]}".encode()
        ).hexdigest()[:16]
        
        # 提取标签
        diseases = self._extract_diseases(section["title"], content)
        symptoms = self._extract_symptoms(section["title"], content)
        nursing_diagnoses = self._extract_nursing_diagnoses(content)
        procedures = self._extract_procedures(content)
        
        return SemanticChunk(
            chunk_id=chunk_id,
            content=content,
            chunk_type=section["chunk_type"].value,
            textbook=file_info["textbook"],
            chapter_num=file_info["chapter_num"],
            chapter_title=file_info["chapter_title"],
            section=section["title"],
            subsection="",
            title=section["title"],
            source_file=filename,
            line_start=section["line_start"],
            line_end=section["line_end"],
            diseases=diseases,
            symptoms=symptoms,
            nursing_diagnoses=nursing_diagnoses,
            procedures=procedures,
            content_preview=content[:200].replace('\n', ' '),
        )
    
    def _extract_diseases(self, title: str, content: str) -> List[str]:
        """提取疾病名称 - 使用医学词典"""
        text = title + " " + content
        return find_diseases_in_text(text)[:5]
    
    def _extract_symptoms(self, title: str, content: str) -> List[str]:
        """提取症状 - 使用医学词典"""
        text = title + " " + content
        return find_symptoms_in_text(text)[:5]
    
    def _extract_nursing_diagnoses(self, content: str) -> List[str]:
        """提取护理诊断 - 使用医学词典"""
        return find_diagnoses_in_text(content)[:5]
    
    def _extract_procedures(self, content: str) -> List[str]:
        """提取护理操作"""
        found = []
        for proc in NURSING_PROCEDURES:
            if proc in content:
                found.append(proc)
        return found[:5]


class NursingIndexBuilder:
    """护理教材索引构建器"""
    
    def __init__(self):
        self.disease_index: Dict[str, List[str]] = {}
        self.symptom_index: Dict[str, List[str]] = {}
        self.diagnosis_index: Dict[str, List[str]] = {}
        self.procedure_index: Dict[str, List[str]] = {}
        self.chapter_index: Dict[str, Dict] = {}
    
    def build(self, chunks: List[SemanticChunk]) -> Dict:
        """构建多维度索引"""
        for chunk in chunks:
            # 疾病索引
            for disease in chunk.diseases:
                if disease not in self.disease_index:
                    self.disease_index[disease] = []
                self.disease_index[disease].append(chunk.chunk_id)
            
            # 症状索引
            for symptom in chunk.symptoms:
                if symptom not in self.symptom_index:
                    self.symptom_index[symptom] = []
                self.symptom_index[symptom].append(chunk.chunk_id)
            
            # 护理诊断索引
            for diagnosis in chunk.nursing_diagnoses:
                if diagnosis not in self.diagnosis_index:
                    self.diagnosis_index[diagnosis] = []
                self.diagnosis_index[diagnosis].append(chunk.chunk_id)
            
            # 操作索引
            for procedure in chunk.procedures:
                if procedure not in self.procedure_index:
                    self.procedure_index[procedure] = []
                self.procedure_index[procedure].append(chunk.chunk_id)
            
            # 章节索引
            textbook = chunk.textbook
            if textbook not in self.chapter_index:
                self.chapter_index[textbook] = {}
            
            chapter = chunk.chapter_title
            if chapter not in self.chapter_index[textbook]:
                self.chapter_index[textbook][chapter] = []
            self.chapter_index[textbook][chapter].append(chunk.chunk_id)
        
        return {
            "disease_index": self.disease_index,
            "symptom_index": self.symptom_index,
            "diagnosis_index": self.diagnosis_index,
            "procedure_index": self.procedure_index,
            "chapter_index": self.chapter_index,
            "stats": {
                "total_chunks": len(chunks),
                "diseases": len(self.disease_index),
                "symptoms": len(self.symptom_index),
                "diagnoses": len(self.diagnosis_index),
                "procedures": len(self.procedure_index),
            }
        }


def create_vectorstore(chunks: List[SemanticChunk], config: Dict[str, Any]):
    """创建向量数据库"""
    print("\n🔄 加载嵌入模型...")
    model_name = config.get("embedding_model", "text2vec-base-chinese")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print("📦 创建 ChromaDB 向量存储...")
    
    # 转换为 LangChain Document
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "textbook": chunk.textbook,
                "chapter_num": chunk.chapter_num,
                "chapter_title": chunk.chapter_title,
                "section": chunk.section,
                "title": chunk.title,
                "source_file": chunk.source_file,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "diseases": json.dumps(chunk.diseases, ensure_ascii=False),
                "symptoms": json.dumps(chunk.symptoms, ensure_ascii=False),
                "nursing_diagnoses": json.dumps(chunk.nursing_diagnoses, ensure_ascii=False),
            }
        )
        documents.append(doc)
    
    # 删除旧数据
    if DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DATA_DIR),
        collection_name="nursing_textbooks_v2",
    )
    
    print(f"💾 已保存到：{DATA_DIR}")
    return vectorstore


def save_chunks(chunks: List[SemanticChunk]):
    """保存分块数据"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    data = [asdict(chunk) for chunk in chunks]
    output_path = INDEX_DIR / "chunks_v2.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 分块数据已保存：{output_path}")


def save_index(index_data: Dict):
    """保存索引数据"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INDEX_DIR / "index_v2.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    print(f"📑 索引数据已保存：{output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("护理教材数据导入 V2 - 语义分块")
    print("=" * 60)
    
    # 加载配置
    config_path = PROJECT_ROOT / "config.yaml"
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    
    # 初始化分块器
    chunker = SemanticChunker(config)
    
    # 处理每本教材
    all_chunks = []
    for textbook_name, textbook_path in TEXTBOOKS.items():
        print(f"\n📚 处理教材：{textbook_name}")
        
        if not textbook_path.exists():
            print(f"  ⚠️ 路径不存在：{textbook_path}")
            continue
        
        md_files = sorted(textbook_path.glob("*.md"))
        print(f"  找到 {len(md_files)} 个文件")
        
        for md_file in md_files:
            chunks = chunker.chunk_file(md_file, textbook_name)
            all_chunks.extend(chunks)
            if chunks:
                print(f"    ✓ {md_file.name} -> {len(chunks)} 块")
    
    print(f"\n✅ 共生成 {len(all_chunks)} 个语义分块")
    
    # 构建索引
    print("\n🔄 构建多维度索引...")
    index_builder = NursingIndexBuilder()
    index_data = index_builder.build(all_chunks)
    
    # 创建向量库
    create_vectorstore(all_chunks, config)
    
    # 保存数据
    save_chunks(all_chunks)
    save_index(index_data)
    
    # 打印统计
    print("\n" + "=" * 60)
    print("📊 索引统计")
    print("=" * 60)
    stats = index_data["stats"]
    print(f"总分块数：{stats['total_chunks']}")
    print(f"疾病索引：{stats['diseases']} 条")
    print(f"症状索引：{stats['symptoms']} 条")
    print(f"护理诊断索引：{stats['diagnoses']} 条")
    print(f"护理操作索引：{stats['procedures']} 条")
    
    # 分块类型统计
    type_counts = {}
    for chunk in all_chunks:
        t = chunk.chunk_type
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\n📋 分块类型分布：")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    
    # 示例
    print("\n📋 疾病索引示例（前5条）：")
    for disease, chunk_ids in list(index_data["disease_index"].items())[:5]:
        print(f"  {disease}: {len(chunk_ids)} 个分块")
    
    print("\n📋 护理诊断索引示例（前5条）：")
    for diagnosis, chunk_ids in list(index_data["diagnosis_index"].items())[:5]:
        print(f"  {diagnosis}: {len(chunk_ids)} 个分块")
    
    print("\n" + "=" * 60)
    print("✅ 数据导入完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()