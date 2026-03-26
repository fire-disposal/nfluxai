#!/usr/bin/env python3
"""
护理教材数据导入
- 基于语义的智能分块
- 多维度索引构建
- 护理程序完整性保持
- 医学词典支持
- 国内镜像加速
"""

import os
import re
import json
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import yaml
import requests

from medical_terms import (
    find_diseases_in_text,
    find_diagnoses_in_text,
    find_symptoms_in_text,
    NURSING_PROCEDURES,
    NURSING_PROCESS_MARKERS,
)





def ensure_python_compatibility():
    """检查当前 Python 版本是否与依赖兼容。"""
    version = sys.version_info
    if version >= (3, 14):
        current = f"{version.major}.{version.minor}.{version.micro}"
        raise RuntimeError(
            "当前 Python 版本与 Chroma 生态依赖不兼容。\n"
            f"检测到: Python {current}\n"
            "请使用 Python 3.10-3.13（推荐 3.12）后重试：\n"
            "  uv python install 3.12\n"
            "  uv sync --python 3.12\n"
            "  uv run --python 3.12 python main.py --run"
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
        self.target_chunk_size = self.config.get("chunk_size", 500)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
    
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
            section_chunks = self._split_large_section(
                section, lines, file_info, filepath.name
            )
            for chunk in section_chunks:
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

        # 文档无标题时，退化为全文分块
        if not sections and lines:
            sections.append({
                "level": 1,
                "title": "正文",
                "line_start": 0,
                "line_end": len(lines),
                "chunk_type": ChunkType.GENERAL,
            })
        
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

    def _create_chunk_from_lines(
        self,
        section: Dict,
        lines: List[str],
        file_info: Dict[str, str],
        filename: str,
        line_start: int,
        line_end: int,
        title_suffix: str = "",
    ) -> Optional[SemanticChunk]:
        """基于指定行范围创建 chunk。"""
        local_section = {
            **section,
            "line_start": line_start,
            "line_end": line_end,
            "title": section["title"] + title_suffix,
        }
        return self._create_semantic_chunk(local_section, lines, file_info, filename)

    def _split_large_section(
        self,
        section: Dict,
        lines: List[str],
        file_info: Dict[str, str],
        filename: str,
    ) -> List[SemanticChunk]:
        """将过大段落按段落边界切分，避免切片偏小或语义割裂。"""
        content_lines = lines[section["line_start"]:section["line_end"]]
        content = "\n".join(content_lines).strip()
        if len(content) <= self.max_chunk_size:
            chunk = self._create_semantic_chunk(section, lines, file_info, filename)
            return [chunk] if chunk else []

        paragraphs: List[Tuple[int, str]] = []
        for offset, raw_line in enumerate(content_lines):
            text = raw_line.strip()
            if not text:
                continue
            paragraphs.append((section["line_start"] + offset, text))

        if not paragraphs:
            return []

        chunks: List[SemanticChunk] = []
        current_group: List[Tuple[int, str]] = []
        current_size = 0
        piece_index = 1

        def flush_group(group: List[Tuple[int, str]], idx: int):
            if not group:
                return
            start = group[0][0]
            end = group[-1][0] + 1
            suffix = f"（片段{idx}）"
            chunk = self._create_chunk_from_lines(
                section, lines, file_info, filename, start, end, suffix
            )
            if chunk:
                chunks.append(chunk)

        for line_no, paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if current_group and current_size + paragraph_len > self.target_chunk_size:
                flush_group(current_group, piece_index)
                piece_index += 1

                overlap_group: List[Tuple[int, str]] = []
                overlap_chars = 0
                for item in reversed(current_group):
                    overlap_group.insert(0, item)
                    overlap_chars += len(item[1])
                    if overlap_chars >= self.chunk_overlap:
                        break

                current_group = overlap_group
                current_size = sum(len(item[1]) for item in current_group)

            current_group.append((line_no, paragraph))
            current_size += paragraph_len

        flush_group(current_group, piece_index)
        return chunks


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
    ensure_python_compatibility()
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
    except Exception as exc:
        raise RuntimeError("向量库依赖加载失败，请确认 Python 与依赖版本兼容。") from exc

    print("\n🔄 加载嵌入模型（Silicon Flow 远程嵌入）...")
    model_name = config.get("embedding_model", "BAAI/bge-large-zh-v1.5")
    device = config.get("embedding_device", "cpu")
    use_remote = config.get("use_remote_embeddings", True)
    use_silicon = config.get("use_silicon_flow", True)
    silicon_api = config.get("silicon_flow_api_url") or os.getenv("SILICON_FLOW_API_URL")
    silicon_key = config.get("silicon_flow_api_key") or os.getenv("SILICON_FLOW_API_KEY")
    silicon_model = config.get("silicon_flow_model") or model_name

    print(f"   模型: {model_name}")
    print(f"   设备: {device}")
    print(f"   use_remote_embeddings: {use_remote} (use_silicon_flow={use_silicon})")

    class SiliconFlowEmbeddings:
        """Silicon Flow 在线嵌入适配器。

        期望 API 接受 JSON: {"model": "<model>", "input": ["text1", ...]} 并返回
        {"data": [{"embedding": [...]}, ...]} 或类似格式。
        请根据供应商文档调整 `silicon_api`。
        """
        def __init__(self, api_url: str, api_key: Optional[str], model: str):
            self.api_url = api_url
            self.api_key = api_key
            self.model = model

        def _call(self, inputs: List[str]):
            if not self.api_url:
                raise RuntimeError("Silicon Flow API 地址未配置(silicon_flow_api_url)。")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # API 有时接受单个字符串作为 input，也接受列表；匹配示例优先使用单字符串
            payload_input = inputs[0] if len(inputs) == 1 else inputs
            payload = {"model": self.model, "input": payload_input}

            try:
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            except requests.RequestException as e:
                raise RuntimeError(
                    "Silicon Flow 请求失败：无法连接到嵌入服务。请检查 `silicon_flow_api_url` 配置、网络连接和 `SILICON_FLOW_API_KEY`。"
                ) from e

            try:
                data = resp.json()
            except Exception:
                raise RuntimeError(f"Silicon Flow 返回非 JSON 响应：{resp.status_code} {resp.text}")

            # 解析常见的返回结构，兼容你提供的示例：{"data":[{"object":"embedding","embedding":[...],"index":...}],...}
            items = None
            if isinstance(data, dict):
                if "data" in data:
                    items = data["data"]
                elif "embedding" in data:
                    items = [data]
            elif isinstance(data, list):
                items = data

            if items is None:
                raise RuntimeError(f"未知的 Silicon Flow Embeddings 返回格式: {data}")

            emb = []
            for it in items:
                if isinstance(it, dict) and "embedding" in it:
                    emb.append(it["embedding"])
                elif isinstance(it, list) and all(isinstance(x, (int, float)) for x in it):
                    emb.append(it)
                else:
                    raise RuntimeError("无法解析 Silicon Flow 返回的 embedding 格式")

            return emb

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self._call(texts)

        def embed_query(self, text: str) -> List[float]:
            return self._call([text])[0]

    if use_remote and use_silicon:
        embeddings = SiliconFlowEmbeddings(api_url=silicon_api, api_key=silicon_key, model=silicon_model)
    else:
        raise RuntimeError("当前仅支持使用 Silicon Flow 远程嵌入。请在 `config.yaml` 中启用 `use_silicon_flow` 并配置 `silicon_flow_api_url` 与 API key。")
    
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
                "section_header": chunk.section,
                "subsection_header": chunk.subsection,
                "title": chunk.title,
                "source_file": chunk.source_file,
                "filename": chunk.source_file,
                "filepath": f"data/textbooks/{chunk.textbook}/{chunk.source_file}",
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "diseases": json.dumps(chunk.diseases, ensure_ascii=False),
                "symptoms": json.dumps(chunk.symptoms, ensure_ascii=False),
                "nursing_diagnoses": json.dumps(chunk.nursing_diagnoses, ensure_ascii=False),
            }
        )
        documents.append(doc)
    
    # 删除旧数据并逐批写入，避免在 CPU 上一次性计算大量嵌入导致长时间无响应
    if DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = int(config.get("embedding_batch_size", 128))
    total = len(documents)
    if total == 0:
        print("⚠️ 无文档需入库，跳过向量库创建。")
        return None

    print(f"📦 分批写入 ChromaDB（batch_size={batch_size}）... 总文档: {total}")
    vectorstore = None
    try:
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            start = i + 1
            end = min(i + batch_size, total)
            print(f"  - 正在处理文档 {start}-{end} / {total} ...")

            # 使用 from_documents 按批次写入（Chroma 会在 persist_directory 下追加）
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(DATA_DIR),
                collection_name="nursing_textbooks",
            )

        print(f"💾 已保存到：{DATA_DIR}")
        return vectorstore
    except Exception as e:
        raise RuntimeError(
            "向量库创建失败：在将文档写入 Chroma 时出错。"
            " 请检查模型是否在 CPU 上耗时较长，或使用更小的嵌入模型/启用 GPU。"
        ) from e


def save_chunks(chunks: List[SemanticChunk]):
    """保存分块数据"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    data = [asdict(chunk) for chunk in chunks]
    output_path = INDEX_DIR / "chunks_index.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 分块数据已保存：{output_path}")


def save_index(index_data: Dict):
    """保存索引数据"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INDEX_DIR / "index.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    print(f"📑 索引数据已保存：{output_path}")


def main():
    """主函数"""
    ensure_python_compatibility()

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
