# 护理教材索引与分块设计文档

## 一、教材结构分析

### 1.1 文档层级结构

```
教材文件 (.md)
├── # 章节标题          → 如 "呼吸系统疾病病人的护理"
│   ├── # 第一节 概述
│   │   ├── # 【小节标题】    → 如 【护理评估】
│   │   │   ├── # （一）子小节
│   │   │   │   └── # 1. 更细层级
│   │   │   └── ...
│   │   └── ...
│   ├── # 第二节 常见症状
│   └── # 第三节 具体疾病
└── ...
```

### 1.2 内容类型分布

| 类型 | 标记关键词 | 说明 |
|------|-----------|------|
| 章节概述 | 概述、绪论 | 系统介绍、结构功能 |
| 疾病信息 | 病因、临床表现、诊断要点 | 具体疾病相关知识 |
| 症状护理 | 咳嗽、呼吸困难、疼痛等 | 症状的护理方法 |
| 护理评估 | 【护理评估】、健康史、身体状况 | 评估内容和方法 |
| 护理诊断 | 【护理诊断】、护理问题 | NANDA 诊断名称 |
| 护理措施 | 【护理措施】、治疗要点 | 具体护理干预 |
| 护理评价 | 【护理评价】 | 效果评价标准 |
| 健康指导 | 【健康指导】、健康教育 | 患者教育内容 |

---

## 二、分块策略设计

### 2.1 设计原则

1. **语义完整性**：按护理程序单元分块，保持内容完整性
2. **层级保留**：保留教材→章→节→小节的层级信息
3. **索引友好**：提取疾病、症状、护理诊断等标签便于检索
4. **上下文保持**：每个分块包含足够的上下文信息

### 2.2 分块规则

```python
# 分块触发条件
1. 遇到标题行（# 开头）时，结束当前分块，开始新分块
2. 分块最小长度：100 字符（过滤过短内容）
3. 分块最大长度：2000 字符（过长内容需进一步分割）

# 分块类型识别
if "【护理评估】" in title:
    chunk_type = "nursing_assessment"
elif "【护理诊断" in title:
    chunk_type = "nursing_diagnosis"
elif "【护理措施" in title:
    chunk_type = "nursing_intervention"
elif "【健康指导】" in title:
    chunk_type = "health_education"
...
```

### 2.3 分块元数据

每个分块包含以下元数据：

```json
{
  "chunk_id": "唯一标识（MD5 哈希）",
  "content": "分块内容",
  "chunk_type": "分块类型",
  
  // 层级信息
  "textbook": "教材名称",
  "chapter_num": "章节号",
  "chapter_title": "章节标题",
  "section": "节标题",
  "subsection": "小节标题",
  "title": "当前标题",
  
  // 位置信息
  "source_file": "源文件名",
  "line_start": "起始行号",
  "line_end": "结束行号",
  
  // 索引标签
  "diseases": ["疾病名称列表"],
  "symptoms": ["症状列表"],
  "nursing_diagnoses": ["护理诊断列表"],
  "procedures": ["护理操作列表"],
  
  // 预览
  "content_preview": "内容前200字符"
}
```

---

## 三、索引机制设计

### 3.1 多维度索引

```
┌─────────────────────────────────────────────────────────────┐
│                      主索引（向量索引）                        │
│  ChromaDB 向量数据库，支持语义检索                              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   疾病索引     │   │   症状索引     │   │ 护理诊断索引   │
│ disease_index │   │ symptom_index │   │diagnosis_index│
├───────────────┤   ├───────────────┤   ├───────────────┤
│ 肺炎: [id1,id2]│   │ 咳嗽: [id1,id3]│   │焦虑: [id1,id4] │
│ 哮喘: [id3,id4]│   │ 呼吸困难: [...] │   │疼痛: [id2,id5] │
│ ...           │   │ ...           │   │...            │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────┐
                    │   章节索引     │
                    │ chapter_index │
                    ├───────────────┤
                    │ 内科/第二章:  │
                    │   [id1,id2...]│
                    └───────────────┘
```

### 3.2 索引数据结构

```json
{
  "disease_index": {
    "肺炎": ["chunk_id_1", "chunk_id_2"],
    "支气管哮喘": ["chunk_id_3", "chunk_id_4"],
    ...
  },
  "symptom_index": {
    "咳嗽": ["chunk_id_1", "chunk_id_5"],
    "呼吸困难": ["chunk_id_2", "chunk_id_6"],
    ...
  },
  "diagnosis_index": {
    "清理呼吸道无效": ["chunk_id_1", "chunk_id_2"],
    "气体交换受损": ["chunk_id_3", "chunk_id_4"],
    "焦虑": ["chunk_id_5", "chunk_id_6"],
    ...
  },
  "procedure_index": {
    "吸氧": ["chunk_id_1", "chunk_id_2"],
    "导尿": ["chunk_id_3"],
    ...
  },
  "chapter_index": {
    "内科护理学": {
      "第二章 呼吸系统疾病病人的护理": ["chunk_id_1", ...],
      ...
    },
    ...
  },
  "stats": {
    "total_chunks": 4916,
    "diseases": 150,
    "symptoms": 25,
    "diagnoses": 50,
    "procedures": 30
  }
}
```

---

## 四、检索策略

### 4.1 混合检索流程

```
用户查询
    │
    ├──→ 1. 关键词提取
    │       ├── 疾病名称识别
    │       ├── 症状识别
    │       └── 护理诊断识别
    │
    ├──→ 2. 索引预筛选（可选）
    │       └── 根据关键词从对应索引获取候选 chunk_id
    │
    ├──→ 3. 向量检索
    │       ├── 全量检索（无预筛选）
    │       └── 过滤检索（有预筛选）
    │
    ├──→ 4. Rerank 重排序
    │       └── bge-reranker 提高相关性
    │
    └──→ 5. 结果返回
            ├── 分块内容
            ├── 来源信息
            └── 相关标签
```

### 4.2 查询示例

```python
# 查询：肺炎病人的护理措施

# 1. 关键词提取
keywords = {
    "diseases": ["肺炎"],
    "content_types": ["nursing_intervention"]
}

# 2. 索引预筛选
candidate_ids = disease_index.get("肺炎", [])
candidate_ids += [id for id, chunk in chunks.items() 
                  if chunk.chunk_type == "nursing_intervention"]

# 3. 向量检索
results = vectorstore.similarity_search_with_score(
    query="肺炎病人的护理措施",
    k=20,
    filter={"chunk_id": {"$in": candidate_ids}}  # 可选过滤
)

# 4. Rerank
reranked = reranker.rank(query, results)

# 5. 返回 Top-K
return reranked[:5]
```

---

## 五、文件结构

```
data/
├── textbooks/              # 教材源文件（已清理图片链接）
│   ├── 内科护理学/
│   ├── 外科护理学/
│   └── 新编护理学基础/
│
├── chroma_db/              # 向量数据库
│
└── index/
    ├── chunks_v2.json      # 分块数据
    └── index_v2.json       # 多维度索引
```

---

## 六、后续优化方向

### 6.1 短期优化

1. **疾病名称识别优化**
   - 使用医学实体识别模型（如 CMeKG）
   - 建立疾病名称词典

2. **护理诊断标准化**
   - 使用 NANDA-I 标准诊断名称
   - 建立诊断-措施关联

### 6.2 中期优化

1. **知识图谱构建**
   - 疾病-症状-诊断-措施 关系图谱
   - 支持推理式检索

2. **查询理解增强**
   - 意图识别（查询疾病信息/护理措施/健康指导）
   - 查询改写和扩展

### 6.3 长期优化

1. **多模态支持**
   - 图片、表格内容提取
   - 流程图解析

2. **个性化检索**
   - 用户画像
   - 学习路径推荐

---

## 七、使用说明

### 7.1 数据导入

```bash
# 运行 V2 版本导入脚本
uv run python src/ingest_v2.py
```

### 7.2 检索使用

```python
from retriever_v2 import NursingRetrieverV2

retriever = NursingRetrieverV2()
retriever.initialize()

# 基础检索
results = retriever.search("肺炎病人的护理措施", top_k=5)

# 带过滤的检索
results = retriever.search(
    query="呼吸困难",
    top_k=5,
    filters={
        "textbook": "内科护理学",
        "chunk_type": "nursing_intervention"
    }
)

# 按疾病检索
results = retriever.search_by_disease("支气管哮喘")

# 按护理诊断检索
results = retriever.search_by_diagnosis("清理呼吸道无效")
```

---

## 八、版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| v1 | 2024-03 | 初始版本，简单标题分块 |
| v2 | 2024-03 | 语义分块，多维度索引，护理程序完整性 |