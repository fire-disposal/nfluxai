# 护理教材 AI 问答系统

基于 RAG (检索增强生成) 技术的护理学教材智能问答原型系统。

## 🚀 快速开始

### 1. 环境准备

确保已安装 [uv](https://github.com/astral-sh/uv)：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装依赖

```bash
cd /home/firedisposal/nflux_ai
uv sync
```

### 3. 导入数据

```bash
# 将三本教材向量化并存入 ChromaDB
# 同时生成索引文件 data/index/chunks_index.json
uv run python src/ingest.py
```

### 4. 启动应用

```bash
# 方式一：使用启动脚本
bash start.sh

# 方式二：直接运行
uv run streamlit run src/app.py
```

访问 http://localhost:8501

---

## 📁 项目结构

```
nflux_ai/
├── src/
│   ├── ingest.py      # 数据导入与分块 (适配教材格式)
│   ├── retriever.py   # 检索引擎 (语义检索 + Rerank + 来源追踪)
│   ├── llm.py         # LLM 集成 (Ollama / API)
│   ├── app.py         # Streamlit 聊天界面 (带来源显示)
│   ├── viewer.py      # 索引查看工具 (命令行)
│   └── test.py        # 测试脚本
├── data/
│   ├── chroma_db/     # ChromaDB 向量数据库
│   └── index/         # 索引文件 (JSON 格式)
│       └── chunks_index.json
├── config.yaml        # 配置文件
├── start.sh           # 快速启动脚本
├── pyproject.toml     # 项目依赖
└── README.md          # 本文件
```

---

## 📚 数据源

仅处理三本核心护理教材（位于 `/home/firedisposal/nflux`）：

| 教材 | 章节数 | 内容 |
|------|--------|------|
| 内科护理学 | 10 章 | 呼吸、循环、消化等系统疾病护理 |
| 外科护理学 | 45 章 | 手术、创伤、各外科专科护理 |
| 新编护理学基础 | 22 章 | 护理程序、基础护理技术 |

---

## 🔧 分块策略

系统采用**智能层级分块**策略，适配教材实际格式：

### 分块逻辑

```
教材文件 (.md)
    │
    ├── # 主标题 (章/节) ────→ Section
    │       │
    │       └── ## 小节标题 ──→ Subsection
    │               │
    │               └── 内容 (<50 字符跳过)
    │
    └── 元数据提取:
            - 教材名称
            - 章节号 (从文件名解析)
            - 标题层级
            - 行号范围 (用于原文定位)
```

### 元数据字段

```json
{
  "chunk_id": "唯一标识",
  "textbook": "教材名称",
  "filename": "文件名",
  "filepath": "完整路径",
  "chapter_num": "章节号",
  "chapter_title": "章节标题",
  "section_header": "节标题",
  "subsection_header": "小节标题",
  "title": "完整标题",
  "line_start": "起始行号",
  "line_end": "结束行号",
  "content_preview": "内容预览 (200 字符)"
}
```

---

## 🔍 检索与来源追踪

### 检索流程

```
用户查询
    │
    ├──→ 向量检索 (ChromaDB)
    │       │
    │       └── 返回 Top-N 相似文档
    │
    ├──→ Rerank 重排序 (可选)
    │       │
    │       └── 提高相关性精度
    │
    └──→ 来源格式化
            │
            └── [教材/章节/小节]
```

### 来源显示

系统支持**完整的原文来源列出**：

1. **引用标注**: 回答中标注 [1][2] 等编号
2. **来源面板**: 显示完整来源信息
   - 教材名称
   - 章节标题
   - 文件名
   - 行号范围
3. **原文查看**: 可选显示原文内容片段

---

## 📋 命令行工具

### 索引查看器

```bash
# 交互模式
uv run python src/viewer.py

# 列出分块统计
uv run python src/viewer.py -l
uv run python src/viewer.py -l 内科护理学

# 搜索来源
uv run python src/viewer.py -s 呼吸系统

# 显示帮助
uv run python src/viewer.py -h
```

### 测试脚本

```bash
# 运行完整测试
uv run python src/test.py
```

---

## 🛠️ 高级用法

### 使用 Ollama 本地 LLM

1. 安装 Ollama: https://ollama.ai
2. 下载模型：`ollama pull qwen2.5:7b`
3. 配置 `config.yaml`:
   ```yaml
   llm:
     provider: "ollama"
     model: "qwen2.5:7b"
   ```

### 使用 API 服务

```yaml
llm:
  provider: "openai"
  api_key: "sk-xxx"
  model: "gpt-3.5-turbo"
```

支持任何 OpenAI 兼容 API（DeepSeek、智谱等）。

---

## 📊 核心功能

| 功能 | 说明 |
|------|------|
| **智能分块** | 按 Markdown 标题层级切割，适配教材格式 |
| **语义检索** | 使用 text2vec 中文向量模型 |
| **Rerank 重排序** | bge-reranker 提高检索精度 |
| **引用溯源** | 回答中标注 [1][2]，可跳转原文 |
| **来源面板** | 显示完整来源信息 (教材/章节/行号) |
| **原文查看** | 支持读取并显示原文内容 |
| **教材过滤** | 按内科/外科/基础护理筛选 |
| **索引工具** | 命令行查看和搜索索引 |

---

## 🔍 示例问题

- 呼吸系统疾病病人的护理评估要点
- 护理程序的基本步骤有哪些
- 慢性阻塞性肺疾病的护理措施
- 手术前后病人的护理注意事项
- 静脉输液反应的预防和处理

---

## 📝 依赖列表

```toml
chromadb>=0.5.0       # 向量数据库
sentence-transformers # 嵌入模型
langchain             # RAG 框架
streamlit             # 前端界面
pyyaml                # 配置解析
rerankers             # Rerank 模型
```

---

## ⚠️ 注意事项

1. **首次运行**需要先执行 `uv run python src/ingest.py` 导入数据
2. **向量库位置**: `data/chroma_db`，如需重置直接删除该目录
3. **索引文件**: `data/index/chunks_index.json` 包含所有分块的来源信息
4. **内存需求**: 建议至少 4GB 可用内存（嵌入模型 + ChromaDB）
5. **响应速度**: 启用 Rerank 会增加 1-2 秒延迟

---

## 🔧 故障排查

### 问题：找不到相关来源

**原因**: 数据未导入或向量库为空

**解决**:
```bash
uv run python src/ingest.py
```

### 问题：无法读取原文内容

**原因**: 教材路径不正确

**解决**: 确认 `nflux` 文件夹位于 `/home/firedisposal/nflux`

### 问题：检索结果不准确

**解决**:
1. 启用 Rerank: `config.yaml` 中设置 `rerank: true`
2. 增加 `top_k` 值
3. 优化查询语句，使用更专业的术语

---

## 📄 License

MIT
