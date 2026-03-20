# 护理教材 AI 问答系统

基于 RAG (检索增强生成) 技术的护理学教材智能问答原型系统。

## 🚀 快速开始

### 1. 环境准备

确保已安装 [uv](https://github.com/astral-sh/uv)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装依赖

```bash
cd /home/firedisposal/nflux_ai
uv sync
```

### 3. 导入数据

```bash
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
│   ├── app.py            # Streamlit 聊天界面
│   ├── retriever.py      # 检索引擎 (语义检索 + Rerank)
│   ├── ingest.py         # 数据导入与智能分块
│   ├── llm.py            # LLM 集成 (Ollama / API)
│   ├── medical_terms.py  # 医学词典 (疾病/诊断/症状)
│   └── test_flow.py      # 功能测试脚本
├── data/
│   ├── textbooks/        # 教材源文件 (已内置)
│   ├── chroma_db/        # ChromaDB 向量数据库
│   └── index/            # 索引文件
├── docs/
│   ├── INDEX_DESIGN.md   # 索引设计文档
│   └── CODE_REVIEW.md    # 代码审查报告
├── scripts/
│   └── copy_textbooks.py # 教材拷贝脚本
├── config.yaml           # 配置文件
├── main.py               # 主入口点
├── start.sh              # 快速启动脚本
└── pyproject.toml        # 项目依赖
```

---

## 📚 数据源

三本核心护理教材已内置到 `data/textbooks/`：

| 教材 | 章节数 | 内容 |
|------|--------|------|
| 内科护理学 | 10 章 | 呼吸、循环、消化等系统疾病护理 |
| 外科护理学 | 45 章 | 手术、创伤、各外科专科护理 |
| 新编护理学基础 | 21 章 | 护理程序、基础护理技术 |

---

## 🔧 核心功能

### 医学词典

系统内置医学词典，支持智能实体识别：

| 类型 | 数量 | 说明 |
|------|------|------|
| 疾病名称 | 283 个 | 按系统分类（呼吸、循环、消化等） |
| 护理诊断 | 121 个 | NANDA-I 标准诊断名称 |
| 症状 | 50+ 个 | 常见临床症状 |
| 护理操作 | 50+ 个 | 基础护理操作技术 |

### 智能分块

- 按护理程序单元分块（评估、诊断、措施、评价）
- 保持内容语义完整性
- 自动提取疾病、症状、护理诊断标签

### 多维度索引

```
向量索引 (ChromaDB)
    │
    ├── 疾病索引 → 按疾病名称快速定位
    ├── 症状索引 → 按症状检索相关护理
    ├── 护理诊断索引 → NANDA 标准诊断名称
    └── 章节索引 → 层级导航
```

---

## 🔍 检索流程

```
用户查询
    │
    ├──→ 1. 实体识别 (疾病/诊断/症状)
    │
    ├──→ 2. 向量检索 (ChromaDB)
    │
    ├──→ 3. Rerank 重排序
    │
    └──→ 4. 返回结果 + 来源信息
```

---

## 🛠️ 配置说明

### config.yaml

```yaml
# 嵌入模型
embedding_model: "shibing624/text2vec-base-chinese"

# 检索配置
top_k: 5
rerank: true

# LLM 配置
llm:
  provider: "ollama"
  model: "qwen2.5:7b"
  temperature: 0.7
```

### 使用 Ollama 本地 LLM

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull qwen2.5:7b

# 启动服务
ollama serve
```

### 使用 API 服务

```yaml
llm:
  provider: "openai"
  api_key: "sk-xxx"
  model: "gpt-3.5-turbo"
```

---

## 📋 命令行工具

```bash
# 导入数据
python main.py --ingest

# 启动应用
python main.py --run

# 检查数据状态
python main.py --check

# 运行测试
uv run python src/test_flow.py
```

---

## 🔍 示例问题

- 肺炎病人的护理措施有哪些
- 糖尿病酮症酸中毒的处理
- 清理呼吸道无效的护理措施
- 高血压病人的健康教育
- 手术前后病人的护理注意事项

---

## 📝 依赖列表

```toml
chromadb>=0.5.0           # 向量数据库
sentence-transformers     # 嵌入模型
langchain                 # RAG 框架
langchain-chroma          # ChromaDB 集成
langchain-community       # 社区组件
streamlit                 # 前端界面
pyyaml                    # 配置解析
rerankers                 # Rerank 模型
```

---

## ⚠️ 注意事项

1. **首次运行**需要先执行数据导入
2. **向量库位置**: `data/chroma_db`
3. **索引文件**: `data/index/`
4. **内存需求**: 建议至少 4GB 可用内存
5. **响应速度**: 启用 Rerank 会增加 1-2 秒延迟

---

## 🔧 故障排查

### 问题：找不到相关来源

```bash
# 重新导入数据
uv run python src/ingest.py
```

### 问题：LLM 服务不可用

```bash
# 检查 Ollama 服务
ollama list

# 启动服务
ollama serve
```

### 问题：检索结果不准确

1. 启用 Rerank: `config.yaml` 中设置 `rerank: true`
2. 增加 `top_k` 值
3. 使用更专业的医学术语

---

## 📄 License

MIT