# 系统配置指南

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     护理教材 AI 问答系统                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户查询 ──→ 检索器 ──→ Rerank ──→ LLM ──→ 回答      │
│               │            │           │                    │
│               ▼            ▼           ▼                    │
│         嵌入模型      Reranker     DeepSeek API             │
│         (本地)        (本地)         (云端)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 模型配置

### 1. 嵌入模型 (本地运行)

推荐使用 BGE 系列中文嵌入模型，通过 HuggingFace 镜像加速下载：

| 模型 | 维度 | 大小 | 效果 | 推荐场景 |
|------|------|------|------|----------|
| `BAAI/bge-large-zh-v1.5` | 1024 | ~1.3GB | ⭐⭐⭐⭐⭐ | 生产环境 |
| `BAAI/bge-base-zh-v1.5` | 768 | ~400MB | ⭐⭐⭐⭐ | 平衡选择 |
| `shibing624/text2vec-base-chinese` | 768 | ~400MB | ⭐⭐⭐ | 轻量部署 |

**配置方式** (`config.yaml`):

```yaml
embedding_model: "BAAI/bge-large-zh-v1.5"
embedding_device: "cpu"           # 或 "cuda" 使用 GPU
huggingface_mirror: "https://hf-mirror.com"
```

### 2. Rerank 模型 (本地运行)

| 模型 | 效果 | 推荐场景 |
|------|------|----------|
| `BAAI/bge-reranker-large` | ⭐⭐⭐⭐⭐ | 生产环境 |
| `BAAI/bge-reranker-base` | ⭐⭐⭐⭐ | 平衡选择 |

**配置方式**:

```yaml
rerank: true
rerank_model: "BAAI/bge-reranker-large"
rerank_top_n: 20
```

### 3. LLM 配置 (DeepSeek API)

**获取 API Key**: https://platform.deepseek.com/

**配置方式** (`config.yaml`):

```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"          # 或 "deepseek-reasoner"
  api_key: ""                     # 建议使用环境变量
  api_base: "https://api.deepseek.com/v1/chat/completions"
  temperature: 0.7
  max_tokens: 2048
```

**环境变量方式** (推荐):

```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

---

## 备选 LLM 配置

### 智谱 GLM

```yaml
llm:
  provider: "zhipu"
  model: "glm-4-flash"
  api_key: ""
  api_base: "https://open.bigmodel.cn/api/paas/v4/chat/completions"
```

```bash
export ZHIPU_API_KEY="xxxxxxxx.xxxxxxxx"
```

### 通义千问

```yaml
llm:
  provider: "qwen"
  model: "qwen-turbo"
  api_key: ""
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
```

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

### Ollama 本地模型 (无需 API Key)

```yaml
llm:
  provider: "ollama"
  model: "qwen2.5:7b"
```

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull qwen2.5:7b

# 启动服务
ollama serve
```

---

## 环境变量汇总

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | `sk-xxx` |
| `ZHIPU_API_KEY` | 智谱 API 密钥 | `xxx.xxx` |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope 密钥 | `sk-xxx` |
| `HF_ENDPOINT` | HuggingFace 镜像地址 | `https://hf-mirror.com` |

---

## 完整配置示例

```yaml
# config.yaml

# 嵌入模型 (国内镜像加速)
embedding_model: "BAAI/bge-large-zh-v1.5"
embedding_device: "cpu"
embedding_batch_size: 32
huggingface_mirror: "https://hf-mirror.com"

# 分块配置
chunk_size: 500
chunk_overlap: 100

# 检索配置
top_k: 5
rerank: true
rerank_model: "BAAI/bge-reranker-large"
rerank_top_n: 20

# LLM 配置 - DeepSeek
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: ""
  api_base: "https://api.deepseek.com/v1/chat/completions"
  temperature: 0.7
  max_tokens: 2048
```

---

## 费用估算

### DeepSeek API

| 模型 | 输入价格 | 输出价格 | 说明 |
|------|----------|----------|------|
| `deepseek-chat` | ¥1/百万token | ¥2/百万token | 日常对话 |
| `deepseek-reasoner` | ¥4/百万token | ¥16/百万token | 深度推理 |

**预估**: 每次问答约消耗 2000-5000 token，成本约 ¥0.01-0.02

### 本地模型

- 嵌入模型: 免费，需 ~1GB 内存
- Rerank 模型: 免费，需 ~500MB 内存
- Ollama LLM: 免费，需 8-16GB 内存

---

## 性能优化建议

| 场景 | 嵌入模型 | Rerank | LLM | 内存需求 |
|------|----------|--------|-----|----------|
| 开发测试 | bge-base | 关闭 | DeepSeek API | ~2GB |
| 生产部署 | bge-large | 开启 | DeepSeek API | ~4GB |
| 完全离线 | bge-base | 关闭 | Ollama 7B | ~12GB |

---

## 故障排查

### 嵌入模型下载慢

```bash
# 设置镜像
export HF_ENDPOINT="https://hf-mirror.com"
```

### DeepSeek API 调用失败

1. 检查 API Key 是否正确
2. 检查账户余额
3. 检查网络连接

### Ollama 连接失败

```bash
# 确保 Ollama 服务运行
ollama serve

# 检查服务状态
curl http://localhost:11434/api/tags
```