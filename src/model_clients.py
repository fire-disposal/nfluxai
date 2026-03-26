#!/usr/bin/env python3
"""统一的 API 服务配置与调用客户端。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_api_key(service_cfg: Dict[str, Any], fallback_env: str = "") -> str:
    key = service_cfg.get("api_key", "")
    if key:
        return key
    env_name = service_cfg.get("api_key_env") or fallback_env
    return os.getenv(env_name, "") if env_name else ""


def get_llm_service_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_config()
    service_cfg = (cfg.get("api_services") or {}).get("llm", {}).copy()

    legacy_llm = cfg.get("llm", {})
    service_cfg.setdefault("provider", legacy_llm.get("provider", "deepseek"))
    service_cfg.setdefault("model", legacy_llm.get("model", "deepseek-chat"))
    service_cfg.setdefault("api_url", legacy_llm.get("api_base", "https://api.deepseek.com/v1/chat/completions"))
    service_cfg.setdefault("temperature", legacy_llm.get("temperature", 0.7))
    service_cfg.setdefault("max_tokens", legacy_llm.get("max_tokens", 2048))
    service_cfg.setdefault("timeout", 120)
    service_cfg.setdefault("api_key_env", "DEEPSEEK_API_KEY")

    provider = service_cfg.get("provider", "deepseek")
    defaults = {
        "deepseek": ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY"),
        "zhipu": ("https://open.bigmodel.cn/api/paas/v4/chat/completions", "ZHIPU_API_KEY"),
        "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "DASHSCOPE_API_KEY"),
    }
    default_url, default_env = defaults.get(provider, defaults["deepseek"])
    service_cfg["api_url"] = os.getenv("LLM_API_BASE", service_cfg.get("api_url") or default_url)
    service_cfg["api_key"] = _resolve_api_key(service_cfg, fallback_env=default_env)
    return service_cfg


def get_embedding_service_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_config()
    service_cfg = (cfg.get("api_services") or {}).get("embedding", {}).copy()

    # 兼容旧配置
    service_cfg.setdefault("provider", "siliconflow")
    service_cfg.setdefault("model", cfg.get("silicon_flow_model", cfg.get("embedding_model", "BAAI/bge-m3")))
    service_cfg.setdefault("api_url", cfg.get("silicon_flow_api_url", ""))
    service_cfg.setdefault("timeout", 120)
    service_cfg.setdefault("batch_size", cfg.get("embedding_batch_size", 32))
    service_cfg.setdefault("api_key_env", "SILICON_FLOW_API_KEY")

    service_cfg["api_url"] = service_cfg.get("api_url") or os.getenv("SILICON_FLOW_API_URL", "")
    service_cfg["api_key"] = _resolve_api_key(service_cfg, fallback_env="SILICON_FLOW_API_KEY")
    return service_cfg


def get_rerank_service_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_config()
    service_cfg = (cfg.get("api_services") or {}).get("rerank", {}).copy()

    # 兼容旧配置
    service_cfg.setdefault("enabled", cfg.get("rerank", True))
    service_cfg.setdefault("provider", "http")
    service_cfg.setdefault("model", cfg.get("rerank_model", "BAAI/bge-reranker-large"))
    service_cfg.setdefault("api_url", cfg.get("rerank_api_url", ""))
    service_cfg.setdefault("top_n", cfg.get("rerank_top_n", 20))
    service_cfg.setdefault("timeout", 30)
    service_cfg.setdefault("api_key_env", "RERANK_API_KEY")

    service_cfg["api_url"] = service_cfg.get("api_url") or os.getenv("RERANK_API_URL", "")
    service_cfg["api_key"] = _resolve_api_key(service_cfg, fallback_env="RERANK_API_KEY")
    return service_cfg


class ApiTextEmbeddings:
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config.get("api_url", "")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "BAAI/bge-m3")
        self.timeout = int(config.get("timeout", 120))

    def _call(self, inputs: List[str]) -> List[List[float]]:
        if not self.api_url:
            raise RuntimeError("嵌入服务 api_url 未配置，请在 config.yaml 的 api_services.embedding.api_url 中设置。")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model, "input": inputs[0] if len(inputs) == 1 else inputs}
        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"嵌入服务请求失败: {e}") from e

        items = data.get("data") if isinstance(data, dict) else data
        if isinstance(data, dict) and "embedding" in data:
            items = [data]
        if not isinstance(items, list):
            raise RuntimeError(f"无法解析嵌入服务返回: {data}")

        vectors: List[List[float]] = []
        for item in items:
            if isinstance(item, dict) and "embedding" in item:
                vectors.append(item["embedding"])
            elif isinstance(item, list):
                vectors.append(item)
            else:
                raise RuntimeError("嵌入向量返回格式不正确")
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]


class ApiReranker:
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config.get("api_url", "")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "BAAI/bge-reranker-large")
        self.timeout = int(config.get("timeout", 30))

    def rank(self, query: str, docs: List[Any], doc_ids: List[int]):
        payload = {
            "model": self.model,
            "query": query,
            "docs": [doc.page_content for doc in docs],
            "doc_ids": doc_ids,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"远程 Rerank 请求失败: {e}") from e

        if isinstance(data, dict) and "scores" in data:
            scores = data["scores"]
            if len(scores) != len(docs):
                raise RuntimeError("远程 Rerank 返回 scores 数量与 docs 不一致")
            return [SimpleNamespace(doc_id=i, score=float(s)) for i, s in enumerate(scores)]
        if isinstance(data, dict) and "ranked" in data:
            return [SimpleNamespace(doc_id=int(i["doc_id"]), score=float(i["score"])) for i in data["ranked"]]
        raise RuntimeError(f"无法解析远程 Rerank 返回: {data}")


def call_chat_completion(prompt: str, config: Dict[str, Any]) -> str:
    api_key = config.get("api_key", "")
    if not api_key:
        env_name = config.get("api_key_env", "LLM_API_KEY")
        return f"[错误：未配置 LLM API Key，请设置 {env_name} 或 config.yaml 中的 api_services.llm.api_key]"

    payload = {
        "model": config.get("model", "deepseek-chat"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens", 2048),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            config.get("api_url", "https://api.deepseek.com/v1/chat/completions"),
            headers=headers,
            json=payload,
            timeout=int(config.get("timeout", 120)),
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "[错误：LLM API 请求超时，请稍后重试]"
    except Exception as e:
        return f"[错误：{e}]"
