#!/usr/bin/env python3
"""DeepSeek API 客户端 - 使用官方 OpenAI SDK"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def get_llm_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or _load_config()
    llm = (cfg.get("api_services") or {}).get("llm", {})
    return {
        "model": llm.get("model", "deepseek-v4-flash"),
        "api_key": llm.get("api_key") or os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": llm.get("base_url", "https://api.deepseek.com"),
        "temperature": llm.get("temperature", 0.0),
        "max_tokens": llm.get("max_tokens", 4096),
    }


def _get_client(config: Optional[Dict[str, Any]] = None) -> OpenAI:
    cfg = config or get_llm_config()
    api_key = cfg.get("api_key", "")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url=cfg.get("base_url", "https://api.deepseek.com"))


def call_deepseek(prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config or get_llm_config()
    client = _get_client(cfg)
    resp = client.chat.completions.create(
        model=cfg.get("model", "deepseek-v4-flash"),
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 4096),
        extra_body={"thinking": {"type": "disabled"}},
    )
    return resp.choices[0].message.content
